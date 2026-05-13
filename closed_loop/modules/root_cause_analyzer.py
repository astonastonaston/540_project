"""
Root-Cause Analyzer
===================
Uses an LLM to analyze episode results and determine why the robot
failed or violated constraints, based on the trajectory data, violations,
and scenario parameters.

Falls back to heuristic analysis if no API key is available.
"""

import json
from typing import List, Dict, Any, Optional
from modules.safety_monitor import Violation


# ------------------------------------------------------------------ #
#  LLM-Based Root Cause Analysis
# ------------------------------------------------------------------ #

RCA_SYSTEM_PROMPT = """You are a robotics safety analyst. You analyze simulation episodes
of a Franka Panda robot arm that must reach a goal position while avoiding a wall obstacle.

The controller uses PD control + potential field obstacle avoidance:
- Attractive PD force pulls end-effector (EE) toward goal
- Repulsive inverse-square force pushes EE away from obstacle within 0.15m influence distance
- Both forces mapped through a fixed approximate Jacobian (degrades far from rest config)

Scene layout:
- Wall at y=0.0 between EE start (negative y) and goal (y=0.40)
- Robot must navigate over or around the wall
- EE start position varies per scenario

Known failure modes:
- Local minima: attractive and repulsive forces cancel out, robot stalls near wall
- Jacobian degradation: far from rest config, control forces map incorrectly, robot drifts
- Sensor noise: corrupted distance readings → wrong repulsion → collision or oscillation
- Control delay: repulsion fires too late → overshoot into obstacle
- High gain: attractive force overwhelms repulsive field → obstacle proximity violation
- Joint limits: extreme arm configurations when reaching far from base

Analyze the episode data and provide:
1. A concise root cause explanation (1-3 sentences)
2. Which subsystem is primarily responsible: PERCEPTION, CONTROL, or PLANNING
3. Confidence level (0.0-1.0)

Respond with ONLY valid JSON:
{"subsystem": "...", "confidence": 0.X, "reasoning": "..."}"""

RCA_PROMPT_TEMPLATE = """Analyze this episode:

SCENARIO:
{scenario}

RESULT:
- Success: {success}
- Steps: {steps} / {max_steps}
- Time: {elapsed:.1f}s / {max_time}s

VIOLATIONS ({num_violations} total):
{violations}

TRAJECTORY SAMPLES (step: ee_position, obstacle_distance, speed):
{trajectory}

Min obstacle distance: {min_obs_dist:.4f}m (threshold: 0.02m)

What caused the robot to {outcome}?"""


class RootCauseAnalyzer:
    """
    LLM-based root-cause attribution for Franka Panda safety violations.
    Falls back to heuristic analysis if no API key is provided.

    Subsystem categories:
    - PERCEPTION: failures caused by sensor noise / inaccurate state estimation
    - PLANNING:   failures caused by path/trajectory choices (obstacle placement, local minima)
    - CONTROL:    failures caused by controller behavior (gains, delay, overshoot, Jacobian)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def analyze(
        self,
        violations: List[Violation],
        scenario: dict,
        trajectory: List[dict] = None,
        episode_result: dict = None,
    ) -> List[dict]:
        """
        Analyze violations from one episode.

        Args:
            violations: List of Violation objects from SafetyMonitor
            scenario: The adversarial scenario dict
            trajectory: List of sampled states from the episode
            episode_result: Full episode result dict (success, steps, etc.)

        Returns:
            List of attribution dicts with keys:
                subsystem, confidence, reasoning
        """
        if not violations and episode_result and not episode_result.get("success"):
            # No violations but also didn't succeed — still worth analyzing
            pass
        elif not violations:
            return []

        if self.api_key and trajectory:
            try:
                return self._analyze_llm(violations, scenario, trajectory, episode_result)
            except Exception as e:
                print(f"  LLM root-cause analysis failed: {e}")

        # Fallback to heuristic
        return self._analyze_heuristic(violations, scenario)

    # ------------------------------------------------------------------ #
    #  LLM Analysis
    # ------------------------------------------------------------------ #

    def _analyze_llm(
        self,
        violations: List[Violation],
        scenario: dict,
        trajectory: List[dict],
        episode_result: dict,
    ) -> List[dict]:
        """Use Claude to analyze the episode."""
        import yaml
        from anthropic import Anthropic

        # Format scenario (only variable parts)
        scenario_str = json.dumps({
            "ee_start": scenario.get("ee_start", {}),
            "sensor_noise": scenario.get("sensor_noise", {}),
            "control": scenario.get("control", {}),
        }, indent=2)

        # Format violations summary
        if violations:
            v_types = {}
            for v in violations:
                v_types[v.type] = v_types.get(v.type, 0) + 1
            violations_str = json.dumps(v_types, indent=2)
            first_v = violations[0]
            violations_str += f"\nFirst violation: {first_v.type} at step {first_v.timestep} (value={first_v.value:.4f}, limit={first_v.limit:.4f})"
        else:
            violations_str = "None (but robot did not reach goal)"

        # Format trajectory samples
        traj_lines = []
        for s in trajectory:
            traj_lines.append(
                f"  step {s['step']:3d}: ee=({s['ee_x']:.3f}, {s['ee_y']:.3f}, {s['ee_z']:.3f}), "
                f"obs_dist={s['obs_dist']:.4f}, speed={s['speed']:.4f}"
            )
        trajectory_str = "\n".join(traj_lines)

        steps = episode_result.get("steps", 0)
        max_steps = episode_result.get("max_steps", 600)
        dt = episode_result.get("dt", 0.05)
        success = episode_result.get("success", False)

        outcome = "succeed" if success else "fail to reach the goal"
        if violations:
            outcome += f" and violate {len(violations)} constraint(s)"

        prompt = RCA_PROMPT_TEMPLATE.format(
            scenario=scenario_str,
            success=success,
            steps=steps,
            max_steps=max_steps,
            elapsed=steps * dt,
            max_time=max_steps * dt,
            num_violations=len(violations),
            violations=violations_str,
            trajectory=trajectory_str,
            min_obs_dist=episode_result.get("min_obs_dist", float('inf')),
            outcome=outcome,
        )

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=RCA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = response.content[0].text.strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw_text)

        return [{
            "subsystem": result.get("subsystem", "unknown").lower(),
            "confidence": result.get("confidence", 0.5),
            "reasoning": result.get("reasoning", "LLM analysis returned no reasoning."),
        }]

    # ------------------------------------------------------------------ #
    #  Heuristic Fallback
    # ------------------------------------------------------------------ #

    def _analyze_heuristic(
        self,
        violations: List[Violation],
        scenario: dict,
    ) -> List[dict]:
        """Heuristic root-cause attribution (no LLM needed)."""
        results = []
        noise = scenario.get("sensor_noise", {})
        control = scenario.get("control", {})

        ee_noise = noise.get("ee_position_std", 0.0)
        joint_noise = noise.get("joint_position_std", 0.0)
        delay = control.get("delay_steps", 0)
        gain_scale = control.get("gain_scale", 1.0)

        has_noise = ee_noise > 0.01 or joint_noise > 0.005
        has_delay = delay > 3
        has_gain_issue = gain_scale > 1.5 or gain_scale < 0.6

        for v in violations:
            attribution = self._attribute_single(v, has_noise, has_delay, has_gain_issue, scenario)
            results.append(attribution)

        return results

    def _attribute_single(
        self,
        v: Violation,
        has_noise: bool,
        has_delay: bool,
        has_gain_issue: bool,
        scenario: dict,
    ) -> dict:
        """Attribute a single violation to a subsystem."""
        vtype = v.type

        if vtype == "obstacle_proximity":
            if has_noise:
                return self._result(v, "perception", 0.8,
                    "Sensor noise likely caused inaccurate distance estimate.")
            elif has_delay:
                return self._result(v, "control", 0.7,
                    "Control delay caused late repulsive response.")
            elif has_gain_issue:
                return self._result(v, "control", 0.8,
                    "High gain overwhelmed repulsive field.")
            else:
                return self._result(v, "planning", 0.7,
                    "Controller path too close to obstacle.")

        elif vtype in ("ee_velocity_exceeded", "joint_velocity_exceeded"):
            if has_gain_issue:
                return self._result(v, "control", 0.85,
                    f"Gain scale ({scenario['control']['gain_scale']:.2f}) caused excessive velocity.")
            elif has_delay:
                return self._result(v, "control", 0.7,
                    "Control delay caused velocity spike from accumulated error.")
            else:
                return self._result(v, "control", 0.6,
                    "Controller produced excessive velocity.")

        elif vtype in ("joint_limit_low", "joint_limit_high"):
            if has_noise:
                return self._result(v, "perception", 0.6,
                    "Noisy joint readings caused drift toward joint limits.")
            else:
                return self._result(v, "control", 0.6,
                    "Controller drove joints near limits — Jacobian degradation likely.")

        elif vtype == "workspace_boundary":
            if has_gain_issue:
                return self._result(v, "control", 0.75,
                    "Aggressive gains caused EE to overshoot workspace boundary.")
            else:
                return self._result(v, "planning", 0.7,
                    "Configuration pushed arm outside workspace.")

        elif vtype == "timeout":
            if has_delay:
                return self._result(v, "control", 0.7,
                    "Control delay slowed convergence to goal.")
            elif has_noise:
                return self._result(v, "perception", 0.6,
                    "Sensor noise caused oscillatory approach, preventing convergence.")
            else:
                return self._result(v, "planning", 0.6,
                    "Robot stuck in local minimum — repulsive field blocked path to goal.")

        return self._result(v, "unknown", 0.3, "Could not determine root cause.")

    @staticmethod
    def _result(v: Violation, subsystem: str, confidence: float, reasoning: str) -> dict:
        return {
            "subsystem": subsystem,
            "confidence": confidence,
            "reasoning": reasoning,
        }
