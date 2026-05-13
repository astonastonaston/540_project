"""
Adversarial Scenario Generator
===============================
Uses an LLM to generate YAML scenario files that are likely
to cause safety violations in the Franka Panda reach task.

Scene geometry (wall and goal) is FIXED. Scenarios only vary:
- ee_start.y: EE start position on y-axis (opposite side of wall from goal)
- sensor_noise: Gaussian noise on EE position and joint readings
- control: delay_steps and gain_scale
"""

import json
import yaml
import numpy as np
from typing import List, Dict, Any, Optional


# ------------------------------------------------------------------ #
#  LLM-Based Generator
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are an adversarial red-team agent for a robotic safety testing system.

Your job: generate simulation scenarios that cause a Franka Panda robot arm to violate
its safety constraints while reaching a target point past a wall obstacle.

FIXED SCENE (you cannot change these):
- Wall obstacle at y=0.0, size [0.15, 0.015, 0.20]
- Goal at (0.05, 0.40, 0.10), on the far side of the wall
- EE starts on the near side of the wall (negative y)

The controller uses PD control + potential field obstacle avoidance:
- Attractive PD force pulls EE toward goal
- Repulsive inverse-square force pushes EE away from obstacle within 0.15m influence distance
- Both forces mapped through a fixed approximate Jacobian (degrades far from rest config)

YOU CONTROL:
- ee_start.y: Where the EE starts on the y-axis (range: -0.68 to -0.06, always negative)
- sensor_noise: Gaussian noise corrupts EE position readings AND obstacle distance estimates
- control.delay_steps: Delayed repulsion response → overshoot into obstacle
- control.gain_scale: High gain can overwhelm repulsive field
- control.influence_distance: Range of the repulsive field (small values = field activates only when EE is already very close, leaving no room to react)

KNOWN EXPLOITABLE FAILURE MODES:
- Local minima: EE gets stuck when attractive and repulsive forces cancel
- Sensor noise: Wrong distance → wrong repulsion magnitude → collision
- Delay + proximity: Repulsion fires too late, EE already past safety margin
- High gain: Attractive force overwhelms repulsion near obstacle
- Jacobian degradation: Far from rest config, both forces map incorrectly
- Combined: noise + delay + high gain create compound failures

Respond with ONLY valid YAML. No explanation, no markdown fences."""

GENERATION_PROMPT_TEMPLATE = """
SAFETY CONSTRAINTS:
{constraints}

PARAMETER RANGES (you MUST stay within these bounds):
{param_ranges}

PREVIOUS VIOLATIONS FOUND (learn from these):
{history}

EPISODE {episode_num}: Generate a new adversarial scenario as YAML.
Try a DIFFERENT strategy than previous episodes.

Required YAML structure:
name: <descriptive_snake_case_name>
strategy: "<brief explanation of adversarial strategy>"
ee_start:
  y: float
sensor_noise:
  ee_position_std: float
  joint_position_std: float
control:
  delay_steps: int
  gain_scale: float
  influence_distance: float
"""


def generate_adversarial_scenario_llm(
    constraints_path: str,
    episode_num: int,
    violation_history: List[dict],
    api_key: Optional[str] = None,
    save_dir: str = "scenarios",
) -> dict:
    """
    Use Claude to generate an adversarial scenario and save it as a YAML file.

    Args:
        constraints_path: Path to constraints.yaml
        episode_num: Current episode number
        violation_history: List of past violation summaries
        api_key: Anthropic API key
        save_dir: Directory to save generated scenario YAML files

    Returns:
        Validated scenario dict (with fixed wall/goal added)
    """
    import os
    from anthropic import Anthropic

    with open(constraints_path) as f:
        cfg = yaml.safe_load(f)

    constraints_str = json.dumps(cfg["constraints"], indent=2)
    params_str = json.dumps(cfg["adversarial_params"], indent=2)

    # Keep only last 5 episodes of history to stay within context
    recent_history = violation_history[-5:] if violation_history else [{"note": "No prior data"}]
    history_str = json.dumps(recent_history, indent=2)

    prompt = GENERATION_PROMPT_TEMPLATE.format(
        constraints=constraints_str,
        param_ranges=params_str,
        history=history_str,
        episode_num=episode_num,
    )

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text
    # Strip markdown fences if present
    raw_text = raw_text.strip().removeprefix("```yaml").removeprefix("```").removesuffix("```").strip()

    scenario_raw = yaml.safe_load(raw_text)
    validated = validate_and_clamp(scenario_raw, cfg["adversarial_params"])

    # Save the generated scenario as a YAML file
    os.makedirs(save_dir, exist_ok=True)
    name = scenario_raw.get("name", f"episode_{episode_num:03d}")
    save_path = os.path.join(save_dir, f"{name}.yaml")
    save_data = {
        "name": name,
        "strategy": scenario_raw.get("strategy", ""),
        "ee_start": validated["ee_start"],
        "sensor_noise": validated["sensor_noise"],
        "control": validated["control"],
    }
    with open(save_path, "w") as f:
        yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved scenario: {save_path}")

    return validated


# ------------------------------------------------------------------ #
#  Random Baseline Generator
# ------------------------------------------------------------------ #

def generate_random_scenario(constraints_path: str) -> dict:
    """
    Generate a uniformly random scenario within parameter bounds.
    Used as a baseline to compare against LLM adversarial generation.
    """
    with open(constraints_path) as f:
        cfg = yaml.safe_load(f)
    params = cfg["adversarial_params"]

    def rand_range(r):
        return np.random.uniform(r[0], r[1])

    scenario = {
        "ee_start": {
            "y": rand_range(params["ee_start"]["y"]),
        },
        "sensor_noise": {
            "ee_position_std": rand_range(params["sensor_noise"]["ee_position_std"]),
            "joint_position_std": rand_range(params["sensor_noise"]["joint_position_std"]),
        },
        "control": {
            "delay_steps": int(rand_range(params["control"]["delay_steps"])),
            "gain_scale": rand_range(params["control"]["gain_scale"]),
            "influence_distance": rand_range(params["control"]["influence_distance"]),
        },
        "reasoning": "random baseline",
    }
    return scenario


# ------------------------------------------------------------------ #
#  Validation
# ------------------------------------------------------------------ #

def validate_and_clamp(scenario: dict, params: dict) -> dict:
    """Clamp all scenario values to stay within allowed parameter ranges."""

    def clamp(val, low, high):
        return max(low, min(high, val))

    # EE start
    ee_start = scenario.get("ee_start", {})
    ee_start["y"] = clamp(ee_start.get("y", -0.40), *params["ee_start"]["y"])

    # Sensor noise
    sn = scenario.get("sensor_noise", {})
    sn_p = params["sensor_noise"]
    sn["ee_position_std"] = clamp(sn.get("ee_position_std", 0.0), *sn_p["ee_position_std"])
    sn["joint_position_std"] = clamp(sn.get("joint_position_std", 0.0), *sn_p["joint_position_std"])

    # Control params
    ctrl = scenario.get("control", {})
    ctrl_p = params["control"]
    ctrl["delay_steps"] = int(clamp(ctrl.get("delay_steps", 0), *ctrl_p["delay_steps"]))
    ctrl["gain_scale"] = clamp(ctrl.get("gain_scale", 1.0), *ctrl_p["gain_scale"])
    if "influence_distance" in ctrl_p:
        ctrl["influence_distance"] = clamp(
            ctrl.get("influence_distance", 0.15), *ctrl_p["influence_distance"]
        )

    scenario["ee_start"] = ee_start
    scenario["sensor_noise"] = sn
    scenario["control"] = ctrl
    scenario["reasoning"] = scenario.get("strategy", scenario.get("reasoning", ""))

    return scenario
