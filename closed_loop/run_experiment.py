"""
Red-Teaming Experiment Runner
=============================
Main entry point that ties together all modules:
1. Creates the FrankaReachObstacle environment
2. Generates adversarial scenarios (LLM or random baseline)
3. Runs the controller
4. Monitors safety constraints
5. Performs root-cause analysis
6. Logs everything

Usage:
    python run_experiment.py --mode single       # Run one scenario for debugging
    python run_experiment.py --mode full          # Run full LLM adversarial experiment
    python run_experiment.py --mode baseline      # Run random baseline
    python run_experiment.py --mode compare       # Run both and compare
"""

import argparse
import json
import yaml
import numpy as np
import gymnasium as gym

# Register our custom environment
from env.franka_reach_obstacle import FrankaReachObstacleEnv  # noqa: F401
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

from controller.pd_ee_controller import PDEEController
from modules.safety_monitor import SafetyMonitor
from modules.adversarial_generator import (
    generate_adversarial_scenario_llm,
    generate_random_scenario,
)
from modules.root_cause_analyzer import RootCauseAnalyzer
from modules.experiment_logger import ExperimentLogger


# ------------------------------------------------------------------ #
#  Core Episode Runner
# ------------------------------------------------------------------ #

def run_episode(
    env,
    controller: PDEEController,
    monitor: SafetyMonitor,
    analyzer: RootCauseAnalyzer,
    scenario: dict,
    max_steps: int = 600,
    dt: float = 0.05,
    render: bool = False,
    step_by_step: bool = False,
    constraints: dict = None,
) -> dict:
    """
    Run a single episode with the given adversarial scenario.

    Args:
        render: If True, call env.render() each step (requires render_mode="human")

    Returns:
        dict with keys: success, steps, violations, attributions, scenario
    """
    # Override max_steps and dt from constraints if provided
    if constraints:
        max_steps = constraints.get("max_steps", max_steps)
        dt = constraints.get("dt", dt)

    # Reset everything
    obs, info = env.reset()
    controller.reset()
    monitor.reset()

    # Access unwrapped env for custom methods
    base_env = env.unwrapped

    # Apply adversarial perturbations to the scene
    base_env.apply_adversarial_scenario(scenario)

    # Extract noise/control params from scenario
    noise = scenario.get("sensor_noise", {})
    ctrl = scenario.get("control", {})
    ee_noise = noise.get("ee_position_std", 0.0)
    joint_noise = noise.get("joint_position_std", 0.0)

    # Extract obstacle geometry for waypoint navigation
    obs_cfg = scenario.get("obstacle", {})
    obs_pos_dict = obs_cfg.get("position", {"x": 0.05, "y": 0.0, "z": 0.0})
    obs_pos = [obs_pos_dict["x"], obs_pos_dict["y"], obs_pos_dict.get("z", 0.0)]
    obs_hs = obs_cfg.get("half_size", [0.15, 0.015, 0.20])

    # Reconfigure controller with adversarial params
    controller_ep = PDEEController(
        gain_scale=ctrl.get("gain_scale", 1.0),
        delay_steps=ctrl.get("delay_steps", 0),
        repulsive_gain=ctrl.get("repulsive_gain", 0.5),
        influence_distance=ctrl.get("influence_distance", 0.15),
        obstacle_pos=obs_pos,
        obstacle_half_size=obs_hs,
    )

    success = False
    init_state = base_env.get_safety_state()
    print(f"  EE start:    {init_state['ee_pos']}")
    print(f"  Goal:        {init_state['goal_pos']}")
    print(f"  Obs dist:    {init_state['obstacle_distance']:.4f}")
    min_obs_dist = init_state['obstacle_distance']
    trajectory = []

    for step in range(max_steps):
        # Get safety-relevant state from the unwrapped env
        state = base_env.get_safety_state()

        # Check safety constraints
        monitor.check(state)

        # Track min obstacle distance and sample trajectory
        if state["obstacle_distance"] < min_obs_dist:
            min_obs_dist = state["obstacle_distance"]
        if step < 5 or step % (max_steps // 10) == 0 or step == max_steps - 1:
            trajectory.append({
                "step": step,
                "ee_x": float(state["ee_pos"][0]),
                "ee_y": float(state["ee_pos"][1]),
                "ee_z": float(state["ee_pos"][2]),
                "obs_dist": float(state["obstacle_distance"]),
                "speed": float(state["ee_speed"]),
            })
            print(f"  step {step:3d}: ee={state['ee_pos']}, obs_dist={state['obstacle_distance']:.4f}, speed={state['ee_speed']:.4f}")

        # Compute controller action
        action = controller_ep.compute_action(
            ee_pos=state["ee_pos"],
            goal_pos=state["goal_pos"],
            qpos=state["qpos"],
            dt=dt,
            noise_ee=ee_noise,
            noise_joint=joint_noise,
            obstacle_distance=state["obstacle_distance"],
            obstacle_direction=state["obstacle_direction"],
        )

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Render if visualization is enabled
        if render:
            env.render()
            if step_by_step:
                input(f"  [Step {step}] Press Enter to continue...")

        # Check success from the unwrapped evaluate
        eval_result = base_env.evaluate()
        # Handle both tensor and numpy success flags
        succ = eval_result["success"]
        if hasattr(succ, 'any'):
            if succ.any():
                success = True
                break
        elif succ:
            success = True
            break

        if terminated or truncated:
            break

    print(f"  Min obstacle dist: {min_obs_dist:.4f}")

    # Check timeout
    elapsed = step * dt
    monitor.check_timeout(elapsed)

    # Root-cause analysis
    violations_list = [v.to_dict() for v in monitor.violations]
    episode_result = {
        "success": success,
        "steps": step + 1,
        "max_steps": max_steps,
        "dt": dt,
        "min_obs_dist": min_obs_dist,
    }
    attributions = analyzer.analyze(
        monitor.violations, scenario,
        trajectory=trajectory,
        episode_result=episode_result,
    )

    return {
        "success": success,
        "steps": step + 1,
        "violations": violations_list,
        "attributions": attributions,
        "scenario": scenario,
        "summary": monitor.get_summary(),
    }


# ------------------------------------------------------------------ #
#  Experiment Modes
# ------------------------------------------------------------------ #

# Fixed scene geometry (same for every scenario)
FIXED_OBSTACLE = {
    "position": {"x": 0.05, "y": 0.0, "z": 0.0},
    "half_size": [0.15, 0.015, 0.20],
    "barrier_enabled": False,
}
FIXED_GOAL = {
    "position": {"x": 0.05, "y": 0.40, "z": 0.10},
}
DEFAULT_EE_START_Y = -0.40


def load_scenario(scenario_path: str) -> dict:
    """Load a scenario from a YAML file.

    Scenarios only specify: ee_start, sensor_noise, control.
    Wall and goal are fixed.
    """
    with open(scenario_path) as f:
        raw = yaml.safe_load(f)

    scenario = {
        "obstacle": FIXED_OBSTACLE,
        "goal": FIXED_GOAL,
        "sensor_noise": raw.get("sensor_noise", {"ee_position_std": 0.0, "joint_position_std": 0.0}),
        "control": raw.get("control", {"delay_steps": 0, "gain_scale": 1.0}),
        "ee_start": raw.get("ee_start", {"y": DEFAULT_EE_START_Y}),
        "reasoning": raw.get("strategy", raw.get("reasoning", "")),
    }
    return scenario


def run_single_test(
    constraints_path: str = "constraints.yaml",
    render: bool = False,
    step_by_step: bool = False,
    scenario_path: str = None,
):
    """Run a single scenario from a YAML file or the default."""
    print("=" * 60)
    if scenario_path:
        print(f"SCENARIO: {scenario_path}")
    else:
        print("SINGLE TEST MODE (default scenario)")
    if render:
        print("  (with visualization)")
    print("=" * 60)

    env_kwargs = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )
    if render:
        env_kwargs["render_mode"] = "human"

    env = gym.make("FrankaReachObstacle-v0", **env_kwargs)
    env = CPUGymWrapper(env)
    controller = PDEEController()
    monitor = SafetyMonitor(constraints_path)
    analyzer = RootCauseAnalyzer()

    with open(constraints_path) as f:
        constraints = yaml.safe_load(f)["constraints"]

    if scenario_path:
        scenario = load_scenario(scenario_path)
    else:
        scenario = {
            "obstacle": FIXED_OBSTACLE,
            "goal": FIXED_GOAL,
            "sensor_noise": {"ee_position_std": 0.0, "joint_position_std": 0.0},
            "control": {"delay_steps": 0, "gain_scale": 2.0},
            "ee_start": {"y": DEFAULT_EE_START_Y},
            "reasoning": "Default scenario",
        }

    result = run_episode(env, controller, monitor, analyzer, scenario, render=render, step_by_step=step_by_step, constraints=constraints)

    print(f"\nSuccess: {result['success']}")
    print(f"Steps: {result['steps']}")
    print(f"Violations: {result['summary']}")
    if result["attributions"]:
        print(f"Primary cause: {result['attributions'][0]['subsystem']}")
        print(f"Reasoning: {result['attributions'][0]['reasoning']}")

    env.close()
    return result


def run_full_experiment(
    mode: str = "llm",
    num_episodes: int = 100,
    constraints_path: str = "constraints.yaml",
    api_key: str = None,
):
    """Run a full experiment (LLM adversarial or random baseline)."""
    print("=" * 60)
    print(f"FULL EXPERIMENT: {mode.upper()} | {num_episodes} episodes")
    print("=" * 60)

    env = gym.make(
        "FrankaReachObstacle-v0",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )
    env = CPUGymWrapper(env)
    controller = PDEEController()
    monitor = SafetyMonitor(constraints_path)
    analyzer = RootCauseAnalyzer()
    logger = ExperimentLogger(output_dir=f"results_{mode}")

    with open(constraints_path) as f:
        constraints = yaml.safe_load(f)["constraints"]

    for ep in range(num_episodes):
        # Generate scenario
        if mode == "llm":
            try:
                scenario = generate_adversarial_scenario_llm(
                    constraints_path=constraints_path,
                    episode_num=ep,
                    violation_history=logger.get_violation_history(),
                    api_key=api_key,
                )
            except Exception as e:
                print(f"  LLM generation failed (ep {ep}): {e}")
                scenario = generate_random_scenario(constraints_path)
        else:
            scenario = generate_random_scenario(constraints_path)

        # Run episode
        result = run_episode(env, controller, monitor, analyzer, scenario, constraints=constraints)

        # Log
        logger.log_episode(
            episode_num=ep,
            scenario=result["scenario"],
            violations=result["violations"],
            attributions=result["attributions"],
            success=result["success"],
            steps=result["steps"],
            metadata={"mode": mode},
        )

        # Progress
        v_count = len(result["violations"])
        status = "PASS" if result["success"] else "FAIL"
        v_str = f"{v_count} violations" if v_count else "clean"
        print(f"  Episode {ep:3d} | {status} | {v_str} | strategy: {scenario.get('reasoning', 'n/a')[:50]}")

    # Save results
    logger.save(f"{mode}_results.json")

    # Print summary
    stats = logger.get_aggregate_stats()
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY ({mode.upper()})")
    print(f"{'=' * 60}")
    print(f"  Episodes:         {stats['total_episodes']}")
    print(f"  Violation rate:   {stats['violation_rate']:.1%}")
    print(f"  Success rate:     {stats['success_rate']:.1%}")
    print(f"  Avg steps to 1st: {stats['avg_steps_to_first_violation']}")
    print(f"  Violation types:  {stats['violation_type_distribution']}")
    print(f"  Subsystem blame:  {stats['subsystem_attribution']}")

    env.close()
    return stats


def run_comparison(num_episodes: int = 50, constraints_path: str = "constraints.yaml", api_key: str = None):
    """Run both LLM and random experiments, then compare."""
    print("\n>>> Running LLM adversarial experiment...")
    llm_stats = run_full_experiment("llm", num_episodes, constraints_path, api_key)

    print("\n>>> Running random baseline experiment...")
    random_stats = run_full_experiment("random", num_episodes, constraints_path)

    print(f"\n{'=' * 60}")
    print("COMPARISON: LLM ADVERSARIAL vs RANDOM BASELINE")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<30} {'LLM':>12} {'Random':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Violation rate':<30} {llm_stats['violation_rate']:>11.1%} {random_stats['violation_rate']:>11.1%}")
    print(f"  {'Success rate':<30} {llm_stats['success_rate']:>11.1%} {random_stats['success_rate']:>11.1%}")

    llm_avg = llm_stats.get("avg_steps_to_first_violation")
    rnd_avg = random_stats.get("avg_steps_to_first_violation")
    print(f"  {'Avg steps to 1st violation':<30} {str(llm_avg):>12} {str(rnd_avg):>12}")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Red-Team Robot Experiment Runner")
    parser.add_argument("--mode", choices=["single", "full", "baseline", "compare"], default="single")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--constraints", type=str, default="constraints.yaml")
    parser.add_argument("--api-key", type=str, default=None, help="Anthropic API key for LLM mode")
    parser.add_argument("--render", action="store_true", help="Open GUI viewer to visualize the experiment")
    parser.add_argument("--step", action="store_true", help="Step-by-step mode: press Enter to advance each step (requires --render)")
    parser.add_argument("--scenario", type=str, default=None, help="Path to a scenario YAML file")
    args = parser.parse_args()

    if args.mode == "single":
        run_single_test(args.constraints, render=args.render, step_by_step=args.step, scenario_path=args.scenario)
    elif args.mode == "full":
        run_full_experiment("llm", args.episodes, args.constraints, args.api_key)
    elif args.mode == "baseline":
        run_full_experiment("random", args.episodes, args.constraints)
    elif args.mode == "compare":
        run_comparison(args.episodes, args.constraints, args.api_key)