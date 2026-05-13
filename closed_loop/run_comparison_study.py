"""
Comparison Study: LLM Adversarial vs Random Baseline
=====================================================
Single-script entry point that:
  1. Generates and runs N episodes of LLM-generated adversarial scenarios
  2. Generates and runs N episodes of random baseline scenarios
  3. Loads both result JSONs and computes ground-truth metrics
  4. Writes a CSV summary table and matplotlib figures suitable for the
     final report and slides 12-13

Usage:
    # Full study (generate + run + analyze)
    python run_comparison_study.py --episodes 30 --api-key $ANTHROPIC_API_KEY

    # Re-analyze existing results without re-running the simulator
    python run_comparison_study.py --skip-run \\
        --llm-results results_llm/llm_results.json \\
        --random-results results_random/random_results.json
"""

import argparse
import csv
import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


CONSTRAINT_TYPES = [
    "obstacle_proximity",
    "ee_velocity_exceeded",
    "joint_velocity_exceeded",
    "joint_limit_low",
    "joint_limit_high",
    "workspace_boundary",
    "timeout",
]

SUBSYSTEMS = ["perception", "control", "planning"]


# ------------------------------------------------------------------ #
#  Loading
# ------------------------------------------------------------------ #

def load_results(path: str) -> dict:
    """Load an ExperimentLogger JSON file."""
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Metrics
# ------------------------------------------------------------------ #

def compute_metrics(results: dict, dt: float) -> dict:
    """Compute all comparison metrics from a results JSON."""
    episodes = results["episodes"]
    n = len(episodes)
    if n == 0:
        return {"n": 0}

    # Overall counts
    n_with_violations = sum(1 for e in episodes if e["num_violations"] > 0)
    n_success = sum(1 for e in episodes if e["success"])

    # Per-type rate (fraction of episodes that hit each type at least once)
    per_type_rate: Dict[str, float] = {}
    for t in CONSTRAINT_TYPES:
        hits = sum(
            1 for e in episodes
            if any(v["type"] == t for v in e["violations"])
        )
        per_type_rate[t] = hits / n

    # Mean total violations per episode
    mean_violations = float(np.mean([e["num_violations"] for e in episodes]))

    # Time-to-first-violation (seconds), only for episodes with violations
    ttf_seconds: List[float] = []
    for e in episodes:
        if e["violations"]:
            first_step = e["violations"][0]["timestep"]
            ttf_seconds.append(first_step * dt)

    # Subsystem attribution (over all violations across all episodes)
    subsystem_counter: Counter = Counter()
    for e in episodes:
        for a in e["attributions"]:
            subsystem_counter[a["subsystem"]] += 1
    subsystem_total = sum(subsystem_counter.values())
    subsystem_dist = {
        s: subsystem_counter.get(s, 0) / subsystem_total if subsystem_total else 0.0
        for s in SUBSYSTEMS
    }

    # Bootstrap 95% CI on overall violation rate
    rng = np.random.default_rng(0)
    boot_rates = []
    indicators = np.array([1 if e["num_violations"] > 0 else 0 for e in episodes])
    for _ in range(1000):
        sample = rng.choice(indicators, size=n, replace=True)
        boot_rates.append(sample.mean())
    ci_low, ci_high = np.percentile(boot_rates, [2.5, 97.5])

    return {
        "n": n,
        "violation_rate": n_with_violations / n,
        "violation_rate_ci": (float(ci_low), float(ci_high)),
        "goal_reach_rate": n_success / n,
        "mean_violations_per_episode": mean_violations,
        "per_type_rate": per_type_rate,
        "ttf_seconds_mean": float(np.mean(ttf_seconds)) if ttf_seconds else None,
        "ttf_seconds_median": float(np.median(ttf_seconds)) if ttf_seconds else None,
        "ttf_seconds_values": ttf_seconds,
        "subsystem_distribution": subsystem_dist,
        "subsystem_counts": dict(subsystem_counter),
    }


# ------------------------------------------------------------------ #
#  CSV summary
# ------------------------------------------------------------------ #

def write_summary_csv(llm: dict, rnd: dict, out_path: str):
    """Write side-by-side comparison CSV with one row per metric."""
    rows = [
        ("episodes", llm["n"], rnd["n"]),
        ("violation_rate", f"{llm['violation_rate']:.4f}", f"{rnd['violation_rate']:.4f}"),
        (
            "violation_rate_ci_95",
            f"[{llm['violation_rate_ci'][0]:.3f}, {llm['violation_rate_ci'][1]:.3f}]",
            f"[{rnd['violation_rate_ci'][0]:.3f}, {rnd['violation_rate_ci'][1]:.3f}]",
        ),
        ("goal_reach_rate", f"{llm['goal_reach_rate']:.4f}", f"{rnd['goal_reach_rate']:.4f}"),
        (
            "mean_violations_per_episode",
            f"{llm['mean_violations_per_episode']:.3f}",
            f"{rnd['mean_violations_per_episode']:.3f}",
        ),
        (
            "ttf_seconds_mean",
            f"{llm['ttf_seconds_mean']:.3f}" if llm["ttf_seconds_mean"] is not None else "n/a",
            f"{rnd['ttf_seconds_mean']:.3f}" if rnd["ttf_seconds_mean"] is not None else "n/a",
        ),
        (
            "ttf_seconds_median",
            f"{llm['ttf_seconds_median']:.3f}" if llm["ttf_seconds_median"] is not None else "n/a",
            f"{rnd['ttf_seconds_median']:.3f}" if rnd["ttf_seconds_median"] is not None else "n/a",
        ),
    ]
    for t in CONSTRAINT_TYPES:
        rows.append((
            f"rate_{t}",
            f"{llm['per_type_rate'][t]:.4f}",
            f"{rnd['per_type_rate'][t]:.4f}",
        ))
    for s in SUBSYSTEMS:
        rows.append((
            f"subsystem_{s}",
            f"{llm['subsystem_distribution'][s]:.4f}",
            f"{rnd['subsystem_distribution'][s]:.4f}",
        ))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "llm", "random"])
        writer.writerows(rows)
    print(f"Summary CSV: {out_path}")


# ------------------------------------------------------------------ #
#  Plots
# ------------------------------------------------------------------ #

def _save(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {path}")


def plot_overall_metrics(llm: dict, rnd: dict, out_path: str):
    """Slide-12 grouped bar chart: the four headline metrics, normalized to [0,1]
    where possible (TtF is shown on a secondary scale below)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: violation rate, goal-reach rate (both fractions)
    labels = ["Violation\nrate", "Goal-reach\nrate"]
    llm_vals = [llm["violation_rate"], llm["goal_reach_rate"]]
    rnd_vals = [rnd["violation_rate"], rnd["goal_reach_rate"]]
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w / 2, llm_vals, w, label="LLM", color="#d62728")
    axes[0].bar(x + w / 2, rnd_vals, w, label="Random", color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of episodes")
    axes[0].set_title("Overall outcomes")
    axes[0].legend()
    for i, (a, b) in enumerate(zip(llm_vals, rnd_vals)):
        axes[0].text(i - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=9)
        axes[0].text(i + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=9)

    # Right: mean violations + TtF (different scale, dual axis)
    ax2 = axes[1]
    ax2b = ax2.twinx()
    metrics = ["Mean violations\nper episode", "Time-to-first-\nviolation (s)"]
    llm_v = [
        llm["mean_violations_per_episode"],
        llm["ttf_seconds_mean"] if llm["ttf_seconds_mean"] is not None else 0.0,
    ]
    rnd_v = [
        rnd["mean_violations_per_episode"],
        rnd["ttf_seconds_mean"] if rnd["ttf_seconds_mean"] is not None else 0.0,
    ]
    x = np.arange(len(metrics))
    # mean violations on left axis
    ax2.bar(x[0] - w / 2, llm_v[0], w, color="#d62728")
    ax2.bar(x[0] + w / 2, rnd_v[0], w, color="#1f77b4")
    ax2.set_ylabel("Mean violations / episode")
    # TtF on right axis
    ax2b.bar(x[1] - w / 2, llm_v[1], w, color="#d62728")
    ax2b.bar(x[1] + w / 2, rnd_v[1], w, color="#1f77b4")
    ax2b.set_ylabel("Time-to-first-violation (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_title("Severity & speed of failure")

    fig.suptitle("LLM adversarial vs random baseline — ReachGoal", y=1.02)
    _save(fig, out_path)


def plot_per_constraint_rates(llm: dict, rnd: dict, out_path: str):
    """Grouped bar chart of per-constraint violation rates."""
    types = CONSTRAINT_TYPES
    llm_vals = [llm["per_type_rate"][t] for t in types]
    rnd_vals = [rnd["per_type_rate"][t] for t in types]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(types))
    w = 0.35
    ax.bar(x - w / 2, llm_vals, w, label="LLM", color="#d62728")
    ax.bar(x + w / 2, rnd_vals, w, label="Random", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in types], fontsize=9)
    ax.set_ylabel("Fraction of episodes that violated")
    ax.set_title("Violation rate by constraint type")
    ax.set_ylim(0, max(max(llm_vals + rnd_vals) * 1.25, 0.1))
    ax.legend()
    for i, (a, b) in enumerate(zip(llm_vals, rnd_vals)):
        if a > 0:
            ax.text(i - w / 2, a + 0.005, f"{a:.2f}", ha="center", fontsize=8)
        if b > 0:
            ax.text(i + w / 2, b + 0.005, f"{b:.2f}", ha="center", fontsize=8)
    _save(fig, out_path)


def plot_subsystem_donuts(llm: dict, rnd: dict, out_path: str):
    """Side-by-side donut charts of subsystem attribution (slide 13)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    colors = {"perception": "#1f77b4", "control": "#d62728", "planning": "#2ca02c"}

    for ax, metrics, title in [
        (axes[0], llm, "LLM adversarial"),
        (axes[1], rnd, "Random baseline"),
    ]:
        dist = metrics["subsystem_distribution"]
        sizes = [dist[s] for s in SUBSYSTEMS]
        if sum(sizes) == 0:
            ax.text(0.5, 0.5, "no violations", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
            continue
        labels = [f"{s}\n{dist[s]*100:.0f}%" for s in SUBSYSTEMS]
        wedges, _ = ax.pie(
            sizes,
            labels=labels,
            colors=[colors[s] for s in SUBSYSTEMS],
            wedgeprops=dict(width=0.4, edgecolor="white"),
            startangle=90,
        )
        ax.set_title(title)

    fig.suptitle("Root-cause subsystem attribution")
    _save(fig, out_path)


def plot_ttf_distribution(llm: dict, rnd: dict, out_path: str):
    """Box plot of time-to-first-violation distributions."""
    llm_ttf = llm["ttf_seconds_values"]
    rnd_ttf = rnd["ttf_seconds_values"]
    fig, ax = plt.subplots(figsize=(7, 4))
    data, labels = [], []
    if llm_ttf:
        data.append(llm_ttf)
        labels.append(f"LLM (n={len(llm_ttf)})")
    if rnd_ttf:
        data.append(rnd_ttf)
        labels.append(f"Random (n={len(rnd_ttf)})")
    if not data:
        ax.text(0.5, 0.5, "no violations recorded in either arm",
                ha="center", va="center")
        ax.axis("off")
    else:
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.set_ylabel("Time-to-first-violation (s)")
        ax.set_title("Distribution of time-to-first-violation\n(lower = adversary triggered failure faster)")
    _save(fig, out_path)


# ------------------------------------------------------------------ #
#  Driver
# ------------------------------------------------------------------ #

def run_preset_arm(preset_dir: str, episodes: int, constraints_path: str,
                   seed: int, output_subdir: str = "results_llm",
                   results_filename: str = "llm_results.json") -> str:
    """Run the LLM arm from a directory of pre-generated scenario YAMLs.

    Walks the directory, loads each YAML, and runs it through the same
    pipeline used by run_experiment.run_full_experiment. If `episodes`
    exceeds the number of preset files, the list cycles. Returns the
    JSON results path.
    """
    import glob
    import gymnasium as gym
    from env.franka_reach_obstacle import FrankaReachObstacleEnv  # noqa: F401
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
    from controller.pd_ee_controller import PDEEController
    from modules.safety_monitor import SafetyMonitor
    from modules.root_cause_analyzer import RootCauseAnalyzer
    from modules.experiment_logger import ExperimentLogger
    from run_experiment import run_episode, load_scenario

    np.random.seed(seed)
    random.seed(seed)

    yaml_paths = sorted(glob.glob(os.path.join(preset_dir, "*.yaml")))
    if not yaml_paths:
        raise SystemExit(f"No scenario YAMLs found in {preset_dir}")

    print(f"\n{'#' * 60}")
    print(f"# Running LLM (preset) arm: {episodes} episodes from {preset_dir}")
    print(f"# Found {len(yaml_paths)} scenario files")
    print(f"{'#' * 60}")

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
    logger = ExperimentLogger(output_dir=output_subdir)

    with open(constraints_path) as f:
        constraints = yaml.safe_load(f)["constraints"]

    for ep in range(episodes):
        scen_path = yaml_paths[ep % len(yaml_paths)]
        scenario = load_scenario(scen_path)
        result = run_episode(env, controller, monitor, analyzer, scenario,
                             constraints=constraints)
        logger.log_episode(
            episode_num=ep,
            scenario=result["scenario"],
            violations=result["violations"],
            attributions=result["attributions"],
            success=result["success"],
            steps=result["steps"],
            metadata={"mode": "llm_preset", "scenario_file": os.path.basename(scen_path)},
        )
        v_count = len(result["violations"])
        status = "PASS" if result["success"] else "FAIL"
        v_str = f"{v_count} violations" if v_count else "clean"
        print(f"  Episode {ep:3d} | {status} | {v_str} | {os.path.basename(scen_path)}")

    logger.save(results_filename)
    env.close()
    return os.path.join(output_subdir, results_filename)


def run_random_arm(episodes: int, constraints_path: str, seed: int) -> str:
    """Run the random baseline arm via run_experiment.run_full_experiment."""
    from run_experiment import run_full_experiment
    np.random.seed(seed)
    random.seed(seed)
    print(f"\n{'#' * 60}\n# Running random baseline arm ({episodes} episodes)\n{'#' * 60}")
    run_full_experiment(
        mode="random",
        num_episodes=episodes,
        constraints_path=constraints_path,
        api_key=None,
    )
    return os.path.join("results_random", "random_results.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per arm (default 30); overridden by --llm-episodes / --random-episodes if provided")
    parser.add_argument("--llm-episodes", type=int, default=None,
                        help="Override episode count for the LLM arm only")
    parser.add_argument("--random-episodes", type=int, default=None,
                        help="Override episode count for the random arm only")
    parser.add_argument("--llm-preset-dir", type=str,
                        default="scenarios/llm_preset",
                        help="Directory of pre-generated LLM scenario YAMLs")
    parser.add_argument("--constraints", type=str, default="constraints.yaml")
    parser.add_argument("--output-dir", type=str, default="study_output",
                        help="Directory for CSV + figures")
    parser.add_argument("--skip-run", action="store_true",
                        help="Skip simulation; only analyze existing JSONs")
    parser.add_argument("--llm-results", type=str,
                        default="results_llm/llm_results.json")
    parser.add_argument("--random-results", type=str,
                        default="results_random/random_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1) Run simulator arms (or skip)
    if not args.skip_run:
        n_llm = args.llm_episodes if args.llm_episodes is not None else args.episodes
        n_random = args.random_episodes if args.random_episodes is not None else args.episodes
        llm_path = run_preset_arm(
            args.llm_preset_dir, n_llm, args.constraints, args.seed,
        )
        random_path = run_random_arm(n_random, args.constraints, args.seed)
    else:
        llm_path = args.llm_results
        random_path = args.random_results
        for p in (llm_path, random_path):
            if not os.path.exists(p):
                raise SystemExit(f"Missing results file: {p}")

    # 2) Load + compute metrics
    with open(args.constraints) as f:
        dt = yaml.safe_load(f)["constraints"].get("dt", 0.05)

    llm_results = load_results(llm_path)
    rnd_results = load_results(random_path)
    llm_metrics = compute_metrics(llm_results, dt)
    rnd_metrics = compute_metrics(rnd_results, dt)

    # 3) Write CSV + figures
    csv_path = os.path.join(args.output_dir, "comparison_summary.csv")
    write_summary_csv(llm_metrics, rnd_metrics, csv_path)

    plot_overall_metrics(llm_metrics, rnd_metrics,
                         os.path.join(fig_dir, "overall_metrics.png"))
    plot_per_constraint_rates(llm_metrics, rnd_metrics,
                              os.path.join(fig_dir, "per_constraint_rates.png"))
    plot_subsystem_donuts(llm_metrics, rnd_metrics,
                          os.path.join(fig_dir, "subsystem_attribution.png"))
    plot_ttf_distribution(llm_metrics, rnd_metrics,
                          os.path.join(fig_dir, "time_to_failure.png"))

    # 4) Console summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  {'metric':<32} {'LLM':>12} {'Random':>12}")
    print(f"  {'-' * 58}")
    print(f"  {'episodes':<32} {llm_metrics['n']:>12d} {rnd_metrics['n']:>12d}")
    print(f"  {'violation_rate':<32} {llm_metrics['violation_rate']:>12.3f} {rnd_metrics['violation_rate']:>12.3f}")
    print(f"  {'goal_reach_rate':<32} {llm_metrics['goal_reach_rate']:>12.3f} {rnd_metrics['goal_reach_rate']:>12.3f}")
    print(f"  {'mean_violations_per_episode':<32} {llm_metrics['mean_violations_per_episode']:>12.3f} {rnd_metrics['mean_violations_per_episode']:>12.3f}")

    def _fmt(x):
        return f"{x:>12.3f}" if isinstance(x, (int, float)) else f"{'n/a':>12}"

    print(f"  {'ttf_seconds_mean':<32} {_fmt(llm_metrics['ttf_seconds_mean'])} {_fmt(rnd_metrics['ttf_seconds_mean'])}")
    print(f"  {'ttf_seconds_median':<32} {_fmt(llm_metrics['ttf_seconds_median'])} {_fmt(rnd_metrics['ttf_seconds_median'])}")
    print(f"\n  Per-constraint violation rate:")
    for t in CONSTRAINT_TYPES:
        print(f"    {t:<30} {llm_metrics['per_type_rate'][t]:>12.3f} {rnd_metrics['per_type_rate'][t]:>12.3f}")
    print(f"\n  Subsystem attribution:")
    for s in SUBSYSTEMS:
        print(f"    {s:<30} {llm_metrics['subsystem_distribution'][s]:>12.3f} {rnd_metrics['subsystem_distribution'][s]:>12.3f}")
    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
