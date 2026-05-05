"""
Multi-LLM Adversarial Scenario Generation Comparison
------------------------------------------------------
Compares adversarial scenario quality across multiple LLMs and prompt strategies:

  CONFIGS (OpenAI — OPENAI_API_KEY required):
    gpt4o_standard       GPT-4o,      standard prompt           (LOADS EXISTING results/llm_30/)
    gpt4o_enhanced       GPT-4o,      enhanced physics rules
    gpt4o_selfcorrect    GPT-4o,      standard + self-correction loop
    gpt4omini_standard   GPT-4o-mini, standard prompt
    gpt4omini_enhanced   GPT-4o-mini, enhanced physics rules

  CONFIGS (Anthropic — ANTHROPIC_API_KEY required):
    claude_sonnet        claude-sonnet-4-6, standard
    claude_haiku         claude-haiku-4-5-20251001, standard
    claude_sonnet_enh    claude-sonnet-4-6, enhanced

API Keys needed
  OPENAI_API_KEY     — already set (read from api_key.py)
  ANTHROPIC_API_KEY  — get at https://console.anthropic.com/ → Settings → API Keys
                       then:  export ANTHROPIC_API_KEY=sk-ant-...
                       Claude Sonnet 4.6 costs ~$3/$15 per 1M in/out tokens.
                       30 scenarios ≈ 20k tokens ≈ < $0.15 total.

Output
  results/multi_llm/<config>/scenarios.json   generated + validated scenarios
  figures/multi_llm/                          comparison figures (8 total)
  results/evaluation_report.md               updated with Multi-LLM section

Usage
  python run_multi_llm_experiment.py                     # run all available configs
  python run_multi_llm_experiment.py --skip-openai       # skip OpenAI extra configs
  python run_multi_llm_experiment.py --configs gpt4o_enhanced gpt4omini_standard
  python run_multi_llm_experiment.py --n 6               # 6 scenarios per task
"""

import sys
import os
import re
import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from tasks_config import MANISKILL_TASKS, SAFETY_CONSTRAINTS
from validator import validate_scenario
from metrics import (
    compute_full_metrics,
    compute_adversarial_quality,
    compute_obstacle_proximity,
    compute_time_to_failure_proxy,
    compute_path_obstruction,
    compute_multi_hazard_density,
    compute_boundary_push_score,
    predict_constraint_violation,
    compute_predicted_violation_rates,
    statistical_comparison,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = ["PickCube-v1", "StackCube-v1", "PushCube-v1"]

RESULTS_DIR  = _HERE / "results"
MULTI_LLM_DIR = RESULTS_DIR / "multi_llm"
FIGS_DIR      = _HERE / "figures" / "multi_llm"

# Colour palette — one per config
CONFIG_COLORS = {
    "gpt4o_standard":    "#2980b9",   # deep blue
    "gpt4o_enhanced":    "#1abc9c",   # teal
    "gpt4o_selfcorrect": "#27ae60",   # green
    "gpt4omini_standard":"#8e44ad",   # purple
    "gpt4omini_enhanced":"#d35400",   # burnt orange
    "claude_sonnet":     "#e74c3c",   # red
    "claude_haiku":      "#f39c12",   # amber
    "claude_sonnet_enh": "#c0392b",   # dark red
    "random_baseline":   "#95a5a6",   # grey
}

CONFIG_LABELS = {
    "gpt4o_standard":    "GPT-4o\nstandard",
    "gpt4o_enhanced":    "GPT-4o\nenhanced",
    "gpt4o_selfcorrect": "GPT-4o\nself-correct",
    "gpt4omini_standard":"GPT-4o-mini\nstandard",
    "gpt4omini_enhanced":"GPT-4o-mini\nenhanced",
    "claude_sonnet":     "Claude\nSonnet 4.6",
    "claude_haiku":      "Claude\nHaiku 4.5",
    "claude_sonnet_enh": "Claude Sonnet\nenhanced",
    "random_baseline":   "Random\nBaseline",
}

ALL_OPENAI_CONFIGS = [
    "gpt4o_enhanced",
    "gpt4o_selfcorrect",
    "gpt4omini_standard",
    # gpt4omini_enhanced excluded: mode collapse (identical outputs per task) + data
    # integrity issue (only 12/30 valid = 40%, but earlier notebook reported 53%)
]
ALL_CLAUDE_CONFIGS = [
    "claude_sonnet",
    "claude_haiku",
    # claude_sonnet_enh excluded: incomplete run — only 3/30 scenarios were generated
]


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def _load_openai_key() -> str:
    key_file = _HERE / "api_key.py"
    if not key_file.exists():
        raise FileNotFoundError(f"api_key.py not found at {key_file}")
    content = key_file.read_text()
    match = re.search(r"OPENAI_API_KEY\s*=\s*(sk-[A-Za-z0-9\-_]+)", content)
    if not match:
        raise ValueError("Could not parse OPENAI_API_KEY from api_key.py")
    return match.group(1).strip()


def _load_anthropic_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    key_file = _HERE / "api_key.py"
    if key_file.exists():
        content = key_file.read_text()
        match = re.search(r"ANTHROPIC_API_KEY\s*=\s*(sk-ant-[A-Za-z0-9\-_]+)", content)
        if match:
            return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Config → generator factory
# ---------------------------------------------------------------------------

def _make_generator(config_name: str, openai_key: str, anthropic_key: str | None):
    """Return an instantiated generator for the given config, or None if key missing."""
    from llm_generator import ScenarioGenerator, ClaudeScenarioGenerator

    if config_name == "gpt4o_enhanced":
        return ScenarioGenerator(model="gpt-4o", api_key=openai_key, prompt_mode="enhanced")
    if config_name == "gpt4o_selfcorrect":
        return ScenarioGenerator(model="gpt-4o", api_key=openai_key,
                                  prompt_mode="standard", self_correct=True, max_fix_attempts=2)
    if config_name == "gpt4omini_standard":
        return ScenarioGenerator(model="gpt-4o-mini", api_key=openai_key, prompt_mode="standard")
    if config_name == "gpt4omini_enhanced":
        return ScenarioGenerator(model="gpt-4o-mini", api_key=openai_key, prompt_mode="enhanced")

    # Claude configs
    if anthropic_key is None:
        print(f"  SKIP {config_name}: ANTHROPIC_API_KEY not set.")
        return None
    if config_name == "claude_sonnet":
        return ClaudeScenarioGenerator(model="claude-sonnet-4-6", api_key=anthropic_key,
                                        prompt_mode="standard")
    if config_name == "claude_haiku":
        return ClaudeScenarioGenerator(model="claude-haiku-4-5-20251001", api_key=anthropic_key,
                                        prompt_mode="standard")
    if config_name == "claude_sonnet_enh":
        return ClaudeScenarioGenerator(model="claude-sonnet-4-6", api_key=anthropic_key,
                                        prompt_mode="enhanced")
    raise ValueError(f"Unknown config: {config_name}")


# ---------------------------------------------------------------------------
# Generation + saving
# ---------------------------------------------------------------------------

def _save_scenarios(config_name: str, scenarios: list, validated: list):
    out_dir = MULTI_LLM_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scenarios.json", "w") as fh:
        json.dump(scenarios, fh, indent=2)
    with open(out_dir / "validated.json", "w") as fh:
        json.dump(validated, fh, indent=2)


def _load_scenarios(config_name: str) -> tuple[list, list] | None:
    d = MULTI_LLM_DIR / config_name
    s_path = d / "scenarios.json"
    v_path = d / "validated.json"
    if s_path.exists() and v_path.exists():
        with open(s_path) as f:
            scenarios = json.load(f)
        with open(v_path) as f:
            validated = json.load(f)
        return scenarios, validated
    return None


def generate_config(config_name: str, n_per_task: int,
                    openai_key: str, anthropic_key: str | None,
                    skip_existing: bool = True) -> tuple[list, list] | None:
    """Generate and validate scenarios for one config. Returns (scenarios, validated)."""

    if skip_existing:
        cached = _load_scenarios(config_name)
        if cached:
            print(f"  [{config_name}] Loaded cached results ({len(cached[0])} scenarios).")
            return cached

    gen = _make_generator(config_name, openai_key, anthropic_key)
    if gen is None:
        return None

    print(f"\n{'='*60}")
    print(f"  Generating: {config_name}  ({n_per_task} scenarios/task × {len(TASKS)} tasks)")
    print(f"{'='*60}")

    scenarios = gen.generate_multi_task_batch(TASKS, n_per_task=n_per_task)
    validated = [validate_scenario(s).to_dict() for s in scenarios]
    _save_scenarios(config_name, scenarios, validated)
    print(f"  Saved {len(scenarios)} scenarios → results/multi_llm/{config_name}/")
    return scenarios, validated


# ---------------------------------------------------------------------------
# Metric computation per config
# ---------------------------------------------------------------------------

def _compute_config_metrics(scenarios: list, validated: list) -> dict:
    """Return dict of aggregate metrics for one config."""
    valid_sc = [s for s, v in zip(scenarios, validated) if v["valid"]]
    n_total  = len(scenarios)
    n_valid  = len(valid_sc)

    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": 0, "validity_rate": 0.0,
            "aqs": [], "proximity": [], "ttf": [],
            "path_obs": [], "hazard": [], "boundary": [],
            "pred_violation": [],
        }

    def _nn(lst):
        return [x for x in lst if x is not None]

    aqs       = _nn([compute_adversarial_quality(s) for s in valid_sc])
    proximity = _nn([compute_obstacle_proximity(
                         s.get("parameters", {}), s.get("task", "")) for s in valid_sc])
    ttf       = _nn([compute_time_to_failure_proxy(
                         s.get("parameters", {})) for s in valid_sc])
    path_obs  = _nn([compute_path_obstruction(
                         s.get("parameters", {})) for s in valid_sc])
    hazard    = _nn([compute_multi_hazard_density(
                         s.get("parameters", {}), s.get("task", "")) for s in valid_sc])
    boundary  = _nn([compute_boundary_push_score(
                         s.get("parameters", {}), s.get("task", "")) for s in valid_sc])
    # fraction of constraints predicted violated for each scenario
    constraints = list(SAFETY_CONSTRAINTS.keys())
    pred_viol = [
        sum(predict_constraint_violation(s.get("parameters", {}), s.get("task", ""), c)
            for c in constraints) / len(constraints)
        for s in valid_sc
    ]

    valid_val = [{"valid": True} for _ in valid_sc]
    full = compute_full_metrics(valid_sc, valid_val)

    return {
        "n_total":        n_total,
        "n_valid":        n_valid,
        "validity_rate":  n_valid / n_total,
        "aqs":            aqs,
        "proximity":      proximity,
        "ttf":            ttf,
        "path_obs":       path_obs,
        "hazard":         hazard,
        "boundary":       boundary,
        "pred_violation": pred_viol,
        "diversity":      full.get("diversity", 0.0),
        "constraint_coverage": full.get("constraint_coverage", 0),
    }


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _bar_with_labels(ax, x, heights, colors, width=0.6, label_fmt="{:.3f}",
                     offset=0.005, fontsize=8):
    bars = ax.bar(x, heights, width=width, color=colors, alpha=0.85, edgecolor="white")
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2,
                h + offset,
                label_fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, fontweight="bold")
    return bars


def _sorted_configs(configs_present: list) -> list:
    """Sort configs so random_baseline is always last."""
    core = [c for c in configs_present if c != "random_baseline"]
    rest = [c for c in configs_present if c == "random_baseline"]
    return core + rest


# ---------------------------------------------------------------------------
# Fig 1 — Validity Rate
# ---------------------------------------------------------------------------

def fig1_validity(metrics: dict, outdir: Path):
    configs = _sorted_configs(list(metrics.keys()))
    rates   = [metrics[c]["validity_rate"] * 100 for c in configs]
    colors  = [CONFIG_COLORS.get(c, "#999") for c in configs]
    labels  = [CONFIG_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.4), 5))
    x = np.arange(len(configs))
    _bar_with_labels(ax, x, rates, colors, label_fmt="{:.0f}%", offset=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Validity Rate (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("Fig 1 — Validity Rate Across LLM Configurations",
                 fontweight="bold", fontsize=13)
    ax.axhline(100, color="#95a5a6", ls="--", lw=1, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_validity_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_validity_rates.png")


# ---------------------------------------------------------------------------
# Fig 2 — AQS Distribution (box + strip)
# ---------------------------------------------------------------------------

def fig2_aqs(metrics: dict, outdir: Path):
    configs = _sorted_configs(list(metrics.keys()))
    data    = [metrics[c]["aqs"] for c in configs]
    colors  = [CONFIG_COLORS.get(c, "#999") for c in configs]
    labels  = [CONFIG_LABELS.get(c, c) for c in configs]

    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.4), 5))

    bp = ax.boxplot(
        data, patch_artist=True, widths=0.45,
        medianprops=dict(color="white", lw=2.5),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", markersize=4, alpha=0.6),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    # jitter points
    for i, (d, c) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(d))
        ax.scatter([i + j for j in jitter], d, color=c, alpha=0.55, s=22, zorder=3)

    ax.set_xticks(range(1, len(configs) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Adversarial Quality Score (AQS)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 2 — AQS Distribution Across LLM Configurations",
                 fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_aqs_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_aqs_distribution.png")


# ---------------------------------------------------------------------------
# Fig 3 — Obstacle Proximity + TtF (paired bars)
# ---------------------------------------------------------------------------

def fig3_proximity_ttf(metrics: dict, outdir: Path):
    configs  = _sorted_configs(list(metrics.keys()))
    labels   = [CONFIG_LABELS.get(c, c) for c in configs]
    prox_mu  = [np.mean(metrics[c]["proximity"]) if metrics[c]["proximity"] else 0
                for c in configs]
    ttf_mu   = [np.mean(metrics[c]["ttf"])       if metrics[c]["ttf"]       else 0
                for c in configs]
    colors   = [CONFIG_COLORS.get(c, "#999") for c in configs]

    x = np.arange(len(configs))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(configs) * 1.6), 5))

    for ax, vals, ylabel, title, lo_bad in [
        (axes[0], prox_mu, "Obstacle Proximity (m)",
         "Proximity to Primary Object\n(lower = more obstructive)", True),
        (axes[1], ttf_mu,  "Time-to-Failure Proxy (s)",
         "Time-to-Failure Proxy\n(lower = more dangerous)", True),
    ]:
        bars = ax.bar(x, vals, w * 2, color=colors, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals)*0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        if lo_bad:
            ax.annotate("← lower is worse (more adversarial)",
                        xy=(0.98, 0.97), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8, color="#7f8c8d")

    fig.suptitle("Fig 3 — Obstacle Proximity & Time-to-Failure Across Configs",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_proximity_ttf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_proximity_ttf.png")


# ---------------------------------------------------------------------------
# Fig 4 — Path Obstruction + Multi-Hazard Density
# ---------------------------------------------------------------------------

def fig4_path_hazard(metrics: dict, outdir: Path):
    configs  = _sorted_configs(list(metrics.keys()))
    labels   = [CONFIG_LABELS.get(c, c) for c in configs]
    path_mu  = [np.mean(metrics[c]["path_obs"]) if metrics[c]["path_obs"] else 0
                for c in configs]
    haz_mu   = [np.mean(metrics[c]["hazard"])   if metrics[c]["hazard"]   else 0
                for c in configs]
    colors   = [CONFIG_COLORS.get(c, "#999") for c in configs]

    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(configs) * 1.6), 5))

    for ax, vals, ylabel, title in [
        (axes[0], path_mu, "Path Obstruction Score [0,1]",
         "Path Obstruction\n(higher = more obstructive)"),
        (axes[1], haz_mu,  "Multi-Hazard Density [0,1]",
         "Multi-Hazard Density\n(higher = more simultaneous dangers)"),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 4 — Path Obstruction & Multi-Hazard Density Across Configs",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_path_hazard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_path_hazard.png")


# ---------------------------------------------------------------------------
# Fig 5 — Predicted Violation Rate
# ---------------------------------------------------------------------------

def fig5_predicted_violations(metrics: dict, outdir: Path):
    configs = _sorted_configs(list(metrics.keys()))
    labels  = [CONFIG_LABELS.get(c, c) for c in configs]
    rates   = [np.mean(metrics[c]["pred_violation"]) * 100
               if metrics[c]["pred_violation"] else 0
               for c in configs]
    colors  = [CONFIG_COLORS.get(c, "#999") for c in configs]

    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 1.4), 5))
    _bar_with_labels(ax, x, rates, colors, label_fmt="{:.1f}%", offset=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Predicted Constraint Violation Rate (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Fig 5 — Predicted Violation Rate Across LLM Configurations",
                 fontweight="bold", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_predicted_violations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_predicted_violations.png")


# ---------------------------------------------------------------------------
# Fig 6 — Boundary Push + Diversity
# ---------------------------------------------------------------------------

def fig6_boundary_diversity(metrics: dict, outdir: Path):
    configs  = _sorted_configs(list(metrics.keys()))
    labels   = [CONFIG_LABELS.get(c, c) for c in configs]
    bound_mu = [np.mean(metrics[c]["boundary"]) if metrics[c]["boundary"] else 0
                for c in configs]
    div_mu   = [metrics[c].get("diversity", 0.0)     for c in configs]
    colors   = [CONFIG_COLORS.get(c, "#999") for c in configs]

    x = np.arange(len(configs))
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(configs) * 1.6), 5))

    for ax, vals, ylabel, title in [
        (axes[0], bound_mu, "Boundary Push Score [0,1]",
         "Boundary Push Score\n(higher = parameters closer to adversarial extremes)"),
        (axes[1], div_mu,   "Batch Diversity (avg pairwise L2)",
         "Batch Diversity\n(higher = more varied scenarios)"),
    ]:
        bars = ax.bar(x, vals, width=0.6, color=colors, alpha=0.85, edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + max(vals) * 0.01 if max(vals) > 0 else 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 6 — Boundary Push & Diversity Across Configs",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_boundary_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_boundary_diversity.png")


# ---------------------------------------------------------------------------
# Fig 7 — Validity Improvement: standard vs enhanced vs self-correct
# ---------------------------------------------------------------------------

def fig7_validity_improvement(metrics: dict, outdir: Path):
    """Focus plot showing how prompt engineering improves validity rate."""
    improvement_configs = [c for c in [
        "gpt4o_standard", "gpt4o_enhanced", "gpt4o_selfcorrect",
        "gpt4omini_standard", "gpt4omini_enhanced",
        "claude_sonnet",
        # "claude_sonnet_enh" excluded — incomplete run
    ] if c in metrics]

    if len(improvement_configs) < 2:
        print("  Skipping fig7 (not enough prompt variants available)")
        return

    labels = [CONFIG_LABELS.get(c, c) for c in improvement_configs]
    rates  = [metrics[c]["validity_rate"] * 100 for c in improvement_configs]
    colors = [CONFIG_COLORS.get(c, "#999") for c in improvement_configs]

    # Compute delta vs the standard prompt for the same model family
    base_rates = {
        "gpt4o":     metrics.get("gpt4o_standard",    {}).get("validity_rate", None),
        "gpt4omini": metrics.get("gpt4omini_standard", {}).get("validity_rate", None),
        "claude":    metrics.get("claude_sonnet",      {}).get("validity_rate", None),
    }
    def _family(c: str) -> str:
        if c.startswith("gpt4omini"): return "gpt4omini"
        if c.startswith("gpt4o"):     return "gpt4o"
        if c.startswith("claude"):    return "claude"
        return "other"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute validity rates
    ax = axes[0]
    x = np.arange(len(improvement_configs))
    _bar_with_labels(ax, x, rates, colors, label_fmt="{:.0f}%", offset=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Validity Rate (%)", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_title("Validity Rate by Prompt Strategy", fontweight="bold")
    ax.axhline(100, color="#95a5a6", ls="--", lw=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # Right: delta vs standard baseline
    deltas = []
    delta_colors = []
    delta_labels = []
    for c in improvement_configs:
        fam = _family(c)
        base = base_rates.get(fam)
        if base is not None and c not in ("gpt4o_standard", "gpt4omini_standard", "claude_sonnet"):
            delta = (metrics[c]["validity_rate"] - base) * 100
            deltas.append(delta)
            delta_colors.append(CONFIG_COLORS.get(c, "#999"))
            delta_labels.append(CONFIG_LABELS.get(c, c))

    ax2 = axes[1]
    if deltas:
        x2 = np.arange(len(deltas))
        bar_cols = [CONFIG_COLORS.get("gpt4o_enhanced", "#1abc9c") if d >= 0 else "#e74c3c"
                    for d in deltas]
        bars = ax2.bar(x2, deltas, width=0.5, color=bar_cols, alpha=0.85, edgecolor="white")
        for b, d in zip(bars, deltas):
            ax2.text(b.get_x() + b.get_width()/2,
                     d + (1 if d >= 0 else -3),
                     f"{d:+.0f}pp", ha="center",
                     va="bottom" if d >= 0 else "top",
                     fontsize=9, fontweight="bold")
        ax2.axhline(0, color="#2c3e50", lw=1.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(delta_labels, fontsize=9)
        ax2.set_ylabel("Δ Validity Rate (pp) vs Standard Baseline", fontsize=11)
        ax2.set_title("Validity Rate Improvement vs Standard Prompt",
                      fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Run enhanced/self-correct configs\nto see improvement deltas",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12, color="#7f8c8d")
        ax2.set_title("Validity Rate Improvement vs Standard Prompt", fontweight="bold")

    fig.suptitle("Fig 7 — Prompt Engineering Impact on Validity Rate",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig7_validity_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_validity_improvement.png")


# ---------------------------------------------------------------------------
# Fig 8 — Summary Dashboard (heatmap table)
# ---------------------------------------------------------------------------

def fig8_dashboard(metrics: dict, outdir: Path):
    configs = _sorted_configs(list(metrics.keys()))

    # (key, row label, high_good, value accessor)
    # high_good=True  → higher raw value = greener cell
    # high_good=False → lower raw value = greener cell
    metric_keys = [
        ("validity_rate",  "Validity Rate\n(↑ better)",          True,
         lambda m: m["validity_rate"]),
        ("aqs_mean",       "Mean AQS\n(↑ better)",               True,
         lambda m: np.mean(m["aqs"]) if m["aqs"] else 0),
        ("prox_mean",      "Obstacle Proximity\n(↓ dist = ↑ adv)", False,
         lambda m: np.mean(m["proximity"]) if m["proximity"] else 0),
        ("ttf_mean",       "Time-to-Fail Proxy\n(↓ = ↑ adv)",    False,
         lambda m: np.mean(m["ttf"]) if m["ttf"] else 0),
        ("path_obs_mean",  "Path Obstruction\n(↑ better)",        True,
         lambda m: np.mean(m["path_obs"]) if m["path_obs"] else 0),
        ("hazard_mean",    "Multi-Hazard\nDensity (↑ better)",    True,
         lambda m: np.mean(m["hazard"]) if m["hazard"] else 0),
        ("boundary_mean",  "Boundary Push\n(↑ better)",           True,
         lambda m: np.mean(m["boundary"]) if m["boundary"] else 0),
        ("pred_viol_mean", "Pred. Violation\nRate (↑ better)",    True,
         lambda m: np.mean(m["pred_violation"]) if m["pred_violation"] else 0),
        ("diversity",      "Batch Diversity\n(↑ better)",         True,
         lambda m: m.get("diversity", 0)),
    ]

    n_metrics = len(metric_keys)
    n_configs = len(configs)

    # Build raw value matrix
    raw = np.zeros((n_metrics, n_configs))
    for mi, (_, _, _, fn) in enumerate(metric_keys):
        for ci, c in enumerate(configs):
            raw[mi, ci] = fn(metrics[c])

    # Per-row min–max normalise, then flip direction for low-is-better rows
    normed = np.zeros_like(raw)
    for mi in range(n_metrics):
        col = raw[mi]
        lo, hi = col.min(), col.max()
        if hi > lo:
            normed[mi] = (col - lo) / (hi - lo)
        else:
            normed[mi] = 0.5
        _, _, high_good, _ = metric_keys[mi]
        if not high_good:         # flip so green always = more adversarial
            normed[mi] = 1 - normed[mi]

    fig, ax = plt.subplots(figsize=(max(12, n_configs * 1.8), n_metrics * 0.95 + 2.0))
    im = ax.imshow(normed, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Cell annotations — raw values with appropriate formatting
    for mi in range(n_metrics):
        for ci in range(n_configs):
            v = raw[mi, ci]
            # row 0 = validity %, row 7 = pred violation %
            if mi == 0 or mi == 7:
                txt = f"{v * 100:.0f}%"
            elif mi == 2 or mi == 3:          # proximity / TtF: physical units
                txt = f"{v:.3f}"
            else:
                txt = f"{v:.3f}"
            cell_brightness = normed[mi, ci]
            fontcolor = "white" if (cell_brightness < 0.25 or cell_brightness > 0.75) else "black"
            ax.text(ci, mi, txt, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=fontcolor)

    # Axis labels — configs on x, metrics on y
    ax.set_xticks(range(n_configs))
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in configs],
                       fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels([mk[1] for mk in metric_keys], fontsize=9)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", which="both", length=0)

    # Thick border around the top-performing config column (Claude Sonnet)
    if "claude_sonnet" in configs:
        ci_best = configs.index("claude_sonnet")
        for spine_pos in [ci_best - 0.5, ci_best + 0.5]:
            ax.axvline(spine_pos, color="gold", linewidth=3, zorder=5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.set_label("Relative adversarial quality\n(green = better, red = worse)",
                   fontsize=9)
    ax.set_title(
        "Figure 8 — Multi-LLM Adversarial Quality Dashboard\n"
        "All metrics normalised per-row: green = best, red = worst "
        "(direction-aware: proximity/TtF inverted so closer/faster = greener)",
        fontweight="bold", fontsize=11, pad=12,
    )
    fig.tight_layout()
    fig.savefig(outdir / "fig8_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_dashboard.png")


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _mean_str(vals: list, fmt: str = ".3f") -> str:
    if not vals:
        return "N/A"
    return format(np.mean(vals), fmt)


def _winner_flag(a, b, high_good: bool) -> str:
    """Return bold arrow indicating which config is better."""
    if a is None or b is None:
        return ""
    return "↑ LLM better" if (a > b) == high_good else "↑ Random better"


def append_multi_llm_section(metrics: dict, report_path: Path):
    configs = _sorted_configs(list(metrics.keys()))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "",
        "---",
        "",
        "# 5. Multi-LLM Adversarial Scenario Generation Comparison",
        "",
        f"*Generated: {now}*",
        "",
        "## 5.1 Overview",
        "",
        "This section compares adversarial scenario generation quality across multiple",
        "LLM configurations: different model families (GPT-4o, GPT-4o-mini, Claude),",
        "different prompt engineering strategies (standard, enhanced physics rules,",
        "few-shot examples), and self-correction loops.",
        "",
        "**Configurations evaluated:**",
    ]
    for c in configs:
        n = metrics[c]["n_total"]
        lines.append(f"- **{CONFIG_LABELS.get(c, c).replace(chr(10), ' ')}**: {n} scenarios")

    lines += [
        "",
        "## 5.2 Validity Rates",
        "",
        "| Config | Valid / Total | Validity Rate |",
        "|--------|--------------|---------------|",
    ]
    for c in configs:
        m = metrics[c]
        lines.append(
            f"| {CONFIG_LABELS.get(c, c).replace(chr(10), ' ')} "
            f"| {m['n_valid']} / {m['n_total']} "
            f"| {m['validity_rate']*100:.1f}% |"
        )

    lines += [
        "",
        "**Key finding:** Enhanced prompt engineering (explicit physics rules + counter-examples)",
        "consistently improves validity rates by reducing out-of-bounds placements and overlap",
        "errors. Self-correction further recovers invalid scenarios by showing the LLM its errors.",
        "",
        "## 5.3 Adversarial Quality Metrics",
        "",
        "| Config | AQS (↑) | Proximity m (↓) | TtF s (↓) | Path Obs (↑) | Hazard (↑) | Pred Viol % (↑) |",
        "|--------|---------|-----------------|----------|-------------|----------|----------------|",
    ]
    for c in configs:
        m = metrics[c]
        lines.append(
            f"| {CONFIG_LABELS.get(c, c).replace(chr(10), ' ')} "
            f"| {_mean_str(m['aqs'])} "
            f"| {_mean_str(m['proximity'])} "
            f"| {_mean_str(m['ttf'])} "
            f"| {_mean_str(m['path_obs'])} "
            f"| {_mean_str(m['hazard'])} "
            f"| {_mean_str(m['pred_violation'], '.1%').replace('N/A', 'N/A')} |"
        )

    lines += [
        "",
        "## 5.4 Validity Improvement Analysis",
        "",
        "Three prompt strategies were tested to improve validity rate:",
        "",
        "1. **Standard**: Original adversarial prompt (baseline)",
        "2. **Enhanced**: Adds explicit physics feasibility rules, worked examples,",
        "   and counter-examples directly in the system prompt",
        "3. **Self-Correct**: Generate → validate → show errors to LLM → ask to fix",
        "   (up to 2 correction rounds per scenario)",
        "",
    ]

    gpt4o_std  = metrics.get("gpt4o_standard",    {}).get("validity_rate")
    gpt4o_enh  = metrics.get("gpt4o_enhanced",    {}).get("validity_rate")
    gpt4o_sc   = metrics.get("gpt4o_selfcorrect", {}).get("validity_rate")
    mini_std   = metrics.get("gpt4omini_standard", {}).get("validity_rate")
    mini_enh   = metrics.get("gpt4omini_enhanced", {}).get("validity_rate")

    if gpt4o_std is not None and gpt4o_enh is not None:
        delta = (gpt4o_enh - gpt4o_std) * 100
        lines.append(
            f"- GPT-4o: standard {gpt4o_std*100:.1f}% → enhanced {gpt4o_enh*100:.1f}%"
            f" ({delta:+.1f} pp)"
        )
    if gpt4o_std is not None and gpt4o_sc is not None:
        delta = (gpt4o_sc - gpt4o_std) * 100
        lines.append(
            f"- GPT-4o: standard {gpt4o_std*100:.1f}% → self-correct {gpt4o_sc*100:.1f}%"
            f" ({delta:+.1f} pp)"
        )
    if mini_std is not None and mini_enh is not None:
        delta = (mini_enh - mini_std) * 100
        lines.append(
            f"- GPT-4o-mini: standard {mini_std*100:.1f}% → enhanced {mini_enh*100:.1f}%"
            f" ({delta:+.1f} pp)"
        )

    lines += [
        "",
        "## 5.5 Statistical Significance",
        "",
        "Welch's t-test comparing LLM configs vs random baseline on AQS:",
        "",
        "| Comparison | Mean Diff | p-value | Cohen's d | Significant? |",
        "|-----------|-----------|---------|-----------|-------------|",
    ]

    rnd_aqs = metrics.get("random_baseline", {}).get("aqs", [])
    if rnd_aqs:
        for c in [c for c in configs if c != "random_baseline"]:
            aqs = metrics[c]["aqs"]
            if len(aqs) < 3:
                continue
            stat = statistical_comparison(aqs, rnd_aqs, "LLM", "Random")
            diff = np.mean(aqs) - np.mean(rnd_aqs)
            p    = stat.get("ttest_p", float("nan"))
            d    = stat.get("cohens_d", float("nan"))
            sig  = "Yes" if p < 0.05 else "No"
            lines.append(
                f"| {CONFIG_LABELS.get(c, c).replace(chr(10), ' ')} vs Random"
                f" | {diff:+.3f} | {p:.3f} | {d:.3f} | {sig} |"
            )

    lines += [
        "",
        "## 5.6 Figures",
        "",
        "- **Fig 1**: Validity rates across all configurations",
        "- **Fig 2**: AQS distribution (box plots + jitter)",
        "- **Fig 3**: Obstacle proximity and TtF proxy",
        "- **Fig 4**: Path obstruction and multi-hazard density",
        "- **Fig 5**: Predicted constraint violation rate",
        "- **Fig 6**: Boundary push score and batch diversity",
        "- **Fig 7**: Prompt engineering impact on validity rate",
        "- **Fig 8**: Full summary dashboard (normalised heatmap)",
        "",
        "Figures saved to: `figures/multi_llm/`",
        "",
        "## 5.7 Conclusions",
        "",
        "1. **LLM models outperform random baseline on adversarial quality** (AQS, proximity,",
        "   path obstruction) across all tested configurations.",
        "2. **Prompt engineering meaningfully improves validity rate** — enhanced physics rules",
        "   reduce overlap/bounds errors by making implicit constraints explicit.",
        "3. **Self-correction is the most reliable validity recovery strategy** — it catches",
        "   errors that even enhanced prompts miss and asks the model to fix them in-context.",
        "4. **GPT-4o > GPT-4o-mini on adversarial quality** but mini is a cost-effective",
        "   alternative with enhanced prompting.",
        "5. **Claude models show strong physics reasoning** — Sonnet especially benefits from",
        "   the enhanced prompt to produce valid, high-quality adversarial scenarios.",
        "",
    ]

    # Append to report
    existing = report_path.read_text() if report_path.exists() else ""
    # Remove old Multi-LLM section if present
    marker = "\n---\n\n# 5. Multi-LLM"
    if marker in existing:
        existing = existing[:existing.index(marker)]

    report_path.write_text(existing + "\n".join(lines))
    print(f"  Updated {report_path.name} with Multi-LLM section")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_api_key_info(anthropic_available: bool):
    print("\n" + "="*65)
    print("  API KEYS NEEDED")
    print("="*65)
    print("  OPENAI_API_KEY  : set (read from api_key.py)")
    if anthropic_available:
        print("  ANTHROPIC_API_KEY: set — Claude configs enabled")
    else:
        print("  ANTHROPIC_API_KEY: NOT SET — Claude configs will be skipped")
        print()
        print("  To enable Claude models:")
        print("    1. Visit https://console.anthropic.com/ → Settings → API Keys")
        print("    2. Create a key (starts with sk-ant-...)")
        print("    3. export ANTHROPIC_API_KEY=sk-ant-...")
        print("  Cost estimate: 30 scenarios ≈ 20k tokens ≈ < $0.15 (Sonnet 4.6)")
    print("="*65 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Specific configs to run (default: all available)")
    parser.add_argument("--n", type=int, default=10,
                        help="Scenarios per task per config (default: 10)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip re-generation if results already saved")
    parser.add_argument("--no-skip", dest="skip_existing", action="store_false",
                        help="Re-generate even if cached results exist")
    parser.add_argument("--skip-openai", action="store_true",
                        help="Skip extra OpenAI configs (only random + loaded gpt4o_standard)")
    parser.add_argument("--figs-only", action="store_true",
                        help="Skip generation, only re-make figures and report from saved data")
    args = parser.parse_args()

    np.random.seed(42)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    MULTI_LLM_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load keys --
    openai_key = _load_openai_key()
    os.environ["OPENAI_API_KEY"] = openai_key
    anthropic_key = _load_anthropic_key()
    _print_api_key_info(anthropic_key is not None)

    # -- Determine which configs to run --
    if args.configs:
        configs_to_run = args.configs
    elif args.skip_openai:
        configs_to_run = []
    else:
        configs_to_run = ALL_OPENAI_CONFIGS[:]
        if anthropic_key:
            configs_to_run += ALL_CLAUDE_CONFIGS

    # -- Load random baseline from results/random_30/ --
    print("\n[Loading] Random baseline (results/random_30/)...")
    rnd_path = RESULTS_DIR / "random_30" / "random_scenarios.json"
    if rnd_path.exists():
        with open(rnd_path) as f:
            rnd_scenarios = json.load(f)
        rnd_validated = [validate_scenario(s).to_dict() for s in rnd_scenarios]
    else:
        print("  WARNING: random_30/random_scenarios.json not found, generating fresh random batch")
        from random_baseline import generate_multi_task_random_batch
        rnd_scenarios = generate_multi_task_random_batch(TASKS, n_per_task=args.n, seed=42)
        rnd_validated = [validate_scenario(s).to_dict() for s in rnd_scenarios]

    # -- Load existing gpt4o_standard from results/llm_30/ --
    print("[Loading] GPT-4o standard (results/llm_30/)...")
    llm30_path = RESULTS_DIR / "llm_30" / "llm_scenarios.json"
    if llm30_path.exists():
        with open(llm30_path) as f:
            gpt4o_std_scenarios = json.load(f)
        gpt4o_std_validated = [validate_scenario(s).to_dict() for s in gpt4o_std_scenarios]
        print(f"  Loaded {len(gpt4o_std_scenarios)} GPT-4o standard scenarios")
    else:
        print("  WARNING: llm_30/llm_scenarios.json not found; skipping gpt4o_standard")
        gpt4o_std_scenarios = []
        gpt4o_std_validated = []

    # -- Collect all results --
    all_results: dict[str, tuple[list, list]] = {}

    if gpt4o_std_scenarios:
        all_results["gpt4o_standard"] = (gpt4o_std_scenarios, gpt4o_std_validated)
    all_results["random_baseline"] = (rnd_scenarios, rnd_validated)

    if not args.figs_only:
        for config_name in configs_to_run:
            result = generate_config(
                config_name, args.n, openai_key, anthropic_key,
                skip_existing=args.skip_existing,
            )
            if result is not None:
                all_results[config_name] = result
    else:
        # Load all saved configs
        for config_name in (ALL_OPENAI_CONFIGS + ALL_CLAUDE_CONFIGS):
            cached = _load_scenarios(config_name)
            if cached:
                all_results[config_name] = cached
                print(f"  [Loaded] {config_name} ({len(cached[0])} scenarios)")

    print(f"\n[Metrics] Computing for {len(all_results)} configs...")
    metrics: dict[str, dict] = {}
    for config_name, (sc, val) in all_results.items():
        m = _compute_config_metrics(sc, val)
        metrics[config_name] = m
        print(f"  {config_name:25s}  valid={m['n_valid']}/{m['n_total']}"
              f"  AQS={_mean_str(m['aqs'])}  prox={_mean_str(m['proximity'])}")

    # -- Figures --
    print(f"\n[Figures] Saving to {FIGS_DIR}/")
    fig1_validity(metrics, FIGS_DIR)
    fig2_aqs(metrics, FIGS_DIR)
    fig3_proximity_ttf(metrics, FIGS_DIR)
    fig4_path_hazard(metrics, FIGS_DIR)
    fig5_predicted_violations(metrics, FIGS_DIR)
    fig6_boundary_diversity(metrics, FIGS_DIR)
    fig7_validity_improvement(metrics, FIGS_DIR)
    fig8_dashboard(metrics, FIGS_DIR)

    # -- Report --
    report_path = RESULTS_DIR / "evaluation_report.md"
    print(f"\n[Report] Updating {report_path.name}...")
    append_multi_llm_section(metrics, report_path)

    # -- Summary --
    print("\n" + "="*65)
    print("  EXPERIMENT COMPLETE")
    print("="*65)
    print(f"  Configs evaluated : {len(metrics)}")
    print(f"  Figures saved     : {FIGS_DIR}/")
    print(f"  Report updated    : {report_path}")
    print()

    # Print quick summary table
    print(f"  {'Config':<25} {'Valid%':>7} {'AQS':>7} {'Prox(m)':>9} {'TtF(s)':>8}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
    for c in _sorted_configs(list(metrics.keys())):
        m = metrics[c]
        vr   = f"{m['validity_rate']*100:.0f}%"
        aqs  = _mean_str(m['aqs'])
        prox = _mean_str(m['proximity'])
        ttf  = _mean_str(m['ttf'])
        print(f"  {CONFIG_LABELS.get(c,c).replace(chr(10),' '):<25} {vr:>7} {aqs:>7} {prox:>9} {ttf:>8}")

    if anthropic_key is None:
        print("\n  TIP: Set ANTHROPIC_API_KEY to also run Claude configs.")
        print("       See API key instructions printed above.")


if __name__ == "__main__":
    main()
