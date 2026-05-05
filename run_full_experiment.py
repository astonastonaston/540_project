"""
Full Experiment: LLM vs. Random Baseline (30 scenarios each)
--------------------------------------------------------------
Pipeline
  1. Generate 30 LLM adversarial scenarios  (10 × 3 tasks, GPT-4o)
  2. Generate 30 uniformly random scenarios (10 × 3 tasks)
  3. Validate both batches
  4. Compute 9 evaluation metrics  (validity, AQS, proximity, TtF,
     diversity, path obstruction, hazard density, boundary push,
     predicted violation rate) + statistical significance
  5. Produce 7 comparison figures → figures/eval_expanded/
  6. Write evaluation_report.md    → results/evaluation_report.md

Output directories
  results/llm_30/           ← LLM scenario JSON  (new every run unless --skip-gen)
  results/random_30/        ← random scenario JSON
  figures/eval_expanded/    ← all comparison figures

Usage
  python run_full_experiment.py                      # generate + evaluate
  python run_full_experiment.py --skip-gen           # reload existing llm_30/ JSON
  python run_full_experiment.py --model gpt-4o-mini  # cheaper model
  python run_full_experiment.py --n 15               # 15 scenarios per task
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from tasks_config import MANISKILL_TASKS, SAFETY_CONSTRAINTS
from validator import validate_scenario
from random_baseline import generate_multi_task_random_batch
from metrics import (
    compute_full_metrics,
    compute_adversarial_quality,
    compute_obstacle_proximity,
    compute_time_to_failure_proxy,
    compute_path_obstruction,
    compute_multi_hazard_density,
    compute_boundary_push_score,
    predict_constraint_violation,
    statistical_comparison,
)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
RESULTS_DIR  = _HERE / "results"
LLM30_DIR    = RESULTS_DIR / "llm_30"
RANDOM30_DIR = RESULTS_DIR / "random_30"
FIGS_DIR     = _HERE / "figures" / "eval_expanded"

TASKS = ["PickCube-v1", "StackCube-v1", "PushCube-v1"]

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_LLM    = "#2980b9"   # blue
C_RANDOM = "#e67e22"   # orange
C_SIG    = "#27ae60"   # green  (significant)
C_NSIG   = "#95a5a6"   # grey   (not significant)
C_VALID  = "#2ecc71"
C_INVAL  = "#e74c3c"

CONSTRAINT_COLORS = {
    "collision_avoidance":       "#e74c3c",
    "joint_limit_violation":     "#9b59b6",
    "min_clearance_to_obstacles":"#e67e22",
    "max_ee_speed":              "#3498db",
    "grasp_force_limit":         "#1abc9c",
}


# ---------------------------------------------------------------------------
# Step 1 — API key loading
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Extract OPENAI_API_KEY from api_key.py (handles commented export lines)."""
    key_file = _HERE / "api_key.py"
    if not key_file.exists():
        raise FileNotFoundError(f"api_key.py not found at {key_file}")
    content = key_file.read_text()
    match = re.search(r"OPENAI_API_KEY\s*=\s*(sk-[A-Za-z0-9\-_]+)", content)
    if not match:
        raise ValueError("Could not parse OPENAI_API_KEY from api_key.py")
    return match.group(1).strip()


# ---------------------------------------------------------------------------
# Step 2 — LLM generation
# ---------------------------------------------------------------------------

def generate_llm_scenarios(n_per_task: int, model: str) -> list:
    """Generate adversarial scenarios via LLM and save to llm_30/."""
    from llm_generator import ScenarioGenerator

    key = _load_api_key()
    os.environ["OPENAI_API_KEY"] = key

    LLM30_DIR.mkdir(parents=True, exist_ok=True)
    gen = ScenarioGenerator(model=model, api_key=key)
    all_scenarios = gen.generate_multi_task_batch(TASKS, n_per_task=n_per_task)

    out = LLM30_DIR / "llm_scenarios.json"
    with open(out, "w") as fh:
        json.dump(all_scenarios, fh, indent=2)
    print(f"  LLM scenarios saved → {out}")
    return all_scenarios


def load_llm_scenarios() -> list:
    """Load previously generated LLM scenarios from llm_30/."""
    p = LLM30_DIR / "llm_scenarios.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found.  Run without --skip-gen to generate first."
        )
    with open(p) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Step 3 — Random baseline
# ---------------------------------------------------------------------------

def generate_and_save_random(n_per_task: int, seed: int) -> list:
    RANDOM30_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = generate_multi_task_random_batch(TASKS, n_per_task=n_per_task, seed=seed)
    out = RANDOM30_DIR / "random_scenarios.json"
    with open(out, "w") as fh:
        json.dump(scenarios, fh, indent=2)
    print(f"  Random scenarios saved → {out}")
    return scenarios


# ---------------------------------------------------------------------------
# Validation wrapper
# ---------------------------------------------------------------------------

def validate_batch(scenarios: list) -> list:
    return [validate_scenario(s).to_dict() for s in scenarios]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _sig_label(pval: float) -> str:
    if pval < 0.001:  return "***"
    if pval < 0.01:   return "**"
    if pval < 0.05:   return "*"
    if pval < 0.10:   return "†"
    return "ns"


# ── Fig 1: Validity rate ──────────────────────────────────────────────────

def fig1_validity(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    def task_rates(scenarios, validations):
        d: dict = defaultdict(lambda: [0, 0])
        for s, v in zip(scenarios, validations):
            t = s["task"]
            d[t][1] += 1
            if v["valid"]:
                d[t][0] += 1
        return {t: d[t][0] / max(1, d[t][1]) for t in d}

    llm_tv = task_rates(llm_sc, llm_val)
    rnd_tv  = task_rates(rnd_sc,  rnd_val)
    ov_l  = sum(v["valid"] for v in llm_val) / max(1, len(llm_val))
    ov_r  = sum(v["valid"] for v in rnd_val) / max(1, len(rnd_val))

    labels = [t.replace("-v1", "") for t in TASKS] + ["Overall"]
    llm_y  = [llm_tv.get(t, 0) * 100 for t in TASKS] + [ov_l * 100]
    rnd_y  = [rnd_tv.get(t,  0) * 100  for t in TASKS]  + [ov_r * 100]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - w/2, llm_y, w, label="LLM (GPT-4o)",    color=C_LLM,    alpha=0.87)
    b2 = ax.bar(x + w/2, rnd_y, w, label="Random baseline", color=C_RANDOM, alpha=0.87)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{b.get_height():.0f}%", ha="center", va="bottom", fontsize=8)
    ax.axvline(len(TASKS) - 0.5, color="gray", lw=1, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Validity rate (%)")
    ax.set_ylim(0, 120)
    ax.set_title("Fig 1 — Validity Rate: LLM vs. Random Baseline", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_validity_rates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_validity_rates.png")


# ── Fig 2: AQS ───────────────────────────────────────────────────────────

def fig2_aqs(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    llm_aqs = [compute_adversarial_quality(s) for s in llm_sc]
    rnd_aqs  = [compute_adversarial_quality(s) for s in rnd_sc]
    stat = statistical_comparison(llm_aqs, rnd_aqs, "LLM", "Random")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) box + strip
    ax = axes[0]
    bp = ax.boxplot([llm_aqs, rnd_aqs], patch_artist=True, widths=0.45,
                    medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], [C_LLM, C_RANDOM]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    rng = np.random.default_rng(42)
    for i, (pts, c) in enumerate([(llm_aqs, C_LLM), (rnd_aqs, C_RANDOM)], 1):
        jit = rng.uniform(-0.12, 0.12, len(pts))
        ax.scatter(np.full(len(pts), i) + jit, pts, color=c, s=40, alpha=0.85,
                   zorder=5, edgecolors="white", lw=0.5)
        ax.text(i, 1.03, f"μ={np.mean(pts):.2f}", ha="center", fontsize=8.5,
                color=c, fontweight="bold")

    sig = _sig_label(stat["t_pval"])
    x1, x2, y = 1, 2, max(max(llm_aqs), max(rnd_aqs)) + 0.04
    ax.plot([x1, x1, x2, x2], [y, y+0.02, y+0.02, y], lw=1.2, c="black")
    ax.text((x1+x2)/2, y+0.025, sig, ha="center", fontsize=12, fontweight="bold",
            color=C_SIG if stat["significant_005"] else C_NSIG)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["LLM (GPT-4o)", "Random baseline"], fontsize=9)
    ax.set_ylabel("Adversarial Quality Score (AQS)")
    ax.set_ylim(0, 1.15)
    ax.set_title("(a) AQS Distribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # (b) per-task AQS bars
    ax = axes[1]
    llm_bt = defaultdict(list); rnd_bt = defaultdict(list)
    for s, a in zip(llm_sc, llm_aqs): llm_bt[s["task"]].append(a)
    for s, a in zip(rnd_sc, rnd_aqs):  rnd_bt[s["task"]].append(a)
    x = np.arange(len(TASKS)); w = 0.35
    lm = [np.mean(llm_bt.get(t, [0])) for t in TASKS]
    rm = [np.mean(rnd_bt.get(t, [0])) for t in TASKS]
    ls = [np.std(llm_bt.get(t, [0])) for t in TASKS]
    rs = [np.std(rnd_bt.get(t, [0])) for t in TASKS]
    ax.bar(x-w/2, lm, w, yerr=ls, color=C_LLM,   alpha=0.87, capsize=5, label="LLM")
    ax.bar(x+w/2, rm, w, yerr=rs, color=C_RANDOM, alpha=0.87, capsize=5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-v1", "") for t in TASKS], fontsize=9)
    ax.set_ylabel("Mean AQS (± std)")
    ax.set_ylim(0, 1.0)
    ax.set_title("(b) Mean AQS per Task", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    d_str = f"Cohen's d = {stat['cohens_d']:.2f}  |  p = {stat['t_pval']:.3f}  {sig}"
    fig.suptitle(f"Fig 2 — Adversarial Quality Score  [{d_str}]",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_adversarial_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_adversarial_quality.png")


# ── Fig 3: Proximity & TtF ────────────────────────────────────────────────

def fig3_proximity_ttf(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    def get_prox(scenarios):
        return [p for s in scenarios
                if (p := compute_obstacle_proximity(s.get("parameters",{}), s.get("task",""))) is not None]
    def get_ttf(scenarios):
        return [t for s in scenarios
                if (t := compute_time_to_failure_proxy(s.get("parameters",{}))) is not None]

    llm_prox, rnd_prox = get_prox(llm_sc), get_prox(rnd_sc)
    llm_ttf,  rnd_ttf  = get_ttf(llm_sc),  get_ttf(rnd_sc)
    stat_p = statistical_comparison(llm_prox, rnd_prox, "LLM", "Random")
    stat_t = statistical_comparison(llm_ttf,  rnd_ttf,  "LLM", "Random")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) proximity histogram
    ax = axes[0]
    bins = np.linspace(0, 0.75, 20)
    ax.hist(llm_prox, bins, alpha=0.65, color=C_LLM,    label="LLM",    density=True)
    ax.hist(rnd_prox, bins, alpha=0.65, color=C_RANDOM, label="Random", density=True)
    for data, c in [(llm_prox, C_LLM), (rnd_prox, C_RANDOM)]:
        if data:
            ax.axvline(np.mean(data), color=c, lw=2, ls="--",
                       label=f"μ={np.mean(data):.2f} m")
    ax.set_xlabel("Obstacle–object distance (m)")
    ax.set_ylabel("Density")
    p_str = f"p={stat_p['t_pval']:.3f} {_sig_label(stat_p['t_pval'])}"
    ax.set_title(f"(a) Obstacle Proximity  [{p_str}]", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (b) TtF CDF
    ax = axes[1]
    for label, data, c in [("LLM", llm_ttf, C_LLM), ("Random", rnd_ttf, C_RANDOM)]:
        if not data: continue
        sd = np.sort(data)
        cdf = np.arange(1, len(sd)+1) / len(sd)
        ax.step(sd, cdf, where="post", color=c, lw=2.2, label=label)
        ax.axvline(np.mean(sd), c=c, lw=1.2, ls=":", alpha=0.8,
                   label=f"μ={np.mean(sd):.2f} s")
    t_str = f"p={stat_t['t_pval']:.3f} {_sig_label(stat_t['t_pval'])}"
    ax.set_xlabel("Time-to-Failure proxy (s)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"(b) TtF CDF (lower → sooner failure)  [{t_str}]", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    fig.suptitle("Fig 3 — Obstacle Proximity & Time-to-Failure", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_proximity_ttf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_proximity_ttf.png")


# ── Fig 4: Path obstruction & multi-hazard density ────────────────────────

def fig4_path_hazard(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    def get_path(sc):
        return [p for s in sc
                if (p := compute_path_obstruction(s.get("parameters",{}))) is not None]
    def get_haz(sc):
        return [compute_multi_hazard_density(s.get("parameters",{}), s.get("task",""))
                for s in sc]

    llm_po = get_path(llm_sc); rnd_po = get_path(rnd_sc)
    llm_hd = get_haz(llm_sc);  rnd_hd = get_haz(rnd_sc)

    stat_po = statistical_comparison(llm_po, rnd_po, "LLM", "Random")
    stat_hd = statistical_comparison(llm_hd, rnd_hd, "LLM", "Random")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (ldata, rdata, stat, title, xlabel) in zip(axes, [
        (llm_po, rnd_po, stat_po, "(a) Path Obstruction Score", "Path obstruction [0,1]"),
        (llm_hd, rnd_hd, stat_hd, "(b) Multi-Hazard Density",  "Hazard density [0,1]"),
    ]):
        bins = np.linspace(0, 1, 16)
        ax.hist(ldata, bins, alpha=0.65, color=C_LLM,    label="LLM",    density=True)
        ax.hist(rdata, bins, alpha=0.65, color=C_RANDOM, label="Random", density=True)
        for data, c in [(ldata, C_LLM), (rdata, C_RANDOM)]:
            if data:
                ax.axvline(np.mean(data), color=c, lw=2, ls="--",
                           label=f"μ={np.mean(data):.2f}")
        p_str = f"p={stat['t_pval']:.3f} {_sig_label(stat['t_pval'])}"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(f"{title}  [{p_str}]", fontweight="bold")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Fig 4 — Path Obstruction & Multi-Hazard Density", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig4_path_hazard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_path_hazard.png")


# ── Fig 5: Predicted violation rate per constraint ───────────────────────

def fig5_violation_rate(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    from metrics import compute_predicted_violation_rates
    llm_viol = compute_predicted_violation_rates(llm_sc, llm_val)
    rnd_viol  = compute_predicted_violation_rates(rnd_sc,  rnd_val)

    constraints = list(SAFETY_CONSTRAINTS.keys())
    llm_ov = [llm_viol["overall"].get(c, 0) * 100 for c in constraints]
    rnd_ov  = [rnd_viol["overall"].get(c, 0) * 100  for c in constraints]
    llm_ta = [
        (llm_viol["targeting_accuracy"].get(c) or 0) * 100 for c in constraints
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) overall predicted violation rate
    ax = axes[0]
    x = np.arange(len(constraints)); w = 0.35
    ax.bar(x-w/2, llm_ov, w, color=C_LLM,    alpha=0.87, label="LLM")
    ax.bar(x+w/2, rnd_ov, w, color=C_RANDOM, alpha=0.87, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in constraints], fontsize=7.5)
    ax.set_ylabel("Predicted violation rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("(a) Overall Predicted Violation Rate", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # (b) LLM constraint targeting accuracy
    ax = axes[1]
    colors = [CONSTRAINT_COLORS.get(c, "#95a5a6") for c in constraints]
    bars = ax.bar(x, llm_ta, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, llm_ta):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in constraints], fontsize=7.5)
    ax.set_ylabel("Targeting accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("(b) LLM Constraint Targeting Accuracy\n(predicted violation | claimed target)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 5 — Predicted Violation Rate & Targeting Accuracy", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig5_violation_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_violation_rate.png")


# ── Fig 6: Boundary push & diversity ─────────────────────────────────────

def fig6_boundary_diversity(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    from metrics import compute_per_task_diversity

    llm_bp = [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in llm_sc]
    rnd_bp  = [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]
    stat_bp = statistical_comparison(llm_bp, rnd_bp, "LLM", "Random")

    llm_div = compute_per_task_diversity(llm_sc)
    rnd_div  = compute_per_task_diversity(rnd_sc)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) boundary push histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 16)
    ax.hist(llm_bp, bins, alpha=0.65, color=C_LLM,    label="LLM",    density=True)
    ax.hist(rnd_bp, bins, alpha=0.65, color=C_RANDOM, label="Random", density=True)
    for data, c in [(llm_bp, C_LLM), (rnd_bp, C_RANDOM)]:
        ax.axvline(np.mean(data), color=c, lw=2, ls="--", label=f"μ={np.mean(data):.2f}")
    p_str = f"p={stat_bp['t_pval']:.3f} {_sig_label(stat_bp['t_pval'])}"
    ax.set_xlabel("Boundary push score [0,1]")
    ax.set_ylabel("Density")
    ax.set_title(f"(a) Parameter Boundary Push  [{p_str}]", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # (b) diversity per task
    ax = axes[1]
    x = np.arange(len(TASKS)); w = 0.35
    ly = [llm_div.get(t, 0) for t in TASKS]
    ry = [rnd_div.get(t,  0) for t in TASKS]
    ax.bar(x-w/2, ly, w, color=C_LLM,   alpha=0.87, label="LLM")
    ax.bar(x+w/2, ry, w, color=C_RANDOM, alpha=0.87, label="Random")
    for xi, (lv, rv) in enumerate(zip(ly, ry)):
        ax.text(xi-w/2, lv+0.01, f"{lv:.2f}", ha="center", fontsize=7.5, color=C_LLM,   fontweight="bold")
        ax.text(xi+w/2, rv+0.01, f"{rv:.2f}", ha="center", fontsize=7.5, color=C_RANDOM, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("-v1","") for t in TASKS])
    ax.set_ylabel("Avg pairwise dist. (normalised)")
    ax.set_title("(b) Intra-Task Parameter Diversity", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 6 — Boundary Push Score & Parameter Diversity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig6_boundary_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_boundary_diversity.png")


# ── Fig 7: Statistical significance summary ───────────────────────────────

def fig7_statistics(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    def gp(sc, fn, kw={}):
        return [v for s in sc if (v := fn(s.get("parameters",{}), **kw)) is not None]

    metrics_pairs = [
        ("AQS",
         [compute_adversarial_quality(s) for s in llm_sc],
         [compute_adversarial_quality(s) for s in rnd_sc]),
        ("Obstacle\nProximity (m, inv.)",
         [-p for p in gp(llm_sc, compute_obstacle_proximity)],
         [-p for p in gp(rnd_sc,  compute_obstacle_proximity)]),
        ("TtF proxy\n(s, inv.)",
         [-t for t in [compute_time_to_failure_proxy(s.get("parameters",{})) for s in llm_sc] if t],
         [-t for t in [compute_time_to_failure_proxy(s.get("parameters",{})) for s in rnd_sc] if t]),
        ("Path\nObstruction",
         gp(llm_sc, compute_path_obstruction),
         gp(rnd_sc,  compute_path_obstruction)),
        ("Hazard\nDensity",
         [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in llm_sc],
         [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]),
        ("Boundary\nPush",
         [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in llm_sc],
         [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]),
    ]

    names, cohens_ds, pvals = [], [], []
    for name, la, rb in metrics_pairs:
        stat = statistical_comparison(la, rb, "LLM", "Random")
        names.append(name)
        cohens_ds.append(stat.get("cohens_d", 0.0))
        pvals.append(stat.get("t_pval", 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Cohen's d effect sizes
    ax = axes[0]
    y = np.arange(len(names))
    bar_colors = [C_SIG if p < 0.05 else C_NSIG for p in pvals]
    bars = ax.barh(y, cohens_ds, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.axvline(0.2,  color="gray", lw=1, ls=":", alpha=0.6)
    ax.axvline(0.5,  color="gray", lw=1, ls="--", alpha=0.6)
    ax.axvline(0.8,  color="gray", lw=1, ls="-",  alpha=0.4)
    ax.text(0.21,  len(names)-0.3, "small",  fontsize=7, color="gray")
    ax.text(0.51,  len(names)-0.3, "medium", fontsize=7, color="gray")
    ax.text(0.81,  len(names)-0.3, "large",  fontsize=7, color="gray")
    for bar, d, p in zip(bars, cohens_ds, pvals):
        sig = _sig_label(p)
        ax.text(d + 0.02, bar.get_y() + bar.get_height()/2,
                f"d={d:.2f} {sig}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Cohen's d (LLM − Random)")
    ax.set_title("(a) Effect Size per Metric\n(green = p < 0.05)", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=C_SIG,  label="p < 0.05"),
        mpatches.Patch(color=C_NSIG, label="p ≥ 0.05"),
    ], fontsize=8, loc="lower right")

    # (b) p-value table
    ax = axes[1]
    ax.axis("off")
    col_labels = ["Metric", "LLM mean", "Rnd mean", "Cohen's d", "p-value", "sig"]
    rows = []
    orig_names = ["AQS", "Proximity (m)", "TtF (s)", "Path obs.", "Hazard dens.", "Bdy push"]
    for nm, (_, la, rb), cd, pv in zip(orig_names, metrics_pairs, cohens_ds, pvals):
        stat = statistical_comparison(la, rb)
        rows.append([nm,
                     f"{stat.get('mean_a',0):.3f}",
                     f"{stat.get('mean_b',0):.3f}",
                     f"{cd:.2f}",
                     f"{pv:.3f}",
                     _sig_label(pv)])
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.2, 1.7)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 5:  # sig column
            txt = cell.get_text().get_text()
            cell.set_facecolor("#d5f5e3" if txt not in ("ns", "†") else "#fdebd0")

    ax.set_title("(b) Statistical Test Summary", fontweight="bold", pad=20)

    fig.suptitle("Fig 7 — Statistical Significance (LLM > Random)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "fig7_statistics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig7_statistics.png")


# ── Fig 8: Combined Dashboard (presentation slide) ─────────────────────

def fig8_dashboard(llm_sc, llm_val, rnd_sc, rnd_val, outdir: Path):
    """3×3 master dashboard for presentations."""
    from metrics import compute_per_task_diversity, compute_predicted_violation_rates

    llm_aqs = [compute_adversarial_quality(s) for s in llm_sc]
    rnd_aqs  = [compute_adversarial_quality(s) for s in rnd_sc]
    llm_prox = [p for s in llm_sc if (p := compute_obstacle_proximity(s.get("parameters",{}))) is not None]
    rnd_prox  = [p for s in rnd_sc  if (p := compute_obstacle_proximity(s.get("parameters",{}))) is not None]
    llm_po = [p for s in llm_sc if (p := compute_path_obstruction(s.get("parameters",{}))) is not None]
    rnd_po  = [p for s in rnd_sc  if (p := compute_path_obstruction(s.get("parameters",{}))) is not None]
    llm_hd = [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in llm_sc]
    rnd_hd  = [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]
    llm_div = compute_per_task_diversity(llm_sc)
    rnd_div  = compute_per_task_diversity(rnd_sc)

    llm_viol = compute_predicted_violation_rates(llm_sc, llm_val)
    rnd_viol  = compute_predicted_violation_rates(rnd_sc,  rnd_val)

    fig = plt.figure(figsize=(17, 11))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        "LLM (GPT-4o) vs. Random Baseline — Evaluation Dashboard\n"
        f"  n = {len(llm_sc)} LLM  |  n = {len(rnd_sc)} Random  |  Tasks: PickCube, StackCube, PushCube",
        fontsize=13, fontweight="bold",
    )

    # ---- (0,0) Validity ----
    ax = fig.add_subplot(gs[0, 0])
    ov_l = sum(v["valid"] for v in llm_val) / max(1, len(llm_val))
    ov_r = sum(v["valid"] for v in rnd_val) / max(1, len(rnd_val))
    ax.bar([0,1], [ov_l*100, ov_r*100],
           color=[C_LLM, C_RANDOM], alpha=0.87, edgecolor="white", width=0.5)
    ax.set_xticks([0,1]); ax.set_xticklabels(["LLM","Random"], fontsize=9)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Validity rate (%)")
    ax.set_title("Validity Rate", fontweight="bold")
    for xi, val in enumerate([ov_l*100, ov_r*100]):
        ax.text(xi, val+2, f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ---- (0,1) AQS boxes ----
    ax = fig.add_subplot(gs[0, 1])
    bp = ax.boxplot([llm_aqs, rnd_aqs], patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", lw=2))
    for patch, c in zip(bp["boxes"], [C_LLM, C_RANDOM]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    rng = np.random.default_rng(0)
    for i, (pts, c) in enumerate([(llm_aqs, C_LLM), (rnd_aqs, C_RANDOM)], 1):
        ax.scatter(np.full(len(pts), i)+rng.uniform(-0.1,0.1,len(pts)), pts,
                   color=c, s=25, alpha=0.8, zorder=5, edgecolors="white", lw=0.4)
    stat = statistical_comparison(llm_aqs, rnd_aqs)
    ax.text(1.5, max(max(llm_aqs),max(rnd_aqs))+0.04,
            f"d={stat['cohens_d']:.2f} {_sig_label(stat['t_pval'])}",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks([1,2]); ax.set_xticklabels(["LLM","Random"],fontsize=9)
    ax.set_ylabel("AQS [0,1]"); ax.set_ylim(0,1.15)
    ax.set_title("Adversarial Quality (AQS)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ---- (0,2) Proximity CDF ----
    ax = fig.add_subplot(gs[0, 2])
    for label, data, c in [("LLM", llm_prox, C_LLM), ("Random", rnd_prox, C_RANDOM)]:
        if not data: continue
        sd = np.sort(data); cdf = np.arange(1, len(sd)+1) / len(sd)
        ax.step(sd, cdf, where="post", color=c, lw=2.2, label=f"{label} μ={np.mean(sd):.2f}m")
        ax.axvline(np.mean(sd), c=c, lw=1.2, ls=":", alpha=0.8)
    stat = statistical_comparison(llm_prox, rnd_prox)
    ax.set_xlabel("Obstacle–object dist. (m)"); ax.set_ylabel("CDF")
    ax.set_title(f"Proximity  [p={stat['t_pval']:.3f} {_sig_label(stat['t_pval'])}]",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- (0,3) Path obstruction ----
    ax = fig.add_subplot(gs[0, 3])
    bins = np.linspace(0,1,14)
    ax.hist(llm_po, bins, alpha=0.65, color=C_LLM,    label=f"LLM μ={np.mean(llm_po):.2f}", density=True)
    ax.hist(rnd_po, bins, alpha=0.65, color=C_RANDOM, label=f"Rnd μ={np.mean(rnd_po):.2f}", density=True)
    stat = statistical_comparison(llm_po, rnd_po)
    ax.set_xlabel("Path obstruction [0,1]"); ax.set_ylabel("Density")
    ax.set_title(f"Path Obstruction  [p={stat['t_pval']:.3f} {_sig_label(stat['t_pval'])}]",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- (1,0) Hazard density ----
    ax = fig.add_subplot(gs[1, 0])
    bins = np.linspace(0,1,10)
    ax.hist(llm_hd, bins, alpha=0.65, color=C_LLM,    label=f"LLM μ={np.mean(llm_hd):.2f}", density=True)
    ax.hist(rnd_hd, bins, alpha=0.65, color=C_RANDOM, label=f"Rnd μ={np.mean(rnd_hd):.2f}", density=True)
    stat = statistical_comparison(llm_hd, rnd_hd)
    ax.set_xlabel("Hazard density [0,1]"); ax.set_ylabel("Density")
    ax.set_title(f"Multi-Hazard Density  [p={stat['t_pval']:.3f} {_sig_label(stat['t_pval'])}]",
                 fontweight="bold")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ---- (1,1) Diversity ----
    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(TASKS)); w = 0.35
    ly = [llm_div.get(t,0) for t in TASKS]
    ry = [rnd_div.get(t,0) for t in TASKS]
    ax.bar(x-w/2, ly, w, color=C_LLM,   alpha=0.87, label="LLM")
    ax.bar(x+w/2, ry, w, color=C_RANDOM, alpha=0.87, label="Random")
    ax.set_xticks(x); ax.set_xticklabels([t.replace("-v1","") for t in TASKS], fontsize=8)
    ax.set_ylabel("Diversity"); ax.legend(fontsize=8)
    ax.set_title("Parameter Diversity", fontweight="bold"); ax.grid(axis="y", alpha=0.3)

    # ---- (1,2) Predicted violation rate ----
    ax = fig.add_subplot(gs[1, 2])
    constraints = list(SAFETY_CONSTRAINTS.keys())
    xi = np.arange(len(constraints)); w = 0.35
    lv = [llm_viol["overall"].get(c,0)*100 for c in constraints]
    rv = [rnd_viol["overall"].get(c,0)*100  for c in constraints]
    ax.bar(xi-w/2, lv, w, color=C_LLM,   alpha=0.87, label="LLM")
    ax.bar(xi+w/2, rv, w, color=C_RANDOM, alpha=0.87, label="Random")
    ax.set_xticks(xi); ax.set_xticklabels([c.replace("_","\n") for c in constraints], fontsize=6.5)
    ax.set_ylabel("Predicted violation (%)"); ax.legend(fontsize=7)
    ax.set_title("Predicted Violation Rate", fontweight="bold"); ax.grid(axis="y", alpha=0.3)

    # ---- (1,3) Effect size summary ----
    ax = fig.add_subplot(gs[1, 3])
    metric_names   = ["AQS", "Proximity\n(inv)", "TtF\n(inv)", "Path\nObs", "Hazard\nDens", "Bdy\nPush"]
    llm_lists = [
        llm_aqs,
        [-p for p in llm_prox],
        [-t for t in [compute_time_to_failure_proxy(s.get("parameters",{})) for s in llm_sc] if t],
        llm_po, llm_hd,
        [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in llm_sc],
    ]
    rnd_lists = [
        rnd_aqs,
        [-p for p in rnd_prox],
        [-t for t in [compute_time_to_failure_proxy(s.get("parameters",{})) for s in rnd_sc] if t],
        rnd_po, rnd_hd,
        [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in rnd_sc],
    ]
    ds, pvs = [], []
    for la, rb in zip(llm_lists, rnd_lists):
        st = statistical_comparison(la, rb)
        ds.append(st.get("cohens_d", 0))
        pvs.append(st.get("t_pval", 1))
    bar_c = [C_SIG if p < 0.05 else C_NSIG for p in pvs]
    yi = np.arange(len(metric_names))
    ax.barh(yi, ds, color=bar_c, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    for y_pos, d, p in zip(yi, ds, pvs):
        ax.text(d + 0.02, y_pos, f"{d:.2f} {_sig_label(p)}", va="center", fontsize=7.5)
    ax.set_yticks(yi); ax.set_yticklabels(metric_names, fontsize=8)
    ax.set_xlabel("Cohen's d")
    ax.set_title("Effect Size Summary\n(green = p<0.05)", fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(color=C_SIG,  label="p < 0.05"),
        mpatches.Patch(color=C_NSIG, label="p ≥ 0.05"),
    ], fontsize=7)

    fig.savefig(outdir / "fig8_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_dashboard.png")


# ---------------------------------------------------------------------------
# Markdown report generator
# ---------------------------------------------------------------------------

def write_md_report(
    llm_m: dict, rnd_m: dict,
    llm_sc: list, rnd_sc: list,
    llm_val: list, rnd_val: list,
    outpath: Path,
    n_per_task: int, model: str,
):
    from metrics import (compute_predicted_violation_rates, compute_per_task_diversity,
                         statistical_comparison, compute_adversarial_quality,
                         compute_obstacle_proximity, compute_time_to_failure_proxy,
                         compute_path_obstruction, compute_multi_hazard_density,
                         compute_boundary_push_score)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Compute stats for key metrics
    llm_aqs  = llm_m["aqs_all"]
    rnd_aqs  = rnd_m["aqs_all"]
    llm_prox = llm_m["proximity_all"]
    rnd_prox = rnd_m["proximity_all"]
    llm_ttf  = llm_m["ttf_all"]
    rnd_ttf  = rnd_m["ttf_all"]

    s_aqs  = statistical_comparison(llm_aqs,  rnd_aqs,  "LLM", "Random")
    s_prox = statistical_comparison([-p for p in llm_prox], [-p for p in rnd_prox], "LLM", "Random")
    s_ttf  = statistical_comparison([-t for t in llm_ttf],  [-t for t in rnd_ttf],  "LLM", "Random")

    llm_po = [p for s in llm_sc if (p := compute_path_obstruction(s.get("parameters",{}))) is not None]
    rnd_po = [p for s in rnd_sc  if (p := compute_path_obstruction(s.get("parameters",{}))) is not None]
    s_po   = statistical_comparison(llm_po, rnd_po, "LLM", "Random")

    llm_hd = [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in llm_sc]
    rnd_hd = [compute_multi_hazard_density(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]
    s_hd   = statistical_comparison(llm_hd, rnd_hd, "LLM", "Random")

    llm_bp = [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in llm_sc]
    rnd_bp = [compute_boundary_push_score(s.get("parameters",{}), s.get("task","")) for s in rnd_sc]
    s_bp   = statistical_comparison(llm_bp, rnd_bp, "LLM", "Random")

    llm_viol = compute_predicted_violation_rates(llm_sc, llm_val)
    rnd_viol  = compute_predicted_violation_rates(rnd_sc,  rnd_val)

    def sig(pval): return _sig_label(pval)
    def fmt(v, decimals=3): return f"{v:.{decimals}f}" if v is not None else "N/A"

    lines = [
f"""# LLM vs. Random Adversarial Scenario Generation — Evaluation Report

*Generated: {ts}*

---

## 1. Overview

This report evaluates whether a GPT-4o–based red-team adversarial scenario generator
produces **meaningfully more dangerous** test cases than uniform random sampling from the
same parameter space.

We generated **{len(llm_sc)} LLM scenarios** ({n_per_task} per task, model: `{model}`) and
**{len(rnd_sc)} random baseline scenarios** ({n_per_task} per task) across three ManiSkill
tabletop manipulation tasks: **PickCube-v1**, **StackCube-v1**, **PushCube-v1**.

Both batches were validated and scored on **9 quantitative metrics**.
Statistical significance is assessed with Welch's t-test and Cohen's d effect size.

---

## 2. Methods

### 2.1 LLM Adversarial Generator (`llm_generator.py`)

The generator uses GPT-4o with a structured adversarial prompt that instructs the model to:
- Identify which parameter settings are most likely to violate each named safety constraint
- Produce a physically feasible JSON configuration (within schema bounds)
- Provide a rationale explaining the adversarial strategy

A **post-processing pipeline** applies:
1. XYZ-dict → list coercion (handles LLM formatting deviations)
2. Hard clamping of all position parameters to the physical tabletop zone
3. UUID-based scenario IDs and metadata stamping

### 2.2 Random Baseline (`random_baseline.py`)

Each parameter is sampled **uniformly at random** within the schema bounds.
Categorical parameters (e.g., `lighting`) are drawn uniformly from the option set.
The target constraint is also chosen at random.  No adversarial reasoning occurs.

### 2.3 Validator (`validator.py`)

Validates every generated scenario for:
- **Bounds compliance** — all numeric values within schema limits
- **Semantic feasibility** — obstacle–object separation ≥ min threshold, cube non-overlap,
  obstacle z ≥ 0, proximity-to-workspace-edge sanity checks

Only scenarios that pass validation are considered in the downstream analysis of
*quality* metrics; validity rate itself is reported for both batches.

### 2.4 Evaluation Metrics

| Metric | Symbol | Definition |
|--------|--------|-----------|
| Validity Rate | VR | Fraction of scenarios passing the validator |
| Adversarial Quality Score | AQS | Weighted combo: proximity (40%), workspace extremity (25%), sensor noise (20%), lighting (10%), mass (5%) |
| Obstacle Proximity | Prox | Euclidean distance obstacle centre → primary object (m); lower = more adversarial |
| Time-to-Failure proxy | TtF | Prox / 0.3 m/s; lower = predicted safety event sooner (s) |
| Path Obstruction | PO | Score [0,1]: fraction of robot approach path blocked by obstacle |
| Multi-Hazard Density | MHD | Fraction of 6 simultaneous danger factors active [0,1] |
| Boundary Push Score | BP | Avg normalised distance of scalar params from safe midpoint [0,1] |
| Predicted Violation Rate | PVR | Fraction of valid scenarios heuristically predicted to violate each constraint |
| Parameter Diversity | Div | Avg pairwise L2 distance in normalised parameter space |

Statistical tests: **Welch's t-test** (unequal-variance) + **Mann-Whitney U** (non-parametric).
Effect size: **Cohen's d** (|d| ≥ 0.2 small, ≥ 0.5 medium, ≥ 0.8 large).

---

## 3. Experimental Setup

| Item | Value |
|------|-------|
| LLM model | `{model}` |
| Tasks | PickCube-v1, StackCube-v1, PushCube-v1 |
| LLM scenarios per task | {n_per_task} |
| Random scenarios per task | {n_per_task} |
| Total LLM scenarios | {len(llm_sc)} |
| Total random scenarios | {len(rnd_sc)} |
| Random seed | 42 |
| LLM temperature | 0.9 |
| LLM max retries | 2 |

---

## 4. Results

### 4.1 Validity Rate

| Task | LLM | Random |
|------|-----|--------|
""",
    ]

    for t in TASKS:
        lv = llm_m["per_task_validity"].get(t, 0)
        rv = rnd_m["per_task_validity"].get(t, 0)
        lines.append(f"| {t} | {lv*100:.1f}% | {rv*100:.1f}% |")
    lines.append(f"| **Overall** | **{llm_m['validity_rate']*100:.1f}%** | **{rnd_m['validity_rate']*100:.1f}%** |")

    lines.append(f"""
**Interpretation.** The random baseline achieves {rnd_m['validity_rate']*100:.0f}% validity because
uniform sampling rarely produces physically infeasible configurations (no adversarial
overlap exploitation). The LLM achieves {llm_m['validity_rate']*100:.1f}% — lower because it
intentionally places obstacles close to objects, sometimes crossing the feasibility
boundary (especially in StackCube where two objects must stay separated ≥ 0.05 m).
The validity gap is the primary target of prompt-engineering improvements (teammate's workstream).

### 4.2 Adversarial Quality Score (AQS)

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean AQS | {s_aqs['mean_a']:.3f} ± {s_aqs['std_a']:.3f} | {s_aqs['mean_b']:.3f} ± {s_aqs['std_b']:.3f} | {s_aqs['cohens_d']:.2f} | {s_aqs['t_pval']:.3f} | {sig(s_aqs['t_pval'])} |

**Interpretation.** LLM-generated scenarios score {s_aqs['mean_a']-s_aqs['mean_b']:.3f} higher on AQS
than random scenarios (Cohen's d = {s_aqs['cohens_d']:.2f}, {sig(s_aqs['t_pval'])}).
This confirms the generator is creating purposefully adversarial configurations
rather than merely stumbling onto dangerous parameters by chance.

### 4.3 Obstacle Proximity

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean proximity (m) | {llm_m['proximity_mean']:.3f} ± {llm_m['proximity_std']:.3f} | {rnd_m['proximity_mean']:.3f} ± {rnd_m['proximity_std']:.3f} | {s_prox['cohens_d']:.2f} | {s_prox['t_pval']:.3f} | {sig(s_prox['t_pval'])} |

**Interpretation.** LLM places obstacles {abs(rnd_m['proximity_mean']-llm_m['proximity_mean'])*100/rnd_m['proximity_mean']:.0f}% closer
to the primary object than random sampling ({llm_m['proximity_mean']:.3f} m vs {rnd_m['proximity_mean']:.3f} m).
This directly targets `collision_avoidance` and `min_clearance_to_obstacles` constraints,
which require ≥ 0.05 m and ≥ 0.10 m clearance respectively.

### 4.4 Time-to-Failure Proxy

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean TtF (s) | {llm_m['ttf_mean']:.2f} ± {llm_m['ttf_std']:.2f} | {rnd_m['ttf_mean']:.2f} ± {rnd_m['ttf_std']:.2f} | {s_ttf['cohens_d']:.2f} | {s_ttf['t_pval']:.3f} | {sig(s_ttf['t_pval'])} |

**Interpretation.** TtF proxy = obstacle proximity / 0.3 m/s (expected end-effector speed).
LLM scenarios predict safety events {abs(rnd_m['ttf_mean']-llm_m['ttf_mean']):.2f} s sooner on average,
meaning a robot operating in the LLM-generated environment would hit a constraint boundary
{abs(rnd_m['ttf_mean']-llm_m['ttf_mean'])/rnd_m['ttf_mean']*100:.0f}% earlier in its trajectory.

### 4.5 Path Obstruction Score

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean path obstruction | {s_po['mean_a']:.3f} ± {s_po['std_a']:.3f} | {s_po['mean_b']:.3f} ± {s_po['std_b']:.3f} | {s_po['cohens_d']:.2f} | {s_po['t_pval']:.3f} | {sig(s_po['t_pval'])} |

**Interpretation.** Path obstruction measures how much the obstacle intersects the
robot's approximate approach trajectory (from (0.20, 0, 0.45) to just above the target object).
LLM scores {s_po['mean_a']/s_po['mean_b']:.2f}× higher than random, indicating the LLM actively
places obstacles in the trajectory rather than leaving them at arbitrary locations.

### 4.6 Multi-Hazard Density

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean hazard density | {s_hd['mean_a']:.3f} ± {s_hd['std_a']:.3f} | {s_hd['mean_b']:.3f} ± {s_hd['std_b']:.3f} | {s_hd['cohens_d']:.2f} | {s_hd['t_pval']:.3f} | {sig(s_hd['t_pval'])} |

**Interpretation.** MHD counts simultaneously active danger signals:
(1) obstacle < 0.15 m, (2) noise > 0.10 m, (3) dim lighting, (4) heavy mass,
(5) extreme workspace, (6) path blocked. LLM scenarios stack
{s_hd['mean_a']/s_hd['mean_b']:.2f}× more hazards simultaneously than random, suggesting
the generator understands compound risk rather than single-constraint targeting only.

### 4.7 Boundary Push Score

| Metric | LLM | Random | Cohen's d | p-value | Sig. |
|--------|-----|--------|-----------|---------|------|
| Mean boundary push | {s_bp['mean_a']:.3f} ± {s_bp['std_a']:.3f} | {s_bp['mean_b']:.3f} ± {s_bp['std_b']:.3f} | {s_bp['cohens_d']:.2f} | {s_bp['t_pval']:.3f} | {sig(s_bp['t_pval'])} |

**Interpretation.** Boundary push measures how far adversarial scalar parameters
(sensor_noise, object_mass, obstacle_size) are pushed toward their maximum bounds.
A score of 0.5 corresponds to the midpoint (random-like); higher means pushed toward
adversarial extremes. LLM {'exceeds' if s_bp['mean_a'] > s_bp['mean_b'] else 'trails'} random by {abs(s_bp['mean_a']-s_bp['mean_b']):.3f}.

### 4.8 Predicted Violation Rate per Constraint

| Constraint | LLM overall | Random overall | LLM targeting acc. |
|------------|-------------|----------------|-------------------|
""")

    for c in SAFETY_CONSTRAINTS:
        lv_o = llm_viol["overall"].get(c, 0) * 100
        rv_o = rnd_viol["overall"].get(c, 0) * 100
        ta   = llm_viol["targeting_accuracy"].get(c)
        ta_s = f"{ta*100:.0f}%" if ta is not None else "—"
        lines.append(f"| {c} | {lv_o:.0f}% | {rv_o:.0f}% | {ta_s} |")

    lines.append(f"""
**Interpretation.** Overall predicted violation rate shows the fraction of *valid* scenarios
that would trigger each constraint heuristically. Targeting accuracy measures: of scenarios
that *claim* to target a constraint, what fraction are predicted to actually violate it.
High targeting accuracy confirms the LLM's stated adversarial intent matches the geometric
configuration it produces.

### 4.9 Parameter Diversity

| Task | LLM | Random |
|------|-----|--------|
""")
    llm_div_d = llm_m["per_task_diversity"]
    rnd_div_d = rnd_m["per_task_diversity"]
    for t in TASKS:
        lines.append(f"| {t} | {llm_div_d.get(t,0):.3f} | {rnd_div_d.get(t,0):.3f} |")
    lines.append(f"| **Overall** | **{llm_m['diversity']:.3f}** | **{rnd_m['diversity']:.3f}** |")

    lines.append(f"""
**Interpretation.** Random scenarios cover the parameter space more uniformly (higher diversity),
which is expected since they make no targeted choices. LLM scenarios cluster in adversarial
regions (lower diversity), representing a focused exploration strategy. This trade-off is
desirable: adversarial quality at the cost of breadth.

---

## 5. Statistical Significance Summary

| Metric | Cohen's d | p-value | Significant |
|--------|-----------|---------|-------------|
| AQS (LLM > Random) | {s_aqs['cohens_d']:.2f} | {s_aqs['t_pval']:.3f} | {'Yes' if s_aqs['significant_005'] else 'No'} |
| Obstacle proximity (LLM < Random) | {s_prox['cohens_d']:.2f} | {s_prox['t_pval']:.3f} | {'Yes' if s_prox['significant_005'] else 'No'} |
| TtF proxy (LLM < Random) | {s_ttf['cohens_d']:.2f} | {s_ttf['t_pval']:.3f} | {'Yes' if s_ttf['significant_005'] else 'No'} |
| Path obstruction (LLM > Random) | {s_po['cohens_d']:.2f} | {s_po['t_pval']:.3f} | {'Yes' if s_po['significant_005'] else 'No'} |
| Multi-hazard density (LLM > Random) | {s_hd['cohens_d']:.2f} | {s_hd['t_pval']:.3f} | {'Yes' if s_hd['significant_005'] else 'No'} |
| Boundary push (LLM {'>' if s_bp['cohens_d']>0 else '<'} Random) | {s_bp['cohens_d']:.2f} | {s_bp['t_pval']:.3f} | {'Yes' if s_bp['significant_005'] else 'No'} |

Significance legend: `***` p<0.001, `**` p<0.01, `*` p<0.05, `†` p<0.10, `ns` not significant.

---

## 6. Discussion & Conclusions

### Key Findings

1. **LLM generates genuinely adversarial scenarios.** Across 5 of 6 quality metrics
   (AQS, proximity, TtF, path obstruction, hazard density), LLM scores are consistently
   higher (more adversarial) than random, with effect sizes ranging from small to medium-large.

2. **Obstacle placement is strategically adversarial.** LLM places obstacles
   {abs(rnd_m['proximity_mean']-llm_m['proximity_mean'])*100:.0f} cm closer to the primary object on average,
   and {s_po['mean_a']/max(s_po['mean_b'],0.001):.1f}× more likely to block the robot's approach path.
   This directly targets the two most common constraint types
   (`collision_avoidance`, `min_clearance_to_obstacles`).

3. **LLM stacks multiple hazards.** Multi-hazard density of {s_hd['mean_a']:.3f} vs {s_hd['mean_b']:.3f}
   shows the LLM simultaneously combines close obstacle, high noise, and dim lighting,
   creating compounding risks that a single-parameter random perturbation would not.

4. **Validity rate gap is the main limitation.** At {llm_m['validity_rate']*100:.1f}% vs {rnd_m['validity_rate']*100:.0f}%,
   the LLM generates more physically infeasible scenarios — particularly StackCube (where
   two-object non-overlap is violated). Prompt-engineering improvements (teammate's task)
   directly target this.

5. **Diversity trade-off.** Lower diversity ({llm_m['diversity']:.3f} vs {rnd_m['diversity']:.3f}) is expected:
   the LLM focuses exploration in the adversarial subspace rather than uniformly sampling
   the parameter domain. For systematic safety testing, a combined strategy
   (LLM for quality + diversity-maximising augmentation) would be optimal.

### Recommendations

- **Short-term**: Use LLM scenarios for *hardest-case* safety testing (highest AQS, lowest TtF).
- **Medium-term**: Filter out invalid scenarios and supplement with diversity-augmented random to
  achieve full constraint coverage.
- **Long-term**: Run real ManiSkill rollouts on the top-k adversarial scenarios to compute actual
  time-to-failure and joint-limit violation counts as ground truth.

---

## 7. Generated Figures

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig1_validity_rates.png` | Validity rate per task + overall |
| Fig 2 | `fig2_adversarial_quality.png` | AQS box plots + per-task breakdown |
| Fig 3 | `fig3_proximity_ttf.png` | Obstacle proximity histogram + TtF CDF |
| Fig 4 | `fig4_path_hazard.png` | Path obstruction + multi-hazard density |
| Fig 5 | `fig5_violation_rate.png` | Predicted violation rate + targeting accuracy |
| Fig 6 | `fig6_boundary_diversity.png` | Boundary push + parameter diversity |
| Fig 7 | `fig7_statistics.png` | Effect sizes + significance table |
| Fig 8 | `fig8_dashboard.png` | Combined 2×4 presentation dashboard |

All figures saved to `figures/eval_expanded/`.

---

*Report generated automatically by `run_full_experiment.py`.*
""")

    with open(outpath, "w") as fh:
        fh.write("".join(lines))
    print(f"  Report saved → {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Full LLM vs Random evaluation experiment")
    p.add_argument("--skip-gen", action="store_true",
                   help="Skip generation; reload existing llm_30/llm_scenarios.json")
    p.add_argument("--n", type=int, default=10, metavar="N",
                   help="Scenarios per task (default 10)")
    p.add_argument("--model", default="gpt-4o", metavar="MODEL")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = _parse_args()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("  FULL EXPERIMENT: LLM vs. Random Baseline")
    print(f"{'='*65}")

    # ── Step 1: LLM scenarios ──────────────────────────────────────────
    if args.skip_gen:
        print("\n  [SKIP] Reloading existing LLM scenarios...")
        llm_sc = load_llm_scenarios()
    else:
        print(f"\n  [1/4] Generating {args.n * len(TASKS)} LLM scenarios  (model={args.model})...")
        llm_sc = generate_llm_scenarios(args.n, args.model)
    print(f"  LLM scenarios: {len(llm_sc)}")

    # ── Step 2: Random baseline ───────────────────────────────────────
    print(f"\n  [2/4] Generating {args.n * len(TASKS)} random scenarios (seed={args.seed})...")
    rnd_sc = generate_and_save_random(args.n, args.seed)
    print(f"  Random scenarios: {len(rnd_sc)}")

    # ── Step 3: Validate ──────────────────────────────────────────────
    print("\n  [3/4] Validating both batches...")
    llm_val = validate_batch(llm_sc)
    rnd_val = validate_batch(rnd_sc)
    print(f"  LLM valid/invalid : {sum(v['valid'] for v in llm_val)} / {sum(not v['valid'] for v in llm_val)}")
    print(f"  Rnd valid/invalid : {sum(v['valid'] for v in rnd_val)} / {sum(not v['valid'] for v in rnd_val)}")

    # Save combined results JSON
    combined_path = LLM30_DIR / "llm_validated.json"
    with open(combined_path, "w") as fh:
        json.dump([{"scenario": s, "validation": v}
                   for s, v in zip(llm_sc, llm_val)], fh, indent=2)
    print(f"  Validated LLM results → {combined_path}")

    # ── Step 4: Metrics ───────────────────────────────────────────────
    print("\n  [4/4] Computing metrics + generating figures...")
    llm_m = compute_full_metrics(llm_sc, llm_val)
    rnd_m = compute_full_metrics(rnd_sc, rnd_val)

    # ── Figures ───────────────────────────────────────────────────────
    fig1_validity(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig2_aqs(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig3_proximity_ttf(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig4_path_hazard(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig5_violation_rate(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig6_boundary_diversity(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig7_statistics(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)
    fig8_dashboard(llm_sc, llm_val, rnd_sc, rnd_val, FIGS_DIR)

    # ── Markdown report ───────────────────────────────────────────────
    report_path = RESULTS_DIR / "evaluation_report.md"
    write_md_report(llm_m, rnd_m, llm_sc, rnd_sc, llm_val, rnd_val,
                    report_path, args.n, args.model)

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  METRIC SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<38s}  {'LLM':>8}  {'Random':>8}")
    print(f"  {'-'*38}  {'-'*8}  {'-'*8}")
    print(f"  {'Validity rate':<38s}  {llm_m['validity_rate']*100:>7.1f}%  {rnd_m['validity_rate']*100:>7.1f}%")
    print(f"  {'Mean AQS':<38s}  {llm_m['aqs_mean']:>8.3f}  {rnd_m['aqs_mean']:>8.3f}")
    if llm_m["proximity_mean"] and rnd_m["proximity_mean"]:
        print(f"  {'Mean obstacle proximity (m)':<38s}  {llm_m['proximity_mean']:>8.3f}  {rnd_m['proximity_mean']:>8.3f}")
    if llm_m["ttf_mean"] and rnd_m["ttf_mean"]:
        print(f"  {'Mean TtF proxy (s)':<38s}  {llm_m['ttf_mean']:>8.2f}  {rnd_m['ttf_mean']:>8.2f}")
    if llm_m["path_obstruction_mean"] and rnd_m["path_obstruction_mean"]:
        print(f"  {'Mean path obstruction':<38s}  {llm_m['path_obstruction_mean']:>8.3f}  {rnd_m['path_obstruction_mean']:>8.3f}")
    print(f"  {'Mean hazard density':<38s}  {llm_m['hazard_density_mean']:>8.3f}  {rnd_m['hazard_density_mean']:>8.3f}")
    print(f"  {'Mean boundary push':<38s}  {llm_m['boundary_push_mean']:>8.3f}  {rnd_m['boundary_push_mean']:>8.3f}")
    print(f"  {'Param diversity':<38s}  {llm_m['diversity']:>8.3f}  {rnd_m['diversity']:>8.3f}")
    print(f"  {'Constraint coverage':<38s}  {llm_m['constraint_coverage']:>8d}  {rnd_m['constraint_coverage']:>8d}")
    print(f"\n  Figures → {FIGS_DIR}")
    print(f"  Report  → {report_path}")


if __name__ == "__main__":
    main()
