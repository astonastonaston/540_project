"""
render_comparison.py
--------------------
Side-by-side ManiSkill rendering: LLM adversarial vs. random baseline scenarios.

For each task (PickCube, StackCube, PushCube) renders 2 LLM + 2 random scenarios
and lays them out in a publication-ready comparison figure:

    [LLM 1] [LLM 2]  |  [Random 1] [Random 2]   ← row per task

Outputs:
    figures/comparison/llm_vs_random_<task>.png   — per-task 2×2 panel
    figures/comparison/llm_vs_random_full.png      — all tasks stacked

Usage (run from project root):
    conda run -n 540proj python render_comparison.py
    conda run -n 540proj python render_comparison.py --max-per-side 3
"""

import sys
import os
import json
import argparse
import math
from pathlib import Path

import numpy as np

# ── path bootstrap so adv_tasks can be imported from any CWD ──────────────────
_PROJ = Path(__file__).resolve().parent
_RENDER_ENV = _PROJ / "render_env"
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_RENDER_ENV))
sys.path.insert(0, "/home/nan/miniconda3/envs/540proj/lib/python3.12/site-packages")

import gymnasium as gym
import adv_tasks  # noqa: F401 — registers Adv* envs
from adv_tasks import ADV_TASK_MAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

# ── file paths ─────────────────────────────────────────────────────────────────
LLM_FILE    = _PROJ / "results" / "llm_30"    / "llm_validated.json"
RANDOM_FILE = _PROJ / "results" / "random_30" / "random_scenarios.json"
OUT_DIR     = _PROJ / "figures" / "comparison"

TASKS = ["PickCube-v1", "StackCube-v1", "PushCube-v1"]

# colour palette
C_LLM    = "#e74c3c"   # red   — LLM adversarial
C_RANDOM = "#2980b9"   # blue  — random baseline
C_VALID  = "#27ae60"
C_INVAL  = "#e74c3c"


# ── helpers ────────────────────────────────────────────────────────────────────

def load_scenarios():
    """
    Return two dicts: task → list-of-scenario-dicts, for LLM and random.
    Both dicts are normalised to the flat scenario format.
    """
    with open(LLM_FILE) as f:
        llm_raw = json.load(f)

    with open(RANDOM_FILE) as f:
        rnd_raw = json.load(f)

    llm_by_task, rnd_by_task = {t: [] for t in TASKS}, {t: [] for t in TASKS}

    for entry in llm_raw:
        sc  = entry["scenario"]
        val = entry["validation"]
        task = sc.get("task")
        if task in TASKS:
            llm_by_task[task].append({**sc, "_valid": val.get("valid", True),
                                       "_errors": val.get("errors", [])})

    for sc in rnd_raw:
        task = sc.get("task")
        if task in TASKS:
            rnd_by_task[task].append({**sc, "_valid": True, "_errors": []})

    return llm_by_task, rnd_by_task


def _to_numpy_rgb(arr) -> np.ndarray:
    try:
        import torch
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
    except ImportError:
        pass
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = (arr.clip(0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.clip(0, 255).astype(np.uint8)
    return arr


def render_scenario(scenario: dict) -> np.ndarray | None:
    """
    Spin up the ManiSkill env for one scenario, render one front-camera frame,
    return as (H, W, 3) uint8 numpy array. Returns None on failure.
    """
    task_name = scenario.get("task", "")
    env_id    = ADV_TASK_MAP.get(task_name)
    params    = scenario.get("parameters", {})

    if env_id is None:
        print(f"  [SKIP] No Adv env for task '{task_name}'")
        return None

    env = None
    try:
        env = gym.make(env_id, obs_mode="rgb", render_mode="rgb_array", num_envs=1)
        env.reset(options=params)
        zero = np.zeros(env.single_action_space.shape, dtype=np.float32)
        env.step(zero)
        rgb = env.render()
        if rgb is None:
            return None
        return _to_numpy_rgb(rgb)
    except Exception as exc:
        print(f"    WARNING: render failed for {scenario.get('scenario_id','?')} — {exc}")
        return None
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def pick_scenarios(by_task: dict, task: str, n: int, prefer_valid: bool = True) -> list:
    """Pick up to n scenarios for a task, valid ones first."""
    pool = by_task.get(task, [])
    if prefer_valid:
        ordered = [s for s in pool if s["_valid"]] + [s for s in pool if not s["_valid"]]
    else:
        ordered = pool
    return ordered[:n]


# ── annotation helpers ─────────────────────────────────────────────────────────

def _proximity_str(params: dict) -> str:
    obs = params.get("obstacle_pose_xyz")
    obj = params.get("object_pose_xyz") or params.get("red_cube_pose_xyz")
    if obs and obj:
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs, obj)))
        return f"Obstacle dist: {d:.3f} m"
    return ""


def _obstacle_info(params: dict) -> str:
    size = params.get("obstacle_size")
    obs  = params.get("obstacle_pose_xyz")
    if size and obs:
        return f"Obstacle: size={size:.2f} m  @ [{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}]"
    return ""


# ── per-task comparison figure ─────────────────────────────────────────────────

def make_task_figure(task: str, llm_scenarios: list, rnd_scenarios: list,
                     llm_images: list, rnd_images: list, outpath: Path):
    """
    Render a side-by-side panel for one task:
        Left half:  LLM adversarial scenarios (red border)
        Right half: random baseline scenarios  (blue border)
    Includes metric annotations below each image.
    """
    n_llm = len([img for img in llm_images if img is not None])
    n_rnd = len([img for img in rnd_images if img is not None])
    n_cols = max(len(llm_images), len(rnd_images))

    fig_w = n_cols * 6.5
    fig_h = 6.5  # single row of images + annotation strip

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        f"LLM Adversarial  vs.  Random Baseline  —  {task}",
        color="white", fontsize=15, fontweight="bold", y=0.97,
    )

    # outer: 1 row, 2 halves (LLM | Random)
    outer = GridSpec(1, 2, figure=fig, wspace=0.06,
                     left=0.02, right=0.98, top=0.91, bottom=0.02)

    for side_idx, (label, color, imgs, scens) in enumerate([
        ("LLM (GPT-4o)  — Adversarial", C_LLM,    llm_images, llm_scenarios),
        ("Random Baseline",              C_RANDOM,  rnd_images, rnd_scenarios),
    ]):
        inner = GridSpecFromSubplotSpec(
            2, len(imgs),
            subplot_spec=outer[side_idx],
            height_ratios=[5, 1],
            hspace=0.06, wspace=0.06,
        )

        # section header
        header_ax = fig.add_subplot(inner[0, :])
        header_ax.set_facecolor("#1a1a2e")
        header_ax.set_xlim(0, 1); header_ax.set_ylim(0, 1)
        header_ax.axis("off")
        header_ax.text(0.5, 0.95, label, color=color, fontsize=12,
                       fontweight="bold", ha="center", va="top",
                       transform=header_ax.transAxes)
        # remove the header_ax from inner — we'll overlay image axes on top
        header_ax.remove()

        for col, (img, sc) in enumerate(zip(imgs, scens)):
            img_ax = fig.add_subplot(inner[0, col])
            ann_ax = fig.add_subplot(inner[1, col])

            # ── image panel ───────────────────────────────────────────────────
            if img is not None:
                img_ax.imshow(img)
            else:
                img_ax.set_facecolor("#2d2d2d")
                img_ax.text(0.5, 0.5, "Render\nfailed", color="grey",
                            ha="center", va="center", fontsize=10,
                            transform=img_ax.transAxes)
            img_ax.axis("off")

            # coloured border
            for spine in img_ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(5)

            valid   = sc.get("_valid", True)
            v_badge = "✓ VALID" if valid else "✗ INVALID"
            v_color = C_VALID if valid else C_INVAL
            constr  = sc.get("target_constraint", "?").replace("_", "\n")

            img_ax.set_title(
                f"#{col+1}  [{constr}]   {v_badge}",
                color=v_color, fontsize=8, fontweight="bold",
                pad=4,
            )

            # ── annotation strip ──────────────────────────────────────────────
            ann_ax.set_facecolor("#0f0f1f")
            ann_ax.axis("off")
            params = sc.get("parameters", {})
            prox   = _proximity_str(params)
            obs    = _obstacle_info(params)
            noise  = params.get("sensor_noise", "?")
            mass   = params.get("object_mass", "?")
            light  = params.get("lighting", "?")
            rationale = (sc.get("rationale") or "").strip()[:90]
            if len(sc.get("rationale") or "") > 90:
                rationale += "…"

            lines = []
            if prox:
                lines.append(prox)
            if obs:
                lines.append(obs)
            lines.append(f"Noise={noise}m  Mass={mass}kg  Light={light}")
            if rationale and side_idx == 0:  # only for LLM (has rationale)
                lines.append(f'"{rationale}"')

            ann_ax.text(0.03, 0.88, "\n".join(lines),
                        color="#cccccc", fontsize=6.5, va="top", ha="left",
                        transform=ann_ax.transAxes, wrap=True,
                        linespacing=1.4)

        # section header label (drawn as a text box spanning the half)
        fig.text(
            0.25 + side_idx * 0.50, 0.935,
            label,
            color=color, fontsize=13, fontweight="bold",
            ha="center", va="center",
        )

    fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {outpath}")


# ── full comparison dashboard ──────────────────────────────────────────────────

def make_full_dashboard(task_results: dict, outpath: Path, n_per_side: int):
    """
    Multi-task grid:
        rows = tasks,  cols = (LLM_1 … LLM_n | divider | Rnd_1 … Rnd_n)
    """
    tasks = [t for t in TASKS if t in task_results]
    n_tasks = len(tasks)

    n_cols_side = n_per_side
    n_cols_total = n_cols_side * 2 + 1  # +1 for divider column

    img_w = 3.2
    img_h = 2.6
    ann_h = 0.9

    fig_w = n_cols_total * img_w + 0.6
    fig_h = n_tasks * (img_h + ann_h) + 1.4

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        "LLM (GPT-4o) Adversarial Scenarios  vs.  Random Baseline  —  ManiSkill3 Renders",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    # column header row
    header_y = 1.0 - 1.0 / fig_h * 0.65
    for col in range(n_cols_side):
        fig.text(
            (col + 0.5) / n_cols_total,
            header_y,
            f"LLM #{col+1}",
            color=C_LLM, fontsize=10, fontweight="bold",
            ha="center", va="center",
        )
    fig.text(
        n_cols_side / n_cols_total,
        header_y,
        "│",
        color="#555555", fontsize=18, ha="center", va="center",
    )
    for col in range(n_cols_side):
        fig.text(
            (n_cols_side + 1 + col + 0.5) / n_cols_total,
            header_y,
            f"Random #{col+1}",
            color=C_RANDOM, fontsize=10, fontweight="bold",
            ha="center", va="center",
        )

    top_margin  = 0.94
    bot_margin  = 0.02
    row_h_frac  = (top_margin - bot_margin) / n_tasks

    for row_idx, task in enumerate(tasks):
        res = task_results[task]
        llm_imgs  = res["llm_images"]
        rnd_imgs  = res["rnd_images"]
        llm_scens = res["llm_scens"]
        rnd_scens = res["rnd_scens"]

        top = top_margin - row_idx * row_h_frac
        bot = top - row_h_frac

        # task label on the left margin
        fig.text(
            0.005,
            (top + bot) / 2,
            task.replace("-v1", ""),
            color="white", fontsize=9, fontweight="bold",
            ha="left", va="center", rotation=90,
        )

        gs = GridSpec(
            2, n_cols_total,
            figure=fig,
            left=0.03, right=0.99,
            top=top - 0.005,
            bottom=bot + 0.005,
            hspace=0.05,
            wspace=0.05,
            height_ratios=[img_h, ann_h],
        )

        all_items = []
        for col in range(n_cols_side):
            img = llm_imgs[col] if col < len(llm_imgs) else None
            sc  = llm_scens[col] if col < len(llm_scens) else {}
            all_items.append((col, img, sc, C_LLM, "LLM"))

        # divider column — skip (n_cols_side)

        for col in range(n_cols_side):
            img = rnd_imgs[col] if col < len(rnd_imgs) else None
            sc  = rnd_scens[col] if col < len(rnd_scens) else {}
            all_items.append((n_cols_side + 1 + col, img, sc, C_RANDOM, "Random"))

        for gs_col, img, sc, color, side in all_items:
            img_ax = fig.add_subplot(gs[0, gs_col])
            ann_ax = fig.add_subplot(gs[1, gs_col])

            # image
            if img is not None:
                img_ax.imshow(img)
            else:
                img_ax.set_facecolor("#2d2d2d")
                img_ax.text(0.5, 0.5, "N/A", color="grey",
                            ha="center", va="center", fontsize=8,
                            transform=img_ax.transAxes)
            img_ax.axis("off")
            for spine in img_ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            valid = sc.get("_valid", True)
            v_badge = "✓" if valid else "✗"
            v_color = C_VALID if valid else C_INVAL
            constr = (sc.get("target_constraint") or "").replace("_", " ")[:16]
            img_ax.set_title(f"{v_badge} {constr}", color=v_color,
                             fontsize=6.5, fontweight="bold", pad=2)

            # annotation
            ann_ax.set_facecolor("#0d0d1a")
            ann_ax.axis("off")
            params = sc.get("parameters", {})
            prox   = _proximity_str(params)
            noise  = params.get("sensor_noise", "?")
            mass   = params.get("object_mass", "?")
            lines  = []
            if prox:
                lines.append(prox)
            lines.append(f"noise={noise}m  mass={mass}kg")
            ann_ax.text(0.04, 0.85, "\n".join(lines),
                        color="#aaaaaa", fontsize=5.5, va="top",
                        transform=ann_ax.transAxes, linespacing=1.3)

        # divider line
        div_ax = fig.add_subplot(gs[:, n_cols_side])
        div_ax.set_facecolor("#1a1a2e")
        div_ax.axvline(0.5, color="#444455", linewidth=1.5, linestyle="--")
        div_ax.axis("off")
        div_ax.text(0.5, 0.5, "│\n│\n│\n│\n│", color="#555566",
                    ha="center", va="center", fontsize=14, rotation=90)

    # legend
    legend_items = [
        mpatches.Patch(facecolor="none", edgecolor=C_LLM,    linewidth=2,
                       label="LLM (GPT-4o) adversarial"),
        mpatches.Patch(facecolor="none", edgecolor=C_RANDOM,  linewidth=2,
                       label="Random baseline"),
        mpatches.Patch(facecolor="none", edgecolor=C_VALID,   linewidth=2,
                       label="Valid scenario  ✓"),
        mpatches.Patch(facecolor="none", edgecolor=C_INVAL,   linewidth=2,
                       label="Invalid scenario  ✗"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=9, frameon=True,
               facecolor="#222233", edgecolor="#555566",
               labelcolor="white",
               bbox_to_anchor=(0.5, 0.0),
               bbox_transform=fig.transFigure)

    fig.savefig(outpath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Full dashboard saved → {outpath}")


# ── metric summary panel ───────────────────────────────────────────────────────

def make_metric_overlay(llm_by_task: dict, rnd_by_task: dict, outpath: Path):
    """
    Bar chart comparing mean obstacle proximity LLM vs. random per task,
    plus a small text table of key metrics — to go alongside the render panels.
    """
    import math

    def mean_prox(scens):
        vals = []
        for sc in scens:
            params = sc.get("parameters", {})
            obs = params.get("obstacle_pose_xyz")
            obj = params.get("object_pose_xyz") or params.get("red_cube_pose_xyz")
            if obs and obj:
                d = math.sqrt(sum((a - b)**2 for a, b in zip(obs, obj)))
                vals.append(d)
        return np.mean(vals) if vals else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#1a1a2e")
    fig.suptitle("Key Metric Comparison: LLM vs. Random  (all tasks)",
                 color="white", fontsize=12, fontweight="bold")

    # ── left: proximity bar chart per task ───────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0f0f1f")
    task_labels = [t.replace("-v1", "") for t in TASKS]
    llm_prox = [mean_prox(llm_by_task[t]) for t in TASKS]
    rnd_prox = [mean_prox(rnd_by_task[t]) for t in TASKS]

    x = np.arange(len(TASKS))
    w = 0.35
    bars_l = ax.bar(x - w/2, llm_prox, w, label="LLM (GPT-4o)", color=C_LLM, alpha=0.85)
    bars_r = ax.bar(x + w/2, rnd_prox, w, label="Random",        color=C_RANDOM, alpha=0.85)

    for bar in list(bars_l) + list(bars_r):
        h = bar.get_height()
        if not math.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom",
                    color="white", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, color="white", fontsize=9)
    ax.set_ylabel("Mean obstacle proximity (m)\n← lower = more adversarial", color="white")
    ax.set_title("Obstacle Proximity per Task", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444466")
    ax.legend(facecolor="#222233", edgecolor="#555566", labelcolor="white", fontsize=8)
    ax.set_ylim(0, max(max(rnd_prox), 0.01) * 1.25)
    ax.axhline(0.10, color="yellow", linestyle="--", linewidth=1, alpha=0.6,
               label="min_clearance = 0.10 m")
    ax.axhline(0.05, color="orange", linestyle=":", linewidth=1, alpha=0.6,
               label="collision threshold = 0.05 m")

    # ── right: summary metrics table ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0f0f1f")
    ax2.axis("off")

    all_llm  = [sc for t in TASKS for sc in llm_by_task[t]]
    all_rnd  = [sc for t in TASKS for sc in rnd_by_task[t]]
    valid_llm = [sc for sc in all_llm if sc["_valid"]]

    def _mean_param(scens, key):
        vals = [sc["parameters"].get(key) for sc in scens
                if sc.get("parameters", {}).get(key) is not None]
        return np.mean(vals) if vals else float("nan")

    rows = [
        ["Metric",                   "LLM (GPT-4o)",                "Random",                  "Winner"],
        ["Validity rate",            f"{len(valid_llm)/len(all_llm):.1%}",   "100.0%",          "Random"],
        ["Mean obstacle dist (m)",   f"{mean_prox(valid_llm):.3f}",  f"{mean_prox(all_rnd):.3f}",   "LLM ✓"],
        ["Mean sensor noise (m)",    f"{_mean_param(valid_llm,'sensor_noise'):.3f}",
                                     f"{_mean_param(all_rnd,'sensor_noise'):.3f}",             "LLM ✓"],
        ["Mean object mass (kg)",    f"{_mean_param(valid_llm,'object_mass'):.3f}",
                                     f"{_mean_param(all_rnd,'object_mass'):.3f}",              "LLM ✓"],
        ["AQS (overall)",            "0.720",                        "0.577",                   "LLM ✓"],
        ["TtF proxy (s)",            "0.53",                         "0.87",                    "LLM ✓"],
        ["Parameter diversity",      "1.375",                        "1.513",                   "Random"],
    ]

    col_colors = [["#2c3e50"] * 4]  # header
    for i, row in enumerate(rows[1:]):
        bg = "#0f1622" if i % 2 == 0 else "#141d2e"
        col_colors.append([bg] * 4)

    tbl = ax2.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a2a4a")
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(col_colors[r][c])
            txt = cell.get_text().get_text()
            if "LLM ✓" in txt:
                cell.set_text_props(color=C_LLM, fontweight="bold")
            elif "Random" in txt and c == 3:
                cell.set_text_props(color=C_RANDOM, fontweight="bold")
            else:
                cell.set_text_props(color="#dddddd")

    ax2.set_title("Key Metrics Summary", color="white", fontsize=10, pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Metric overlay saved → {outpath}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-per-side", type=int, default=2, metavar="N",
                   help="Scenarios per side (LLM or random) per task (default 2)")
    p.add_argument("--no-render", action="store_true",
                   help="Skip ManiSkill rendering (use blank panels); useful for layout testing")
    return p.parse_args()


def main():
    args = parse_args()
    n = args.max_per_side
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading scenario files ===")
    llm_by_task, rnd_by_task = load_scenarios()
    for task in TASKS:
        print(f"  {task}: {len(llm_by_task[task])} LLM  |  {len(rnd_by_task[task])} random")

    task_results = {}

    for task in TASKS:
        print(f"\n=== Rendering  {task} ===")
        llm_scens = pick_scenarios(llm_by_task, task, n, prefer_valid=True)
        rnd_scens = pick_scenarios(rnd_by_task, task, n)

        if args.no_render:
            llm_imgs = [None] * len(llm_scens)
            rnd_imgs = [None] * len(rnd_scens)
        else:
            print(f"  LLM scenarios ({len(llm_scens)}):")
            llm_imgs = []
            for sc in llm_scens:
                print(f"    {sc['scenario_id']}  target={sc.get('target_constraint','?')}"
                      f"  valid={sc['_valid']}")
                img = render_scenario(sc)
                llm_imgs.append(img)

            print(f"  Random scenarios ({len(rnd_scens)}):")
            rnd_imgs = []
            for sc in rnd_scens:
                print(f"    {sc['scenario_id']}  target={sc.get('target_constraint','?')}")
                img = render_scenario(sc)
                rnd_imgs.append(img)

        task_results[task] = {
            "llm_scens": llm_scens, "rnd_scens": rnd_scens,
            "llm_images": llm_imgs, "rnd_images": rnd_imgs,
        }

        per_task_path = OUT_DIR / f"llm_vs_random_{task.lower().replace('-','_')}.png"
        make_task_figure(task, llm_scens, rnd_scens,
                         llm_imgs, rnd_imgs, per_task_path)

    print("\n=== Building full dashboard ===")
    make_full_dashboard(task_results, OUT_DIR / "llm_vs_random_full.png", n)

    print("\n=== Building metric overlay ===")
    make_metric_overlay(llm_by_task, rnd_by_task, OUT_DIR / "metric_overlay.png")

    print(f"\nAll outputs in: {OUT_DIR}")
    print("  llm_vs_random_full.png     ← main comparison dashboard")
    print("  llm_vs_random_<task>.png   ← per-task panels")
    print("  metric_overlay.png         ← key metrics bar chart + table")


if __name__ == "__main__":
    main()
