"""
Render Adversarial Scenarios with ManiSkill / SAPIEN
------------------------------------------------------
For each scenario in a result JSON file this script:

  1. Loads the corresponding Adv* custom task (which includes the obstacle)
  2. Resets the environment with the scenario's exact parameter values
  3. Captures renders from multiple camera angles using SAPIEN's renderer
  4. Saves per-scenario PNG files and a composite contact-sheet PNG

Requirements:
  - ManiSkill 3 installed  (pip install mani-skill)
  - GPU not required;  render_mode="rgb_array" works on CPU with software rendering
    but a GPU (or at least Vulkan / EGL) is strongly recommended for speed.

Usage:
    # Render the most recent result file (demo or generated):
    python render_env/render_scenarios.py

    # Specify an input file:
    python render_env/render_scenarios.py --file results/demo_scenarios_<ts>.json

    # Render only valid scenarios:
    python render_env/render_scenarios.py --valid-only

    # Limit to first N scenarios (fast preview):
    python render_env/render_scenarios.py --max 6

    # Save output to a custom directory:
    python render_env/render_scenarios.py --outdir renders/

Output layout (inside --outdir):
    <scenario_id>_front.png      — 3/4-angle front view (640×480)
    <scenario_id>_overhead.png   — overhead view captured via sensor camera
    contact_sheet.png            — all scenarios tiled in one figure

NOTE: If you see a black frame, the scene has not been stepped yet.  The script
calls env.step(env.action_space.sample()) once before rendering so that
lighting/shadows are fully initialized.
"""

import sys
import os
import json
import glob
import argparse
import math
from pathlib import Path

import numpy as np
import gymnasium as gym

# Pull in our custom task definitions before any gym.make() calls
_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_HERE))

import adv_tasks  # registers Adv* envs  # noqa: F401
from adv_tasks import ADV_TASK_MAP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

C_VALID   = "#2ecc71"
C_INVALID = "#e74c3c"

CAMERA_ANGLES = {
    "front":    "render_camera",    # human render camera (3/4-angle views)
    "overhead": "base_camera",      # sensor camera pointing somewhat down
}

# Human-readable task descriptions for contact-sheet captions
_TASK_DESC = {
    "PickCube-v1":         "Robot must grasp and lift the cube off the table",
    "StackCube-v1":        "Robot must pick the red cube and stack it on the blue cube",
    "PushCube-v1":         "Robot must push the cube to the red/white goal zone without grasping",
    "PlaceSphere-v1":      "Robot must pick up the sphere and drop it into the bowl",
    "PegInsertionSide-v1": "Robot must grasp the peg and insert it sideways into the box hole",
}

_CONSTRAINT_DESC = {
    "collision_avoidance":        "obstacle blocks the direct pick/push path",
    "min_clearance_to_obstacles": "obstacle crowds the end-effector clearance zone",
    "joint_limit_violation":      "tight layout pushes joints near their limits",
    "max_ee_speed":               "congested layout forces dangerously high EE speed",
    "grasp_force_limit":          "positioning risks excessive gripper contact force",
}


def _scene_caption(r: dict) -> str:
    """Build a one-sentence description of the rendered scene."""
    task       = r.get("task", "?")
    constraint = r.get("target_constraint", "?")
    rationale  = (r.get("rationale") or "").strip()
    task_desc  = _TASK_DESC.get(task, task)
    constr_desc = _CONSTRAINT_DESC.get(constraint, constraint.replace("_", " "))
    if rationale:
        caption = rationale[:150].rstrip()
        if len(rationale) > 150:
            caption += "\u2026"
    else:
        caption = f"{task_desc}. Adversarial setup: {constr_desc}."
    return caption


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _to_numpy_rgb(arr) -> np.ndarray:
    """
    Safely convert whatever render() returns (CUDA tensor, CPU tensor, ndarray)
    into a (H, W, 3) uint8 numpy array.
    """
    import torch
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:        # (N, H, W, C) batched → take first
        arr = arr[0]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def _render_scenario(scenario: dict, validation: dict, outdir: Path) -> dict:
    """
    Instantiate the task env, set the scenario parameters, render, save PNGs.

    Returns:
        dict with keys "scenario_id", "valid", "images" {view_name: np.array}
    """
    task_name = scenario.get("task", "")
    env_id    = ADV_TASK_MAP.get(task_name)
    sid       = scenario.get("scenario_id", "unknown")
    params    = scenario.get("parameters", {})
    valid     = validation.get("valid", False)
    errors    = validation.get("errors", [])

    if env_id is None:
        print(f"  [SKIP] No Adv env registered for task '{task_name}'")
        return {"scenario_id": sid, "valid": valid, "images": {}}

    print(f"  {'✓' if valid else '✗'}  {sid}  ({task_name})")

    images = {}
    env = None
    try:
        env = gym.make(
            env_id,
            obs_mode="rgb",
            render_mode="rgb_array",
            num_envs=1,
            # Pass obstacle_size here for _load_scene; also redundantly in options below
            **_extract_load_kwargs(params),
        )
        obs, info = env.reset(options=params)

        # Step once with a zero action so the physics + shadows initialize
        zero_action = np.zeros(env.single_action_space.shape, dtype=np.float32)
        env.step(zero_action)

        # --- front / human render camera ---
        rgb_front = env.render()   # returns tensor or ndarray via render_mode="rgb_array"
        if rgb_front is not None:
            img_arr = _to_numpy_rgb(rgb_front)
            images["front"] = img_arr
            out = outdir / f"{sid}_front.png"
            Image.fromarray(img_arr).save(out)

        # --- sensor camera (overhead-ish) ---
        sensor_imgs = env.unwrapped.get_sensor_images()
        if sensor_imgs:
            cam_key = list(sensor_imgs.keys())[0]
            rgb_s = sensor_imgs[cam_key].get("rgb")
            if rgb_s is not None:
                img_s = _to_numpy_rgb(rgb_s)
                images["sensor"] = img_s
                out = outdir / f"{sid}_sensor.png"
                Image.fromarray(img_s).save(out)

    except Exception as exc:
        print(f"    WARNING: render failed — {exc}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    return {"scenario_id": sid, "valid": valid, "images": images, "errors": errors,
            "task": task_name, "target_constraint": scenario.get("target_constraint", "?"),
            "rationale": scenario.get("rationale", "")}


def _extract_load_kwargs(params: dict) -> dict:
    """
    Pass obstacle_size to gym.make so _load_scene can size the box correctly.
    ManiSkill forwards unknown kwargs to the env constructor via **kwargs.
    """
    # We pass it as env kwargs; adv_tasks reads from options dict in _load_scene
    # so this is informational only — the tasks re-read from options inside the env
    return {}


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def _make_contact_sheet(render_results: list, outpath: Path):
    """
    Tile rendered front images in a large grid (max 3 cols) with a
    description table below — separating images from text entirely so
    both are clearly readable.
    """
    entries = [r for r in render_results if "front" in r.get("images", {})]
    if not entries:
        print("  No rendered images available for contact sheet.")
        return

    n      = len(entries)
    ncols  = min(3, n)
    nrows  = math.ceil(n / ncols)

    # Each image cell is 5.5 × 4.5 inches; add 0.5 header row + table rows below
    # Table: ~0.35 in per row, plus 1 in for header
    n_table_rows = n + 1          # header + one row per scenario
    table_h      = n_table_rows * 0.38 + 0.6
    img_h        = nrows * 4.5

    fig = plt.figure(figsize=(ncols * 5.5, img_h + table_h + 0.6),
                     facecolor="white")

    # GridSpec: top block = images, bottom block = description table
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[img_h, table_h],
        hspace=0.08,
        left=0.03, right=0.97, top=0.96, bottom=0.01,
    )

    # ---- inner grid for image cells ----
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=gs[0],
                                    hspace=0.12, wspace=0.06)

    img_axes = []
    for idx in range(nrows * ncols):
        ax = fig.add_subplot(inner[idx // ncols, idx % ncols])
        img_axes.append(ax)

    for idx, r in enumerate(entries):
        ax    = img_axes[idx]
        valid = r["valid"]
        color = C_VALID if valid else C_INVALID
        status = "✓ VALID" if valid else "✗ INVALID"

        ax.imshow(r["images"]["front"], aspect="auto")
        ax.axis("off")

        # Thin coloured border using Rectangle spanning the axes
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)

        # Compact title: index + task + status badge
        task_short = r["task"].replace("InsertionSide", "Ins.").replace("-v1", "")
        ax.set_title(
            f"#{idx+1}  {task_short}   [{r.get('target_constraint','?')}]   {status}",
            fontsize=8, color=color, fontweight="bold", pad=5,
        )

    # Hide unused cells
    for idx in range(n, nrows * ncols):
        img_axes[idx].set_visible(False)

    # ---- description table ----
    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")

    col_labels  = ["#", "Task", "Constraint", "Status", "Scene Description"]
    col_widths  = [0.03, 0.12, 0.16, 0.07, 0.62]   # fractions of table width

    rows_data = []
    for idx, r in enumerate(entries):
        valid  = r["valid"]
        status = "✓ Valid" if valid else "✗ Invalid"
        err_suffix = ""
        if not valid and r.get("errors"):
            err_suffix = "  [ERR: " + r["errors"][0][:60] + "]"
        caption = _scene_caption(r)
        rows_data.append([
            str(idx + 1),
            r["task"].replace("-v1", ""),
            r.get("target_constraint", "?").replace("_", " "),
            status,
            caption + err_suffix,
        ])

    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)   # row height

    # Style header
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")

    # Style data rows — alternate shading + validity colour for status cell
    for i, r in enumerate(entries):
        valid  = r["valid"]
        bg     = "#f0faf4" if (i % 2 == 0) else "#ffffff"
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#dee2e6")
            if j == 3:   # status column
                cell.set_text_props(color=C_VALID if valid else C_INVALID,
                                    fontweight="bold")

    # ---- legend above the table ----
    legend_items = [
        mpatches.Patch(facecolor="none", edgecolor=C_VALID,   linewidth=2, label="Valid scenario"),
        mpatches.Patch(facecolor="none", edgecolor=C_INVALID, linewidth=2, label="Invalid scenario"),
        mpatches.Patch(color="#c0392b",  label="Adversarial obstacle"),
        mpatches.Patch(color="#1abc9c",  label="Robot (Franka Panda)"),
    ]
    ax_tbl.legend(handles=legend_items, loc="upper right", ncol=4,
                  fontsize=8, frameon=True,
                  bbox_to_anchor=(1.0, 1.05), bbox_transform=ax_tbl.transAxes)

    fig.suptitle(
        "ManiSkill Adversarial Scenario Initial States  "
        "(SAPIEN photo-realistic renderer)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Contact sheet saved → {outpath}")


# ---------------------------------------------------------------------------
# Stats figure from render results
# ---------------------------------------------------------------------------

def _make_render_stats(render_results: list, outpath: Path):
    """Bar chart: rendered count by task × validity."""
    from collections import Counter

    tasks   = [r["task"]  for r in render_results if r["images"]]
    valid_f = [r["valid"] for r in render_results if r["images"]]

    if not tasks:
        return

    task_list  = sorted(set(tasks))
    valid_cnt  = Counter(t for t, v in zip(tasks, valid_f) if v)
    invalid_cnt= Counter(t for t, v in zip(tasks, valid_f) if not v)

    fig, ax = plt.subplots(figsize=(max(6, len(task_list) * 1.6), 4),
                           constrained_layout=True)
    x = range(len(task_list))
    bv = [valid_cnt.get(t, 0)   for t in task_list]
    bi = [invalid_cnt.get(t, 0) for t in task_list]
    ax.bar(x, bv, label="Valid",   color=C_VALID,   alpha=0.85)
    ax.bar(x, bi, bottom=bv,       label="Invalid", color=C_INVALID, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("-v1", "") for t in task_list], rotation=20, ha="right")
    ax.set_ylabel("Rendered scenarios")
    ax.set_title("Rendered Adversarial Scenarios per Task")
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Annotate total rendered
    total = len(tasks)
    ax.text(0.98, 0.95, f"Total rendered: {total}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="goldenrod"))

    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stats figure saved   → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_latest_results():
    results_dir = _PROJ / "results"
    files = sorted(
        glob.glob(str(results_dir / "*.json")),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {results_dir}.  "
            "Run demo.py or run_generation.py first."
        )
    return files[0]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render adversarial scenario initial states with ManiSkill/SAPIEN"
    )
    parser.add_argument("--file", "-f", default=None,
                        help="Path to result JSON (default: most recent in results/)")
    parser.add_argument("--outdir", "-o", default=None,
                        help="Output directory for renders (default: results/renders/)")
    parser.add_argument("--valid-only", action="store_true",
                        help="Only render valid scenarios")
    parser.add_argument("--invalid-only", action="store_true",
                        help="Only render invalid scenarios")
    parser.add_argument("--max", type=int, default=None, metavar="N",
                        help="Render at most N scenarios (useful for quick preview)")
    return parser.parse_args()


def main():
    args = _parse_args()

    json_path = args.file or _find_latest_results()
    print(f"\n  Loading scenarios from: {json_path}")

    with open(json_path) as fh:
        entries = json.load(fh)

    # Filter
    if args.valid_only:
        entries = [e for e in entries if e["validation"]["valid"]]
        print(f"  Filtering to valid only: {len(entries)} scenarios")
    elif args.invalid_only:
        entries = [e for e in entries if not e["validation"]["valid"]]
        print(f"  Filtering to invalid only: {len(entries)} scenarios")

    if args.max:
        entries = entries[: args.max]
        print(f"  Capped at {args.max} scenarios")

    # Output directory
    stem = Path(json_path).stem
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(json_path).parent / "renders" / stem
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {outdir}\n")

    # Render
    render_results = []
    for entry in entries:
        rr = _render_scenario(entry["scenario"], entry["validation"], outdir)
        render_results.append(rr)

    # Composite figures
    print("\n  Building composite figures...")
    _make_contact_sheet(render_results, outdir / "contact_sheet.png")
    _make_render_stats(render_results,  outdir / "render_stats.png")

    # Summary
    n_rendered = sum(1 for r in render_results if r["images"])
    print(f"\n  ({'✓' * n_rendered}) Rendered {n_rendered}/{len(entries)} scenarios")
    print(f"  All files in: {outdir}")


if __name__ == "__main__":
    main()
