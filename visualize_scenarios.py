"""
Scenario Visualizer
--------------------
Generates publication-ready figures from adversarial scenario JSON result files
(output of run_generation.py or demo.py).  No ManiSkill / GPU required.

Two figures are produced:

  Figure 1 — Workspace Layout Grid
      One subplot per scenario showing a top-down 2D view of the robot workspace:
        • Robot base (blue star)
        • Primary object — cube / sphere / peg  (green filled shape)
        • Obstacle  (red filled square)
        • Goal / target object if applicable  (orange dashed square)
        • Subplot border: green = valid, red = invalid

  Figure 2 — Summary Statistics
      Four subplots:
        a) Valid vs Invalid counts per ManiSkill task
        b) Scenarios per targeted constraint
        c) Scatter: obstacle_size vs sensor_noise, coloured by validity
        d) Horizontal stacked bar: pass/fail breakdown per constraint

Usage:
    # Visualise the most recent results file automatically:
    python visualize_scenarios.py

    # Specify a file explicitly:
    python visualize_scenarios.py --file results/generated_scenarios_20260320_032831.json

    # Combine multiple result files (e.g. demo + real):
    python visualize_scenarios.py --file results/demo_scenarios_*.json

    # Save to a custom directory:
    python visualize_scenarios.py --outdir figures/
"""

import sys
import json
import math
import glob
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.lines as mlines

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Workspace constants (metres, top-down XY)
# ---------------------------------------------------------------------------
TABLE_X = (0.0, 0.60)    # robot reach in x
TABLE_Y = (-0.35, 0.35)  # robot reach in y
ROBOT_BASE = (0.0, 0.0)  # fixed robot base


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
C_VALID      = "#2ecc71"   # green
C_INVALID    = "#e74c3c"   # red
C_OBSTACLE   = "#c0392b"   # dark red
C_OBJECT     = "#2980b9"   # blue
C_GOAL       = "#f39c12"   # orange
C_SECONDARY  = "#8e44ad"   # purple (bowl / box)
C_ROBOT      = "#2c3e50"   # dark blue
C_TABLE_FILL = "#ecf0f1"   # light grey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(paths) -> list:
    """Load and merge one or more result JSON files."""
    entries = []
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            entries.extend(data)
        else:
            entries.append(data)
    return entries


def _get_xyz(params: dict, key: str):
    """Safely return [x, y] from a pose parameter (ignore z for top-down view)."""
    val = params.get(key)
    if isinstance(val, list) and len(val) >= 2:
        return val[0], val[1]
    return None


def _draw_cross(ax, x, y, size=0.03, color="black", lw=1.5):
    ax.plot([x - size, x + size], [y, y], color=color, lw=lw, zorder=5)
    ax.plot([x, x], [y - size, y + size], color=color, lw=lw, zorder=5)


def _annotate_obj(ax, x, y, label: str, color: str, dy: float = 0.06):
    """Place a descriptive text label just below (or above) an object marker."""
    ax.text(
        x, y + dy, label,
        ha="center", va="bottom", fontsize=4.5, color=color,
        fontweight="bold", zorder=9,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                  edgecolor=color, linewidth=0.6, alpha=0.85),
    )


# ---------------------------------------------------------------------------
# Figure 1 — Workspace layout grid
# ---------------------------------------------------------------------------

def _draw_workspace(ax, scenario: dict, result: dict):
    """Draw a single top-down workspace subplot for one scenario."""
    valid   = result.get("valid", False)
    task    = scenario.get("task", "?")
    params  = scenario.get("parameters", {})
    target  = scenario.get("target_constraint", "?")
    sid     = scenario.get("scenario_id", "?")[-8:]

    border_color = C_VALID if valid else C_INVALID
    status_label = "VALID ✓" if valid else "INVALID ✗"

    # --- table background ---
    ax.set_facecolor(C_TABLE_FILL)
    table_rect = mpatches.FancyBboxPatch(
        (TABLE_X[0], TABLE_Y[0]),
        TABLE_X[1] - TABLE_X[0],
        TABLE_Y[1] - TABLE_Y[0],
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor=border_color,
        facecolor=C_TABLE_FILL,
        zorder=0,
    )
    ax.add_patch(table_rect)

    # --- robot base ---
    ax.plot(
        ROBOT_BASE[0], ROBOT_BASE[1],
        marker="*", markersize=12, color=C_ROBOT,
        zorder=8, label="Robot base",
    )

    # --- obstacle ---
    obs_xy  = _get_xyz(params, "obstacle_pose_xyz")
    obs_sz  = params.get("obstacle_size", 0.06)
    if obs_xy and isinstance(obs_sz, (int, float)):
        half = obs_sz / 2
        obs_rect = mpatches.Rectangle(
            (obs_xy[0] - half, obs_xy[1] - half),
            obs_sz, obs_sz,
            linewidth=1.5, edgecolor="white",
            facecolor=C_OBSTACLE, alpha=0.85, zorder=6,
        )
        ax.add_patch(obs_rect)
        ax.text(
            obs_xy[0], obs_xy[1], "✕", color="white",
            ha="center", va="center", fontsize=5, fontweight="bold", zorder=7,
        )
        _annotate_obj(ax, obs_xy[0], obs_xy[1] + half, "Obstacle", C_OBSTACLE)

    # --- primary object (task-specific) ---
    _draw_task_objects(ax, task, params)

    # --- axis formatting ---
    margin = 0.05
    ax.set_xlim(TABLE_X[0] - margin, TABLE_X[1] + margin)
    ax.set_ylim(TABLE_Y[0] - margin, TABLE_Y[1] + margin)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.3, 0.6])
    ax.set_yticks([-0.3, 0.0, 0.3])
    ax.tick_params(labelsize=5)
    ax.set_xlabel("x (m)", fontsize=5, labelpad=1)
    ax.set_ylabel("y (m)", fontsize=5, labelpad=1)

    title = f"{task}\n[{target}]\n{sid}  —  {status_label}"
    ax.set_title(title, fontsize=5.5, pad=3,
                 color=border_color, fontweight="bold")

    # --- validity watermark ---
    ax.text(
        0.98, 0.02, status_label,
        transform=ax.transAxes,
        color=border_color, fontsize=6, alpha=0.5,
        ha="right", va="bottom", fontweight="bold",
    )


def _draw_task_objects(ax, task: str, params: dict):
    """Draw task-specific primary object(s) onto ax."""
    r = 0.025  # default circle radius

    if task in ("PickCube-v1", "PushCube-v1"):
        obj_xy = _get_xyz(params, "object_pose_xyz")
        if obj_xy:
            circ = mpatches.Circle(obj_xy, r, color=C_OBJECT, zorder=6)
            ax.add_patch(circ)
            ax.text(obj_xy[0], obj_xy[1], "■", color="white",
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=7)
            _annotate_obj(ax, obj_xy[0], obj_xy[1] + r, "Grasp target", C_OBJECT)

        if task == "PushCube-v1":
            goal_xy = _get_xyz(params, "goal_pose_xyz")
            if goal_xy:
                goal_rect = mpatches.Rectangle(
                    (goal_xy[0] - 0.04, goal_xy[1] - 0.04), 0.08, 0.08,
                    linewidth=1.5, edgecolor=C_GOAL, facecolor="none",
                    linestyle="--", zorder=5,
                )
                ax.add_patch(goal_rect)
                ax.text(goal_xy[0], goal_xy[1], "⬡", color=C_GOAL,
                        ha="center", va="center", fontsize=5, fontweight="bold", zorder=6)
                _annotate_obj(ax, goal_xy[0], goal_xy[1] + 0.04, "Goal zone", C_GOAL)

    elif task == "StackCube-v1":
        red_xy  = _get_xyz(params, "red_cube_pose_xyz")
        blue_xy = _get_xyz(params, "blue_cube_pose_xyz")
        if red_xy:
            circ = mpatches.Circle(red_xy, r, color="#e74c3c", zorder=6)
            ax.add_patch(circ)
            ax.text(red_xy[0], red_xy[1], "■", color="white",
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=7)
            _annotate_obj(ax, red_xy[0], red_xy[1] + r, "Grasp (red)", "#e74c3c")
        if blue_xy:
            circ = mpatches.Circle(blue_xy, r, color=C_OBJECT, zorder=6)
            ax.add_patch(circ)
            ax.text(blue_xy[0], blue_xy[1], "■", color="white",
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=7)
            _annotate_obj(ax, blue_xy[0], blue_xy[1] + r, "Stack base", C_OBJECT)
        if red_xy and blue_xy:
            ax.annotate(
                "", xy=blue_xy, xytext=red_xy,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                zorder=4,
            )

    elif task == "PlaceSphere-v1":
        sph_xy  = _get_xyz(params, "sphere_pose_xyz")
        bowl_xy = _get_xyz(params, "bowl_pose_xyz")
        sph_r   = params.get("sphere_radius", 0.03)
        _sph_r  = sph_r if isinstance(sph_r, float) else r
        if sph_xy:
            circ = mpatches.Circle(sph_xy, _sph_r, color=C_OBJECT, zorder=6)
            ax.add_patch(circ)
            ax.text(sph_xy[0], sph_xy[1], "●", color="white",
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=7)
            _annotate_obj(ax, sph_xy[0], sph_xy[1] + _sph_r, "Grasp target", C_OBJECT)
        if bowl_xy:
            bowl_circ = mpatches.Circle(bowl_xy, 0.04,
                                        linewidth=1.5, edgecolor=C_SECONDARY,
                                        facecolor="none", linestyle="--", zorder=5)
            ax.add_patch(bowl_circ)
            ax.text(bowl_xy[0], bowl_xy[1], "⬤", color=C_SECONDARY,
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=6)
            _annotate_obj(ax, bowl_xy[0], bowl_xy[1] + 0.04, "Target bin", C_SECONDARY)
        if sph_xy and bowl_xy:
            ax.annotate(
                "", xy=bowl_xy, xytext=sph_xy,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                zorder=4,
            )

    elif task == "PegInsertionSide-v1":
        peg_xy = _get_xyz(params, "peg_pose_xyz")
        box_xy = _get_xyz(params, "box_pose_xyz")
        peg_l  = params.get("peg_length", 0.12)
        if peg_xy:
            if isinstance(peg_l, (int, float)):
                peg_rect = mpatches.Rectangle(
                    (peg_xy[0] - peg_l / 2, peg_xy[1] - 0.012),
                    peg_l, 0.024,
                    color=C_OBJECT, zorder=6,
                )
                ax.add_patch(peg_rect)
            ax.text(peg_xy[0], peg_xy[1], "—", color="white",
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=7)
            _annotate_obj(ax, peg_xy[0], peg_xy[1] + 0.012, "Peg (grasp)", C_OBJECT)
        if box_xy:
            box_rect = mpatches.Rectangle(
                (box_xy[0] - 0.04, box_xy[1] - 0.04), 0.08, 0.08,
                linewidth=1.5, edgecolor=C_SECONDARY, facecolor="none",
                linestyle="--", zorder=5,
            )
            ax.add_patch(box_rect)
            ax.text(box_xy[0], box_xy[1], "□", color=C_SECONDARY,
                    ha="center", va="center", fontsize=5, fontweight="bold", zorder=6)
            _annotate_obj(ax, box_xy[0], box_xy[1] + 0.04, "Insert hole", C_SECONDARY)
        if peg_xy and box_xy:
            ax.annotate(
                "", xy=box_xy, xytext=peg_xy,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                zorder=4,
            )


def make_workspace_figure(entries: list, outpath: Path):
    n = len(entries)
    ncols = min(3, n)   # max 3 columns → 2×3 grid for 6 scenarios (not cluttered)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.2),
        constrained_layout=True,
    )
    if n == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, entry in enumerate(entries):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        _draw_workspace(ax, entry["scenario"], entry["validation"])

    # Hide any unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared legend
    legend_items = [
        mpatches.Patch(color=C_ROBOT,    label="Robot base (★)"),
        mpatches.Patch(color=C_OBJECT,   label="Primary object"),
        mpatches.Patch(color=C_OBSTACLE, label="Obstacle"),
        mpatches.Patch(color=C_GOAL,     label="Goal / target"),
        mpatches.Patch(color=C_SECONDARY,label="Secondary obj"),
        mpatches.Patch(facecolor="none", edgecolor=C_VALID,   linewidth=2, label="Valid scenario"),
        mpatches.Patch(facecolor="none", edgecolor=C_INVALID, linewidth=2, label="Invalid scenario"),
    ]
    fig.legend(
        handles=legend_items,
        loc="lower center",
        ncol=4,
        fontsize=7,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Adversarial Scenario Workspace Layouts (Top-Down View)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved workspace figure → {outpath}")


# ---------------------------------------------------------------------------
# Figure 2 — Summary statistics
# ---------------------------------------------------------------------------

def make_statistics_figure(entries: list, outpath: Path):
    from collections import Counter

    tasks        = [e["scenario"].get("task", "?") for e in entries]
    constraints  = [e["scenario"].get("target_constraint", "?") for e in entries]
    valid_flags  = [e["validation"]["valid"] for e in entries]
    obs_sizes    = []
    noise_vals   = []
    for e in entries:
        p = e["scenario"].get("parameters", {})
        obs_sizes.append(p.get("obstacle_size") if isinstance(p.get("obstacle_size"), (int, float)) else None)
        noise_vals.append(p.get("sensor_noise") if isinstance(p.get("sensor_noise"), (int, float)) else None)

    task_list  = sorted(set(tasks))
    const_list = sorted(set(constraints))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    fig.suptitle(
        "Adversarial Scenario Generation — Summary Statistics",
        fontsize=13, fontweight="bold",
    )

    # --- (a) Valid / Invalid per task ---
    ax = axes[0][0]
    task_valid   = Counter(t for t, v in zip(tasks, valid_flags) if v)
    task_invalid = Counter(t for t, v in zip(tasks, valid_flags) if not v)
    x = range(len(task_list))
    bars_v = [task_valid.get(t, 0)   for t in task_list]
    bars_i = [task_invalid.get(t, 0) for t in task_list]
    ax.bar(x, bars_v, label="Valid",   color=C_VALID,   alpha=0.85)
    ax.bar(x, bars_i, bottom=bars_v,   label="Invalid", color=C_INVALID, alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels([t.replace("-v1", "") for t in task_list], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("(a) Valid / Invalid per Task")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- (b) Scenarios per targeted constraint ---
    ax = axes[0][1]
    const_counts = Counter(constraints)
    const_valid_counts = Counter(c for c, v in zip(constraints, valid_flags) if v)
    const_invalid_counts = Counter(c for c, v in zip(constraints, valid_flags) if not v)
    y = range(len(const_list))
    v_vals = [const_valid_counts.get(c, 0)   for c in const_list]
    i_vals = [const_invalid_counts.get(c, 0) for c in const_list]
    ax.barh(list(y), v_vals, label="Valid",   color=C_VALID,   alpha=0.85)
    ax.barh(list(y), i_vals, left=v_vals,     label="Invalid", color=C_INVALID, alpha=0.85)
    ax.set_yticks(list(y))
    short_labels = [c.replace("_", "\n") for c in const_list]
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel("Count")
    ax.set_title("(b) Scenarios per Constraint")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # --- (c) Scatter: obstacle_size vs sensor_noise ---
    ax = axes[1][0]
    for valid, color, label in [(True, C_VALID, "Valid"), (False, C_INVALID, "Invalid")]:
        xs = [s for s, v in zip(obs_sizes,  valid_flags) if v == valid and s is not None]
        ys = [n for n, v in zip(noise_vals, valid_flags) if v == valid and n is not None]
        ax.scatter(xs, ys, color=color, alpha=0.75, s=60, label=label, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Obstacle size (m)")
    ax.set_ylabel("Sensor noise (m)")
    ax.set_title("(c) Obstacle Size vs Sensor Noise")
    ax.legend(fontsize=8)

    # Add schema bounds reference lines
    ax.axvline(0.02, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.axvline(0.15, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.axhline(0.00, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.axhline(0.20, color="gray", lw=0.8, linestyle=":", alpha=0.5)
    ax.text(0.155, 0.21, "schema\nbounds", color="gray", fontsize=6)

    # --- (d) Overall pass rate pie + summary text ---
    ax = axes[1][1]
    n_valid   = sum(valid_flags)
    n_invalid = len(valid_flags) - n_valid
    total     = len(valid_flags)
    if total > 0:
        wedge_colors = [C_VALID, C_INVALID]
        wedge_sizes  = [n_valid, n_invalid]
        labels_pie   = [
            f"Valid\n{n_valid}/{total}\n({n_valid/total*100:.0f}%)",
            f"Invalid\n{n_invalid}/{total}\n({n_invalid/total*100:.0f}%)",
        ]
        wedges, texts = ax.pie(
            wedge_sizes, labels=labels_pie, colors=wedge_colors,
            startangle=90, textprops={"fontsize": 9},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
        )
    ax.set_title("(d) Overall Validation Pass Rate")

    # Error type breakdown as text annotation
    all_errors = []
    for e in entries:
        all_errors.extend(e["validation"].get("errors", []))
    if all_errors:
        from collections import Counter as C2
        # Classify errors into broad buckets
        buckets = {
            "Bounds violation": sum(1 for err in all_errors if "out of bounds" in err),
            "Dict instead of list": sum(1 for err in all_errors if "must be a list" in err),
            "Physics infeasible": sum(1 for err in all_errors if "infeasible" in err or "overlaps" in err),
            "Missing field": sum(1 for err in all_errors if "Missing" in err),
            "Other": 0,
        }
        buckets["Other"] = sum(1 for err in all_errors) - sum(buckets.values())
        buckets = {k: v for k, v in buckets.items() if v > 0}
        if buckets:
            txt = "Validation error types:\n" + "\n".join(
                f"  {k}: {v}" for k, v in buckets.items()
            )
            ax.text(
                0.5, -0.15, txt,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffeeba", edgecolor="#e0a800", alpha=0.9),
            )

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved statistics figure → {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_latest_results():
    """Return the most recently modified JSON file in results/."""
    results_dir = _HERE / "results"
    pattern = str(results_dir / "*.json")
    files = sorted(glob.glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {results_dir}.  "
            "Run demo.py or run_generation.py first."
        )
    return [files[0]]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize adversarial scenario results (workspace layout + statistics)"
    )
    parser.add_argument(
        "--file", "-f",
        nargs="*",
        default=None,
        metavar="JSON",
        help=(
            "Path(s) to result JSON file(s).  Glob patterns accepted.  "
            "Default: most recent file in results/"
        ),
    )
    parser.add_argument(
        "--outdir", "-o",
        default=None,
        metavar="DIR",
        help="Output directory for figures (default: same directory as input file)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # Resolve input files
    if args.file:
        paths = []
        for pattern in args.file:
            expanded = glob.glob(pattern)
            paths.extend(expanded if expanded else [pattern])
    else:
        paths = _find_latest_results()

    print(f"\n  Loading {len(paths)} result file(s)...")
    entries = _load_results(paths)
    print(f"  Total scenarios loaded: {len(entries)}")

    n_valid = sum(1 for e in entries if e["validation"]["valid"])
    print(f"  Valid: {n_valid}  /  Invalid: {len(entries) - n_valid}")

    # Resolve output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(paths[0]).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Derive a stem from the first filename
    stem = Path(paths[0]).stem

    print("\n  Generating figures...")

    workspace_path = outdir / f"{stem}_workspace.png"
    make_workspace_figure(entries, workspace_path)

    stats_path = outdir / f"{stem}_statistics.png"
    make_statistics_figure(entries, stats_path)

    print("\n  Done.  Files written:")
    print(f"    {workspace_path}")
    print(f"    {stats_path}")


if __name__ == "__main__":
    main()
