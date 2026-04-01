"""
Offline Demo — Adversarial Scenario Generation + Validation (No API Key Required)
----------------------------------------------------------------------------------
Demonstrates the full Step 1 → Step 2 pipeline using pre-crafted mock scenarios
that mimic realistic LLM outputs.  Useful for:
  • Project demos / status reports without an active API key
  • Unit-testing the validator logic
  • Showing both valid and invalid scenario handling

Run:
    python demo.py

The script:
  1. Prints each mock scenario (as if just returned by the LLM)
  2. Runs the validator on every scenario
  3. Saves results to  LLMScenarios/results/demo_<timestamp>.json
  4. Prints a summary table
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from validator import validate_scenario, summarise

RESULTS_DIR = _HERE / "results"

# ---------------------------------------------------------------------------
# Mock scenarios
# Annotations show the EXPECTED validation outcome and why.
# ---------------------------------------------------------------------------

MOCK_SCENARIOS = [

    # ── Scenario 1 ──────────────────────────────────────────────────────────
    # Task: PickCube-v1
    # Target: collision_avoidance
    # Strategy: obstacle placed directly on the straight-line path from the robot
    #            base to the cube; large obstacle size + dim lighting.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_pickcube_v1_a1b2c3d4",
        "task": "PickCube-v1",
        "target_constraint": "collision_avoidance",
        "rationale": (
            "The obstacle is centred at (0.37, 0.03, 0.07), which lies directly "
            "on the end-effector's approach trajectory to the cube at (0.47, 0.05, 0.02). "
            "The large obstacle size (0.09 m) combined with dim lighting forces the "
            "robot to navigate blind through a cluttered region."
        ),
        "parameters": {
            "object_pose_xyz": [0.47, 0.05, 0.02],
            "obstacle_pose_xyz": [0.37, 0.03, 0.07],
            "obstacle_size": 0.09,
            "object_mass": 0.45,
            "sensor_noise": 0.14,
            "lighting": "dim",
        },
        "generated_at": "2026-03-21T10:00:00Z",
        "model": "gpt-4o",
    },

    # ── Scenario 2 ──────────────────────────────────────────────────────────
    # Task: PickCube-v1
    # Target: grasp_force_limit
    # Strategy: cube mass near maximum + high sensor noise → robot misjudges
    #            grip force needed.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_pickcube_v1_e5f6g7h8",
        "task": "PickCube-v1",
        "target_constraint": "grasp_force_limit",
        "rationale": (
            "Object mass of 1.90 kg approaches the upper bound; combined with "
            "sensor_noise = 0.17 m the robot's force estimator will significantly "
            "underestimate grip requirements, causing the gripper to overshoot "
            "the 15 N force limit."
        ),
        "parameters": {
            "object_pose_xyz": [0.35, 0.10, 0.02],
            "obstacle_pose_xyz": [0.60, 0.30, 0.08],
            "obstacle_size": 0.04,
            "object_mass": 1.90,
            "sensor_noise": 0.17,
            "lighting": "dim",
        },
        "generated_at": "2026-03-21T10:00:01Z",
        "model": "gpt-4o",
    },

    # ── Scenario 3 ──────────────────────────────────────────────────────────
    # Task: StackCube-v1
    # Target: joint_limit_violation
    # Strategy: red cube at far negative-Y extreme, blue cube at far positive-Y
    #            extreme — robot must execute a near-180° horizontal reach that
    #            pushes joint 7 toward its limit.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_stackcube_v1_i9j0k1l2",
        "task": "StackCube-v1",
        "target_constraint": "joint_limit_violation",
        "rationale": (
            "Placing the red cube at y = -0.23 and the blue cube at y = +0.23 "
            "on opposite sides of the workspace forces a combined reach and rotation "
            "that drives joint 1 and joint 7 toward their angular limits during the "
            "transfer phase."
        ),
        "parameters": {
            "red_cube_pose_xyz": [0.45, -0.23, 0.02],
            "blue_cube_pose_xyz": [0.10, 0.23, 0.02],
            "obstacle_pose_xyz": [0.27, 0.00, 0.10],
            "obstacle_size": 0.06,
            "object_mass": 0.80,
            "sensor_noise": 0.05,
            "lighting": "normal",
        },
        "generated_at": "2026-03-21T10:00:02Z",
        "model": "gpt-4o",
    },

    # ── Scenario 4 ──────────────────────────────────────────────────────────
    # Task: PushCube-v1
    # Target: collision_avoidance
    # Strategy: obstacle placed directly on the push vector midpoint; robot
    #            must deviate but the narrow gap triggers a collision.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_pushcube_v1_m3n4o5p6",
        "task": "PushCube-v1",
        "target_constraint": "collision_avoidance",
        "rationale": (
            "The obstacle at (0.35, 0.08, 0.06) sits on the straight-line push "
            "path from cube start (0.22, 0.0, 0.02) to goal (0.50, 0.18, 0.02), "
            "forcing the end-effector into contact with the obstacle mid-push."
        ),
        "parameters": {
            "object_pose_xyz": [0.22, 0.00, 0.02],
            "goal_pose_xyz": [0.50, 0.18, 0.02],
            "obstacle_pose_xyz": [0.35, 0.08, 0.06],
            "obstacle_size": 0.07,
            "object_mass": 0.30,
            "sensor_noise": 0.10,
            "lighting": "normal",
        },
        "generated_at": "2026-03-21T10:00:03Z",
        "model": "gpt-4o",
    },

    # ── Scenario 5 ──────────────────────────────────────────────────────────
    # Task: PlaceSphere-v1
    # Target: min_clearance_to_obstacles
    # Strategy: obstacle placed exactly between sphere and bowl on the approach
    #            arc, with its top surface at the same height as the sphere.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_placesphere_v1_q7r8s9t0",
        "task": "PlaceSphere-v1",
        "target_constraint": "min_clearance_to_obstacles",
        "rationale": (
            "The obstacle is centred at (0.37, 0.01, 0.08), midpoint between "
            "sphere (0.28, -0.08, 0.03) and bowl (0.46, 0.10, 0.01). "
            "Its top surface (z≈0.12 m) lies in the end-effector's descent "
            "path, reducing clearance below the 0.10 m threshold."
        ),
        "parameters": {
            "sphere_pose_xyz": [0.28, -0.08, 0.03],
            "bowl_pose_xyz": [0.46, 0.10, 0.01],
            "obstacle_pose_xyz": [0.37, 0.01, 0.08],
            "obstacle_size": 0.08,
            "sphere_radius": 0.03,
            "object_mass": 0.20,
            "sensor_noise": 0.09,
            "lighting": "normal",
        },
        "generated_at": "2026-03-21T10:00:04Z",
        "model": "gpt-4o",
    },

    # ── Scenario 6 ──────────────────────────────────────────────────────────
    # Task: PegInsertionSide-v1
    # Target: max_ee_speed
    # Strategy: peg at far edge, box near robot base — long fast travel then
    #            abrupt deceleration at the insertion point.
    # Expected: VALID ✓
    {
        "scenario_id": "mock_peginsert_v1_u1v2w3x4",
        "task": "PegInsertionSide-v1",
        "target_constraint": "max_ee_speed",
        "rationale": (
            "Peg at (0.42, -0.22, 0.05) is far from the box at (0.22, 0.18, 0.04). "
            "The learned policy will attempt a fast pick-and-carry motion over the "
            "large distance; the high sensor noise (0.13 m) causes a late approach "
            "correction that spikes end-effector speed past 0.6 m/s."
        ),
        "parameters": {
            "peg_pose_xyz": [0.42, -0.22, 0.05],
            "box_pose_xyz": [0.22, 0.18, 0.04],
            "obstacle_pose_xyz": [0.32, -0.02, 0.12],
            "obstacle_size": 0.05,
            "peg_length": 0.17,
            "object_mass": 0.30,
            "sensor_noise": 0.13,
            "lighting": "dim",
        },
        "generated_at": "2026-03-21T10:00:05Z",
        "model": "gpt-4o",
    },

    # ── Scenario 7 ──────────────────────────────────────────────────────────
    # Task: PickCube-v1
    # *** INVALID — obstacle physically overlaps the cube ***
    # Both the object and the obstacle are centred at the exact same position.
    # Expected: INVALID ✗  (semantic overlap error)
    {
        "scenario_id": "mock_pickcube_v1_INVALID_overlap",
        "task": "PickCube-v1",
        "target_constraint": "collision_avoidance",
        "rationale": "Place obstacle at the same location as the cube.",
        "parameters": {
            "object_pose_xyz": [0.40, 0.05, 0.02],
            "obstacle_pose_xyz": [0.40, 0.05, 0.02],  # ← same position as cube
            "obstacle_size": 0.08,
            "object_mass": 0.50,
            "sensor_noise": 0.10,
            "lighting": "normal",
        },
        "generated_at": "2026-03-21T10:00:06Z",
        "model": "gpt-4o",
    },

    # ── Scenario 8 ──────────────────────────────────────────────────────────
    # Task: PickCube-v1
    # *** INVALID — sensor_noise exceeds schema bound (> 0.20) ***
    # Expected: INVALID ✗  (bounds violation)
    {
        "scenario_id": "mock_pickcube_v1_INVALID_noise",
        "task": "PickCube-v1",
        "target_constraint": "max_ee_speed",
        "rationale": "Extremely high noise to disorient the robot's position estimates.",
        "parameters": {
            "object_pose_xyz": [0.30, 0.00, 0.02],
            "obstacle_pose_xyz": [0.55, 0.25, 0.10],
            "obstacle_size": 0.05,
            "object_mass": 0.60,
            "sensor_noise": 0.35,          # ← exceeds bound [0.00, 0.20]
            "lighting": "dim",
        },
        "generated_at": "2026-03-21T10:00:07Z",
        "model": "gpt-4o",
    },

    # ── Scenario 9 ──────────────────────────────────────────────────────────
    # Task: StackCube-v1
    # *** INVALID — red and blue cubes placed at same position ***
    # Expected: INVALID ✗  (StackCube semantic overlap)
    {
        "scenario_id": "mock_stackcube_v1_INVALID_cubes",
        "task": "StackCube-v1",
        "target_constraint": "joint_limit_violation",
        "rationale": "Both cubes at same position to stress the init.",
        "parameters": {
            "red_cube_pose_xyz": [0.30, 0.05, 0.02],
            "blue_cube_pose_xyz": [0.30, 0.05, 0.02],   # ← same as red
            "obstacle_pose_xyz": [0.50, 0.10, 0.08],
            "obstacle_size": 0.05,
            "object_mass": 0.40,
            "sensor_noise": 0.08,
            "lighting": "bright",
        },
        "generated_at": "2026-03-21T10:00:08Z",
        "model": "gpt-4o",
    },

    # ── Scenario 10 ─────────────────────────────────────────────────────────
    # Task: PushCube-v1
    # *** INVALID — obstacle z is below the table (negative z) ***
    # Expected: INVALID ✗  (obstacle below table surface)
    {
        "scenario_id": "mock_pushcube_v1_INVALID_z",
        "task": "PushCube-v1",
        "target_constraint": "collision_avoidance",
        "rationale": "Obstacle placed below table to block trajectory.",
        "parameters": {
            "object_pose_xyz": [0.25, 0.0, 0.02],
            "goal_pose_xyz": [0.50, 0.10, 0.02],
            "obstacle_pose_xyz": [0.35, 0.05, -0.05],  # ← z < 0
            "obstacle_size": 0.06,
            "object_mass": 0.30,
            "sensor_noise": 0.05,
            "lighting": "normal",
        },
        "generated_at": "2026-03-21T10:00:09Z",
        "model": "gpt-4o",
    },
]


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

def _print_scenario(s: dict, result, idx: int, total: int):
    tag = "VALID  ✓" if result.valid else "INVALID ✗"
    bar = "─" * 65
    print(f"\n{bar}")
    print(f"  [{idx}/{total}]  {s['scenario_id']}")
    print(f"  Task     : {s['task']}")
    print(f"  Target   : {s.get('target_constraint', 'n/a')}")
    print(f"  Status   : {tag}")
    if result.errors:
        for e in result.errors:
            print(f"  ✗ ERROR  : {e}")
    if result.warnings:
        for w in result.warnings:
            print(f"  ⚠ WARN   : {w}")
    print(f"  Rationale: {s.get('rationale', '')[:120]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  LLM ADVERSARIAL SCENARIO GENERATOR — OFFLINE DEMO")
    print("  Steps 1 (LLM generation) + 2 (Validation)")
    print("=" * 65)
    print(f"  Scenarios to validate : {len(MOCK_SCENARIOS)}")
    print(f"  Tasks covered         : PickCube-v1, StackCube-v1,")
    print(f"                          PushCube-v1, PlaceSphere-v1,")
    print(f"                          PegInsertionSide-v1")

    combined = []
    for i, scenario in enumerate(MOCK_SCENARIOS, start=1):
        result = validate_scenario(scenario)
        _print_scenario(scenario, result, i, len(MOCK_SCENARIOS))
        combined.append(
            {
                "scenario": scenario,
                "validation": result.to_dict(),
            }
        )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    vr_list = [
        {
            "scenario_id": e["scenario"]["scenario_id"],
            "task": e["scenario"]["task"],
            "target_constraint": e["scenario"].get("target_constraint", "?"),
            "validation": e["validation"],
        }
        for e in combined
    ]
    stats = summarise(vr_list)

    print(f"\n{'=' * 65}")
    print(f"  SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total scenarios   : {stats['total']}")
    print(f"  Passed validation : {stats['valid']}  (physically feasible)")
    print(f"  Failed validation : {stats['invalid']}  (infeasible / out-of-bounds)")
    print(f"  Pass rate         : {stats['pass_rate_pct']}%")
    print(f"\n  Scenarios per targeted constraint:")
    for c, count in sorted(stats["scenarios_per_constraint"].items()):
        print(f"    {c:<38}  {count}")

    # ------------------------------------------------------------------
    # Save to results/
    # ------------------------------------------------------------------
    out_path = RESULTS_DIR / f"demo_scenarios_{timestamp}.json"
    with open(out_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
