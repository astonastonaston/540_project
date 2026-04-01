"""
Scenario Validator
------------------
Validates LLM-generated adversarial scenarios against:
  1. Schema completeness  — all required top-level fields present
  2. Parameter completeness  — every schema parameter provided
  3. Bounds checking  — numeric values within allowed ranges, categoricals in allowed sets
  4. Semantic / physics feasibility  — task-specific checks (overlap, trivial configs, etc.)

Usage:
    from validator import validate_scenario, ValidationResult
    result = validate_scenario(scenario_dict)
    print(result)          # human-readable summary
    result.valid           # True / False
    result.errors          # list of error strings
    result.warnings        # list of warning strings
"""

import math
from dataclasses import dataclass, field

from tasks_config import MANISKILL_TASKS, SAFETY_CONSTRAINTS

# ---------------------------------------------------------------------------
# Minimum physical separation between the primary object centre and the
# obstacle centre at scenario initialisation (metres).  Prevents embedded /
# overlapping initial states that are physically impossible.
# ---------------------------------------------------------------------------
_MIN_OBJ_OBSTACLE_SEP = 0.06   # object radius ~3 cm + half small obstacle


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def __str__(self):
        status = "VALID  ✓" if self.valid else "INVALID ✗"
        lines = [f"  Status : {status}"]
        for e in self.errors:
            lines.append(f"  ERROR  : {e}")
        for w in self.warnings:
            lines.append(f"  WARN   : {w}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _dist3(a: list, b: list) -> float:
    """Euclidean distance between two 3-D points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _check_float_in_bounds(value, lo: float, hi: float, name: str, errors: list):
    if not isinstance(value, (int, float)):
        errors.append(f"'{name}' must be numeric, got {type(value).__name__}")
        return
    if not (lo <= value <= hi):
        errors.append(f"'{name}' = {value:.4f} is out of bounds [{lo}, {hi}]")


def _check_list_in_bounds(value, bounds: list, name: str, length: int, errors: list):
    if not isinstance(value, list):
        errors.append(f"'{name}' must be a list, got {type(value).__name__}")
        return
    if len(value) != length:
        errors.append(
            f"'{name}' must have exactly {length} elements, got {len(value)}"
        )
        return
    axes = ["x", "y", "z"] + [str(i) for i in range(3, length)]
    for i, (v, (lo, hi)) in enumerate(zip(value, bounds)):
        _check_float_in_bounds(v, lo, hi, f"{name}[{axes[i]}]", errors)


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_scenario(scenario: dict) -> ValidationResult:
    """
    Validate an adversarial scenario dictionary.

    Returns a ValidationResult with .valid, .errors, and .warnings populated.
    """
    errors: list = []
    warnings: list = []

    # ------------------------------------------------------------------
    # 1.  Top-level required fields
    # ------------------------------------------------------------------
    required_top = {"scenario_id", "task", "target_constraint", "rationale", "parameters"}
    for f in required_top:
        if f not in scenario:
            errors.append(f"Missing required top-level field: '{f}'")

    # Early exit — cannot proceed without task or parameters
    if "task" not in scenario or "parameters" not in scenario:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    task_name = scenario["task"]
    if task_name not in MANISKILL_TASKS:
        errors.append(
            f"Unknown task: '{task_name}'. "
            f"Supported tasks: {list(MANISKILL_TASKS.keys())}"
        )
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    task = MANISKILL_TASKS[task_name]
    schema = task["parameter_schema"]
    params = scenario["parameters"]

    # ------------------------------------------------------------------
    # 2.  target_constraint validity
    # ------------------------------------------------------------------
    target = scenario.get("target_constraint", "")
    if target and target not in SAFETY_CONSTRAINTS:
        errors.append(
            f"Unknown target_constraint: '{target}'. "
            f"Valid IDs: {list(SAFETY_CONSTRAINTS.keys())}"
        )
    if target and target not in task["safety_constraints"]:
        warnings.append(
            f"Constraint '{target}' is not typically active for task '{task_name}'. "
            "Scenario may not be meaningful."
        )

    # ------------------------------------------------------------------
    # 3.  Parameter bounds checking
    # ------------------------------------------------------------------
    for param_name, spec in schema.items():
        if param_name not in params:
            errors.append(f"Missing parameter: '{param_name}'")
            continue

        value = params[param_name]

        if spec["type"] == "categorical":
            if value not in spec["options"]:
                errors.append(
                    f"'{param_name}' = '{value}' is not in allowed options {spec['options']}"
                )

        elif spec["type"].startswith("list"):
            _check_list_in_bounds(
                value, spec["bounds"], param_name, spec["length"], errors
            )

        else:  # scalar float / int
            lo, hi = spec["bounds"]
            _check_float_in_bounds(value, lo, hi, param_name, errors)

    # Extra parameters that are not in schema (informational warning only)
    known_params = set(schema.keys())
    for p in params:
        if p not in known_params:
            warnings.append(
                f"Extra parameter '{p}' is not in the task schema and will be ignored."
            )

    # ------------------------------------------------------------------
    # 4.  Semantic / physics-feasibility checks
    # ------------------------------------------------------------------
    _semantic_checks(task_name, scenario, params, schema, errors, warnings)

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Semantic checks
# ---------------------------------------------------------------------------

def _semantic_checks(task_name, scenario, params, schema, errors, warnings):
    """Task-specific physics- and logic-plausibility checks."""

    # ---- Identify the primary object position --------------------------------
    primary_key = None
    for candidate in [
        "object_pose_xyz",
        "sphere_pose_xyz",
        "red_cube_pose_xyz",
        "peg_pose_xyz",
    ]:
        if candidate in params and isinstance(params[candidate], list):
            primary_key = candidate
            break

    # ---- Obstacle overlaps primary object ------------------------------------
    if (
        primary_key
        and "obstacle_pose_xyz" in params
        and isinstance(params["obstacle_pose_xyz"], list)
        and "obstacle_size" in params
        and isinstance(params["obstacle_size"], (int, float))
    ):
        obj_pos = params[primary_key]
        obs_pos = params["obstacle_pose_xyz"]
        obs_size = params["obstacle_size"]
        dist = _dist3(obj_pos, obs_pos)
        # obstacle halfwidth + object half-size (~2 cm) + safety margin (1 cm)
        min_sep = obs_size / 2.0 + 0.03
        if dist < min_sep:
            errors.append(
                f"Obstacle physically overlaps '{primary_key}': "
                f"centre distance {dist:.3f} m < minimum separation {min_sep:.3f} m. "
                "Scenario is physically infeasible at initialisation."
            )
        elif dist < min_sep + 0.02:
            warnings.append(
                f"Obstacle is extremely close to '{primary_key}' "
                f"(distance {dist:.3f} m). "
                "Initial state may cause immediate collision."
            )

    # ---- StackCube: two cubes must not overlap at start ----------------------
    if task_name == "StackCube-v1":
        r = params.get("red_cube_pose_xyz")
        b = params.get("blue_cube_pose_xyz")
        if (
            isinstance(r, list) and len(r) == 3
            and isinstance(b, list) and len(b) == 3
        ):
            dist = _dist3(r, b)
            if dist < 0.05:
                errors.append(
                    f"Red and blue cubes overlap at initialisation "
                    f"(centre distance {dist:.3f} m < 0.05 m). "
                    "Scenario is physically infeasible."
                )

    # ---- PushCube: trivial scenario when object ≈ goal -----------------------
    if task_name == "PushCube-v1":
        obj = params.get("object_pose_xyz")
        goal = params.get("goal_pose_xyz")
        if (
            isinstance(obj, list) and len(obj) == 3
            and isinstance(goal, list) and len(goal) == 3
        ):
            dist = _dist3(obj, goal)
            if dist < 0.05:
                warnings.append(
                    f"Object and goal are nearly coincident (distance {dist:.3f} m). "
                    "Task is trivially solved — not an effective adversarial scenario."
                )

    # ---- Obstacle z below table surface --------------------------------------
    obs = params.get("obstacle_pose_xyz")
    if isinstance(obs, list) and len(obs) >= 3:
        if obs[2] < -0.01:
            errors.append(
                f"Obstacle z = {obs[2]:.3f} m is below the table surface (z = 0). "
                "Scenario is physically infeasible."
            )

    # ---- PlaceSphere: bowl must be reachable from sphere  --------------------
    if task_name == "PlaceSphere-v1":
        sphere = params.get("sphere_pose_xyz")
        bowl = params.get("bowl_pose_xyz")
        if (
            isinstance(sphere, list) and len(sphere) == 3
            and isinstance(bowl, list) and len(bowl) == 3
        ):
            dist = _dist3(sphere, bowl)
            if dist > 0.55:
                warnings.append(
                    f"Sphere and bowl are {dist:.3f} m apart. "
                    "This may exceed the robot's workspace reach."
                )

    # ---- High noise + dim lighting warning -----------------------------------
    noise = params.get("sensor_noise", 0)
    lighting = params.get("lighting", "normal")
    if isinstance(noise, (int, float)) and noise > 0.15 and lighting == "dim":
        warnings.append(
            "High sensor noise (>0.15 m) combined with dim lighting may render "
            "the scenario degenerate — perception is essentially random."
        )

    # ---- rationale should not be empty ---------------------------------------
    rationale = scenario.get("rationale", "")
    if not rationale or len(rationale.strip()) < 10:
        warnings.append(
            "Field 'rationale' is empty or too short. "
            "A clear adversarial strategy explanation is expected."
        )


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def validate_batch(scenarios: list) -> list:
    """
    Validate a list of scenarios and return a list of dicts:
        {"scenario_id": ..., "task": ..., "validation": ValidationResult.to_dict()}
    """
    results = []
    for s in scenarios:
        r = validate_scenario(s)
        results.append(
            {
                "scenario_id": s.get("scenario_id", "unknown"),
                "task": s.get("task", "unknown"),
                "target_constraint": s.get("target_constraint", "unknown"),
                "validation": r.to_dict(),
            }
        )
    return results


def summarise(validation_results: list) -> dict:
    """
    Print and return a summary dict from a list of validate_batch output items.
    """
    total = len(validation_results)
    valid = sum(1 for r in validation_results if r["validation"]["valid"])
    invalid = total - valid
    pass_rate = (valid / total * 100) if total else 0.0

    summary = {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "pass_rate_pct": round(pass_rate, 1),
    }

    # Constraint-level breakdown
    constraint_counts: dict = {}
    for r in validation_results:
        c = r.get("target_constraint", "unknown")
        constraint_counts[c] = constraint_counts.get(c, 0) + 1

    summary["scenarios_per_constraint"] = constraint_counts
    return summary
