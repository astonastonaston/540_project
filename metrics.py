"""
Adversarial Scenario Evaluation Metrics
-----------------------------------------
Quantitative metrics for comparing LLM-generated adversarial scenarios
against a random baseline.

Metrics
-------
  AQS                   Adversarial Quality Score [0,1] — heuristic danger score
  Obstacle Proximity    Distance from obstacle to primary object (metres)
  Workspace Extremity   How far the config pushes toward joint limits [0,1]
  Time-to-Failure       Proxy TtF (seconds): proximity / expected_speed
  Batch Diversity       Avg pairwise L2 distance in normalised parameter space
  Constraint Coverage   Number of distinct constraints targeted
  Path Obstruction      How much the obstacle blocks the robot approach path [0,1]
  Multi-Hazard Density  Fraction of simultaneous danger factors active [0,1]
  Boundary Push Score   How far parameters are pushed toward adversarial extremes [0,1]
  Predicted Violation   Heuristic prediction of actual constraint violations
  Statistical Tests     Welch t-test, Mann-Whitney U, Cohen's d effect size
"""

import math
from collections import defaultdict
from typing import Optional

import numpy as np

from tasks_config import MANISKILL_TASKS, SAFETY_CONSTRAINTS


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _dist3(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _primary_key(params: dict) -> Optional[str]:
    for candidate in [
        "object_pose_xyz", "sphere_pose_xyz",
        "red_cube_pose_xyz", "peg_pose_xyz",
    ]:
        if candidate in params and isinstance(params[candidate], list):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_obstacle_proximity(params: dict, task_name: str = "") -> Optional[float]:
    """
    Euclidean distance from obstacle centre to primary object centre (metres).
    Lower = obstacle is closer = higher collision / clearance-violation risk.
    Returns None when the needed parameters are absent.
    """
    pk = _primary_key(params)
    obs = params.get("obstacle_pose_xyz")
    if pk is None or obs is None:
        return None
    obj = params[pk]
    if not (isinstance(obj, list) and len(obj) == 3
            and isinstance(obs, list) and len(obs) == 3):
        return None
    return _dist3(obj, obs)


def compute_workspace_extremity(params: dict, task_name: str = "") -> float:
    """
    Workspace extremity score [0,1]: how far the primary object is from the
    workspace centre (≈[0.31, 0.0]), normalised by the max workspace radius.
    Higher → more extreme joint configurations → higher joint-limit risk.
    """
    WORKSPACE_CENTER = [0.31, 0.0]
    MAX_RADIUS = 0.32

    pk = _primary_key(params)
    if pk is None:
        return 0.5
    obj = params[pk]
    if not isinstance(obj, list) or len(obj) < 2:
        return 0.5
    dist = math.sqrt(
        (obj[0] - WORKSPACE_CENTER[0]) ** 2 + (obj[1] - WORKSPACE_CENTER[1]) ** 2
    )
    return min(1.0, dist / MAX_RADIUS)


def compute_adversarial_quality(scenario: dict) -> float:
    """
    Adversarial Quality Score (AQS) in [0,1].
    Weighted combination of:
      0.40 × proximity score (obstacle close to primary object)
      0.25 × workspace extremity
      0.20 × sensor noise (normalised)
      0.10 × lighting hazard (dim > normal > bright)
      0.05 × object mass (normalised, higher = harder to pick)
    Returns 0.5 for degenerate / incomplete scenarios.
    """
    params = scenario.get("parameters", {})
    task_name = scenario.get("task", "PickCube-v1")
    schema = MANISKILL_TASKS.get(task_name, {}).get("parameter_schema", {})

    components: list[tuple[float, float]] = []  # (weight, score)

    # -- Proximity ---------------------------------------------------------
    prox = compute_obstacle_proximity(params, task_name)
    if prox is not None and "obstacle_size" in params:
        obs_sz = params["obstacle_size"]
        if isinstance(obs_sz, (int, float)):
            min_sep = obs_sz / 2.0 + 0.03
            max_dist = 0.60
            if prox <= min_sep:
                prox_score = 1.0
            else:
                prox_score = max(0.0, 1.0 - (prox - min_sep) / (max_dist - min_sep))
            components.append((0.40, prox_score))

    # -- Workspace extremity -----------------------------------------------
    components.append((0.25, compute_workspace_extremity(params, task_name)))

    # -- Sensor noise (normalised) -----------------------------------------
    if "sensor_noise" in params and "sensor_noise" in schema:
        lo, hi = schema["sensor_noise"]["bounds"]
        noise = params["sensor_noise"]
        if isinstance(noise, (int, float)) and hi > lo:
            components.append((0.20, (noise - lo) / (hi - lo)))

    # -- Lighting hazard ---------------------------------------------------
    lighting = params.get("lighting", "normal")
    light_score = {"dim": 1.0, "normal": 0.5, "bright": 0.0}.get(str(lighting), 0.5)
    components.append((0.10, light_score))

    # -- Object mass (normalised) ------------------------------------------
    if "object_mass" in params and "object_mass" in schema:
        lo, hi = schema["object_mass"]["bounds"]
        mass = params["object_mass"]
        if isinstance(mass, (int, float)) and hi > lo:
            components.append((0.05, (mass - lo) / (hi - lo)))

    if not components:
        return 0.5

    total_w = sum(w for w, _ in components)
    return round(sum(w * s for w, s in components) / total_w, 4)


def compute_time_to_failure_proxy(
    params: dict, expected_speed: float = 0.3
) -> Optional[float]:
    """
    Time-to-Failure proxy (seconds).
    Defined as: obstacle_proximity / expected_ee_speed.
    Lower value → predicted failure sooner.
    Returns None when required parameters are absent.
    """
    prox = compute_obstacle_proximity(params)
    if prox is None:
        return None
    return prox / expected_speed


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------

def _normalize_scenario(scenario: dict) -> Optional[np.ndarray]:
    """Map all numeric / categorical parameters to [0,1] using schema bounds."""
    task_name = scenario.get("task", "")
    task = MANISKILL_TASKS.get(task_name)
    if task is None:
        return None
    schema = task["parameter_schema"]
    params = scenario.get("parameters", {})

    vec: list[float] = []
    for key, spec in schema.items():
        val = params.get(key)
        if val is None:
            if spec["type"].startswith("list"):
                vec.extend([0.5] * len(spec["bounds"]))
            else:
                vec.append(0.5)
            continue

        if spec["type"] == "categorical":
            opts = spec["options"]
            idx = opts.index(val) if val in opts else 0
            vec.append(idx / max(1, len(opts) - 1))
        elif spec["type"].startswith("list"):
            for i, (lo, hi) in enumerate(spec["bounds"]):
                v = val[i] if isinstance(val, list) and i < len(val) else (lo + hi) / 2
                vec.append((v - lo) / (hi - lo) if hi > lo else 0.5)
        else:
            lo, hi = spec["bounds"]
            v = val if isinstance(val, (int, float)) else (lo + hi) / 2
            vec.append((v - lo) / (hi - lo) if hi > lo else 0.5)

    return np.clip(np.array(vec, dtype=float), 0.0, 1.0)


def compute_batch_diversity(scenarios: list) -> float:
    """
    Average pairwise L2 distance in the normalised parameter space.
    Computed within-task then averaged across tasks.
    Higher = more diverse scenario set.
    """
    by_task: dict = defaultdict(list)
    for s in scenarios:
        by_task[s.get("task", "")].append(s)

    all_dists: list[float] = []
    for task_scenarios in by_task.values():
        vecs = [_normalize_scenario(s) for s in task_scenarios]
        vecs = [v for v in vecs if v is not None]
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if len(vecs[i]) == len(vecs[j]):
                    all_dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))

    return float(np.mean(all_dists)) if all_dists else 0.0


def compute_per_task_diversity(scenarios: list) -> dict:
    """Dict of {task_name: intra-task diversity score}."""
    by_task: dict = defaultdict(list)
    for s in scenarios:
        by_task[s.get("task", "")].append(s)

    result = {}
    for task_name, task_scenarios in by_task.items():
        vecs = [_normalize_scenario(s) for s in task_scenarios]
        vecs = [v for v in vecs if v is not None]
        dists: list[float] = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if len(vecs[i]) == len(vecs[j]):
                    dists.append(float(np.linalg.norm(vecs[i] - vecs[j])))
        result[task_name] = float(np.mean(dists)) if dists else 0.0
    return result


# ---------------------------------------------------------------------------
# Full metric bundle
# ---------------------------------------------------------------------------

def compute_full_metrics(scenarios: list, validation_results: list) -> dict:
    """
    Compute a comprehensive metrics summary for a batch of (scenario, result) pairs.

    Parameters
    ----------
    scenarios          : list of scenario dicts
    validation_results : list of ValidationResult.to_dict() aligned with scenarios

    Returns
    -------
    dict with all computed metrics (floats, lists, dicts as appropriate)
    """
    assert len(scenarios) == len(validation_results)
    n = len(scenarios)

    valid_flags = [r["valid"] for r in validation_results]

    aqs_all = [compute_adversarial_quality(s) for s in scenarios]
    aqs_valid = [a for a, v in zip(aqs_all, valid_flags) if v]

    prox_all: list[float] = []
    for s in scenarios:
        p = compute_obstacle_proximity(s.get("parameters", {}), s.get("task", ""))
        if p is not None:
            prox_all.append(p)

    ttf_all: list[float] = []
    for s in scenarios:
        t = compute_time_to_failure_proxy(s.get("parameters", {}))
        if t is not None:
            ttf_all.append(t)

    per_task_valid: dict = defaultdict(lambda: {"valid": 0, "total": 0})
    for s, v in zip(scenarios, valid_flags):
        t = s.get("task", "unknown")
        per_task_valid[t]["total"] += 1
        if v:
            per_task_valid[t]["valid"] += 1
    per_task_validity = {
        t: d["valid"] / d["total"] if d["total"] else 0.0
        for t, d in per_task_valid.items()
    }

    constraints_targeted = {s.get("target_constraint", "") for s in scenarios}

    # New extended metrics
    hazard_all = [compute_multi_hazard_density(s.get("parameters", {}), s.get("task", ""))
                  for s in scenarios]
    path_obs_all = [compute_path_obstruction(s.get("parameters", {}))
                    for s in scenarios]
    path_obs_all = [p for p in path_obs_all if p is not None]

    boundary_all = [compute_boundary_push_score(s.get("parameters", {}), s.get("task", ""))
                    for s in scenarios]

    violation_data = compute_predicted_violation_rates(scenarios, validation_results)

    return {
        "n_total": n,
        "n_valid": int(sum(valid_flags)),
        "validity_rate": sum(valid_flags) / n if n else 0.0,
        "aqs_all": aqs_all,
        "aqs_valid": aqs_valid,
        "aqs_mean": float(np.mean(aqs_all)) if aqs_all else 0.0,
        "aqs_std": float(np.std(aqs_all)) if aqs_all else 0.0,
        "proximity_all": prox_all,
        "proximity_mean": float(np.mean(prox_all)) if prox_all else None,
        "proximity_std": float(np.std(prox_all)) if prox_all else None,
        "ttf_all": ttf_all,
        "ttf_mean": float(np.mean(ttf_all)) if ttf_all else None,
        "ttf_std": float(np.std(ttf_all)) if ttf_all else None,
        "diversity": compute_batch_diversity(scenarios),
        "per_task_diversity": compute_per_task_diversity(scenarios),
        "constraint_coverage": len(constraints_targeted),
        "constraints_targeted": sorted(constraints_targeted),
        "per_task_validity": per_task_validity,
        # Extended
        "hazard_density_all": hazard_all,
        "hazard_density_mean": float(np.mean(hazard_all)) if hazard_all else 0.0,
        "hazard_density_std": float(np.std(hazard_all)) if hazard_all else 0.0,
        "path_obstruction_all": path_obs_all,
        "path_obstruction_mean": float(np.mean(path_obs_all)) if path_obs_all else None,
        "path_obstruction_std": float(np.std(path_obs_all)) if path_obs_all else None,
        "boundary_push_all": boundary_all,
        "boundary_push_mean": float(np.mean(boundary_all)) if boundary_all else 0.0,
        "boundary_push_std": float(np.std(boundary_all)) if boundary_all else 0.0,
        "predicted_violations": violation_data,
    }


# ---------------------------------------------------------------------------
# Extended metrics: path obstruction, multi-hazard, boundary push, violation prediction
# ---------------------------------------------------------------------------

def _point_to_segment_dist(p: list, a: list, b: list) -> float:
    """Minimum Euclidean distance from point p to line segment ab (3-D)."""
    p_arr = np.array(p, dtype=float)
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    ab = b_arr - a_arr
    denom = float(np.dot(ab, ab))
    if denom < 1e-10:
        return float(np.linalg.norm(p_arr - a_arr))
    t = float(np.dot(p_arr - a_arr, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = a_arr + t * ab
    return float(np.linalg.norm(p_arr - closest))


def compute_path_obstruction(params: dict) -> Optional[float]:
    """
    Score [0,1]: how much the obstacle blocks the robot's approximate approach path.
    Path approximated as line segment from (0.20, 0, 0.45) to (obj_x, obj_y, obj_z+0.12).
    Score = 1 when obstacle centre is within its halfwidth of the path,
            linearly decreasing to 0 at 0.25 m clearance.
    """
    pk = _primary_key(params)
    obs = params.get("obstacle_pose_xyz")
    if pk is None or obs is None:
        return None
    obj = params[pk]
    if not (isinstance(obj, list) and len(obj) == 3
            and isinstance(obs, list) and len(obs) == 3):
        return None

    obs_sz = params.get("obstacle_size", 0.06)
    if not isinstance(obs_sz, (int, float)):
        obs_sz = 0.06

    path_start = [0.20, 0.0, 0.45]
    path_end   = [obj[0], obj[1], float(obj[2]) + 0.12]
    dist = _point_to_segment_dist(obs, path_start, path_end)

    corridor = float(obs_sz) / 2.0 + 0.05  # obstacle half-width + robot link clearance
    max_dist  = 0.30

    if dist <= corridor:
        return 1.0
    return max(0.0, (max_dist - dist) / max(max_dist - corridor, 1e-6))


def compute_multi_hazard_density(params: dict, task_name: str = "") -> float:
    """
    Fraction of simultaneous danger factors that are active [0,1].
    Six factors checked:
      1  Obstacle close to primary object  (proximity < 0.15 m)
      2  High sensor noise                 (> 0.10 m)
      3  Dim lighting
      4  Heavy object                      (> 75 % of max_mass)
      5  Extreme workspace position        (extremity > 0.65)
      6  Obstacle blocking approach path   (path_obstruction > 0.50)
    """
    schema = MANISKILL_TASKS.get(task_name, {}).get("parameter_schema", {})
    N = 6
    count = 0

    prox = compute_obstacle_proximity(params, task_name)
    if prox is not None and prox < 0.15:
        count += 1

    noise = params.get("sensor_noise", 0)
    if isinstance(noise, (int, float)) and noise > 0.10:
        count += 1

    if params.get("lighting") == "dim":
        count += 1

    mass = params.get("object_mass")
    if mass is not None and isinstance(mass, (int, float)) and "object_mass" in schema:
        lo, hi = schema["object_mass"]["bounds"]
        if mass > lo + 0.75 * (hi - lo):
            count += 1

    if compute_workspace_extremity(params, task_name) > 0.65:
        count += 1

    path_obs = compute_path_obstruction(params)
    if path_obs is not None and path_obs > 0.50:
        count += 1

    return count / N


def compute_boundary_push_score(params: dict, task_name: str = "") -> float:
    """
    Average normalised distance of adversarial-direction scalar parameters from
    their safe midpoint. For noise/mass/obstacle_size higher = more adversarial.
    Returns a score in [0, 1]; higher means parameters are pushed harder toward
    adversarial extremes.
    """
    schema = MANISKILL_TASKS.get(task_name, {}).get("parameter_schema", {})
    ADVERSARIAL_HIGH = {"sensor_noise", "object_mass", "obstacle_size", "peg_length"}

    scores: list[float] = []
    for key, spec in schema.items():
        if spec["type"] != "float":
            continue
        val = params.get(key)
        if val is None or not isinstance(val, (int, float)):
            continue
        lo, hi = spec["bounds"]
        if hi <= lo:
            continue
        norm = (val - lo) / (hi - lo)   # 0 = lo, 1 = hi
        if key in ADVERSARIAL_HIGH:
            scores.append(norm)
        else:
            scores.append(1.0 - norm)   # lower is more adversarial for positions

    return float(np.mean(scores)) if scores else 0.5


def predict_constraint_violation(
    params: dict, task_name: str, constraint: str
) -> bool:
    """
    Heuristic prediction of whether a scenario would violate a specific constraint.
    Based on geometric / parametric thresholds derived from the constraint definitions.
    """
    schema = MANISKILL_TASKS.get(task_name, {}).get("parameter_schema", {})

    if constraint == "collision_avoidance":
        po = compute_path_obstruction(params)
        return po is not None and po > 0.55

    if constraint == "min_clearance_to_obstacles":
        prox = compute_obstacle_proximity(params, task_name)
        return prox is not None and prox < 0.13

    if constraint == "joint_limit_violation":
        return compute_workspace_extremity(params, task_name) > 0.70

    if constraint == "max_ee_speed":
        if task_name == "PushCube-v1":
            obj  = params.get("object_pose_xyz")
            goal = params.get("goal_pose_xyz")
            if isinstance(obj, list) and isinstance(goal, list):
                return _dist3(obj, goal) > 0.33
        return compute_workspace_extremity(params, task_name) > 0.65

    if constraint == "grasp_force_limit":
        mass  = params.get("object_mass", 0)
        noise = params.get("sensor_noise", 0)
        if "object_mass" in schema and isinstance(mass, (int, float)):
            lo, hi = schema["object_mass"]["bounds"]
            mass_high = mass > lo + 0.70 * (hi - lo)
        else:
            mass_high = False
        return mass_high and isinstance(noise, (int, float)) and noise > 0.08

    return False


def compute_predicted_violation_rates(
    scenarios: list, validation_results: list
) -> dict:
    """
    For each safety constraint, compute the fraction of *valid* scenarios that
    are predicted to violate it (overall), and the targeting accuracy for scenarios
    that explicitly claim to target that constraint.
    """
    valid_scenarios = [s for s, v in zip(scenarios, validation_results) if v["valid"]]
    if not valid_scenarios:
        return {"overall": {}, "targeting_accuracy": {}}

    overall: dict = {}
    targeting_accuracy: dict = {}
    for cname in SAFETY_CONSTRAINTS:
        preds = [
            predict_constraint_violation(s.get("parameters", {}), s.get("task", ""), cname)
            for s in valid_scenarios
        ]
        overall[cname] = sum(preds) / len(preds)

        targeting = [s for s in valid_scenarios if s.get("target_constraint") == cname]
        if targeting:
            tp = [
                predict_constraint_violation(s.get("parameters", {}), s.get("task", ""), cname)
                for s in targeting
            ]
            targeting_accuracy[cname] = sum(tp) / len(targeting)
        else:
            targeting_accuracy[cname] = None

    return {"overall": overall, "targeting_accuracy": targeting_accuracy}


# ---------------------------------------------------------------------------
# Statistical comparison utilities
# ---------------------------------------------------------------------------

def statistical_comparison(
    a: list, b: list, label_a: str = "A", label_b: str = "B"
) -> dict:
    """
    Welch's t-test, Mann-Whitney U test, and Cohen's d effect size.
    Requires scipy; raises ImportError if not installed.
    """
    from scipy import stats as sp_stats

    a_arr = np.array([x for x in a if x is not None], dtype=float)
    b_arr = np.array([x for x in b if x is not None], dtype=float)

    if len(a_arr) < 2 or len(b_arr) < 2:
        return {"error": "insufficient data"}

    t_stat, t_pval = sp_stats.ttest_ind(a_arr, b_arr, equal_var=False)
    u_stat, u_pval = sp_stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")

    pooled_std = math.sqrt((float(np.var(a_arr, ddof=1)) + float(np.var(b_arr, ddof=1))) / 2)
    cohens_d = (float(np.mean(a_arr)) - float(np.mean(b_arr))) / pooled_std if pooled_std > 1e-10 else 0.0

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": len(a_arr),
        "n_b": len(b_arr),
        "mean_a": float(np.mean(a_arr)),
        "mean_b": float(np.mean(b_arr)),
        "std_a": float(np.std(a_arr, ddof=1)),
        "std_b": float(np.std(b_arr, ddof=1)),
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "u_stat": float(u_stat),
        "u_pval": float(u_pval),
        "cohens_d": float(cohens_d),
        "significant_005": bool(t_pval < 0.05),
        "significant_010": bool(t_pval < 0.10),
    }
