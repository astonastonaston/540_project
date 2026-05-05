"""
Random Baseline Scenario Generator
------------------------------------
Uniformly samples scenarios from the task parameter schema bounds.
Used as a null model against which LLM adversarial generation is evaluated.

Usage:
    from random_baseline import generate_random_batch, generate_multi_task_random_batch
    scenarios = generate_random_batch("PickCube-v1", n=10, seed=42)
"""

import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from tasks_config import MANISKILL_TASKS


def generate_random_scenario(task_name: str) -> dict:
    """
    Sample one scenario uniformly at random from the task's parameter schema.
    No adversarial intent — purely random placement within schema bounds.
    """
    if task_name not in MANISKILL_TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. Supported: {list(MANISKILL_TASKS.keys())}"
        )

    task = MANISKILL_TASKS[task_name]
    schema = task["parameter_schema"]
    constraints = task["safety_constraints"]

    params: dict = {}
    for key, spec in schema.items():
        if spec["type"] == "categorical":
            params[key] = random.choice(spec["options"])
        elif spec["type"].startswith("list"):
            params[key] = [
                round(random.uniform(lo, hi), 4)
                for lo, hi in spec["bounds"]
            ]
        else:
            lo, hi = spec["bounds"]
            params[key] = round(random.uniform(lo, hi), 4)

    return {
        "scenario_id": (
            f"rnd_{task_name.replace('-', '_').lower()}_{uuid.uuid4().hex[:8]}"
        ),
        "task": task_name,
        "target_constraint": random.choice(constraints),
        "rationale": "Uniformly random baseline scenario (no adversarial strategy).",
        "parameters": params,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "random_baseline",
    }


def generate_random_batch(
    task_name: str,
    n: int = 10,
    seed: Optional[int] = None,
) -> list:
    """Generate *n* random scenarios for a single task."""
    if seed is not None:
        random.seed(seed)
    return [generate_random_scenario(task_name) for _ in range(n)]


def generate_multi_task_random_batch(
    tasks: list,
    n_per_task: int = 10,
    seed: Optional[int] = None,
) -> list:
    """Generate *n_per_task* random scenarios for each task in *tasks*."""
    if seed is not None:
        random.seed(seed)
    all_scenarios: list = []
    for task_name in tasks:
        all_scenarios.extend(generate_random_batch(task_name, n=n_per_task))
    return all_scenarios
