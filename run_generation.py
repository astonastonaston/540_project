"""
Run Generation — Main Orchestration Script
-------------------------------------------
End-to-end pipeline:

  Step 1  →  LLM generates adversarial scenarios (JSON)
  Step 2  →  Validator checks bounds + semantic feasibility

Results are saved to  LLMScenarios/results/<timestamp>.json

Usage:
    # Generate 3 scenarios per task for 3 tasks:
    python run_generation.py

    # Custom tasks and count:
    python run_generation.py --tasks PickCube-v1 PlaceSphere-v1 --n 5

    # Use a cheaper/faster model:
    python run_generation.py --model gpt-4o-mini --n 5

    # Use a .env file:
    python run_generation.py --env .env
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Resolve imports relative to this file's location
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from tasks_config import MANISKILL_TASKS
from llm_generator import ScenarioGenerator
from validator import validate_scenario, validate_batch, summarise


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
RESULTS_DIR = _HERE / "results"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    tasks: list,
    n_per_task: int = 3,
    model: str = "gpt-4o",
    save: bool = True,
) -> list:
    """
    Run the full generation + validation pipeline.

    Parameters
    ----------
    tasks : list[str]
        Task names from MANISKILL_TASKS.
    n_per_task : int
        Number of adversarial scenarios to generate per task.
    model : str
        OpenAI model identifier.
    save : bool
        If True, write results to RESULTS_DIR.

    Returns
    -------
    list  — combined results list (scenario + validation dict per entry)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gen = ScenarioGenerator(model=model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    combined: list = []

    for task_name in tasks:
        if task_name not in MANISKILL_TASKS:
            print(f"[SKIP] Unknown task: '{task_name}'")
            continue

        print(f"\n{'=' * 65}")
        print(f"  TASK: {task_name}")
        print(f"{'=' * 65}")

        scenarios = gen.generate_batch(task_name, n=n_per_task)

        for scenario in scenarios:
            result = validate_scenario(scenario)

            print(f"\n  Scenario : {scenario['scenario_id']}")
            print(f"  Target   : {scenario.get('target_constraint')}")
            print(f"  Rationale: {scenario.get('rationale', '')}")
            print(result)

            combined.append(
                {
                    "scenario": scenario,
                    "validation": result.to_dict(),
                }
            )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if save and combined:
        out_path = RESULTS_DIR / f"generated_scenarios_{timestamp}.json"
        with open(out_path, "w") as fh:
            json.dump(combined, fh, indent=2)
        print(f"\n  Results saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    vr_list = [
        {
            "scenario_id": e["scenario"].get("scenario_id", "?"),
            "task": e["scenario"].get("task", "?"),
            "target_constraint": e["scenario"].get("target_constraint", "?"),
            "validation": e["validation"],
        }
        for e in combined
    ]
    stats = summarise(vr_list)

    print(f"\n{'=' * 65}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total scenarios   : {stats['total']}")
    print(f"  Passed validation : {stats['valid']}")
    print(f"  Failed validation : {stats['invalid']}")
    print(f"  Pass rate         : {stats['pass_rate_pct']}%")
    print(f"  Scenarios per constraint:")
    for c, count in stats["scenarios_per_constraint"].items():
        print(f"    {c:35s}  {count}")

    return combined


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    # 3 tasks × 2 each = 6 scenarios → clean 2×3 workspace layout grid
    default_tasks = ["PickCube-v1", "StackCube-v1", "PushCube-v1"]

    parser = argparse.ArgumentParser(
        description="LLM Adversarial Scenario Generator + Validator (Steps 1 & 2)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=default_tasks,
        metavar="TASK",
        help=(
            "ManiSkill task names to generate scenarios for. "
            f"Default: {default_tasks}. "
            f"All supported: {list(MANISKILL_TASKS.keys())}"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        metavar="N",
        help="Number of adversarial scenarios per task (default: 2)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        metavar="MODEL",
        help="OpenAI model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write results to disk",
    )
    parser.add_argument(
        "--env",
        default=None,
        metavar="FILE",
        help="Path to a .env file containing OPENAI_API_KEY",
    )
    return parser.parse_args()


def _load_dotenv(env_file: str):
    """Minimal .env loader — avoids requiring python-dotenv."""
    path = Path(env_file)
    if not path.is_file():
        print(f"WARNING: .env file not found: {env_file}")
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


if __name__ == "__main__":
    args = _parse_args()

    if args.env:
        _load_dotenv(args.env)

    run_pipeline(
        tasks=args.tasks,
        n_per_task=args.n,
        model=args.model,
        save=not args.no_save,
    )

# export OPENAI_API_KEY=<your_openai_api_key>

