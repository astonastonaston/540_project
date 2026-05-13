"""
Experiment Logger
=================
Records structured outputs for every episode:
scenario, violations, root-cause attributions, and summary metrics.
Supports JSON export for reproducibility and downstream analysis.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class ExperimentLogger:
    """Structured logger for red-teaming experiments."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.episodes: List[dict] = []
        self.start_time = datetime.now().isoformat()

    def log_episode(
        self,
        episode_num: int,
        scenario: dict,
        violations: List[dict],
        attributions: List[dict],
        success: bool,
        steps: int,
        metadata: Optional[dict] = None,
    ):
        """Log results from a single episode."""
        entry = {
            "episode": episode_num,
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario,
            "success": success,
            "steps": steps,
            "num_violations": len(violations),
            "violations": violations,
            "attributions": attributions,
            "metadata": metadata or {},
        }
        self.episodes.append(entry)

    def get_violation_history(self, last_n: int = 5) -> List[dict]:
        """
        Get condensed violation history for LLM context.
        Returns summaries of last N episodes.
        """
        history = []
        for ep in self.episodes[-last_n:]:
            summary = {
                "episode": ep["episode"],
                "had_violations": ep["num_violations"] > 0,
                "violation_types": list(set(v["type"] for v in ep["violations"])),
                "strategy": ep["scenario"].get("reasoning", "unknown"),
                "success": ep["success"],
            }
            history.append(summary)
        return history

    def get_aggregate_stats(self) -> dict:
        """Compute aggregate statistics across all episodes."""
        if not self.episodes:
            return {"episodes": 0}

        total = len(self.episodes)
        with_violations = sum(1 for e in self.episodes if e["num_violations"] > 0)

        # Violation rate
        violation_rate = with_violations / total

        # Violation type distribution
        all_types = []
        for ep in self.episodes:
            all_types.extend(v["type"] for v in ep["violations"])
        type_counts = {t: all_types.count(t) for t in set(all_types)}

        # Time to first violation (average steps)
        first_steps = []
        for ep in self.episodes:
            if ep["violations"]:
                first_steps.append(ep["violations"][0]["timestep"])
        avg_first_step = sum(first_steps) / len(first_steps) if first_steps else None

        # Subsystem attribution distribution
        all_subsystems = []
        for ep in self.episodes:
            all_subsystems.extend(a["subsystem"] for a in ep["attributions"])
        subsystem_counts = {s: all_subsystems.count(s) for s in set(all_subsystems)}

        # Success rate
        success_rate = sum(1 for e in self.episodes if e["success"]) / total

        return {
            "total_episodes": total,
            "violation_rate": violation_rate,
            "success_rate": success_rate,
            "episodes_with_violations": with_violations,
            "violation_type_distribution": type_counts,
            "avg_steps_to_first_violation": avg_first_step,
            "subsystem_attribution": subsystem_counts,
        }

    def save(self, filename: Optional[str] = None):
        """Save all results to JSON."""
        if filename is None:
            filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)

        output = {
            "experiment_start": self.start_time,
            "experiment_end": datetime.now().isoformat(),
            "aggregate_stats": self.get_aggregate_stats(),
            "episodes": self.episodes,
        }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Results saved to {filepath}")
        return filepath