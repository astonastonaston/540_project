"""
Safety Monitor
==============
Continuously checks all safety constraints at every timestep.
Records structured violation events for downstream analysis.
"""

import yaml
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Violation:
    """A single safety violation event."""
    type: str               # e.g., "collision", "velocity_exceeded", "joint_limit"
    timestep: int
    value: float            # the violating measurement
    limit: float            # the constraint threshold
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class SafetyMonitor:
    """
    Monitors robot state against formal safety constraints.
    Instantiate once, call check() every timestep, read violations after episode.
    """

    # Franka Panda joint limits (radians)
    JOINT_LIMITS_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JOINT_LIMITS_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    def __init__(self, config_path: str = "constraints.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.c = cfg["constraints"]
        self.violations: List[Violation] = []
        self._timestep = 0

    def reset(self):
        """Clear violations for a new episode."""
        self.violations = []
        self._timestep = 0

    def check(self, state: dict) -> List[Violation]:
        """
        Check all safety constraints against current state.

        Args:
            state: dict from env.get_safety_state() with keys:
                ee_pos, ee_vel, ee_speed, qpos, qvel,
                max_joint_velocity, obstacle_distance, goal_distance

        Returns:
            List of new violations detected this timestep.
        """
        new_violations = []
        t = self._timestep

        # 1) Obstacle collision / proximity
        if state["obstacle_distance"] < self.c["min_obstacle_distance"]:
            v = Violation(
                type="obstacle_proximity",
                timestep=t,
                value=state["obstacle_distance"],
                limit=self.c["min_obstacle_distance"],
                details={"ee_pos": state["ee_pos"].tolist()},
            )
            new_violations.append(v)

        # 2) End-effector velocity
        if state["ee_speed"] > self.c["max_ee_velocity"]:
            v = Violation(
                type="ee_velocity_exceeded",
                timestep=t,
                value=state["ee_speed"],
                limit=self.c["max_ee_velocity"],
                details={"ee_vel": state["ee_vel"].tolist()},
            )
            new_violations.append(v)

        # 3) Joint velocity limits
        if state["max_joint_velocity"] > self.c["max_joint_velocity"]:
            v = Violation(
                type="joint_velocity_exceeded",
                timestep=t,
                value=state["max_joint_velocity"],
                limit=self.c["max_joint_velocity"],
                details={"qvel": state["qvel"].tolist()},
            )
            new_violations.append(v)

        # 4) Joint position limits (with margin)
        margin = self.c["joint_limit_margin"]
        qpos = state["qpos"][:7]  # arm joints only
        for i in range(7):
            if qpos[i] < self.JOINT_LIMITS_LOW[i] + margin:
                v = Violation(
                    type="joint_limit_low",
                    timestep=t,
                    value=qpos[i],
                    limit=self.JOINT_LIMITS_LOW[i] + margin,
                    details={"joint_index": i},
                )
                new_violations.append(v)
            elif qpos[i] > self.JOINT_LIMITS_HIGH[i] - margin:
                v = Violation(
                    type="joint_limit_high",
                    timestep=t,
                    value=qpos[i],
                    limit=self.JOINT_LIMITS_HIGH[i] - margin,
                    details={"joint_index": i},
                )
                new_violations.append(v)

        # 5) Workspace boundary
        ws = self.c["workspace"]
        ee = state["ee_pos"]
        if (ee[0] < ws["x_min"] or ee[0] > ws["x_max"] or
            ee[1] < ws["y_min"] or ee[1] > ws["y_max"] or
            ee[2] < ws["z_min"] or ee[2] > ws["z_max"]):
            v = Violation(
                type="workspace_boundary",
                timestep=t,
                value=0.0,
                limit=0.0,
                details={
                    "ee_pos": ee.tolist(),
                    "workspace": ws,
                },
            )
            new_violations.append(v)

        self.violations.extend(new_violations)
        self._timestep += 1
        return new_violations

    def check_timeout(self, elapsed_time: float) -> Optional[Violation]:
        """Check if episode exceeded maximum allowed time without reaching goal."""
        if elapsed_time > self.c["max_episode_time"]:
            v = Violation(
                type="timeout",
                timestep=self._timestep,
                value=elapsed_time,
                limit=self.c["max_episode_time"],
            )
            self.violations.append(v)
            return v
        return None

    def get_summary(self) -> dict:
        """Summary statistics for the episode."""
        types = [v.type for v in self.violations]
        return {
            "total_violations": len(self.violations),
            "violation_types": list(set(types)),
            "counts_by_type": {t: types.count(t) for t in set(types)},
            "first_violation_step": self.violations[0].timestep if self.violations else None,
        }