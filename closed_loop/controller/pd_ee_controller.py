"""
PD End-Effector Controller with Potential Field Obstacle Avoidance
==================================================================
Computes delta joint position commands for the Franka Panda to reach
a Cartesian goal while avoiding obstacles. Compatible with ManiSkill3's
pd_joint_delta_pos control mode (8-dim action: 7 arm joints + 1 gripper,
normalized [-1, 1]).

Obstacle avoidance uses a repulsive potential field: when the EE is
within a configurable influence distance of the obstacle, a repulsive
force pushes it away. The final command is the sum of the attractive
(goal-seeking) and repulsive (obstacle-avoiding) forces.

Waypoint navigation: when an obstacle blocks the direct path, the controller
first targets a waypoint above the wall, then descends to the goal. Phase
transitions are decided from (possibly noisy) sensor readings.

Known failure modes (exploitable by adversarial scenarios):
- Waypoint transition: noise corrupts "am I past the wall?" check → premature
  switch crashes into wall, or late switch causes timeout
- Overshoot: delay + high gain → EE overshoots waypoint into obstacle zone
- Noise on EE z → wrong clearance estimate → clips wall top
- Gain mismatch: high attractive gain can overwhelm repulsion
- Singularities: near singular configs, Jacobian pseudoinverse amplifies noise
"""

import numpy as np
from typing import Optional


class PDEEController:
    """
    Cartesian PD controller with potential field obstacle avoidance.

    Attractive force: PD control pulling EE toward goal.
    Repulsive force: inverse-square repulsion pushing EE away from obstacle
    when within the influence distance.

    Uses the analytical Jacobian of the Franka Panda computed from DH
    parameters at the current joint configuration. The Cartesian EE delta
    is mapped to joint deltas via the damped pseudoinverse of the 3x7
    position Jacobian.

    For ManiSkill3 pd_joint_delta_pos:
    - Action dimension: 8 (7 arm joints + 1 gripper)
    - Action range: [-1, 1] (normalized)
    """

    # Franka Panda URDF joint parameters: (xyz, rpy) for each revolute joint.
    # Extracted from panda_v3.urdf. All joints rotate about local z-axis.
    # Each entry: ([tx, ty, tz], [roll, pitch, yaw])
    _URDF_JOINTS = [
        ([0,       0,      0.333], [0,       0, 0]),        # joint 1
        ([0,       0,      0    ], [-np.pi/2, 0, 0]),       # joint 2
        ([0,      -0.316,  0    ], [np.pi/2,  0, 0]),       # joint 3
        ([0.0825,  0,      0    ], [np.pi/2,  0, 0]),       # joint 4
        ([-0.0825, 0.384,  0    ], [-np.pi/2, 0, 0]),       # joint 5
        ([0,       0,      0    ], [np.pi/2,  0, 0]),       # joint 6
        ([0.088,   0,      0    ], [np.pi/2,  0, 0]),       # joint 7
    ]

    # Fixed transforms from joint 7 frame to TCP:
    # joint8 (flange): xyz="0 0 0.107"
    # hand_joint:      rpy="0 0 -π/4"
    # hand_tcp_joint:  xyz="0 0 0.1034"
    _FIXED_TRANSFORMS = [
        ([0, 0, 0.107],  [0, 0, 0]),
        ([0, 0, 0],      [0, 0, -np.pi/4]),
        ([0, 0, 0.1034], [0, 0, 0]),
    ]

    # Damping factor for pseudoinverse (avoids singularity blow-up)
    _DAMPING = 0.01

    def __init__(
        self,
        kp: float = 4.0,
        kd: float = 0.5,
        max_delta: float = 0.8,
        gain_scale: float = 1.0,
        delay_steps: int = 0,
        repulsive_gain: float = 0.5,
        influence_distance: float = 0.15,
        obstacle_pos: Optional[list] = None,
        obstacle_half_size: Optional[list] = None,
        waypoint_clearance: float = 0.08,
    ):
        self.kp = kp * gain_scale
        self.kd = kd * gain_scale
        self.max_delta = max_delta
        self.delay_steps = delay_steps

        # Potential field parameters
        self.repulsive_gain = repulsive_gain
        self.influence_distance = influence_distance

        # Waypoint navigation: obstacle geometry for computing the over-wall waypoint
        self._obstacle_pos = obstacle_pos      # [x, y, z] base position (z=table surface)
        self._obstacle_hs = obstacle_half_size  # [hx, hy, hz]
        self._waypoint_clearance = waypoint_clearance

        # State
        self._action_buffer = []
        self._prev_error = np.zeros(3)
        self._phase = "climb"  # "climb" → waypoint above wall, "reach" → final goal

    def reset(self):
        """Reset controller state between episodes."""
        self._action_buffer = []
        self._prev_error = np.zeros(3)
        self._phase = "climb"

    def _get_target(self, ee_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
        """
        Waypoint navigation: return the current target position.

        In the "climb" phase, target a waypoint above the wall so the EE
        clears the obstacle. Once the EE is above and past the wall,
        switch to the "reach" phase and go to the real goal.

        The phase decision uses the (possibly noisy) ee_pos, which makes
        it vulnerable to sensor noise causing premature/late switching.
        """
        if self._obstacle_pos is None or self._obstacle_hs is None:
            return goal_pos

        wall_y = self._obstacle_pos[1] + self._obstacle_hs[1]  # far edge of wall
        wall_top_z = self._obstacle_pos[2] + self._obstacle_hs[2] * 2  # top surface
        clearance_z = wall_top_z + self._waypoint_clearance

        if self._phase == "climb":
            waypoint = np.array([goal_pos[0], wall_y, clearance_z])
            dist_to_wp = np.linalg.norm(ee_pos - waypoint)
            # Switch once close enough to waypoint (distance check is
            # vulnerable to noise — noisy readings can trigger early/late)
            if dist_to_wp < 0.05:
                self._phase = "reach"
                return goal_pos
            return waypoint
        else:
            return goal_pos

    @staticmethod
    def _rpy_to_rotation(roll, pitch, yaw):
        """Convert URDF RPY (fixed-axis XYZ) to a 3x3 rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [  -sp,           cp*sr,           cp*cr    ],
        ])

    @staticmethod
    def _rot_z(theta):
        """Rotation matrix about the z-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _compute_jacobian(self, qpos):
        """
        Compute the 3x7 position Jacobian at the current joint configuration.

        Uses the URDF joint parameterization directly: each joint transform is
        T_i = Trans(xyz) @ Rot(rpy) @ Rot_z(q_i). The Jacobian column for
        revolute joint i is z_i x (p_tcp - p_i), where z_i is the joint axis
        in world frame after applying the joint rotation.
        """
        R = np.eye(3)
        p = np.zeros(3)
        joint_axes = []
        joint_origins = []

        for i in range(7):
            xyz, rpy = self._URDF_JOINTS[i]
            # Apply the fixed origin transform
            R_fixed = self._rpy_to_rotation(*rpy)
            p = p + R @ np.array(xyz)
            R = R @ R_fixed
            # The joint axis (z) in world frame, before joint rotation
            joint_axes.append(R[:, 2].copy())
            joint_origins.append(p.copy())
            # Apply the joint rotation
            R = R @ self._rot_z(qpos[i])

        # Fixed transforms to TCP (flange + hand rotation + tcp offset)
        for xyz, rpy in self._FIXED_TRANSFORMS:
            R_fixed = self._rpy_to_rotation(*rpy)
            p = p + R @ np.array(xyz)
            R = R @ R_fixed

        # Build the position Jacobian
        J = np.zeros((3, 7))
        for i in range(7):
            J[:, i] = np.cross(joint_axes[i], p - joint_origins[i])

        return J

    def _damped_pinv(self, J):
        """
        Compute the damped pseudoinverse J^T (J J^T + λ²I)^{-1}.

        This avoids the numerical blow-up of the standard pseudoinverse
        near kinematic singularities, at the cost of a small tracking error.
        """
        JJT = J @ J.T
        return J.T @ np.linalg.inv(JJT + self._DAMPING**2 * np.eye(JJT.shape[0]))

    def compute_action(
        self,
        ee_pos: np.ndarray,
        goal_pos: np.ndarray,
        qpos: Optional[np.ndarray] = None,
        dt: float = 0.05,
        noise_ee: float = 0.0,
        noise_joint: float = 0.0,
        obstacle_distance: float = float('inf'),
        obstacle_direction: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute a delta joint position action to move EE toward goal
        while avoiding obstacles via potential field repulsion.

        Args:
            ee_pos: Current end-effector position (3,)
            goal_pos: Target position (3,)
            qpos: Current joint positions (7+ dim); used to compute the Jacobian
            dt: Control timestep
            noise_ee: Std of Gaussian noise on EE position reading
            noise_joint: Std of Gaussian noise on joint readings
            obstacle_distance: Distance from EE to nearest obstacle surface
            obstacle_direction: Unit vector pointing from obstacle toward EE

        Returns:
            Action array (8,) normalized to [-1, 1] for pd_joint_delta_pos
        """
        # --- Apply sensor noise ---
        if noise_ee > 0:
            ee_pos = ee_pos + np.random.normal(0, noise_ee, size=3)
            # Noise also corrupts obstacle distance estimate
            obstacle_distance += np.random.normal(0, noise_ee)
            obstacle_distance = max(obstacle_distance, 0.001)
        if noise_joint > 0 and qpos is not None:
            qpos = qpos + np.random.normal(0, noise_joint, size=qpos.shape)

        # --- Waypoint navigation: pick current target (uses noisy ee_pos) ---
        target_pos = self._get_target(ee_pos, goal_pos)

        # --- Attractive force: PD control toward target ---
        error = target_pos - ee_pos
        d_error = (error - self._prev_error) / dt
        self._prev_error = error.copy()

        attractive = self.kp * error + self.kd * d_error

        # --- Repulsive force: potential field from obstacle ---
        repulsive = np.zeros(3)
        if obstacle_distance < self.influence_distance and obstacle_direction is not None:
            # Inverse-square repulsion: stronger as EE gets closer
            # F_rep = gain * (1/dist - 1/d0) * (1/dist^2) * direction
            inv_dist = 1.0 / obstacle_distance
            inv_d0 = 1.0 / self.influence_distance
            repulsive = self.repulsive_gain * (inv_dist - inv_d0) * (inv_dist ** 2) * obstacle_direction

        # --- Combined Cartesian displacement ---
        ee_delta = (attractive + repulsive) * dt

        # --- Map to joint space via current-configuration Jacobian ---
        if qpos is not None and len(qpos) >= 7:
            J = self._compute_jacobian(qpos[:7])
            J_pinv = self._damped_pinv(J)
            joint_delta = J_pinv @ ee_delta  # (7,)
        else:
            # Fallback: zero action if no joint state available
            joint_delta = np.zeros(7)

        # --- Normalize to [-1, 1] ---
        # pd_joint_delta_pos expects normalized actions
        joint_delta = np.clip(joint_delta / self.max_delta, -1.0, 1.0)

        # --- Build full action: 7 arm joints + 1 gripper ---
        action = np.zeros(8)
        action[:7] = joint_delta
        action[7] = 1.0  # gripper closed

        # --- Apply control delay ---
        if self.delay_steps > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self.delay_steps:
                return self._action_buffer.pop(0)
            else:
                return np.zeros(8)

        return action
