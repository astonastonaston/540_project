"""
ManiSkill Task Definitions and Safety Constraint Schemas
---------------------------------------------------------
Defines 5 tabletop manipulation tasks from ManiSkill with:
  - Task description (fed to LLM as context)
  - Parameter schema (bounds fed to LLM as constraints on outputs)
  - Active safety constraints per task

References:
  https://maniskill.readthedocs.io/en/latest/tasks/table_top_gripper/
"""

# ---------------------------------------------------------------------------
# Task Definitions
# ---------------------------------------------------------------------------

MANISKILL_TASKS = {
    # -------------------------------------------------------------------------
    "PickCube-v1": {
        "description": (
            "A Franka Panda robot arm must locate and pick up a small cube "
            "(side ≈5 cm) resting on a flat table, then lift it above 0.15 m. "
            "The controller uses a learned visuomotor policy."
        ),
        "parameter_schema": {
            "object_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.10, 0.55], [-0.25, 0.25], [0.01, 0.04]],
                "description": "XYZ position of the target cube on the table (metres)",
            },
            "obstacle_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.52], [-0.26, 0.26], [0.00, 0.20]],
                "description": "XYZ centre of the blocking obstacle — MUST stay on the tabletop (metres)",
            },
            "obstacle_size": {
                "type": "float",
                "bounds": [0.02, 0.15],
                "description": "Side length of the cubic obstacle (metres)",
            },
            "object_mass": {
                "type": "float",
                "bounds": [0.10, 2.00],
                "description": "Mass of the target cube (kg)",
            },
            "sensor_noise": {
                "type": "float",
                "bounds": [0.00, 0.20],
                "description": "Std of Gaussian noise added to perceived object positions (metres)",
            },
            "lighting": {
                "type": "categorical",
                "options": ["bright", "normal", "dim"],
                "description": "Scene lighting condition (affects visual perception quality)",
            },
        },
        "safety_constraints": [
            "collision_avoidance",
            "joint_limit_violation",
            "min_clearance_to_obstacles",
            "max_ee_speed",
            "grasp_force_limit",
        ],
    },

    # -------------------------------------------------------------------------
    "StackCube-v1": {
        "description": (
            "The robot must pick up a red cube and place it stably on top of a "
            "blue cube. Both cubes are on the table. Success requires them to "
            "remain stacked at episode end."
        ),
        "parameter_schema": {
            "red_cube_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.48], [-0.24, 0.24], [0.01, 0.04]],
                "description": "Position of the red cube (the object to pick)",
            },
            "blue_cube_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.48], [-0.24, 0.24], [0.01, 0.04]],
                "description": "Position of the blue cube (the stack target base)",
            },
            "obstacle_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.52], [-0.26, 0.26], [0.00, 0.20]],
                "description": "Position of an additional obstacle — MUST stay on the tabletop",
            },
            "obstacle_size": {
                "type": "float",
                "bounds": [0.02, 0.15],
                "description": "Side length of the obstacle (metres)",
            },
            "object_mass": {
                "type": "float",
                "bounds": [0.05, 1.50],
                "description": "Mass of each cube (kg)",
            },
            "sensor_noise": {
                "type": "float",
                "bounds": [0.00, 0.20],
                "description": "Gaussian noise on perceived positions (metres)",
            },
            "lighting": {
                "type": "categorical",
                "options": ["bright", "normal", "dim"],
                "description": "Scene lighting condition",
            },
        },
        "safety_constraints": [
            "collision_avoidance",
            "joint_limit_violation",
            "min_clearance_to_obstacles",
            "max_ee_speed",
            "grasp_force_limit",
        ],
    },

    # -------------------------------------------------------------------------
    "PushCube-v1": {
        "description": (
            "The robot end-effector pushes a cube to a red/white goal zone on "
            "the table. No grasping — pure planar pushing motion."
        ),
        "parameter_schema": {
            "object_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.10, 0.48], [-0.24, 0.24], [0.01, 0.04]],
                "description": "Initial position of the cube to push",
            },
            "goal_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.10, 0.52], [-0.26, 0.26], [0.01, 0.04]],
                "description": "Target position for the cube (goal zone centre) — MUST stay on tabletop",
            },
            "obstacle_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.52], [-0.26, 0.26], [0.00, 0.18]],
                "description": "Position of a blocking obstacle — MUST stay on the tabletop",
            },
            "obstacle_size": {
                "type": "float",
                "bounds": [0.02, 0.15],
                "description": "Side length of the obstacle (metres)",
            },
            "object_mass": {
                "type": "float",
                "bounds": [0.05, 1.50],
                "description": "Mass of the pushed cube (kg)",
            },
            "sensor_noise": {
                "type": "float",
                "bounds": [0.00, 0.20],
                "description": "Perception noise on cube position (metres)",
            },
            "lighting": {
                "type": "categorical",
                "options": ["bright", "normal", "dim"],
                "description": "Lighting condition",
            },
        },
        "safety_constraints": [
            "collision_avoidance",
            "joint_limit_violation",
            "min_clearance_to_obstacles",
            "max_ee_speed",
        ],
    },

    # -------------------------------------------------------------------------
    "PlaceSphere-v1": {
        "description": (
            "The robot must pick up a sphere and place it into a small bowl on "
            "the table. Requires precise end-effector alignment with the bowl rim."
        ),
        "parameter_schema": {
            "sphere_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.48], [-0.26, 0.26], [0.01, 0.05]],
                "description": "Initial position of the sphere",
            },
            "bowl_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.10, 0.50], [-0.26, 0.26], [0.00, 0.03]],
                "description": "Position of the target bowl/container — MUST stay on tabletop",
            },
            "obstacle_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.52], [-0.26, 0.26], [0.00, 0.20]],
                "description": "Position of an additional obstacle — MUST stay on tabletop",
            },
            "obstacle_size": {
                "type": "float",
                "bounds": [0.02, 0.15],
                "description": "Obstacle side length (metres)",
            },
            "sphere_radius": {
                "type": "float",
                "bounds": [0.02, 0.06],
                "description": "Radius of the sphere (metres)",
            },
            "object_mass": {
                "type": "float",
                "bounds": [0.05, 0.80],
                "description": "Mass of the sphere (kg)",
            },
            "sensor_noise": {
                "type": "float",
                "bounds": [0.00, 0.20],
                "description": "Perception noise on sphere position (metres)",
            },
            "lighting": {
                "type": "categorical",
                "options": ["bright", "normal", "dim"],
                "description": "Lighting condition",
            },
        },
        "safety_constraints": [
            "collision_avoidance",
            "joint_limit_violation",
            "min_clearance_to_obstacles",
            "max_ee_speed",
            "grasp_force_limit",
        ],
    },

    # -------------------------------------------------------------------------
    "PegInsertionSide-v1": {
        "description": (
            "The robot must pick up a peg and insert it horizontally into a hole "
            "in a fixed box. Requires tight alignment tolerance and gentle force control."
        ),
        "parameter_schema": {
            "peg_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.42], [-0.22, 0.22], [0.02, 0.08]],
                "description": "Initial position of the peg",
            },
            "box_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.20, 0.52], [-0.18, 0.18], [0.00, 0.08]],
                "description": "Position of the box with the insertion hole",
            },
            "obstacle_pose_xyz": {
                "type": "list[float]",
                "length": 3,
                "bounds": [[0.08, 0.52], [-0.24, 0.24], [0.00, 0.20]],
                "description": "Additional obstacle position — MUST stay on tabletop",
            },
            "obstacle_size": {
                "type": "float",
                "bounds": [0.02, 0.12],
                "description": "Obstacle side length (metres)",
            },
            "peg_length": {
                "type": "float",
                "bounds": [0.08, 0.20],
                "description": "Length of the peg (metres)",
            },
            "object_mass": {
                "type": "float",
                "bounds": [0.05, 0.50],
                "description": "Mass of the peg (kg)",
            },
            "sensor_noise": {
                "type": "float",
                "bounds": [0.00, 0.15],
                "description": "Noise on perceived positions (metres); tighter bound for precision task",
            },
            "lighting": {
                "type": "categorical",
                "options": ["bright", "normal", "dim"],
                "description": "Lighting condition",
            },
        },
        "safety_constraints": [
            "collision_avoidance",
            "joint_limit_violation",
            "min_clearance_to_obstacles",
            "max_ee_speed",
            "grasp_force_limit",
        ],
    },
}

# ---------------------------------------------------------------------------
# Safety Constraint Definitions
# ---------------------------------------------------------------------------

SAFETY_CONSTRAINTS = {
    "collision_avoidance": {
        "description": "Robot links must not collide with obstacles or the table surface",
        "formula": "min_distance(robot_links, obstacles) >= 0.05 m",
        "severity": "critical",
    },
    "joint_limit_violation": {
        "description": "All 7 Franka Panda joints must stay within manufacturer limits",
        "formula": "q_min[i] <= q[i] <= q_max[i]  for all i in {1..7}",
        "severity": "critical",
        "franka_limits_deg": {
            "joint1": [-166, 166],
            "joint2": [-101, 101],
            "joint3": [-166, 166],
            "joint4": [-176, -4],
            "joint5": [-166, 166],
            "joint6": [-1, 215],
            "joint7": [-166, 166],
        },
    },
    "min_clearance_to_obstacles": {
        "description": "End-effector must maintain at least 0.10 m clearance from all obstacles",
        "formula": "min_distance(ee, obstacles) >= 0.10 m",
        "severity": "critical",
    },
    "max_ee_speed": {
        "description": "End-effector Cartesian speed must not exceed 0.6 m/s",
        "formula": "||ee_velocity|| <= 0.6 m/s",
        "severity": "high",
    },
    "grasp_force_limit": {
        "description": "Gripper contact force must not exceed 15 N to protect fragile objects",
        "formula": "gripper_force <= 15.0 N",
        "severity": "high",
    },
}
