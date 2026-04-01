"""
Adversarial ManiSkill Task Classes
------------------------------------
Each class inherits from its corresponding ManiSkill base task and adds:
  - An adversarial obstacle box whose pose and size are read from
    ``options["obstacle_pose_xyz"]`` / ``options["obstacle_size"]``
  - Deterministic episode initialization driven by the scenario ``parameters``
    dict (no randomization applied when these keys are present)
  - A high-quality overhead + side-angle camera pair for rendering

Import this module BEFORE calling gym.make() so the tasks get registered:

    import render_env.adv_tasks  # registers all Adv* envs

Then:
    env = gym.make("AdvPickCube-v0", render_mode="rgb_array")
    env.reset(options=scenario["parameters"])
    rgb = env.render()
"""

from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs  # ensure base envs are registered
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Distinct, vivid obstacle colours per task for better photo-realistic visuals
_OBSTACLE_COLOR          = [0.85, 0.20, 0.10, 1.0]   # default bright red
_OBSTACLE_COLOR_PICK     = [0.90, 0.65, 0.05, 1.0]   # gold   — PickCube
_OBSTACLE_COLOR_STACK    = [0.95, 0.38, 0.05, 1.0]   # deep orange — StackCube
_OBSTACLE_COLOR_PUSH     = [0.85, 0.10, 0.10, 1.0]   # vivid red   — PushCube
_OBSTACLE_COLOR_SPHERE   = [0.05, 0.55, 0.65, 1.0]   # teal / cyan — PlaceSphere
_OBSTACLE_COLOR_PEG      = [0.50, 0.05, 0.70, 1.0]   # purple      — PegInsertion


def _xyz_from_options(options: dict, key: str, fallback: list) -> list:
    """
    Return an [x, y, z] list from options, coercing dict-style values if needed.
    Falls back to *fallback* if the key is absent.
    """
    val = options.get(key, fallback)
    if isinstance(val, dict):
        if all(k in val for k in ("x", "y", "z")):
            val = [val["x"], val["y"], val["z"]]
        else:
            val = list(val.values())
    return [float(v) for v in val]


def _float_from_options(options: dict, key: str, fallback: float) -> float:
    v = options.get(key, fallback)
    return float(v)


def _build_obstacle(env, half_size: float, name: str = "obstacle",
                    color=None) -> Any:
    """Build and return a static cubic obstacle actor."""
    return actors.build_box(
        env.scene,
        half_sizes=[half_size] * 3,
        color=color if color is not None else _OBSTACLE_COLOR,
        name=name,
        body_type="static",
        initial_pose=sapien.Pose(p=[0.5, 0.5, 0.5]),  # off to the side initially
    )


def _place_obstacle(env, obstacle_actor, options: dict, fallback_xyz: list):
    """Set the obstacle pose from options."""
    xyz = _xyz_from_options(options, "obstacle_pose_xyz", fallback_xyz)
    obstacle_actor.set_pose(sapien.Pose(p=xyz))


def _adv_camera_configs(eye_front, target):
    """Return two high-quality CameraConfig objects: front-angle + top-down."""
    pose_front = sapien_utils.look_at(eye=eye_front, target=target)
    # Top-down slightly angled
    eye_top = [target[0], target[1] - 0.05, target[2] + 0.65]
    pose_top = sapien_utils.look_at(eye=eye_top, target=target)
    return [
        CameraConfig("render_camera",   pose_front, 640, 480, fov=1.0,  near=0.01, far=100),
        CameraConfig("overhead_camera", pose_top,   640, 480, fov=0.75, near=0.01, far=100),
    ]


# ---------------------------------------------------------------------------
# AdvPickCube-v0
# ---------------------------------------------------------------------------

@register_env("AdvPickCube-v0", max_episode_steps=50)
class AdvPickCubeEnv(PickCubeEnv):
    """
    PickCube with an adversarial obstacle.  Set initial state via::

        env.reset(options={
            "object_pose_xyz":   [x, y, z],
            "obstacle_pose_xyz": [x, y, z],
            "obstacle_size":     0.08,
            ...
        })
    """

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Wide low-angle view: robot arm + cube (x≈0.2-0.5) + obstacle all in frame
        pose = sapien_utils.look_at(eye=[0.80, 0.62, 0.35], target=[0.28, 0.0, 0.03])
        return CameraConfig("render_camera", pose, 640, 480, fov=1.15, near=0.01, far=100)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.4, -0.3, 0.5], target=[0.05, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        half = _float_from_options(options, "obstacle_size", 0.06) / 2.0
        self._obs_actor = _build_obstacle(self, half, name="adv_obstacle",
                                          color=_OBSTACLE_COLOR_PICK)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if "object_pose_xyz" in options:
                xyz = _xyz_from_options(options, "object_pose_xyz", [0.2, 0.0, self.cube_half_size])
                p = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                q = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(b, 4)
                self.cube.set_pose(Pose.create_from_pq(p, q))
            else:
                # fall back to parent randomization
                super()._initialize_episode(env_idx, options)
                return

            if "obstacle_pose_xyz" in options:
                xyz_obs = _xyz_from_options(options, "obstacle_pose_xyz", [0.3, 0.0, 0.05])
                # Obstacles are static; set directly via sapien pose on each sub-scene
                self._obs_actor.set_pose(sapien.Pose(p=xyz_obs))

            # Keep goal site somewhere visible but out of the way
            goal_xyz = [xyz[0] + 0.05, xyz[1], xyz[2] + 0.2]
            goal_p = torch.tensor(goal_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
            goal_q = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(b, 4)
            self.goal_site.set_pose(Pose.create_from_pq(goal_p, goal_q))


# ---------------------------------------------------------------------------
# AdvStackCube-v0
# ---------------------------------------------------------------------------

@register_env("AdvStackCube-v0", max_episode_steps=50)
class AdvStackCubeEnv(StackCubeEnv):
    """StackCube with an adversarial obstacle."""

    def __init__(self, *args, robot_uids="panda_wristcam", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Wide low view: both cubes (x≈0.1-0.35) + obstacle + robot arm visible
        pose = sapien_utils.look_at(eye=[0.72, 0.62, 0.35], target=[0.22, 0.0, 0.03])
        return CameraConfig("render_camera", pose, 640, 480, fov=1.15, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        half = _float_from_options(options, "obstacle_size", 0.05) / 2.0
        self._obs_actor = _build_obstacle(self, half, name="adv_obstacle",
                                          color=_OBSTACLE_COLOR_STACK)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if "red_cube_pose_xyz" in options:
                half = 0.02
                red_xyz = _xyz_from_options(options, "red_cube_pose_xyz", [0.1, -0.1, half])
                red_p = torch.tensor(red_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                q_id = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(b, 4)
                self.cubeA.set_pose(Pose.create_from_pq(red_p, q_id))

                blue_xyz = _xyz_from_options(options, "blue_cube_pose_xyz", [0.1, 0.1, half])
                blue_p = torch.tensor(blue_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                self.cubeB.set_pose(Pose.create_from_pq(blue_p, q_id))

                if "obstacle_pose_xyz" in options:
                    obs_xyz = _xyz_from_options(options, "obstacle_pose_xyz", [0.2, 0.0, 0.05])
                    self._obs_actor.set_pose(sapien.Pose(p=obs_xyz))
            else:
                super()._initialize_episode(env_idx, options)


# ---------------------------------------------------------------------------
# AdvPushCube-v0
# ---------------------------------------------------------------------------

@register_env("AdvPushCube-v0", max_episode_steps=50)
class AdvPushCubeEnv(PushCubeEnv):
    """PushCube with an adversarial obstacle."""

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Low wide-angle view along the push axis: captures cube → obstacle → goal zone
        pose = sapien_utils.look_at(eye=[0.78, 0.42, 0.26], target=[0.28, 0.0, 0.02])
        return CameraConfig("render_camera", pose, 640, 480, fov=1.15, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        half = _float_from_options(options, "obstacle_size", 0.05) / 2.0
        self._obs_actor = _build_obstacle(self, half, name="adv_obstacle",
                                          color=_OBSTACLE_COLOR_PUSH)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if "object_pose_xyz" in options:
                cube_xyz = _xyz_from_options(options, "object_pose_xyz", [0.2, 0.0, self.cube_half_size])
                cube_p = torch.tensor(cube_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                q_id = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(b, 4)
                self.obj.set_pose(Pose.create_from_pq(cube_p, q_id))

                goal_xyz = _xyz_from_options(options, "goal_pose_xyz",
                                             [cube_xyz[0] + 0.15, cube_xyz[1], cube_xyz[2]])
                goal_p = torch.tensor(goal_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                self.goal_region.set_pose(Pose.create_from_pq(goal_p, q_id))

                if "obstacle_pose_xyz" in options:
                    obs_xyz = _xyz_from_options(options, "obstacle_pose_xyz", [0.3, 0.0, 0.05])
                    self._obs_actor.set_pose(sapien.Pose(p=obs_xyz))
            else:
                super()._initialize_episode(env_idx, options)


# ---------------------------------------------------------------------------
# AdvPlaceSphere-v0
# ---------------------------------------------------------------------------

@register_env("AdvPlaceSphere-v0", max_episode_steps=50)
class AdvPlaceSphereEnv(PlaceSphereEnv):
    """PlaceSphere with an adversarial obstacle."""

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Sphere (x≈0.1-0.25) + bowl (x≈0.15-0.3) + obstacle + robot arm
        pose = sapien_utils.look_at(eye=[0.68, 0.58, 0.32], target=[0.20, 0.0, 0.03])
        return CameraConfig("render_camera", pose, 640, 480, fov=1.15, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        half = _float_from_options(options, "obstacle_size", 0.05) / 2.0
        self._obs_actor = _build_obstacle(self, half, name="adv_obstacle",
                                          color=_OBSTACLE_COLOR_SPHERE)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            if "sphere_pose_xyz" in options:
                sph_xyz = _xyz_from_options(options, "sphere_pose_xyz", [0.1, -0.05, self.radius])
                sph_p = torch.tensor(sph_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                q_id = torch.tensor([1.0, 0.0, 0.0, 0.0]).unsqueeze(0).expand(b, 4)
                self.obj.set_pose(Pose.create_from_pq(sph_p, q_id))   # PlaceSphereEnv uses self.obj

                bowl_xyz = _xyz_from_options(options, "bowl_pose_xyz", [0.1, 0.05, 0.0])
                bowl_p = torch.tensor(bowl_xyz, dtype=torch.float32).unsqueeze(0).expand(b, 3)
                self.bin.set_pose(Pose.create_from_pq(bowl_p, q_id))

                if "obstacle_pose_xyz" in options:
                    obs_xyz = _xyz_from_options(options, "obstacle_pose_xyz", [0.2, 0.0, 0.05])
                    self._obs_actor.set_pose(sapien.Pose(p=obs_xyz))
            else:
                super()._initialize_episode(env_idx, options)


# ---------------------------------------------------------------------------
# AdvPegInsertion-v0
# ---------------------------------------------------------------------------

@register_env("AdvPegInsertion-v0", max_episode_steps=50)
class AdvPegInsertionEnv(PegInsertionSideEnv):
    """PegInsertionSide with an adversarial obstacle."""

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Peg (x≈0.1-0.3) + insertion box (x≈0.15-0.35) + obstacle + robot arm
        pose = sapien_utils.look_at(eye=[0.68, 0.55, 0.32], target=[0.22, 0.0, 0.04])
        return CameraConfig("render_camera", pose, 640, 480, fov=1.10, near=0.01, far=100)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        half = _float_from_options(options, "obstacle_size", 0.04) / 2.0
        self._obs_actor = _build_obstacle(self, half, name="adv_obstacle",
                                          color=_OBSTACLE_COLOR_PEG)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # Delegate to parent for correct peg+box initialization
        super()._initialize_episode(env_idx, options)
        if "obstacle_pose_xyz" in options:
            obs_xyz = _xyz_from_options(options, "obstacle_pose_xyz", [0.25, 0.0, 0.05])
            self._obs_actor.set_pose(sapien.Pose(p=obs_xyz))


# ---------------------------------------------------------------------------
# Task registry mapping scenario task names → Adv env IDs
# ---------------------------------------------------------------------------

ADV_TASK_MAP = {
    "PickCube-v1":         "AdvPickCube-v0",
    "StackCube-v1":        "AdvStackCube-v0",
    "PushCube-v1":         "AdvPushCube-v0",
    "PlaceSphere-v1":      "AdvPlaceSphere-v0",
    "PegInsertionSide-v1": "AdvPegInsertion-v0",
}
