from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from script.inverse_kinematics import _normalize_quat, _quat_to_euler_xyz

from .config_export import capture_init_config, export_env_config
from .grasping_env import GraspingEnv
from .grasping_env_ik import (
    DEFAULT_CAMERA_CONFIG,
    DEFAULT_XML_PATH as DEFAULT_GRASP_XML_PATH,
    GraspingEnvIK,
)
from .grasping_env_v1 import GraspingEnvV1
from .grasping_env_v2 import GraspingEnvV2

try:
    from .grasping_env_v3 import GraspingEnvV3
except ModuleNotFoundError:
    GraspingEnvV3 = None

DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"

GRASP_ENV_REGISTRY = {
    "GraspingEnv": GraspingEnv,
    "GraspingEnvIK": GraspingEnvIK,
    "GraspingEnvV1": GraspingEnvV1,
    "GraspingEnvV2": GraspingEnvV2,
}
if GraspingEnvV3 is not None:
    GRASP_ENV_REGISTRY["GraspingEnvV3"] = GraspingEnvV3

INSIDE_PLACE_BOX_HALF_EXTENTS = np.array([0.018, 0.018], dtype=np.float64)
INSIDE_PLACE_TRIANGLE_RADIUS = 0.014
INSIDE_PLACE_CYLINDER_RADIUS = 0.018
INSIDE_PLACE_Z_OFFSET_MIN = -0.025
INSIDE_PLACE_Z_OFFSET_MAX = 0.008


class InsertTargetEnvIK(GraspingEnvIK):
    """Insert env with Cartesian IK actions and a grasp-policy reset.

    Reset is sampled from a GraspingEnvIK policy as soon as the gripper has closed.
    The insert task then keeps the gripper closed until the object reaches the
    release target 5 cm above the active place with sufficiently aligned yaw.
    """

    def __init__(
        self,
        xml_file: str = str(DEFAULT_XML_PATH),
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 0.0,
        reward_orientation_weight: float = 0.0,
        reward_inside_place_bonus: float = 30.0,
        control_penalty_weight: float = 0.0,
        distance_tanh_scale: float = 0.05,
        success_distance: float = 0.012,
        success_angle_deg: float = 10.0,
        success_steps_required: int = 5,
        max_episode_steps: int = 180,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.22, 0.00),
        ik_workspace_high: tuple[float, float, float] = (0.35, 0.22, 0.45),
        ik_position_only: bool = False,
        ik_max_iters: int = 80,
        ik_position_tolerance: float = 1e-3,
        ik_rotation_tolerance_deg: float = 3.0,
        ik_damping: float = 1e-3,
        ik_step_size: float = 0.4,
        ik_max_delta_deg: float = 5.0,
        ik_rotation_weight: float = 0.35,
        ik_random_restarts: int = 0,
        ik_seed: int | None = 0,
        control_interpolation_steps: int = 10,
        max_joint_ctrl_delta_deg: float = 5.0,
        smooth_cartesian_target: bool = True,
        debug_ik: bool = False,
        target_x_range: tuple[float, float] = (0.16, 0.27),
        target_y_range: tuple[float, float] = (-0.16, 0.16),
        target_place_z: float = 0.02,
        target_place_yaw_range: tuple[float, float] = (-np.pi / 6.0, np.pi / 6.0),
        target_height_above_place: float = 0.05,
        target_min_object_xy_distance: float = 0.06,
        gripper_release_distance: float | None = None,
        gripper_release_angle_deg: float = 10.0,
        terminate_ee_obj_distance: float = 0.08,
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        ee_frame_body_name: str = "ee_frame_vis",
        object_frame_body_name: str = "object_frame_vis",
        target_frame_body_name: str = "target_frame_vis",
        grasp_model_path: str | None = None,
        grasp_env_name: str = "GraspingEnvIK",
        grasp_xml_file: str | None = None,
        grasp_max_steps: int = 220,
        grasp_attempts_per_reset: int = 6,
        grasp_deterministic: bool = True,
        grasp_success_min_lift: float | None = None,
        grasp_success_max_lift: float | None = 0.02,
        grasp_success_ee_obj_dist: float = 0.025,
        grasp_success_hold_steps: int = 1,
        grasp_ctrl_close_threshold: float = 0.005,
        grasp_qpos_close_threshold: float = 0.002,
        grasp_transfer_settle_steps: int = 20,
        allow_grasp_fallback_snapshot: bool = False,
        **kwargs,
    ):
        init_config = capture_init_config(locals())
        self._insert_env_ready = False

        if grasp_model_path is None:
            raise ValueError(
                "InsertTargetEnvIK requires `grasp_model_path` so reset can start "
                "from a trained GraspingEnvIK policy."
            )

        grasp_model_path_obj = Path(grasp_model_path).expanduser()
        if not grasp_model_path_obj.is_absolute():
            grasp_model_path_obj = grasp_model_path_obj.resolve()
        if not grasp_model_path_obj.exists():
            raise FileNotFoundError(f"Grasp model not found: {grasp_model_path_obj}")

        grasp_xml_path_obj = (
            DEFAULT_GRASP_XML_PATH
            if grasp_xml_file is None
            else Path(grasp_xml_file).expanduser()
        )
        if not grasp_xml_path_obj.is_absolute():
            grasp_xml_path_obj = grasp_xml_path_obj.resolve()
        if not grasp_xml_path_obj.exists():
            raise FileNotFoundError(f"Grasp XML not found: {grasp_xml_path_obj}")

        if grasp_env_name not in GRASP_ENV_REGISTRY:
            supported = ", ".join(sorted(GRASP_ENV_REGISTRY))
            raise ValueError(
                f"Unsupported grasp env `{grasp_env_name}`. Expected one of: {supported}"
            )

        self._reward_target_weight = float(reward_target_weight)
        self._reward_target_tanh_weight = float(reward_target_tanh_weight)
        self._reward_orientation_weight = float(reward_orientation_weight)
        self._reward_inside_place_bonus = float(reward_inside_place_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._distance_tanh_scale = float(distance_tanh_scale)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_steps_required = int(success_steps_required)
        self.max_episode_steps = int(max_episode_steps)
        self._target_x_range = tuple(float(value) for value in target_x_range)
        self._target_y_range = tuple(float(value) for value in target_y_range)
        self._target_place_z = float(target_place_z)
        self._target_place_yaw_range = tuple(
            float(value) for value in target_place_yaw_range
        )
        self._target_height_above_place = float(target_height_above_place)
        self._target_min_object_xy_distance = float(target_min_object_xy_distance)
        self._gripper_release_distance = (
            self._success_distance
            if gripper_release_distance is None
            else float(gripper_release_distance)
        )
        self._gripper_release_angle_rad = np.deg2rad(float(gripper_release_angle_deg))
        self._terminate_ee_obj_distance = float(terminate_ee_obj_distance)

        if self._distance_tanh_scale <= 0.0:
            raise ValueError("distance_tanh_scale must be greater than 0.")
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        if self._success_angle_rad <= 0.0:
            raise ValueError("success_angle_deg must be greater than 0.")
        if self._success_steps_required <= 0:
            raise ValueError("success_steps_required must be greater than 0.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than 0.")
        if self._target_x_range[0] > self._target_x_range[1]:
            raise ValueError("target_x_range must be ordered as (min_x, max_x).")
        if self._target_y_range[0] > self._target_y_range[1]:
            raise ValueError("target_y_range must be ordered as (min_y, max_y).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError(
                "target_place_yaw_range must be ordered as (min_yaw, max_yaw)."
            )
        if self._target_height_above_place <= 0.0:
            raise ValueError("target_height_above_place must be greater than 0.")
        if self._target_min_object_xy_distance < 0.0:
            raise ValueError(
                "target_min_object_xy_distance must be greater than or equal to 0."
            )
        if self._gripper_release_distance <= 0.0:
            raise ValueError("gripper_release_distance must be greater than 0.")
        if self._gripper_release_angle_rad <= 0.0:
            raise ValueError("gripper_release_angle_deg must be greater than 0.")
        if self._terminate_ee_obj_distance <= 0.0:
            raise ValueError("terminate_ee_obj_distance must be greater than 0.")
        grasp_qpos_close_threshold = float(grasp_qpos_close_threshold)
        if grasp_qpos_close_threshold <= 0.0:
            raise ValueError("grasp_qpos_close_threshold must be greater than 0.")

        self._grasp_model_path = grasp_model_path_obj
        self._grasp_env_name = str(grasp_env_name)
        self._grasp_xml_path = grasp_xml_path_obj
        self._grasp_max_steps = int(grasp_max_steps)
        self._grasp_attempts_per_reset = max(1, int(grasp_attempts_per_reset))
        self._grasp_deterministic = bool(grasp_deterministic)
        self._grasp_success_min_lift = grasp_success_min_lift
        self._grasp_success_max_lift = (
            None if grasp_success_max_lift is None else float(grasp_success_max_lift)
        )
        self._grasp_success_ee_obj_dist = float(grasp_success_ee_obj_dist)
        self._grasp_success_hold_steps = max(1, int(grasp_success_hold_steps))
        self._grasp_ctrl_close_threshold = float(grasp_ctrl_close_threshold)
        self._grasp_qpos_close_threshold = grasp_qpos_close_threshold
        self._grasp_transfer_settle_steps = max(0, int(grasp_transfer_settle_steps))
        self._allow_grasp_fallback_snapshot = bool(allow_grasp_fallback_snapshot)
        self._grasp_env = None
        self._grasp_policy = None

        self._last_grasp_reset_attempts = 0
        self._last_grasp_init_lift_height = 0.0
        self._last_grasp_init_ee_obj_dist = np.inf
        self._last_grasp_reset_source = "uninitialized"

        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            default_camera_config=default_camera_config,
            reward_distance_weight=reward_target_weight,
            reward_distance_tanh_weight=reward_target_tanh_weight,
            reward_orientation_weight=reward_orientation_weight,
            reward_target_bonus=reward_inside_place_bonus,
            control_penalty_weight=control_penalty_weight,
            distance_tanh_scale=distance_tanh_scale,
            success_distance=success_distance,
            success_angle_deg=success_angle_deg,
            success_requires_orientation=True,
            success_steps_required=success_steps_required,
            max_episode_steps=max_episode_steps,
            cartesian_action_scale=cartesian_action_scale,
            cartesian_rotation_scale_deg=cartesian_rotation_scale_deg,
            ik_workspace_low=ik_workspace_low,
            ik_workspace_high=ik_workspace_high,
            ik_position_only=ik_position_only,
            ik_max_iters=ik_max_iters,
            ik_position_tolerance=ik_position_tolerance,
            ik_rotation_tolerance_deg=ik_rotation_tolerance_deg,
            ik_damping=ik_damping,
            ik_step_size=ik_step_size,
            ik_max_delta_deg=ik_max_delta_deg,
            ik_rotation_weight=ik_rotation_weight,
            ik_random_restarts=ik_random_restarts,
            ik_seed=ik_seed,
            control_interpolation_steps=control_interpolation_steps,
            max_joint_ctrl_delta_deg=max_joint_ctrl_delta_deg,
            smooth_cartesian_target=smooth_cartesian_target,
            debug_ik=debug_ik,
            ee_site_name=ee_site_name,
            target_site_name=target_site_name,
            ee_frame_body_name=ee_frame_body_name,
            object_frame_body_name=object_frame_body_name,
            target_frame_body_name=target_frame_body_name,
            **kwargs,
        )
        self._init_config = init_config

        self.place_name_by_object = {
            "box": "cube_place",
            "triangle": "tri_place",
            "cylinder": "cyl_place",
        }
        self.place_site_name_by_object = {
            "box": "cube_place_site",
            "triangle": "tri_place_site",
            "cylinder": "cyl_place_site",
        }
        self.place_geom_name_by_object = {
            "box": "cube_place_geom",
            "triangle": "tri_place_geom",
            "cylinder": "cyl_place_geom",
        }
        self.place_info: dict[str, dict[str, int | str]] = {}
        for obj_name in self.object_names:
            body_name = self.place_name_by_object[obj_name]
            site_name = self.place_site_name_by_object[obj_name]
            geom_name = self.place_geom_name_by_object[obj_name]
            body_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_BODY, body_name, "body"
            )
            site_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_SITE, site_name, "site"
            )
            geom_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_GEOM, geom_name, "geom"
            )
            self.place_info[obj_name] = {
                "body_name": body_name,
                "site_name": site_name,
                "geom_name": geom_name,
                "body_id": body_id,
                "site_id": site_id,
                "geom_id": geom_id,
            }

        self.place_geom_rgba = {
            obj_name: self.model.geom_rgba[int(info["geom_id"])].copy()
            for obj_name, info in self.place_info.items()
        }
        self.target_body_id = int(self.model.site_bodyid[self.target_site_id])
        self.target_body_name = str(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id)
        )
        self._target_site_local_pos = self.model.site_pos[self.target_site_id].copy()
        self._target_site_local_quat = _normalize_quat(
            self.model.site_quat[self.target_site_id].copy()
        )
        self._place_site_local_pose_by_object: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        for obj_name, info in self.place_info.items():
            site_id = int(info["site_id"])
            self._place_site_local_pose_by_object[obj_name] = (
                self.model.site_pos[site_id].copy(),
                _normalize_quat(self.model.site_quat[site_id].copy()),
            )

        self.sampled_target_place_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_site_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.sampled_target_place_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0
        self.gripper_release_latched = False
        self.last_gripper_should_release = False
        self.last_release_target_dist = np.inf
        self.last_release_target_angle = np.inf
        self.last_inside_place = False
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

        self._insert_env_ready = True
        self._sync_target_site_to_release_pose()
        self.sync_visual_frames()
        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _quat_rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        quat = _normalize_quat(np.asarray(quat, dtype=np.float64))
        vec_quat = np.array(
            [0.0, *np.asarray(vec, dtype=np.float64)],
            dtype=np.float64,
        )
        rotated = GraspingEnvIK._quat_multiply(
            GraspingEnvIK._quat_multiply(
                quat,
                vec_quat,
            ),
            GraspingEnvIK._quat_conjugate(quat),
        )
        return rotated[1:]

    def _get_active_place_info(self) -> dict[str, int | str]:
        return self.place_info[self.active_obj_name]

    def _get_active_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_place_info()["site_name"]))

    def _get_pose_in_body_frame(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_quat = _normalize_quat(np.asarray(body_quat, dtype=np.float64))
        body_quat_conj = self._quat_conjugate(body_quat)
        local_pos = self._quat_rotate_vector(
            body_quat_conj,
            np.asarray(world_pos, dtype=np.float64)
            - np.asarray(body_pos, dtype=np.float64),
        )
        local_quat = _normalize_quat(
            self._quat_multiply(
                body_quat_conj,
                np.asarray(world_quat, dtype=np.float64),
            )
        )
        return local_pos, local_quat

    def _pose_to_body_transform(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        local_pos: np.ndarray,
        local_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        world_pos = np.asarray(world_pos, dtype=np.float64)
        world_quat = _normalize_quat(np.asarray(world_quat, dtype=np.float64))
        local_pos = np.asarray(local_pos, dtype=np.float64)
        local_quat = _normalize_quat(np.asarray(local_quat, dtype=np.float64))
        body_quat = _normalize_quat(
            self._quat_multiply(world_quat, self._quat_conjugate(local_quat))
        )
        body_pos = world_pos - self._quat_rotate_vector(body_quat, local_pos)
        return body_pos, body_quat

    def _target_site_pose_to_target_body_pose(
        self,
        target_site_pos: np.ndarray,
        target_site_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._pose_to_body_transform(
            target_site_pos,
            target_site_quat,
            self._target_site_local_pos,
            self._target_site_local_quat,
        )

    def _target_place_site_pose_to_place_body_pose(
        self,
        target_place_site_pos: np.ndarray,
        target_place_site_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_pos, local_quat = self._place_site_local_pose_by_object[
            self.active_obj_name
        ]
        return self._pose_to_body_transform(
            target_place_site_pos,
            target_place_site_quat,
            local_pos,
            local_quat,
        )

    def _set_target_site_pose_in_model(
        self,
        target_site_pos: np.ndarray,
        target_site_quat: np.ndarray,
    ) -> None:
        body_pos, body_quat = self._target_site_pose_to_target_body_pose(
            target_site_pos,
            target_site_quat,
        )
        self.model.body_pos[self.target_body_id] = body_pos
        self.model.body_quat[self.target_body_id] = body_quat

    def _sync_target_site_to_release_pose(self) -> None:
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
        release_target_pos = place_site_pos.copy()
        release_target_pos[2] += self._target_height_above_place
        self._set_target_site_pose_in_model(release_target_pos, place_site_quat)
        mujoco.mj_forward(self.model, self.data)

    def _set_active_place_visual(self) -> None:
        for obj_name, info in self.place_info.items():
            rgba = self.place_geom_rgba[obj_name].copy()
            rgba[3] = (
                self.place_geom_rgba[obj_name][3]
                if obj_name == self.active_obj_name
                else 0.0
            )
            self.model.geom_rgba[int(info["geom_id"])] = rgba

    def _set_place_poses_in_model(
        self,
        active_place_pos: np.ndarray,
        active_place_quat: np.ndarray,
    ) -> None:
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for index, obj_name in enumerate(self.object_names):
            info = self.place_info[obj_name]
            body_id = int(info["body_id"])
            if obj_name == self.active_obj_name:
                self.model.body_pos[body_id] = np.asarray(
                    active_place_pos, dtype=np.float64
                ).reshape(3)
                self.model.body_quat[body_id] = _normalize_quat(active_place_quat)
            else:
                self.model.body_pos[body_id] = np.array(
                    [2.0 + index, 2.0, 0.2],
                    dtype=np.float64,
                )
                self.model.body_quat[body_id] = identity_quat

    def _sample_target_place_pose(
        self,
        object_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        chosen_site_pos = None
        chosen_site_quat = None
        chosen_yaw = 0.0

        for _ in range(50):
            site_pos = np.array(
                [
                    self.np_random.uniform(*self._target_x_range),
                    self.np_random.uniform(*self._target_y_range),
                    self._target_place_z,
                ],
                dtype=np.float64,
            )
            yaw = float(self.np_random.uniform(*self._target_place_yaw_range))
            site_quat = self._yaw_to_quat(yaw)
            chosen_site_pos = site_pos
            chosen_site_quat = site_quat
            chosen_yaw = yaw

            if object_pos is None or self._target_min_object_xy_distance <= 0.0:
                break

            xy_distance = float(
                np.linalg.norm(
                    site_pos[:2] - np.asarray(object_pos, dtype=np.float64)[:2]
                )
            )
            if xy_distance >= self._target_min_object_xy_distance:
                break

        assert chosen_site_pos is not None
        assert chosen_site_quat is not None
        body_pos, body_quat = self._target_place_site_pose_to_place_body_pose(
            chosen_site_pos,
            chosen_site_quat,
        )
        return chosen_site_pos, chosen_site_quat, body_pos, body_quat, chosen_yaw

    @staticmethod
    def _joint_name_map(model) -> dict[str, int]:
        joint_map: dict[str, int] = {}
        for joint_id in range(int(model.njnt)):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name:
                joint_map[str(joint_name)] = joint_id
        return joint_map

    @staticmethod
    def _joint_qpos_size(model, joint_id: int) -> int:
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 7
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 4
        return 1

    @staticmethod
    def _joint_dof_size(model, joint_id: int) -> int:
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        return 1

    def _ensure_grasp_policy_loaded(self) -> None:
        if self._grasp_env is not None and self._grasp_policy is not None:
            return

        try:
            from stable_baselines3 import SAC
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "InsertTargetEnvIK requires stable-baselines3 to load the grasping policy."
            ) from exc

        grasp_env_cls = GRASP_ENV_REGISTRY[self._grasp_env_name]
        grasp_env_kwargs = {
            "xml_file": str(self._grasp_xml_path),
            "render_mode": None,
        }
        if self._grasp_env_name == "GraspingEnvV2":
            grasp_env_kwargs["gripper_assist_steps"] = 0

        self._grasp_env = grasp_env_cls(**grasp_env_kwargs)
        self._grasp_policy = SAC.load(
            str(self._grasp_model_path),
            env=self._grasp_env,
            device="auto",
        )

    def _get_grasp_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        active_obj_name = str(grasp_env.active_obj_name)
        info = grasp_env.object_info[active_obj_name]
        site_name = str(info["site_name"])
        obj_pos = grasp_env.data.site(site_name).xpos.copy()
        obj_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(obj_quat, grasp_env.data.site(site_name).xmat)
        return obj_pos, _normalize_quat(obj_quat)

    def _get_grasp_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        ee_pos = grasp_env.data.site(self.ee_site_name).xpos.copy()
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, grasp_env.data.site(self.ee_site_name).xmat)
        return ee_pos, _normalize_quat(ee_quat)

    def _get_grasp_object_speed(self) -> float:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        active_obj_name = str(grasp_env.active_obj_name)
        info = grasp_env.object_info[active_obj_name]
        dofadr = int(info["dofadr"])
        return float(np.linalg.norm(grasp_env.data.qvel[dofadr : dofadr + 3]))

    def _capture_grasp_snapshot(self, initial_obj_pos: np.ndarray) -> dict:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        gripL_qadr = int(getattr(grasp_env, "gripL_qadr", self.gripL_qadr))
        gripR_qadr = int(getattr(grasp_env, "gripR_qadr", self.gripR_qadr))
        gripL_dadr = int(getattr(grasp_env, "gripL_dadr", self.gripL_dadr))
        gripR_dadr = int(getattr(grasp_env, "gripR_dadr", self.gripR_dadr))
        gripL_act_id = int(getattr(grasp_env, "gripL_act_id", self.gripL_act_id))
        gripR_act_id = int(getattr(grasp_env, "gripR_act_id", self.gripR_act_id))
        obj_pos, obj_quat = self._get_grasp_obj_pose()
        ee_pos, ee_quat = self._get_grasp_ee_pose()
        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        lift_height = float(obj_pos[2] - initial_obj_pos[2])
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))

        return {
            "qpos": grasp_env.data.qpos.copy(),
            "qvel": grasp_env.data.qvel.copy(),
            "ctrl": grasp_env.data.ctrl.copy(),
            "active_object": str(grasp_env.active_obj_name),
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "lift_height": lift_height,
            "ee_obj_dist": ee_obj_dist,
            "object_speed": self._get_grasp_object_speed(),
            "gripper_qpos": grasp_env.data.qpos[[gripL_qadr, gripR_qadr]].copy(),
            "gripper_qvel": grasp_env.data.qvel[[gripL_dadr, gripR_dadr]].copy(),
            "gripper_ctrl": grasp_env.data.ctrl[[gripL_act_id, gripR_act_id]].copy(),
            "gripper_state": str(getattr(grasp_env, "gripper_state", "unknown")),
            "grasp_latched": bool(getattr(grasp_env, "grasp_latched", False)),
        }

    def _snapshot_gripper_closed(self, snapshot: dict) -> bool:
        ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
        qpos = np.asarray(snapshot["gripper_qpos"], dtype=np.float64)
        return bool(
            ctrl[0] < -self._grasp_ctrl_close_threshold
            and ctrl[1] > self._grasp_ctrl_close_threshold
            and qpos[0] < -self._grasp_qpos_close_threshold
            and qpos[1] > self._grasp_qpos_close_threshold
        )

    def _is_good_grasp_snapshot(self, snapshot: dict) -> bool:
        lift_ok = True
        if self._grasp_success_max_lift is not None:
            lift_ok = float(snapshot["lift_height"]) <= self._grasp_success_max_lift

        return bool(
            self._snapshot_gripper_closed(snapshot)
            and lift_ok
            and float(snapshot["ee_obj_dist"]) <= self._grasp_success_ee_obj_dist
        )

    def _score_grasp_snapshot(self, snapshot: dict) -> float:
        is_closed = float(self._snapshot_gripper_closed(snapshot))
        lift_height = max(0.0, float(snapshot["lift_height"]))
        return (
            2.0 * is_closed
            - 4.0 * float(snapshot["ee_obj_dist"])
            - 0.5 * float(snapshot["object_speed"])
            - 4.0 * lift_height
        )

    def _sample_grasp_reset_snapshot(self) -> tuple[dict, str, int]:
        self._ensure_grasp_policy_loaded()
        grasp_env = self._grasp_env
        grasp_policy = self._grasp_policy
        assert grasp_env is not None
        assert grasp_policy is not None

        best_snapshot: dict | None = None
        best_score = -np.inf

        for attempt in range(1, self._grasp_attempts_per_reset + 1):
            grasp_seed = int(self.np_random.integers(0, 2**31 - 1))
            observation, _ = grasp_env.reset(seed=grasp_seed)
            initial_obj_pos, _ = self._get_grasp_obj_pose()
            consecutive_good_steps = 0

            for _ in range(self._grasp_max_steps):
                action, _ = grasp_policy.predict(
                    observation,
                    deterministic=self._grasp_deterministic,
                )
                observation, _reward, terminated, truncated, _info = grasp_env.step(
                    action
                )

                snapshot = self._capture_grasp_snapshot(initial_obj_pos)
                snapshot_score = self._score_grasp_snapshot(snapshot)
                if snapshot_score > best_score:
                    best_score = snapshot_score
                    best_snapshot = snapshot

                if self._is_good_grasp_snapshot(snapshot):
                    consecutive_good_steps += 1
                else:
                    consecutive_good_steps = 0

                if consecutive_good_steps >= self._grasp_success_hold_steps:
                    return snapshot, "gripper_closed_success", attempt

                if terminated or truncated:
                    break

        if best_snapshot is None or not self._allow_grasp_fallback_snapshot:
            raise RuntimeError(
                "Failed to obtain a closed-gripper state from the grasping policy. "
                "Try increasing grasp_max_steps or grasp_attempts_per_reset."
            )

        return (
            best_snapshot,
            "gripper_closed_fallback_best_snapshot",
            self._grasp_attempts_per_reset,
        )

    def _restore_grasp_snapshot(self, snapshot: dict) -> None:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        ctrl = np.asarray(snapshot["ctrl"], dtype=np.float64).copy()
        if ctrl.shape != self.data.ctrl.shape:
            raise ValueError(
                "Transferred ctrl shape does not match InsertTargetEnvIK scene. "
                f"Expected {self.data.ctrl.shape}, got {ctrl.shape}."
            )

        self.active_obj_name = str(snapshot["active_object"])
        self._set_place_poses_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()

        grasp_env = self._grasp_env
        assert grasp_env is not None
        source_qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
        source_qvel = np.asarray(snapshot["qvel"], dtype=np.float64)
        source_model = grasp_env.model
        source_joint_map = self._joint_name_map(source_model)
        target_joint_map = self._joint_name_map(self.model)
        transfer_joint_names = sorted(
            set(source_joint_map).intersection(target_joint_map)
        )

        for joint_name in transfer_joint_names:
            source_joint_id = source_joint_map[joint_name]
            target_joint_id = target_joint_map[joint_name]
            source_qposadr = int(source_model.jnt_qposadr[source_joint_id])
            source_dofadr = int(source_model.jnt_dofadr[source_joint_id])
            target_qposadr = int(self.model.jnt_qposadr[target_joint_id])
            target_dofadr = int(self.model.jnt_dofadr[target_joint_id])
            qpos_size = self._joint_qpos_size(source_model, source_joint_id)
            dof_size = self._joint_dof_size(source_model, source_joint_id)
            target_qpos_size = self._joint_qpos_size(self.model, target_joint_id)
            target_dof_size = self._joint_dof_size(self.model, target_joint_id)

            if qpos_size != target_qpos_size or dof_size != target_dof_size:
                raise ValueError(
                    "Transferred joint shape mismatch for "
                    f"`{joint_name}`: source qpos/dof=({qpos_size}, {dof_size}) "
                    f"target=({target_qpos_size}, {target_dof_size})."
                )

            qpos[target_qposadr : target_qposadr + qpos_size] = source_qpos[
                source_qposadr : source_qposadr + qpos_size
            ]
            qvel[target_dofadr : target_dofadr + dof_size] = source_qvel[
                source_dofadr : source_dofadr + dof_size
            ]

        self.set_state(qpos, qvel)
        self._set_closed_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        self.gripper_state = "closed"
        self._sync_target_site_to_release_pose()
        mujoco.mj_forward(self.model, self.data)

        if self._grasp_transfer_settle_steps > 0:
            settle_ctrl = self.data.ctrl.copy()
            self._set_closed_gripper_target(settle_ctrl)
            settle_ctrl = np.clip(settle_ctrl, self._ctrl_low, self._ctrl_high)
            for _ in range(self._grasp_transfer_settle_steps):
                self.do_simulation(settle_ctrl, 1)

        self._reset_ik_state()

    def _get_inside_place_metrics(self) -> dict[str, np.ndarray | float | int]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        active_place_info = self._get_active_place_info()
        place_body = self.data.body(str(active_place_info["body_name"]))
        place_body_pos = place_body.xpos.copy()
        place_body_quat = _normalize_quat(place_body.xquat.copy())
        place_site_local_pos = self.model.site_pos[
            int(active_place_info["site_id"])
        ].copy()

        obj_local_pos, obj_local_quat = self._get_pose_in_body_frame(
            obj_pos,
            obj_quat,
            place_body_pos,
            place_body_quat,
        )
        local_xy_error = obj_local_pos[:2] - place_site_local_pos[:2]
        local_z_offset = float(obj_local_pos[2] - place_site_local_pos[2])

        if self.active_obj_name == "box":
            xy_inside = bool(
                np.all(np.abs(local_xy_error) <= INSIDE_PLACE_BOX_HALF_EXTENTS)
            )
            xy_margin = float(
                np.min(INSIDE_PLACE_BOX_HALF_EXTENTS - np.abs(local_xy_error))
            )
        elif self.active_obj_name == "triangle":
            radial_error = float(np.linalg.norm(local_xy_error))
            xy_inside = bool(radial_error <= INSIDE_PLACE_TRIANGLE_RADIUS)
            xy_margin = float(INSIDE_PLACE_TRIANGLE_RADIUS - radial_error)
        else:
            radial_error = float(np.linalg.norm(local_xy_error))
            xy_inside = bool(radial_error <= INSIDE_PLACE_CYLINDER_RADIUS)
            xy_margin = float(INSIDE_PLACE_CYLINDER_RADIUS - radial_error)

        z_inside = bool(
            INSIDE_PLACE_Z_OFFSET_MIN <= local_z_offset <= INSIDE_PLACE_Z_OFFSET_MAX
        )
        inside_place = bool(xy_inside and z_inside)

        return {
            "object_local_pos": obj_local_pos,
            "object_local_quat": obj_local_quat,
            "place_site_local_pos": place_site_local_pos,
            "inside_place_xy_error": local_xy_error,
            "inside_place_z_offset": float(local_z_offset),
            "inside_place_xy_margin": float(xy_margin),
            "inside_place_xy_ok": int(xy_inside),
            "inside_place_z_ok": int(z_inside),
            "inside_place": int(inside_place),
        }

    def _get_release_target_alignment(self) -> tuple[float, float, bool]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        release_pose_aligned = bool(
            target_dist <= self._gripper_release_distance
            and target_angle <= self._gripper_release_angle_rad
        )
        return target_dist, target_angle, release_pose_aligned

    def _apply_insert_gripper_heuristic(self, ctrl: np.ndarray) -> None:
        target_dist, target_angle, release_pose_aligned = (
            self._get_release_target_alignment()
        )

        self.gripper_release_latched = bool(
            self.gripper_release_latched or release_pose_aligned
        )
        self.last_gripper_should_release = bool(release_pose_aligned)
        self.last_release_target_dist = target_dist
        self.last_release_target_angle = target_angle

        if self.gripper_release_latched:
            self._set_open_gripper_target(ctrl)
        else:
            self._set_closed_gripper_target(ctrl)

    def step(self, action):
        self.current_step += 1
        self._sync_target_site_to_release_pose()
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)
        self._apply_insert_gripper_heuristic(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        start_ctrl = self.data.ctrl.copy()
        for interp_idx in range(1, self._control_interpolation_steps + 1):
            alpha = interp_idx / self._control_interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            smooth_ctrl = np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high)
            self.do_simulation(smooth_ctrl, self.frame_skip)

        self._sync_target_site_to_release_pose()
        self.sync_visual_frames()
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated_success = self.success_counter >= self._success_steps_required
        terminated_ee_obj_far = bool(
            not self.gripper_release_latched
            and float(reward_info["ee_object_dist"]) >= self._terminate_ee_obj_distance
        )
        terminated = terminated_success or terminated_ee_obj_far
        truncated = self.current_step >= self.max_episode_steps
        reward_info.update(
            terminated_success=int(terminated_success),
            terminated_ee_obj_far=int(terminated_ee_obj_far),
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_pos, ee_quat = self._get_ee_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        inside_place_metrics = self._get_inside_place_metrics()
        inside_place = bool(inside_place_metrics["inside_place"])
        self.last_inside_place = inside_place
        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)

        reward_target = 0.0
        reward_target_tanh = 0.0
        reward_orientation = 0.0
        control_penalty = 0.0
        if not self.gripper_release_latched:
            reward_target = -target_dist * self._reward_target_weight
            reward_target_tanh = (
                1.0 - float(np.tanh(target_dist / self._distance_tanh_scale))
            ) * self._reward_target_tanh_weight
            reward_orientation = -target_angle * self._reward_orientation_weight
            control_penalty = -self._control_penalty_weight * float(
                np.sum(np.square(action))
            )

        reward_inside_place_bonus = (
            self._reward_inside_place_bonus
            if self.gripper_release_latched and inside_place
            else 0.0
        )

        if self.gripper_release_latched and inside_place:
            self.success_counter += 1
        else:
            self.success_counter = 0

        reward = (
            reward_target
            + reward_target_tanh
            + reward_orientation
            + reward_inside_place_bonus
            + control_penalty
        )

        reward_info = {
            "active_object": self.active_obj_name,
            "ee_object_dist": ee_obj_dist,
            "object_target_dist": target_dist,
            "object_target_angle_rad": target_angle,
            "release_target_dist": target_dist,
            "release_target_angle_rad": target_angle,
            "release_target_height_above_place": float(
                self._target_height_above_place
            ),
            "release_pose_aligned": int(
                target_dist <= self._gripper_release_distance
                and target_angle <= self._gripper_release_angle_rad
            ),
            "gripper_release_latched": int(self.gripper_release_latched),
            "gripper_should_release": int(self.last_gripper_should_release),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_orientation": float(reward_orientation),
            "reward_inside_place_bonus": float(reward_inside_place_bonus),
            "reward_bonus": float(reward_inside_place_bonus),
            "control_penalty": float(control_penalty),
            "inside_place": int(inside_place),
            "inside_place_xy_ok": int(inside_place_metrics["inside_place_xy_ok"]),
            "inside_place_z_ok": int(inside_place_metrics["inside_place_z_ok"]),
            "inside_place_x_error": float(
                np.asarray(inside_place_metrics["inside_place_xy_error"])[0]
            ),
            "inside_place_y_error": float(
                np.asarray(inside_place_metrics["inside_place_xy_error"])[1]
            ),
            "inside_place_z_offset": float(
                inside_place_metrics["inside_place_z_offset"]
            ),
            "inside_place_xy_margin": float(
                inside_place_metrics["inside_place_xy_margin"]
            ),
            "success_counter": int(self.success_counter),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
        }
        return float(reward), reward_info

    def _reset_insert_episode_state(self) -> None:
        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.gripper_release_latched = False
        self.last_gripper_should_release = False
        self.last_release_target_dist = np.inf
        self.last_release_target_angle = np.inf
        self.last_inside_place = False
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

    def _initialize_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        reset_source: str,
        attempt_count: int,
    ) -> np.ndarray:
        self._reset_insert_episode_state()
        self.active_obj_name = str(snapshot["active_object"])

        (
            self.sampled_target_place_site_pos,
            self.sampled_target_place_site_quat,
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
        ) = self._sample_target_place_pose(np.asarray(snapshot["obj_pos"]))

        self._restore_grasp_snapshot(snapshot)
        self._sync_target_site_to_release_pose()

        self.sampled_target_site_pos, self.sampled_target_site_quat = (
            self._get_target_pose()
        )
        self.sampled_object_yaw = float(
            self._quat_to_yaw(np.asarray(snapshot["obj_quat"], dtype=np.float64))
        )
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        target_place_body_quat = _normalize_quat(
            self.data.body(
                str(self._get_active_place_info()["body_name"])
            ).xquat.copy()
        )
        self.applied_target_place_yaw = float(self._quat_to_yaw(target_place_body_quat))
        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        target_dist, target_angle, _ = self._get_release_target_alignment()
        self.initial_object_target_dist = target_dist
        self.best_object_target_dist = target_dist
        self.last_release_target_dist = target_dist
        self.last_release_target_angle = target_angle
        self._last_grasp_reset_attempts = int(attempt_count)
        self._last_grasp_init_lift_height = float(snapshot["lift_height"])
        self._last_grasp_init_ee_obj_dist = float(snapshot["ee_obj_dist"])
        self._last_grasp_reset_source = str(reset_source)
        self._reset_ik_state()
        self.sync_visual_frames()
        return self._get_obs()

    def reset_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        seed: int | None = None,
        reset_source: str = "external_gripper_closed_snapshot",
        attempt_count: int = 1,
    ) -> np.ndarray:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        return self._initialize_from_grasp_snapshot(
            snapshot,
            reset_source=reset_source,
            attempt_count=attempt_count,
        )

    def reset_model(self):
        if not getattr(self, "_insert_env_ready", False):
            return super().reset_model()

        snapshot, reset_source, attempt_count = self._sample_grasp_reset_snapshot()
        return self._initialize_from_grasp_snapshot(
            snapshot,
            reset_source=reset_source,
            attempt_count=attempt_count,
        )

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
        if not getattr(self, "_insert_env_ready", False):
            return super()._get_obs_components()

        qpos = self.data.qpos
        qvel = self.data.qvel
        first_object_qposadr = min(
            int(info["qposadr"]) for info in self.object_info.values()
        )
        first_object_dofadr = min(
            int(info["dofadr"]) for info in self.object_info.values()
        )
        robot_qpos = qpos[:first_object_qposadr]
        robot_qvel = qvel[:first_object_dofadr]
        gripper_qpos = qpos[[self.gripL_qadr, self.gripR_qadr]].copy()
        gripper_qvel = qvel[[self.gripL_dadr, self.gripR_dadr]].copy()
        gripper_ctrl = self.data.ctrl[[self.gripL_act_id, self.gripR_act_id]].copy()
        gripper_closed = np.array(
            [1.0 if self.gripper_state == "closed" else 0.0], dtype=np.float64
        )
        release_latched = np.array(
            [1.0 if self.gripper_release_latched else 0.0], dtype=np.float64
        )

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
        place_body = self.data.body(str(self._get_active_place_info()["body_name"]))
        place_body_pos = place_body.xpos.copy()
        place_body_quat = _normalize_quat(place_body.xquat.copy())

        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        target_delta_euler = self._wrap_vector_to_pi(
            _quat_to_euler_xyz(target_quat) - _quat_to_euler_xyz(obj_quat)
        )
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        inside_place_metrics = self._get_inside_place_metrics()

        metrics = np.array(
            [
                target_dist,
                target_angle,
                float(self.success_counter),
                float(self._ik_failure_count),
                float(self.gripper_release_latched),
                float(inside_place_metrics["inside_place"]),
                float(self._target_height_above_place),
            ],
            dtype=np.float64,
        )

        return [
            ("robot_qpos", robot_qpos),
            ("robot_qvel", robot_qvel),
            ("gripper_qpos", gripper_qpos),
            ("gripper_qvel", gripper_qvel),
            ("gripper_ctrl", gripper_ctrl),
            ("gripper_closed", gripper_closed),
            ("gripper_release_latched", release_latched),
            ("object_type", self.object_one_hot[self.active_obj_name]),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("release_target_pos", target_pos),
            ("release_target_quat", target_quat),
            ("place_site_pos", place_site_pos),
            ("place_site_quat", place_site_quat),
            ("place_body_pos", place_body_pos),
            ("place_body_quat", place_body_quat),
            ("object_release_pos_error", obj_target_pos_error),
            ("object_release_rot_error", obj_target_rot_error),
            ("target_delta_euler", target_delta_euler),
            ("ik_target_pos", self._ik_target_pos),
            ("ik_target_quat", self._ik_target_quat),
            ("last_action", self.last_action),
            ("metrics", metrics),
        ]

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(component, dtype=np.float64).reshape(-1)
                for _, component in self._get_obs_components()
            ]
        )
        return obs.astype(np.float32)

    def export_config(self) -> dict:
        config = export_env_config(self, self._get_obs_components())
        config["action"]["controller"] = "standalone_cartesian_ik"
        config["action"]["action_components"] = list(self.ACTION_COMPONENTS)
        config["action"]["cartesian_action_scale_m"] = float(
            self._cartesian_action_scale
        )
        config["action"]["cartesian_rotation_scale_deg"] = float(
            np.rad2deg(self._cartesian_rotation_scale_rad)
        )
        config["action"]["ik_workspace_low"] = self._ik_workspace_low.tolist()
        config["action"]["ik_workspace_high"] = self._ik_workspace_high.tolist()
        config["action"]["ik_position_only"] = bool(self._ik_position_only)
        config["action"]["ik_max_iters"] = int(self._ik_max_iters)
        config["action"]["ik_position_tolerance"] = float(self._ik_position_tolerance)
        config["action"]["ik_rotation_tolerance_deg"] = float(
            np.rad2deg(self._ik_rotation_tolerance_rad)
        )
        config["action"]["ik_damping"] = float(self._ik_damping)
        config["action"]["ik_step_size"] = float(self._ik_step_size)
        config["action"]["ik_max_delta_deg"] = float(np.rad2deg(self._ik_max_delta_rad))
        config["action"]["ik_rotation_weight"] = float(self._ik_rotation_weight)
        config["action"]["ik_random_restarts"] = int(self._ik_random_restarts)
        config["action"]["ik_seed"] = self._ik_seed
        config["action"]["control_interpolation_steps"] = int(
            self._control_interpolation_steps
        )
        config["action"]["max_joint_ctrl_delta_deg"] = float(
            np.rad2deg(self._max_joint_ctrl_delta_rad)
        )
        config["action"]["smooth_cartesian_target"] = bool(
            self._smooth_cartesian_target
        )
        config["action"]["ik_joint_names"] = list(self._ik_solver.joint_names)
        config["action"]["ee_site_name"] = self.ee_site_name
        config["action"]["gripper_policy"] = "release_above_place_heuristic"
        config["action"]["gripper_release_distance"] = float(
            self._gripper_release_distance
        )
        config["action"]["gripper_release_angle_deg"] = float(
            np.rad2deg(self._gripper_release_angle_rad)
        )
        config["action"]["gripper_open_target"] = [0.01, -0.01]
        config["action"]["gripper_closed_target"] = [-0.02, 0.02]
        config["task"]["target_mode"] = "release_target_5cm_above_active_place"
        config["task"]["target_height_above_place"] = float(
            self._target_height_above_place
        )
        config["task"]["reward_phase_before_release"] = (
            "object_distance_to_release_target"
        )
        config["task"]["reward_phase_after_release"] = "inside_place_bonus_only"
        config["task"]["target_place_body_names"] = {
            obj_name: str(info["body_name"])
            for obj_name, info in self.place_info.items()
        }
        config["task"]["target_place_site_names"] = {
            obj_name: str(info["site_name"])
            for obj_name, info in self.place_info.items()
        }
        config["task"]["target_place_randomization"] = {
            "target_x_range": list(self._target_x_range),
            "target_y_range": list(self._target_y_range),
            "target_place_z": float(self._target_place_z),
            "target_place_yaw_range": list(self._target_place_yaw_range),
            "target_min_object_xy_distance": float(
                self._target_min_object_xy_distance
            ),
        }
        config["task"]["grasp_policy_reset"] = {
            "grasp_env_name": self._grasp_env_name,
            "grasp_model_path": str(self._grasp_model_path),
            "grasp_xml_file": str(self._grasp_xml_path),
            "grasp_max_steps": int(self._grasp_max_steps),
            "grasp_attempts_per_reset": int(self._grasp_attempts_per_reset),
            "grasp_success_condition": "actual_gripper_qpos_closed_without_lift",
            "grasp_success_max_lift": self._grasp_success_max_lift,
            "grasp_success_ee_obj_dist": float(self._grasp_success_ee_obj_dist),
            "grasp_success_hold_steps": int(self._grasp_success_hold_steps),
            "grasp_ctrl_close_threshold": float(self._grasp_ctrl_close_threshold),
            "grasp_qpos_close_threshold": float(self._grasp_qpos_close_threshold),
        }
        config["task"]["inside_place_bonus_criterion"] = {
            "box_half_extents_xy": INSIDE_PLACE_BOX_HALF_EXTENTS.astype(
                np.float64
            ).tolist(),
            "triangle_radius_xy": float(INSIDE_PLACE_TRIANGLE_RADIUS),
            "cylinder_radius_xy": float(INSIDE_PLACE_CYLINDER_RADIUS),
            "z_offset_min": float(INSIDE_PLACE_Z_OFFSET_MIN),
            "z_offset_max": float(INSIDE_PLACE_Z_OFFSET_MAX),
        }
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
        place_body = self.data.body(str(self._get_active_place_info()["body_name"]))
        place_body_quat = _normalize_quat(place_body.xquat.copy())
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            obj_pos,
            obj_quat,
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        inside_place_metrics = self._get_inside_place_metrics()

        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_place_pos": place_body.xpos.copy(),
            "target_place_quat": place_body_quat,
            "place_site_pos": place_site_pos,
            "place_site_quat": place_site_quat,
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "obj_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "object_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "release_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "release_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "release_target_height_above_place": float(
                self._target_height_above_place
            ),
            "release_distance_threshold": float(self._gripper_release_distance),
            "release_angle_threshold_deg": float(
                np.rad2deg(self._gripper_release_angle_rad)
            ),
            "gripper_state": self.gripper_state,
            "gripper_qpos": self.data.qpos[
                [self.gripL_qadr, self.gripR_qadr]
            ].copy(),
            "gripper_qvel": self.data.qvel[
                [self.gripL_dadr, self.gripR_dadr]
            ].copy(),
            "gripper_ctrl": self.data.ctrl[
                [self.gripL_act_id, self.gripR_act_id]
            ].copy(),
            "gripper_qpos_close_threshold": float(self._grasp_qpos_close_threshold),
            "gripper_release_latched": bool(self.gripper_release_latched),
            "gripper_should_release": bool(self.last_gripper_should_release),
            "inside_place": bool(inside_place_metrics["inside_place"]),
            "inside_place_xy_ok": bool(inside_place_metrics["inside_place_xy_ok"]),
            "inside_place_z_ok": bool(inside_place_metrics["inside_place_z_ok"]),
            "inside_place_xy_error": np.asarray(
                inside_place_metrics["inside_place_xy_error"], dtype=np.float64
            ),
            "inside_place_z_offset": float(
                inside_place_metrics["inside_place_z_offset"]
            ),
            "inside_place_xy_margin": float(
                inside_place_metrics["inside_place_xy_margin"]
            ),
            "sampled_target_site_pos": self.sampled_target_site_pos.copy(),
            "sampled_target_place_site_pos": self.sampled_target_place_site_pos.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "success_counter": int(self.success_counter),
            "initial_object_target_dist": float(self.initial_object_target_dist),
            "best_object_target_dist": float(self.best_object_target_dist),
            "grasp_reset_attempts": int(self._last_grasp_reset_attempts),
            "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
            "grasp_reset_source": self._last_grasp_reset_source,
            "last_action": self.last_action.copy(),
            "task_mode": "ik_insert_release_above_place_then_inside_bonus",
            **self._get_ik_debug_state(),
        }

    def close(self):
        if self._grasp_env is not None:
            self._grasp_env.close()
            self._grasp_env = None
            self._grasp_policy = None
        return super().close()
