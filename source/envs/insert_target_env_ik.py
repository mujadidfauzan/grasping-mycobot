from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .cartesian_ik import CartesianIKActionMixin
from .config_export import capture_init_config, export_env_config
from .grasping_env import GraspingEnv
from .grasping_env_ik import GraspingEnvIK
from .grasping_env_v1 import GraspingEnvV1
from .grasping_env_v2 import GraspingEnvV2
from .place_above_site_env import PlaceAboveSiteEnv

try:
    from .grasping_env_v3 import GraspingEnvV3
except ModuleNotFoundError:
    GraspingEnvV3 = None

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"
DEFAULT_GRASP_XML_PATH = (
    Path(__file__).resolve().parents[1] / "robot" / "object_lift.xml"
)

GRASP_ENV_REGISTRY = {
    "GraspingEnv": GraspingEnv,
    "GraspingEnvIK": GraspingEnvIK,
    "GraspingEnvV1": GraspingEnvV1,
    "GraspingEnvV2": GraspingEnvV2,
}
if GraspingEnvV3 is not None:
    GRASP_ENV_REGISTRY["GraspingEnvV3"] = GraspingEnvV3


class InsertTargetEnvIK(CartesianIKActionMixin, MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = str(DEFAULT_XML_PATH),
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 1.0,
        reward_target_orient_weight: float = 2.0,
        reward_target_tanh_orient_weight: float = 1.0,
        reward_bonus: float = 10.0,
        success_distance: float = 0.008,
        success_angle_deg: float = 10.0,
        success_steps_required: int = 10,
        terminate_ee_obj_distance: float = 0.05,
        max_episode_steps: int = 100,
        cartesian_action_scale: float = 0.1,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.10, -0.20, 0.02),
        ik_workspace_high: tuple[float, float, float] = (0.35, 0.20, 0.30),
        ik_max_iters: int = 80,
        ik_position_tolerance: float = 1e-3,
        ik_rotation_tolerance_deg: float = 3.0,
        ik_damping: float = 1e-3,
        ik_step_size: float = 0.75,
        ik_max_delta_deg: float = 10.0,
        ik_rotation_weight: float = 0.35,
        ik_random_restarts: int = 0,
        ik_seed: int | None = 0,
        gripper_command_threshold: float = -0.01,
        gripper_open_xy_threshold: float = 0.008,
        gripper_open_z_threshold: float = 0.05,
        gripper_open_angle_deg: float = 10.0,
        target_x_range: tuple[float, float] = (0.0, 0.27),
        target_y_range: tuple[float, float] = (-0.20, 0.20),
        target_place_z: float = 0.025,
        target_z_range: tuple[float, float] | None = None,
        target_place_yaw_range: tuple[float, float] = (-np.pi / 6.0, np.pi / 6.0),
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        grasp_model_path: str | None = None,
        grasp_env_name: str = "GraspingEnvIK",
        grasp_xml_file: str | None = None,
        grasp_max_steps: int = 300,
        grasp_attempts_per_reset: int = 6,
        grasp_deterministic: bool = True,
        grasp_success_min_lift: float = 0.025,
        grasp_success_ee_obj_dist: float = 0.035,
        grasp_success_hold_steps: int = 3,
        grasp_ctrl_close_threshold: float = 0.005,
        grasp_transfer_settle_steps: int = 5,
        allow_grasp_fallback_snapshot: bool = True,
        place_above_model_path: str | None = None,
        place_above_xml_file: str | None = None,
        place_above_max_steps: int = 150,
        place_above_attempts_per_reset: int = 4,
        place_above_deterministic: bool = True,
        place_above_success_distance: float = 0.015,
        place_above_success_ee_obj_dist: float = 0.05,
        place_above_success_hold_steps: int = 10,
        place_above_ctrl_close_threshold: float = 0.005,
        place_above_target_height_above_place: float = 0.04,
        **kwargs,
    ):
        self._init_config = capture_init_config(locals())
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_target_weight,
            reward_target_tanh_weight,
            reward_target_orient_weight,
            reward_target_tanh_orient_weight,
            reward_bonus,
            success_distance,
            success_angle_deg,
            success_steps_required,
            terminate_ee_obj_distance,
            max_episode_steps,
            cartesian_action_scale,
            cartesian_rotation_scale_deg,
            ik_workspace_low,
            ik_workspace_high,
            ik_max_iters,
            ik_position_tolerance,
            ik_rotation_tolerance_deg,
            ik_damping,
            ik_step_size,
            ik_max_delta_deg,
            ik_rotation_weight,
            ik_random_restarts,
            ik_seed,
            gripper_command_threshold,
            gripper_open_xy_threshold,
            gripper_open_z_threshold,
            gripper_open_angle_deg,
            target_x_range,
            target_y_range,
            target_place_z,
            target_z_range,
            target_place_yaw_range,
            ee_site_name,
            target_site_name,
            grasp_model_path,
            grasp_env_name,
            grasp_xml_file,
            grasp_max_steps,
            grasp_attempts_per_reset,
            grasp_deterministic,
            grasp_success_min_lift,
            grasp_success_ee_obj_dist,
            grasp_success_hold_steps,
            grasp_ctrl_close_threshold,
            grasp_transfer_settle_steps,
            allow_grasp_fallback_snapshot,
            place_above_model_path,
            place_above_xml_file,
            place_above_max_steps,
            place_above_attempts_per_reset,
            place_above_deterministic,
            place_above_success_distance,
            place_above_success_ee_obj_dist,
            place_above_success_hold_steps,
            place_above_ctrl_close_threshold,
            place_above_target_height_above_place,
            **kwargs,
        )

        if grasp_model_path is None:
            raise ValueError(
                "InsertTargetEnv requires `grasp_model_path` so the place-above reset "
                "pipeline can start from the grasping policy state."
            )
        if place_above_model_path is None:
            raise ValueError(
                "InsertTargetEnv requires `place_above_model_path` so reset can start "
                "from the trained PlaceAboveSite policy state."
            )

        grasp_model_path_obj = Path(grasp_model_path).expanduser()
        if not grasp_model_path_obj.is_absolute():
            grasp_model_path_obj = grasp_model_path_obj.resolve()
        if not grasp_model_path_obj.exists():
            raise FileNotFoundError(f"Grasp model not found: {grasp_model_path_obj}")

        place_above_model_path_obj = Path(place_above_model_path).expanduser()
        if not place_above_model_path_obj.is_absolute():
            place_above_model_path_obj = place_above_model_path_obj.resolve()
        if not place_above_model_path_obj.exists():
            raise FileNotFoundError(
                f"Place-above model not found: {place_above_model_path_obj}"
            )

        grasp_xml_path_obj = (
            DEFAULT_GRASP_XML_PATH
            if grasp_xml_file is None
            else Path(grasp_xml_file).expanduser()
        )
        if not grasp_xml_path_obj.is_absolute():
            grasp_xml_path_obj = grasp_xml_path_obj.resolve()
        if not grasp_xml_path_obj.exists():
            raise FileNotFoundError(f"Grasp XML not found: {grasp_xml_path_obj}")

        resolved_place_above_xml = (
            xml_file if place_above_xml_file is None else place_above_xml_file
        )
        place_above_xml_path_obj = Path(resolved_place_above_xml).expanduser()
        if not place_above_xml_path_obj.is_absolute():
            place_above_xml_path_obj = place_above_xml_path_obj.resolve()
        if not place_above_xml_path_obj.exists():
            raise FileNotFoundError(
                f"Place-above XML not found: {place_above_xml_path_obj}"
            )

        if grasp_env_name not in GRASP_ENV_REGISTRY:
            supported = ", ".join(sorted(GRASP_ENV_REGISTRY))
            raise ValueError(
                f"Unsupported grasp env `{grasp_env_name}`. Expected one of: {supported}"
            )

        self._reward_target_weight = float(reward_target_weight)
        self._reward_target_tanh_weight = float(reward_target_tanh_weight)
        self._reward_target_orient_weight = float(reward_target_orient_weight)
        self._reward_target_tanh_orient_weight = float(reward_target_tanh_orient_weight)
        self._reward_bonus = float(reward_bonus)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_steps_required = int(success_steps_required)
        self._terminate_ee_obj_distance = float(terminate_ee_obj_distance)
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        if self._success_angle_rad <= 0.0:
            raise ValueError("success_angle_deg must be greater than 0.")
        if self._success_steps_required <= 0:
            raise ValueError("success_steps_required must be greater than 0.")
        if self._terminate_ee_obj_distance <= 0.0:
            raise ValueError("terminate_ee_obj_distance must be greater than 0.")
        self.max_episode_steps = int(max_episode_steps)
        self._gripper_command_threshold = float(gripper_command_threshold)
        self._gripper_open_xy_threshold = float(gripper_open_xy_threshold)
        self._gripper_open_z_threshold = float(gripper_open_z_threshold)
        self._gripper_open_angle_rad = np.deg2rad(float(gripper_open_angle_deg))
        if self._gripper_open_xy_threshold <= 0.0:
            raise ValueError("gripper_open_xy_threshold must be greater than 0.")
        if self._gripper_open_z_threshold <= 0.0:
            raise ValueError("gripper_open_z_threshold must be greater than 0.")
        if self._gripper_open_angle_rad <= 0.0:
            raise ValueError("gripper_open_angle_deg must be greater than 0.")
        self._target_x_range = tuple(float(value) for value in target_x_range)
        self._target_y_range = tuple(float(value) for value in target_y_range)
        self._target_place_z = float(target_place_z)
        self._target_z_range = (
            (self._target_place_z, self._target_place_z)
            if target_z_range is None
            else tuple(float(value) for value in target_z_range)
        )
        self._target_place_yaw_range = tuple(
            float(value) for value in target_place_yaw_range
        )
        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)
        if self._target_z_range[0] > self._target_z_range[1]:
            raise ValueError("target_z_range must be ordered as (min_z, max_z).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError(
                "target_place_yaw_range must be ordered as (min_yaw, max_yaw)."
            )

        self._gripper_open_target = np.array([0.01, -0.01], dtype=np.float64)
        self._gripper_closed_target = np.array([-0.02, 0.02], dtype=np.float64)
        self._grasp_model_path = grasp_model_path_obj
        self._grasp_env_name = str(grasp_env_name)
        self._grasp_xml_path = grasp_xml_path_obj
        self._grasp_max_steps = int(grasp_max_steps)
        self._grasp_attempts_per_reset = max(1, int(grasp_attempts_per_reset))
        self._grasp_deterministic = bool(grasp_deterministic)
        self._grasp_success_min_lift = float(grasp_success_min_lift)
        self._grasp_success_ee_obj_dist = float(grasp_success_ee_obj_dist)
        self._grasp_success_hold_steps = max(1, int(grasp_success_hold_steps))
        self._grasp_ctrl_close_threshold = float(grasp_ctrl_close_threshold)
        self._grasp_transfer_settle_steps = max(0, int(grasp_transfer_settle_steps))
        self._allow_grasp_fallback_snapshot = bool(allow_grasp_fallback_snapshot)

        self._place_above_model_path = place_above_model_path_obj
        self._place_above_xml_path = place_above_xml_path_obj
        self._place_above_max_steps = int(place_above_max_steps)
        self._place_above_attempts_per_reset = max(
            1, int(place_above_attempts_per_reset)
        )
        self._place_above_deterministic = bool(place_above_deterministic)
        self._place_above_success_distance = float(place_above_success_distance)
        self._place_above_success_ee_obj_dist = float(place_above_success_ee_obj_dist)
        self._place_above_success_hold_steps = max(
            1, int(place_above_success_hold_steps)
        )
        self._place_above_ctrl_close_threshold = float(place_above_ctrl_close_threshold)
        self._place_above_target_height_above_place = float(
            place_above_target_height_above_place
        )

        self._place_above_env = None
        self._place_above_policy = None
        self._last_place_above_reset_attempts = 0
        self._last_place_above_reset_source = "uninitialized"
        self._last_place_above_init_target_dist = np.inf
        self._last_place_above_init_ee_obj_dist = np.inf
        self._last_grasp_reset_attempts = 0
        self._last_grasp_init_lift_height = 0.0
        self._last_grasp_init_ee_obj_dist = np.inf
        self._last_grasp_reset_source = "uninitialized"

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            camera_name="watching",
            **kwargs,
        )

        self.object_names = ["box", "cylinder", "triangle"]
        self.object_info: dict[str, dict[str, int | str]] = {}
        self.object_one_hot: dict[str, np.ndarray] = {}
        for index, obj_name in enumerate(self.object_names):
            body_name = f"obj_{obj_name}"
            joint_name = f"obj_{obj_name}_joint"
            site_name = f"obj_{obj_name}_ref"
            geom_name = f"obj_{obj_name}_geom"

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

            self.object_info[obj_name] = {
                "body_name": body_name,
                "joint_name": joint_name,
                "site_name": site_name,
                "geom_name": geom_name,
                "body_id": body_id,
                "joint_id": joint_id,
                "site_id": site_id,
                "geom_id": geom_id,
                "qposadr": int(self.model.jnt_qposadr[joint_id]),
                "dofadr": int(self.model.jnt_dofadr[joint_id]),
            }
            one_hot = np.zeros(len(self.object_names), dtype=np.float64)
            one_hot[index] = 1.0
            self.object_one_hot[obj_name] = one_hot

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

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

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
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site_name
        )
        self.target_body_id = int(self.model.site_bodyid[self.target_site_id])
        self.target_body_name = str(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_body_id)
        )
        self._target_site_local_pos = self.model.site_pos[self.target_site_id].copy()
        self._target_site_local_quat = self._normalize_quat(
            self.model.site_quat[self.target_site_id].copy()
        )
        self._place_site_local_pose_by_object: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        for obj_name, info in self.place_info.items():
            site_id = int(info["site_id"])
            self._place_site_local_pose_by_object[obj_name] = (
                self.model.site_pos[site_id].copy(),
                self._normalize_quat(self.model.site_quat[site_id].copy()),
            )

        self.active_obj_name = self.object_names[0]

        self.gripL_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_10"
        )
        self.gripR_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_11"
        )
        self.gripL_qadr = int(self.model.jnt_qposadr[self.gripL_jid])
        self.gripR_qadr = int(self.model.jnt_qposadr[self.gripR_jid])
        self.gripL_dadr = int(self.model.jnt_dofadr[self.gripL_jid])
        self.gripR_dadr = int(self.model.jnt_dofadr[self.gripR_jid])

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        if self.model.nu < 3:
            raise ValueError(
                "InsertTargetEnv expects arm actuators plus 2 gripper actuators."
            )
        self._arm_ctrl_dim = int(self.model.nu - 2)
        self._setup_cartesian_ik_action(
            xml_file=str(self.fullpath),
            ee_site_name=self.ee_site_name,
            cartesian_action_scale=cartesian_action_scale,
            cartesian_rotation_scale_deg=cartesian_rotation_scale_deg,
            ik_workspace_low=ik_workspace_low,
            ik_workspace_high=ik_workspace_high,
            ik_max_iters=ik_max_iters,
            ik_position_tolerance=ik_position_tolerance,
            ik_rotation_tolerance_deg=ik_rotation_tolerance_deg,
            ik_damping=ik_damping,
            ik_step_size=ik_step_size,
            ik_max_delta_deg=ik_max_delta_deg,
            ik_rotation_weight=ik_rotation_weight,
            ik_random_restarts=ik_random_restarts,
            ik_seed=ik_seed,
        )

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.gripper_release_latched = False
        self.last_gripper_should_open = False
        self.last_gripper_open_xy_error = np.full(2, np.inf, dtype=np.float64)
        self.last_gripper_open_z_error = np.inf
        self.last_gripper_open_angle = np.inf
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_site_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.sampled_target_site_yaw = 0.0
        self.applied_target_site_yaw = 0.0
        self.sampled_target_place_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0

        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    @staticmethod
    def _normalize_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    @staticmethod
    def _quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        wa, xa, ya, za = quat_a
        wb, xb, yb, zb = quat_b
        return np.array(
            [
                wa * wb - xa * xb - ya * yb - za * zb,
                wa * xb + xa * wb + ya * zb - za * yb,
                wa * yb - xa * zb + ya * wb + za * xb,
                wa * zb + xa * yb - ya * xb + za * wb,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _quat_rotate_vector(cls, quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        vec_quat = np.array([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
        rotated = cls._quat_multiply(
            cls._quat_multiply(cls._normalize_quat(quat), vec_quat),
            cls._quat_conjugate(cls._normalize_quat(quat)),
        )
        return rotated[1:]

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        quat = InsertTargetEnvIK._normalize_quat(quat)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half_yaw = float(yaw) / 2.0
        return np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    def _get_site_quat(self, site_name: str) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site(site_name).xmat)
        return self._normalize_quat(quat)

    def _get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site(site_name).xpos.copy(), self._get_site_quat(site_name)

    def _rotation_vector(
        self, source_quat: np.ndarray, target_quat: np.ndarray
    ) -> np.ndarray:
        source_quat = self._normalize_quat(source_quat)
        target_quat = self._normalize_quat(target_quat)
        delta = self._quat_multiply(target_quat, self._quat_conjugate(source_quat))
        delta = self._normalize_quat(delta)
        if delta[0] < 0.0:
            delta = -delta

        xyz = delta[1:]
        sin_half = np.linalg.norm(xyz)
        if sin_half < 1e-12:
            return np.zeros(3, dtype=np.float64)

        angle = 2.0 * np.arctan2(sin_half, np.clip(delta[0], -1.0, 1.0))
        axis = xyz / sin_half
        return axis * angle

    def _get_pose_error(
        self,
        source_pos: np.ndarray,
        source_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_error = np.asarray(target_pos, dtype=np.float64) - np.asarray(
            source_pos, dtype=np.float64
        )
        rot_error = self._rotation_vector(source_quat, target_quat)
        return pos_error, rot_error

    def _get_active_obj_info(self) -> dict[str, int | str]:
        return self.object_info[self.active_obj_name]

    def _get_active_place_info(self) -> dict[str, int | str]:
        return self.place_info[self.active_obj_name]

    def _count_contacts_between_geoms(self, geom1_id: int, geom2_id: int) -> int:
        contact_count = 0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            if (int(contact.geom1) == geom1_id and int(contact.geom2) == geom2_id) or (
                int(contact.geom1) == geom2_id and int(contact.geom2) == geom1_id
            ):
                contact_count += 1
        return contact_count

    def _get_pose_in_body_frame(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_quat = self._normalize_quat(np.asarray(body_quat, dtype=np.float64))
        body_quat_conj = self._quat_conjugate(body_quat)
        local_pos = self._quat_rotate_vector(
            body_quat_conj,
            np.asarray(world_pos, dtype=np.float64)
            - np.asarray(body_pos, dtype=np.float64),
        )
        local_quat = self._normalize_quat(
            self._quat_multiply(
                body_quat_conj, np.asarray(world_quat, dtype=np.float64)
            )
        )
        return local_pos, local_quat

    def _get_insertion_metrics(self) -> dict[str, np.ndarray | float | int]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        active_obj_info = self._get_active_obj_info()
        active_place_info = self._get_active_place_info()
        place_body_name = str(active_place_info["body_name"])
        place_body = self.data.body(place_body_name)
        place_body_pos = place_body.xpos.copy()
        place_body_quat = self._normalize_quat(place_body.xquat.copy())

        obj_local_pos, obj_local_quat = self._get_pose_in_body_frame(
            obj_pos,
            obj_quat,
            place_body_pos,
            place_body_quat,
        )
        target_local_pos, target_local_quat = self._place_site_local_pose_by_object[
            self.active_obj_name
        ]
        local_pos_error, local_rot_error = self._get_pose_error(
            obj_local_pos,
            obj_local_quat,
            target_local_pos,
            target_local_quat,
        )

        radial_error = float(np.linalg.norm(local_pos_error[:2]))
        height_error = float(local_pos_error[2])
        rot_error = float(np.linalg.norm(local_rot_error))
        object_place_contact_count = self._count_contacts_between_geoms(
            int(active_obj_info["geom_id"]),
            int(active_place_info["geom_id"]),
        )
        pose_aligned = bool(
            radial_error < self._success_distance
            and abs(height_error) < self._success_distance
            and rot_error < self._success_angle_rad
        )

        return {
            "object_local_pos": obj_local_pos,
            "object_local_quat": obj_local_quat,
            "target_local_pos": target_local_pos.copy(),
            "target_local_quat": target_local_quat.copy(),
            "object_target_local_pos_error": local_pos_error,
            "object_target_local_rot_error": local_rot_error,
            "object_target_local_radial_error": radial_error,
            "object_target_local_height_error": height_error,
            "object_target_local_angle_error": rot_error,
            "object_place_contact_count": int(object_place_contact_count),
            "object_place_in_contact": int(object_place_contact_count > 0),
            "insert_pose_aligned": int(pose_aligned),
            "inserted_contact_candidate": int(
                pose_aligned and object_place_contact_count > 0
            ),
        }

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_active_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_place_info()["site_name"]))

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _get_target_pose_alignment(self) -> tuple[float, float, bool]:
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
        target_pose_aligned = bool(
            target_dist < self._success_distance
            and target_angle < self._success_angle_rad
        )
        return target_dist, target_angle, target_pose_aligned

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[-2:] = self._gripper_closed_target

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[-2:] = self._gripper_open_target

    def _update_gripper_state_from_target(self, target: np.ndarray) -> None:
        self.gripper_state = "closed" if target[-2] < target[-1] else "open"

    def _set_active_place_visual(self) -> None:
        for obj_name, info in self.place_info.items():
            rgba = self.place_geom_rgba[obj_name].copy()
            rgba[3] = (
                self.place_geom_rgba[obj_name][3]
                if obj_name == self.active_obj_name
                else 0.0
            )
            self.model.geom_rgba[int(info["geom_id"])] = rgba

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
                self.model.body_pos[body_id] = active_place_pos
                self.model.body_quat[body_id] = active_place_quat
            else:
                self.model.body_pos[body_id] = np.array(
                    [2.0 + index, 2.0, 0.2],
                    dtype=np.float64,
                )
                self.model.body_quat[body_id] = identity_quat

    def _pose_to_body_transform(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        local_pos: np.ndarray,
        local_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        world_pos = np.asarray(world_pos, dtype=np.float64)
        world_quat = self._normalize_quat(np.asarray(world_quat, dtype=np.float64))
        local_pos = np.asarray(local_pos, dtype=np.float64)
        local_quat = self._normalize_quat(np.asarray(local_quat, dtype=np.float64))
        body_quat = self._normalize_quat(
            self._quat_multiply(world_quat, self._quat_conjugate(local_quat))
        )
        body_pos = world_pos - self._quat_rotate_vector(body_quat, local_pos)
        return body_pos, body_quat

    def _target_site_pose_to_target_body_pose(
        self, target_site_pos: np.ndarray, target_site_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._pose_to_body_transform(
            target_site_pos,
            target_site_quat,
            self._target_site_local_pos,
            self._target_site_local_quat,
        )

    def _target_site_pose_to_place_body_pose(
        self, target_site_pos: np.ndarray, target_site_quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        local_pos, local_quat = self._place_site_local_pose_by_object[
            self.active_obj_name
        ]

        return self._pose_to_body_transform(
            target_site_pos,
            target_site_quat,
            local_pos,
            local_quat,
        )

    def _set_target_site_pose_in_model(
        self, target_site_pos: np.ndarray, target_site_quat: np.ndarray
    ) -> None:
        body_pos, body_quat = self._target_site_pose_to_target_body_pose(
            target_site_pos,
            target_site_quat,
        )
        self.model.body_pos[self.target_body_id] = body_pos
        self.model.body_quat[self.target_body_id] = body_quat

    def _sync_target_site_to_active_place(self) -> None:
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
        self._set_target_site_pose_in_model(place_site_pos, place_site_quat)

    def _sample_target_site_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        target_site_pos = np.array(
            [
                self.np_random.uniform(*self._target_x_range),
                self.np_random.uniform(*self._target_y_range),
                self.np_random.uniform(*self._target_z_range),
            ],
            dtype=np.float64,
        )
        target_site_yaw = float(self.np_random.uniform(*self._target_place_yaw_range))
        target_site_quat = self._yaw_to_quat(target_site_yaw)
        return target_site_pos, target_site_quat, target_site_yaw

    def _ensure_place_above_policy_loaded(self) -> None:
        if self._place_above_env is not None and self._place_above_policy is not None:
            return

        try:
            from stable_baselines3 import SAC
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "InsertTargetEnv requires stable-baselines3 to load the place-above policy."
            ) from exc

        self._place_above_env = PlaceAboveSiteEnv(
            xml_file=str(self._place_above_xml_path),
            render_mode=None,
            grasp_model_path=str(self._grasp_model_path),
            grasp_env_name=self._grasp_env_name,
            grasp_xml_file=str(self._grasp_xml_path),
            grasp_max_steps=self._grasp_max_steps,
            grasp_attempts_per_reset=self._grasp_attempts_per_reset,
            grasp_deterministic=self._grasp_deterministic,
            grasp_success_min_lift=self._grasp_success_min_lift,
            grasp_success_ee_obj_dist=self._grasp_success_ee_obj_dist,
            grasp_success_hold_steps=self._grasp_success_hold_steps,
            grasp_ctrl_close_threshold=self._grasp_ctrl_close_threshold,
            grasp_transfer_settle_steps=self._grasp_transfer_settle_steps,
            allow_grasp_fallback_snapshot=self._allow_grasp_fallback_snapshot,
        )
        self._place_above_policy = SAC.load(
            str(self._place_above_model_path),
            env=self._place_above_env,
            device="auto",
        )

    def _get_place_above_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        place_above_env = self._place_above_env
        assert place_above_env is not None

        active_obj_name = str(place_above_env.active_obj_name)
        info = place_above_env.object_info[active_obj_name]
        site_name = str(info["site_name"])

        obj_pos = place_above_env.data.site(site_name).xpos.copy()
        obj_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(obj_quat, place_above_env.data.site(site_name).xmat)
        return obj_pos, self._normalize_quat(obj_quat)

    def _get_place_above_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        place_above_env = self._place_above_env
        assert place_above_env is not None

        ee_pos = place_above_env.data.site(self.ee_site_name).xpos.copy()
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, place_above_env.data.site(self.ee_site_name).xmat)
        return ee_pos, self._normalize_quat(ee_quat)

    def _get_place_above_object_speed(self) -> float:
        place_above_env = self._place_above_env
        assert place_above_env is not None

        active_obj_name = str(place_above_env.active_obj_name)
        info = place_above_env.object_info[active_obj_name]
        dofadr = int(info["dofadr"])
        return float(np.linalg.norm(place_above_env.data.qvel[dofadr : dofadr + 3]))

    def _capture_place_above_snapshot(self) -> dict:
        place_above_env = self._place_above_env
        assert place_above_env is not None

        obj_pos, obj_quat = self._get_place_above_obj_pose()
        ee_pos, ee_quat = self._get_place_above_ee_pose()
        target_pos, target_quat = place_above_env._get_target_pose()
        place_body_name = str(place_above_env._get_active_place_info()["body_name"])
        target_place_pos = place_above_env.data.body(place_body_name).xpos.copy()
        target_place_quat = self._normalize_quat(
            place_above_env.data.body(place_body_name).xquat.copy()
        )

        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        obj_target_pos_error, _ = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        target_dist = float(np.linalg.norm(obj_target_pos_error))

        return {
            "qpos": place_above_env.data.qpos.copy(),
            "qvel": place_above_env.data.qvel.copy(),
            "ctrl": place_above_env.data.ctrl.copy(),
            "active_object": str(place_above_env.active_obj_name),
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "ee_obj_dist": ee_obj_dist,
            "target_dist": target_dist,
            "target_place_pos": target_place_pos,
            "target_place_quat": target_place_quat,
            "object_speed": self._get_place_above_object_speed(),
            "gripper_ctrl": place_above_env.data.ctrl[-2:].copy(),
            "success_counter": int(getattr(place_above_env, "success_counter", 0)),
            "nested_grasp_reset_source": str(
                getattr(place_above_env, "_last_grasp_reset_source", "unknown")
            ),
            "nested_grasp_reset_attempts": int(
                getattr(place_above_env, "_last_grasp_reset_attempts", 0)
            ),
            "nested_grasp_init_lift_height": float(
                getattr(place_above_env, "_last_grasp_init_lift_height", 0.0)
            ),
            "nested_grasp_init_ee_obj_dist": float(
                getattr(place_above_env, "_last_grasp_init_ee_obj_dist", np.inf)
            ),
        }

    def _is_good_place_above_snapshot(self, snapshot: dict) -> bool:
        gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
        is_closed = bool(
            gripper_ctrl[0] < -self._place_above_ctrl_close_threshold
            and gripper_ctrl[1] > self._place_above_ctrl_close_threshold
        )
        return bool(
            is_closed
            and float(snapshot["target_dist"]) <= self._place_above_success_distance
            and float(snapshot["ee_obj_dist"]) <= self._place_above_success_ee_obj_dist
        )

    def _score_place_above_snapshot(self, snapshot: dict) -> float:
        gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
        is_closed = float(
            gripper_ctrl[0] < -self._place_above_ctrl_close_threshold
            and gripper_ctrl[1] > self._place_above_ctrl_close_threshold
        )
        return (
            -6.0 * float(snapshot["target_dist"])
            - 2.0 * float(snapshot["ee_obj_dist"])
            - 0.2 * float(snapshot["object_speed"])
            + 0.05 * is_closed
            + 0.01 * float(snapshot["success_counter"])
        )

    def _sample_place_above_reset_snapshot(
        self,
        target_place_pos: np.ndarray,
        target_place_quat: np.ndarray,
    ) -> tuple[dict, str, int]:
        self._ensure_place_above_policy_loaded()
        place_above_env = self._place_above_env
        place_above_policy = self._place_above_policy
        assert place_above_env is not None
        assert place_above_policy is not None

        best_snapshot: dict | None = None
        best_score = -np.inf

        for attempt in range(1, self._place_above_attempts_per_reset + 1):
            place_above_seed = int(self.np_random.integers(0, 2**31 - 1))
            observation, _ = place_above_env.reset(seed=place_above_seed)
            consecutive_good_steps = 0

            for _ in range(self._place_above_max_steps):
                action, _ = place_above_policy.predict(
                    observation,
                    deterministic=self._place_above_deterministic,
                )
                observation, _reward, terminated, truncated, _info = (
                    place_above_env.step(action)
                )

                snapshot = self._capture_place_above_snapshot()
                snapshot_score = self._score_place_above_snapshot(snapshot)
                if snapshot_score > best_score:
                    best_score = snapshot_score
                    best_snapshot = snapshot

                if self._is_good_place_above_snapshot(snapshot):
                    consecutive_good_steps += 1
                else:
                    consecutive_good_steps = 0

                if consecutive_good_steps >= self._place_above_success_hold_steps:
                    return snapshot, "place_above_success", attempt

                if terminated or truncated:
                    break

        if best_snapshot is None:
            raise RuntimeError(
                "Failed to obtain a valid place-above state from the PlaceAboveSite policy. "
                "Try increasing place_above_max_steps or place_above_attempts_per_reset."
            )

        return (
            best_snapshot,
            "place_above_fallback_best_snapshot",
            self._place_above_attempts_per_reset,
        )

    def _restore_place_above_snapshot(self, snapshot: dict) -> None:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        ctrl = np.asarray(snapshot["ctrl"], dtype=np.float64).copy()
        if ctrl.shape != self.data.ctrl.shape:
            raise ValueError(
                "Transferred ctrl shape does not match InsertTargetEnv scene. "
                f"Expected {self.data.ctrl.shape}, got {ctrl.shape}."
            )

        self.active_obj_name = str(snapshot["active_object"])
        self.sampled_target_place_pos = np.asarray(
            snapshot["target_place_pos"], dtype=np.float64
        ).copy()
        self.sampled_target_place_quat = self._normalize_quat(
            np.asarray(snapshot["target_place_quat"], dtype=np.float64)
        )
        self.sampled_target_place_yaw = float(
            self._quat_to_yaw(self.sampled_target_place_quat)
        )
        self._set_place_poses_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()

        place_above_env = self._place_above_env
        assert place_above_env is not None
        source_qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
        source_qvel = np.asarray(snapshot["qvel"], dtype=np.float64)
        source_model = place_above_env.model
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
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)

        if self._grasp_transfer_settle_steps > 0:
            settle_ctrl = self.data.ctrl.copy()
            settle_ctrl = np.clip(settle_ctrl, self._ctrl_low, self._ctrl_high)
            for _ in range(self._grasp_transfer_settle_steps):
                self.do_simulation(settle_ctrl, 1)

    def step(self, action):
        self.current_step += 1

        action, target_ctrl, _ik_result = self._cartesian_action_to_target_ctrl(action)

        insertion_metrics = self._get_insertion_metrics()
        pos_error = np.asarray(
            insertion_metrics["object_target_local_pos_error"], dtype=np.float64
        )
        angle_error = float(insertion_metrics["object_target_local_angle_error"])
        xy_error = np.abs(pos_error[:2])
        z_error = float(abs(pos_error[2]))
        should_open = bool(
            np.all(xy_error < self._gripper_open_xy_threshold)
            and z_error < self._gripper_open_z_threshold
            and angle_error < self._gripper_open_angle_rad
        )

        self.gripper_release_latched = bool(self.gripper_release_latched or should_open)
        self.last_gripper_should_open = should_open
        self.last_gripper_open_xy_error = xy_error.astype(np.float64)
        self.last_gripper_open_z_error = z_error
        self.last_gripper_open_angle = angle_error

        if self.gripper_release_latched:
            self._set_open_gripper_target(target_ctrl)
        else:
            self._set_closed_gripper_target(target_ctrl)

        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(target_ctrl)
        self.do_simulation(target_ctrl, self.frame_skip)
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        reward, reward_info = self._get_rew()
        terminated_success = self.success_counter >= self._success_steps_required
        terminated_ee_obj_far = bool(
            float(reward_info["ee_object_dist"]) >= self._terminate_ee_obj_distance
            and not bool(reward_info["target_pose_aligned"])
        )
        if terminated_success:
            print(f"Episode terminated with success at step {self.current_step}.")
        terminated = terminated_ee_obj_far
        truncated = self.current_step >= self.max_episode_steps
        reward_info.update(
            terminated_success=int(terminated_success),
            terminated_ee_obj_far=int(terminated_ee_obj_far),
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self) -> tuple[float, dict]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_pos, ee_quat = self._get_ee_pose()

        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        target_dist, target_angle, target_pose_aligned = (
            self._get_target_pose_alignment()
        )
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        insertion_metrics = self._get_insertion_metrics()
        moved_away = bool(
            ee_obj_dist >= self._terminate_ee_obj_distance and not target_pose_aligned
        )
        reward_bonus = 0.0
        if target_pose_aligned:
            reward_bonus = self._reward_bonus
        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / 0.05))
        ) * self._reward_target_tanh_weight
        reward_target_orient = -target_angle * self._reward_target_orient_weight
        reward_target_tanh_orient = (
            1.0 - float(np.tanh(target_angle / 0.5))
        ) * self._reward_target_tanh_orient_weight

        if target_pose_aligned:
            self.success_counter += 1
        else:
            self.success_counter = 0

        reward = (
            reward_target
            + reward_target_tanh
            + reward_target_orient
            + reward_target_tanh_orient
            + reward_bonus
        )

        reward_info = {
            "ee_object_dist": ee_obj_dist,
            "object_target_dist": target_dist,
            "object_target_rot_error": target_angle,
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_target_orient": float(reward_target_orient),
            "reward_target_tanh_orient": float(reward_target_tanh_orient),
            "reward_bonus": float(reward_bonus),
            "target_pose_aligned": int(target_pose_aligned),
            "moved_away": int(moved_away),
            "gripper_open": int(self.gripper_state == "open"),
            "gripper_release_latched": int(self.gripper_release_latched),
            "gripper_should_open": int(self.last_gripper_should_open),
            "gripper_open_x_error": float(self.last_gripper_open_xy_error[0]),
            "gripper_open_y_error": float(self.last_gripper_open_xy_error[1]),
            "gripper_open_z_error": float(self.last_gripper_open_z_error),
            "gripper_open_angle_error": float(self.last_gripper_open_angle),
            "object_target_local_radial_error": float(
                insertion_metrics["object_target_local_radial_error"]
            ),
            "object_target_local_height_error": float(
                insertion_metrics["object_target_local_height_error"]
            ),
            "object_place_contact_count": int(
                insertion_metrics["object_place_contact_count"]
            ),
            "inserted_contact_candidate": int(
                insertion_metrics["inserted_contact_candidate"]
            ),
        }

        return float(reward), reward_info

    def reset_model(self):
        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_release_latched = False
        self.last_gripper_should_open = False
        self.last_gripper_open_xy_error = np.full(2, np.inf, dtype=np.float64)
        self.last_gripper_open_z_error = np.inf
        self.last_gripper_open_angle = np.inf
        (
            self.sampled_target_site_pos,
            self.sampled_target_site_quat,
            self.sampled_target_site_yaw,
        ) = self._sample_target_site_pose()
        (
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        ) = self._target_site_pose_to_place_body_pose(
            self.sampled_target_site_pos,
            self.sampled_target_site_quat,
        )
        self.sampled_target_place_yaw = float(self.sampled_target_site_yaw)

        snapshot, reset_source, attempt_count = self._sample_place_above_reset_snapshot(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._restore_place_above_snapshot(snapshot)
        closed_ctrl = self.data.ctrl.copy()
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(self.data.ctrl)
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)
        self.sampled_target_site_pos, self.sampled_target_site_quat = (
            self._get_site_pose(self.target_site_name)
        )
        self.sampled_target_site_yaw = float(
            self._quat_to_yaw(self.sampled_target_site_quat)
        )

        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        self.sampled_object_yaw = float(
            self._quat_to_yaw(np.asarray(snapshot["obj_quat"], dtype=np.float64))
        )
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        target_place_body_quat = self._normalize_quat(
            self.data.body(str(self._get_active_place_info()["body_name"])).xquat.copy()
        )
        self.applied_target_site_yaw = float(
            self._quat_to_yaw(self._get_target_pose()[1])
        )
        self.applied_target_place_yaw = float(self._quat_to_yaw(target_place_body_quat))

        self._last_place_above_reset_attempts = int(attempt_count)
        self._last_place_above_reset_source = str(reset_source)
        self._last_place_above_init_target_dist = float(snapshot["target_dist"])
        self._last_place_above_init_ee_obj_dist = float(snapshot["ee_obj_dist"])
        self._last_grasp_reset_source = str(snapshot["nested_grasp_reset_source"])
        self._last_grasp_reset_attempts = int(snapshot["nested_grasp_reset_attempts"])
        self._last_grasp_init_lift_height = float(
            snapshot["nested_grasp_init_lift_height"]
        )
        self._last_grasp_init_ee_obj_dist = float(
            snapshot["nested_grasp_init_ee_obj_dist"]
        )
        self._reset_cartesian_ik_state()

        return self._get_obs()

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
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
        gripper_ctrl = self.data.ctrl[-2:].copy()
        gripper_closed = np.array(
            [1.0 if self.gripper_state == "closed" else 0.0], dtype=np.float64
        )

        ee_pos, ee_quat = self._get_ee_pose()
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

        return [
            ("robot_qpos", robot_qpos),
            ("robot_qvel", robot_qvel),
            ("gripper_qpos", gripper_qpos),
            ("gripper_qvel", gripper_qvel),
            ("gripper_ctrl", gripper_ctrl),
            ("gripper_closed", gripper_closed),
            ("object_type", self.object_one_hot[self.active_obj_name]),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("object_target_dist", np.array([target_dist], dtype=np.float64)),
            ("object_target_angle", np.array([target_angle], dtype=np.float64)),
        ]

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(component, dtype=np.float64)
                for _, component in self._get_obs_components()
            ]
        )
        return obs.astype(np.float32)

    def export_config(self) -> dict:
        config = export_env_config(self, self._get_obs_components())
        self._append_cartesian_ik_config(config)
        config["action"]["gripper_policy"] = "manual_release_heuristic"
        config["action"]["gripper_open_xy_threshold"] = float(
            self._gripper_open_xy_threshold
        )
        config["action"]["gripper_open_z_threshold"] = float(
            self._gripper_open_z_threshold
        )
        config["action"]["gripper_open_angle_deg"] = float(
            np.rad2deg(self._gripper_open_angle_rad)
        )
        config["action"]["gripper_open_target"] = self._gripper_open_target.tolist()
        config["action"]["gripper_closed_target"] = self._gripper_closed_target.tolist()
        config["reward"]["params"]["reward_target_weight"] = float(
            self._reward_target_weight
        )
        config["reward"]["params"]["reward_target_tanh_weight"] = float(
            self._reward_target_tanh_weight
        )
        config["reward"]["params"]["reward_target_orient_weight"] = float(
            self._reward_target_orient_weight
        )
        config["reward"]["params"]["reward_target_tanh_orient_weight"] = float(
            self._reward_target_tanh_orient_weight
        )
        config["task"]["termination_enabled"] = True
        config["task"]["terminate_ee_obj_distance"] = float(
            self._terminate_ee_obj_distance
        )
        config["task"][
            "target_mode"
        ] = "object_insert_into_xml_target_site_synced_to_active_place_site"
        config["task"]["target_site_name"] = self.target_site_name
        config["task"]["target_body_name"] = self.target_body_name
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
            "target_z_range": list(self._target_z_range),
            "target_place_yaw_range": list(self._target_place_yaw_range),
        }
        config["task"]["grasp_policy_reset"] = {
            "grasp_env_name": self._grasp_env_name,
            "grasp_model_path": str(self._grasp_model_path),
            "grasp_xml_file": str(self._grasp_xml_path),
            "grasp_max_steps": int(self._grasp_max_steps),
            "grasp_attempts_per_reset": int(self._grasp_attempts_per_reset),
            "grasp_success_min_lift": float(self._grasp_success_min_lift),
            "grasp_success_ee_obj_dist": float(self._grasp_success_ee_obj_dist),
            "grasp_success_hold_steps": int(self._grasp_success_hold_steps),
        }
        config["task"]["place_above_policy_reset"] = {
            "place_above_model_path": str(self._place_above_model_path),
            "place_above_xml_file": str(self._place_above_xml_path),
            "place_above_max_steps": int(self._place_above_max_steps),
            "place_above_attempts_per_reset": int(self._place_above_attempts_per_reset),
            "place_above_success_distance": float(self._place_above_success_distance),
            "place_above_success_ee_obj_dist": float(
                self._place_above_success_ee_obj_dist
            ),
            "place_above_success_hold_steps": int(self._place_above_success_hold_steps),
            "place_above_target_height_above_place": float(
                self._place_above_target_height_above_place
            ),
        }
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        target_body_quat = self._normalize_quat(
            self.data.body(self.target_body_name).xquat.copy()
        )
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        insertion_metrics = self._get_insertion_metrics()

        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_body_pos": self.data.body(self.target_body_name).xpos.copy(),
            "target_body_quat": target_body_quat,
            "target_place_pos": self.data.body(
                str(self._get_active_place_info()["body_name"])
            ).xpos.copy(),
            "target_place_quat": self._normalize_quat(
                self.data.body(
                    str(self._get_active_place_info()["body_name"])
                ).xquat.copy()
            ),
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "obj_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "object_local_pos": np.asarray(
                insertion_metrics["object_local_pos"], dtype=np.float64
            ).copy(),
            "object_local_quat": np.asarray(
                insertion_metrics["object_local_quat"], dtype=np.float64
            ).copy(),
            "target_local_pos": np.asarray(
                insertion_metrics["target_local_pos"], dtype=np.float64
            ).copy(),
            "target_local_quat": np.asarray(
                insertion_metrics["target_local_quat"], dtype=np.float64
            ).copy(),
            "object_target_local_pos_error": np.asarray(
                insertion_metrics["object_target_local_pos_error"], dtype=np.float64
            ).copy(),
            "object_target_local_rot_error": np.asarray(
                insertion_metrics["object_target_local_rot_error"], dtype=np.float64
            ).copy(),
            "object_target_local_radial_error": float(
                insertion_metrics["object_target_local_radial_error"]
            ),
            "object_target_local_height_error": float(
                insertion_metrics["object_target_local_height_error"]
            ),
            "object_target_local_angle_error": float(
                insertion_metrics["object_target_local_angle_error"]
            ),
            "object_place_contact_count": int(
                insertion_metrics["object_place_contact_count"]
            ),
            "object_place_in_contact": bool(
                insertion_metrics["object_place_in_contact"]
            ),
            "insert_pose_aligned": bool(insertion_metrics["insert_pose_aligned"]),
            "inserted_contact_candidate": bool(
                insertion_metrics["inserted_contact_candidate"]
            ),
            "success_angle_rad": float(self._success_angle_rad),
            "success_angle_deg": float(np.rad2deg(self._success_angle_rad)),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "target_site_yaw": float(self._quat_to_yaw(target_quat)),
            "target_body_yaw": float(self._quat_to_yaw(target_body_quat)),
            "sampled_target_site_yaw": float(self.sampled_target_site_yaw),
            "applied_target_site_yaw": float(self.applied_target_site_yaw),
            "target_place_yaw": float(
                self._quat_to_yaw(
                    self.data.body(
                        str(self._get_active_place_info()["body_name"])
                    ).xquat.copy()
                )
            ),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "gripper_state": self.gripper_state,
            "gripper_release_latched": bool(self.gripper_release_latched),
            "gripper_should_open": bool(self.last_gripper_should_open),
            "gripper_open_xy_error": self.last_gripper_open_xy_error.copy(),
            "gripper_open_z_error": float(self.last_gripper_open_z_error),
            "gripper_open_angle_error": float(self.last_gripper_open_angle),
            "success_counter": int(self.success_counter),
            "last_action": self.last_action.copy(),
            "grasp_reset_attempts": int(self._last_grasp_reset_attempts),
            "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
            "grasp_reset_source": self._last_grasp_reset_source,
            "place_above_reset_attempts": int(self._last_place_above_reset_attempts),
            "place_above_reset_source": self._last_place_above_reset_source,
            "place_above_init_target_dist": float(
                self._last_place_above_init_target_dist
            ),
            "place_above_init_ee_obj_dist": float(
                self._last_place_above_init_ee_obj_dist
            ),
            "reward_target_weight": float(self._reward_target_weight),
            "reward_target_tanh_weight": float(self._reward_target_tanh_weight),
            "reward_target_orient_weight": float(self._reward_target_orient_weight),
            "reward_target_tanh_orient_weight": float(
                self._reward_target_tanh_orient_weight
            ),
            "terminate_ee_obj_distance": float(self._terminate_ee_obj_distance),
            "ee_obj_too_far": bool(
                np.linalg.norm(ee_obj_pos_error) >= self._terminate_ee_obj_distance
            ),
            "termination_enabled": True,
            "task_mode": "object_insert_into_xml_target_site_synced_to_active_place_site",
            **self._get_cartesian_ik_debug_state(),
        }

    def close(self):
        if self._place_above_env is not None:
            self._place_above_env.close()
            self._place_above_env = None
            self._place_above_policy = None
        return super().close()

    def render(self):
        return super().render()
