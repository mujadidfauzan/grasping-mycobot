from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from script.inverse_kinematics import (
    IKResult,
    MyCobotIK,
    _normalize_quat,
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)

from .config_export import capture_init_config, export_env_config
from .grasping_env_ik import GraspingEnvIK

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"
DEFAULT_GRASP_XML_PATH = (
    Path(__file__).resolve().parents[1] / "robot" / "object_lift.xml"
)
DEFAULT_GRASP_MODEL_PATH = PROJECT_ROOT / "logs" / "models" / "grasp-best.zip"


class InsertTargetEnvIK(MujocoEnv, utils.EzPickle):
    """Box insert/place-above task controlled by 6-DoF Cartesian IK actions.

    Reset starts from a lifted object state produced by a trained GraspingEnvIK
    policy. The policy action controls only the end-effector Cartesian delta;
    the gripper is held closed during training. For evaluation, call
    ``set_eval_gripper_open(True)`` or ``open_gripper_for_eval()`` after the
    object is aligned above the place.
    """

    ACTION_COMPONENTS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")

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
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 2.0,
        reward_orientation_weight: float = 1.5,
        reward_orientation_tanh_weight: float = 0.8,
        reward_success_bonus: float = 20.0,
        control_penalty_weight: float = 0.001,
        target_tanh_scale: float = 0.05,
        orientation_tanh_scale: float = 0.5,
        success_distance: float = 0.01,
        success_angle_deg: float = 10.0,
        success_steps_required: int = 5,
        max_episode_steps: int = 150,
        terminate_ee_obj_distance: float = 0.08,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.24, 0.02),
        ik_workspace_high: tuple[float, float, float] = (0.36, 0.24, 0.45),
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
        target_height_above_place: float = 0.035,
        target_x_range: tuple[float, float] = (0.15, 0.27),
        target_y_range: tuple[float, float] = (-0.14, 0.14),
        target_place_z: float = 0.0,
        target_place_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        min_initial_object_target_distance: float = 0.08,
        target_resample_attempts: int = 100,
        object_name: str = "box",
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        target_body_name: str = "target_body",
        place_body_name: str = "cube_place",
        place_site_name: str = "cube_place_site",
        place_geom_name: str = "cube_place_geom",
        grasp_model_path: str | None = (
            str(DEFAULT_GRASP_MODEL_PATH) if DEFAULT_GRASP_MODEL_PATH.exists() else None
        ),
        grasp_xml_file: str | None = None,
        grasp_max_steps: int = 300,
        grasp_attempts_per_reset: int = 6,
        grasp_deterministic: bool = True,
        grasp_post_grasp_mode: str = "auto",
        grasp_success_min_lift: float = 0.025,
        grasp_success_ee_obj_dist: float = 0.04,
        grasp_success_hold_steps: int = 3,
        grasp_ctrl_close_threshold: float = 0.005,
        grasp_transfer_settle_steps: int = 5,
        allow_grasp_fallback_snapshot: bool = True,
        strict_grasp_object_match: bool = False,
        cross_object_pose_source: str = "site",
        grasp_env_kwargs: dict[str, Any] | None = None,
        allow_eval_gripper_release: bool = True,
        reset_gripper_eval_open: bool = True,
        debug_ik: bool = False,
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
            reward_orientation_weight,
            reward_orientation_tanh_weight,
            reward_success_bonus,
            control_penalty_weight,
            target_tanh_scale,
            orientation_tanh_scale,
            success_distance,
            success_angle_deg,
            success_steps_required,
            max_episode_steps,
            terminate_ee_obj_distance,
            cartesian_action_scale,
            cartesian_rotation_scale_deg,
            ik_workspace_low,
            ik_workspace_high,
            ik_position_only,
            ik_max_iters,
            ik_position_tolerance,
            ik_rotation_tolerance_deg,
            ik_damping,
            ik_step_size,
            ik_max_delta_deg,
            ik_rotation_weight,
            ik_random_restarts,
            ik_seed,
            control_interpolation_steps,
            max_joint_ctrl_delta_deg,
            smooth_cartesian_target,
            target_height_above_place,
            target_x_range,
            target_y_range,
            target_place_z,
            target_place_yaw_range,
            min_initial_object_target_distance,
            target_resample_attempts,
            object_name,
            ee_site_name,
            target_site_name,
            target_body_name,
            place_body_name,
            place_site_name,
            place_geom_name,
            grasp_model_path,
            grasp_xml_file,
            grasp_max_steps,
            grasp_attempts_per_reset,
            grasp_deterministic,
            grasp_post_grasp_mode,
            grasp_success_min_lift,
            grasp_success_ee_obj_dist,
            grasp_success_hold_steps,
            grasp_ctrl_close_threshold,
            grasp_transfer_settle_steps,
            allow_grasp_fallback_snapshot,
            strict_grasp_object_match,
            cross_object_pose_source,
            grasp_env_kwargs,
            allow_eval_gripper_release,
            reset_gripper_eval_open,
            debug_ik,
            **kwargs,
        )

        if str(object_name) != "box":
            raise ValueError(
                "InsertTargetEnvIK is intentionally box-only for now. "
                "Use object_name='box'."
            )
        if grasp_model_path is None:
            raise ValueError(
                "InsertTargetEnvIK requires `grasp_model_path` so reset can start "
                "from a lifted GraspingEnvIK state."
            )

        grasp_model_path_obj = Path(grasp_model_path).expanduser()
        if not grasp_model_path_obj.is_absolute():
            grasp_model_path_obj = grasp_model_path_obj.resolve()
        if not grasp_model_path_obj.exists():
            raise FileNotFoundError(f"Grasp IK model not found: {grasp_model_path_obj}")

        grasp_xml_path_obj = (
            DEFAULT_GRASP_XML_PATH
            if grasp_xml_file is None
            else Path(grasp_xml_file).expanduser()
        )
        if not grasp_xml_path_obj.is_absolute():
            grasp_xml_path_obj = grasp_xml_path_obj.resolve()
        if not grasp_xml_path_obj.exists():
            raise FileNotFoundError(f"Grasp XML not found: {grasp_xml_path_obj}")

        self._reward_target_weight = float(reward_target_weight)
        self._reward_target_tanh_weight = float(reward_target_tanh_weight)
        self._reward_orientation_weight = float(reward_orientation_weight)
        self._reward_orientation_tanh_weight = float(reward_orientation_tanh_weight)
        self._reward_success_bonus = float(reward_success_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._target_tanh_scale = float(target_tanh_scale)
        self._orientation_tanh_scale = float(orientation_tanh_scale)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_steps_required = int(success_steps_required)
        self.max_episode_steps = int(max_episode_steps)
        self._terminate_ee_obj_distance = float(terminate_ee_obj_distance)
        self._control_interpolation_steps = max(1, int(control_interpolation_steps))
        self._max_joint_ctrl_delta_rad = np.deg2rad(float(max_joint_ctrl_delta_deg))
        self._smooth_cartesian_target = bool(smooth_cartesian_target)
        self._target_height_above_place = float(target_height_above_place)
        self._target_x_range = tuple(float(value) for value in target_x_range)
        self._target_y_range = tuple(float(value) for value in target_y_range)
        self._target_place_z = float(target_place_z)
        self._target_place_yaw_range = tuple(
            float(value) for value in target_place_yaw_range
        )
        self._min_initial_object_target_distance = float(
            min_initial_object_target_distance
        )
        self._target_resample_attempts = max(1, int(target_resample_attempts))
        self.object_name = str(object_name)
        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)
        self.target_body_name = str(target_body_name)
        self.place_body_name = str(place_body_name)
        self.place_site_name = str(place_site_name)
        self.place_geom_name = str(place_geom_name)
        self._debug_ik = bool(debug_ik)

        self._gripper_open_target = np.array([0.01, -0.01], dtype=np.float64)
        self._gripper_closed_target = np.array([-0.02, 0.02], dtype=np.float64)
        self._allow_eval_gripper_release = bool(allow_eval_gripper_release)
        self._reset_gripper_eval_open = bool(reset_gripper_eval_open)
        self._eval_gripper_open = False

        self._grasp_model_path = grasp_model_path_obj
        self._grasp_xml_path = grasp_xml_path_obj
        self._grasp_max_steps = int(grasp_max_steps)
        self._grasp_attempts_per_reset = max(1, int(grasp_attempts_per_reset))
        self._grasp_deterministic = bool(grasp_deterministic)
        self._grasp_post_grasp_mode = str(grasp_post_grasp_mode)
        self._grasp_success_min_lift = float(grasp_success_min_lift)
        self._grasp_success_ee_obj_dist = float(grasp_success_ee_obj_dist)
        self._grasp_success_hold_steps = max(1, int(grasp_success_hold_steps))
        self._grasp_ctrl_close_threshold = float(grasp_ctrl_close_threshold)
        self._grasp_transfer_settle_steps = max(0, int(grasp_transfer_settle_steps))
        self._allow_grasp_fallback_snapshot = bool(allow_grasp_fallback_snapshot)
        self._strict_grasp_object_match = bool(strict_grasp_object_match)
        self._cross_object_pose_source = str(cross_object_pose_source).lower()
        if self._cross_object_pose_source not in {"site", "body"}:
            raise ValueError(
                "cross_object_pose_source must be either 'site' or 'body'."
            )
        self._grasp_env_kwargs = dict(grasp_env_kwargs or {})
        self._grasp_env = None
        self._grasp_policy = None

        if self._target_tanh_scale <= 0.0:
            raise ValueError("target_tanh_scale must be greater than 0.")
        if self._orientation_tanh_scale <= 0.0:
            raise ValueError("orientation_tanh_scale must be greater than 0.")
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        if self._success_angle_rad <= 0.0:
            raise ValueError("success_angle_deg must be greater than 0.")
        if self._success_steps_required <= 0:
            raise ValueError("success_steps_required must be greater than 0.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than 0.")
        if self._terminate_ee_obj_distance <= 0.0:
            raise ValueError("terminate_ee_obj_distance must be greater than 0.")
        if self._target_height_above_place <= 0.0:
            raise ValueError("target_height_above_place must be greater than 0.")
        if self._target_x_range[0] > self._target_x_range[1]:
            raise ValueError("target_x_range must be ordered as (min_x, max_x).")
        if self._target_y_range[0] > self._target_y_range[1]:
            raise ValueError("target_y_range must be ordered as (min_y, max_y).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError(
                "target_place_yaw_range must be ordered as (min_yaw, max_yaw)."
            )
        if self._min_initial_object_target_distance < 0.0:
            raise ValueError(
                "min_initial_object_target_distance must be greater than or equal to 0."
            )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            camera_name="watching",
            **kwargs,
        )

        self.object_names = [self.object_name]
        self.object_info = {
            self.object_name: self._build_object_info(self.object_name),
        }
        self.object_one_hot = {
            self.object_name: np.array([1.0], dtype=np.float64),
        }
        self.active_obj_name = self.object_name

        self.place_info = self._build_place_info()
        self._default_place_pos = self.model.body_pos[
            int(self.place_info["body_id"])
        ].copy()
        self._default_place_quat = self.model.body_quat[
            int(self.place_info["body_id"])
        ].copy()
        self._place_site_local_pos = self.model.site_pos[
            int(self.place_info["site_id"])
        ].copy()
        self._place_site_local_quat = _normalize_quat(
            self.model.site_quat[int(self.place_info["site_id"])].copy()
        )

        self.ee_site_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name, "site"
        )
        self.target_site_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site_name, "site"
        )
        self.target_body_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_body_name, "body"
        )

        self.gripL_jid = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_10", "joint"
        )
        self.gripR_jid = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_11", "joint"
        )
        self.gripL_qadr = int(self.model.jnt_qposadr[self.gripL_jid])
        self.gripR_qadr = int(self.model.jnt_qposadr[self.gripR_jid])
        self.gripL_dadr = int(self.model.jnt_dofadr[self.gripL_jid])
        self.gripR_dadr = int(self.model.jnt_dofadr[self.gripR_jid])
        self.gripL_act_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_l", "actuator"
        )
        self.gripR_act_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_r", "actuator"
        )

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._setup_ik_action(
            xml_file=str(self.fullpath),
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
        )

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf
        self.sampled_target_place_pos = self._default_place_pos.copy()
        self.sampled_target_place_quat = self._default_place_quat.copy()
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0
        self.sampled_target_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self._last_target_resample_attempts = 0
        self._last_target_resample_distance = np.inf
        self._last_grasp_reset_attempts = 0
        self._last_grasp_init_lift_height = 0.0
        self._last_grasp_init_ee_obj_dist = np.inf
        self._last_grasp_reset_source = "uninitialized"
        self._last_grasp_source_object = "uninitialized"

        self._sync_target_site_to_active_place()
        closed_ctrl = self.data.ctrl.copy()
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)
        self._reset_ik_state()

        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(
                np.round(1.0 / (self.dt * self._control_interpolation_steps))
            ),
        }

    @staticmethod
    def _require_named_id(model, obj_type: mujoco.mjtObj, name: str, label: str) -> int:
        obj_id = int(mujoco.mj_name2id(model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"MuJoCo {label} `{name}` not found in model.")
        return obj_id

    @staticmethod
    def _optional_named_id(model, obj_type: mujoco.mjtObj, name: str) -> int | None:
        obj_id = int(mujoco.mj_name2id(model, obj_type, name))
        return None if obj_id < 0 else obj_id

    def _build_object_info(self, obj_name: str) -> dict[str, int | str]:
        body_name = f"obj_{obj_name}"
        joint_name = f"obj_{obj_name}_joint"
        site_name = f"obj_{obj_name}_ref"
        geom_name = f"obj_{obj_name}_geom"

        body_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name, "body"
        )
        joint_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name, "joint"
        )
        site_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, site_name, "site"
        )
        geom_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name, "geom"
        )

        return {
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

    def _build_place_info(self) -> dict[str, int | str]:
        body_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.place_body_name, "body"
        )
        site_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.place_site_name, "site"
        )
        geom_id = self._require_named_id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, self.place_geom_name, "geom"
        )
        return {
            "body_name": self.place_body_name,
            "site_name": self.place_site_name,
            "geom_name": self.place_geom_name,
            "body_id": body_id,
            "site_id": site_id,
            "geom_id": geom_id,
        }

    def _setup_ik_action(
        self,
        *,
        xml_file: str,
        cartesian_action_scale: float,
        cartesian_rotation_scale_deg: float,
        ik_workspace_low: tuple[float, float, float],
        ik_workspace_high: tuple[float, float, float],
        ik_position_only: bool,
        ik_max_iters: int,
        ik_position_tolerance: float,
        ik_rotation_tolerance_deg: float,
        ik_damping: float,
        ik_step_size: float,
        ik_max_delta_deg: float,
        ik_rotation_weight: float,
        ik_random_restarts: int,
        ik_seed: int | None,
    ) -> None:
        self._ik_solver = MyCobotIK(
            xml_file=Path(xml_file).expanduser().resolve(),
            ee_site_name=self.ee_site_name,
        )
        self._arm_joint_names = tuple(self._ik_solver.joint_names)
        self._arm_qpos_indices = np.array(
            [
                self.model.jnt_qposadr[
                    self._require_named_id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        joint_name,
                        "joint",
                    )
                ]
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int64,
        )
        self._arm_ctrl_indices = np.array(
            [
                self._require_named_id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    joint_name,
                    "actuator",
                )
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int64,
        )
        self._arm_ctrl_dim = int(len(self._arm_ctrl_indices))
        if self._arm_ctrl_dim != len(self._ik_solver.joint_names):
            raise ValueError(
                "IK action expects the arm actuator count to match IK joint count. "
                f"Got arm_ctrl_dim={self._arm_ctrl_dim} and "
                f"ik_joints={len(self._ik_solver.joint_names)}."
            )

        self._cartesian_action_scale = float(cartesian_action_scale)
        self._cartesian_rotation_scale_rad = np.deg2rad(
            float(cartesian_rotation_scale_deg)
        )
        self._ik_workspace_low = np.asarray(ik_workspace_low, dtype=np.float64).reshape(
            3
        )
        self._ik_workspace_high = np.asarray(
            ik_workspace_high, dtype=np.float64
        ).reshape(3)
        if np.any(self._ik_workspace_low > self._ik_workspace_high):
            raise ValueError(
                "ik_workspace_low must be ordered below ik_workspace_high."
            )

        self._ik_position_only = bool(ik_position_only)
        self._ik_max_iters = int(ik_max_iters)
        self._ik_position_tolerance = float(ik_position_tolerance)
        self._ik_rotation_tolerance_rad = np.deg2rad(float(ik_rotation_tolerance_deg))
        self._ik_damping = float(ik_damping)
        self._ik_step_size = float(ik_step_size)
        self._ik_max_delta_rad = np.deg2rad(float(ik_max_delta_deg))
        self._ik_rotation_weight = float(ik_rotation_weight)
        self._ik_random_restarts = max(0, int(ik_random_restarts))
        self._ik_seed = None if ik_seed is None else int(ik_seed)

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.ACTION_COMPONENTS),),
            dtype=np.float32,
        )
        self._ik_target_pos = np.zeros(3, dtype=np.float64)
        self._ik_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_ik_result: IKResult | None = None
        self._ik_failure_count = 0

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    @staticmethod
    def _quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        wa, xa, ya, za = np.asarray(quat_a, dtype=np.float64)
        wb, xb, yb, zb = np.asarray(quat_b, dtype=np.float64)
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
            cls._quat_multiply(_normalize_quat(quat), vec_quat),
            cls._quat_conjugate(_normalize_quat(quat)),
        )
        return rotated[1:]

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half_yaw = float(yaw) * 0.5
        return np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    @staticmethod
    def _wrap_to_pi(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _wrap_vector_to_pi(self, angles_rad: np.ndarray) -> np.ndarray:
        return np.array(
            [self._wrap_to_pi(float(value)) for value in angles_rad],
            dtype=np.float64,
        )

    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        quat = _normalize_quat(quat)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _get_site_quat(self, site_name: str) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site(site_name).xmat)
        return _normalize_quat(quat)

    def _get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site(site_name).xpos.copy(), self._get_site_quat(site_name)

    def _rotation_vector(
        self,
        source_quat: np.ndarray,
        target_quat: np.ndarray,
    ) -> np.ndarray:
        source_quat = _normalize_quat(source_quat)
        target_quat = _normalize_quat(target_quat)
        delta = self._quat_multiply(target_quat, self._quat_conjugate(source_quat))
        delta = _normalize_quat(delta)
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

    def _get_pose_in_body_frame(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_quat = _normalize_quat(body_quat)
        body_quat_conj = self._quat_conjugate(body_quat)
        local_pos = self._quat_rotate_vector(
            body_quat_conj,
            np.asarray(world_pos, dtype=np.float64)
            - np.asarray(body_pos, dtype=np.float64),
        )
        local_quat = _normalize_quat(
            self._quat_multiply(body_quat_conj, np.asarray(world_quat))
        )
        return local_pos, local_quat

    def _get_active_obj_info(self) -> dict[str, int | str]:
        return self.object_info[self.active_obj_name]

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.place_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _current_arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self._arm_qpos_indices], dtype=np.float64
        ).copy()

    def _reset_ik_state(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        self._ik_target_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        self._ik_target_quat = _normalize_quat(np.asarray(ee_quat, dtype=np.float64))
        self._last_ik_result = None
        self._ik_failure_count = 0

    def _compute_ik_target(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        delta_pos = self._cartesian_action_scale * action[:3]
        delta_rpy = self._cartesian_rotation_scale_rad * action[3:6]

        if self._smooth_cartesian_target:
            base_pos = self._ik_target_pos.copy()
            base_quat = self._ik_target_quat.copy()
        else:
            base_pos, base_quat = self._get_ee_pose()
            base_pos = np.asarray(base_pos, dtype=np.float64)
            base_quat = np.asarray(base_quat, dtype=np.float64)

        target_pos = np.clip(
            base_pos + delta_pos,
            self._ik_workspace_low,
            self._ik_workspace_high,
        )

        base_rpy = _quat_to_euler_xyz(base_quat)
        target_rpy = self._wrap_vector_to_pi(base_rpy + delta_rpy)
        target_quat = _quat_from_euler_xyz(*target_rpy)
        return target_pos, target_quat

    def _ik_action_to_target_ctrl(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, IKResult]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != self.action_space.shape:
            expected = ", ".join(self.ACTION_COMPONENTS)
            raise ValueError(
                f"Unexpected action shape for {type(self).__name__}. "
                f"Expected {self.action_space.shape} ({expected}), got {action.shape}."
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos, target_quat = self._compute_ik_target(action)
        ik_result = self._ik_solver.solve(
            target_pos,
            target_quat,
            initial_q=self._current_arm_joint_positions(),
            position_only=self._ik_position_only,
            max_iters=self._ik_max_iters,
            position_tolerance=self._ik_position_tolerance,
            rotation_tolerance=self._ik_rotation_tolerance_rad,
            damping=self._ik_damping,
            step_size=self._ik_step_size,
            max_delta=self._ik_max_delta_rad,
            rotation_weight=self._ik_rotation_weight,
            random_restarts=self._ik_random_restarts,
            seed=self._ik_seed,
        )

        target_ctrl = self.data.ctrl.copy()
        current_arm_ctrl = self.data.ctrl[self._arm_ctrl_indices].copy()
        desired_arm_ctrl = ik_result.q_rad.copy()

        if self._max_joint_ctrl_delta_rad > 0.0:
            delta_q = desired_arm_ctrl - current_arm_ctrl
            delta_q = np.clip(
                delta_q,
                -self._max_joint_ctrl_delta_rad,
                self._max_joint_ctrl_delta_rad,
            )
            desired_arm_ctrl = current_arm_ctrl + delta_q

        target_ctrl[self._arm_ctrl_indices] = desired_arm_ctrl
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self.last_action = action.astype(np.float32)
        self._ik_target_pos = target_pos.copy()
        self._ik_target_quat = target_quat.copy()
        self._last_ik_result = ik_result
        if not ik_result.success:
            self._ik_failure_count += 1

        return action, target_ctrl, ik_result

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[self.gripL_act_id] = self._gripper_open_target[0]
        ctrl[self.gripR_act_id] = self._gripper_open_target[1]

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[self.gripL_act_id] = self._gripper_closed_target[0]
        ctrl[self.gripR_act_id] = self._gripper_closed_target[1]

    def _apply_gripper_policy(self, ctrl: np.ndarray) -> None:
        if self._allow_eval_gripper_release and self._eval_gripper_open:
            self._set_open_gripper_target(ctrl)
        else:
            self._set_closed_gripper_target(ctrl)

    def set_eval_gripper_open(self, open_gripper: bool = True) -> None:
        self._eval_gripper_open = bool(open_gripper)
        ctrl = self.data.ctrl.copy()
        self._apply_gripper_policy(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

    def open_gripper_for_eval(self) -> None:
        self.set_eval_gripper_open(True)

    def close_gripper(self) -> None:
        self.set_eval_gripper_open(False)

    def _sample_target_place_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        place_pos = self._default_place_pos.copy()
        place_pos[0] = self.np_random.uniform(*self._target_x_range)
        place_pos[1] = self.np_random.uniform(*self._target_y_range)
        place_pos[2] = self._target_place_z
        yaw = float(self.np_random.uniform(*self._target_place_yaw_range))
        place_quat = self._yaw_to_quat(yaw)
        return place_pos, place_quat, yaw

    def _target_pos_for_place_pose(
        self,
        place_pos: np.ndarray,
        place_quat: np.ndarray,
    ) -> np.ndarray:
        target_site_local_pos = self._place_site_local_pos.copy()
        target_site_local_pos[2] += self._target_height_above_place
        return np.asarray(place_pos, dtype=np.float64) + self._quat_rotate_vector(
            place_quat,
            target_site_local_pos,
        )

    def _sample_target_place_pose_away_from_object(
        self,
        object_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, int]:
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        best_pose: tuple[np.ndarray, np.ndarray, float] | None = None
        best_distance = -np.inf

        for attempt in range(1, self._target_resample_attempts + 1):
            place_pos, place_quat, yaw = self._sample_target_place_pose()
            target_pos = self._target_pos_for_place_pose(place_pos, place_quat)
            distance = float(np.linalg.norm(target_pos - object_pos))
            if distance > best_distance:
                best_distance = distance
                best_pose = (place_pos, place_quat, yaw)
            if distance >= self._min_initial_object_target_distance:
                return place_pos, place_quat, yaw, distance, attempt

        assert best_pose is not None
        if self._min_initial_object_target_distance <= 0.0:
            return (*best_pose, best_distance, self._target_resample_attempts)

        raise RuntimeError(
            "Failed to sample an insert target far enough from the object. "
            f"Required at least {self._min_initial_object_target_distance:.3f} m, "
            f"best sampled distance was {best_distance:.3f} m after "
            f"{self._target_resample_attempts} attempts. "
            "Widen target_x_range/target_y_range or reduce "
            "min_initial_object_target_distance."
        )

    def _set_place_pose_in_model(
        self,
        place_pos: np.ndarray,
        place_quat: np.ndarray,
    ) -> None:
        body_id = int(self.place_info["body_id"])
        self.model.body_pos[body_id] = np.asarray(place_pos, dtype=np.float64).reshape(
            3
        )
        self.model.body_quat[body_id] = _normalize_quat(
            np.asarray(place_quat, dtype=np.float64).reshape(4)
        )

    def _sync_target_site_to_active_place(self) -> None:
        place_body_id = int(self.place_info["body_id"])
        place_site_id = int(self.place_info["site_id"])
        self.model.body_pos[self.target_body_id] = self.model.body_pos[
            place_body_id
        ].copy()
        self.model.body_quat[self.target_body_id] = self.model.body_quat[
            place_body_id
        ].copy()

        target_site_local_pos = self.model.site_pos[place_site_id].copy()
        target_site_local_pos[2] += self._target_height_above_place
        self.model.site_pos[self.target_site_id] = target_site_local_pos
        self.model.site_quat[self.target_site_id] = self.model.site_quat[
            place_site_id
        ].copy()

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
                "InsertTargetEnvIK requires stable-baselines3 to load the "
                "GraspingEnvIK reset policy."
            ) from exc

        grasp_env_kwargs = {
            "xml_file": str(self._grasp_xml_path),
            "render_mode": None,
        }
        grasp_env_signature = inspect.signature(GraspingEnvIK.__init__)
        if "post_grasp_mode" in grasp_env_signature.parameters:
            grasp_env_kwargs["post_grasp_mode"] = self._grasp_post_grasp_mode

        grasp_env_kwargs.update(self._grasp_env_kwargs)
        if "post_grasp_mode" not in grasp_env_signature.parameters:
            grasp_env_kwargs.pop("post_grasp_mode", None)

        self._grasp_env = GraspingEnvIK(**grasp_env_kwargs)
        self._grasp_policy = SAC.load(
            str(self._grasp_model_path),
            env=self._grasp_env,
            device="auto",
        )

    def _get_grasp_active_obj_info(self) -> dict[str, int | str]:
        grasp_env = self._grasp_env
        assert grasp_env is not None
        active_obj_name = str(grasp_env.active_obj_name)
        return grasp_env.object_info[active_obj_name]

    def _get_grasp_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        grasp_env = self._grasp_env
        assert grasp_env is not None
        info = self._get_grasp_active_obj_info()
        site_name = str(info["site_name"])
        obj_pos = grasp_env.data.site(site_name).xpos.copy()
        obj_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(obj_quat, grasp_env.data.site(site_name).xmat)
        return obj_pos, _normalize_quat(obj_quat)

    def _get_grasp_obj_body_pose(self) -> tuple[np.ndarray, np.ndarray]:
        grasp_env = self._grasp_env
        assert grasp_env is not None
        info = self._get_grasp_active_obj_info()
        body_name = str(info["body_name"])
        body = grasp_env.data.body(body_name)
        return body.xpos.copy(), _normalize_quat(body.xquat.copy())

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
        info = self._get_grasp_active_obj_info()
        dofadr = int(info["dofadr"])
        return float(np.linalg.norm(grasp_env.data.qvel[dofadr : dofadr + 3]))

    def _capture_grasp_snapshot(self, initial_obj_pos: np.ndarray) -> dict:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        obj_pos, obj_quat = self._get_grasp_obj_pose()
        obj_body_pos, obj_body_quat = self._get_grasp_obj_body_pose()
        ee_pos, ee_quat = self._get_grasp_ee_pose()
        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        lift_height = float(obj_pos[2] - initial_obj_pos[2])
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        object_speed = self._get_grasp_object_speed()

        return {
            "qpos": grasp_env.data.qpos.copy(),
            "qvel": grasp_env.data.qvel.copy(),
            "ctrl": grasp_env.data.ctrl.copy(),
            "active_object": str(grasp_env.active_obj_name),
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "obj_body_pos": obj_body_pos,
            "obj_body_quat": obj_body_quat,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "lift_height": lift_height,
            "ee_obj_dist": ee_obj_dist,
            "object_speed": object_speed,
            "gripper_ctrl": grasp_env.data.ctrl[-2:].copy(),
            "success_counter": int(getattr(grasp_env, "success_counter", 0)),
            "lift_success_counter": int(getattr(grasp_env, "lift_success_counter", 0)),
            "ik_failure_count": int(getattr(grasp_env, "_ik_failure_count", 0)),
        }

    def _is_good_grasp_snapshot(self, snapshot: dict) -> bool:
        source_object = str(snapshot["active_object"])
        if self._strict_grasp_object_match and source_object != self.object_name:
            return False

        gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
        is_closed = bool(
            gripper_ctrl[0] < -self._grasp_ctrl_close_threshold
            and gripper_ctrl[1] > self._grasp_ctrl_close_threshold
        )
        return bool(
            is_closed
            and float(snapshot["ee_obj_dist"]) <= self._grasp_success_ee_obj_dist
            and float(snapshot["lift_height"]) >= self._grasp_success_min_lift
        )

    def _score_grasp_snapshot(self, snapshot: dict) -> float:
        gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
        is_closed = float(
            gripper_ctrl[0] < -self._grasp_ctrl_close_threshold
            and gripper_ctrl[1] > self._grasp_ctrl_close_threshold
        )
        object_match_bonus = 0.5 * float(
            str(snapshot["active_object"]) == self.object_name
        )
        return (
            4.0 * float(snapshot["lift_height"])
            - 2.5 * float(snapshot["ee_obj_dist"])
            - 0.2 * float(snapshot["object_speed"])
            + 0.05 * is_closed
            + 0.02 * float(snapshot["lift_success_counter"])
            + object_match_bonus
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
                    return snapshot, "grasp_ik_success", attempt

                if terminated or truncated:
                    break

        if best_snapshot is None or not self._allow_grasp_fallback_snapshot:
            raise RuntimeError(
                "Failed to obtain a lifted state from the GraspingEnvIK policy. "
                "Try increasing grasp_max_steps or grasp_attempts_per_reset."
            )

        return (
            best_snapshot,
            "grasp_ik_fallback_best_snapshot",
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

        source_object = str(snapshot["active_object"])
        if self._strict_grasp_object_match and source_object != self.object_name:
            raise ValueError(
                "Grasp snapshot object mismatch. "
                f"Expected `{self.object_name}`, got `{source_object}`."
            )

        grasp_env = self._grasp_env
        assert grasp_env is not None
        source_model = grasp_env.model
        source_qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
        source_qvel = np.asarray(snapshot["qvel"], dtype=np.float64)
        source_joint_map = self._joint_name_map(source_model)
        target_joint_map = self._joint_name_map(self.model)
        source_object_joint = f"obj_{source_object}_joint"
        target_object_joint = str(self._get_active_obj_info()["joint_name"])
        transfer_joint_names = sorted(
            set(source_joint_map).intersection(target_joint_map)
        )

        for joint_name in transfer_joint_names:
            if joint_name in {source_object_joint, target_object_joint}:
                continue

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

        object_info = self._get_active_obj_info()
        object_qposadr = int(object_info["qposadr"])
        object_dofadr = int(object_info["dofadr"])
        if source_object == self.object_name:
            obj_pos = np.asarray(snapshot["obj_body_pos"], dtype=np.float64)
            obj_quat = _normalize_quat(np.asarray(snapshot["obj_body_quat"]))
        elif self._cross_object_pose_source == "body":
            obj_pos = np.asarray(snapshot["obj_body_pos"], dtype=np.float64)
            obj_quat = _normalize_quat(np.asarray(snapshot["obj_body_quat"]))
        else:
            obj_pos = np.asarray(snapshot["obj_pos"], dtype=np.float64)
            obj_quat = _normalize_quat(np.asarray(snapshot["obj_quat"]))

        qpos[object_qposadr : object_qposadr + 3] = obj_pos
        qpos[object_qposadr + 3 : object_qposadr + 7] = obj_quat
        qvel[object_dofadr : object_dofadr + 6] = 0.0

        self.set_state(qpos, qvel)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        closed_ctrl = self.data.ctrl.copy()
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        if self._grasp_transfer_settle_steps > 0:
            settle_ctrl = self.data.ctrl.copy()
            self._set_closed_gripper_target(settle_ctrl)
            settle_ctrl = np.clip(settle_ctrl, self._ctrl_low, self._ctrl_high)
            for _ in range(self._grasp_transfer_settle_steps):
                self.do_simulation(settle_ctrl, 1)
            mujoco.mj_forward(self.model, self.data)

    def _initialize_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        reset_source: str,
        attempt_count: int,
    ) -> np.ndarray:
        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        if self._reset_gripper_eval_open:
            self._eval_gripper_open = False
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

        self._set_place_pose_in_model(self._default_place_pos, self._default_place_quat)
        self._sync_target_site_to_active_place()

        self._restore_grasp_snapshot(snapshot)
        mujoco.mj_forward(self.model, self.data)

        obj_pos, obj_quat = self._get_active_obj_pose()
        (
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
            self._last_target_resample_distance,
            self._last_target_resample_attempts,
        ) = self._sample_target_place_pose_away_from_object(obj_pos)
        self._set_place_pose_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

        target_pos, target_quat = self._get_target_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        obj_target_pos_error, _ = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        self.initial_object_target_dist = float(np.linalg.norm(obj_target_pos_error))
        self.best_object_target_dist = float(self.initial_object_target_dist)
        self.initial_obj_site_pos = obj_pos.copy()
        self.sampled_target_pos = target_pos.copy()
        self.sampled_target_quat = target_quat.copy()
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(
                _normalize_quat(self.data.body(self.place_body_name).xquat.copy())
            )
        )
        self.sampled_object_yaw = float(
            self._quat_to_yaw(np.asarray(snapshot["obj_quat"], dtype=np.float64))
        )
        self.applied_object_yaw = float(self._quat_to_yaw(obj_quat))
        self._last_grasp_reset_attempts = int(attempt_count)
        self._last_grasp_init_lift_height = float(snapshot["lift_height"])
        self._last_grasp_init_ee_obj_dist = float(snapshot["ee_obj_dist"])
        self._last_grasp_reset_source = str(reset_source)
        self._last_grasp_source_object = str(snapshot["active_object"])
        self._reset_ik_state()

        return self._get_obs()

    def reset_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        seed: int | None = None,
        reset_source: str = "external_grasp_ik_snapshot",
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
        snapshot, reset_source, attempt_count = self._sample_grasp_reset_snapshot()
        return self._initialize_from_grasp_snapshot(
            snapshot,
            reset_source=reset_source,
            attempt_count=attempt_count,
        )

    def step(self, action):
        self.current_step += 1
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)
        self._apply_gripper_policy(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        start_ctrl = self.data.ctrl.copy()
        for interp_idx in range(1, self._control_interpolation_steps + 1):
            alpha = interp_idx / self._control_interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            smooth_ctrl = np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high)
            self.do_simulation(smooth_ctrl, self.frame_skip)

        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

        print(
            f"Object pos : {self._get_active_obj_pose()[0]}, Target pos: {self._get_target_pose()[0]}"
        )
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated_success = self.success_counter >= self._success_steps_required
        terminated_ee_obj_far = bool(
            float(reward_info["ee_object_dist"]) >= self._terminate_ee_obj_distance
            and not bool(reward_info["target_pose_aligned"])
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

    def _get_box_place_metrics(self) -> dict[str, np.ndarray | float | int]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        place_body = self.data.body(self.place_body_name)
        place_body_pos = place_body.xpos.copy()
        place_body_quat = _normalize_quat(place_body.xquat.copy())
        obj_local_pos, obj_local_quat = self._get_pose_in_body_frame(
            obj_pos,
            obj_quat,
            place_body_pos,
            place_body_quat,
        )
        target_local_pos = self._place_site_local_pos.copy()
        target_local_pos[2] += self._target_height_above_place
        target_local_quat = self._place_site_local_quat.copy()
        local_pos_error, local_rot_error = self._get_pose_error(
            obj_local_pos,
            obj_local_quat,
            target_local_pos,
            target_local_quat,
        )
        return {
            "object_local_pos": obj_local_pos,
            "object_local_quat": obj_local_quat,
            "target_local_pos": target_local_pos,
            "target_local_quat": target_local_quat,
            "object_target_local_pos_error": local_pos_error,
            "object_target_local_rot_error": local_rot_error,
            "object_target_local_radial_error": float(
                np.linalg.norm(local_pos_error[:2])
            ),
            "object_target_local_height_error": float(local_pos_error[2]),
            "object_target_local_angle_error": float(np.linalg.norm(local_rot_error)),
        }

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_pos, ee_quat = self._get_ee_pose()

        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        target_pose_aligned = bool(
            target_dist < self._success_distance
            and target_angle < self._success_angle_rad
        )
        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)

        if target_pose_aligned:
            self.success_counter += 1
        else:
            self.success_counter = 0

        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / self._target_tanh_scale))
        ) * self._reward_target_tanh_weight
        reward_orientation = -target_angle * self._reward_orientation_weight
        reward_orientation_tanh = (
            1.0 - float(np.tanh(target_angle / self._orientation_tanh_scale))
        ) * self._reward_orientation_tanh_weight
        reward_success_bonus = (
            self._reward_success_bonus if target_pose_aligned else 0.0
        )
        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )
        reward = (
            reward_target
            + reward_target_tanh
            + reward_orientation
            + reward_orientation_tanh
            + reward_success_bonus
            + control_penalty
        )
        box_place_metrics = self._get_box_place_metrics()

        reward_info = {
            "ee_object_dist": ee_obj_dist,
            "object_target_dist": target_dist,
            "object_target_angle_rad": target_angle,
            "object_target_rot_error": target_angle,
            "target_pose_aligned": int(target_pose_aligned),
            "success_counter": int(self.success_counter),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_orientation": float(reward_orientation),
            "reward_orientation_tanh": float(reward_orientation_tanh),
            "reward_success_bonus": float(reward_success_bonus),
            "control_penalty": float(control_penalty),
            "gripper_closed": int(self.gripper_state == "closed"),
            "eval_gripper_open": int(self._eval_gripper_open),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
            "object_target_local_radial_error": float(
                box_place_metrics["object_target_local_radial_error"]
            ),
            "object_target_local_height_error": float(
                box_place_metrics["object_target_local_height_error"]
            ),
            "object_target_local_angle_error": float(
                box_place_metrics["object_target_local_angle_error"]
            ),
        }
        return float(reward), reward_info

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
        qpos = self.data.qpos
        qvel = self.data.qvel
        object_qposadr = int(self._get_active_obj_info()["qposadr"])
        object_dofadr = int(self._get_active_obj_info()["dofadr"])
        robot_qpos = qpos[:object_qposadr]
        robot_qvel = qvel[:object_dofadr]
        gripper_qpos = qpos[[self.gripL_qadr, self.gripR_qadr]].copy()
        gripper_qvel = qvel[[self.gripL_dadr, self.gripR_dadr]].copy()
        gripper_ctrl = self.data.ctrl[[self.gripL_act_id, self.gripR_act_id]].copy()
        gripper_closed = np.array(
            [1.0 if self.gripper_state == "closed" else 0.0], dtype=np.float64
        )

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_pos, place_quat = self._get_place_site_pose()
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            obj_pos,
            obj_quat,
        )
        target_delta_euler = self._wrap_vector_to_pi(
            _quat_to_euler_xyz(target_quat) - _quat_to_euler_xyz(obj_quat)
        )
        metrics = np.array(
            [
                np.linalg.norm(obj_target_pos_error),
                np.linalg.norm(obj_target_rot_error),
                np.linalg.norm(ee_obj_pos_error),
                np.linalg.norm(ee_obj_rot_error),
                float(self.success_counter),
                float(self._ik_failure_count),
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
            ("object_type", self.object_one_hot[self.active_obj_name]),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("place_pos", place_pos),
            ("place_quat", place_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("object_target_pos_error", obj_target_pos_error),
            ("object_target_rot_error", obj_target_rot_error),
            ("ee_object_pos_error", ee_obj_pos_error),
            ("ee_object_rot_error", ee_obj_rot_error),
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

    def _get_ik_debug_state(self) -> dict:
        state = {
            "ik_target_pos": self._ik_target_pos.copy(),
            "ik_target_quat": self._ik_target_quat.copy(),
            "ik_target_rpy_deg": np.rad2deg(_quat_to_euler_xyz(self._ik_target_quat)),
            "ik_position_only": bool(self._ik_position_only),
            "ik_failure_count": int(self._ik_failure_count),
            "control_interpolation_steps": int(self._control_interpolation_steps),
            "max_joint_ctrl_delta_deg": float(
                np.rad2deg(self._max_joint_ctrl_delta_rad)
            ),
            "smooth_cartesian_target": bool(self._smooth_cartesian_target),
        }
        if self._last_ik_result is None:
            state.update(
                {
                    "ik_success": None,
                    "ik_iterations": 0,
                    "ik_attempt": 0,
                }
            )
            return state

        state.update(
            {
                "ik_success": bool(self._last_ik_result.success),
                "ik_message": self._last_ik_result.message,
                "ik_iterations": int(self._last_ik_result.iterations),
                "ik_attempt": int(self._last_ik_result.attempt),
                "ik_position_error": self._last_ik_result.position_error.copy(),
                "ik_rotation_error": self._last_ik_result.rotation_error.copy(),
                "ik_position_error_norm": float(
                    self._last_ik_result.position_error_norm
                ),
                "ik_rotation_error_deg": float(
                    np.rad2deg(self._last_ik_result.rotation_error_norm)
                ),
            }
        )
        return state

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
        config["action"]["gripper_policy"] = "fixed_closed_with_eval_release_method"
        config["action"]["gripper_open_target"] = self._gripper_open_target.tolist()
        config["action"]["gripper_closed_target"] = self._gripper_closed_target.tolist()
        config["action"]["allow_eval_gripper_release"] = bool(
            self._allow_eval_gripper_release
        )
        config["reward"]["params"]["target_tanh_scale"] = float(self._target_tanh_scale)
        config["reward"]["params"]["orientation_tanh_scale"] = float(
            self._orientation_tanh_scale
        )
        config["task"]["target_mode"] = "box_pose_matches_cube_place_site_plus_offset"
        config["task"]["object_name"] = self.object_name
        config["task"]["target_site_name"] = self.target_site_name
        config["task"]["place_body_name"] = self.place_body_name
        config["task"]["place_site_name"] = self.place_site_name
        config["task"]["target_height_above_place"] = float(
            self._target_height_above_place
        )
        config["task"]["target_place_randomization"] = {
            "target_x_range": list(self._target_x_range),
            "target_y_range": list(self._target_y_range),
            "target_place_z": float(self._target_place_z),
            "target_place_yaw_range": list(self._target_place_yaw_range),
            "min_initial_object_target_distance": float(
                self._min_initial_object_target_distance
            ),
            "target_resample_attempts": int(self._target_resample_attempts),
        }
        config["task"]["success_criterion"] = {
            "distance_m": float(self._success_distance),
            "angle_deg": float(np.rad2deg(self._success_angle_rad)),
            "steps_required": int(self._success_steps_required),
        }
        config["task"]["grasp_ik_reset"] = {
            "grasp_model_path": str(self._grasp_model_path),
            "grasp_xml_file": str(self._grasp_xml_path),
            "grasp_max_steps": int(self._grasp_max_steps),
            "grasp_attempts_per_reset": int(self._grasp_attempts_per_reset),
            "grasp_post_grasp_mode": self._grasp_post_grasp_mode,
            "grasp_success_min_lift": float(self._grasp_success_min_lift),
            "grasp_success_ee_obj_dist": float(self._grasp_success_ee_obj_dist),
            "grasp_success_hold_steps": int(self._grasp_success_hold_steps),
            "strict_grasp_object_match": bool(self._strict_grasp_object_match),
            "cross_object_pose_source": self._cross_object_pose_source,
        }
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        place_pos, place_quat = self._get_place_site_pose()
        target_pos, target_quat = self._get_target_pose()
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
        box_place_metrics = self._get_box_place_metrics()
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))

        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "place_pos": place_pos,
            "place_quat": place_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_height_above_place": float(self._target_height_above_place),
            "target_place_body_pos": self.data.body(self.place_body_name).xpos.copy(),
            "target_place_body_quat": _normalize_quat(
                self.data.body(self.place_body_name).xquat.copy()
            ),
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": target_dist,
            "obj_target_angle_rad": target_angle,
            "object_target_dist": target_dist,
            "object_target_rot_error": target_angle,
            "target_pose_aligned": bool(
                target_dist < self._success_distance
                and target_angle < self._success_angle_rad
            ),
            "success_counter": int(self.success_counter),
            "initial_object_target_dist": float(self.initial_object_target_dist),
            "best_object_target_dist": float(self.best_object_target_dist),
            "gripper_state": self.gripper_state,
            "eval_gripper_open": bool(self._eval_gripper_open),
            "gripper_qpos": self.data.qpos[[self.gripL_qadr, self.gripR_qadr]].copy(),
            "gripper_ctrl": self.data.ctrl[
                [self.gripL_act_id, self.gripR_act_id]
            ].copy(),
            "last_action": self.last_action.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "sampled_target_place_quat": self.sampled_target_place_quat.copy(),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "min_initial_object_target_distance": float(
                self._min_initial_object_target_distance
            ),
            "target_resample_attempts": int(self._last_target_resample_attempts),
            "target_resample_distance": float(self._last_target_resample_distance),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "grasp_reset_attempts": int(self._last_grasp_reset_attempts),
            "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
            "grasp_reset_source": self._last_grasp_reset_source,
            "grasp_source_object": self._last_grasp_source_object,
            "object_local_pos": np.asarray(
                box_place_metrics["object_local_pos"], dtype=np.float64
            ).copy(),
            "object_local_quat": np.asarray(
                box_place_metrics["object_local_quat"], dtype=np.float64
            ).copy(),
            "target_local_pos": np.asarray(
                box_place_metrics["target_local_pos"], dtype=np.float64
            ).copy(),
            "target_local_quat": np.asarray(
                box_place_metrics["target_local_quat"], dtype=np.float64
            ).copy(),
            "object_target_local_pos_error": np.asarray(
                box_place_metrics["object_target_local_pos_error"], dtype=np.float64
            ).copy(),
            "object_target_local_rot_error": np.asarray(
                box_place_metrics["object_target_local_rot_error"], dtype=np.float64
            ).copy(),
            "object_target_local_radial_error": float(
                box_place_metrics["object_target_local_radial_error"]
            ),
            "object_target_local_height_error": float(
                box_place_metrics["object_target_local_height_error"]
            ),
            "object_target_local_angle_error": float(
                box_place_metrics["object_target_local_angle_error"]
            ),
            **self._get_ik_debug_state(),
        }

    def render(self):
        return super().render()

    def close(self):
        if self._grasp_env is not None:
            self._grasp_env.close()
            self._grasp_env = None
            self._grasp_policy = None
        return super().close()
