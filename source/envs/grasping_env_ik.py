from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from script.inverse_kinematics import (
    IKResult,
    MyCobotIK,
    _normalize_quat,
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)

from .config_export import capture_init_config, export_env_config

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_lift.xml"


class GraspingEnvIK(MujocoEnv, utils.EzPickle):
    """Standalone grasping env with 6-DoF Cartesian actions solved by IK."""

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
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        control_interpolation: bool = True,
        control_interpolation_profile: str = "smoothstep",
        control_smoothing_factor: float = 0.35,
        reward_ee_obj_dist_weight: float = 3.0,
        reward_ee_obj_dist_tanh_weight: float = 1.5,
        reward_ee_obj_orient_weight: float = 1.0,
        reward_obj_target_dist_weight: float = 5.0,
        reward_obj_target_dist_tanh_weight: float = 3.0,
        reward_obj_target_orient_weight: float = 1.0,
        reward_target_bonus: float = 10.0,
        control_penalty_weight: float = 0.001,
        ee_obj_tanh_scale: float = 0.05,
        obj_target_tanh_scale: float = 0.05,
        success_distance: float = 0.01,
        success_angle_deg: float = 25.0,
        success_steps_required: int = 10,
        max_episode_steps: int = 100,
        cartesian_action_scale: float = 1.0,
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
        object_x_range: tuple[float, float] = (0.08, 0.27),
        object_y_range: tuple[float, float] = (-0.20, 0.20),
        object_z: float = 0.025,
        object_yaw_limit_rad: float = 1.05,
        lift_height: float = 0.10,
        grasp_close_distance: float = 0.015,
        grasp_release_distance: float = 0.055,
        grasp_close_angle_deg: float = 25.0,
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        **kwargs,
    ):
        self._init_config = capture_init_config(locals())
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            control_interpolation,
            control_interpolation_profile,
            control_smoothing_factor,
            reward_ee_obj_dist_weight,
            reward_ee_obj_dist_tanh_weight,
            reward_ee_obj_orient_weight,
            reward_obj_target_dist_weight,
            reward_obj_target_dist_tanh_weight,
            reward_obj_target_orient_weight,
            reward_target_bonus,
            control_penalty_weight,
            ee_obj_tanh_scale,
            obj_target_tanh_scale,
            success_distance,
            success_angle_deg,
            success_steps_required,
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
            object_x_range,
            object_y_range,
            object_z,
            object_yaw_limit_rad,
            lift_height,
            grasp_close_distance,
            grasp_release_distance,
            grasp_close_angle_deg,
            ee_site_name,
            target_site_name,
            **kwargs,
        )

        self._control_interpolation = bool(control_interpolation)
        self._control_interpolation_profile = str(control_interpolation_profile)
        self._control_smoothing_factor = float(control_smoothing_factor)
        self._reward_ee_obj_dist_weight = float(reward_ee_obj_dist_weight)
        self._reward_ee_obj_dist_tanh_weight = float(reward_ee_obj_dist_tanh_weight)
        self._reward_ee_obj_orient_weight = float(reward_ee_obj_orient_weight)
        self._reward_obj_target_dist_weight = float(reward_obj_target_dist_weight)
        self._reward_obj_target_dist_tanh_weight = float(
            reward_obj_target_dist_tanh_weight
        )
        self._reward_obj_target_orient_weight = float(reward_obj_target_orient_weight)
        self._reward_target_bonus = float(reward_target_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._ee_obj_tanh_scale = float(ee_obj_tanh_scale)
        self._obj_target_tanh_scale = float(obj_target_tanh_scale)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_steps_required = int(success_steps_required)
        self.max_episode_steps = int(max_episode_steps)
        self._object_x_range = tuple(float(value) for value in object_x_range)
        self._object_y_range = tuple(float(value) for value in object_y_range)
        self._object_z = float(object_z)
        self._object_yaw_limit_rad = float(object_yaw_limit_rad)
        self._lift_height = float(lift_height)
        self._grasp_close_distance = float(grasp_close_distance)
        self._grasp_release_distance = float(grasp_release_distance)
        self._grasp_close_angle_rad = np.deg2rad(float(grasp_close_angle_deg))
        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)

        if self._control_interpolation_profile not in {"linear", "smoothstep"}:
            raise ValueError(
                "control_interpolation_profile must be `linear` or `smoothstep`."
            )
        if not 0.0 < self._control_smoothing_factor <= 1.0:
            raise ValueError("control_smoothing_factor must be in the range (0, 1].")
        if self._ee_obj_tanh_scale <= 0.0:
            raise ValueError("ee_obj_tanh_scale must be greater than 0.")
        if self._obj_target_tanh_scale <= 0.0:
            raise ValueError("obj_target_tanh_scale must be greater than 0.")
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        if self._success_angle_rad <= 0.0:
            raise ValueError("success_angle_deg must be greater than 0.")
        if self._success_steps_required <= 0:
            raise ValueError("success_steps_required must be greater than 0.")

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

            body_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_BODY, body_name, "body"
            )
            joint_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_JOINT, joint_name, "joint"
            )
            site_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_SITE, site_name, "site"
            )
            geom_id = self._require_named_id(
                mujoco.mjtObj.mjOBJ_GEOM, geom_name, "geom"
            )

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

        self.active_obj_name = self.object_names[0]
        self.target_site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.target_site_name, "site"
        )

        self.gripL_jid = self._require_named_id(
            mujoco.mjtObj.mjOBJ_JOINT, "Slider_10", "joint"
        )
        self.gripR_jid = self._require_named_id(
            mujoco.mjtObj.mjOBJ_JOINT, "Slider_11", "joint"
        )
        self.gripL_qadr = int(self.model.jnt_qposadr[self.gripL_jid])
        self.gripR_qadr = int(self.model.jnt_qposadr[self.gripR_jid])
        self.gripL_dadr = int(self.model.jnt_dofadr[self.gripL_jid])
        self.gripR_dadr = int(self.model.jnt_dofadr[self.gripR_jid])
        self.gripL_act_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_l", "actuator"
        )
        self.gripR_act_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_r", "actuator"
        )

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self._last_control_start = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control_target = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control_applied_target = np.zeros(self.model.nu, dtype=np.float64)
        self._last_control_delta_norm = 0.0
        self._last_arm_control_delta_max_abs = 0.0
        self._setup_ik_action(
            xml_file=str(self.fullpath),
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
        self.gripper_state = "open"
        self.grasp_latched = False
        self.last_grasp_should_close = False
        self.last_grasp_dist = np.inf
        self.last_grasp_angle = np.inf
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0

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
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def _require_named_id(
        self,
        obj_type: mujoco.mjtObj,
        name: str,
        label: str,
    ) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"MuJoCo {label} `{name}` not found in model.")
        return obj_id

    def _setup_ik_action(
        self,
        *,
        xml_file: str,
        cartesian_action_scale: float,
        cartesian_rotation_scale_deg: float,
        ik_workspace_low: tuple[float, float, float],
        ik_workspace_high: tuple[float, float, float],
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
                        mujoco.mjtObj.mjOBJ_JOINT, joint_name, "joint"
                    )
                ]
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int64,
        )
        self._arm_ctrl_indices = np.array(
            [
                self._require_named_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name, "actuator"
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
            low=-0.1,
            high=0.1,
            shape=(len(self.ACTION_COMPONENTS),),
            dtype=np.float32,
        )
        self._ik_target_pos = np.zeros(3, dtype=np.float64)
        self._ik_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_ik_result: IKResult | None = None
        self._ik_failure_count = 0
        self._reset_ik_state()

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

    def _get_active_obj_info(self) -> dict[str, int | str]:
        return self.object_info[self.active_obj_name]

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

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
        ee_pos, ee_quat = self._get_ee_pose()
        action = np.asarray(action, dtype=np.float64)
        delta_pos = self._cartesian_action_scale * action[:3]
        delta_rpy = self._cartesian_rotation_scale_rad * action[3:6]
        # print(f"Delta pos: {delta_pos}, delta_rpy: {delta_rpy}")
        target_pos = np.clip(
            np.asarray(ee_pos, dtype=np.float64) + delta_pos,
            self._ik_workspace_low,
            self._ik_workspace_high,
        )
        current_rpy = _quat_to_euler_xyz(np.asarray(ee_quat, dtype=np.float64))
        target_rpy = np.array(
            [self._wrap_to_pi(value) for value in current_rpy + delta_rpy],
            dtype=np.float64,
        )
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
        # print(f"IK target pos: {target_pos}, target_quat: {target_quat}")
        ik_result = self._ik_solver.solve(
            target_pos,
            target_quat,
            initial_q=self._current_arm_joint_positions(),
            position_only=False,
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
        target_ctrl[self._arm_ctrl_indices] = ik_result.q_rad
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
        ctrl[self.gripL_act_id] = 0.01
        ctrl[self.gripR_act_id] = -0.01

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[self.gripL_act_id] = -0.02
        ctrl[self.gripR_act_id] = 0.02

    def _sample_object_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = self.np_random.uniform(*self._object_x_range)
        y = self.np_random.uniform(*self._object_y_range)
        z = self._object_z
        yaw = self.np_random.uniform(
            -self._object_yaw_limit_rad,
            self._object_yaw_limit_rad,
        )
        pos = np.array([x, y, z], dtype=np.float64)
        quat = self._yaw_to_quat(yaw)
        return pos, quat, float(yaw)

    def _update_target_site(self) -> None:
        target_pos = self.initial_obj_site_pos + np.array(
            [0.0, 0.0, self._lift_height],
            dtype=np.float64,
        )
        self.model.site_pos[self.target_site_id] = target_pos

    def _apply_grasp_heuristic(self, ctrl: np.ndarray) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            obj_pos,
            obj_quat,
        )

        dist = float(np.linalg.norm(ee_obj_pos_error))
        angle = float(np.linalg.norm(ee_obj_rot_error))
        should_close = (
            dist < self._grasp_close_distance and angle < self._grasp_close_angle_rad
        )
        keep_closed = self.grasp_latched and dist < self._grasp_release_distance
        self.grasp_latched = bool(should_close or keep_closed)
        self.last_grasp_should_close = bool(should_close)
        self.last_grasp_dist = dist
        self.last_grasp_angle = angle

        if self.grasp_latched:
            self._set_closed_gripper_target(ctrl)
        else:
            self._set_open_gripper_target(ctrl)

    def _control_interp_alpha(self, step_index: int, total_steps: int) -> float:
        alpha = float(step_index + 1) / float(max(total_steps, 1))
        if self._control_interpolation_profile == "smoothstep":
            return float(alpha * alpha * (3.0 - 2.0 * alpha))
        return alpha

    def _apply_control_smoothing(
        self,
        start_ctrl: np.ndarray,
        target_ctrl: np.ndarray,
    ) -> np.ndarray:
        smoothed_ctrl = start_ctrl + self._control_smoothing_factor * (
            target_ctrl - start_ctrl
        )
        return np.clip(smoothed_ctrl, self._ctrl_low, self._ctrl_high)

    def _record_control_interpolation_debug(
        self,
        start_ctrl: np.ndarray,
        target_ctrl: np.ndarray,
        applied_target_ctrl: np.ndarray,
    ) -> None:
        self._last_control_start = start_ctrl.copy()
        self._last_control_target = target_ctrl.copy()
        self._last_control_applied_target = applied_target_ctrl.copy()

        control_delta = applied_target_ctrl - start_ctrl
        arm_control_delta = control_delta[self._arm_ctrl_indices]
        self._last_control_delta_norm = float(np.linalg.norm(control_delta))
        self._last_arm_control_delta_max_abs = float(
            np.max(np.abs(arm_control_delta)) if arm_control_delta.size else 0.0
        )

    def _do_interpolated_simulation(self, target_ctrl: np.ndarray) -> None:
        target_ctrl = np.asarray(target_ctrl, dtype=np.float64).reshape(-1)
        if target_ctrl.shape != (self.model.nu,):
            raise ValueError(
                f"Expected control shape {(self.model.nu,)}, got {target_ctrl.shape}."
            )

        start_ctrl = self.data.ctrl.copy()
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        applied_target_ctrl = self._apply_control_smoothing(start_ctrl, target_ctrl)
        self._record_control_interpolation_debug(
            start_ctrl,
            target_ctrl,
            applied_target_ctrl,
        )

        if not self._control_interpolation or self.frame_skip <= 1:
            self.do_simulation(applied_target_ctrl, self.frame_skip)
            return

        for frame_idx in range(self.frame_skip):
            alpha = self._control_interp_alpha(frame_idx, self.frame_skip)
            self.data.ctrl[:] = start_ctrl + alpha * (
                applied_target_ctrl - start_ctrl
            )
            mujoco.mj_step(self.model, self.data)

    def step(self, action):
        self.current_step += 1
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)
        self._apply_grasp_heuristic(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self._do_interpolated_simulation(target_ctrl)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = self.success_counter >= self._success_steps_required
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
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

        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        obj_target_dist = float(np.linalg.norm(obj_target_pos_error))
        obj_target_angle = float(np.linalg.norm(obj_target_rot_error))

        reward_ee_obj_dist = -ee_obj_dist * self._reward_ee_obj_dist_weight
        reward_ee_obj_dist_tanh = (
            1.0 - float(np.tanh(ee_obj_dist / self._ee_obj_tanh_scale))
        ) * self._reward_ee_obj_dist_tanh_weight
        reward_ee_obj_orient = -ee_obj_angle * self._reward_ee_obj_orient_weight

        reward_obj_target_dist = -obj_target_dist * self._reward_obj_target_dist_weight
        reward_obj_target_dist_tanh = (
            1.0 - float(np.tanh(obj_target_dist / self._obj_target_tanh_scale))
        ) * self._reward_obj_target_dist_tanh_weight
        reward_obj_target_orient = (
            -obj_target_angle * self._reward_obj_target_orient_weight
        )

        target_reached = (
            obj_target_dist < self._success_distance
            and obj_target_angle < self._success_angle_rad
        )
        if target_reached:
            self.success_counter += 1
            reward_target_bonus = self._reward_target_bonus
        else:
            self.success_counter = 0
            reward_target_bonus = 0.0

        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )

        reward = (
            reward_ee_obj_dist
            + reward_ee_obj_dist_tanh
            + reward_ee_obj_orient
            + reward_obj_target_dist
            + reward_obj_target_dist_tanh
            + reward_obj_target_orient
            + reward_target_bonus
            + control_penalty
        )

        reward_info = {
            "active_object": self.active_obj_name,
            "ee_object_dist": ee_obj_dist,
            "ee_object_rot_error": ee_obj_angle,
            "object_target_dist": obj_target_dist,
            "object_target_rot_error": obj_target_angle,
            "target_reached": int(target_reached),
            "success_counter": int(self.success_counter),
            "reward_ee_object_dist": float(reward_ee_obj_dist),
            "reward_ee_object_dist_tanh": float(reward_ee_obj_dist_tanh),
            "reward_ee_object_orient": float(reward_ee_obj_orient),
            "reward_object_target_dist": float(reward_obj_target_dist),
            "reward_object_target_dist_tanh": float(reward_obj_target_dist_tanh),
            "reward_object_target_orient": float(reward_obj_target_orient),
            "reward_target_bonus": float(reward_target_bonus),
            "control_penalty": float(control_penalty),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
            "control_delta_norm": float(self._last_control_delta_norm),
            "arm_control_delta_max_abs": float(self._last_arm_control_delta_max_abs),
            # Backward-compatible metric names used by the training/eval overlays.
            "reward_dist": float(reward_ee_obj_dist),
            "reward_dist_tanh": float(reward_ee_obj_dist_tanh),
            "reward_orient": float(reward_ee_obj_orient),
            "reward_target": float(reward_obj_target_dist),
            "reward_target_tanh": float(reward_obj_target_dist_tanh),
            "reward_target_orient": float(reward_obj_target_orient),
        }
        return float(reward), reward_info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.active_obj_name = str(self.np_random.choice(self.object_names))
        obj_pos, obj_quat, yaw = self._sample_object_pose()
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for obj_name in self.object_names:
            info = self.object_info[obj_name]
            qposadr = int(info["qposadr"])
            dofadr = int(info["dofadr"])

            if obj_name == self.active_obj_name:
                qpos[qposadr : qposadr + 3] = obj_pos
                qpos[qposadr + 3 : qposadr + 7] = obj_quat
            else:
                qpos[qposadr : qposadr + 3] = np.array(
                    [6.0, 1.0, 1.0],
                    dtype=np.float64,
                )
                qpos[qposadr + 3 : qposadr + 7] = identity_quat

            qvel[dofadr : dofadr + 6] = 0.0

        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)

        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_ctrl_indices] = qpos[self._arm_qpos_indices]
        self._set_open_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        self._update_target_site()
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.grasp_latched = False
        self.last_grasp_should_close = False
        self.last_grasp_dist = np.inf
        self.last_grasp_angle = np.inf
        self.sampled_object_yaw = float(yaw)
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        self._reset_ik_state()
        self._record_control_interpolation_debug(
            self.data.ctrl.copy(),
            self.data.ctrl.copy(),
            self.data.ctrl.copy(),
        )

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
        gripper_ctrl = self.data.ctrl[[self.gripL_act_id, self.gripR_act_id]].copy()
        gripper_closed = np.array(
            [1.0 if self.gripper_state == "closed" else 0.0],
            dtype=np.float64,
        )

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
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
        lift_height = float(obj_pos[2] - self.initial_obj_site_pos[2])

        metrics = np.array(
            [
                np.linalg.norm(ee_obj_pos_error),
                np.linalg.norm(ee_obj_rot_error),
                np.linalg.norm(obj_target_pos_error),
                np.linalg.norm(obj_target_rot_error),
                lift_height,
                float(self.grasp_latched),
                float(self._ik_failure_count),
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
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("ee_object_pos_error", ee_obj_pos_error),
            ("ee_object_rot_error", ee_obj_rot_error),
            ("object_target_pos_error", obj_target_pos_error),
            ("object_target_rot_error", obj_target_rot_error),
            ("ik_target_pos", self._ik_target_pos),
            ("ik_target_quat", self._ik_target_quat),
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
        debug_state = {
            "ik_target_pos": self._ik_target_pos.copy(),
            "ik_target_quat": self._ik_target_quat.copy(),
            "ik_target_rpy_deg": np.rad2deg(_quat_to_euler_xyz(self._ik_target_quat)),
            "ik_failure_count": int(self._ik_failure_count),
        }
        if self._last_ik_result is None:
            debug_state.update(
                {
                    "ik_success": None,
                    "ik_iterations": 0,
                    "ik_attempt": 0,
                }
            )
            return debug_state

        debug_state.update(
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
        return debug_state

    def export_config(self) -> dict:
        config = export_env_config(self, self._get_obs_components())
        config["action"]["controller"] = "standalone_cartesian_ik"
        config["action"]["action_components"] = list(self.ACTION_COMPONENTS)
        config["action"]["control_interpolation"] = {
            "enabled": bool(self._control_interpolation),
            "profile": self._control_interpolation_profile,
            "physics_steps": int(self.frame_skip),
            "smoothing_factor": float(self._control_smoothing_factor),
        }
        config["action"]["cartesian_action_scale_m"] = float(
            self._cartesian_action_scale
        )
        config["action"]["cartesian_rotation_scale_deg"] = float(
            np.rad2deg(self._cartesian_rotation_scale_rad)
        )
        config["action"]["ik_workspace_low"] = self._ik_workspace_low.tolist()
        config["action"]["ik_workspace_high"] = self._ik_workspace_high.tolist()
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
        config["action"]["ik_joint_names"] = list(self._ik_solver.joint_names)
        config["action"]["ee_site_name"] = self.ee_site_name
        config["action"]["gripper_policy"] = "distance_orientation_heuristic"
        config["action"]["gripper_open_target"] = [0.01, -0.01]
        config["action"]["gripper_closed_target"] = [-0.02, 0.02]
        config["task"]["target_mode"] = "lift_active_object_above_initial_site"
        config["task"]["target_site_name"] = self.target_site_name
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
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
        obj_target_dist = float(np.linalg.norm(obj_target_pos_error))
        obj_target_angle = float(np.linalg.norm(obj_target_rot_error))
        target_reached = (
            obj_target_dist < self._success_distance
            and obj_target_angle < self._success_angle_rad
        )

        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": obj_target_dist,
            "obj_target_angle_rad": obj_target_angle,
            "lift_height": float(obj_pos[2] - self.initial_obj_site_pos[2]),
            "required_lift_height": float(self._lift_height),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "gripper_should_close": bool(self.last_grasp_should_close),
            "grasp_latched": bool(self.grasp_latched),
            "gripper_state": self.gripper_state,
            "gripper_qpos": self.data.qpos[[self.gripL_qadr, self.gripR_qadr]].copy(),
            "gripper_ctrl": self.data.ctrl[
                [self.gripL_act_id, self.gripR_act_id]
            ].copy(),
            "success_counter": int(self.success_counter),
            "target_reached": bool(target_reached),
            "last_action": self.last_action.copy(),
            "control_interpolation": bool(self._control_interpolation),
            "control_interpolation_profile": self._control_interpolation_profile,
            "control_smoothing_factor": float(self._control_smoothing_factor),
            "last_control_start": self._last_control_start.copy(),
            "last_control_target": self._last_control_target.copy(),
            "last_control_applied_target": self._last_control_applied_target.copy(),
            "last_control_delta_norm": float(self._last_control_delta_norm),
            "last_arm_control_delta_max_abs": float(
                self._last_arm_control_delta_max_abs
            ),
            **self._get_ik_debug_state(),
        }

    def render(self):
        return super().render()
