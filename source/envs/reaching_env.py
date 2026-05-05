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
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "reaching.xml"


class ReachingEnvIK(MujocoEnv, utils.EzPickle):
    """Standalone target-reaching env controlled through Cartesian IK actions."""

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
        reward_distance_weight: float = 5.0,
        reward_distance_tanh_weight: float = 2.0,
        reward_orientation_weight: float = 2.0,
        reward_target_bonus: float = 20.0,
        control_penalty_weight: float = 0.001,
        distance_tanh_scale: float = 0.05,
        success_distance: float = 0.01,
        success_angle_deg: float = 10.0,
        success_requires_orientation: bool = True,
        success_steps_required: int = 5,
        max_episode_steps: int = 200,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.22, 0.015),
        ik_workspace_high: tuple[float, float, float] = (0.35, 0.22, 0.30),
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
        target_x_range: tuple[float, float] = (0.10, 0.27),
        target_y_range: tuple[float, float] = (-0.14, 0.14),
        target_z: float = 0.015,
        target_yaw_range: tuple[float, float] = (-np.pi / 2, np.pi / 2),
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        target_body_name: str = "target_body",
        ee_frame_body_name: str = "ee_frame_vis",
        target_frame_body_name: str = "target_frame_vis",
        **kwargs,
    ):
        self._init_config = capture_init_config(locals())
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_distance_weight,
            reward_distance_tanh_weight,
            reward_orientation_weight,
            reward_target_bonus,
            control_penalty_weight,
            distance_tanh_scale,
            success_distance,
            success_angle_deg,
            success_requires_orientation,
            success_steps_required,
            max_episode_steps,
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
            debug_ik,
            target_x_range,
            target_y_range,
            target_z,
            target_yaw_range,
            ee_site_name,
            target_site_name,
            target_body_name,
            ee_frame_body_name,
            target_frame_body_name,
            **kwargs,
        )

        self._reward_distance_weight = float(reward_distance_weight)
        self._reward_distance_tanh_weight = float(reward_distance_tanh_weight)
        self._reward_orientation_weight = float(reward_orientation_weight)
        self._reward_target_bonus = float(reward_target_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._distance_tanh_scale = float(distance_tanh_scale)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_requires_orientation = bool(success_requires_orientation)
        self._success_steps_required = int(success_steps_required)
        self.max_episode_steps = int(max_episode_steps)
        self._control_interpolation_steps = max(1, int(control_interpolation_steps))
        self._max_joint_ctrl_delta_rad = np.deg2rad(float(max_joint_ctrl_delta_deg))
        self._smooth_cartesian_target = bool(smooth_cartesian_target)
        self._debug_ik = bool(debug_ik)
        self._target_x_range = tuple(float(value) for value in target_x_range)
        self._target_y_range = tuple(float(value) for value in target_y_range)
        self._target_z = float(target_z)
        self._target_yaw_range = tuple(float(value) for value in target_yaw_range)
        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)
        self.target_body_name = str(target_body_name)
        self.ee_frame_body_name = str(ee_frame_body_name)
        self.target_frame_body_name = str(target_frame_body_name)
        self.object_names: list[str] = []

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

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            camera_name="watching",
            **kwargs,
        )

        self.ee_site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name, "site"
        )
        self.target_site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.target_site_name, "site"
        )
        self.target_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY, self.target_body_name, "body"
        )
        self.ee_frame_body_id = self._optional_named_id(
            mujoco.mjtObj.mjOBJ_BODY, self.ee_frame_body_name
        )
        self.target_frame_body_id = self._optional_named_id(
            mujoco.mjtObj.mjOBJ_BODY, self.target_frame_body_name
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
        self.sampled_target_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_yaw = 0.0
        self.applied_target_yaw = 0.0

        self.sync_visual_frames()
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

    def _optional_named_id(self, obj_type: mujoco.mjtObj, name: str) -> int | None:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        return None if obj_id < 0 else obj_id

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

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _current_arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self._arm_qpos_indices],
            dtype=np.float64,
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
        # print(f"Action: {action}, Delta Pos: {delta_pos}, Delta RPY: {delta_rpy}")

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
        print(f"Target EE: {target_pos}")

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
        self._set_open_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self.last_action = action.astype(np.float32)
        self._ik_target_pos = target_pos.copy()
        self._ik_target_quat = target_quat.copy()
        self._last_ik_result = ik_result
        if not ik_result.success:
            self._ik_failure_count += 1

        return action, target_ctrl, ik_result

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        ctrl[self.gripL_act_id] = 0.01
        ctrl[self.gripR_act_id] = -0.01

    def _sample_target_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = self.np_random.uniform(*self._target_x_range)
        y = self.np_random.uniform(*self._target_y_range)
        yaw = self.np_random.uniform(*self._target_yaw_range)
        pos = np.array([x, y, self._target_z], dtype=np.float64)
        quat = self._yaw_to_quat(yaw)
        return pos, quat, float(yaw)

    def _set_target_body_pose(self, pos: np.ndarray, quat: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        quat = _normalize_quat(np.asarray(quat, dtype=np.float64).reshape(4))
        pos[2] = self._target_z
        self.model.body_pos[self.target_body_id] = pos
        self.model.body_quat[self.target_body_id] = quat

    def _set_mocap_body_pose(
        self,
        body_id: int | None,
        pos: np.ndarray,
        quat: np.ndarray,
    ) -> None:
        if body_id is None:
            return
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            return
        self.data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64).reshape(3)
        self.data.mocap_quat[mocap_id] = _normalize_quat(
            np.asarray(quat, dtype=np.float64).reshape(4)
        )

    def sync_visual_frames(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        target_pos, target_quat = self._get_target_pose()
        self._set_mocap_body_pose(self.ee_frame_body_id, ee_pos, ee_quat)
        self._set_mocap_body_pose(self.target_frame_body_id, target_pos, target_quat)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.current_step += 1
        print(f"Step {self.current_step}")
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)

        start_ctrl = self.data.ctrl.copy()
        for interp_idx in range(1, self._control_interpolation_steps + 1):
            alpha = interp_idx / self._control_interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            smooth_ctrl = np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high)
            self.do_simulation(smooth_ctrl, self.frame_skip)

        self.sync_visual_frames()
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = self.success_counter >= self._success_steps_required
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _target_reached(self, dist: float, angle: float) -> bool:
        if dist >= self._success_distance:
            return False
        if self._success_requires_orientation and angle >= self._success_angle_rad:
            return False
        return True

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_target_pos_error, ee_target_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            target_pos,
            target_quat,
        )

        ee_target_dist = float(np.linalg.norm(ee_target_pos_error))
        print(f"EE-Target Distance: {ee_target_dist}")
        ee_target_angle = float(np.linalg.norm(ee_target_rot_error))
        reward_distance = -ee_target_dist * self._reward_distance_weight
        reward_distance_tanh = (
            1.0 - float(np.tanh(ee_target_dist / self._distance_tanh_scale))
        ) * self._reward_distance_tanh_weight
        reward_orientation = -ee_target_angle * self._reward_orientation_weight

        target_reached = self._target_reached(ee_target_dist, ee_target_angle)
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
            reward_distance
            + reward_distance_tanh
            + reward_orientation
            + reward_target_bonus
            + control_penalty
        )

        reward_info = {
            "ee_target_dist": ee_target_dist,
            "ee_target_angle_rad": ee_target_angle,
            "target_dist": ee_target_dist,
            "target_reached": int(target_reached),
            "success_counter": int(self.success_counter),
            "reward_distance": float(reward_distance),
            "reward_distance_tanh": float(reward_distance_tanh),
            "reward_orientation": float(reward_orientation),
            "reward_target_bonus": float(reward_target_bonus),
            "control_penalty": float(control_penalty),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
        }
        return float(reward), reward_info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        target_pos, target_quat, sampled_yaw = self._sample_target_pose()
        self._set_target_body_pose(target_pos, target_quat)
        self.set_state(qpos, qvel)

        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_ctrl_indices] = qpos[self._arm_qpos_indices]
        self._set_open_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.sampled_target_site_pos = target_pos.copy()
        self.sampled_target_yaw = float(sampled_yaw)
        self.applied_target_yaw = float(self._quat_to_yaw(self._get_target_pose()[1]))
        self._reset_ik_state()
        self.sync_visual_frames()

        return self._get_obs()

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
        qpos = self.data.qpos
        qvel = self.data.qvel
        ee_pos, ee_quat = self._get_ee_pose()
        print(f"EE Pos: {ee_pos}, EE Quat: {ee_quat}")
        target_pos, target_quat = self._get_target_pose()
        print(f"Target Pos: {target_pos}, Target Quat: {target_quat}")
        ee_target_pos_error, ee_target_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            target_pos,
            target_quat,
        )
        target_delta_euler = self._wrap_vector_to_pi(
            _quat_to_euler_xyz(target_quat) - _quat_to_euler_xyz(ee_quat)
        )

        metrics = np.array(
            [
                np.linalg.norm(ee_target_pos_error),
                np.linalg.norm(ee_target_rot_error),
                float(self.success_counter),
                float(self._ik_failure_count),
                float(self.sampled_target_yaw),
                float(self._target_z),
            ],
            dtype=np.float64,
        )

        return [
            ("robot_qpos", qpos),
            ("robot_qvel", qvel),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("ee_target_pos_error", ee_target_pos_error),
            ("ee_target_rot_error", ee_target_rot_error),
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
        debug_state = {
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
        config["action"]["gripper_policy"] = "fixed_open"
        config["task"]["target_mode"] = "random_site_reaching"
        config["task"]["target_site_name"] = self.target_site_name
        config["task"]["target_body_name"] = self.target_body_name
        config["task"]["target_x_range"] = list(self._target_x_range)
        config["task"]["target_y_range"] = list(self._target_y_range)
        config["task"]["target_z"] = float(self._target_z)
        config["task"]["target_yaw_range"] = list(self._target_yaw_range)
        config["task"]["success_requires_orientation"] = bool(
            self._success_requires_orientation
        )
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_target_pos_error, ee_target_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            target_pos,
            target_quat,
        )
        ee_target_dist = float(np.linalg.norm(ee_target_pos_error))
        ee_target_angle = float(np.linalg.norm(ee_target_rot_error))
        target_reached = self._target_reached(ee_target_dist, ee_target_angle)
        target_delta_euler = self._wrap_vector_to_pi(
            _quat_to_euler_xyz(target_quat) - _quat_to_euler_xyz(ee_quat)
        )

        return {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "ee_target_pos_error": ee_target_pos_error,
            "ee_target_rot_error": ee_target_rot_error,
            "ee_target_dist": ee_target_dist,
            "ee_target_angle_rad": ee_target_angle,
            "target_dist": ee_target_dist,
            "target_delta_euler_deg": np.rad2deg(target_delta_euler),
            "target_yaw": float(self._quat_to_yaw(target_quat)),
            "sampled_target_yaw": float(self.sampled_target_yaw),
            "applied_target_yaw": float(self.applied_target_yaw),
            "sampled_target_site_pos": self.sampled_target_site_pos.copy(),
            "target_z": float(target_pos[2]),
            "fixed_target_z": float(self._target_z),
            "success_counter": int(self.success_counter),
            "target_reached": bool(target_reached),
            "success_requires_orientation": bool(self._success_requires_orientation),
            "last_action": self.last_action.copy(),
            **self._get_ik_debug_state(),
        }

    def render(self):
        self.sync_visual_frames()
        return super().render()


ReachingEnv = ReachingEnvIK
