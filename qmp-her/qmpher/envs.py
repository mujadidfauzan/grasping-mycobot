from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, Dict

from .utils import ensure_project_root_on_path

ensure_project_root_on_path()

from script.inverse_kinematics import (  # noqa: E402
    IKResult,
    MyCobotIK,
    _normalize_quat,
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class QMPGraspInsertEnv(MujocoEnv, utils.EzPickle):
    """Standalone Q-switch grasp+insert environment.

    This env does not subclass InsertTargetEnvIK. It owns its reset logic,
    manual gripper policy, sparse goal reward, termination, observation, and IK
    action path. The observation layout intentionally mirrors InsertTargetEnvIK
    so a trained InsertTargetEnvIK primitive can still propose candidate 6-DoF
    actions.
    """

    ACTION_COMPONENTS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
    }

    def __init__(
        self,
        xml_file: str,
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        close_distance: float = 0.01,
        close_angle_deg: float = 10.0,
        release_distance: float = 0.01,
        release_angle_deg: float = 10.0,
        success_steps_required: int = 5,
        success_distance: float = 0.008,
        success_angle_deg: float = 10.0,
        her_success_reward: float = 0.0,
        her_failure_reward: float = -1.0,
        max_episode_steps: int = 500,
        terminate_lost_object_distance: float = 0.08,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.24, 0.00),
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
        object_x_range: tuple[float, float] = (0.18, 0.27),
        object_y_range: tuple[float, float] = (-0.14, 0.14),
        object_z: float = 0.025,
        object_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        target_height_above_place: float = 0.035,
        target_x_range: tuple[float, float] = (0.18, 0.27),
        target_y_range: tuple[float, float] = (-0.14, 0.14),
        target_place_z: float = 0.0,
        target_place_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        min_initial_object_target_distance: float = 0.08,
        target_resample_attempts: int = 100,
        reset_settle_steps: int = 20,
        object_name: str = "box",
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        target_body_name: str = "target_body",
        place_body_name: str = "cube_place",
        place_site_name: str = "cube_place_site",
        place_geom_name: str = "cube_place_geom",
        debug_ik: bool = False,
        **kwargs: Any,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, default_camera_config)

        self._close_distance = float(close_distance)
        self._close_angle_rad = (
            None if close_angle_deg is None else np.deg2rad(float(close_angle_deg))
        )
        self._release_distance = float(release_distance)
        self._release_angle_rad = np.deg2rad(float(release_angle_deg))
        self._success_steps_required = max(1, int(success_steps_required))
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._her_success_reward = float(her_success_reward)
        self._her_failure_reward = float(her_failure_reward)
        self.max_episode_steps = int(max_episode_steps)
        self._terminate_lost_object_distance = float(terminate_lost_object_distance)
        self._control_interpolation_steps = max(1, int(control_interpolation_steps))
        self._max_joint_ctrl_delta_rad = np.deg2rad(float(max_joint_ctrl_delta_deg))
        self._smooth_cartesian_target = bool(smooth_cartesian_target)
        self._object_x_range = tuple(float(value) for value in object_x_range)
        self._object_y_range = tuple(float(value) for value in object_y_range)
        self._object_z = float(object_z)
        self._object_yaw_range = tuple(float(value) for value in object_yaw_range)
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
        self._reset_settle_steps = max(0, int(reset_settle_steps))
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
        self.object_info = {self.object_name: self._build_object_info(self.object_name)}
        self.object_one_hot = {self.object_name: np.array([1.0], dtype=np.float64)}
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
            mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name, "site"
        )
        self.target_site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.target_site_name, "site"
        )
        self.target_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY, self.target_body_name, "body"
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
            xml_file=xml_file,
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
        self.release_steps = 0
        self.release_event_count = 0
        self.success_counter = 0
        self.qswitch_insert_active = False
        self.gripper_phase = "open"
        self.gripper_state = "open"
        self.last_manual_event = "init"
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_pos = self._default_place_pos.copy()
        self.sampled_target_place_quat = self._default_place_quat.copy()
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0
        self.sampled_target_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self._last_target_resample_attempts = 0
        self._last_target_resample_distance = np.inf
        self._last_qswitch_debug: dict[str, Any] = {}

        self._sync_target_site_to_active_place()
        self._set_open_gripper_target(self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)
        self._reset_ik_state()

        dummy_flat_obs = self._get_flat_obs()
        dummy_goal = self._get_achieved_goal()
        self.observation_space = Dict(
            {
                "observation": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=dummy_flat_obs.shape,
                    dtype=np.float32,
                ),
                "achieved_goal": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=dummy_goal.shape,
                    dtype=np.float32,
                ),
                "desired_goal": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=dummy_goal.shape,
                    dtype=np.float32,
                ),
            }
        )
        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
            "render_fps": int(
                np.round(1.0 / (self.dt * self._control_interpolation_steps))
            ),
        }

    def _require_named_id(self, obj_type: mujoco.mjtObj, name: str, label: str) -> int:
        obj_id = int(mujoco.mj_name2id(self.model, obj_type, name))
        if obj_id < 0:
            raise ValueError(f"MuJoCo {label} `{name}` not found in model.")
        return obj_id

    def _build_object_info(self, obj_name: str) -> dict[str, int | str]:
        body_name = f"obj_{obj_name}"
        joint_name = f"obj_{obj_name}_joint"
        site_name = f"obj_{obj_name}_ref"
        geom_name = f"obj_{obj_name}_geom"
        body_id = self._require_named_id(mujoco.mjtObj.mjOBJ_BODY, body_name, "body")
        joint_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_JOINT, joint_name, "joint"
        )
        site_id = self._require_named_id(mujoco.mjtObj.mjOBJ_SITE, site_name, "site")
        geom_id = self._require_named_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name, "geom")
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
            mujoco.mjtObj.mjOBJ_BODY, self.place_body_name, "body"
        )
        site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.place_site_name, "site"
        )
        geom_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_GEOM, self.place_geom_name, "geom"
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
        self._ik_solver = MyCobotIK(xml_file=xml_file, ee_site_name=self.ee_site_name)
        self._arm_joint_names = tuple(self._ik_solver.joint_names)
        self._arm_qpos_indices = np.array(
            [
                self.model.jnt_qposadr[
                    self._require_named_id(mujoco.mjtObj.mjOBJ_JOINT, name, "joint")
                ]
                for name in self._arm_joint_names
            ],
            dtype=np.int64,
        )
        self._arm_ctrl_indices = np.array(
            [
                self._require_named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name, "actuator")
                for name in self._arm_joint_names
            ],
            dtype=np.int64,
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
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64
        )

    @staticmethod
    def _wrap_to_pi(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _wrap_vector_to_pi(self, angles_rad: np.ndarray) -> np.ndarray:
        return np.array(
            [self._wrap_to_pi(float(v)) for v in angles_rad], dtype=np.float64
        )

    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        quat = _normalize_quat(quat)
        w, x, y, z = quat
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def _get_site_quat(self, site_name: str) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site(site_name).xmat)
        return _normalize_quat(quat)

    def _get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self.data.site(site_name).xpos.copy(), self._get_site_quat(site_name)

    def _rotation_vector(
        self, source_quat: np.ndarray, target_quat: np.ndarray
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
        return xyz / sin_half * angle

    def _get_pose_error(
        self,
        source_pos: np.ndarray,
        source_quat: np.ndarray,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(target_pos, dtype=np.float64)
            - np.asarray(source_pos, dtype=np.float64),
            self._rotation_vector(source_quat, target_quat),
        )

    def _get_pose_in_body_frame(
        self,
        world_pos: np.ndarray,
        world_quat: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        body_quat_conj = self._quat_conjugate(_normalize_quat(body_quat))
        local_pos = self._quat_rotate_vector(
            body_quat_conj,
            np.asarray(world_pos, dtype=np.float64)
            - np.asarray(body_pos, dtype=np.float64),
        )
        local_quat = _normalize_quat(self._quat_multiply(body_quat_conj, world_quat))
        return local_pos, local_quat

    def _get_active_obj_info(self) -> dict[str, int | str]:
        return self.object_info[self.active_obj_name]

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.place_site_name)

    def _get_insert_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Pose that represents the object being inside the physical place."""
        return self._get_place_site_pose()

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _current_arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self._arm_qpos_indices], dtype=np.float64
        ).copy()

    def _reset_ik_state(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        self._ik_target_pos = ee_pos.copy()
        self._ik_target_quat = _normalize_quat(ee_quat)
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
        target_pos = np.clip(
            base_pos + delta_pos, self._ik_workspace_low, self._ik_workspace_high
        )
        target_rpy = self._wrap_vector_to_pi(_quat_to_euler_xyz(base_quat) + delta_rpy)
        return target_pos, _quat_from_euler_xyz(*target_rpy)

    def _ik_action_to_target_ctrl(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, IKResult]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != self.action_space.shape:
            raise ValueError(
                f"Expected action shape {self.action_space.shape}, got {action.shape}."
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos, target_quat = self._compute_ik_target(action)

        # print(f"Target Pos : {target_pos}, Target Quat : {target_quat}", flush=True)

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

        # PENTING:
        # Pakai q_rad meskipun ik_result.success False,
        # sama seperti GraspingEnvIK.
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

        # Untuk test awal, commit target seperti GraspingEnvIK.
        # Ini membuat smooth target bisa bergerak step-by-step.
        self._ik_target_pos = target_pos.copy()
        self._ik_target_quat = target_quat.copy()

        self._last_ik_result = ik_result

        if ik_result.success:
            # print("IK Sukses", flush=True)
            pass
        else:
            self._ik_failure_count += 1
            # print("IK Gagal, tapi tetap apply q_rad best-effort", flush=True)

        return action, target_ctrl, ik_result

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[self.gripL_act_id] = self._gripper_open_target[0]
        ctrl[self.gripR_act_id] = self._gripper_open_target[1]

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[self.gripL_act_id] = self._gripper_closed_target[0]
        ctrl[self.gripR_act_id] = self._gripper_closed_target[1]

    def _apply_manual_gripper_to_ctrl(self, ctrl: np.ndarray) -> None:
        if self.gripper_phase in {"open", "released"}:
            self._set_open_gripper_target(ctrl)
        else:
            self._set_closed_gripper_target(ctrl)

    def _sample_object_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = float(self.np_random.uniform(*self._object_x_range))
        y = float(self.np_random.uniform(*self._object_y_range))
        yaw = float(self.np_random.uniform(*self._object_yaw_range))
        return (
            np.array([x, y, self._object_z], dtype=np.float64),
            self._yaw_to_quat(yaw),
            yaw,
        )

    def _sample_target_place_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        place_pos = self._default_place_pos.copy()
        place_pos[0] = self.np_random.uniform(*self._target_x_range)
        place_pos[1] = self.np_random.uniform(*self._target_y_range)
        place_pos[2] = self._target_place_z
        yaw = float(self.np_random.uniform(*self._target_place_yaw_range))
        return place_pos, self._yaw_to_quat(yaw), yaw

    def _target_pos_for_place_pose(
        self, place_pos: np.ndarray, place_quat: np.ndarray
    ) -> np.ndarray:
        local_pos = self._place_site_local_pos.copy()
        local_pos[2] += self._target_height_above_place
        return np.asarray(place_pos, dtype=np.float64) + self._quat_rotate_vector(
            place_quat, local_pos
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
        return (*best_pose, best_distance, self._target_resample_attempts)

    def _set_place_pose_in_model(
        self, place_pos: np.ndarray, place_quat: np.ndarray
    ) -> None:
        body_id = int(self.place_info["body_id"])
        self.model.body_pos[body_id] = np.asarray(place_pos, dtype=np.float64).reshape(
            3
        )
        self.model.body_quat[body_id] = _normalize_quat(
            np.asarray(place_quat, dtype=np.float64)
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
        target_local_pos = self.model.site_pos[place_site_id].copy()
        target_local_pos[2] += self._target_height_above_place
        self.model.site_pos[self.target_site_id] = target_local_pos
        self.model.site_quat[self.target_site_id] = self.model.site_quat[
            place_site_id
        ].copy()

    def _box_place_metrics(self) -> dict[str, np.ndarray | float]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        place_body = self.data.body(self.place_body_name)
        obj_local_pos, obj_local_quat = self._get_pose_in_body_frame(
            obj_pos,
            obj_quat,
            place_body.xpos.copy(),
            _normalize_quat(place_body.xquat.copy()),
        )
        target_local_pos = self._place_site_local_pos.copy()
        target_local_quat = self._place_site_local_quat.copy()
        local_pos_error, local_rot_error = self._get_pose_error(
            obj_local_pos,
            obj_local_quat,
            target_local_pos,
            target_local_quat,
        )
        radial_error = float(np.linalg.norm(local_pos_error[:2]))
        height_error = float(local_pos_error[2])
        angle_error = float(np.linalg.norm(local_rot_error))
        pose_aligned = bool(
            radial_error < self._success_distance
            and abs(height_error) < self._success_distance
            and angle_error < self._success_angle_rad
        )
        return {
            "object_local_pos": obj_local_pos,
            "object_local_quat": obj_local_quat,
            "target_local_pos": target_local_pos,
            "target_local_quat": target_local_quat,
            "object_target_local_pos_error": local_pos_error,
            "object_target_local_rot_error": local_rot_error,
            "object_target_local_radial_error": radial_error,
            "object_target_local_height_error": height_error,
            "object_target_local_angle_error": angle_error,
            "place_pose_aligned": int(pose_aligned),
        }

    def _task_metrics(self) -> dict[str, Any]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        lift_height = float(obj_pos[2] - self.initial_obj_site_pos[2])
        box_place_metrics = self._box_place_metrics()
        place_pose_aligned = bool(box_place_metrics["place_pose_aligned"])
        return {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "ee_obj_dist": ee_obj_dist,
            "ee_obj_angle": ee_obj_angle,
            "target_dist": target_dist,
            "target_angle": target_angle,
            "lift_height": lift_height,
            "target_pose_aligned": bool(
                target_dist < self._success_distance
                and target_angle < self._success_angle_rad
            ),
            "place_pose_aligned": place_pose_aligned,
            "place_radial_error": float(
                box_place_metrics["object_target_local_radial_error"]
            ),
            "place_height_error": float(
                box_place_metrics["object_target_local_height_error"]
            ),
            "place_angle_error": float(
                box_place_metrics["object_target_local_angle_error"]
            ),
        }

    def _goal_from_pose(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(pos, dtype=np.float64).reshape(3),
                _normalize_quat(np.asarray(quat, dtype=np.float64).reshape(4)),
            ]
        ).astype(np.float32)

    def _get_achieved_goal(self) -> np.ndarray:
        obj_pos, obj_quat = self._get_active_obj_pose()
        return self._goal_from_pose(obj_pos, obj_quat)

    def _get_desired_goal(self) -> np.ndarray:
        target_pos, target_quat = self._get_insert_target_pose()
        return self._goal_from_pose(target_pos, target_quat)

    def _goal_success_mask(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        single = achieved_goal.ndim == 1
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)

        pos_delta = achieved_goal[:, :3] - desired_goal[:, :3]
        radial_error = np.linalg.norm(pos_delta[:, :2], axis=1)
        height_error = np.abs(pos_delta[:, 2])
        achieved_quat = achieved_goal[:, 3:7]
        desired_quat = desired_goal[:, 3:7]
        achieved_quat = achieved_quat / np.maximum(
            np.linalg.norm(achieved_quat, axis=1, keepdims=True),
            1e-12,
        )
        desired_quat = desired_quat / np.maximum(
            np.linalg.norm(desired_quat, axis=1, keepdims=True),
            1e-12,
        )
        dot = np.abs(np.sum(achieved_quat * desired_quat, axis=1))
        angle_error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
        # print(f"Pos Error : {pos_error}, Angle Error : {angle_error}", flush=True)
        success = np.logical_and.reduce(
            (
                radial_error < self._success_distance,
                height_error < self._success_distance,
                angle_error < self._success_angle_rad,
            )
        )
        return success[0] if single else success

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info=None,
    ):
        del info

        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)

        single = achieved_goal.ndim == 1
        achieved_goal = np.atleast_2d(achieved_goal)
        desired_goal = np.atleast_2d(desired_goal)

        pos_delta = achieved_goal[:, :3] - desired_goal[:, :3]
        pos_error = np.linalg.norm(pos_delta, axis=1)
        radial_error = np.linalg.norm(pos_delta[:, :2], axis=1)
        height_error = np.abs(pos_delta[:, 2])

        achieved_quat = achieved_goal[:, 3:7]
        desired_quat = desired_goal[:, 3:7]

        achieved_quat = achieved_quat / np.maximum(
            np.linalg.norm(achieved_quat, axis=1, keepdims=True),
            1e-12,
        )
        desired_quat = desired_quat / np.maximum(
            np.linalg.norm(desired_quat, axis=1, keepdims=True),
            1e-12,
        )

        dot = np.abs(np.sum(achieved_quat * desired_quat, axis=1))
        angle_error = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))

        success = np.logical_and.reduce(
            (
                radial_error < self._success_distance,
                height_error < self._success_distance,
                angle_error < self._success_angle_rad,
            )
        )

        # Reward dasar: sparse HER style.
        reward = -1.0 * np.ones_like(pos_error, dtype=np.float32)

        # Dense kecil, hanya untuk membantu precision refinement.
        # 2-3 cm dapat reward lebih baik daripada jauh,
        # tapi tetap negatif selama belum masuk success threshold.
        pos_bonus = 0.45 * np.exp(-np.square(pos_error / 0.03))
        ori_bonus = 0.10 * np.exp(-np.square(angle_error / 0.50))

        reward = reward + pos_bonus.astype(np.float32) + ori_bonus.astype(np.float32)

        # Penting: non-success tetap negatif.
        # Jangan biarkan 2-3 cm menjadi "cukup bagus".
        reward = np.minimum(reward, -0.05)

        # Success sejati tetap reward terbaik.
        reward = np.where(success, 0.0, reward).astype(np.float32)

        return float(reward.item()) if single else reward

    def _update_manual_gripper_phase(self, metrics: dict[str, Any]) -> str:
        if self.gripper_phase == "released":
            self.last_manual_event = "released_hold"
            return "released_hold"

        event = "hold"
        close_angle_ok = (
            True
            if self._close_angle_rad is None
            else float(metrics["ee_obj_angle"]) <= self._close_angle_rad
        )
        if (
            self.gripper_phase == "open"
            and float(metrics["ee_obj_dist"]) <= self._close_distance
            and close_angle_ok
        ):
            self.gripper_phase = "closed"
            event = "close_near_object"

        release_angle_ok = float(metrics["target_angle"]) <= self._release_angle_rad
        release_distance_ok = float(metrics["target_dist"]) <= self._release_distance
        if self.gripper_phase == "closed" and (
            bool(metrics["target_pose_aligned"])
            or (release_distance_ok and release_angle_ok)
        ):
            self.gripper_phase = "released"
            self.release_steps = 0
            self.release_event_count += 1
            event = "release_on_target_align"

        self.last_manual_event = event
        return event

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.current_step = 0
        self.release_steps = 0
        self.release_event_count = 0
        self.success_counter = 0
        self.qswitch_insert_active = False
        self.gripper_phase = "open"
        self.gripper_state = "open"
        self.last_manual_event = "reset_open"
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

        self._set_place_pose_in_model(self._default_place_pos, self._default_place_quat)
        self._sync_target_site_to_active_place()

        obj_pos, obj_quat, object_yaw = self._sample_object_pose()
        object_info = self._get_active_obj_info()
        object_qposadr = int(object_info["qposadr"])
        object_dofadr = int(object_info["dofadr"])
        qpos[object_qposadr : object_qposadr + 3] = obj_pos
        qpos[object_qposadr + 3 : object_qposadr + 7] = obj_quat
        qvel[object_dofadr : object_dofadr + 6] = 0.0
        self.set_state(qpos, qvel)
        reset_ctrl = self.data.ctrl.copy()
        self._set_open_gripper_target(reset_ctrl)
        self.data.ctrl[:] = np.clip(reset_ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        if self._reset_settle_steps > 0:
            for _ in range(self._reset_settle_steps):
                self.do_simulation(self.data.ctrl.copy(), 1)
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
            self.sampled_target_place_pos, self.sampled_target_place_quat
        )
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

        target_pos, target_quat = self._get_target_pose()
        obj_target_pos_error, _ = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        self.initial_object_target_dist = float(np.linalg.norm(obj_target_pos_error))
        self.best_object_target_dist = float(self.initial_object_target_dist)
        self.initial_obj_site_pos = obj_pos.copy()
        self.sampled_target_pos = target_pos.copy()
        self.sampled_target_quat = target_quat.copy()
        self.sampled_object_yaw = float(object_yaw)
        self.applied_object_yaw = float(self._quat_to_yaw(obj_quat))
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(
                _normalize_quat(self.data.body(self.place_body_name).xquat.copy())
            )
        )
        self._reset_ik_state()
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        # print(f"Step : {self.current_step }")
        # print(f"Action di QMP ENV : {np.round(np.asarray(action).reshape(-1), 6)}")
        pre_metrics = self._task_metrics()
        manual_event = self._update_manual_gripper_phase(pre_metrics)
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)
        self._apply_manual_gripper_to_ctrl(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        start_ctrl = self.data.ctrl.copy()
        for interp_idx in range(1, self._control_interpolation_steps + 1):
            alpha = interp_idx / self._control_interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            self.do_simulation(
                np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high),
                self.frame_skip,
            )
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

        metrics = self._task_metrics()
        target_pose_aligned = bool(metrics["target_pose_aligned"])
        place_pose_aligned = bool(metrics["place_pose_aligned"])
        # print(
        #     f"Object-Target Dist: {metrics['target_dist']:.4f}, Angle: {np.rad2deg(metrics['target_angle']):.2f} deg, Pose Aligned: {target_pose_aligned}, Manual Event: {manual_event}"
        # )
        # print(f"Objek pos {metrics['obj_pos']}, Target pos {metrics['target_pos']}")
        if place_pose_aligned:
            self.success_counter += 1
        else:
            self.success_counter = 0

        if self.gripper_phase == "released":
            self.release_steps += 1

        reward, reward_info = self._get_rew(metrics)
        terminated_success = self.success_counter >= self._success_steps_required
        terminated_lost_object = bool(
            self.gripper_phase == "closed"
            and float(metrics["ee_obj_dist"]) >= self._terminate_lost_object_distance
            and not target_pose_aligned
        )
        terminated = terminated_success or terminated_lost_object
        # print(
        #     f"Terminated Lost Object: {terminated_lost_object} or Terminated Success: {terminated_success}"
        # )
        truncated = self.current_step >= self.max_episode_steps
        # print(f"Truncated: {truncated}")
        reward_info.update(
            {
                "terminated_success": int(terminated_success),
                "terminated_lost_object": int(terminated_lost_object),
                "manual_gripper_phase": self.gripper_phase,
                "manual_gripper_event": manual_event,
                "manual_release_steps": int(self.release_steps),
                "manual_release_event_count": int(self.release_event_count),
            }
        )
        if self._last_qswitch_debug:
            reward_info["qswitch"] = dict(self._last_qswitch_debug)
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), terminated, truncated, reward_info

    def _get_rew(
        self,
        metrics: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        released = self.gripper_phase == "released"
        target_dist = float(metrics["target_dist"])
        target_angle = float(metrics["target_angle"])
        lift_height = float(metrics["lift_height"])
        lift_progress = float(
            np.clip(
                lift_height / max(self._target_height_above_place, 1e-9),
                0.0,
                1.5,
            )
        )
        her_sparse_reward = float(
            self.compute_reward(self._get_achieved_goal(), self._get_desired_goal(), {})
        )
        # print(f"HER Sparse Reward: {her_sparse_reward:.4f}")
        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)
        target_pose_aligned = bool(metrics["target_pose_aligned"])
        place_pose_aligned = bool(metrics["place_pose_aligned"])

        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)
        return her_sparse_reward, {
            "ee_object_dist": float(metrics["ee_obj_dist"]),
            "ee_object_rot_error": float(metrics["ee_obj_angle"]),
            "object_target_dist": target_dist,
            "object_target_angle_rad": target_angle,
            "object_target_rot_error": target_angle,
            "object_place_radial_error": float(metrics["place_radial_error"]),
            "object_place_height_error": float(metrics["place_height_error"]),
            "object_place_angle_error": float(metrics["place_angle_error"]),
            "lift_height": lift_height,
            "lift_progress": lift_progress,
            "target_pose_aligned": int(target_pose_aligned),
            "place_pose_aligned": int(place_pose_aligned),
            "is_success": int(place_pose_aligned),
            "success_counter": int(self.success_counter),
            "gripper_closed": int(self.gripper_phase == "closed"),
            "gripper_released": int(released),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
            "reward_her_sparse": float(her_sparse_reward),
        }

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
            [1.0 if self.gripper_phase in {"closed", "released"} else 0.0],
            dtype=np.float64,
        )
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_pos, place_quat = self._get_place_site_pose()
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
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

    def _get_flat_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                np.asarray(component, dtype=np.float64).reshape(-1)
                for _, component in self._get_obs_components()
            ]
        ).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "observation": self._get_flat_obs(),
            "achieved_goal": self._get_achieved_goal(),
            "desired_goal": self._get_desired_goal(),
        }

    def set_qswitch_debug(self, debug: dict[str, Any]) -> None:
        self._last_qswitch_debug = dict(debug)

    def _get_ik_debug_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "ik_target_pos": self._ik_target_pos.copy(),
            "ik_target_quat": self._ik_target_quat.copy(),
            "ik_failure_count": int(self._ik_failure_count),
        }
        if self._last_ik_result is not None:
            state.update(
                {
                    "ik_success": bool(self._last_ik_result.success),
                    "ik_iterations": int(self._last_ik_result.iterations),
                    "ik_position_error_norm": float(
                        self._last_ik_result.position_error_norm
                    ),
                    "ik_rotation_error_deg": float(
                        np.rad2deg(self._last_ik_result.rotation_error_norm)
                    ),
                }
            )
        return state

    def get_debug_state(self) -> dict[str, Any]:
        metrics = self._task_metrics()
        place_pos, place_quat = self._get_place_site_pose()
        box_place_metrics = self._box_place_metrics()
        return {
            "current_step": int(self.current_step),
            "active_object": self.active_obj_name,
            "ee_pos": metrics["ee_pos"],
            "ee_quat": metrics["ee_quat"],
            "obj_pos": metrics["obj_pos"],
            "obj_quat": metrics["obj_quat"],
            "place_pos": place_pos,
            "place_quat": place_quat,
            "target_pos": metrics["target_pos"],
            "target_quat": metrics["target_quat"],
            "target_height_above_place": float(self._target_height_above_place),
            "success_distance": float(self._success_distance),
            "success_angle_deg": float(np.rad2deg(self._success_angle_rad)),
            "lift_height": float(metrics["lift_height"]),
            "ee_obj_pos_error": metrics["ee_obj_pos_error"],
            "ee_obj_rot_error": metrics["ee_obj_rot_error"],
            "ee_obj_dist": float(metrics["ee_obj_dist"]),
            "ee_obj_angle_rad": float(metrics["ee_obj_angle"]),
            "ee_object_dist": float(metrics["ee_obj_dist"]),
            "obj_target_pos_error": metrics["obj_target_pos_error"],
            "obj_target_rot_error": metrics["obj_target_rot_error"],
            "obj_target_dist": float(metrics["target_dist"]),
            "obj_target_angle_rad": float(metrics["target_angle"]),
            "object_target_dist": float(metrics["target_dist"]),
            "object_target_rot_error": float(metrics["target_angle"]),
            "target_pose_aligned": bool(metrics["target_pose_aligned"]),
            "place_pose_aligned": bool(metrics["place_pose_aligned"]),
            "object_place_radial_error": float(metrics["place_radial_error"]),
            "object_place_height_error": float(metrics["place_height_error"]),
            "object_place_angle_error": float(metrics["place_angle_error"]),
            "success_counter": int(self.success_counter),
            "initial_object_target_dist": float(self.initial_object_target_dist),
            "best_object_target_dist": float(self.best_object_target_dist),
            "gripper_state": self.gripper_state,
            "manual_gripper_phase": self.gripper_phase,
            "qswitch_insert_active": int(bool(self.qswitch_insert_active)),
            "manual_gripper_event": self.last_manual_event,
            "manual_release_steps": int(self.release_steps),
            "manual_release_event_count": int(self.release_event_count),
            "gripper_qpos": self.data.qpos[[self.gripL_qadr, self.gripR_qadr]].copy(),
            "gripper_ctrl": self.data.ctrl[
                [self.gripL_act_id, self.gripR_act_id]
            ].copy(),
            "last_action": self.last_action.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "sampled_target_place_quat": self.sampled_target_place_quat.copy(),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "target_resample_attempts": int(self._last_target_resample_attempts),
            "target_resample_distance": float(self._last_target_resample_distance),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "object_local_pos": np.asarray(
                box_place_metrics["object_local_pos"]
            ).copy(),
            "object_local_quat": np.asarray(
                box_place_metrics["object_local_quat"]
            ).copy(),
            "target_local_pos": np.asarray(
                box_place_metrics["target_local_pos"]
            ).copy(),
            "target_local_quat": np.asarray(
                box_place_metrics["target_local_quat"]
            ).copy(),
            "object_target_local_pos_error": np.asarray(
                box_place_metrics["object_target_local_pos_error"]
            ).copy(),
            "object_target_local_rot_error": np.asarray(
                box_place_metrics["object_target_local_rot_error"]
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
            "qswitch": dict(self._last_qswitch_debug),
            **self._get_ik_debug_state(),
        }
