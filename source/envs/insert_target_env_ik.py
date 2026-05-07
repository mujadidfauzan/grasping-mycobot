from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium.spaces import Box

from script.inverse_kinematics import (
    _normalize_quat,
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)

from .config_export import capture_init_config, export_env_config
from .grasping_env_ik import DEFAULT_CAMERA_CONFIG, GraspingEnvIK

DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"


class InsertTargetEnvIK(GraspingEnvIK):
    """Insertion/release env with 6-DoF Cartesian IK actions.

    Reset is generated procedurally with IK only: the object is sampled first,
    the IK target follows that object pose, the gripper closes, and the reset is
    accepted when the object is held.
    The task target is the active place site plus a configurable vertical offset
    (default 3 cm). Once the object reaches that target, the gripper opens
    manually and the dense distance/orientation shaping is disabled.
    """

    def __init__(
        self,
        xml_file: str = str(DEFAULT_XML_PATH),
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 2.0,
        reward_orientation_weight: float = 1.0,
        reward_orientation_tanh_weight: float = 1.0,
        reward_bonus: float = 20.0,
        control_penalty_weight: float = 0.001,
        distance_tanh_scale: float = 0.05,
        orientation_tanh_scale: float = 0.50,
        success_distance: float = 0.008,
        success_steps_required: int = 5,
        max_episode_steps: int = 160,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.22, 0.00),
        ik_workspace_high: tuple[float, float, float] = (0.35, 0.22, 0.45),
        ik_position_only: bool = False,
        ik_max_iters: int = 100,
        ik_position_tolerance: float = 1e-3,
        ik_rotation_tolerance_deg: float = 4.0,
        ik_damping: float = 1e-3,
        ik_step_size: float = 0.45,
        ik_max_delta_deg: float = 6.0,
        ik_rotation_weight: float = 0.35,
        ik_random_restarts: int = 2,
        ik_seed: int | None = 0,
        control_interpolation_steps: int = 10,
        max_joint_ctrl_delta_deg: float = 5.0,
        smooth_cartesian_target: bool = True,
        debug_ik: bool = False,
        object_x_range: tuple[float, float] = (0.15, 0.24),
        object_y_range: tuple[float, float] = (-0.12, 0.12),
        object_z: float = 0.025,
        object_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        target_x_range: tuple[float, float] = (0.17, 0.27),
        target_y_range: tuple[float, float] = (-0.14, 0.14),
        target_place_z: float = 0.02,
        target_place_yaw_range: tuple[float, float] = (-np.pi / 6.0, np.pi / 6.0),
        target_height_above_place: float = 0.03,
        target_min_object_xy_distance: float = 0.06,
        gripper_open_distance: float | None = None,
        release_open_steps: int = 0,
        terminate_ee_obj_distance: float = 0.08,
        reset_attempts: int = 8,
        reset_pregrasp_height: float = 0.035,
        reset_approach_steps: int = 35,
        reset_pregrasp_settle_steps: int = 15,
        reset_close_steps: int = 80,
        reset_lift_height: float = 0.012,
        reset_lift_steps: int = 45,
        reset_position_noise: float = 0.003,
        reset_orientation_noise_deg: float = 3.0,
        reset_accept_ee_obj_dist: float = 0.028,
        reset_accept_min_lift: float = 0.0,
        reset_qpos_close_threshold: float = 0.001,
        allow_reset_fallback_snapshot: bool = True,
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        ee_frame_body_name: str = "ee_frame_vis",
        object_frame_body_name: str = "object_frame_vis",
        target_frame_body_name: str = "target_frame_vis",
        # Backward-compatible arguments from the old policy-reset InsertTargetEnvIK.
        grasp_model_path: str | None = None,
        grasp_env_name: str = "GraspingEnvIK",
        grasp_xml_file: str | None = None,
        grasp_max_steps: int = 220,
        grasp_attempts_per_reset: int = 6,
        grasp_deterministic: bool = True,
        grasp_success_min_lift: float | None = None,
        grasp_success_max_lift: float | None = None,
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
        self._gripper_open_target = np.array([0.01, -0.01], dtype=np.float64)
        self._gripper_closed_target = np.array([-0.02, 0.02], dtype=np.float64)

        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            default_camera_config=default_camera_config,
            reward_distance_weight=reward_target_weight,
            reward_distance_tanh_weight=reward_target_tanh_weight,
            reward_orientation_weight=reward_orientation_weight,
            reward_target_bonus=reward_bonus,
            control_penalty_weight=control_penalty_weight,
            distance_tanh_scale=distance_tanh_scale,
            success_distance=success_distance,
            success_angle_deg=180.0,
            success_requires_orientation=False,
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
            object_x_range=object_x_range,
            object_y_range=object_y_range,
            object_z=object_z,
            object_yaw_range=object_yaw_range,
            lift_height=target_height_above_place,
            ee_site_name=ee_site_name,
            target_site_name=target_site_name,
            ee_frame_body_name=ee_frame_body_name,
            object_frame_body_name=object_frame_body_name,
            target_frame_body_name=target_frame_body_name,
            **kwargs,
        )
        self._init_config = init_config

        self._reward_target_weight = float(reward_target_weight)
        self._reward_target_tanh_weight = float(reward_target_tanh_weight)
        self._reward_orientation_weight = float(reward_orientation_weight)
        self._reward_orientation_tanh_weight = float(reward_orientation_tanh_weight)
        self._reward_bonus = float(reward_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._distance_tanh_scale = float(distance_tanh_scale)
        self._orientation_tanh_scale = float(orientation_tanh_scale)
        self._success_distance = float(success_distance)
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
        self._gripper_open_distance = (
            self._success_distance
            if gripper_open_distance is None
            else float(gripper_open_distance)
        )
        self._release_open_steps = max(0, int(release_open_steps))
        self._terminate_ee_obj_distance = float(terminate_ee_obj_distance)
        self._reset_attempts = max(1, int(reset_attempts))
        self._reset_pregrasp_height = float(reset_pregrasp_height)
        self._reset_approach_steps = max(1, int(reset_approach_steps))
        self._reset_pregrasp_settle_steps = max(0, int(reset_pregrasp_settle_steps))
        self._reset_close_steps = max(1, int(reset_close_steps))
        self._reset_lift_height = float(reset_lift_height)
        self._reset_lift_steps = max(1, int(reset_lift_steps))
        self._reset_position_noise = float(reset_position_noise)
        self._reset_orientation_noise_rad = np.deg2rad(float(reset_orientation_noise_deg))
        self._reset_accept_ee_obj_dist = float(reset_accept_ee_obj_dist)
        self._reset_accept_min_lift = float(reset_accept_min_lift)
        self._reset_qpos_close_threshold = float(reset_qpos_close_threshold)
        self._allow_reset_fallback_snapshot = bool(allow_reset_fallback_snapshot)
        self._grasp_ctrl_close_threshold = float(grasp_ctrl_close_threshold)

        if self._distance_tanh_scale <= 0.0:
            raise ValueError("distance_tanh_scale must be greater than 0.")
        if self._orientation_tanh_scale <= 0.0:
            raise ValueError("orientation_tanh_scale must be greater than 0.")
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        if self._target_height_above_place <= 0.0:
            raise ValueError("target_height_above_place must be greater than 0.")
        if self._gripper_open_distance <= 0.0:
            raise ValueError("gripper_open_distance must be greater than 0.")
        if self._terminate_ee_obj_distance <= 0.0:
            raise ValueError("terminate_ee_obj_distance must be greater than 0.")
        if self._reset_pregrasp_height < 0.0:
            raise ValueError("reset_pregrasp_height must be non-negative.")
        if self._reset_lift_height < 0.0:
            raise ValueError("reset_lift_height must be non-negative.")
        if self._reset_position_noise < 0.0:
            raise ValueError("reset_position_noise must be non-negative.")
        if self._target_x_range[0] > self._target_x_range[1]:
            raise ValueError("target_x_range must be ordered as (min_x, max_x).")
        if self._target_y_range[0] > self._target_y_range[1]:
            raise ValueError("target_y_range must be ordered as (min_y, max_y).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError("target_place_yaw_range must be ordered as (min_yaw, max_yaw).")

        self._arm_dof_indices = np.array(
            [
                self.model.jnt_dofadr[
                    self._require_named_id(
                        mujoco.mjtObj.mjOBJ_JOINT, joint_name, "joint"
                    )
                ]
                for joint_name in self._arm_joint_names
            ],
            dtype=np.int64,
        )
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
            self.place_info[obj_name] = {
                "body_name": body_name,
                "site_name": site_name,
                "geom_name": geom_name,
                "body_id": self._require_named_id(mujoco.mjtObj.mjOBJ_BODY, body_name, "body"),
                "site_id": self._require_named_id(mujoco.mjtObj.mjOBJ_SITE, site_name, "site"),
                "geom_id": self._require_named_id(mujoco.mjtObj.mjOBJ_GEOM, geom_name, "geom"),
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

        self.gripper_release_latched = False
        self.object_grasp_attached = False
        self._grasp_site_offset_ee = np.zeros(3, dtype=np.float64)
        self._grasp_relative_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_gripper_should_open = False
        self.last_release_target_dist = np.inf
        self.last_release_target_angle = np.inf
        self.last_reset_source = "uninitialized"
        self.last_reset_attempts = 0
        self.last_reset_lift_height = 0.0
        self.last_reset_ee_obj_dist = np.inf
        self.sampled_target_place_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_site_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.sampled_target_place_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0
        self.sampled_target_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_site_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.reset_grasp_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.reset_grasp_position_noise = np.zeros(3, dtype=np.float64)

        self._insert_env_ready = True
        self._sync_target_site_to_above_place()
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
        vec_quat = np.array([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
        rotated = GraspingEnvIK._quat_multiply(
            GraspingEnvIK._quat_multiply(quat, vec_quat),
            GraspingEnvIK._quat_conjugate(quat),
        )
        return rotated[1:]

    def _get_active_place_info(self) -> dict[str, int | str]:
        return self.place_info[self.active_obj_name]

    def _get_active_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_place_info()["site_name"]))

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

    def _set_place_poses_in_model(
        self,
        active_place_pos: np.ndarray,
        active_place_quat: np.ndarray,
    ) -> None:
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for index, obj_name in enumerate(self.object_names):
            body_id = int(self.place_info[obj_name]["body_id"])
            if obj_name == self.active_obj_name:
                self.model.body_pos[body_id] = np.asarray(active_place_pos, dtype=np.float64)
                self.model.body_quat[body_id] = _normalize_quat(active_place_quat)
            else:
                self.model.body_pos[body_id] = np.array([2.0 + index, 2.0, 0.2], dtype=np.float64)
                self.model.body_quat[body_id] = identity_quat

    def _set_active_place_visual(self) -> None:
        for obj_name, info in self.place_info.items():
            rgba = self.place_geom_rgba[obj_name].copy()
            if obj_name != self.active_obj_name:
                rgba[3] = 0.0
            self.model.geom_rgba[int(info["geom_id"])] = rgba

    def _sample_target_place_site_pose(
        self,
        object_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        object_xy = np.asarray(object_pos, dtype=np.float64)[:2]
        best_site_pos = None
        best_dist = -np.inf
        target_place_yaw = 0.0

        for _ in range(100):
            site_pos = np.array(
                [
                    self.np_random.uniform(*self._target_x_range),
                    self.np_random.uniform(*self._target_y_range),
                    self._target_place_z,
                ],
                dtype=np.float64,
            )
            target_place_yaw = float(
                self.np_random.uniform(*self._target_place_yaw_range)
            )
            dist = float(np.linalg.norm(site_pos[:2] - object_xy))
            if dist > best_dist:
                best_site_pos = site_pos
                best_dist = dist
            if dist >= self._target_min_object_xy_distance:
                break

        assert best_site_pos is not None
        site_quat = self._yaw_to_quat(target_place_yaw)
        place_pos, place_quat = self._target_place_site_pose_to_place_body_pose(
            best_site_pos,
            site_quat,
        )
        return best_site_pos, site_quat, place_pos, place_quat, target_place_yaw

    def _sync_target_site_to_above_place(self) -> None:
        if not getattr(self, "_insert_env_ready", False):
            return

        mujoco.mj_forward(self.model, self.data)
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
        target_pos = place_site_pos + np.array(
            [0.0, 0.0, self._target_height_above_place],
            dtype=np.float64,
        )
        self._set_target_site_pose_in_model(target_pos, place_site_quat)
        mujoco.mj_forward(self.model, self.data)
        self.sampled_target_site_pos, self.sampled_target_site_quat = self._get_target_pose()

    def _sample_reset_grasp_quat(self, obj_quat: np.ndarray) -> np.ndarray:
        base_rpy = _quat_to_euler_xyz(obj_quat)
        if self._reset_orientation_noise_rad > 0.0:
            noise = self.np_random.uniform(
                -self._reset_orientation_noise_rad,
                self._reset_orientation_noise_rad,
                size=3,
            )
            base_rpy = self._wrap_vector_to_pi(base_rpy + noise)
        return _quat_from_euler_xyz(*base_rpy)

    def _sample_reset_grasp_pos(self, obj_pos: np.ndarray) -> np.ndarray:
        obj_pos = np.asarray(obj_pos, dtype=np.float64).reshape(3)
        if self._reset_position_noise > 0.0:
            self.reset_grasp_position_noise = self.np_random.uniform(
                -self._reset_position_noise,
                self._reset_position_noise,
                size=3,
            ).astype(np.float64)
        else:
            self.reset_grasp_position_noise = np.zeros(3, dtype=np.float64)
        return np.clip(
            obj_pos + self.reset_grasp_position_noise,
            self._ik_workspace_low,
            self._ik_workspace_high,
        )

    def _set_active_obj_site_pose(
        self,
        site_world_pos: np.ndarray,
        site_world_quat: np.ndarray,
    ) -> None:
        info = self._get_active_obj_info()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qposadr = int(info["qposadr"])
        dofadr = int(info["dofadr"])
        site_id = int(info["site_id"])
        site_world_pos = np.asarray(site_world_pos, dtype=np.float64).reshape(3)
        site_world_quat = _normalize_quat(
            np.asarray(site_world_quat, dtype=np.float64).reshape(4)
        )
        site_local_pos = self.model.site_pos[site_id].copy()
        body_pos = site_world_pos - self._quat_rotate_vector(
            site_world_quat,
            site_local_pos,
        )

        qpos[qposadr : qposadr + 3] = body_pos
        qpos[qposadr + 3 : qposadr + 7] = site_world_quat
        qvel[dofadr : dofadr + 6] = 0.0
        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

    def _attach_active_object_to_gripper(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_quat_conj = self._quat_conjugate(ee_quat)
        self.object_grasp_attached = True
        self.grasp_latched = True
        self.gripper_release_latched = False
        self._grasp_site_offset_ee = self._quat_rotate_vector(
            ee_quat_conj,
            obj_pos - ee_pos,
        )
        self._grasp_relative_quat = _normalize_quat(
            self._quat_multiply(ee_quat_conj, obj_quat)
        )

    def _sync_grasped_object_to_ee(self) -> None:
        if not self.object_grasp_attached or self.gripper_release_latched:
            return

        ee_pos, ee_quat = self._get_ee_pose()
        obj_site_pos = ee_pos + self._quat_rotate_vector(
            ee_quat,
            self._grasp_site_offset_ee,
        )
        obj_site_quat = _normalize_quat(
            self._quat_multiply(ee_quat, self._grasp_relative_quat)
        )
        self._set_active_obj_site_pose(obj_site_pos, obj_site_quat)

    def _apply_arm_qpos(self, arm_qpos: np.ndarray) -> None:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self._arm_qpos_indices] = np.asarray(arm_qpos, dtype=np.float64)
        qvel[self._arm_dof_indices] = 0.0
        self.set_state(qpos, qvel)

    def _solve_reset_ik(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        seed_offset: int,
    ):
        return self._ik_solver.solve(
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
            random_restarts=max(2, self._ik_random_restarts),
            seed=None if self._ik_seed is None else int(self._ik_seed + seed_offset),
        )

    def _solve_best_reset_ik(
        self,
        target_pos: np.ndarray,
        preferred_quat: np.ndarray,
        *,
        seed_offset: int,
    ):
        candidate_quats = [
            _normalize_quat(preferred_quat),
            _normalize_quat(self._get_active_obj_pose()[1]),
        ]
        best_result = None
        best_score = np.inf

        for index, quat in enumerate(candidate_quats):
            result = self._solve_reset_ik(
                target_pos,
                quat,
                seed_offset=seed_offset + index,
            )
            score = float(
                result.position_error_norm
                + self._ik_rotation_weight * result.rotation_error_norm
            )
            if score < best_score:
                best_score = score
                best_result = result
                self.reset_grasp_target_quat = quat.copy()
            if result.success:
                break

        assert best_result is not None
        return best_result

    def _settle_with_ctrl(self, ctrl: np.ndarray, steps: int) -> None:
        ctrl = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        for _ in range(max(0, int(steps))):
            self.do_simulation(ctrl, self.frame_skip)

    def _drive_arm_to_pose(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        close_gripper: bool,
        steps: int,
        seed_offset: int,
    ) -> bool:
        ik_result = self._solve_reset_ik(
            target_pos,
            target_quat,
            seed_offset=seed_offset,
        )
        self._last_ik_result = ik_result
        if not ik_result.success:
            self._ik_failure_count += 1

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[self._arm_ctrl_indices] = ik_result.q_rad.copy()
        if close_gripper:
            self._set_closed_gripper_target(target_ctrl)
        else:
            self._set_open_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        start_ctrl = self.data.ctrl.copy()
        for step_idx in range(1, max(1, int(steps)) + 1):
            alpha = step_idx / max(1, int(steps))
            ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            self.do_simulation(np.clip(ctrl, self._ctrl_low, self._ctrl_high), self.frame_skip)

        self.data.ctrl[:] = target_ctrl
        mujoco.mj_forward(self.model, self.data)
        return bool(ik_result.success)

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[self.gripL_act_id] = self._gripper_open_target[0]
        ctrl[self.gripR_act_id] = self._gripper_open_target[1]

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[self.gripL_act_id] = self._gripper_closed_target[0]
        ctrl[self.gripR_act_id] = self._gripper_closed_target[1]

    def _gripper_qpos_closed(self) -> bool:
        qpos = self.data.qpos[[self.gripL_qadr, self.gripR_qadr]]
        return bool(
            qpos[0] <= -self._reset_qpos_close_threshold
            and qpos[1] >= self._reset_qpos_close_threshold
        )

    def _reset_episode_state(self) -> None:
        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.gripper_release_latched = False
        self.object_grasp_attached = False
        self.last_gripper_should_open = False
        self.last_release_target_dist = np.inf
        self.last_release_target_angle = np.inf
        self.grasp_latched = True
        self.last_grasp_should_close = True
        self.last_grasp_dist = np.inf
        self.last_grasp_angle = np.inf
        self._grasp_site_offset_ee = np.zeros(3, dtype=np.float64)
        self._grasp_relative_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.reset_grasp_position_noise = np.zeros(3, dtype=np.float64)
        self._reset_ik_state()

    def _initialize_scene_for_reset(self) -> np.ndarray:
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
                qpos[qposadr : qposadr + 3] = np.array([6.0, 1.0, 1.0], dtype=np.float64)
                qpos[qposadr + 3 : qposadr + 7] = identity_quat
            qvel[dofadr : dofadr + 6] = 0.0

        qpos[self.gripL_qadr] = self._gripper_open_target[0]
        qpos[self.gripR_qadr] = self._gripper_open_target[1]
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0
        self.set_state(qpos, qvel)

        (
            self.sampled_target_place_site_pos,
            self.sampled_target_place_site_quat,
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
        ) = self._sample_target_place_site_pose(obj_pos)
        self._set_place_poses_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()
        mujoco.mj_forward(self.model, self.data)
        self._sync_target_site_to_above_place()

        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_ctrl_indices] = qpos[self._arm_qpos_indices]
        self._set_open_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        self.sampled_object_yaw = float(yaw)
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(self._get_active_place_site_pose()[1])
        )
        return self.initial_obj_site_pos.copy()

    def _run_procedural_grasp_reset(self, attempt_index: int) -> None:
        obj_pos, obj_quat = self._get_active_obj_pose()
        grasp_quat = self._sample_reset_grasp_quat(obj_quat)
        self.reset_grasp_target_quat = grasp_quat.copy()

        grasp_pos = self._sample_reset_grasp_pos(obj_pos)
        grasp_result = self._solve_best_reset_ik(
            grasp_pos,
            grasp_quat,
            seed_offset=attempt_index * 10,
        )
        self._last_ik_result = grasp_result
        if not grasp_result.success:
            self._ik_failure_count += 1

        self._apply_arm_qpos(grasp_result.q_rad)
        ctrl = self.data.ctrl.copy()
        ctrl[self._arm_ctrl_indices] = grasp_result.q_rad
        self._set_closed_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)
        self._attach_active_object_to_gripper()

        self._reset_ik_state()
        self.sync_visual_frames()

    def _reset_acceptance_metrics(
        self,
        reset_start_obj_pos: np.ndarray,
    ) -> tuple[bool, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos,
            ee_quat,
            obj_pos,
            obj_quat,
        )
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        lift_height = float(obj_pos[2] - np.asarray(reset_start_obj_pos, dtype=np.float64)[2])
        max_reset_angle = max(
            np.deg2rad(10.0),
            self._reset_orientation_noise_rad + self._ik_rotation_tolerance_rad,
        )
        gripper_ctrl = self.data.ctrl[[self.gripL_act_id, self.gripR_act_id]]
        gripper_command_closed = bool(
            gripper_ctrl[0] < -self._grasp_ctrl_close_threshold
            and gripper_ctrl[1] > self._grasp_ctrl_close_threshold
        )
        accepted = bool(
            self.object_grasp_attached
            and gripper_command_closed
            and ee_obj_dist <= self._reset_accept_ee_obj_dist
            and ee_obj_angle <= max_reset_angle
            and lift_height >= self._reset_accept_min_lift
        )
        metrics = {
            "ee_obj_dist": ee_obj_dist,
            "ee_obj_angle": ee_obj_angle,
            "lift_height": lift_height,
            "max_reset_angle": float(max_reset_angle),
            "gripper_qpos_closed": int(self._gripper_qpos_closed()),
            "gripper_command_closed": int(gripper_command_closed),
            "object_grasp_attached": int(self.object_grasp_attached),
            "score": -ee_obj_dist
            + 0.5 * float(gripper_command_closed)
            + 0.5 * float(self.object_grasp_attached)
            + lift_height,
        }
        return accepted, metrics

    def _capture_reset_snapshot(self, metrics: dict, source: str, attempts: int) -> dict:
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "active_object": self.active_obj_name,
            "sampled_target_place_site_pos": self.sampled_target_place_site_pos.copy(),
            "sampled_target_place_site_quat": self.sampled_target_place_site_quat.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "sampled_target_place_quat": self.sampled_target_place_quat.copy(),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "sampled_target_site_pos": self.sampled_target_site_pos.copy(),
            "sampled_target_site_quat": self.sampled_target_site_quat.copy(),
            "reset_grasp_target_quat": self.reset_grasp_target_quat.copy(),
            "reset_grasp_position_noise": self.reset_grasp_position_noise.copy(),
            "grasp_site_offset_ee": self._grasp_site_offset_ee.copy(),
            "grasp_relative_quat": self._grasp_relative_quat.copy(),
            "source": source,
            "attempts": int(attempts),
            "metrics": dict(metrics),
        }

    def _restore_reset_snapshot(self, snapshot: dict) -> None:
        self.active_obj_name = str(snapshot["active_object"])
        self.sampled_target_place_site_pos = np.asarray(
            snapshot["sampled_target_place_site_pos"],
            dtype=np.float64,
        ).copy()
        self.sampled_target_place_site_quat = _normalize_quat(
            np.asarray(snapshot["sampled_target_place_site_quat"], dtype=np.float64)
        )
        self.sampled_target_place_pos = np.asarray(
            snapshot["sampled_target_place_pos"],
            dtype=np.float64,
        ).copy()
        self.sampled_target_place_quat = _normalize_quat(
            np.asarray(snapshot["sampled_target_place_quat"], dtype=np.float64)
        )
        self.sampled_target_place_yaw = float(snapshot["sampled_target_place_yaw"])
        self.sampled_target_site_pos = np.asarray(
            snapshot["sampled_target_site_pos"],
            dtype=np.float64,
        ).copy()
        self.sampled_target_site_quat = _normalize_quat(
            np.asarray(snapshot["sampled_target_site_quat"], dtype=np.float64)
        )
        self.reset_grasp_target_quat = _normalize_quat(
            np.asarray(snapshot["reset_grasp_target_quat"], dtype=np.float64)
        )
        self.reset_grasp_position_noise = np.asarray(
            snapshot.get("reset_grasp_position_noise", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).copy()
        self._grasp_site_offset_ee = np.asarray(
            snapshot.get("grasp_site_offset_ee", np.zeros(3, dtype=np.float64)),
            dtype=np.float64,
        ).copy()
        self._grasp_relative_quat = _normalize_quat(
            np.asarray(
                snapshot.get(
                    "grasp_relative_quat",
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                ),
                dtype=np.float64,
            )
        )
        self._set_place_poses_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()
        self._set_target_site_pose_in_model(
            self.sampled_target_site_pos,
            self.sampled_target_site_quat,
        )
        self.set_state(
            np.asarray(snapshot["qpos"], dtype=np.float64),
            np.asarray(snapshot["qvel"], dtype=np.float64),
        )
        self.data.ctrl[:] = np.clip(
            np.asarray(snapshot["ctrl"], dtype=np.float64),
            self._ctrl_low,
            self._ctrl_high,
        )
        mujoco.mj_forward(self.model, self.data)
        self.gripper_state = "closed"
        self.grasp_latched = True
        self.gripper_release_latched = False
        self.object_grasp_attached = True
        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(self._get_active_place_site_pose()[1])
        )
        metrics = dict(snapshot["metrics"])
        self.last_reset_source = str(snapshot["source"])
        self.last_reset_attempts = int(snapshot["attempts"])
        self.last_reset_lift_height = float(metrics.get("lift_height", 0.0))
        self.last_reset_ee_obj_dist = float(metrics.get("ee_obj_dist", np.inf))
        self._reset_ik_state()
        self.sync_visual_frames()

    def reset_model(self):
        if not getattr(self, "_insert_env_ready", False):
            return super().reset_model()

        self._reset_episode_state()
        best_snapshot = None
        best_score = -np.inf

        for attempt in range(1, self._reset_attempts + 1):
            reset_start_obj_pos = self._initialize_scene_for_reset()
            self._run_procedural_grasp_reset(attempt)
            accepted, metrics = self._reset_acceptance_metrics(reset_start_obj_pos)
            score = float(metrics["score"])
            if score > best_score:
                best_score = score
                best_snapshot = self._capture_reset_snapshot(
                    metrics,
                    "ik_fallback_best_grasp_snapshot",
                    attempt,
                )
            if accepted:
                snapshot = self._capture_reset_snapshot(metrics, "ik_grasp_success", attempt)
                self._restore_reset_snapshot(snapshot)
                return self._get_obs()

        fallback_metrics = {} if best_snapshot is None else dict(best_snapshot["metrics"])
        fallback_grasped = bool(
            fallback_metrics.get("object_grasp_attached", 0)
            and fallback_metrics.get("gripper_command_closed", 0)
            and float(fallback_metrics.get("ee_obj_dist", np.inf))
            <= self._reset_accept_ee_obj_dist
        )
        if (
            best_snapshot is not None
            and self._allow_reset_fallback_snapshot
            and fallback_grasped
        ):
            self._restore_reset_snapshot(best_snapshot)
            return self._get_obs()

        raise RuntimeError(
            "InsertTargetEnvIK failed to create a grasped reset state with IK. "
            "Increase reset_attempts/reset_close_steps or loosen reset_accept_* thresholds."
        )

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
        target_close = bool(target_dist < self._success_distance)
        return target_dist, target_angle, target_close

    def _apply_manual_release_if_ready(self) -> None:
        target_dist, target_angle, _ = self._get_target_pose_alignment()
        should_open = bool(target_dist <= self._gripper_open_distance)
        self.last_gripper_should_open = should_open
        self.last_release_target_dist = target_dist
        self.last_release_target_angle = target_angle
        if not should_open:
            return

        self.gripper_release_latched = True
        self.object_grasp_attached = False
        open_ctrl = self.data.ctrl.copy()
        self._set_open_gripper_target(open_ctrl)
        self.data.ctrl[:] = np.clip(open_ctrl, self._ctrl_low, self._ctrl_high)
        self._settle_with_ctrl(self.data.ctrl.copy(), self._release_open_steps)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.current_step += 1
        action, target_ctrl, _ik_result = self._ik_action_to_target_ctrl(action)
        if self.gripper_release_latched:
            self._set_open_gripper_target(target_ctrl)
        else:
            self._set_closed_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        start_ctrl = self.data.ctrl.copy()
        for interp_idx in range(1, self._control_interpolation_steps + 1):
            alpha = interp_idx / self._control_interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            self.do_simulation(np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high), self.frame_skip)

        if not self.gripper_release_latched:
            self._sync_grasped_object_to_ee()
        self.sync_visual_frames()
        if not self.gripper_release_latched:
            self._apply_manual_release_if_ready()
            self.sync_visual_frames()

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated_ee_obj_far = bool(
            not self.gripper_release_latched
            and reward_info["ee_object_dist"] >= self._terminate_ee_obj_distance
            and reward_info["object_target_dist"] >= self._success_distance
        )
        terminated = terminated_ee_obj_far
        truncated = self.current_step >= self.max_episode_steps
        reward_info["terminated_ee_obj_far"] = int(terminated_ee_obj_far)
        reward_info["terminated_success"] = int(
            self.success_counter >= self._success_steps_required
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
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
        target_close = bool(target_dist < self._success_distance)

        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / self._distance_tanh_scale))
        ) * self._reward_target_tanh_weight
        reward_orientation = -target_angle * self._reward_orientation_weight
        reward_orientation_tanh = (
            1.0 - float(np.tanh(target_angle / self._orientation_tanh_scale))
        ) * self._reward_orientation_tanh_weight
        control_penalty = -self._control_penalty_weight * float(np.sum(np.square(action)))
        reward_bonus = self._reward_bonus if target_close else 0.0

        dense_reward_active = not self.gripper_release_latched
        if dense_reward_active:
            reward = (
                reward_target
                + reward_target_tanh
                + reward_orientation
                + reward_orientation_tanh
                + control_penalty
                + reward_bonus
            )
        else:
            reward_target = 0.0
            reward_target_tanh = 0.0
            reward_orientation = 0.0
            reward_orientation_tanh = 0.0
            control_penalty = 0.0
            reward = reward_bonus

        if self.gripper_release_latched and target_close:
            self.success_counter += 1
        else:
            self.success_counter = 0

        reward_info = {
            "active_object": self.active_obj_name,
            "ee_object_dist": ee_obj_dist,
            "object_target_dist": target_dist,
            "object_target_rot_error": target_angle,
            "release_target_dist": target_dist,
            "release_target_angle_rad": target_angle,
            "target_height_above_place": float(self._target_height_above_place),
            "dense_reward_active": int(dense_reward_active),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_dist": float(reward_target),
            "reward_dist_tanh": float(reward_target_tanh),
            "reward_orientation": float(reward_orientation),
            "reward_orientation_tanh": float(reward_orientation_tanh),
            "reward_orient": float(reward_orientation),
            "reward_orient_tanh": float(reward_orientation_tanh),
            "reward_bonus": float(reward_bonus),
            "control_penalty": float(control_penalty),
            "target_pose_aligned": int(target_close),
            "gripper_open": int(self.gripper_state == "open"),
            "gripper_release_latched": int(self.gripper_release_latched),
            "gripper_should_open": int(self.last_gripper_should_open),
            "object_grasp_attached": int(self.object_grasp_attached),
            "success_counter": int(self.success_counter),
            "ik_success": (
                None
                if self._last_ik_result is None
                else int(bool(self._last_ik_result.success))
            ),
            "ik_failure_count": int(self._ik_failure_count),
        }
        return float(reward), reward_info

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
            [1.0 if self.gripper_state == "closed" else 0.0],
            dtype=np.float64,
        )
        release_latched = np.array(
            [1.0 if self.gripper_release_latched else 0.0],
            dtype=np.float64,
        )

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
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
                float(self.gripper_release_latched),
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
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("place_site_pos", place_site_pos),
            ("place_site_quat", place_site_quat),
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

    def export_config(self) -> dict:
        config = export_env_config(self, self._get_obs_components())
        config["action"]["controller"] = "standalone_cartesian_ik"
        config["action"]["action_components"] = list(self.ACTION_COMPONENTS)
        config["action"]["gripper_policy"] = "manual_open_at_above_target"
        config["action"]["gripper_open_distance"] = float(self._gripper_open_distance)
        config["action"]["gripper_open_target"] = self._gripper_open_target.tolist()
        config["action"]["gripper_closed_target"] = self._gripper_closed_target.tolist()
        config["task"]["target_mode"] = "active_place_site_plus_3cm"
        config["task"]["target_height_above_place"] = float(
            self._target_height_above_place
        )
        config["task"]["reset_mode"] = "procedural_ik_follows_sampled_object_pose"
        config["task"]["reset_position_noise_m"] = float(
            self._reset_position_noise
        )
        config["task"]["reset_orientation_noise_deg"] = float(
            np.rad2deg(self._reset_orientation_noise_rad)
        )
        config["task"]["reward_after_release"] = "bonus_only_when_distance_below_success_distance"
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        place_site_pos, place_site_quat = self._get_active_place_site_pose()
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
        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "place_site_pos": place_site_pos,
            "place_site_quat": place_site_quat,
            "sampled_target_site_pos": self.sampled_target_site_pos.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "sampled_target_place_site_pos": self.sampled_target_place_site_pos.copy(),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
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
            "target_height_above_place": float(self._target_height_above_place),
            "success_distance": float(self._success_distance),
            "gripper_state": self.gripper_state,
            "gripper_release_latched": bool(self.gripper_release_latched),
            "gripper_should_open": bool(self.last_gripper_should_open),
            "object_grasp_attached": bool(self.object_grasp_attached),
            "gripper_qpos": self.data.qpos[[self.gripL_qadr, self.gripR_qadr]].copy(),
            "gripper_ctrl": self.data.ctrl[
                [self.gripL_act_id, self.gripR_act_id]
            ].copy(),
            "reset_source": self.last_reset_source,
            "reset_attempts": int(self.last_reset_attempts),
            "reset_lift_height": float(self.last_reset_lift_height),
            "reset_ee_obj_dist": float(self.last_reset_ee_obj_dist),
            "reset_position_noise_m": float(self._reset_position_noise),
            "reset_grasp_position_noise": self.reset_grasp_position_noise.copy(),
            "reset_grasp_target_quat": self.reset_grasp_target_quat.copy(),
            "grasp_site_offset_ee": self._grasp_site_offset_ee.copy(),
            "grasp_relative_quat": self._grasp_relative_quat.copy(),
            "reset_orientation_noise_deg": float(
                np.rad2deg(self._reset_orientation_noise_rad)
            ),
            "success_counter": int(self.success_counter),
            "dense_reward_active": bool(not self.gripper_release_latched),
            "last_action": self.last_action.copy(),
            "task_mode": "ik_insert_grasped_reset_release_above_target_3cm",
            **self._get_ik_debug_state(),
        }

    def render(self):
        self.sync_visual_frames()
        return super().render()
