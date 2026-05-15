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
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)

from .config_export import capture_init_config, export_env_config

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"


class EndToEndInsertEnv(MujocoEnv, utils.EzPickle):
    """End-to-end grasp-to-insert task with heuristic manual gripper switching."""

    ACTION_COMPONENTS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    PHASES = ("approach", "grasp_pause", "carry", "release_pause", "released")

    def __init__(
        self,
        xml_file: str = str(DEFAULT_XML_PATH),
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_position_weight: float = 5.0,
        reward_position_tanh_weight: float = 2.0,
        reward_orientation_weight: float = 1.5,
        reward_orientation_tanh_weight: float = 1.0,
        reward_grasp_bonus: float = 8.0,
        reward_release_bonus: float = 30.0,
        control_penalty_weight: float = 0.001,
        position_tanh_scale: float = 0.05,
        orientation_tanh_scale: float = 0.5,
        grasp_distance: float = 0.01,
        grasp_angle_deg: float = 10.0,
        release_distance: float = 0.01,
        release_angle_deg: float = 10.0,
        release_height_above_place: float = 0.04,
        pause_steps_before_grasp: int = 1,
        pause_steps_before_release: int = 1,
        terminate_on_release: bool = True,
        max_episode_steps: int = 400,
        arm_action_scale: float = 0.01,
        cartesian_action_scale: float = 0.01,
        cartesian_rotation_scale_deg: float = 10.0,
        ik_workspace_low: tuple[float, float, float] = (0.08, -0.22, 0.015),
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
        control_interpolation_steps: int = 5,
        max_joint_ctrl_delta_deg: float = 5.0,
        smooth_cartesian_target: bool = True,
        object_x_range: tuple[float, float] = (0.16, 0.24),
        object_y_range: tuple[float, float] = (-0.10, 0.10),
        object_z: float = 0.025,
        object_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        target_x_range: tuple[float, float] = (0.20, 0.28),
        target_y_range: tuple[float, float] = (-0.10, 0.10),
        target_place_z: float = 0.0,
        target_place_yaw_range: tuple[float, float] = (
            -np.pi / 6.0,
            np.pi / 6.0,
        ),
        min_initial_object_target_distance: float = 0.06,
        target_resample_attempts: int = 100,
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        target_body_name: str = "target_body",
        **kwargs,
    ):
        self._init_config = capture_init_config(locals())
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_position_weight,
            reward_position_tanh_weight,
            reward_orientation_weight,
            reward_orientation_tanh_weight,
            reward_grasp_bonus,
            reward_release_bonus,
            control_penalty_weight,
            position_tanh_scale,
            orientation_tanh_scale,
            grasp_distance,
            grasp_angle_deg,
            release_distance,
            release_angle_deg,
            release_height_above_place,
            pause_steps_before_grasp,
            pause_steps_before_release,
            terminate_on_release,
            max_episode_steps,
            arm_action_scale,
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
            object_x_range,
            object_y_range,
            object_z,
            object_yaw_range,
            target_x_range,
            target_y_range,
            target_place_z,
            target_place_yaw_range,
            min_initial_object_target_distance,
            target_resample_attempts,
            ee_site_name,
            target_site_name,
            target_body_name,
            **kwargs,
        )

        self._reward_position_weight = float(reward_position_weight)
        self._reward_position_tanh_weight = float(reward_position_tanh_weight)
        self._reward_orientation_weight = float(reward_orientation_weight)
        self._reward_orientation_tanh_weight = float(reward_orientation_tanh_weight)
        self._reward_grasp_bonus = float(reward_grasp_bonus)
        self._reward_release_bonus = float(reward_release_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._position_tanh_scale = float(position_tanh_scale)
        self._orientation_tanh_scale = float(orientation_tanh_scale)
        self._grasp_distance = float(grasp_distance)
        self._grasp_angle_rad = np.deg2rad(float(grasp_angle_deg))
        self._release_distance = float(release_distance)
        self._release_angle_rad = np.deg2rad(float(release_angle_deg))
        self._release_height_above_place = float(release_height_above_place)
        self._pause_steps_before_grasp = max(0, int(pause_steps_before_grasp))
        self._pause_steps_before_release = max(0, int(pause_steps_before_release))
        self._terminate_on_release = bool(terminate_on_release)
        self.max_episode_steps = int(max_episode_steps)
        self._arm_action_scale = float(arm_action_scale)
        self._cartesian_action_scale = float(cartesian_action_scale)
        self._cartesian_rotation_scale_rad = np.deg2rad(
            float(cartesian_rotation_scale_deg)
        )
        self._ik_workspace_low = np.asarray(ik_workspace_low, dtype=np.float64).reshape(3)
        self._ik_workspace_high = np.asarray(ik_workspace_high, dtype=np.float64).reshape(
            3
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
        self._control_interpolation_steps = max(1, int(control_interpolation_steps))
        self._max_joint_ctrl_delta_rad = np.deg2rad(float(max_joint_ctrl_delta_deg))
        self._smooth_cartesian_target = bool(smooth_cartesian_target)
        self._object_x_range = tuple(float(value) for value in object_x_range)
        self._object_y_range = tuple(float(value) for value in object_y_range)
        self._object_z = float(object_z)
        self._object_yaw_range = tuple(float(value) for value in object_yaw_range)
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
        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)
        self.target_body_name = str(target_body_name)

        if self._position_tanh_scale <= 0.0:
            raise ValueError("position_tanh_scale must be greater than 0.")
        if self._orientation_tanh_scale <= 0.0:
            raise ValueError("orientation_tanh_scale must be greater than 0.")
        if self._grasp_distance <= 0.0:
            raise ValueError("grasp_distance must be greater than 0.")
        if self._grasp_angle_rad <= 0.0:
            raise ValueError("grasp_angle_deg must be greater than 0.")
        if self._release_distance <= 0.0:
            raise ValueError("release_distance must be greater than 0.")
        if self._release_angle_rad <= 0.0:
            raise ValueError("release_angle_deg must be greater than 0.")
        if self._release_height_above_place < 0.0:
            raise ValueError("release_height_above_place must be non-negative.")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be greater than 0.")
        if self._cartesian_action_scale <= 0.0:
            raise ValueError("cartesian_action_scale must be greater than 0.")
        if self._cartesian_rotation_scale_rad <= 0.0:
            raise ValueError("cartesian_rotation_scale_deg must be greater than 0.")
        if np.any(self._ik_workspace_low > self._ik_workspace_high):
            raise ValueError("ik_workspace_low must be ordered below ik_workspace_high.")
        if self._ik_max_iters <= 0:
            raise ValueError("ik_max_iters must be greater than 0.")
        if self._ik_position_tolerance <= 0.0:
            raise ValueError("ik_position_tolerance must be greater than 0.")
        if self._ik_rotation_tolerance_rad <= 0.0:
            raise ValueError("ik_rotation_tolerance_deg must be greater than 0.")
        if self._ik_damping < 0.0:
            raise ValueError("ik_damping must be non-negative.")
        if self._ik_step_size <= 0.0:
            raise ValueError("ik_step_size must be greater than 0.")
        if self._ik_max_delta_rad <= 0.0:
            raise ValueError("ik_max_delta_deg must be greater than 0.")
        if self._ik_rotation_weight < 0.0:
            raise ValueError("ik_rotation_weight must be non-negative.")
        if self._control_interpolation_steps <= 0:
            raise ValueError("control_interpolation_steps must be greater than 0.")
        if self._max_joint_ctrl_delta_rad < 0.0:
            raise ValueError("max_joint_ctrl_delta_deg must be non-negative.")
        if self._object_x_range[0] > self._object_x_range[1]:
            raise ValueError("object_x_range must be ordered as (min_x, max_x).")
        if self._object_y_range[0] > self._object_y_range[1]:
            raise ValueError("object_y_range must be ordered as (min_y, max_y).")
        if self._object_yaw_range[0] > self._object_yaw_range[1]:
            raise ValueError("object_yaw_range must be ordered as (min_yaw, max_yaw).")
        if self._target_x_range[0] > self._target_x_range[1]:
            raise ValueError("target_x_range must be ordered as (min_x, max_x).")
        if self._target_y_range[0] > self._target_y_range[1]:
            raise ValueError("target_y_range must be ordered as (min_y, max_y).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError(
                "target_place_yaw_range must be ordered as (min_yaw, max_yaw)."
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

        self._gripper_open_target = np.array([0.01, -0.01], dtype=np.float64)
        self._gripper_closed_target = np.array([-0.02, 0.02], dtype=np.float64)

        self.object_names = self._discover_available_objects()
        if not self.object_names:
            raise ValueError(
                "EndToEndInsertEnv requires at least one object and matching place "
                "body/site/geom in the XML."
            )
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
        self._place_site_local_pose_by_object: dict[
            str, tuple[np.ndarray, np.ndarray]
        ] = {}
        for obj_name, info in self.place_info.items():
            site_id = int(info["site_id"])
            self._place_site_local_pose_by_object[obj_name] = (
                self.model.site_pos[site_id].copy(),
                self._normalize_quat(self.model.site_quat[site_id].copy()),
            )

        self.target_site_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_SITE, self.target_site_name, "site"
        )
        self.target_body_id = self._require_named_id(
            mujoco.mjtObj.mjOBJ_BODY, self.target_body_name, "body"
        )
        self._target_site_local_pos = self.model.site_pos[self.target_site_id].copy()
        self._target_site_local_quat = self._normalize_quat(
            self.model.site_quat[self.target_site_id].copy()
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

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        if self.model.nu < 3:
            raise ValueError(
                "EndToEndInsertEnv expects arm actuators plus 2 gripper actuators."
            )
        self._setup_ik_action(xml_file=str(self.fullpath))
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.ACTION_COMPONENTS),),
            dtype=np.float32,
        )

        self.active_obj_name = self.object_names[0]
        self.phase = "approach"
        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_effective_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_zero_action_reason = "none"
        self.last_grasp_ready = False
        self.last_release_ready = False
        self.last_grasp_event = False
        self.last_release_event = False
        self.gripper_state = "open"
        self.grasp_latched = False
        self.release_latched = False
        self.grasp_bonus_given = False
        self.release_bonus_given = False
        self.grasp_pause_steps_left = 0
        self.release_pause_steps_left = 0
        self.success_counter = 0
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
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

    def _name2id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, obj_type, name))

    def _require_named_id(self, obj_type: mujoco.mjtObj, name: str, label: str) -> int:
        obj_id = self._name2id(obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Missing {label} `{name}` in XML scene.")
        return obj_id

    def _discover_available_objects(self) -> list[str]:
        object_names = []
        for obj_name in ("box", "triangle", "cylinder"):
            names = (
                (mujoco.mjtObj.mjOBJ_BODY, f"obj_{obj_name}"),
                (mujoco.mjtObj.mjOBJ_JOINT, f"obj_{obj_name}_joint"),
                (mujoco.mjtObj.mjOBJ_SITE, f"obj_{obj_name}_ref"),
                (mujoco.mjtObj.mjOBJ_GEOM, f"obj_{obj_name}_geom"),
                (
                    mujoco.mjtObj.mjOBJ_BODY,
                    {
                        "box": "cube_place",
                        "triangle": "tri_place",
                        "cylinder": "cyl_place",
                    }[obj_name],
                ),
                (
                    mujoco.mjtObj.mjOBJ_SITE,
                    {
                        "box": "cube_place_site",
                        "triangle": "tri_place_site",
                        "cylinder": "cyl_place_site",
                    }[obj_name],
                ),
                (
                    mujoco.mjtObj.mjOBJ_GEOM,
                    {
                        "box": "cube_place_geom",
                        "triangle": "tri_place_geom",
                        "cylinder": "cyl_place_geom",
                    }[obj_name],
                ),
            )
            if all(self._name2id(obj_type, name) >= 0 for obj_type, name in names):
                object_names.append(obj_name)
        return object_names

    def _setup_ik_action(self, *, xml_file: str) -> None:
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
                "Cartesian IK action expects arm actuator count to match IK joint count. "
                f"Got arm_ctrl_dim={self._arm_ctrl_dim} and "
                f"ik_joints={len(self._ik_solver.joint_names)}."
            )

        self._ik_target_pos = np.zeros(3, dtype=np.float64)
        self._ik_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_ik_result: IKResult | None = None
        self._ik_failure_count = 0
        self._reset_ik_state()

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
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half_yaw = float(yaw) / 2.0
        return np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        quat = EndToEndInsertEnv._normalize_quat(quat)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _wrap_vector_to_pi(self, angles_rad: np.ndarray) -> np.ndarray:
        return np.array(
            [self._wrap_to_pi(float(value)) for value in angles_rad],
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

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _get_active_place_site_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_place_info()["site_name"]))

    def _get_release_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        target_pos, target_quat = self._get_active_place_site_pose()
        target_pos = target_pos.copy()
        target_pos[2] += self._release_height_above_place
        return target_pos, target_quat

    def _current_arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self._arm_qpos_indices], dtype=np.float64
        ).copy()

    def _reset_ik_state(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        self._ik_target_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        self._ik_target_quat = self._normalize_quat(
            np.asarray(ee_quat, dtype=np.float64)
        )
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
        target_rpy = self._wrap_vector_to_pi(_quat_to_euler_xyz(base_quat) + delta_rpy)
        target_quat = _quat_from_euler_xyz(*target_rpy)
        return target_pos, target_quat

    def _apply_ik_action_to_ctrl(
        self, target_ctrl: np.ndarray, effective_action: np.ndarray
    ) -> np.ndarray:
        target_pos, target_quat = self._compute_ik_target(effective_action)
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
        self._ik_target_pos = target_pos.copy()
        self._ik_target_quat = self._normalize_quat(target_quat)
        self._last_ik_result = ik_result
        if not ik_result.success:
            self._ik_failure_count += 1
        return target_ctrl

    def _do_interpolated_simulation(self, target_ctrl: np.ndarray) -> None:
        frame_count = max(1, int(self.frame_skip))
        interpolation_steps = min(
            max(1, int(self._control_interpolation_steps)), frame_count
        )
        start_ctrl = self.data.ctrl.copy()
        frames_done = 0

        for interp_idx in range(1, interpolation_steps + 1):
            remaining_frames = frame_count - frames_done
            remaining_interps = interpolation_steps - interp_idx + 1
            frames_this_step = max(1, remaining_frames // remaining_interps)
            alpha = interp_idx / interpolation_steps
            smooth_ctrl = (1.0 - alpha) * start_ctrl + alpha * target_ctrl
            smooth_ctrl = np.clip(smooth_ctrl, self._ctrl_low, self._ctrl_high)
            self.do_simulation(smooth_ctrl, frames_this_step)
            frames_done += frames_this_step

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[-2:] = self._gripper_closed_target

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[-2:] = self._gripper_open_target

    def _update_gripper_state_from_target(self, target: np.ndarray) -> None:
        self.gripper_state = "closed" if target[-2] < target[-1] else "open"

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

    def _set_target_site_pose_in_model(
        self, target_site_pos: np.ndarray, target_site_quat: np.ndarray
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

    def _set_active_place_visual(self) -> None:
        for obj_name, info in self.place_info.items():
            rgba = self.place_geom_rgba[obj_name].copy()
            rgba[3] = (
                self.place_geom_rgba[obj_name][3]
                if obj_name == self.active_obj_name
                else 0.0
            )
            self.model.geom_rgba[int(info["geom_id"])] = rgba

    def _sample_object_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = self.np_random.uniform(*self._object_x_range)
        y = self.np_random.uniform(*self._object_y_range)
        yaw = self.np_random.uniform(*self._object_yaw_range)
        pos = np.array([x, y, self._object_z], dtype=np.float64)
        quat = self._yaw_to_quat(float(yaw))
        return pos, quat, float(yaw)

    def _sample_target_place_pose(
        self, object_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        best_pos = None
        best_quat = None
        best_yaw = 0.0
        best_dist = -np.inf

        for _ in range(self._target_resample_attempts):
            x = self.np_random.uniform(*self._target_x_range)
            y = self.np_random.uniform(*self._target_y_range)
            yaw = float(self.np_random.uniform(*self._target_place_yaw_range))
            pos = np.array([x, y, self._target_place_z], dtype=np.float64)
            quat = self._yaw_to_quat(yaw)

            place_site_local_pos, _ = self._place_site_local_pose_by_object[
                self.active_obj_name
            ]
            place_site_pos = pos + self._quat_rotate_vector(quat, place_site_local_pos)
            release_target_pos = place_site_pos.copy()
            release_target_pos[2] += self._release_height_above_place
            dist = float(np.linalg.norm(release_target_pos - object_pos))

            if dist > best_dist:
                best_pos = pos
                best_quat = quat
                best_yaw = yaw
                best_dist = dist
            if dist >= self._min_initial_object_target_distance:
                return pos, quat, yaw

        assert best_pos is not None
        assert best_quat is not None
        return best_pos, best_quat, best_yaw

    def _get_alignment_metrics(self) -> dict[str, np.ndarray | float | bool]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_release_target_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        obj_target_dist = float(np.linalg.norm(obj_target_pos_error))
        obj_target_angle = float(np.linalg.norm(obj_target_rot_error))

        grasp_ready = bool(
            ee_obj_dist < self._grasp_distance and ee_obj_angle < self._grasp_angle_rad
        )
        release_ready = bool(
            obj_target_dist < self._release_distance
            and obj_target_angle < self._release_angle_rad
        )

        return {
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": ee_obj_dist,
            "ee_obj_angle": ee_obj_angle,
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": obj_target_dist,
            "obj_target_angle": obj_target_angle,
            "grasp_ready": grasp_ready,
            "release_ready": release_ready,
        }

    def _compute_ctrl_and_events(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        metrics = self._get_alignment_metrics()
        self.last_grasp_ready = bool(metrics["grasp_ready"])
        self.last_release_ready = bool(metrics["release_ready"])
        self.last_grasp_event = False
        self.last_release_event = False
        self.last_zero_action_reason = "none"

        target_ctrl = self.data.ctrl.copy()
        effective_action = action.copy()
        if self.phase == "approach":
            self._set_open_gripper_target(target_ctrl)
            if self.last_grasp_ready:
                self.grasp_pause_steps_left = self._pause_steps_before_grasp
                if self.grasp_pause_steps_left > 0:
                    self.phase = "grasp_pause"
                    effective_action = np.zeros_like(action)
                    self.grasp_pause_steps_left -= 1
                    self.last_zero_action_reason = "pre_grasp_pause"
                else:
                    self.phase = "carry"
                    self.grasp_latched = True
                    self.last_grasp_event = True
                    effective_action = np.zeros_like(action)
                    self.last_zero_action_reason = "grasp_close"
                    self._set_closed_gripper_target(target_ctrl)

        elif self.phase == "grasp_pause":
            effective_action = np.zeros_like(action)
            if self.grasp_pause_steps_left > 0:
                self.grasp_pause_steps_left -= 1
                self._set_open_gripper_target(target_ctrl)
                self.last_zero_action_reason = "pre_grasp_pause"
            else:
                self.phase = "carry"
                self.grasp_latched = True
                self.last_grasp_event = True
                self.last_zero_action_reason = "grasp_close"
                self._set_closed_gripper_target(target_ctrl)

        elif self.phase == "carry":
            self._set_closed_gripper_target(target_ctrl)
            if self.last_release_ready:
                self.release_pause_steps_left = self._pause_steps_before_release
                if self.release_pause_steps_left > 0:
                    self.phase = "release_pause"
                    effective_action = np.zeros_like(action)
                    self.release_pause_steps_left -= 1
                    self.last_zero_action_reason = "pre_release_pause"
                else:
                    self.phase = "released"
                    self.release_latched = True
                    self.last_release_event = True
                    effective_action = np.zeros_like(action)
                    self.last_zero_action_reason = "release_open"
                    self._set_open_gripper_target(target_ctrl)

        elif self.phase == "release_pause":
            effective_action = np.zeros_like(action)
            if self.release_pause_steps_left > 0:
                self.release_pause_steps_left -= 1
                self._set_closed_gripper_target(target_ctrl)
                self.last_zero_action_reason = "pre_release_pause"
            else:
                self.phase = "released"
                self.release_latched = True
                self.last_release_event = True
                self.last_zero_action_reason = "release_open"
                self._set_open_gripper_target(target_ctrl)

        elif self.phase == "released":
            effective_action = np.zeros_like(action)
            self.release_latched = True
            self.last_zero_action_reason = "released"
            self._set_open_gripper_target(target_ctrl)

        else:
            raise RuntimeError(f"Unknown phase `{self.phase}`.")

        target_ctrl = self._apply_ik_action_to_ctrl(target_ctrl, effective_action)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(target_ctrl)

        return target_ctrl, effective_action, metrics

    def _pose_reward_terms(
        self,
        position_error: float,
        orientation_error: float,
        prefix: str,
    ) -> tuple[float, dict]:
        reward_position = -position_error * self._reward_position_weight
        reward_position_tanh = (
            1.0 - float(np.tanh(position_error / self._position_tanh_scale))
        ) * self._reward_position_tanh_weight
        reward_orientation = -orientation_error * self._reward_orientation_weight
        reward_orientation_tanh = (
            1.0 - float(np.tanh(orientation_error / self._orientation_tanh_scale))
        ) * self._reward_orientation_tanh_weight
        reward = (
            reward_position
            + reward_position_tanh
            + reward_orientation
            + reward_orientation_tanh
        )
        return reward, {
            f"reward_{prefix}_position": float(reward_position),
            f"reward_{prefix}_position_tanh": float(reward_position_tanh),
            f"reward_{prefix}_orientation": float(reward_orientation),
            f"reward_{prefix}_orientation_tanh": float(reward_orientation_tanh),
            f"reward_{prefix}_total": float(reward),
        }

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != self.action_space.shape:
            expected = ", ".join(self.ACTION_COMPONENTS)
            raise ValueError(
                "Unexpected action shape for EndToEndInsertEnv. "
                f"Expected {self.action_space.shape} ({expected}), got {action.shape}."
            )
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.astype(np.float32)

        target_ctrl, effective_action, pre_metrics = self._compute_ctrl_and_events(
            action
        )
        self.last_effective_action = effective_action.astype(np.float32)

        self._do_interpolated_simulation(target_ctrl)
        target_pos, target_quat = self._get_release_target_pose()
        self._set_target_site_pose_in_model(target_pos, target_quat)
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(effective_action, pre_metrics)
        terminated_success = bool(self.release_latched and self._terminate_on_release)
        if self.release_latched:
            self.success_counter = 1
        terminated = terminated_success
        truncated = self.current_step >= self.max_episode_steps
        reward_info["terminated_success"] = int(terminated_success)

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(
        self, effective_action: np.ndarray, pre_metrics: dict[str, np.ndarray | float]
    ) -> tuple[float, dict]:
        metrics = self._get_alignment_metrics()
        reward_grasp_bonus = (
            self._reward_grasp_bonus
            if self.last_grasp_event and not self.grasp_bonus_given
            else 0.0
        )
        if self.last_grasp_event:
            self.grasp_bonus_given = True

        reward_release_bonus = (
            self._reward_release_bonus
            if self.last_release_event and not self.release_bonus_given
            else 0.0
        )
        if self.last_release_event:
            self.release_bonus_given = True

        if self.release_latched:
            reward = reward_release_bonus
            dense_info = {
                "reward_position": 0.0,
                "reward_position_tanh": 0.0,
                "reward_orientation": 0.0,
                "reward_orientation_tanh": 0.0,
                "reward_ee_object_position": 0.0,
                "reward_ee_object_position_tanh": 0.0,
                "reward_ee_object_orientation": 0.0,
                "reward_ee_object_orientation_tanh": 0.0,
                "reward_ee_object_total": 0.0,
                "reward_object_target_position": 0.0,
                "reward_object_target_position_tanh": 0.0,
                "reward_object_target_orientation": 0.0,
                "reward_object_target_orientation_tanh": 0.0,
                "reward_object_target_total": 0.0,
                "control_penalty": 0.0,
            }
            reward_grasp_bonus = 0.0
            active_position_error = float(metrics["obj_target_dist"])
            active_orientation_error = float(metrics["obj_target_angle"])
            active_reward_target = "released_bonus_only"
        else:
            object_target_reward, object_target_info = self._pose_reward_terms(
                float(metrics["obj_target_dist"]),
                float(metrics["obj_target_angle"]),
                "object_target",
            )
            dense_reward = object_target_reward
            dense_info = dict(object_target_info)

            if self.phase in {"approach", "grasp_pause"}:
                active_position_error = float(metrics["ee_obj_dist"])
                active_orientation_error = float(metrics["ee_obj_angle"])
                active_reward_target = "ee_to_object_plus_object_to_release_target"
                ee_object_reward, ee_object_info = self._pose_reward_terms(
                    active_position_error,
                    active_orientation_error,
                    "ee_object",
                )
                dense_reward += ee_object_reward
                dense_info.update(ee_object_info)
            else:
                active_position_error = float(metrics["obj_target_dist"])
                active_orientation_error = float(metrics["obj_target_angle"])
                active_reward_target = "object_to_release_target"
                dense_info.update(
                    {
                        "reward_ee_object_position": 0.0,
                        "reward_ee_object_position_tanh": 0.0,
                        "reward_ee_object_orientation": 0.0,
                        "reward_ee_object_orientation_tanh": 0.0,
                        "reward_ee_object_total": 0.0,
                    }
                )

            reward_position = float(
                dense_info["reward_object_target_position"]
                + dense_info["reward_ee_object_position"]
            )
            reward_position_tanh = float(
                dense_info["reward_object_target_position_tanh"]
                + dense_info["reward_ee_object_position_tanh"]
            )
            reward_orientation = float(
                dense_info["reward_object_target_orientation"]
                + dense_info["reward_ee_object_orientation"]
            )
            reward_orientation_tanh = float(
                dense_info["reward_object_target_orientation_tanh"]
                + dense_info["reward_ee_object_orientation_tanh"]
            )
            control_penalty = -self._control_penalty_weight * float(
                np.sum(np.square(effective_action))
            )
            dense_info.update(
                {
                    "reward_position": reward_position,
                    "reward_position_tanh": reward_position_tanh,
                    "reward_orientation": reward_orientation,
                    "reward_orientation_tanh": reward_orientation_tanh,
                    "control_penalty": float(control_penalty),
                }
            )
            reward = dense_reward + control_penalty + reward_grasp_bonus

        ik_success = (
            -1
            if self._last_ik_result is None
            else int(bool(self._last_ik_result.success))
        )
        ik_position_error_norm = (
            0.0
            if self._last_ik_result is None
            else float(self._last_ik_result.position_error_norm)
        )
        ik_rotation_error_norm = (
            0.0
            if self._last_ik_result is None
            else float(self._last_ik_result.rotation_error_norm)
        )
        reward_info = {
            "active_object": self.active_obj_name,
            "phase": self.phase,
            "active_reward_target": active_reward_target,
            "position_error": float(active_position_error),
            "orientation_error": float(active_orientation_error),
            "ee_object_dist": float(metrics["ee_obj_dist"]),
            "ee_object_rot_error": float(metrics["ee_obj_angle"]),
            "object_target_dist": float(metrics["obj_target_dist"]),
            "object_target_rot_error": float(metrics["obj_target_angle"]),
            "pre_ee_object_dist": float(pre_metrics["ee_obj_dist"]),
            "pre_ee_object_rot_error": float(pre_metrics["ee_obj_angle"]),
            "pre_object_target_dist": float(pre_metrics["obj_target_dist"]),
            "pre_object_target_rot_error": float(pre_metrics["obj_target_angle"]),
            "grasp_ready": int(self.last_grasp_ready),
            "release_ready": int(self.last_release_ready),
            "grasp_latched": int(self.grasp_latched),
            "release_latched": int(self.release_latched),
            "grasp_event": int(self.last_grasp_event),
            "release_event": int(self.last_release_event),
            "reward_grasp_bonus": float(reward_grasp_bonus),
            "reward_release_bonus": float(reward_release_bonus),
            "zero_action_reason": self.last_zero_action_reason,
            "gripper_state": self.gripper_state,
            "gripper_ctrl_left": float(self.data.ctrl[-2]),
            "gripper_ctrl_right": float(self.data.ctrl[-1]),
            "success_counter": int(self.success_counter),
            "ik_success": ik_success,
            "ik_failure_count": int(self._ik_failure_count),
            "ik_position_error_norm": ik_position_error_norm,
            "ik_rotation_error_norm": ik_rotation_error_norm,
            "ik_target_x": float(self._ik_target_pos[0]),
            "ik_target_y": float(self._ik_target_pos[1]),
            "ik_target_z": float(self._ik_target_pos[2]),
        }
        reward_info.update(dense_info)

        return float(reward), reward_info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.active_obj_name = str(self.np_random.choice(self.object_names))
        obj_pos, obj_quat, object_yaw = self._sample_object_pose()
        (
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
        ) = self._sample_target_place_pose(obj_pos)
        self._set_place_poses_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()

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
                    [6.0, 1.0, 1.0], dtype=np.float64
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
        target_pos, target_quat = self._get_release_target_pose()
        self._set_target_site_pose_in_model(target_pos, target_quat)
        mujoco.mj_forward(self.model, self.data)

        self.phase = "approach"
        self.current_step = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_effective_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_zero_action_reason = "none"
        self.last_grasp_ready = False
        self.last_release_ready = False
        self.last_grasp_event = False
        self.last_release_event = False
        self.gripper_state = "open"
        self.grasp_latched = False
        self.release_latched = False
        self.grasp_bonus_given = False
        self.release_bonus_given = False
        self.grasp_pause_steps_left = 0
        self.release_pause_steps_left = 0
        self.success_counter = 0
        self.initial_obj_site_pos = self._get_active_obj_pose()[0].copy()
        self.sampled_object_yaw = float(object_yaw)
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(
                self.data.body(
                    str(self._get_active_place_info()["body_name"])
                ).xquat.copy()
            )
        )
        self._reset_ik_state()

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
        arm_ctrl = self.data.ctrl[self._arm_ctrl_indices].copy()
        gripper_qpos = qpos[[self.gripL_qadr, self.gripR_qadr]].copy()
        gripper_qvel = qvel[[self.gripL_dadr, self.gripR_dadr]].copy()
        gripper_ctrl = self.data.ctrl[-2:].copy()
        gripper_closed = np.array(
            [1.0 if self.gripper_state == "closed" else 0.0], dtype=np.float64
        )
        phase_one_hot = np.zeros(len(self.PHASES), dtype=np.float64)
        phase_one_hot[self.PHASES.index(self.phase)] = 1.0

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_release_target_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        ik_success = (
            -1.0
            if self._last_ik_result is None
            else float(bool(self._last_ik_result.success))
        )

        return [
            ("robot_qpos", robot_qpos),
            ("robot_qvel", robot_qvel),
            ("arm_ctrl", arm_ctrl),
            ("gripper_qpos", gripper_qpos),
            ("gripper_qvel", gripper_qvel),
            ("gripper_ctrl", gripper_ctrl),
            ("gripper_closed", gripper_closed),
            ("phase", phase_one_hot),
            ("object_type", self.object_one_hot[self.active_obj_name]),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("ik_target_pos", self._ik_target_pos),
            ("ik_target_quat", self._ik_target_quat),
            ("last_action", self.last_action),
            ("last_effective_action", self.last_effective_action),
            ("ee_object_pos_error", ee_obj_pos_error),
            ("ee_object_rot_error", ee_obj_rot_error),
            ("object_target_pos_error", obj_target_pos_error),
            ("object_target_rot_error", obj_target_rot_error),
            (
                "alignment_scalars",
                np.array(
                    [
                        np.linalg.norm(ee_obj_pos_error),
                        np.linalg.norm(ee_obj_rot_error),
                        np.linalg.norm(obj_target_pos_error),
                        np.linalg.norm(obj_target_rot_error),
                        float(self.grasp_latched),
                        float(self.release_latched),
                        ik_success,
                        float(self._ik_failure_count),
                    ],
                    dtype=np.float64,
                ),
            ),
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
        config["action"]["controller"] = "cartesian_ik"
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
        config["action"]["ik_position_tolerance"] = float(
            self._ik_position_tolerance
        )
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
        config["action"]["gripper_policy"] = "manual_grasp_release_heuristic"
        config["action"]["grasp_distance"] = float(self._grasp_distance)
        config["action"]["grasp_angle_deg"] = float(np.rad2deg(self._grasp_angle_rad))
        config["action"]["release_distance"] = float(self._release_distance)
        config["action"]["release_angle_deg"] = float(
            np.rad2deg(self._release_angle_rad)
        )
        config["action"]["release_height_above_place"] = float(
            self._release_height_above_place
        )
        config["action"]["pause_steps_before_grasp"] = int(
            self._pause_steps_before_grasp
        )
        config["action"]["pause_steps_before_release"] = int(
            self._pause_steps_before_release
        )
        config["action"]["gripper_open_target"] = self._gripper_open_target.tolist()
        config["action"]["gripper_closed_target"] = self._gripper_closed_target.tolist()
        config["reward"]["params"]["release_reward_mode"] = "release_bonus_only"
        config["reward"]["params"]["object_target_reward_active_from_reset"] = True
        config["reward"]["params"]["ee_object_reward_active_until_grasp"] = True
        config["reward"]["params"]["position_tanh_scale"] = float(
            self._position_tanh_scale
        )
        config["reward"]["params"]["orientation_tanh_scale"] = float(
            self._orientation_tanh_scale
        )
        config["task"]["target_mode"] = "release_target_4cm_above_active_place_site"
        config["task"]["phases"] = list(self.PHASES)
        config["task"]["terminate_on_release"] = bool(self._terminate_on_release)
        config["task"]["available_objects"] = list(self.object_names)
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
            "min_initial_object_target_distance": float(
                self._min_initial_object_target_distance
            ),
        }
        return config

    def get_debug_state(self) -> dict:
        metrics = self._get_alignment_metrics()
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_release_target_pose()
        target_place_quat = self._normalize_quat(
            self.data.body(str(self._get_active_place_info()["body_name"])).xquat.copy()
        )
        ik_debug = {
            "ik_target_pos": self._ik_target_pos.copy(),
            "ik_target_quat": self._ik_target_quat.copy(),
            "ik_failure_count": int(self._ik_failure_count),
            "control_interpolation_steps": int(self._control_interpolation_steps),
            "smooth_cartesian_target": bool(self._smooth_cartesian_target),
        }
        if self._last_ik_result is None:
            ik_debug.update(
                {
                    "ik_success": None,
                    "ik_iterations": 0,
                    "ik_position_error_norm": 0.0,
                    "ik_rotation_error_rad": 0.0,
                }
            )
        else:
            ik_debug.update(
                {
                    "ik_success": bool(self._last_ik_result.success),
                    "ik_message": self._last_ik_result.message,
                    "ik_iterations": int(self._last_ik_result.iterations),
                    "ik_position_error_norm": float(
                        self._last_ik_result.position_error_norm
                    ),
                    "ik_rotation_error_rad": float(
                        self._last_ik_result.rotation_error_norm
                    ),
                }
            )

        return {
            "active_object": self.active_obj_name,
            "phase": self.phase,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_place_pos": self.data.body(
                str(self._get_active_place_info()["body_name"])
            ).xpos.copy(),
            "target_place_quat": target_place_quat,
            "ee_obj_pos_error": np.asarray(
                metrics["ee_obj_pos_error"], dtype=np.float64
            ).copy(),
            "ee_obj_rot_error": np.asarray(
                metrics["ee_obj_rot_error"], dtype=np.float64
            ).copy(),
            "ee_obj_dist": float(metrics["ee_obj_dist"]),
            "ee_obj_angle_rad": float(metrics["ee_obj_angle"]),
            "obj_target_pos_error": np.asarray(
                metrics["obj_target_pos_error"], dtype=np.float64
            ).copy(),
            "obj_target_rot_error": np.asarray(
                metrics["obj_target_rot_error"], dtype=np.float64
            ).copy(),
            "obj_target_dist": float(metrics["obj_target_dist"]),
            "obj_target_angle_rad": float(metrics["obj_target_angle"]),
            "grasp_ready": bool(metrics["grasp_ready"]),
            "release_ready": bool(metrics["release_ready"]),
            "grasp_latched": bool(self.grasp_latched),
            "release_latched": bool(self.release_latched),
            "gripper_state": self.gripper_state,
            "last_zero_action_reason": self.last_zero_action_reason,
            "last_action": self.last_action.copy(),
            "last_effective_action": self.last_effective_action.copy(),
            "arm_ctrl": self.data.ctrl[self._arm_ctrl_indices].copy(),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "release_height_above_place": float(self._release_height_above_place),
            "success_counter": int(self.success_counter),
            **ik_debug,
        }

    def render(self):
        return super().render()
