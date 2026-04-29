from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.utils import seeding

from .config_export import capture_init_config, export_env_config
from .place_target_env import DEFAULT_GRASP_XML_PATH, GRASP_ENV_REGISTRY

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_place.xml"
TARGET_HEIGHT_ABOVE_PLACE = 0.03
TARGET_PLACE_YAW_DEG = 30.0
TARGET_PLACE_YAW_RAD = np.deg2rad(TARGET_PLACE_YAW_DEG)
GRIPPER_OPEN_DISTANCE_THRESHOLD = 0.008
GRIPPER_OPEN_ANGLE_DEG = 20.0
GRIPPER_OPEN_ANGLE_RAD = np.deg2rad(GRIPPER_OPEN_ANGLE_DEG)
INACTIVE_PLACE_BASE_POS = np.array([10.0, 0.0, 0.0], dtype=np.float64)
INACTIVE_PLACE_Y_OFFSETS = {
    "box": 0.00,
    "triangle": 0.20,
    "cylinder": -0.20,
}


class PlaceAboveSiteEnv(MujocoEnv, utils.EzPickle):
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
        reward_target_tanh_weight: float = 3.0,
        reward_orientation_weight: float = 2.0,
        reward_target_bonus: float = 15.0,
        control_penalty_weight: float = 0.001,
        success_distance: float = 0.015,
        max_episode_steps: int = 150,
        arm_action_scale: float = 0.01,
        target_x_range: tuple[float, float] = (0.15, 0.27),
        target_y_range: tuple[float, float] = (-0.20, 0.20),
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        grasp_model_path: str | None = None,
        grasp_env_name: str = "GraspingEnv",
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
            reward_target_bonus,
            control_penalty_weight,
            success_distance,
            max_episode_steps,
            arm_action_scale,
            target_x_range,
            target_y_range,
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
            **kwargs,
        )

        if grasp_model_path is None:
            raise ValueError(
                "PlaceAboveSiteEnv requires `grasp_model_path` so reset can start from the grasping policy state."
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
        self._reward_target_bonus = float(reward_target_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._success_distance = float(success_distance)
        if self._success_distance <= 0.0:
            raise ValueError("success_distance must be greater than 0.")
        self.max_episode_steps = int(max_episode_steps)
        self._arm_action_scale = float(arm_action_scale)
        self._target_x_range = tuple(float(value) for value in target_x_range)
        self._target_y_range = tuple(float(value) for value in target_y_range)
        self._target_height_above_place = TARGET_HEIGHT_ABOVE_PLACE
        self._target_place_yaw_rad = TARGET_PLACE_YAW_RAD
        if self._target_x_range[0] > self._target_x_range[1]:
            raise ValueError("target_x_range must be ordered as (min_x, max_x).")
        if self._target_y_range[0] > self._target_y_range[1]:
            raise ValueError("target_y_range must be ordered as (min_y, max_y).")

        self.ee_site_name = str(ee_site_name)
        self.target_site_name = str(target_site_name)
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

        self._grasp_env = None
        self._grasp_policy = None
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

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)

            self.object_info[obj_name] = {
                "body_name": body_name,
                "joint_name": joint_name,
                "site_name": site_name,
                "body_id": body_id,
                "joint_id": joint_id,
                "site_id": site_id,
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
        self.place_info = self._build_place_info()
        if not self.place_info:
            raise ValueError(
                "PlaceAboveSiteEnv requires place bodies/sites/geoms in the XML scene."
            )
        self._default_place_pose_by_object = {
            obj_name: {
                "pos": self.model.body_pos[int(info["body_id"])].copy(),
                "quat": self.model.body_quat[int(info["body_id"])].copy(),
            }
            for obj_name, info in self.place_info.items()
        }

        self.active_obj_name = self.object_names[0]
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_body"
        )
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site_name
        )

        gripL_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_10"
        )
        gripR_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "Slider_11"
        )
        self.gripL_qadr = int(self.model.jnt_qposadr[gripL_jid])
        self.gripR_qadr = int(self.model.jnt_qposadr[gripR_jid])
        self.gripL_dadr = int(self.model.jnt_dofadr[gripL_jid])
        self.gripR_dadr = int(self.model.jnt_dofadr[gripR_jid])

        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        if self.model.nu < 3:
            raise ValueError(
                "PlaceAboveSiteEnv expects arm actuators plus 2 gripper actuators."
            )
        self._arm_ctrl_dim = int(self.model.nu - 2)
        self._policy_action_dim = self._arm_ctrl_dim
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_action_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self.sampled_target_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_pos = np.zeros(3, dtype=np.float64)

        self._sync_target_site_to_active_place()
        closed_ctrl = np.zeros(int(self.model.nu), dtype=np.float64)
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

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

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        quat = PlaceAboveSiteEnv._normalize_quat(quat)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _yaw_to_quat(yaw_rad: float) -> np.ndarray:
        half_yaw = 0.5 * float(yaw_rad)
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

    @staticmethod
    def _get_named_id_or_none(model, obj_type, obj_name: str) -> int | None:
        obj_id = int(mujoco.mj_name2id(model, obj_type, obj_name))
        return obj_id if obj_id >= 0 else None

    def _build_place_info(self) -> dict[str, dict[str, int | str]]:
        place_info: dict[str, dict[str, int | str]] = {}
        for obj_name in self.object_names:
            body_name = self.place_name_by_object[obj_name]
            site_name = self.place_site_name_by_object[obj_name]
            geom_name = self.place_geom_name_by_object[obj_name]

            body_id = self._get_named_id_or_none(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            site_id = self._get_named_id_or_none(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            geom_id = self._get_named_id_or_none(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if body_id is None or site_id is None or geom_id is None:
                return {}

            place_info[obj_name] = {
                "body_name": body_name,
                "site_name": site_name,
                "geom_name": geom_name,
                "body_id": body_id,
                "site_id": site_id,
                "geom_id": geom_id,
            }

        return place_info

    def _get_active_place_info(self) -> dict[str, int | str]:
        if not self.place_info:
            raise ValueError("Active place info is unavailable in the loaded XML.")
        return self.place_info[self.active_obj_name]

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        target_pos, target_quat = self._get_site_pose(
            str(self._get_active_place_info()["site_name"])
        )
        target_pos = target_pos.copy()
        target_pos[2] += self._target_height_above_place
        return target_pos, target_quat

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[-2:] = self._gripper_closed_target

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[-2:] = self._gripper_open_target

    def _get_object_target_metrics(self) -> tuple[float, float]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        return target_dist, target_angle

    def _should_open_gripper(self) -> bool:
        target_dist, target_angle = self._get_object_target_metrics()
        return bool(
            target_dist < GRIPPER_OPEN_DISTANCE_THRESHOLD
            and target_angle < GRIPPER_OPEN_ANGLE_RAD
        )

    def _coerce_policy_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape == self.action_space.shape:
            return action

        place_action_shape = (self._arm_ctrl_dim + 1,)
        if action.shape == place_action_shape:
            return action[: self._arm_ctrl_dim]

        legacy_shape = (int(self.model.nu),)
        if action.shape == legacy_shape:
            return action[: self._arm_ctrl_dim]

        raise ValueError(
            "Unexpected action shape for PlaceAboveSiteEnv. "
            f"Expected {self.action_space.shape} (arm only), "
            f"{place_action_shape} (arm + gripper command), "
            f"or legacy {legacy_shape}, got {action.shape}."
        )

    def _restore_default_place_poses(self) -> None:
        for obj_name, pose in self._default_place_pose_by_object.items():
            body_id = int(self.place_info[obj_name]["body_id"])
            self.model.body_pos[body_id] = np.asarray(pose["pos"], dtype=np.float64)
            self.model.body_quat[body_id] = np.asarray(pose["quat"], dtype=np.float64)

    def _move_inactive_places_away(self) -> None:
        for obj_name, place_info in self.place_info.items():
            if obj_name == self.active_obj_name:
                continue

            body_id = int(place_info["body_id"])
            far_pos = INACTIVE_PLACE_BASE_POS.copy()
            far_pos[1] = INACTIVE_PLACE_Y_OFFSETS.get(obj_name, 0.0)
            self.model.body_pos[body_id] = far_pos
            self.model.body_quat[body_id] = np.asarray(
                self._default_place_pose_by_object[obj_name]["quat"],
                dtype=np.float64,
            )

    def _randomize_active_place_pose(self) -> None:
        active_place_info = self._get_active_place_info()
        body_id = int(active_place_info["body_id"])
        place_pos = np.asarray(
            self._default_place_pose_by_object[self.active_obj_name]["pos"],
            dtype=np.float64,
        ).copy()
        place_pos[0] = self.np_random.uniform(*self._target_x_range)
        place_pos[1] = self.np_random.uniform(*self._target_y_range)
        place_quat = self._yaw_to_quat(self._target_place_yaw_rad)

        self.model.body_pos[body_id] = place_pos
        self.model.body_quat[body_id] = place_quat

    def _sync_target_site_to_active_place(self) -> None:
        active_place_info = self._get_active_place_info()
        body_id = int(active_place_info["body_id"])
        site_id = int(active_place_info["site_id"])

        self.model.body_pos[self.target_body_id] = self.model.body_pos[body_id].copy()
        self.model.body_quat[self.target_body_id] = self.model.body_quat[body_id].copy()

        target_site_local_pos = self.model.site_pos[site_id].copy()
        target_site_local_pos[2] += self._target_height_above_place
        self.model.site_pos[self.target_site_id] = target_site_local_pos

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
                "PlaceAboveSiteEnv requires stable-baselines3 to load the grasping policy."
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
        return obj_pos, self._normalize_quat(obj_quat)

    def _get_grasp_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        grasp_env = self._grasp_env
        assert grasp_env is not None

        ee_pos = grasp_env.data.site(self.ee_site_name).xpos.copy()
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, grasp_env.data.site(self.ee_site_name).xmat)
        return ee_pos, self._normalize_quat(ee_quat)

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

        obj_pos, obj_quat = self._get_grasp_obj_pose()
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
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "lift_height": lift_height,
            "ee_obj_dist": ee_obj_dist,
            "object_speed": object_speed,
            "gripper_ctrl": grasp_env.data.ctrl[-2:].copy(),
            "terminated_like": bool(getattr(grasp_env, "success_counter", 0) > 0),
        }

    def _is_good_grasp_snapshot(self, snapshot: dict) -> bool:
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
        return (
            4.0 * float(snapshot["lift_height"])
            - 2.5 * float(snapshot["ee_obj_dist"])
            - 0.2 * float(snapshot["object_speed"])
            + 0.05 * is_closed
            + 0.02 * float(snapshot["terminated_like"])
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
                    return snapshot, "grasp_success", attempt

                if terminated or truncated:
                    break

        if best_snapshot is None or not self._allow_grasp_fallback_snapshot:
            raise RuntimeError(
                "Failed to obtain a grasped state from the grasping policy. "
                "Try increasing grasp_max_steps or grasp_attempts_per_reset."
            )

        return (
            best_snapshot,
            "grasp_fallback_best_snapshot",
            self._grasp_attempts_per_reset,
        )

    def _restore_grasp_snapshot(self, snapshot: dict) -> None:
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        ctrl = np.asarray(snapshot["ctrl"], dtype=np.float64).copy()
        if ctrl.shape != self.data.ctrl.shape:
            raise ValueError(
                "Transferred ctrl shape does not match PlaceAboveSiteEnv scene. "
                f"Expected {self.data.ctrl.shape}, got {ctrl.shape}."
            )

        self.active_obj_name = str(snapshot["active_object"])

        source_qpos = np.asarray(snapshot["qpos"], dtype=np.float64)
        source_qvel = np.asarray(snapshot["qvel"], dtype=np.float64)
        source_model = self._grasp_env.model
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
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

        self._restore_grasp_snapshot(snapshot)
        self._restore_default_place_poses()
        self._move_inactive_places_away()
        self._randomize_active_place_pose()
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)
        self.sampled_target_site_pos = self.data.site(self.target_site_name).xpos.copy()
        self.sampled_target_place_pos = self.data.site(
            str(self._get_active_place_info()["site_name"])
        ).xpos.copy()

        self.sampled_object_yaw = float(
            self._quat_to_yaw(np.asarray(snapshot["obj_quat"], dtype=np.float64))
        )
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        self._last_grasp_reset_attempts = int(attempt_count)
        self._last_grasp_init_lift_height = float(snapshot["lift_height"])
        self._last_grasp_init_ee_obj_dist = float(snapshot["ee_obj_dist"])
        self._last_grasp_reset_source = str(reset_source)

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

        return self._get_obs()

    def reset_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        seed: int | None = None,
        reset_source: str = "external_grasp_snapshot",
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
        action = self._coerce_policy_action(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.astype(np.float32)

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[: self._arm_ctrl_dim] += self._arm_action_scale * action
        if self._should_open_gripper():
            self._set_open_gripper_target(target_ctrl)
        else:
            self._set_closed_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self.do_simulation(target_ctrl, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_pos, ee_quat = self._get_ee_pose()

        ee_obj_pos_error, _ = self._get_pose_error(ee_pos, ee_quat, obj_pos, obj_quat)
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))

        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)

        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / 0.05))
        ) * self._reward_target_tanh_weight
        reward_orientation = -target_angle * self._reward_orientation_weight
        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )

        target_pose_aligned = bool(target_dist < self._success_distance)
        reward_target_bonus = self._reward_target_bonus if target_pose_aligned else 0.0

        if target_pose_aligned:
            self.success_counter += 1
        else:
            self.success_counter = 0

        reward = (
            reward_target
            + reward_target_tanh
            + reward_orientation
            + reward_target_bonus
            + control_penalty
        )

        reward_info = {
            "ee_object_dist": ee_obj_dist,
            "object_target_dist": target_dist,
            "object_target_angle_rad": target_angle,
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_orientation": float(reward_orientation),
            "reward_target_bonus": float(reward_target_bonus),
            "control_penalty": float(control_penalty),
        }

        return float(reward), reward_info

    def export_config(self) -> dict:
        config = export_env_config(self, self._get_obs_components())
        config["action"]["gripper_policy"] = "heuristic_release"
        config["action"]["gripper_open_distance_threshold"] = float(
            GRIPPER_OPEN_DISTANCE_THRESHOLD
        )
        config["action"]["gripper_open_angle_deg"] = float(GRIPPER_OPEN_ANGLE_DEG)
        config["action"]["gripper_open_target"] = self._gripper_open_target.astype(
            np.float64
        ).tolist()
        config["action"]["gripper_closed_target"] = self._gripper_closed_target.astype(
            np.float64
        ).tolist()
        config["task"]["target_site_name"] = self.target_site_name
        config["task"][
            "target_mode"
        ] = "object_position_above_randomized_xml_target_place"
        config["task"]["target_visual"] = "xml_target_site_above_place"
        config["task"]["target_height_above_place"] = float(
            self._target_height_above_place
        )
        config["task"]["target_place_yaw_deg"] = float(TARGET_PLACE_YAW_DEG)
        config["task"]["target_place_site_names"] = {
            obj_name: str(info["site_name"])
            for obj_name, info in self.place_info.items()
        }
        config["task"]["target_place_randomization"] = {
            "target_x_range": list(self._target_x_range),
            "target_y_range": list(self._target_y_range),
        }
        config["task"]["success_criterion"] = {
            "object_target_distance_only": float(self._success_distance)
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
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        target_place_pos = None
        target_place_quat = None
        active_place_site_name = str(self._get_active_place_info()["site_name"])
        target_place_pos, target_place_quat = self._get_site_pose(
            active_place_site_name
        )
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        state = {
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
            "obj_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "obj_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "sampled_target_site_pos": self.sampled_target_site_pos.copy(),
            "sampled_target_place_pos": self.sampled_target_place_pos.copy(),
            "gripper_policy": "heuristic_release",
            "gripper_state": self.gripper_state,
            "gripper_should_open": bool(self._should_open_gripper()),
            "success_counter": int(self.success_counter),
            "initial_object_target_dist": float(self.initial_object_target_dist),
            "best_object_target_dist": float(self.best_object_target_dist),
            "last_action": self.last_action.copy(),
            "grasp_reset_attempts": int(self._last_grasp_reset_attempts),
            "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
            "grasp_reset_source": self._last_grasp_reset_source,
            "task_mode": "object_above_randomized_xml_target_place",
        }
        if target_place_pos is not None and target_place_quat is not None:
            state["target_place_pos"] = target_place_pos
            state["target_place_quat"] = target_place_quat
            state["target_place_yaw"] = float(self._quat_to_yaw(target_place_quat))
            state["target_place_yaw_deg"] = float(TARGET_PLACE_YAW_DEG)
            state["target_height_above_place"] = float(self._target_height_above_place)
            state["gripper_open_distance_threshold"] = float(
                GRIPPER_OPEN_DISTANCE_THRESHOLD
            )
            state["gripper_open_angle_deg"] = float(GRIPPER_OPEN_ANGLE_DEG)
        return state

    def close(self):
        if self._grasp_env is not None:
            self._grasp_env.close()
            self._grasp_env = None
            self._grasp_policy = None
        return super().close()
