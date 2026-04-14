from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config_export import capture_init_config, export_env_config
from .grasping_env import GraspingEnv
from .grasping_env_v1 import GraspingEnvV1
from .grasping_env_v2 import GraspingEnvV2

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
    "GraspingEnvV1": GraspingEnvV1,
    "GraspingEnvV2": GraspingEnvV2,
}
if GraspingEnvV3 is not None:
    GRASP_ENV_REGISTRY["GraspingEnvV3"] = GraspingEnvV3


class PlaceTargetEnv(MujocoEnv, utils.EzPickle):
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
        reward_target_weight: float = 7.0,
        reward_target_tanh_weight: float = 3.0,
        reward_target_bonus: float = 10.0,
        reward_stay_bonus: float = 16.0,
        control_penalty_weight: float = 0.001,
        success_distance: float = 0.015,
        success_steps_required: int = 10,
        terminate_ee_obj_distance: float = 0.08,
        max_episode_steps: int = 100,
        arm_action_scale: float = 0.01,
        target_x_range: tuple[float, float] = (0.15, 0.27),
        target_y_range: tuple[float, float] = (-0.10, 0.10),
        target_place_z: float = 0.001,
        target_z_range: tuple[float, float] | None = None,
        target_place_yaw_range: tuple[float, float] = (-np.pi, np.pi),
        target_height_above_place: float = 0.1,
        ee_site_name: str = "attachment_site",
        target_site_name: str = "target",
        target_place_body_name: str = "target_place_body",
        grasp_model_path: str | None = None,
        grasp_env_name: str = "GraspingEnvV2",
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
            reward_target_bonus,
            reward_stay_bonus,
            control_penalty_weight,
            success_distance,
            success_steps_required,
            terminate_ee_obj_distance,
            max_episode_steps,
            arm_action_scale,
            target_x_range,
            target_y_range,
            target_place_z,
            target_z_range,
            target_place_yaw_range,
            target_height_above_place,
            ee_site_name,
            target_site_name,
            target_place_body_name,
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
                "PlaceTargetEnv requires `grasp_model_path` so reset can start from the grasping policy state."
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
        self._reward_target_bonus = float(reward_target_bonus)
        self._reward_stay_bonus = float(reward_stay_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._success_distance = float(success_distance)
        self._success_steps_required = int(success_steps_required)
        self._terminate_ee_obj_distance = float(terminate_ee_obj_distance)
        if self._terminate_ee_obj_distance <= 0.0:
            raise ValueError("terminate_ee_obj_distance must be greater than 0.")
        self.max_episode_steps = int(max_episode_steps)
        self._arm_action_scale = float(arm_action_scale)
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
        self._target_height_above_place = float(target_height_above_place)
        self.ee_site_name = ee_site_name
        self.target_site_name = target_site_name
        self.target_place_body_name = target_place_body_name
        if self._target_z_range[0] > self._target_z_range[1]:
            raise ValueError("target_z_range must be ordered as (min_z, max_z).")
        if self._target_place_yaw_range[0] > self._target_place_yaw_range[1]:
            raise ValueError(
                "target_place_yaw_range must be ordered as (min_yaw, max_yaw)."
            )

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
        self.place_joint_name_by_object = {
            "box": "cube_place_joint",
            "triangle": "tri_place_joint",
            "cylinder": "cyl_place_joint",
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
            joint_name = self.place_joint_name_by_object[obj_name]
            site_name = self.place_site_name_by_object[obj_name]
            geom_name = self.place_geom_name_by_object[obj_name]

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

            self.place_info[obj_name] = {
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

        self.place_geom_rgba = {
            obj_name: self.model.geom_rgba[int(info["geom_id"])].copy()
            for obj_name, info in self.place_info.items()
        }

        self.active_obj_name = self.object_names[0]
        self.active_place_name = self.place_name_by_object[self.active_obj_name]

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
                "PlaceTargetEnv expects arm actuators plus 2 gripper actuators."
            )
        self._arm_ctrl_dim = int(self.model.nu - 2)
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self._arm_ctrl_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self.initial_obj_site_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_pos = np.zeros(3, dtype=np.float64)
        self.sampled_target_place_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.sampled_target_place_yaw = 0.0
        self.applied_target_place_yaw = 0.0

        self._disable_grasp_constraints()

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
        quat = PlaceTargetEnv._normalize_quat(quat)
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

    def _sample_target_place_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = self.np_random.uniform(*self._target_x_range)
        y = self.np_random.uniform(*self._target_y_range)
        z = self.np_random.uniform(*self._target_z_range)
        yaw = self.np_random.uniform(*self._target_place_yaw_range)
        return (
            np.array([x, y, z], dtype=np.float64),
            self._yaw_to_quat(yaw),
            float(yaw),
        )

    def _set_active_place_visual(self) -> None:
        for obj_name, info in self.place_info.items():
            rgba = self.place_geom_rgba[obj_name].copy()
            rgba[3] = (
                self.place_geom_rgba[obj_name][3]
                if obj_name == self.active_obj_name
                else 0.0
            )
            self.model.geom_rgba[int(info["geom_id"])] = rgba

    def _disable_grasp_constraints(self) -> None:
        return None

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

    def _set_place_joints_in_state(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        active_place_pos: np.ndarray,
        active_place_quat: np.ndarray,
    ) -> None:
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.active_place_name = self.place_name_by_object[self.active_obj_name]

        for index, obj_name in enumerate(self.object_names):
            info = self.place_info[obj_name]
            qposadr = int(info["qposadr"])
            dofadr = int(info["dofadr"])

            if obj_name == self.active_obj_name:
                qpos[qposadr : qposadr + 3] = active_place_pos
                qpos[qposadr + 3 : qposadr + 7] = active_place_quat
            else:
                qpos[qposadr : qposadr + 3] = np.array(
                    [2.0 + index, 2.0, 0.2],
                    dtype=np.float64,
                )
                qpos[qposadr + 3 : qposadr + 7] = identity_quat
            qvel[dofadr : dofadr + 6] = 0.0

    def _ensure_grasp_policy_loaded(self) -> None:
        if self._grasp_env is not None and self._grasp_policy is not None:
            return

        try:
            from stable_baselines3 import SAC
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PlaceTargetEnv requires stable-baselines3 to load the grasping policy."
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

        ee_pos = grasp_env.data.site("attachment_site").xpos.copy()
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, grasp_env.data.site("attachment_site").xmat)
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
                "Transferred ctrl shape does not match PlaceTargetEnv scene. "
                f"Expected {self.data.ctrl.shape}, got {ctrl.shape}."
            )

        self.active_obj_name = str(snapshot["active_object"])
        self._set_place_joints_in_state(
            qpos,
            qvel,
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._set_active_place_visual()

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
        self._disable_grasp_constraints()
        mujoco.mj_forward(self.model, self.data)

        if self._grasp_transfer_settle_steps > 0:
            settle_ctrl = self.data.ctrl.copy()
            settle_ctrl[-2:] = self._gripper_closed_target
            settle_ctrl = np.clip(settle_ctrl, self._ctrl_low, self._ctrl_high)
            for _ in range(self._grasp_transfer_settle_steps):
                self.do_simulation(settle_ctrl, 1)

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float64).copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.astype(np.float32)

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[: self._arm_ctrl_dim] += self._arm_action_scale * action
        self._set_closed_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        self._disable_grasp_constraints()

        self.do_simulation(target_ctrl, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated_success = self.success_counter >= self._success_steps_required
        terminated_ee_obj_far = (
            float(reward_info["ee_object_dist"]) >= self._terminate_ee_obj_distance
        )
        terminated = terminated_success or terminated_ee_obj_far
        truncated = self.current_step >= self.max_episode_steps
        reward_info["terminate_ee_obj_distance"] = float(
            self._terminate_ee_obj_distance
        )
        reward_info["terminated_success"] = bool(terminated_success)
        reward_info["terminated_ee_obj_far"] = bool(terminated_ee_obj_far)
        reward_info["termination_reason"] = (
            "success"
            if terminated_success
            else "ee_obj_too_far" if terminated_ee_obj_far else None
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(self, action: np.ndarray) -> tuple[float, dict]:
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        ee_pos, ee_quat = self._get_ee_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))
        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))

        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / 0.05))
        ) * self._reward_target_tanh_weight
        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )

        reward_target_bonus = (
            self._reward_target_bonus if target_dist < self._success_distance else 0.0
        )

        success_now = target_dist < self._success_distance
        if success_now:
            self.success_counter += 1
            stay_bonus = self._reward_stay_bonus
        else:
            self.success_counter = 0
            stay_bonus = 0.0

        reward = (
            reward_target
            + reward_target_tanh
            + reward_target_bonus
            + stay_bonus
            + control_penalty
        )

        reward_info = {
            "active_object": self.active_obj_name,
            "ee_object_dist": ee_obj_dist,
            "ee_object_rot_error": ee_obj_angle,
            "object_target_dist": target_dist,
            "object_target_rot_error": target_angle,
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_target_bonus": float(reward_target_bonus),
            "stay_bonus": float(stay_bonus),
            "control_penalty": float(control_penalty),
            # "grasp_reset_attempts": float(self._last_grasp_reset_attempts),
            # "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            # "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
        }

        return float(reward), reward_info

    def reset_model(self):
        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        snapshot, reset_source, attempt_count = self._sample_grasp_reset_snapshot()
        self.active_obj_name = str(snapshot["active_object"])
        (
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
        ) = self._sample_target_place_pose()
        self._restore_grasp_snapshot(snapshot)

        self.initial_obj_site_pos = np.asarray(
            snapshot["obj_pos"], dtype=np.float64
        ).copy()
        self.sampled_object_yaw = float(
            self._quat_to_yaw(np.asarray(snapshot["obj_quat"], dtype=np.float64))
        )
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )
        target_place_body_quat = self._normalize_quat(
            self.data.body(str(self._get_active_place_info()["body_name"])).xquat.copy()
        )
        self.applied_target_place_yaw = float(self._quat_to_yaw(target_place_body_quat))
        self._last_grasp_reset_attempts = int(attempt_count)
        self._last_grasp_init_lift_height = float(snapshot["lift_height"])
        self._last_grasp_init_ee_obj_dist = float(snapshot["ee_obj_dist"])
        self._last_grasp_reset_source = str(reset_source)

        return self._get_obs()

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
        qpos = self.data.qpos
        qvel = self.data.qvel

        first_place_qposadr = min(
            int(info["qposadr"]) for info in self.place_info.values()
        )
        first_place_dofadr = min(
            int(info["dofadr"]) for info in self.place_info.values()
        )
        first_obj_qposadr = min(
            int(info["qposadr"]) for info in self.object_info.values()
        )
        first_obj_dofadr = min(
            int(info["dofadr"]) for info in self.object_info.values()
        )

        robot_qpos = qpos[:first_place_qposadr]
        robot_qvel = qvel[:first_place_dofadr]

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        target_place_pos = self.data.body(
            str(self._get_active_place_info()["body_name"])
        ).xpos.copy()
        target_place_quat = self._normalize_quat(
            self.data.body(str(self._get_active_place_info()["body_name"])).xquat.copy()
        )

        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos,
            obj_quat,
            target_pos,
            target_quat,
        )

        target_xy_dist = float(np.linalg.norm(obj_target_pos_error[:2]))
        target_z_error = float(obj_target_pos_error[2])
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))

        return [
            ("robot_qpos", robot_qpos),
            ("robot_qvel", robot_qvel),
            ("object_type", self.object_one_hot[self.active_obj_name]),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("target_place_pos", target_place_pos),
            ("target_place_quat", target_place_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("object_target_pos_error", obj_target_pos_error),
            ("object_target_rot_error", obj_target_rot_error),
            ("object_target_xy_dist", np.array([target_xy_dist], dtype=np.float64)),
            ("object_target_z_error", np.array([target_z_error], dtype=np.float64)),
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
        config["action"]["gripper_policy"] = "fixed_closed_contact"
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
        return config

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
        target_place_body_quat = self._normalize_quat(
            self.data.body(str(self._get_active_place_info()["body_name"])).xquat.copy()
        )
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        return {
            "active_object": self.active_obj_name,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "obj_pos": obj_pos,
            "obj_quat": obj_quat,
            "target_pos": target_pos,
            "target_quat": target_quat,
            "target_place_pos": self.data.body(
                str(self._get_active_place_info()["body_name"])
            ).xpos.copy(),
            "target_place_quat": target_place_body_quat,
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "obj_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "target_height_above_place": float(self._target_height_above_place),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "target_place_yaw": float(self._quat_to_yaw(target_place_body_quat)),
            "sampled_target_place_yaw": float(self.sampled_target_place_yaw),
            "applied_target_place_yaw": float(self.applied_target_place_yaw),
            "gripper_assist_mix": 0.0,
            "gripper_should_close": True,
            "gripper_state": self.gripper_state,
            "success_counter": int(self.success_counter),
            "last_action": self.last_action.copy(),
            "grasp_reset_attempts": int(self._last_grasp_reset_attempts),
            "grasp_init_lift_height": float(self._last_grasp_init_lift_height),
            "grasp_init_ee_obj_dist": float(self._last_grasp_init_ee_obj_dist),
            "grasp_reset_source": self._last_grasp_reset_source,
            "terminate_ee_obj_distance": float(self._terminate_ee_obj_distance),
            "ee_obj_too_far": bool(
                np.linalg.norm(ee_obj_pos_error) >= self._terminate_ee_obj_distance
            ),
        }

    def close(self):
        if self._grasp_env is not None:
            self._grasp_env.close()
            self._grasp_env = None
            self._grasp_policy = None
        return super().close()

    def render(self):
        return super().render()
