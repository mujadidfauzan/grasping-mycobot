from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from .config_export import capture_init_config, export_env_config

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}
DEFAULT_XML_PATH = Path(__file__).resolve().parents[1] / "robot" / "object_lift.xml"


class ReachingEnv(MujocoEnv, utils.EzPickle):
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
        reward_dist_weight: float = 3.0,
        reward_dist_tanh_weight: float = 1.5,
        reward_orient_weight: float = 1.0,
        reward_orient_tanh_weight: float = 0.8,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 3.0,
        reward_target_orient_weight: float = 1.0,
        reward_grasp_bonus: float = 4.0,
        reward_target_bonus: float = 8.0,
        reward_stay_bonus: float = 12.0,
        control_penalty_weight: float = 0.001,
        success_distance: float = 0.02,
        success_angle_deg: float = 25.0,
        success_steps_required: int = 10,
        max_episode_steps: int = 300,
        arm_action_scale: float = 0.01,
        object_x_range: tuple[float, float] = (0.15, 0.27),
        object_y_range: tuple[float, float] = (-0.10, 0.10),
        object_z: float = 0.025,
        object_yaw_limit_rad: float = 1.05,
        lift_height: float = 0.10,
        grasp_close_distance: float = 0.035,
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
            reward_dist_weight,
            reward_dist_tanh_weight,
            reward_orient_weight,
            reward_orient_tanh_weight,
            reward_target_weight,
            reward_target_tanh_weight,
            reward_target_orient_weight,
            reward_grasp_bonus,
            reward_target_bonus,
            reward_stay_bonus,
            control_penalty_weight,
            success_distance,
            success_angle_deg,
            success_steps_required,
            max_episode_steps,
            arm_action_scale,
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

        self._reward_dist_weight = float(reward_dist_weight)
        self._reward_dist_tanh_weight = float(reward_dist_tanh_weight)
        self._reward_orient_weight = float(reward_orient_weight)
        self._reward_orient_tanh_weight = float(reward_orient_tanh_weight)
        self._reward_target_weight = float(reward_target_weight)
        self._reward_target_tanh_weight = float(reward_target_tanh_weight)
        self._reward_target_orient_weight = float(reward_target_orient_weight)
        self._reward_grasp_bonus = float(reward_grasp_bonus)
        self._reward_target_bonus = float(reward_target_bonus)
        self._reward_stay_bonus = float(reward_stay_bonus)
        self._control_penalty_weight = float(control_penalty_weight)
        self._success_distance = float(success_distance)
        self._success_angle_rad = np.deg2rad(float(success_angle_deg))
        self._success_steps_required = int(success_steps_required)
        self.max_episode_steps = int(max_episode_steps)
        self._arm_action_scale = float(arm_action_scale)
        self._object_x_range = tuple(float(value) for value in object_x_range)
        self._object_y_range = tuple(float(value) for value in object_y_range)
        self._object_z = float(object_z)
        self._object_yaw_limit_rad = float(object_yaw_limit_rad)
        self._lift_height = float(lift_height)
        self._grasp_close_distance = float(grasp_close_distance)
        self._grasp_release_distance = float(grasp_release_distance)
        self._grasp_close_angle_rad = np.deg2rad(float(grasp_close_angle_deg))
        self.ee_site_name = ee_site_name
        self.target_site_name = target_site_name

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
        for obj_name in self.object_names:
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

        self.active_obj_name = self.object_names[0]
        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_site_name
        )

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
                "ReachingEnv expects arm actuators plus 2 gripper actuators."
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
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        half_yaw = float(yaw) / 2.0
        return np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_to_yaw(quat: np.ndarray) -> float:
        quat = ReachingEnv._normalize_quat(quat)
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

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

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(str(self._get_active_obj_info()["site_name"]))

    def _set_open_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "open"
        ctrl[-2] = 0.01
        ctrl[-1] = -0.01

    def _set_closed_gripper_target(self, ctrl: np.ndarray) -> None:
        self.gripper_state = "closed"
        ctrl[-2] = -0.02
        ctrl[-1] = 0.02

    def _sample_object_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = self.np_random.uniform(*self._object_x_range)
        y = self.np_random.uniform(*self._object_y_range)
        z = self._object_z
        yaw = self.np_random.uniform(
            -self._object_yaw_limit_rad, self._object_yaw_limit_rad
        )
        pos = np.array([x, y, z], dtype=np.float64)
        quat = self._yaw_to_quat(yaw)
        return pos, quat, float(yaw)

    def _update_target_site(self) -> None:
        target_pos = self.initial_obj_site_pos + np.array(
            [0.0, 0.0, self._lift_height], dtype=np.float64
        )
        self.model.site_pos[self.target_site_id] = target_pos

    def _apply_grasp_heuristic(self, ctrl: np.ndarray) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
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

    def step(self, action):
        self.current_step += 1
        action = np.asarray(action, dtype=np.float64).copy()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.astype(np.float32)

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[: self._arm_ctrl_dim] += self._arm_action_scale * action
        self._apply_grasp_heuristic(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self.do_simulation(target_ctrl, self.frame_skip)

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
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        target_dist = float(np.linalg.norm(obj_target_pos_error))
        target_angle = float(np.linalg.norm(obj_target_rot_error))

        reward_dist = -ee_obj_dist * self._reward_dist_weight
        reward_dist_tanh = (
            1.0 - float(np.tanh(ee_obj_dist / 0.05))
        ) * self._reward_dist_tanh_weight
        reward_orient = -ee_obj_angle * self._reward_orient_weight
        reward_orient_tanh = (
            1.0 - float(np.tanh(ee_obj_angle / 0.5))
        ) * self._reward_orient_tanh_weight
        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / 0.05))
        ) * self._reward_target_tanh_weight
        reward_target_orient = -target_angle * self._reward_target_orient_weight
        reward_grasp = self._reward_grasp_bonus if self.grasp_latched else 0.0
        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )

        reward_target_bonus = (
            self._reward_target_bonus if target_dist < self._success_distance else 0.0
        )

        success_now = (
            target_dist < self._success_distance
            and target_angle < self._success_angle_rad
        )
        if success_now:
            self.success_counter += 1
            stay_bonus = self._reward_stay_bonus
        else:
            self.success_counter = 0
            stay_bonus = 0.0

        reward = (
            reward_dist
            + reward_dist_tanh
            + reward_orient
            + reward_orient_tanh
            + reward_target
            + reward_target_tanh
            + reward_target_orient
            + reward_grasp
            + control_penalty
            + reward_target_bonus
            + stay_bonus
        )

        reward_info = {
            "active_object": self.active_obj_name,
            "ee_object_dist": ee_obj_dist,
            "ee_object_rot_error": ee_obj_angle,
            "object_target_dist": target_dist,
            "object_target_rot_error": float(np.linalg.norm(obj_target_rot_error)),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "reward_orient": float(reward_orient),
            "reward_orient_tanh": float(reward_orient_tanh),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_target_orient": float(reward_target_orient),
            "reward_grasp": float(reward_grasp),
            "control_penalty": float(control_penalty),
            "reward_target_bonus": float(reward_target_bonus),
            "stay_bonus": float(stay_bonus),
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
        ctrl[: self._arm_ctrl_dim] = qpos[: self._arm_ctrl_dim]
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

        return self._get_obs()

    def _get_obs_components(self) -> list[tuple[str, np.ndarray]]:
        qpos = self.data.qpos
        qvel = self.data.qvel

        first_obj_qposadr = min(
            int(info["qposadr"]) for info in self.object_info.values()
        )
        first_obj_dofadr = min(
            int(info["dofadr"]) for info in self.object_info.values()
        )

        robot_qpos = qpos[:first_obj_qposadr]
        robot_qvel = qvel[:first_obj_dofadr]
        gripper_state = robot_qpos[-2:]

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, _ = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        lift_height = float(obj_pos[2] - self.initial_obj_site_pos[2])

        return [
            ("robot_qpos", robot_qpos),
            ("robot_qvel", robot_qvel),
            ("gripper_state", gripper_state),
            ("ee_pos", ee_pos),
            ("ee_quat", ee_quat),
            ("object_pos", obj_pos),
            ("object_quat", obj_quat),
            ("target_pos", target_pos),
            ("target_quat", target_quat),
            ("ee_object_pos_error", ee_obj_pos_error),
            ("ee_object_rot_error", ee_obj_rot_error),
            ("object_target_pos_error", obj_target_pos_error),
            ("ee_object_dist", np.array([np.linalg.norm(ee_obj_pos_error)])),
            ("ee_object_rot_error_norm", np.array([np.linalg.norm(ee_obj_rot_error)])),
            ("object_target_dist", np.array([np.linalg.norm(obj_target_pos_error)])),
            ("lift_height", np.array([lift_height])),
            ("grasp_latched", np.array([float(self.grasp_latched)])),
        ]

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([
            np.asarray(component, dtype=np.float64)
            for _, component in self._get_obs_components()
        ])
        return obs.astype(np.float32)

    def export_config(self) -> dict:
        return export_env_config(self, self._get_obs_components())

    def get_debug_state(self) -> dict:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()
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
            "ee_obj_pos_error": ee_obj_pos_error,
            "ee_obj_rot_error": ee_obj_rot_error,
            "ee_obj_dist": float(np.linalg.norm(ee_obj_pos_error)),
            "ee_obj_angle_rad": float(np.linalg.norm(ee_obj_rot_error)),
            "obj_target_pos_error": obj_target_pos_error,
            "obj_target_rot_error": obj_target_rot_error,
            "obj_target_dist": float(np.linalg.norm(obj_target_pos_error)),
            "obj_target_angle_rad": float(np.linalg.norm(obj_target_rot_error)),
            "lift_height": float(obj_pos[2] - self.initial_obj_site_pos[2]),
            "required_lift_height": float(self._lift_height),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "gripper_assist_mix": 1.0,
            "gripper_should_close": bool(self.last_grasp_should_close),
            "grasp_latched": bool(self.grasp_latched),
            "gripper_state": self.gripper_state,
            "success_counter": int(self.success_counter),
            "last_action": self.last_action.copy(),
        }

    def render(self):
        return super().render()
