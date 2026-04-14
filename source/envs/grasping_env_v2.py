import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class GraspingEnvV2(MujocoEnv, utils.EzPickle):

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
        xml_file: str = "object_lift.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_dist_weight: float = 2.0,
        reward_dist_target_weight: float = 1.0,
        reward_pose_orient_weight: float = 0.35,
        reward_target_orient_weight: float = 0.5,
        object_yaw_limit_deg: float = 45.0,
        success_steps_required: int = 10,
        target_height: float = 0.1,
        arm_action_scale: float = 0.01,
        gripper_action_scale: float = 0.003,
        gripper_assist_steps: int = 0,
        gripper_assist_close_dist: float = 0.035,
        gripper_assist_close_angle_deg: float = 35.0,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_target_weight,
            reward_pose_orient_weight,
            reward_target_orient_weight,
            object_yaw_limit_deg,
            success_steps_required,
            target_height,
            arm_action_scale,
            gripper_action_scale,
            gripper_assist_steps,
            gripper_assist_close_dist,
            gripper_assist_close_angle_deg,
            **kwargs,
        )

        self._reward_dist_weight = float(reward_dist_weight)
        self._reward_dist_target_weight = float(reward_dist_target_weight)
        self._reward_pose_orient_weight = float(reward_pose_orient_weight)
        self._reward_target_orient_weight = float(reward_target_orient_weight)
        self._object_yaw_limit_rad = np.deg2rad(float(object_yaw_limit_deg))
        self._success_steps_required = int(success_steps_required)
        self._target_height = float(target_height)
        self._arm_action_scale = float(arm_action_scale)
        self._gripper_action_scale = float(gripper_action_scale)
        self._gripper_assist_steps = int(gripper_assist_steps)
        self._gripper_assist_close_dist = float(gripper_assist_close_dist)
        self._gripper_assist_close_angle_rad = np.deg2rad(
            float(gripper_assist_close_angle_deg)
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

        self.ee_site_name = "attachment_site"
        self.target_site_name = "target"
        self.object_names = ["box", "cylinder", "triangle"]
        self.object_info = {}

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

        self.gripper_state = "open"
        self.max_episode_steps = 500
        self.current_step = 0
        self.success_counter = 0
        self.sampled_object_yaw = 0.0
        self.applied_object_yaw = 0.0
        self.training_num_timesteps = 0
        self.last_gripper_assist_mix = 0.0
        self.last_gripper_should_close = False

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

    def _yaw_to_quat(self, yaw: float) -> np.ndarray:
        half = float(yaw) / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _quat_to_yaw(self, quat: np.ndarray) -> float:
        quat = self._normalize_quat(quat)
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

    def _rotation_angle(self, source_quat: np.ndarray, target_quat: np.ndarray) -> float:
        return float(np.linalg.norm(self._rotation_vector(source_quat, target_quat)))

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

    def _get_active_obj_info(self) -> dict:
        return self.object_info[self.active_obj_name]

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.ee_site_name)

    def _get_target_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self.target_site_name)

    def _get_active_obj_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self._get_site_pose(self._get_active_obj_info()["site_name"])

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
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "gripper_assist_mix": float(self.last_gripper_assist_mix),
            "gripper_should_close": bool(self.last_gripper_should_close),
        }

    def gripper_ctrl(self, close: bool, target: np.ndarray) -> None:
        if close:
            self.gripper_state = "closed"
            target[-2] = -0.02
            target[-1] = 0.02
        else:
            self.gripper_state = "open"
            target[-2] = 0.01
            target[-1] = -0.01

    def set_training_num_timesteps(self, num_timesteps: int) -> None:
        self.training_num_timesteps = max(0, int(num_timesteps))

    def _get_gripper_assist_mix(self) -> float:
        if self._gripper_assist_steps <= 0:
            return 0.0
        progress = np.clip(
            float(self.training_num_timesteps) / float(self._gripper_assist_steps),
            0.0,
            1.0,
        )
        return float(1.0 - progress)

    def _get_policy_gripper_target(
        self, current_target: np.ndarray, action: np.ndarray
    ) -> np.ndarray:
        policy_target = current_target.copy()
        policy_target[-2:] += self._gripper_action_scale * action[-2:]
        return policy_target

    def _get_assist_gripper_target(
        self, current_target: np.ndarray
    ) -> tuple[np.ndarray, float, float, bool]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )

        ee_obj_dist = float(np.linalg.norm(ee_obj_pos_error))
        ee_obj_angle = float(np.linalg.norm(ee_obj_rot_error))
        should_close = (
            ee_obj_dist < self._gripper_assist_close_dist
            and ee_obj_angle < self._gripper_assist_close_angle_rad
        )

        assist_target = current_target.copy()
        self.gripper_ctrl(close=should_close, target=assist_target)
        return assist_target, ee_obj_dist, ee_obj_angle, should_close

    def _update_gripper_state_from_target(self, target: np.ndarray) -> None:
        self.gripper_state = "closed" if target[-2] < target[-1] else "open"

    def step(self, action):
        self.current_step += 1
        action = action.copy()

        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()
        target[:-2] += self._arm_action_scale * action[:-2]

        policy_target = self._get_policy_gripper_target(target, action)
        assist_target, assist_dist, assist_angle, should_close = (
            self._get_assist_gripper_target(target)
        )
        assist_mix = self._get_gripper_assist_mix()
        target[-2:] = (
            (1.0 - assist_mix) * policy_target[-2:]
            + assist_mix * assist_target[-2:]
        )
        self.last_gripper_assist_mix = assist_mix
        self.last_gripper_should_close = should_close

        target = np.clip(target, self.action_space.low, self.action_space.high)
        self._update_gripper_state_from_target(target)
        self.do_simulation(target, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(
            action=action,
            assist_dist=assist_dist,
            assist_angle=assist_angle,
            assist_mix=assist_mix,
            should_close=should_close,
        )
        terminated = self.success_counter >= self._success_steps_required
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, reward_info

    def _get_rew(
        self,
        action: np.ndarray,
        assist_dist: float,
        assist_angle: float,
        assist_mix: float,
        should_close: bool,
    ) -> tuple[float, dict]:
        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        dist = np.linalg.norm(ee_obj_pos_error)
        ee_obj_angle = np.linalg.norm(ee_obj_rot_error)
        target_dist = np.linalg.norm(obj_target_pos_error)
        target_angle = np.linalg.norm(obj_target_rot_error)

        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))
        reward_pose_orient = -ee_obj_angle * self._reward_pose_orient_weight

        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))
        reward_target_orient = -target_angle * self._reward_target_orient_weight
        reward_orient = reward_pose_orient + reward_target_orient

        reward_dist_bonus = (
            8.0 if dist < 0.01 and ee_obj_angle < np.deg2rad(12.0) else 0.0
        )
        reward_target_bonus = (
            15.0 if target_dist < 0.01 and target_angle < np.deg2rad(10.0) else 0.0
        )
        control_penalty = -0.001 * np.sum(np.square(action))

        active_info = self._get_active_obj_info()
        obj_twist = self.data.qvel[active_info["dofadr"] : active_info["dofadr"] + 6]
        obj_speed = float(np.linalg.norm(obj_twist[:3]))
        obj_ang_speed = float(np.linalg.norm(obj_twist[3:6]))

        stay_bonus = 0.0
        if (
            target_dist < 0.02
            and target_angle < np.deg2rad(10.0)
            and obj_speed < 0.01
            and obj_ang_speed < 0.1
        ):
            self.success_counter += 1
            stay_bonus = 25.0
        else:
            self.success_counter = 0

        reward = (
            reward_dist
            + reward_dist_tanh
            + reward_pose_orient
            + reward_target
            + reward_target_tanh
            + reward_target_orient
            + control_penalty
            + stay_bonus
            + reward_dist_bonus
            + reward_target_bonus
        )

        reward_info = {
            "active_object": self.active_obj_name,
            "dist": float(dist),
            "ee_object_dist": float(dist),
            "ee_object_rot_error": float(ee_obj_angle),
            "object_target_dist": float(target_dist),
            "object_target_rot_error": float(target_angle),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "reward_pose_orient": float(reward_pose_orient),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_target_orient": float(reward_target_orient),
            "reward_orient": float(reward_orient),
            "control_penalty": float(control_penalty),
            "stay_bonus": float(stay_bonus),
            "reward_dist_bonus": float(reward_dist_bonus),
            "reward_target_bonus": float(reward_target_bonus),
            "object_yaw": float(self._quat_to_yaw(obj_quat)),
            "sampled_object_yaw": float(self.sampled_object_yaw),
            "applied_object_yaw": float(self.applied_object_yaw),
            "gripper_assist_mix": float(assist_mix),
            "gripper_assist_steps": float(self._gripper_assist_steps),
            "gripper_assist_dist": float(assist_dist),
            "gripper_assist_rot_error": float(assist_angle),
            "gripper_should_close": float(should_close),
            "gripper_ctrl_left": float(self.data.ctrl[-2]),
            "gripper_ctrl_right": float(self.data.ctrl[-1]),
        }

        return float(reward), reward_info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.active_obj_name = self.np_random.choice(self.object_names)
        x = self.np_random.uniform(0.15, 0.27)
        y = self.np_random.uniform(-0.10, 0.10)
        z = 0.025

        yaw = self.np_random.uniform(
            -self._object_yaw_limit_rad, self._object_yaw_limit_rad
        )
        quat = self._yaw_to_quat(yaw)
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for obj_name in self.object_names:
            info = self.object_info[obj_name]
            adr = info["qposadr"]
            dadr = info["dofadr"]

            if obj_name == self.active_obj_name:
                qpos[adr + 0] = x
                qpos[adr + 1] = y
                qpos[adr + 2] = z
                qpos[adr + 3 : adr + 7] = quat
            else:
                qpos[adr + 0] = 6.0
                qpos[adr + 1] = 1.0
                qpos[adr + 2] = 1.0
                qpos[adr + 3 : adr + 7] = identity_quat

            qvel[dadr : dadr + 6] = 0.0

        self.model.site_pos[self.target_site_id] = np.array(
            [x, y, self._target_height], dtype=np.float64
        )

        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0
        self.success_counter = 0
        self.sampled_object_yaw = float(yaw)
        self.applied_object_yaw = float(
            self._quat_to_yaw(self._get_active_obj_pose()[1])
        )

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel

        first_obj_qposadr = min(info["qposadr"] for info in self.object_info.values())
        first_obj_dofadr = min(info["dofadr"] for info in self.object_info.values())

        robot_qpos = qpos[:first_obj_qposadr]
        robot_qvel = qvel[:first_obj_dofadr]
        gripper_state = robot_qpos[-2:]

        ee_pos, ee_quat = self._get_ee_pose()
        obj_pos, obj_quat = self._get_active_obj_pose()
        target_pos, target_quat = self._get_target_pose()

        ee_obj_pos_error, ee_obj_rot_error = self._get_pose_error(
            ee_pos, ee_quat, obj_pos, obj_quat
        )
        obj_target_pos_error, obj_target_rot_error = self._get_pose_error(
            obj_pos, obj_quat, target_pos, target_quat
        )

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                gripper_state,
                ee_pos,
                ee_quat,
                obj_pos,
                obj_quat,
                target_pos,
                target_quat,
                ee_obj_pos_error,
                ee_obj_rot_error,
                obj_target_pos_error,
                obj_target_rot_error,
                np.array(
                    [
                        np.linalg.norm(ee_obj_pos_error),
                        np.linalg.norm(ee_obj_rot_error),
                        np.linalg.norm(obj_target_pos_error),
                        np.linalg.norm(obj_target_rot_error),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

        return obs.astype(np.float32)
