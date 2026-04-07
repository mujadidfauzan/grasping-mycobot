import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class GraspingEnvV3(MujocoEnv, utils.EzPickle):

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
        reward_dist_weight: float = 2,
        reward_dist_target_weight: float = 1,
        random_yaw: bool = True,
        yaw_range: tuple[float, float] = (-np.pi, np.pi),
        close_radius_xy: float = 0.02,
        close_radius_z: float = 0.025,
        reward_upright_weight: float = 0.5,
        debug_randomization: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_target_weight,
            random_yaw,
            yaw_range,
            close_radius_xy,
            close_radius_z,
            reward_upright_weight,
            debug_randomization,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_dist_target_weight = reward_dist_target_weight
        self._random_yaw = random_yaw
        self._yaw_range = tuple(float(v) for v in yaw_range)
        self._close_radius_xy = close_radius_xy
        self._close_radius_z = close_radius_z
        self._reward_upright_weight = reward_upright_weight
        self._debug_randomization = debug_randomization

        self._last_reset_yaw = 0.0
        self._last_applied_yaw = 0.0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
            camera_name="watching",
            **kwargs,
        )

        self.object_names = ["box", "cylinder", "triangle"]
        self.object_info = {}
        self.object_ref_site_ids = {}

        for obj_name in self.object_names:
            body_name = f"obj_{obj_name}"
            joint_name = f"obj_{obj_name}_joint"
            ref_site_name = f"obj_{obj_name}_ref"

            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            ref_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, ref_site_name
            )

            self.object_info[obj_name] = {
                "body_name": body_name,
                "joint_name": joint_name,
                "body_id": body_id,
                "joint_id": joint_id,
                "ref_site_name": ref_site_name,
                "ref_site_id": ref_site_id,
                "qposadr": int(self.model.jnt_qposadr[joint_id]),
                "dofadr": int(self.model.jnt_dofadr[joint_id]),
            }
            self.object_ref_site_ids[obj_name] = ref_site_id

        self.active_obj_name = "box"

        self.object_z_offset = {
            "box": 0.0,
            "cylinder": 0.015,
            "triangle": 0.015,
        }

        self.target_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )
        self.target_default_pos = self.model.site_pos[self.target_site_id].copy()
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

        dummy_obs = self._get_obs()
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

        self.gripper_state = "open"
        self.max_episode_steps = 500
        self.current_step = 0
        self.success_counter = 0

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def gripper_ctrl(self, close: bool, target):
        if close:
            self.gripper_state = "closed"
            target[6] = -0.02
            target[7] = 0.02
        else:
            self.gripper_state = "open"
            target[6] = 0.01
            target[7] = -0.01

    def _get_active_obj_info(self):
        return self.object_info[self.active_obj_name]

    def _get_active_obj_pos(self):
        return self.data.body(self._get_active_obj_info()["body_name"]).xpos.copy()

    def _get_active_obj_task_pos(self):
        site_id = self.object_ref_site_ids.get(self.active_obj_name)
        if site_id is not None and site_id >= 0:
            return self.data.site_xpos[site_id].copy()

        pos = self._get_active_obj_pos().copy()
        pos[2] += self.object_z_offset[self.active_obj_name]
        return pos

    def _get_active_obj_quat(self):
        adr = self._get_active_obj_info()["qposadr"]
        return self.data.qpos[adr + 3 : adr + 7].copy()

    def _get_active_obj_upright(self):
        xmat = self.data.body(self._get_active_obj_info()["body_name"]).xmat.reshape(
            3, 3
        )
        obj_z_axis = xmat[:, 2]
        return float(np.clip(np.dot(obj_z_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))

    def _yaw_to_quat(self, yaw):
        half = yaw / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)

    def _quat_to_yaw(self, quat):
        w, x, y, z = quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _get_active_obj_yaw(self):
        return self._quat_to_yaw(self._get_active_obj_quat())

    def _sample_object_yaw(self):
        if not self._random_yaw:
            return 0.0
        return float(self.np_random.uniform(*self._yaw_range))

    def _get_yaw_info(self):
        return {
            "object_yaw": float(self._get_active_obj_yaw()),
            "sampled_object_yaw": float(self._last_reset_yaw),
            "applied_object_yaw": float(self._last_applied_yaw),
        }

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info.update(self._get_yaw_info())
        info["active_object"] = str(self.active_obj_name)
        return observation, info

    def step(self, action):
        self.current_step += 1
        action = action.copy()

        scale_arm = 0.01
        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()
        target[:-2] += scale_arm * action[:-2]

        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self._get_active_obj_task_pos()
        xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        z_dist = abs(float(ee_pos[2] - obj_pos[2]))

        if xy_dist < self._close_radius_xy and z_dist < self._close_radius_z:
            self.gripper_ctrl(close=True, target=target)
        else:
            self.gripper_ctrl(close=False, target=target)

        target = np.clip(target, self.action_space.low, self.action_space.high)
        self.do_simulation(target, self.frame_skip)

        if self.current_step % 250 == 0 or self.current_step == 1:
            print(
                f"Step: {self.current_step}, "
                f"Objek Pos: {[f'{p:.3f}' for p in self._get_active_obj_task_pos()]}, "
                f"Target Pos: {[f'{p:.3f}' for p in self.data.site('target').xpos]}, "
                f"Objek Aktif: {self.active_obj_name}, "
                # f"Yaw: {self._get_active_obj_yaw():.3f}"
            )

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = dict(reward_info)
        info["active_object"] = str(self.active_obj_name)
        terminated = self.success_counter >= 10
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_rew(self, action):
        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self._get_active_obj_task_pos()
        target_pos = self.data.site("target").xpos.copy()

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))
        reward_dist_bonus = 5.0 if dist < 0.01 else 0.0

        target_dist = np.linalg.norm(target_pos - obj_pos)
        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))
        reward_target_bonus = 15.0 if target_dist < 0.01 else 0.0

        upright = self._get_active_obj_upright()
        reward_upright = -(1.0 - upright) * self._reward_upright_weight

        control_penalty = -0.001 * np.sum(np.square(action))

        obj_dofadr = self._get_active_obj_info()["dofadr"]
        obj_vel = np.linalg.norm(self.data.qvel[obj_dofadr : obj_dofadr + 3])

        stay_bonus = 0.0
        if target_dist < 0.02 and obj_vel < 0.01:
            self.success_counter += 1
            stay_bonus = 20.0
        else:
            self.success_counter = 0

        reward_info = {
            "dist": float(dist),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "control_penalty": float(control_penalty),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_upright": float(reward_upright),
            "stay_bonus": float(stay_bonus),
            "reward_dist_bonus": float(reward_dist_bonus),
            "reward_target_bonus": float(reward_target_bonus),
            "upright": float(upright),
        }
        reward_info.update(self._get_yaw_info())

        reward = (
            reward_dist
            + reward_dist_tanh
            + control_penalty
            + reward_target
            + reward_target_tanh
            + reward_upright
            + stay_bonus
            + reward_dist_bonus
            + reward_target_bonus
        )

        return reward, reward_info

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.active_obj_name = self.np_random.choice(self.object_names)

        x = self.np_random.uniform(0.15, 0.27)
        y = self.np_random.uniform(-0.10, 0.10)
        z = 0.025

        yaw = self._sample_object_yaw()
        quat = self._yaw_to_quat(yaw)
        quat /= np.linalg.norm(quat)
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
            [x, y, 0.1], dtype=np.float64
        )

        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)

        self._last_reset_yaw = float(yaw)
        self._last_applied_yaw = float(self._get_active_obj_yaw())
        self.current_step = 0
        self.success_counter = 0

        if self._debug_randomization:
            print(
                "Reset randomization:",
                {
                    "active_object": self.active_obj_name,
                    "sampled_yaw": round(self._last_reset_yaw, 4),
                    "applied_yaw": round(self._last_applied_yaw, 4),
                },
            )

        return self._get_obs()

    def _get_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel

        first_obj_qposadr = min(info["qposadr"] for info in self.object_info.values())
        first_obj_dofadr = min(info["dofadr"] for info in self.object_info.values())

        robot_qpos = qpos[:first_obj_qposadr]
        robot_qvel = qvel[:first_obj_dofadr]
        gripper_state = robot_qpos[-2:]

        obj_pos = self._get_active_obj_task_pos()
        obj_quat = self._get_active_obj_quat()

        ee_pos = self.data.site("attachment_site").xpos
        ee_xmat = self.data.site("attachment_site").xmat
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat)

        target_pos = self.data.site("target").xpos

        rel_obj_ee = obj_pos - ee_pos
        rel_obj_target = obj_pos - target_pos

        obs = np.concatenate(
            [
                robot_qpos,
                robot_qvel,
                gripper_state,
                obj_pos,
                obj_quat,
                target_pos,
                rel_obj_ee,
                rel_obj_target,
                ee_pos,
                ee_quat,
            ]
        )

        return obs.astype(np.float32)
