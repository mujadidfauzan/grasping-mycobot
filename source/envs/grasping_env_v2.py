import mujoco
import numpy as np

from .grasping_env_v1 import GraspingEnvV1


class GraspingEnvV2(GraspingEnvV1):
    def __init__(
        self,
        random_yaw: bool = True,
        close_radius_xy: float = 0.02,
        close_radius_z: float = 0.025,
        reward_upright_weight: float = 0.5,
        **kwargs,
    ):
        self._random_yaw = random_yaw
        self._close_radius_xy = close_radius_xy
        self._close_radius_z = close_radius_z
        self._reward_upright_weight = reward_upright_weight
        self.object_ref_site_ids = {}

        super().__init__(**kwargs)

        for obj_name in self.object_names:
            site_name = f"obj_{obj_name}_ref"
            self.object_ref_site_ids[obj_name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )

    def _get_active_obj_task_pos(self):
        if not self.object_ref_site_ids or self.active_obj_name not in self.object_ref_site_ids:
            return super()._get_active_obj_task_pos()
        site_id = self.object_ref_site_ids[self.active_obj_name]
        return self.data.site_xpos[site_id].copy()

    def _get_active_obj_upright(self):
        xmat = self.data.body(self._get_active_obj_info()["body_name"]).xmat.reshape(3, 3)
        obj_z_axis = xmat[:, 2]
        return float(np.clip(np.dot(obj_z_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))

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
                f"Objek Aktif: {self.active_obj_name}"
            )

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info
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
        reward_dist_bonus = 3.0 if dist < 0.01 else 0.0

        target_dist = np.linalg.norm(target_pos - obj_pos)
        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))
        reward_target_bonus = 5.0 if target_dist < 0.01 else 0.0

        upright = self._get_active_obj_upright()
        reward_upright = -(1.0 - upright) * self._reward_upright_weight

        control_penalty = -0.001 * np.sum(np.square(action))

        obj_vel = np.linalg.norm(
            self.data.qvel[
                self._get_active_obj_info()["dofadr"] : self._get_active_obj_info()[
                    "dofadr"
                ]
                + 3
            ]
        )

        stay_bonus = 0.0
        if target_dist < 0.02 and obj_vel < 0.01:
            self.success_counter += 1
            stay_bonus = 2.0
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

        yaw = self.np_random.uniform(-np.pi, np.pi) if self._random_yaw else 0.0
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

        self.model.site_pos[self.target_site_id] = np.array([x, y, 0.1], dtype=np.float64)

        qpos[self.gripL_qadr] = 0.0
        qpos[self.gripR_qadr] = 0.0
        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        self.success_counter = 0

        return self._get_obs()
