import time

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


class GraspingEnv(MujocoEnv, utils.EzPickle):

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
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_dist_weight,
            reward_dist_target_weight,
            **kwargs,
        )

        self._reward_dist_weight = reward_dist_weight
        self._reward_dist_target_weight = reward_dist_target_weight

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=None,
            camera_name="watching",
            **kwargs,
        )

        obj_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint"
        )
        self.obj_qposadr = int(self.model.jnt_qposadr[obj_joint_id])
        self.obj_dofadr = int(self.model.jnt_dofadr[obj_joint_id])

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

    def step(self, action):
        self.current_step += 1
        action = action.copy()

        scale_arm = 0.01

        current_ctrl = self.data.ctrl.copy()
        target = current_ctrl.copy()

        target[:-2] += scale_arm * action[:-2]
        # target[6] = 0.02
        # target[7] = -0.02

        if (
            np.linalg.norm(
                self.data.site("attachment_site").xpos.copy()
                - self.data.body("obj").xpos.copy()
            )
            < 0.03
        ):
            self.gripper_ctrl(close=True, target=target)
        else:
            self.gripper_ctrl(close=False, target=target)

        target = np.clip(target, self.action_space.low, self.action_space.high)
        # print("Target after clipping:", target)

        self.do_simulation(target, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def _get_rew(self, action):
        ee_pos = self.data.site("attachment_site").xpos.copy()
        obj_pos = self.data.body("obj").xpos.copy()
        target_pos = self.data.site("target").xpos.copy()

        dist = np.linalg.norm(ee_pos - obj_pos)
        reward_dist = -dist * self._reward_dist_weight
        reward_dist_tanh = 1.0 - float(np.tanh(float(dist) / 0.10))

        target_dist = np.linalg.norm(target_pos - obj_pos)
        reward_target = -target_dist * self._reward_dist_target_weight
        reward_target_tanh = 1.0 - float(np.tanh(float(target_dist) / 0.10))

        obj_quat = self.data.qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7]
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat_dot = np.abs(np.dot(obj_quat, target_quat))
        quat_dot = np.clip(quat_dot, -1.0, 1.0)
        orientation_error = 1.0 - quat_dot
        reward_orient = -orientation_error * 0.5

        control_penalty = -0.001 * np.sum(np.square(action))
        reward_info = {
            "dist": float(dist),
            "reward_dist": float(reward_dist),
            "reward_dist_tanh": float(reward_dist_tanh),
            "control_penalty": float(control_penalty),
            "reward_target": float(reward_target),
            "reward_target_tanh": float(reward_target_tanh),
            "reward_orient": float(reward_orient),
        }

        reward = (
            reward_dist
            + reward_dist_tanh
            # + touch_bonus
            + control_penalty
            + reward_target
            # + reward_lift
            + reward_target_tanh
            + reward_orient
        )

        return reward, reward_info

    def reset_model(self):

        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        adr = self.obj_qposadr
        vadr = self.obj_dofadr

        # Random XY
        x = self.np_random.uniform(0.15, 0.27)
        y = self.np_random.uniform(-0.10, 0.10)
        z = 0.025

        # Set position
        qpos[adr + 0] = x
        qpos[adr + 1] = y
        qpos[adr + 2] = z

        # Identity quaternion
        qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]

        # Zero object velocity
        qvel[vadr : vadr + 6] = 0.0

        tz = 0.1

        self.model.site_pos[self.target_site_id] = np.array(
            [x, y, tz], dtype=np.float64
        )

        # Open Gripper
        qpos[self.gripL_qadr] = 0
        qpos[self.gripR_qadr] = 0

        qvel[self.gripL_dadr] = 0.0
        qvel[self.gripR_dadr] = 0.0

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0

        return self._get_obs()

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel

        # Robot joint positions & velocities
        robot_qpos = qpos[: self.obj_qposadr]
        robot_qvel = qvel[: self.obj_dofadr]

        # Gripper state (2 finger joint terakhir)
        gripper_state = robot_qpos[-2:]

        # Object
        obj_pos = qpos[self.obj_qposadr : self.obj_qposadr + 3]
        obj_quat = qpos[self.obj_qposadr + 3 : self.obj_qposadr + 7]

        # End effector
        ee_pos = self.data.site("attachment_site").xpos
        ee_xmat = self.data.site("attachment_site").xmat
        ee_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(ee_quat, ee_xmat)

        # Target
        target_pos = self.data.site("target").xpos

        # Relative positions
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

    def get_physics_state(self):
        return {
            "step": self.current_step,
            "obj_pos": self.data.body("obj").xpos.copy(),
            "target_pos": self.data.site("target").xpos.copy(),
            "ee_pos": self.data.site("attachment_site").xpos.copy(),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "gripper_state": self.gripper_state,
        }