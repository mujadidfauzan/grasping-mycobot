from __future__ import annotations

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box

from .config_export import capture_init_config
from .place_target_env import DEFAULT_CAMERA_CONFIG, DEFAULT_XML_PATH, PlaceTargetEnv


class PlaceAboveTargetEnv(PlaceTargetEnv):
    def __init__(
        self,
        xml_file: str = str(DEFAULT_XML_PATH),
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        reward_target_weight: float = 5.0,
        reward_target_tanh_weight: float = 3.0,
        reward_target_bonus: float = 12.0,
        reward_stay_bonus: float = 20.0,
        reward_hold_weight: float = 0.75,
        reward_drop_penalty: float = 8.0,
        control_penalty_weight: float = 0.001,
        success_distance: float = 0.02,
        success_steps_required: int = 10,
        terminate_ee_obj_distance: float = 0.08,
        target_height_above_place: float = 0.03,
        **kwargs,
    ):
        init_config = capture_init_config(locals())
        super().__init__(
            xml_file=xml_file,
            frame_skip=frame_skip,
            default_camera_config=default_camera_config,
            reward_target_weight=reward_target_weight,
            reward_target_tanh_weight=reward_target_tanh_weight,
            reward_target_orient_weight=0.0,
            reward_target_bonus=reward_target_bonus,
            reward_stay_bonus=reward_stay_bonus,
            reward_drop_penalty=reward_drop_penalty,
            control_penalty_weight=control_penalty_weight,
            success_distance=success_distance,
            success_angle_deg=180.0,
            success_steps_required=success_steps_required,
            terminate_ee_obj_distance=terminate_ee_obj_distance,
            target_height_above_place=target_height_above_place,
            **kwargs,
        )
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            reward_target_weight,
            reward_target_tanh_weight,
            reward_target_bonus,
            reward_stay_bonus,
            reward_hold_weight,
            reward_drop_penalty,
            control_penalty_weight,
            success_distance,
            success_steps_required,
            terminate_ee_obj_distance,
            target_height_above_place,
            **kwargs,
        )

        self._init_config = init_config
        self._reward_hold_weight = float(reward_hold_weight)
        self._policy_action_dim = self._arm_ctrl_dim
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_action_dim,),
            dtype=np.float32,
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "closed"

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
            "Unexpected action shape for PlaceAboveTargetEnv. "
            f"Expected {self.action_space.shape} (arm only), "
            f"{place_action_shape} (arm + gripper command), "
            f"or legacy {legacy_shape}, got {action.shape}."
        )

    def reset_model(self):
        super().reset_model()
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        closed_ctrl = self.data.ctrl.copy()
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def reset_from_grasp_snapshot(
        self,
        snapshot: dict,
        *,
        seed: int | None = None,
        reset_source: str = "external_grasp_snapshot",
        attempt_count: int = 1,
    ) -> np.ndarray:
        super().reset_from_grasp_snapshot(
            snapshot,
            seed=seed,
            reset_source=reset_source,
            attempt_count=attempt_count,
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        closed_ctrl = self.data.ctrl.copy()
        self._set_closed_gripper_target(closed_ctrl)
        self.data.ctrl[:] = np.clip(closed_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(self.data.ctrl)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.current_step += 1
        action = self._coerce_policy_action(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.astype(np.float32)

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[: self._arm_ctrl_dim] += (
            self._arm_action_scale * action[: self._arm_ctrl_dim]
        )
        self._set_closed_gripper_target(target_ctrl)
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)
        self._update_gripper_state_from_target(target_ctrl)
        self._disable_grasp_constraints()

        self.do_simulation(target_ctrl, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated_success = self.success_counter >= self._success_steps_required

        terminated_ee_obj_far = bool(
            float(reward_info["ee_object_dist"]) >= self._terminate_ee_obj_distance
            and float(reward_info["object_target_dist"]) > self._success_distance
        )
        terminated = terminated_success or terminated_ee_obj_far
        truncated = self.current_step >= self.max_episode_steps
        reward_info["terminated_success"] = int(terminated_success)
        reward_info["terminated_ee_obj_far"] = int(terminated_ee_obj_far)

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

        self.best_object_target_dist = min(self.best_object_target_dist, target_dist)
        target_progress = self._get_target_progress()
        required_target_progress = self._get_required_target_progress()
        has_approached_target = bool(target_progress >= required_target_progress)
        terminated_ee_obj_far = bool(
            ee_obj_dist >= self._terminate_ee_obj_distance
            and target_dist > self._success_distance
        )

        reward_target = -target_dist * self._reward_target_weight
        reward_target_tanh = (
            1.0 - float(np.tanh(target_dist / 0.05))
        ) * self._reward_target_tanh_weight
        # reward_hold = (
        #     1.0 - float(np.tanh(ee_obj_dist / 0.04))
        # ) * self._reward_hold_weight
        control_penalty = -self._control_penalty_weight * float(
            np.sum(np.square(action))
        )

        target_pose_aligned = bool(target_dist < self._success_distance)
        reward_target_bonus = self._reward_target_bonus if target_pose_aligned else 0.0
        drop_penalty = (
            -self._reward_drop_penalty
            if terminated_ee_obj_far and not has_approached_target
            else 0.0
        )

        if target_pose_aligned:
            self.success_counter += 1
            stay_bonus = self._reward_stay_bonus
        else:
            self.success_counter = 0
            stay_bonus = 0.0

        reward = (
            reward_target
            + reward_target_tanh
            # + reward_hold
            + reward_target_bonus
            + stay_bonus
            + drop_penalty
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
            "reward_target_orient": 0.0,
            # "reward_hold": float(reward_hold),
            "reward_target_bonus": float(reward_target_bonus),
            "target_pose_aligned": int(target_pose_aligned),
            "stay_bonus": float(stay_bonus),
            "drop_penalty": float(drop_penalty),
            "control_penalty": float(control_penalty),
            "target_progress": float(target_progress),
            "required_target_progress": float(required_target_progress),
            "has_approached_target": bool(has_approached_target),
        }

        return float(reward), reward_info

    def export_config(self) -> dict:
        config = super().export_config()
        config["action"]["gripper_policy"] = "fixed_closed"
        config["reward"]["params"]["reward_hold_weight"] = float(
            self._reward_hold_weight
        )
        config["task"]["target_mode"] = "object_position_above_place"
        config["task"]["success_criterion"] = {
            "object_target_distance_only": float(self._success_distance)
        }
        return config

    def get_debug_state(self) -> dict:
        state = super().get_debug_state()
        state["gripper_assist_mix"] = 1.0
        state["gripper_should_close"] = True
        state["gripper_policy"] = "fixed_closed"
        state["task_mode"] = "object_above_target_place"
        state["reward_hold_weight"] = float(self._reward_hold_weight)
        return state
