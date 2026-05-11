from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .utils import ensure_project_root_on_path, safe_debug_state

ensure_project_root_on_path()

from script.inverse_kinematics import _normalize_quat  # noqa: E402
from source.envs.insert_target_env_ik import InsertTargetEnvIK  # noqa: E402


class QMPInsertEndToEndEnv(InsertTargetEnvIK):
    """InsertTargetEnvIK variant that starts from object-on-table states.

    The original InsertTargetEnvIK starts each episode from a snapshot produced by
    a grasp policy. Q-switch training needs the grasp primitive to matter, so this
    subclass keeps the same observation/action/reward contract but samples the
    object on the table and lets a wrapper handle manual gripper open/close logic.

    This class lives in qmp-her only; the source env remains untouched.
    """

    def __init__(
        self,
        *args: Any,
        object_x_range: tuple[float, float] = (0.15, 0.27),
        object_y_range: tuple[float, float] = (-0.12, 0.12),
        object_z: float = 0.025,
        object_yaw_range: tuple[float, float] = (-np.pi / 4.0, np.pi / 4.0),
        reset_settle_steps: int = 20,
        **kwargs: Any,
    ):
        self._qmp_object_x_range = tuple(float(v) for v in object_x_range)
        self._qmp_object_y_range = tuple(float(v) for v in object_y_range)
        self._qmp_object_z = float(object_z)
        self._qmp_object_yaw_range = tuple(float(v) for v in object_yaw_range)
        self._qmp_reset_settle_steps = max(0, int(reset_settle_steps))
        super().__init__(*args, **kwargs)

    def _sample_qmp_object_pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        x = float(self.np_random.uniform(*self._qmp_object_x_range))
        y = float(self.np_random.uniform(*self._qmp_object_y_range))
        yaw = float(self.np_random.uniform(*self._qmp_object_yaw_range))
        pos = np.array([x, y, self._qmp_object_z], dtype=np.float64)
        quat = self._yaw_to_quat(yaw)
        return pos, quat, yaw

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        self.current_step = 0
        self.success_counter = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.gripper_state = "open"
        self._eval_gripper_open = True
        self.initial_object_target_dist = np.inf
        self.best_object_target_dist = np.inf

        self._set_place_pose_in_model(self._default_place_pos, self._default_place_quat)
        self._sync_target_site_to_active_place()

        obj_pos, obj_quat, obj_yaw = self._sample_qmp_object_pose()
        object_info = self._get_active_obj_info()
        object_qposadr = int(object_info["qposadr"])
        object_dofadr = int(object_info["dofadr"])
        qpos[object_qposadr : object_qposadr + 3] = obj_pos
        qpos[object_qposadr + 3 : object_qposadr + 7] = obj_quat
        qvel[object_dofadr : object_dofadr + 6] = 0.0

        self.set_state(qpos, qvel)
        ctrl = self.data.ctrl.copy()
        self._set_open_gripper_target(ctrl)
        self.data.ctrl[:] = np.clip(ctrl, self._ctrl_low, self._ctrl_high)
        mujoco.mj_forward(self.model, self.data)

        if self._qmp_reset_settle_steps > 0:
            settle_ctrl = self.data.ctrl.copy()
            self._set_open_gripper_target(settle_ctrl)
            settle_ctrl = np.clip(settle_ctrl, self._ctrl_low, self._ctrl_high)
            for _ in range(self._qmp_reset_settle_steps):
                self.do_simulation(settle_ctrl, 1)
            mujoco.mj_forward(self.model, self.data)

        obj_pos, obj_quat = self._get_active_obj_pose()
        (
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
            self.sampled_target_place_yaw,
            self._last_target_resample_distance,
            self._last_target_resample_attempts,
        ) = self._sample_target_place_pose_away_from_object(obj_pos)
        self._set_place_pose_in_model(
            self.sampled_target_place_pos,
            self.sampled_target_place_quat,
        )
        self._sync_target_site_to_active_place()
        mujoco.mj_forward(self.model, self.data)

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
        self.initial_obj_site_pos = obj_pos.copy()
        self.sampled_target_pos = target_pos.copy()
        self.sampled_target_quat = target_quat.copy()
        self.sampled_object_yaw = float(obj_yaw)
        self.applied_object_yaw = float(self._quat_to_yaw(obj_quat))
        self.applied_target_place_yaw = float(
            self._quat_to_yaw(
                _normalize_quat(self.data.body(self.place_body_name).xquat.copy())
            )
        )
        self._last_grasp_reset_attempts = 0
        self._last_grasp_init_lift_height = 0.0
        self._last_grasp_init_ee_obj_dist = np.inf
        self._last_grasp_reset_source = "qmp_end_to_end_table_reset"
        self._last_grasp_source_object = self.object_name
        self._reset_ik_state()

        return self._get_obs()


class ManualGripperRewardWrapper(gym.Wrapper):
    """Manual gripper controller and release-aware reward shaping.

    The target policy remains 6-DoF Cartesian IK only. This wrapper turns gripper
    control into deterministic training logic:
    - open at reset,
    - close when EE is close to the object,
    - open/release when object pose is aligned with the insertion target.

    After release, the parent env's object-target distance reward can be removed
    so the falling object is not punished for moving away from the 5 cm-above-place
    target site.
    """

    POSE_REWARD_KEYS = (
        "reward_target",
        "reward_target_tanh",
        "reward_orientation",
        "reward_orientation_tanh",
        "reward_success_bonus",
    )

    def __init__(
        self,
        env: gym.Env,
        *,
        start_open: bool = True,
        close_distance: float = 0.018,
        close_angle_rad: float | None = None,
        release_distance: float = 0.012,
        release_angle_rad: float = np.deg2rad(10.0),
        require_closed_before_release: bool = True,
        disable_pose_reward_after_release: bool = True,
        release_bonus: float = 30.0,
        post_release_reward: float = 1.0,
        terminate_after_release_steps: int = 5,
    ):
        super().__init__(env)
        self.start_open = bool(start_open)
        self.close_distance = float(close_distance)
        self.close_angle_rad = None if close_angle_rad is None else float(close_angle_rad)
        self.release_distance = float(release_distance)
        self.release_angle_rad = float(release_angle_rad)
        self.require_closed_before_release = bool(require_closed_before_release)
        self.disable_pose_reward_after_release = bool(disable_pose_reward_after_release)
        self.release_bonus = float(release_bonus)
        self.post_release_reward = float(post_release_reward)
        self.terminate_after_release_steps = max(0, int(terminate_after_release_steps))

        self.gripper_phase = "open" if self.start_open else "closed"
        self.release_steps = 0
        self.release_event_count = 0
        self.last_manual_event = "reset"
        self._last_qswitch_debug: dict[str, Any] = {}

    def set_qswitch_debug(self, debug: Mapping[str, Any]) -> None:
        self._last_qswitch_debug = dict(debug)

    def _base_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _set_gripper_open(self, open_gripper: bool) -> None:
        base_env = self._base_env()
        setter = getattr(base_env, "set_eval_gripper_open", None)
        if callable(setter):
            setter(bool(open_gripper))
            return
        opener = getattr(base_env, "open_gripper_for_eval", None)
        closer = getattr(base_env, "close_gripper", None)
        if bool(open_gripper) and callable(opener):
            opener()
        elif not bool(open_gripper) and callable(closer):
            closer()

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _metric(state: Mapping[str, Any], *keys: str, default: float = np.inf) -> float:
        for key in keys:
            if key not in state:
                continue
            try:
                value = float(state[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return float(default)

    def _should_close(self, state: Mapping[str, Any]) -> bool:
        dist = self._metric(state, "ee_object_dist", "ee_obj_dist")
        if dist > self.close_distance:
            return False
        if self.close_angle_rad is None:
            return True
        angle = self._metric(state, "ee_obj_angle_rad", "ee_object_rot_error")
        return angle <= self.close_angle_rad

    def _should_release(self, state: Mapping[str, Any]) -> bool:
        if self._truthy(state.get("target_pose_aligned", False)):
            return True
        dist = self._metric(state, "object_target_dist", "obj_target_dist")
        angle = self._metric(
            state,
            "object_target_angle_rad",
            "obj_target_angle_rad",
            "object_target_rot_error",
        )
        return bool(dist <= self.release_distance and angle <= self.release_angle_rad)

    def _apply_manual_gripper(self) -> str:
        state = safe_debug_state(self._base_env())
        event = "hold"

        if self.gripper_phase == "released":
            self._set_gripper_open(True)
            self.last_manual_event = "released_hold"
            return "released_hold"

        if self.gripper_phase == "open" and self._should_close(state):
            self._set_gripper_open(False)
            self.gripper_phase = "closed"
            event = "close_near_object"

        closed_enough = self.gripper_phase == "closed" or not self.require_closed_before_release
        if closed_enough and self._should_release(state):
            self._set_gripper_open(True)
            self.gripper_phase = "released"
            self.release_steps = 0
            self.release_event_count += 1
            event = "release_on_target_align"

        self.last_manual_event = event
        return event

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        self.gripper_phase = "open" if self.start_open else "closed"
        self.release_steps = 0
        self.release_event_count = 0
        self.last_manual_event = "reset_open" if self.start_open else "reset_closed"
        self._last_qswitch_debug = {}
        self._set_gripper_open(self.start_open)
        obs_getter = getattr(self._base_env(), "_get_obs", None)
        if callable(obs_getter):
            observation = obs_getter()
        return observation, info

    def step(self, action):
        manual_event = self._apply_manual_gripper()
        observation, reward, terminated, truncated, info = self.env.step(action)

        original_reward = float(reward)
        adjusted_reward = original_reward
        released = self.gripper_phase == "released"
        release_first_step = manual_event == "release_on_target_align"

        if released:
            self.release_steps += 1
            if self.disable_pose_reward_after_release:
                for key in self.POSE_REWARD_KEYS:
                    try:
                        adjusted_reward -= float(info.get(key, 0.0))
                    except (TypeError, ValueError):
                        pass
            adjusted_reward += self.post_release_reward
            if release_first_step:
                adjusted_reward += self.release_bonus

            if (
                self.terminate_after_release_steps > 0
                and self.release_steps >= self.terminate_after_release_steps
            ):
                terminated = True

        info = dict(info)
        info.update(
            {
                "manual_gripper_phase": self.gripper_phase,
                "manual_gripper_event": manual_event,
                "manual_release_steps": int(self.release_steps),
                "manual_release_event_count": int(self.release_event_count),
                "manual_original_reward": original_reward,
                "manual_adjusted_reward": float(adjusted_reward),
                "manual_pose_reward_disabled": int(
                    released and self.disable_pose_reward_after_release
                ),
            }
        )
        if self._last_qswitch_debug:
            info["qswitch"] = dict(self._last_qswitch_debug)

        return observation, float(adjusted_reward), terminated, truncated, info

    def get_debug_state(self) -> dict[str, Any]:
        state = safe_debug_state(self._base_env())
        state.update(
            {
                "manual_gripper_phase": self.gripper_phase,
                "manual_gripper_event": self.last_manual_event,
                "manual_release_steps": int(self.release_steps),
                "manual_release_event_count": int(self.release_event_count),
                "qswitch": dict(self._last_qswitch_debug),
            }
        )
        return state
