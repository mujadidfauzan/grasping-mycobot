from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

from script.inverse_kinematics import (
    IKResult,
    MyCobotIK,
    _normalize_quat,
    _quat_from_euler_xyz,
    _quat_to_euler_xyz,
)


class CartesianIKActionMixin:
    _IK_ACTION_COMPONENTS = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")

    def _setup_cartesian_ik_action(
        self,
        *,
        xml_file: str,
        ee_site_name: str,
        cartesian_action_scale: float,
        cartesian_rotation_scale_deg: float,
        ik_workspace_low: tuple[float, float, float],
        ik_workspace_high: tuple[float, float, float],
        ik_max_iters: int,
        ik_position_tolerance: float,
        ik_rotation_tolerance_deg: float,
        ik_damping: float,
        ik_step_size: float,
        ik_max_delta_deg: float,
        ik_rotation_weight: float,
        ik_random_restarts: int,
        ik_seed: int | None,
    ) -> None:
        self._ik_solver = MyCobotIK(
            xml_file=Path(xml_file).expanduser().resolve(),
            ee_site_name=ee_site_name,
        )
        if self._arm_ctrl_dim != len(self._ik_solver.joint_names):
            raise ValueError(
                "Cartesian IK expects the arm actuator dimension to match the IK "
                f"joint count. Got arm_ctrl_dim={self._arm_ctrl_dim} and "
                f"ik_joints={len(self._ik_solver.joint_names)}."
            )

        self._cartesian_action_scale = float(cartesian_action_scale)
        self._cartesian_rotation_scale_rad = np.deg2rad(
            float(cartesian_rotation_scale_deg)
        )
        self._ik_workspace_low = np.asarray(ik_workspace_low, dtype=np.float64).reshape(3)
        self._ik_workspace_high = np.asarray(ik_workspace_high, dtype=np.float64).reshape(3)
        if np.any(self._ik_workspace_low > self._ik_workspace_high):
            raise ValueError(
                "ik_workspace_low must be ordered element-wise below ik_workspace_high."
            )

        self._ik_max_iters = int(ik_max_iters)
        self._ik_position_tolerance = float(ik_position_tolerance)
        self._ik_rotation_tolerance_rad = np.deg2rad(float(ik_rotation_tolerance_deg))
        self._ik_damping = float(ik_damping)
        self._ik_step_size = float(ik_step_size)
        self._ik_max_delta_rad = np.deg2rad(float(ik_max_delta_deg))
        self._ik_rotation_weight = float(ik_rotation_weight)
        self._ik_random_restarts = max(0, int(ik_random_restarts))
        self._ik_seed = None if ik_seed is None else int(ik_seed)

        self._policy_action_dim = len(self._IK_ACTION_COMPONENTS)
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_action_dim,),
            dtype=np.float32,
        )
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._ik_failure_count = 0
        self._ik_target_pos = np.zeros(3, dtype=np.float64)
        self._ik_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_ik_result: IKResult | None = None
        self._reset_cartesian_ik_state()

    @staticmethod
    def _wrap_to_pi(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    def _reset_cartesian_ik_state(self) -> None:
        ee_pos, ee_quat = self._get_ee_pose()
        self._ik_target_pos = np.asarray(ee_pos, dtype=np.float64).copy()
        self._ik_target_quat = _normalize_quat(np.asarray(ee_quat, dtype=np.float64))
        self._last_ik_result = None
        self._ik_failure_count = 0
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def _current_arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            self.data.qpos[self._ik_solver.qpos_indices],
            dtype=np.float64,
        ).copy()

    def _compute_cartesian_ik_target(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ee_pos, ee_quat = self._get_ee_pose()
        delta_pos = self._cartesian_action_scale * np.asarray(action[:3], dtype=np.float64)
        delta_rpy = self._cartesian_rotation_scale_rad * np.asarray(
            action[3:6], dtype=np.float64
        )

        target_pos = np.clip(
            np.asarray(ee_pos, dtype=np.float64) + delta_pos,
            self._ik_workspace_low,
            self._ik_workspace_high,
        )
        current_rpy = _quat_to_euler_xyz(np.asarray(ee_quat, dtype=np.float64))
        target_rpy = np.array(
            [self._wrap_to_pi(value) for value in current_rpy + delta_rpy],
            dtype=np.float64,
        )
        target_quat = _quat_from_euler_xyz(*target_rpy)
        return target_pos, target_quat

    def _cartesian_action_to_target_ctrl(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, IKResult]:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != self.action_space.shape:
            expected_components = ", ".join(self._IK_ACTION_COMPONENTS)
            raise ValueError(
                f"Unexpected action shape for {type(self).__name__}. "
                f"Expected {self.action_space.shape} ({expected_components}), "
                f"got {action.shape}."
            )

        action = np.clip(action, self.action_space.low, self.action_space.high)
        target_pos, target_quat = self._compute_cartesian_ik_target(action)
        ik_result = self._ik_solver.solve(
            target_pos,
            target_quat,
            initial_q=self._current_arm_joint_positions(),
            position_only=False,
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

        target_ctrl = self.data.ctrl.copy()
        target_ctrl[: self._arm_ctrl_dim] = ik_result.q_rad
        target_ctrl = np.clip(target_ctrl, self._ctrl_low, self._ctrl_high)

        self.last_action = action.astype(np.float32)
        self._ik_target_pos = target_pos.copy()
        self._ik_target_quat = target_quat.copy()
        self._last_ik_result = ik_result
        if not ik_result.success:
            self._ik_failure_count += 1

        return action, target_ctrl, ik_result

    def _append_cartesian_ik_config(self, config: dict) -> dict:
        action_config = config.setdefault("action", {})
        action_config["controller"] = "cartesian_ik"
        action_config["action_components"] = list(self._IK_ACTION_COMPONENTS)
        action_config["cartesian_action_scale_m"] = float(self._cartesian_action_scale)
        action_config["cartesian_rotation_scale_deg"] = float(
            np.rad2deg(self._cartesian_rotation_scale_rad)
        )
        action_config["ik_workspace_low"] = self._ik_workspace_low.tolist()
        action_config["ik_workspace_high"] = self._ik_workspace_high.tolist()
        action_config["ik_max_iters"] = int(self._ik_max_iters)
        action_config["ik_position_tolerance"] = float(self._ik_position_tolerance)
        action_config["ik_rotation_tolerance_deg"] = float(
            np.rad2deg(self._ik_rotation_tolerance_rad)
        )
        action_config["ik_damping"] = float(self._ik_damping)
        action_config["ik_step_size"] = float(self._ik_step_size)
        action_config["ik_max_delta_deg"] = float(np.rad2deg(self._ik_max_delta_rad))
        action_config["ik_rotation_weight"] = float(self._ik_rotation_weight)
        action_config["ik_random_restarts"] = int(self._ik_random_restarts)
        action_config["ik_seed"] = self._ik_seed
        action_config["ik_joint_names"] = list(self._ik_solver.joint_names)
        action_config["ee_site_name"] = str(self.ee_site_name)
        return config

    def _get_cartesian_ik_debug_state(self) -> dict:
        debug_state = {
            "ik_target_pos": self._ik_target_pos.copy(),
            "ik_target_quat": self._ik_target_quat.copy(),
            "ik_target_rpy_deg": np.rad2deg(
                _quat_to_euler_xyz(self._ik_target_quat)
            ),
            "ik_failure_count": int(self._ik_failure_count),
        }
        if self._last_ik_result is None:
            debug_state.update(
                {
                    "ik_success": None,
                    "ik_iterations": 0,
                    "ik_attempt": 0,
                }
            )
            return debug_state

        debug_state.update(
            {
                "ik_success": bool(self._last_ik_result.success),
                "ik_message": self._last_ik_result.message,
                "ik_iterations": int(self._last_ik_result.iterations),
                "ik_attempt": int(self._last_ik_result.attempt),
                "ik_position_error": self._last_ik_result.position_error.copy(),
                "ik_rotation_error": self._last_ik_result.rotation_error.copy(),
                "ik_position_error_norm": float(
                    self._last_ik_result.position_error_norm
                ),
                "ik_rotation_error_deg": float(
                    np.rad2deg(self._last_ik_result.rotation_error_norm)
                ),
            }
        )
        return debug_state
