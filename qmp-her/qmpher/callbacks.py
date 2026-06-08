from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import torch as th
import torch.nn.functional as F

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .utils import coerce_scalar

KNOWN_SELECTION_LABELS = (
    "primitive_grasp",
    "primitive_insert",
    "script_lift",
    "target_policy",
    "random_action",
    "zero_action",
)

SHORT_LABELS = {
    "primitive_grasp": "pg",
    "primitive_insert": "pi",
    "script_lift": "lift",
    "target_policy": "tp",
    "random_action": "rand",
    "zero_action": "zero",
}

ENV_METRICS = (
    ("ee_object_dist", "task/ee_obj_d"),
    ("object_target_dist", "task/obj_tgt_d"),
    ("object_target_angle_rad", "task/obj_tgt_ang"),
    ("object_place_radial_error", "task/place_radial"),
    ("object_place_height_error", "task/place_height"),
    ("object_place_angle_error", "task/place_ang"),
    ("lift_height", "task/lift_h"),
    ("lift_progress", "task/lift_p"),
    ("target_pose_aligned", "task/aligned"),
    ("place_pose_aligned", "task/place_aligned"),
    ("success_counter", "task/succ_cnt"),
    ("terminated_success", "ep/term_succ"),
    ("terminated_lost_object", "ep/term_lost"),
    ("gripper_closed", "grip/closed"),
    ("ik_success", "ik/success"),
    ("ik_failure_count", "ik/fail_cnt"),
    ("reward_total", "rew/total"),
    ("reward_her_sparse", "rew/her"),
    ("reward_grasp_approach", "rew/grasp_app"),
    ("reward_grasp_closed", "rew/grasp_cls"),
    ("reward_lift", "rew/lift"),
    ("reward_target_position", "rew/tgt_pos"),
    ("reward_target_orientation", "rew/tgt_ori"),
)

QSWITCH_METRICS = (
    ("enabled", "qs/en"),
    ("num_candidates", "qs/n_cand"),
    ("selected_q", "qs/sel_q"),
    ("epsilon", "qs/eps"),
    ("target_included", "qs/tp_in"),
    ("target_step_allowed", "qs/tp_step_ok"),
    ("target_phase_allowed", "qs/tp_phase_ok"),
    ("target_forced_fallback", "qs/tp_fallback"),
    ("target_policy_only", "qs/tp_only"),
    ("q_abs_max", "qs/q_abs_max"),
    ("q_unstable", "qs/q_unstable"),
    ("stickiness_applied", "qs/sticky_applied"),
    ("sticky_steps_left", "qs/sticky_steps_left"),
    ("teacher_stored", "qs/teacher_stored"),
)

PHASE_FILTER_METRICS = (
    ("lift_h", "qs/pf_lift_h"),
    ("before", "qs/pf_before"),
    ("after", "qs/pf_after"),
)

TARGET_PHASE_METRICS = (
    ("allowed", "qs/tp_phase_allowed"),
    ("lift_h", "qs/tp_lift_h"),
    ("min_lift_h", "qs/tp_min_lift_h"),
)


def _safe_metric_label(label: str) -> str:
    return SHORT_LABELS.get(str(label), str(label).replace("/", "_").replace(" ", "_"))


def _append_metric(metrics: dict[str, list[float]], key: str, value: Any) -> None:
    scalar = coerce_scalar(value)
    if scalar is not None:
        metrics.setdefault(key, []).append(scalar)


class QSwitchInfoCallback(BaseCallback):
    """Log a compact, non-duplicated subset of env and q-switch metrics."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_freq = int(log_freq)

    def _on_step(self) -> bool:
        if self.log_freq <= 0 or self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        metrics: dict[str, list[float]] = {}
        for info in infos:
            if not isinstance(info, Mapping):
                continue

            for source_key, log_key in ENV_METRICS:
                _append_metric(metrics, log_key, info.get(source_key))

            qswitch = info.get("qswitch")
            if isinstance(qswitch, Mapping):
                for source_key, log_key in QSWITCH_METRICS:
                    _append_metric(metrics, log_key, qswitch.get(source_key))

                phase_filter = qswitch.get("pf")
                if isinstance(phase_filter, Mapping):
                    for source_key, log_key in PHASE_FILTER_METRICS:
                        _append_metric(metrics, log_key, phase_filter.get(source_key))

                target_phase = qswitch.get("tp_phase")
                if isinstance(target_phase, Mapping):
                    for source_key, log_key in TARGET_PHASE_METRICS:
                        _append_metric(metrics, log_key, target_phase.get(source_key))

                selected_label = qswitch.get("selected_label")
                candidate_labels = qswitch.get("candidate_labels", [])
                label_set = set(KNOWN_SELECTION_LABELS)
                if isinstance(candidate_labels, (list, tuple)):
                    label_set.update(str(label) for label in candidate_labels)
                if isinstance(selected_label, str):
                    label_set.add(selected_label)

                for label in sorted(label_set):
                    safe_label = _safe_metric_label(label)
                    _append_metric(
                        metrics,
                        f"qs/cand/{safe_label}",
                        float(
                            isinstance(candidate_labels, (list, tuple))
                            and label in candidate_labels
                        ),
                    )
                    _append_metric(
                        metrics,
                        f"qs/sel/{safe_label}",
                        float(
                            isinstance(selected_label, str) and selected_label == label
                        ),
                    )

        for key, values in metrics.items():
            if values:
                self.logger.record(f"custom/{key}", float(np.mean(values)))

        return True


class QSwitchBCCallback(BaseCallback):
    """Auxiliary behavior cloning update for the target actor.

    This trains the SAC actor to imitate the current non-target teacher action
    stored by QSwitchSAC: primitive_grasp, script_lift, or primitive_insert.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        config = getattr(self.model, "qswitch_config", None)
        if config is None or not getattr(config, "bc_enabled", False):
            return

        if self.num_timesteps < int(config.bc_start_steps):
            return

        coef = float(self.model.bc_coef())
        if coef <= 0.0:
            return

        batch_size = int(config.bc_batch_size)
        gradient_steps = int(config.bc_gradient_steps)

        losses = []

        self.model.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            batch = self.model.sample_teacher_batch(batch_size)
            if batch is None:
                return

            obs_tensor, teacher_actions = batch

            # SB3 SAC actor returns scaled action in [-1, 1].
            pred_actions = self.model.policy.actor(
                obs_tensor,
                deterministic=True,
            )

            bc_loss = F.mse_loss(pred_actions, teacher_actions)
            loss = coef * bc_loss

            self.model.policy.actor.optimizer.zero_grad()
            loss.backward()

            th.nn.utils.clip_grad_norm_(
                self.model.policy.actor.parameters(),
                max_norm=float(config.bc_max_grad_norm),
            )

            self.model.policy.actor.optimizer.step()
            losses.append(float(bc_loss.detach().cpu().item()))

        if losses:
            self.logger.record("train/bc_loss", float(np.mean(losses)))
            self.logger.record("train/bc_coef", coef)
            self.logger.record(
                "train/bc_teacher_buffer_size",
                float(self.model.teacher_buffer_size()),
            )


class PrintQSwitchCallback(BaseCallback):
    """Occasionally print the last selector decision for terminal debugging."""

    def __init__(self, print_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_freq = int(print_freq)

    def _on_step(self) -> bool:
        if self.print_freq <= 0 or self.n_calls % self.print_freq != 0:
            return True
        debug = getattr(self.model, "last_qswitch_debug", None)
        if not isinstance(debug, Mapping):
            return True
        label = debug.get("selected_label", "n/a")
        q_value = debug.get("selected_q", np.nan)
        epsilon = debug.get("epsilon", np.nan)
        num_candidates = debug.get("num_candidates", 0)
        print(
            "[Q-switch] "
            f"step={self.num_timesteps} selected={label} "
            f"q={q_value} eps={epsilon} candidates={num_candidates}"
        )
        return True
