from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
    ("lift_height", "task/lift_h"),
    ("lift_progress", "task/lift_p"),
    ("target_pose_aligned", "task/aligned"),
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
    ("q_abs_max", "qs/q_abs_max"),
    ("q_unstable", "qs/q_unstable"),
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
                        float(isinstance(selected_label, str) and selected_label == label),
                    )

        for key, values in metrics.items():
            if values:
                self.logger.record(f"custom/{key}", float(np.mean(values)))

        return True


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
