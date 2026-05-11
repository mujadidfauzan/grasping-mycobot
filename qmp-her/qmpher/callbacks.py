from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .utils import coerce_scalar, flatten_numeric_info

KNOWN_SELECTION_LABELS = (
    "primitive_grasp",
    "primitive_insert",
    "target_policy",
    "random_action",
    "zero_action",
)

SHORT_LABELS = {
    "primitive_grasp": "pg",
    "primitive_insert": "pi",
    "target_policy": "tp",
    "random_action": "rand",
    "zero_action": "zero",
}


def _safe_metric_label(label: str) -> str:
    return SHORT_LABELS.get(str(label), str(label).replace("/", "_").replace(" ", "_"))


class QSwitchInfoCallback(BaseCallback):
    """Log q-switch and manual gripper metrics from env info dicts."""

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
            flat = flatten_numeric_info(info)
            qswitch = info.get("qswitch")
            if isinstance(qswitch, Mapping):
                flat.update(
                    {
                        f"qswitch/{key}": value
                        for key, value in flatten_numeric_info(qswitch).items()
                    }
                )
                selected_label = qswitch.get("selected_label")
                candidate_labels = qswitch.get("candidate_labels", [])
                label_set = set(KNOWN_SELECTION_LABELS)
                if isinstance(candidate_labels, (list, tuple)):
                    label_set.update(str(label) for label in candidate_labels)
                if isinstance(selected_label, str):
                    label_set.add(selected_label)

                for label in sorted(label_set):
                    safe_label = _safe_metric_label(label)
                    flat[f"qswitch/sel_rate/{safe_label}"] = float(
                        isinstance(selected_label, str) and selected_label == label
                    )
                    flat[f"qswitch/cand_rate/{safe_label}"] = float(
                        isinstance(candidate_labels, (list, tuple))
                        and label in candidate_labels
                    )

            for key, value in flat.items():
                scalar = coerce_scalar(value)
                if scalar is not None:
                    metrics.setdefault(key, []).append(scalar)

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
