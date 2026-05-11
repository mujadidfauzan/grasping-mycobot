from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import obs_as_tensor

from .primitives import PrimitiveCandidate, PrimitiveEnsemble
from .utils import call_vec_env_method, first_env_from_vec


@dataclass
class QSwitchConfig:
    enabled: bool = True
    include_target_during_warmup: bool = False
    include_target_after_learning_starts: bool = True
    force_primitives_during_warmup: bool = True
    include_zero_action: bool = False
    epsilon_initial: float = 0.20
    epsilon_final: float = 0.02
    epsilon_decay_steps: int = 100_000
    epsilon_random_action: bool = True
    q_aggregation: str = "min"
    debug_to_env: bool = True


class QSwitchSAC(SAC):
    """SAC with Q-switch action selection during rollout collection.

    At every rollout step this class asks primitive policies for candidate
    6-DoF actions, asks the current SAC actor for its own candidate action, and
    executes the candidate with the highest target critic value Q(s, a).
    """

    def __init__(
        self,
        *args: Any,
        primitive_ensemble: PrimitiveEnsemble | None = None,
        qswitch_config: QSwitchConfig | None = None,
        **kwargs: Any,
    ):
        self.primitive_ensemble = primitive_ensemble
        self.qswitch_config = qswitch_config or QSwitchConfig()
        self._qswitch_rng = np.random.default_rng()
        self.last_qswitch_debug: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        return excluded + ["primitive_ensemble", "_qswitch_rng"]

    def set_qswitch_ensemble(
        self,
        primitive_ensemble: PrimitiveEnsemble | None,
        qswitch_config: QSwitchConfig | None = None,
    ) -> None:
        self.primitive_ensemble = primitive_ensemble
        if qswitch_config is not None:
            self.qswitch_config = qswitch_config

    def _current_epsilon(self) -> float:
        config = self.qswitch_config
        if config.epsilon_decay_steps <= 0:
            return float(config.epsilon_final)
        progress = min(1.0, float(self.num_timesteps) / float(config.epsilon_decay_steps))
        return float(
            config.epsilon_initial
            + progress * (config.epsilon_final - config.epsilon_initial)
        )

    def _should_include_target(self, learning_starts: int) -> bool:
        config = self.qswitch_config
        if self.num_timesteps < learning_starts:
            return bool(config.include_target_during_warmup)
        return bool(config.include_target_after_learning_starts)

    def _candidate_q_values(
        self,
        *,
        obs: np.ndarray,
        candidate_actions: np.ndarray,
    ) -> np.ndarray:
        obs_batch = np.repeat(np.asarray(obs).reshape(1, -1), candidate_actions.shape[0], axis=0)
        scaled_actions = self.policy.scale_action(candidate_actions)

        obs_tensor = obs_as_tensor(obs_batch, self.device)
        action_tensor = th.as_tensor(scaled_actions, device=self.device).float()

        with th.no_grad():
            q_values = self.policy.critic(obs_tensor, action_tensor)
            q_stack = th.cat([q.reshape(-1, 1) for q in q_values], dim=1)
            if self.qswitch_config.q_aggregation == "mean":
                selected_q = q_stack.mean(dim=1)
            elif self.qswitch_config.q_aggregation == "q1":
                selected_q = q_stack[:, 0]
            else:
                selected_q = q_stack.min(dim=1).values
        return selected_q.detach().cpu().numpy().astype(np.float64)

    def _push_debug(self, debug: dict[str, Any]) -> None:
        self.last_qswitch_debug = debug
        if self.qswitch_config.debug_to_env and self.env is not None:
            call_vec_env_method(self.env, "set_qswitch_debug", debug)

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        parent_action, parent_buffer_action = super()._sample_action(
            learning_starts,
            action_noise,
            n_envs,
        )

        config = self.qswitch_config
        if (
            not config.enabled
            or self.primitive_ensemble is None
            or self._last_obs is None
            or n_envs != 1
        ):
            return parent_action, parent_buffer_action

        obs = np.asarray(self._last_obs[0], dtype=np.float32)
        action_shape = tuple(int(v) for v in self.action_space.shape)
        target_env = first_env_from_vec(self.env)
        candidates: list[PrimitiveCandidate] = []

        primitive_candidates = self.primitive_ensemble.candidate_actions(
            current_obs=obs,
            target_env=target_env,
            expected_action_shape=action_shape,
        )
        candidates.extend(primitive_candidates)

        include_target = self._should_include_target(learning_starts)
        if include_target or not candidates:
            candidates.append(
                PrimitiveCandidate(
                    label="target_policy",
                    action=np.asarray(parent_action[0], dtype=np.float32).reshape(-1),
                    source="target_policy",
                )
            )

        if config.include_zero_action:
            candidates.append(
                PrimitiveCandidate(
                    label="zero_action",
                    action=np.zeros(action_shape, dtype=np.float32),
                    source="debug",
                )
            )

        if not candidates:
            return parent_action, parent_buffer_action

        actions = np.asarray([candidate.action for candidate in candidates], dtype=np.float32)
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        labels = [candidate.label for candidate in candidates]
        epsilon = self._current_epsilon()

        selected_index: int
        q_values: np.ndarray
        selection_mode = "critic"
        if self._qswitch_rng.random() < epsilon:
            selection_mode = "epsilon_random"
            if config.epsilon_random_action:
                selected_action = np.asarray(self.action_space.sample(), dtype=np.float32).reshape(-1)
                selected_index = -1
                q_values = np.full(actions.shape[0], np.nan, dtype=np.float64)
                selected_label = "random_action"
            else:
                selected_index = int(self._qswitch_rng.integers(0, len(candidates)))
                selected_action = actions[selected_index]
                q_values = np.full(actions.shape[0], np.nan, dtype=np.float64)
                selected_label = labels[selected_index]
        else:
            q_values = self._candidate_q_values(obs=obs, candidate_actions=actions)
            selected_index = int(np.argmax(q_values))
            selected_action = actions[selected_index]
            selected_label = labels[selected_index]

        selected_action = np.asarray(selected_action, dtype=np.float32).reshape(1, -1)
        selected_buffer_action = self.policy.scale_action(selected_action)

        selected_q = (
            float(q_values[selected_index])
            if 0 <= selected_index < len(q_values) and np.isfinite(q_values[selected_index])
            else np.nan
        )
        debug = {
            "enabled": int(config.enabled),
            "num_candidates": int(len(candidates)),
            "candidate_labels": labels,
            "candidate_q_values": q_values.tolist(),
            "candidate_sources": [candidate.source for candidate in candidates],
            "primitive_errors": list(self.primitive_ensemble.last_errors),
            "selected_index": int(selected_index),
            "selected_label": selected_label,
            "selected_q": selected_q,
            "epsilon": float(epsilon),
            "selection_mode": selection_mode,
            "num_timesteps": int(self.num_timesteps),
            "learning_starts": int(learning_starts),
            "target_included": int(include_target),
        }
        self._push_debug(debug)

        return selected_action, selected_buffer_action
