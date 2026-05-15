from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections import deque

import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.utils import obs_as_tensor

from .primitives import PrimitiveCandidate, PrimitiveEnsemble
from .utils import call_vec_env_method, first_env_from_vec, unwrap_env

SHORT_LABELS = {
    "primitive_grasp": "pg",
    "primitive_insert": "pi",
    "target_policy": "tp",
    "random_action": "rand",
    "zero_action": "zero",
}


def _short_label(label: str) -> str:
    return SHORT_LABELS.get(str(label), str(label).replace("/", "_").replace(" ", "_"))


@dataclass
class QSwitchConfig:
    enabled: bool = True
    include_target_during_warmup: bool = False
    include_target_after_learning_starts: bool = True
    include_zero_action: bool = False
    epsilon_initial: float = 0.20
    epsilon_final: float = 0.02
    epsilon_decay_steps: int = 300_000
    epsilon_random_action: bool = False
    q_aggregation: str = "min"
    debug_to_env: bool = True
    target_candidate_starts: int = 300_000
    target_phase_gate: bool = True
    target_min_lift_height: float = 0.035
    target_q_margin: float = 1.0
    q_value_abs_limit: float = 5_000.0
    use_target_critic_for_selection: bool = True
    target_include_prob_final: float = 0.30
    target_include_prob_ramp_steps: int = 700_000

    # behavior cloning from primitive/scripted teacher
    bc_enabled: bool = True
    bc_buffer_size: int = 200_000
    bc_batch_size: int = 256
    bc_gradient_steps: int = 1
    bc_start_steps: int = 10_000
    bc_initial_coef: float = 1.0
    bc_final_coef: float = 0.05
    bc_decay_steps: int = 800_000
    bc_max_grad_norm: float = 10.0


class QSwitchSAC(SAC):
    """SAC with Q-switch action selection during rollout collection.

    At every rollout step this class asks primitive policies for candidate
    6-DoF actions, asks the current SAC actor for its own candidate action, and
    executes the candidate with the highest value according to the current SAC
    critic Q(s, a).
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
        self._teacher_obs_buffer = deque(maxlen=int(self.qswitch_config.bc_buffer_size))
        self._teacher_action_buffer = deque(
            maxlen=int(self.qswitch_config.bc_buffer_size)
        )
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
        progress = min(
            1.0, float(self.num_timesteps) / float(config.epsilon_decay_steps)
        )
        return float(
            config.epsilon_initial
            + progress * (config.epsilon_final - config.epsilon_initial)
        )

    def _target_include_probability(self, learning_starts: int) -> float:
        config = self.qswitch_config

        if self.num_timesteps < learning_starts:
            return 0.0

        if self.num_timesteps < config.target_candidate_starts:
            return 0.0

        ramp_steps = max(1, int(config.target_include_prob_ramp_steps))
        progress = min(
            1.0,
            float(self.num_timesteps - config.target_candidate_starts)
            / float(ramp_steps),
        )
        return float(config.target_include_prob_final * progress)

    def _should_include_target(self, learning_starts: int) -> bool:
        config = self.qswitch_config

        if self.num_timesteps < learning_starts:
            return bool(config.include_target_during_warmup)

        if not config.include_target_after_learning_starts:
            return False

        prob = self._target_include_probability(learning_starts)
        return bool(self._qswitch_rng.random() < prob)

    @staticmethod
    def _repeat_obs_for_candidates(obs: Any, n_candidates: int) -> Any:
        """Build a batch where the same state is paired with every action candidate."""
        if isinstance(obs, dict):
            return {
                key: np.repeat(
                    np.asarray(value, dtype=np.float32).reshape(1, -1),
                    n_candidates,
                    axis=0,
                )
                for key, value in obs.items()
            }
        return np.repeat(
            np.asarray(obs, dtype=np.float32).reshape(1, -1),
            n_candidates,
            axis=0,
        )

    @staticmethod
    def _extract_single_obs(last_obs: Any) -> Any:
        if isinstance(last_obs, dict):
            return {
                key: np.asarray(value, dtype=np.float32)[0]
                for key, value in last_obs.items()
            }
        return np.asarray(last_obs[0], dtype=np.float32)

    @staticmethod
    def _primitive_obs_from_single_obs(single_obs: Any) -> np.ndarray:
        if isinstance(single_obs, dict):
            return np.asarray(single_obs["observation"], dtype=np.float32).reshape(-1)
        return np.asarray(single_obs, dtype=np.float32).reshape(-1)

    @staticmethod
    def _candidate_has_label(candidate: PrimitiveCandidate, label: str) -> bool:
        candidate_label = str(candidate.label)
        return candidate_label == label or candidate_label.startswith(f"{label}:")

    def _phase_filter_primitive_candidates(
        self,
        *,
        candidates: list[PrimitiveCandidate],
        target_env: Any,
    ) -> tuple[list[PrimitiveCandidate], dict[str, Any]]:
        """Filter primitive candidates based on the manual gripper phase.

        This prevents the selector from comparing semantically invalid actions,
        especially during warmup when the critic is still random.
        """
        if not candidates:
            return candidates, {
                "phase_filter_enabled": 1,
                "phase_filter_reason": "no_candidates",
                "phase_filter_phase": "unknown",
                "phase_filter_lift_height": 0.0,
            }

        env = unwrap_env(target_env)
        phase = str(getattr(env, "gripper_phase", "open"))

        lift_height = 0.0
        try:
            metrics = env._task_metrics()
            lift_height = float(metrics.get("lift_height", 0.0))
        except Exception:
            lift_height = 0.0

        min_lift_height = float(self.qswitch_config.target_min_lift_height)

        grasp_candidates = [
            c for c in candidates if self._candidate_has_label(c, "primitive_grasp")
        ]
        insert_candidates = [
            c for c in candidates if self._candidate_has_label(c, "primitive_insert")
        ]

        reason = "pass_through"

        if phase == "open":
            filtered = grasp_candidates
            reason = "open_use_grasp"

        elif phase == "closed" and lift_height < min_lift_height:
            lift_action = np.zeros_like(candidates[0].action, dtype=np.float32).reshape(
                -1
            )
            lift_action[2] = 0.1

            filtered = [
                PrimitiveCandidate(
                    label="script_lift",
                    action=lift_action,
                    source="scripted_lift",
                    info={
                        "phase": phase,
                        "lift_h": float(lift_height),
                        "min_lift_h": float(min_lift_height),
                    },
                )
            ]
            reason = "closed_not_lifted_script_lift"

        elif phase == "closed":
            filtered = insert_candidates
            reason = "closed_use_insert"

        elif phase == "released":
            filtered = []
            reason = "released_use_target_fallback"

        else:
            filtered = candidates
            reason = f"unknown_phase_{phase}"

        return filtered, {
            "en": 1,
            "reason": reason,
            "phase": phase,
            "lift_h": float(lift_height),
            "min_lift_h": float(min_lift_height),
            "before": int(len(candidates)),
            "after": int(len(filtered)),
        }

    def _target_candidate_phase_status(
        self,
        *,
        target_env: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Return whether the learned actor may compete with phase primitives.

        The actor is trained to maximize this same critic. If it is allowed to
        compete too early, it can exploit critic overestimation and steal the
        rollout before it has learned the grasp/lift prerequisites. The phase
        gate keeps primitive behavior as the data source until the task state is
        semantically ready for target insertion behavior.
        """
        config = self.qswitch_config
        env = unwrap_env(target_env)
        phase = str(getattr(env, "gripper_phase", "open"))
        lift_height = 0.0
        try:
            metrics = env._task_metrics()
            lift_height = float(metrics.get("lift_height", 0.0))
        except Exception:
            lift_height = 0.0

        if not config.target_phase_gate:
            allowed = True
            reason = "phase_gate_disabled"
        elif phase == "open":
            allowed = False
            reason = "open_use_grasp_primitive"
        elif phase == "closed" and lift_height < config.target_min_lift_height:
            allowed = False
            reason = "closed_wait_for_lift"
        elif phase == "closed":
            allowed = True
            reason = "closed_lifted"
        elif phase == "released":
            allowed = False
            reason = "released_no_target_compete"
        else:
            allowed = False
            reason = f"unknown_phase_{phase}"

        return allowed, {
            "enabled": int(config.target_phase_gate),
            "allowed": int(allowed),
            "reason": reason,
            "phase": phase,
            "lift_h": float(lift_height),
            "min_lift_h": float(config.target_min_lift_height),
        }

    def _candidate_q_values(
        self,
        *,
        obs: Any,
        candidate_actions: np.ndarray,
    ) -> np.ndarray:
        """Score every candidate action with the target policy's learned critic.

        Q-switch compares several actions for the same state `s`:

        - primitive grasp action;
        - primitive insert action;
        - optionally the current SAC actor action;
        - optionally zero/debug action.

        The critic estimates expected future return Q(s, a). The selected action
        is the one with the largest scalar Q score. SAC uses twin critics, so the
        two Q estimates are reduced with `q_aggregation`; `min` is the default
        conservative choice.
        """
        obs_batch = self._repeat_obs_for_candidates(obs, candidate_actions.shape[0])

        # SB3 SAC trains the critic on scaled replay-buffer actions, so convert
        # env-space actions from [-1, 1] Box semantics into critic action space.
        scaled_actions = self.policy.scale_action(candidate_actions)

        obs_tensor = obs_as_tensor(obs_batch, self.device)
        action_tensor = th.as_tensor(scaled_actions, device=self.device).float()

        with th.no_grad():
            # The batch shape is:
            #   obs_tensor      : [num_candidates, obs_dim or dict obs]
            #   action_tensor   : [num_candidates, action_dim]
            # The critic returns Q1 and Q2, each [num_candidates, 1].
            critic = self.policy.critic
            if self.qswitch_config.use_target_critic_for_selection and hasattr(
                self.policy, "critic_target"
            ):
                critic = self.policy.critic_target
            q_values = critic(obs_tensor, action_tensor)
            q_stack = th.cat([q.reshape(-1, 1) for q in q_values], dim=1)
            if self.qswitch_config.q_aggregation == "mean":
                selected_q = q_stack.mean(dim=1)
            elif self.qswitch_config.q_aggregation == "q1":
                selected_q = q_stack[:, 0]
            else:
                selected_q = q_stack.min(dim=1).values
        return selected_q.detach().cpu().numpy().astype(np.float64)

    def _best_non_target_index(
        self,
        *,
        labels: list[str],
        q_values: np.ndarray,
    ) -> int | None:
        candidate_indices = [
            index
            for index, label in enumerate(labels)
            if label != "target_policy" and np.isfinite(q_values[index])
        ]
        if not candidate_indices:
            candidate_indices = [
                index for index, label in enumerate(labels) if label != "target_policy"
            ]
        if not candidate_indices:
            return None
        return max(
            candidate_indices,
            key=lambda index: (
                q_values[index] if np.isfinite(q_values[index]) else -np.inf
            ),
        )

    @staticmethod
    def _copy_obs_for_teacher(obs: Any) -> Any:
        if isinstance(obs, dict):
            return {
                key: np.asarray(value, dtype=np.float32).copy()
                for key, value in obs.items()
            }
        return np.asarray(obs, dtype=np.float32).copy()

    def _store_teacher_example(self, obs: Any, teacher_action: np.ndarray) -> None:
        if not self.qswitch_config.bc_enabled:
            return

        teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(1, -1)
        teacher_action = np.clip(
            teacher_action,
            self.action_space.low,
            self.action_space.high,
        )

        # SAC actor/critic memakai scaled action di replay/training space.
        teacher_buffer_action = self.policy.scale_action(teacher_action)[0]

        self._teacher_obs_buffer.append(self._copy_obs_for_teacher(obs))
        self._teacher_action_buffer.append(
            np.asarray(teacher_buffer_action, dtype=np.float32).copy()
        )

    def teacher_buffer_size(self) -> int:
        return int(len(self._teacher_action_buffer))

    def bc_coef(self) -> float:
        config = self.qswitch_config
        if self.num_timesteps < config.bc_start_steps:
            return 0.0

        progress = min(
            1.0,
            float(self.num_timesteps - config.bc_start_steps)
            / float(max(1, config.bc_decay_steps)),
        )
        return float(
            config.bc_initial_coef
            + progress * (config.bc_final_coef - config.bc_initial_coef)
        )

    def sample_teacher_batch(self, batch_size: int):
        if len(self._teacher_action_buffer) < batch_size:
            return None

        indices = self._qswitch_rng.integers(
            0,
            len(self._teacher_action_buffer),
            size=int(batch_size),
        )

        first_obs = self._teacher_obs_buffer[0]

        if isinstance(first_obs, dict):
            obs_batch = {
                key: np.stack(
                    [self._teacher_obs_buffer[int(i)][key] for i in indices],
                    axis=0,
                ).astype(np.float32)
                for key in first_obs.keys()
            }
        else:
            obs_batch = np.stack(
                [self._teacher_obs_buffer[int(i)] for i in indices],
                axis=0,
            ).astype(np.float32)

        action_batch = np.stack(
            [self._teacher_action_buffer[int(i)] for i in indices],
            axis=0,
        ).astype(np.float32)

        obs_tensor = obs_as_tensor(obs_batch, self.device)
        action_tensor = th.as_tensor(action_batch, device=self.device).float()
        return obs_tensor, action_tensor

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

        obs = self._extract_single_obs(self._last_obs)
        primitive_obs = self._primitive_obs_from_single_obs(obs)
        action_shape = tuple(int(v) for v in self.action_space.shape)
        target_env = first_env_from_vec(self.env)
        candidates: list[PrimitiveCandidate] = []

        primitive_candidates = self.primitive_ensemble.candidate_actions(
            current_obs=primitive_obs,
            target_env=target_env,
            expected_action_shape=action_shape,
        )

        real_env = getattr(target_env, "unwrapped", target_env)
        try:
            metrics = real_env._task_metrics()
        except Exception:
            metrics = {}

        # print(
        #     "[QSWITCH CANDIDATE DEBUG] "
        #     f"step={self.num_timesteps} "
        #     f"phase={getattr(real_env, 'gripper_phase', 'n/a')} "
        #     f"gripper_state={getattr(real_env, 'gripper_state', 'n/a')} "
        #     f"lift_h={metrics.get('lift_height', 'n/a')} "
        #     f"ee_obj_dist={metrics.get('ee_obj_dist', 'n/a')} "
        #     f"primitive_labels={[c.label for c in primitive_candidates]} "
        #     f"primitive_sources={[c.source for c in primitive_candidates]} "
        #     f"primitive_errors={self.primitive_ensemble.last_errors}",
        #     flush=True,
        # )

        primitive_labels_before_phase_filter = [
            candidate.label for candidate in primitive_candidates
        ]

        primitive_candidates, phase_filter_debug = (
            self._phase_filter_primitive_candidates(
                candidates=primitive_candidates,
                target_env=target_env,
            )
        )

        primitive_labels_after_phase_filter = [
            candidate.label for candidate in primitive_candidates
        ]

        # print(
        #     "[PHASE FILTER DEBUG] "
        #     f"step={self.num_timesteps} "
        #     f"before={primitive_labels_before_phase_filter} "
        #     f"after={primitive_labels_after_phase_filter} "
        #     f"pf={phase_filter_debug}",
        #     flush=True,
        # )

        candidates.extend(primitive_candidates)

        # After warmup, include the actor from the target SAC policy as another
        # candidate. During warmup the selector can be forced to rely only on
        # primitives unless no primitive candidate is available.
        target_step_allowed = self._should_include_target(learning_starts)
        target_phase_allowed, target_phase_debug = self._target_candidate_phase_status(
            target_env=target_env
        )
        include_target = target_step_allowed and target_phase_allowed
        target_forced_fallback = not candidates
        if include_target or target_forced_fallback:
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

        actions = np.asarray(
            [candidate.action for candidate in candidates], dtype=np.float32
        )
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        labels = [candidate.label for candidate in candidates]
        epsilon = self._current_epsilon()

        selected_index: int
        q_values: np.ndarray
        selection_mode = "critic"
        if self._qswitch_rng.random() < epsilon:
            # Exploration deliberately bypasses critic ranking, so logged Q
            # values are NaN for this step.
            selection_mode = "epsilon_random"
            if config.epsilon_random_action:
                selected_action = np.asarray(
                    self.action_space.sample(), dtype=np.float32
                ).reshape(-1)
                selected_index = -1
                q_values = np.full(actions.shape[0], np.nan, dtype=np.float64)
                selected_label = "random_action"
            else:
                selected_index = int(self._qswitch_rng.integers(0, len(candidates)))
                selected_action = actions[selected_index]
                q_values = np.full(actions.shape[0], np.nan, dtype=np.float64)
                selected_label = labels[selected_index]
        else:
            # Normal Q-switch path: score every candidate under the same state,
            # then execute the label/action with maximal predicted return.
            q_values = self._candidate_q_values(obs=obs, candidate_actions=actions)
            finite_q = np.isfinite(q_values)
            q_abs_max = (
                float(np.max(np.abs(q_values[finite_q])))
                if np.any(finite_q)
                else np.inf
            )
            q_unstable = bool(
                not np.all(finite_q) or q_abs_max > config.q_value_abs_limit
            )
            if np.any(finite_q):
                ranked_q_values = np.where(finite_q, q_values, -np.inf)
                selected_index = int(np.argmax(ranked_q_values))
            else:
                selected_index = 0
                selection_mode = "critic_all_nonfinite_fallback"

            selected_label = labels[selected_index]
            if selected_label == "target_policy":
                non_target_index = self._best_non_target_index(
                    labels=labels,
                    q_values=q_values,
                )
                if q_unstable and non_target_index is not None:
                    selected_index = int(non_target_index)
                    selection_mode = "critic_unstable_primitive_fallback"
                elif (
                    non_target_index is not None
                    and config.target_q_margin > 0.0
                    and np.isfinite(q_values[non_target_index])
                    and q_values[selected_index]
                    < q_values[non_target_index] + config.target_q_margin
                ):
                    selected_index = int(non_target_index)
                    selection_mode = "target_margin_primitive_fallback"
            selected_action = actions[selected_index]
            selected_label = labels[selected_index]

        # setelah proses critic/epsilon selesai
        teacher_index = self._best_non_target_index(
            labels=labels,
            q_values=q_values,
        )
        teacher_label = None
        target_q_gap = np.nan

        if teacher_index is not None:
            teacher_label = labels[teacher_index]
            self._store_teacher_example(obs, actions[teacher_index])

        target_index = None
        for idx, label in enumerate(labels):
            if label == "target_policy":
                target_index = idx
                break

        if (
            target_index is not None
            and teacher_index is not None
            and np.isfinite(q_values[target_index])
            and np.isfinite(q_values[teacher_index])
        ):
            target_q_gap = float(q_values[target_index] - q_values[teacher_index])

        selected_action = np.asarray(selected_action, dtype=np.float32).reshape(1, -1)
        selected_buffer_action = self.policy.scale_action(selected_action)

        selected_q = (
            float(q_values[selected_index])
            if 0 <= selected_index < len(q_values)
            and np.isfinite(q_values[selected_index])
            else np.nan
        )
        debug = {
            "enabled": int(config.enabled),
            "num_candidates": int(len(candidates)),
            "candidate_labels": labels,
            "candidate_q_values": q_values.tolist(),
            "candidate_sources": [candidate.source for candidate in candidates],
            "primitive_errors": list(self.primitive_ensemble.last_errors),
            "prim_before_pf": primitive_labels_before_phase_filter,
            "prim_after_pf": primitive_labels_after_phase_filter,
            "pf": phase_filter_debug,
            "tp_phase": target_phase_debug,
            "selected_index": int(selected_index),
            "selected_label": selected_label,
            "selected_q": selected_q,
            "epsilon": float(epsilon),
            "selection_mode": selection_mode,
            "num_timesteps": int(self.num_timesteps),
            "learning_starts": int(learning_starts),
            "target_step_allowed": int(target_step_allowed),
            "target_phase_allowed": int(target_phase_allowed),
            "target_forced_fallback": int(target_forced_fallback),
            "target_included": int("target_policy" in labels),
            "q_abs_max": float(
                np.nanmax(np.abs(q_values)) if np.any(np.isfinite(q_values)) else np.inf
            ),
            "q_unstable": int(
                (not np.all(np.isfinite(q_values)))
                or (
                    np.any(np.isfinite(q_values))
                    and np.nanmax(np.abs(q_values)) > config.q_value_abs_limit
                )
            ),
            "q_value_abs_limit": float(config.q_value_abs_limit),
            "target_q_margin": float(config.target_q_margin),
            "target_include_prob": float(
                self._target_include_probability(learning_starts)
            ),
            "teacher_label": teacher_label if teacher_label is not None else "none",
            "teacher_buffer_size": int(self.teacher_buffer_size()),
            "target_q_gap": float(target_q_gap),
            "bc_coef": float(self.bc_coef()),
            "target_include_prob": float(
                self._target_include_probability(learning_starts)
            ),
        }
        for label in sorted(
            set(labels + [selected_label, "random_action", "target_policy"])
        ):
            metric_label = _short_label(label)
            debug[f"cand_{metric_label}"] = int(label in labels)
            debug[f"sel_{metric_label}"] = int(label == selected_label)
        self._push_debug(debug)

        # print(f"[Q-SWITCH] Selected: {selected_label} (mode: {selection_mode})")

        return selected_action, selected_buffer_action
