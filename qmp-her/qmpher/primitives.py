from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .sync import sync_grasp_env_from_target
from .utils import ensure_project_root_on_path, resolve_repo_path, unwrap_env

ensure_project_root_on_path()

from source.envs.grasping_env_ik import GraspingEnvIK  # noqa: E402


@dataclass
class PrimitiveCandidate:
    label: str
    action: np.ndarray
    source: str
    info: dict[str, Any] = field(default_factory=dict)


class PrimitiveAdapter(Protocol):
    label: str

    def candidate_actions(
        self,
        *,
        current_obs: np.ndarray,
        target_env: Any,
    ) -> list[PrimitiveCandidate]: ...

    def close(self) -> None: ...


class _LazySACModel:
    def __init__(
        self, model_path: str | Path, *, env: Any | None = None, device: str = "auto"
    ):
        self.model_path = resolve_repo_path(model_path)
        self.env = env
        self.device = device
        self.model = None

    def get(self):
        if self.model is None:
            try:
                from stable_baselines3 import SAC
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "QMP-HER primitives require stable-baselines3. Install the "
                    "same training dependencies used by script/train_sac.py."
                ) from exc
            self.model = SAC.load(
                str(self.model_path), env=self.env, device=self.device
            )
        return self.model


class InsertPrimitiveAdapter:
    """Candidate action from an already trained InsertTargetEnvIK policy."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        label: str = "primitive_insert",
        deterministic: bool = True,
        device: str = "auto",
        strict: bool = True,
    ):
        self.label = str(label)
        self.deterministic = bool(deterministic)
        self.strict = bool(strict)
        self._loader = _LazySACModel(model_path, device=device)

    def candidate_actions(
        self,
        *,
        current_obs: np.ndarray,
        target_env: Any,
    ) -> list[PrimitiveCandidate]:
        env = unwrap_env(target_env)

        # Insert primitive hanya valid setelah object sudah digenggam.
        # Jangan gunakan insert saat gripper masih open/released.
        gripper_phase = getattr(env, "gripper_phase", "open")
        if gripper_phase != "closed":
            return []

        # Insert primitive juga hanya valid kalau object sudah cukup terangkat.
        # Ini mencegah model insert menerima state OOD: object masih di meja.
        # min_lift_height = (
        #     0.035  # 3.5 cm relatif dari initial object z; boleh tuning nanti
        # )

        try:
            metrics = env._task_metrics()
            lift_height = float(metrics.get("lift_height", 0.0))
        except Exception:
            lift_height = 0.0

        # if lift_height < min_lift_height:
        #     return []

        try:
            # print("\n" + "=" * 120, flush=True)
            # print(
            #     "[INSERT OBS DEBUG] "
            #     f"qmp_step={getattr(env, 'current_step', -1)} "
            #     f"phase={getattr(env, 'gripper_phase', 'n/a')} "
            #     f"gripper_state={getattr(env, 'gripper_state', 'n/a')} "
            #     f"current_obs_shape={np.asarray(current_obs).shape}",
            #     flush=True,
            # )

            # try:
            #     metrics = env._task_metrics()
            #     print(
            #         "[INSERT STATE DEBUG] "
            #         f"obj_pos={np.round(metrics['obj_pos'], 5)} "
            #         f"target_pos={np.round(metrics['target_pos'], 5)} "
            #         f"obj_target_err={np.round(metrics['obj_target_pos_error'], 5)} "
            #         f"target_dist={float(metrics['target_dist']):.6f} "
            #         f"target_angle_deg={np.rad2deg(float(metrics['target_angle'])):.3f} "
            #         f"ee_obj_dist={float(metrics['ee_obj_dist']):.6f} "
            #         f"lift_h={float(metrics['lift_height']):.6f} "
            #         f"aligned={int(metrics['target_pose_aligned'])}",
            #         flush=True,
            #     )
            # except Exception as e:
            #     print(f"[INSERT STATE DEBUG ERROR] {repr(e)}", flush=True)

            # try:
            #     start_idx = 0
            #     for name, component in env._get_obs_components():
            #         arr = np.asarray(component, dtype=np.float64).reshape(-1)
            #         end_idx = start_idx + arr.size

            #         print(
            #             f"[INSERT OBS] idx={start_idx}:{end_idx} "
            #             f"name={name} "
            #             f"shape={arr.shape} "
            #             f"value={np.array2string(arr, precision=5, suppress_small=True, threshold=np.inf, max_line_width=220)}",
            #             flush=True,
            #         )

            #         start_idx = end_idx

            #     print(
            #         f"[INSERT OBS CHECK] "
            #         f"component_total_len={start_idx} "
            #         f"current_obs_len={np.asarray(current_obs).reshape(-1).shape[0]}",
            #         flush=True,
            #     )
            # except Exception as e:
            #     print(f"[INSERT OBS COMPONENT DEBUG ERROR] {repr(e)}", flush=True)

            # print("=" * 120 + "\n", flush=True)
            model = self._loader.get()
            action, _ = model.predict(
                np.asarray(current_obs, dtype=np.float32),
                deterministic=self.deterministic,
            )
            # print(
            #     "[INSERT ACTION DEBUG] "
            #     f"action={np.array2string(np.asarray(action).reshape(-1), precision=5, suppress_small=True)}",
            #     flush=True,
            # )
        except Exception as exc:
            if self.strict:
                raise
            return [
                PrimitiveCandidate(
                    label=f"{self.label}:error",
                    action=np.zeros(0, dtype=np.float32),
                    source=self.label,
                    info={"error": repr(exc)},
                )
            ]

        return [
            PrimitiveCandidate(
                label=self.label,
                action=np.asarray(action, dtype=np.float32).reshape(-1),
                source="insert_policy",
                info={
                    "gripper_phase": gripper_phase,
                    "lift_height": lift_height,
                    # "min_lift_height": min_lift_height,
                },
            )
        ]

    def close(self) -> None:
        pass


class GraspPrimitiveAdapter:
    """Candidate action from a trained GraspingEnvIK policy.

    A hidden GraspingEnvIK is synchronized from the live target env before each
    prediction. This avoids brittle hand-built observation slicing and keeps the
    primitive observation contract identical to its training environment.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        xml_file: str | Path,
        label: str = "primitive_grasp",
        deterministic: bool = True,
        device: str = "auto",
        env_kwargs: dict[str, Any] | None = None,
        strict: bool = True,
    ):
        self.label = str(label)
        self.xml_file = resolve_repo_path(xml_file)
        self.deterministic = bool(deterministic)
        self.device = device
        self.env_kwargs = dict(env_kwargs or {})
        self.strict = bool(strict)
        self._env: GraspingEnvIK | None = None
        self._loader: _LazySACModel | None = None
        self._model_path = resolve_repo_path(model_path)

    def _ensure_loaded(self) -> tuple[Any, GraspingEnvIK]:
        if self._env is None:
            env_kwargs = {
                "xml_file": str(self.xml_file),
                "render_mode": None,
            }
            env_kwargs.update(self.env_kwargs)
            self._env = GraspingEnvIK(**env_kwargs)
            self._loader = _LazySACModel(
                self._model_path,
                env=self._env,
                device=self.device,
            )
        assert self._loader is not None
        return self._loader.get(), self._env

    def candidate_actions(
        self,
        *,
        current_obs: np.ndarray,
        target_env: Any,
    ) -> list[PrimitiveCandidate]:
        del current_obs
        try:
            model, grasp_env = self._ensure_loaded()
            # Menyamakan state dari target_env ke grasp_env dengan cara yang sama seperti saat training.
            sync_grasp_env_from_target(
                target_env=unwrap_env(target_env),
                grasp_env=grasp_env,
                target_object_name="box",
            )
            obs = grasp_env._get_obs()
            # ===================== DEBUG OBS GRASPING ENV =====================
            # env = unwrap_env(target_env)

            # print("\n" + "=" * 120, flush=True)
            # print(
            #     f"[GRASP OBS COMPONENT DEBUG] "
            #     f"qmp_step={getattr(env, 'current_step', -1)} "
            #     f"qmp_phase={getattr(env, 'gripper_phase', 'n/a')} "
            #     f"obs_shape={np.asarray(obs).shape}",
            #     flush=True,
            # )

            # start_idx = 0
            # components = grasp_env._get_obs_components()

            # for name, component in components:
            #     arr = np.asarray(component, dtype=np.float64).reshape(-1)
            #     end_idx = start_idx + arr.size

            #     print(
            #         f"[OBS] idx={start_idx}:{end_idx} "
            #         f"name={name} "
            #         f"shape={arr.shape} "
            #         f"value={np.array2string(arr, precision=5, suppress_small=True, threshold=np.inf, max_line_width=200)}",
            #         flush=True,
            #     )

            #     start_idx = end_idx

            # print(
            #     f"[OBS CHECK] total_component_len={start_idx} "
            #     f"obs_len={np.asarray(obs).reshape(-1).shape[0]}",
            #     flush=True,
            # )

            # print("=" * 120 + "\n", flush=True)
            # # =================== END DEBUG OBS GRASPING ENV ===================

            # print("\n[TEMPORAL OBS DEBUG]", flush=True)
            # print(f"qmp_step={getattr(env, 'current_step', -1)}", flush=True)

            # print(
            #     "QMP last_action      =",
            #     np.round(
            #         np.asarray(
            #             getattr(env, "last_action", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            # print(
            #     "GRASP last_action    =",
            #     np.round(
            #         np.asarray(
            #             getattr(grasp_env, "last_action", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            # print(
            #     "QMP ik_target_pos    =",
            #     np.round(
            #         np.asarray(
            #             getattr(env, "_ik_target_pos", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            # print(
            #     "GRASP ik_target_pos  =",
            #     np.round(
            #         np.asarray(
            #             getattr(grasp_env, "_ik_target_pos", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            # print(
            #     "QMP ik_target_quat   =",
            #     np.round(
            #         np.asarray(
            #             getattr(env, "_ik_target_quat", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            # print(
            #     "GRASP ik_target_quat =",
            #     np.round(
            #         np.asarray(
            #             getattr(grasp_env, "_ik_target_quat", []), dtype=np.float64
            #         ).reshape(-1),
            #         5,
            #     ),
            #     flush=True,
            # )

            action, _ = model.predict(obs, deterministic=self.deterministic)

        except Exception as exc:
            if self.strict:
                raise
            return [
                PrimitiveCandidate(
                    label=f"{self.label}:error",
                    action=np.zeros(0, dtype=np.float32),
                    source=self.label,
                    info={"error": repr(exc)},
                )
            ]
        return [
            PrimitiveCandidate(
                label=self.label,
                action=np.asarray(action, dtype=np.float32).reshape(-1),
                source="grasp_policy",
                info={
                    "primitive_obs_shape": tuple(int(v) for v in obs.shape),
                    "primitive_env": type(grasp_env).__name__,
                },
            )
        ]

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._loader = None


class PrimitiveEnsemble:
    def __init__(self, primitives: list[PrimitiveAdapter]):
        if not primitives:
            raise ValueError("PrimitiveEnsemble requires at least one primitive.")
        self.primitives = list(primitives)
        self.last_errors: list[dict[str, str]] = []

    def candidate_actions(
        self,
        *,
        current_obs: np.ndarray,
        target_env: Any,
        expected_action_shape: tuple[int, ...],
    ) -> list[PrimitiveCandidate]:
        candidates: list[PrimitiveCandidate] = []
        errors: list[dict[str, str]] = []
        for primitive in self.primitives:
            try:
                primitive_candidates = primitive.candidate_actions(
                    current_obs=current_obs,
                    target_env=target_env,
                )
            except Exception as exc:
                errors.append({"primitive": primitive.label, "error": repr(exc)})
                continue

            for candidate in primitive_candidates:
                action = np.asarray(candidate.action, dtype=np.float32).reshape(-1)
                if action.shape != expected_action_shape:
                    errors.append(
                        {
                            "primitive": candidate.label,
                            "error": (
                                f"action_shape={action.shape} does not match "
                                f"expected={expected_action_shape}"
                            ),
                        }
                    )
                    continue
                candidate.action = action
                candidates.append(candidate)

        self.last_errors = errors
        return candidates

    def close(self) -> None:
        for primitive in self.primitives:
            primitive.close()
