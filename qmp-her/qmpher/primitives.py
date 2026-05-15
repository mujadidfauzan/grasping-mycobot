from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .sync import sync_grasp_env_from_target, sync_insert_env_from_target
from .utils import ensure_project_root_on_path, resolve_repo_path, unwrap_env

ensure_project_root_on_path()

from source.envs.grasping_env_ik import GraspingEnvIK  # noqa: E402
from source.envs.insert_target_env_ik import InsertTargetEnvIK  # noqa: E402


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
    """Candidate action from a trained InsertTargetEnvIK policy.

    A hidden InsertTargetEnvIK is synchronized from the live QMP env before each
    prediction. This keeps the insert policy on its original observation path
    instead of relying on a hand-mirrored QMP observation layout.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        xml_file: str | Path,
        grasp_model_path: str | Path,
        grasp_xml_file: str | Path,
        label: str = "primitive_insert",
        deterministic: bool = True,
        device: str = "auto",
        env_kwargs: dict[str, Any] | None = None,
        min_lift_height: float = 0.035,
        strict: bool = True,
    ):
        self.label = str(label)
        self.xml_file = resolve_repo_path(xml_file)
        self.grasp_model_path = resolve_repo_path(grasp_model_path)
        self.grasp_xml_file = resolve_repo_path(grasp_xml_file)
        self.deterministic = bool(deterministic)
        self.device = device
        self.env_kwargs = dict(env_kwargs or {})
        self.min_lift_height = float(min_lift_height)
        self.strict = bool(strict)
        self._env: InsertTargetEnvIK | None = None
        self._loader: _LazySACModel | None = None
        self._model_path = resolve_repo_path(model_path)

    def _ensure_loaded(self) -> tuple[Any, InsertTargetEnvIK]:
        if self._env is None:
            env_kwargs = {
                "xml_file": str(self.xml_file),
                "grasp_model_path": str(self.grasp_model_path),
                "grasp_xml_file": str(self.grasp_xml_file),
                "render_mode": None,
            }
            env_kwargs.update(self.env_kwargs)
            self._env = InsertTargetEnvIK(**env_kwargs)
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
        env = unwrap_env(target_env)

        # Insert primitive hanya valid setelah object sudah digenggam.
        # Jangan gunakan insert saat gripper masih open/released.
        gripper_phase = getattr(env, "gripper_phase", "open")
        if gripper_phase != "closed":
            return []

        try:
            metrics = env._task_metrics()
            lift_height = float(metrics.get("lift_height", 0.0))
        except Exception:
            lift_height = 0.0

        if lift_height < self.min_lift_height:
            return []

        try:
            model, insert_env = self._ensure_loaded()
            sync_insert_env_from_target(
                target_env=env,
                insert_env=insert_env,
                target_object_name="box",
            )
            obs = insert_env._get_obs()
            action, _ = model.predict(
                obs,
                deterministic=self.deterministic,
            )
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
                    "min_lift_height": self.min_lift_height,
                    "primitive_obs_shape": tuple(int(v) for v in obs.shape),
                    "primitive_env": type(insert_env).__name__,
                },
            )
        ]

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._loader = None


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
