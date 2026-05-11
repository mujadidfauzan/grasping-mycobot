from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


QMPHER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = QMPHER_ROOT.parent


def ensure_project_root_on_path() -> None:
    """Allow scripts inside qmp-her to import the main repository modules."""
    project_root_str = str(PROJECT_ROOT)
    qmp_root_str = str(QMPHER_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if qmp_root_str not in sys.path:
        sys.path.insert(0, qmp_root_str)


def resolve_repo_path(path: str | Path | None, default: Path | None = None) -> Path:
    if path is None:
        if default is None:
            raise ValueError("A path must be provided when no default is available.")
        resolved = default
    else:
        resolved = Path(path).expanduser()

    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    return resolved


def unwrap_env(env: Any) -> Any:
    """Peel common Gym/SB3 wrappers until the underlying env is reached."""
    current = env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "unwrapped"):
            unwrapped = getattr(current, "unwrapped")
            if unwrapped is not current:
                current = unwrapped
                continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return current


def first_env_from_vec(vec_env: Any) -> Any:
    """Return the first real env from DummyVecEnv/VecVideoRecorder-like wrappers."""
    current = vec_env
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "envs") and getattr(current, "envs"):
            return current.envs[0]
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break
    return current


def coerce_scalar(value: Any) -> float | None:
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return None
        value = value.item()
    elif not np.isscalar(value):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(scalar):
        return None
    return scalar


def flatten_numeric_info(info: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in info.items():
        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_numeric_info(value, metric_key))
            continue
        scalar = coerce_scalar(value)
        if scalar is not None:
            flat[metric_key] = scalar
    return flat


def safe_debug_state(env: Any) -> dict[str, Any]:
    getter = getattr(env, "get_debug_state", None)
    if not callable(getter):
        return {}
    try:
        state = getter()
    except Exception:
        return {}
    return dict(state) if isinstance(state, Mapping) else {}


def call_vec_env_method(vec_env: Any, method_name: str, *args: Any, **kwargs: Any) -> None:
    method = getattr(vec_env, "env_method", None)
    if callable(method):
        try:
            method(method_name, *args, **kwargs)
            return
        except Exception:
            return

    first_env = first_env_from_vec(vec_env)
    method = getattr(first_env, method_name, None)
    if callable(method):
        try:
            method(*args, **kwargs)
        except Exception:
            return
