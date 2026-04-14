from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OBJECT_LIFT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
OBJECT_PLACE_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
DEFAULT_EVAL_LOG_DIR = PROJECT_ROOT / "logs_eval"


def parse_args() -> argparse.Namespace:
    env_names = resolve_env_names()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SAC policy and export env debug state to CSV."
    )
    parser.add_argument(
        "--env",
        choices=env_names,
        default="GraspingEnv",
        help="Environment name.",
    )
    parser.add_argument(
        "--xml-file",
        default=None,
        help="Path to MuJoCo XML model. Defaults to the scene associated with --env.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .zip model. If omitted, picks the newest .zip under logs/models/<env>/.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per episode (defaults to env.max_episode_steps if available).",
    )
    parser.add_argument(
        "--render",
        choices=["none", "human", "rgb_array"],
        default="human",
        help="Render mode for the environment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for env.reset().",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (deterministic=False).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVAL_LOG_DIR),
        help="Directory where evaluation CSV files will be saved.",
    )
    parser.add_argument(
        "--grasp-model",
        default=None,
        help="For PlaceTargetEnv: path to the trained grasping SAC .zip used to generate reset states.",
    )
    parser.add_argument(
        "--grasp-env",
        default="GraspingEnvV2",
        help="For PlaceTargetEnv: grasping environment class name used by --grasp-model.",
    )
    parser.add_argument(
        "--grasp-xml-file",
        default=None,
        help="For PlaceTargetEnv: XML scene used by the grasping policy. Defaults to object_lift.xml.",
    )
    parser.add_argument(
        "--grasp-max-steps",
        type=int,
        default=300,
        help="For PlaceTargetEnv: max rollout steps per grasp-policy reset attempt.",
    )
    parser.add_argument(
        "--grasp-attempts",
        type=int,
        default=6,
        help="For PlaceTargetEnv: how many grasp-policy reset attempts to try before falling back to the best snapshot.",
    )
    parser.add_argument(
        "--grasp-min-lift",
        type=float,
        default=0.025,
        help="For PlaceTargetEnv: minimum object lift height required before a grasp snapshot is accepted.",
    )
    parser.add_argument(
        "--grasp-ee-obj-dist",
        type=float,
        default=0.035,
        help="For PlaceTargetEnv: max EE-object distance allowed for a grasp snapshot.",
    )
    parser.add_argument(
        "--grasp-hold-steps",
        type=int,
        default=3,
        help="For PlaceTargetEnv: required consecutive valid grasp steps before transferring the state.",
    )
    parser.add_argument(
        "--terminate-ee-obj-dist",
        type=float,
        default=0.08,
        help="For PlaceTargetEnv: terminate the episode when the object is this far or farther from the EE.",
    )
    return parser.parse_args()


def resolve_env_names() -> list[str]:
    try:
        from source.envs import GraspingEnvV3
    except ModuleNotFoundError:
        GraspingEnvV3 = None

    env_names = [
        "GraspingEnv",
        "GraspingEnvV1",
        "GraspingEnvV2",
        "PlaceTargetEnv",
        "ReachingEnv",
    ]
    if GraspingEnvV3 is not None:
        env_names.append("GraspingEnvV3")
    return sorted(env_names)


def resolve_default_xml_path(env_name: str) -> Path:
    if env_name == "PlaceTargetEnv":
        return OBJECT_PLACE_XML_PATH
    return OBJECT_LIFT_XML_PATH


def resolve_xml_path(env_name: str, xml_file_arg: str | None) -> Path:
    if xml_file_arg is None:
        xml_path = resolve_default_xml_path(env_name)
    else:
        xml_path = Path(xml_file_arg).expanduser()
        if not xml_path.is_absolute():
            xml_path = (PROJECT_ROOT / xml_path).resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return xml_path


def resolve_latest_model(env_name: str) -> Path:
    models_root = PROJECT_ROOT / "logs" / "models" / env_name
    if not models_root.exists():
        raise FileNotFoundError(
            f"Model folder not found: {models_root}. Pass --model explicitly."
        )
    candidates = sorted(models_root.glob("**/*.zip"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No .zip model found under {models_root}. Pass --model explicitly."
        )
    return candidates[-1]


def sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "evaluation"


def resolve_output_csv_path(
    output_dir_arg: str,
    env_name: str,
    model_path: Path,
) -> Path:
    output_dir = Path(output_dir_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model_label = sanitize_filename_part(model_path.stem)
    return output_dir / env_name / f"{timestamp}_{model_label}_debug_state.csv"


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    formatted = f"{value:.8f}".rstrip("0").rstrip(".")
    if formatted in {"", "-0"}:
        return "0"
    return formatted


def normalize_csv_value(value: Any) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    if np is not None and isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return format_float(value)
    if value is None:
        return ""
    return str(value)


def infer_column_suffixes(name: str, size: int) -> list[str]:
    if size == 4 and "quat" in name:
        return ["w", "x", "y", "z"]
    if size == 3:
        return ["x", "y", "z"]
    if size == 2 and "gripper" in name:
        return ["left", "right"]
    return [str(index) for index in range(size)]


def flatten_debug_state(debug_state: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    flattened: dict[str, Any] = {}

    for key, value in debug_state.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            flattened[key] = normalize_csv_value(value)
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            flattened[key] = normalize_csv_value(array.item())
            continue

        flat_values = array.reshape(-1)
        suffixes = infer_column_suffixes(key, int(flat_values.size))
        for suffix, item in zip(suffixes, flat_values, strict=True):
            flattened[f"{key}_{suffix}"] = normalize_csv_value(item.item())

    return flattened


def build_debug_row(
    *,
    episode: int,
    step: int,
    phase: str,
    terminated: bool,
    truncated: bool,
    debug_state: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "episode": episode,
        "step": step,
        "phase": phase,
        "terminated": terminated,
        "truncated": truncated,
    }
    row.update(flatten_debug_state(debug_state))
    return row


class DebugStateCsvWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._file = None
        self._writer = None
        self._fieldnames: list[str] | None = None

    def write_row(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.output_path.open("w", newline="", encoding="utf-8")
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        assert self._fieldnames is not None
        extra_columns = [key for key in row if key not in self._fieldnames]
        if extra_columns:
            raise ValueError(
                "Debug state columns changed during evaluation. "
                f"Unexpected columns: {extra_columns}"
            )

        normalized_row = {key: row.get(key, "") for key in self._fieldnames}
        self._writer.writerow(normalized_row)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._fieldnames = None


def main() -> None:
    args = parse_args()
    if args.env == "PlaceTargetEnv" and not args.grasp_model:
        raise ValueError(
            "PlaceTargetEnv requires --grasp-model so reset can start from the trained grasping policy state."
        )

    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        from source.envs import (
            GraspingEnv,
            GraspingEnvV1,
            GraspingEnvV2,
            GraspingEnvV3,
            PlaceTargetEnv,
            ReachingEnv,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import environments from source.envs. Run this script from the project root "
            "or ensure the project root is on PYTHONPATH."
        ) from exc

    env_registry = {
        "GraspingEnv": GraspingEnv,
        "GraspingEnvV1": GraspingEnvV1,
        "GraspingEnvV2": GraspingEnvV2,
        "PlaceTargetEnv": PlaceTargetEnv,
        "ReachingEnv": ReachingEnv,
    }
    if GraspingEnvV3 is not None:
        env_registry["GraspingEnvV3"] = GraspingEnvV3

    xml_path = resolve_xml_path(args.env, args.xml_file)

    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model
        else resolve_latest_model(args.env)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    render_mode = None if args.render == "none" else args.render
    env_cls = env_registry[args.env]
    env_kwargs = {}
    if args.env == "PlaceTargetEnv":
        env_kwargs.update(
            {
                "grasp_model_path": args.grasp_model,
                "grasp_env_name": args.grasp_env,
                "grasp_xml_file": args.grasp_xml_file,
                "grasp_max_steps": args.grasp_max_steps,
                "grasp_attempts_per_reset": args.grasp_attempts,
                "grasp_success_min_lift": args.grasp_min_lift,
                "grasp_success_ee_obj_dist": args.grasp_ee_obj_dist,
                "grasp_success_hold_steps": args.grasp_hold_steps,
                "terminate_ee_obj_distance": args.terminate_ee_obj_dist,
            }
        )
    env = env_cls(xml_file=str(xml_path), render_mode=render_mode, **env_kwargs)
    debug_state_getter = getattr(env.unwrapped, "get_debug_state", None)
    if not callable(debug_state_getter):
        raise AttributeError(
            f"{type(env.unwrapped).__name__} does not implement get_debug_state()."
        )

    # Attach env to ensure action/obs spaces match what we are evaluating.
    model = SAC.load(str(model_path), env=env, device="auto")
    output_csv_path = resolve_output_csv_path(args.output_dir, args.env, model_path)
    csv_writer = DebugStateCsvWriter(output_csv_path)

    deterministic = not args.stochastic
    max_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(getattr(env, "max_episode_steps", 500))
    )

    print(f"[OK] Model: {model_path}")
    print(f"[OK] XML  : {xml_path}")
    print(f"[OK] CSV  : {output_csv_path}")
    print(
        f"[OK] Env  : {args.env} (render={args.render}, deterministic={deterministic})"
    )

    try:
        for ep in range(args.episodes):
            episode_seed = (args.seed + ep) if args.seed is not None else None
            obs, _info = env.reset(seed=episode_seed)
            csv_writer.write_row(
                build_debug_row(
                    episode=ep + 1,
                    step=0,
                    phase="reset",
                    terminated=False,
                    truncated=False,
                    debug_state=debug_state_getter(),
                )
            )

            for _step in range(max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, _reward, terminated, truncated, _step_info = env.step(action)
                csv_writer.write_row(
                    build_debug_row(
                        episode=ep + 1,
                        step=_step + 1,
                        phase="step",
                        terminated=terminated,
                        truncated=truncated,
                        debug_state=debug_state_getter(),
                    )
                )
                # if terminated or truncated:
                #     print(
                #         f"Terminated episode {ep + 1} at step {_step + 1} (terminated={terminated}, truncated={truncated})"
                #     )
                #     break
    finally:
        csv_writer.close()
        env.close()

    print("[OK] Done.")


if __name__ == "__main__":
    main()
