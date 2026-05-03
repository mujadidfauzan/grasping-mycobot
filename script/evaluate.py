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
        help="For placement/insertion envs: path to the trained grasping SAC .zip used to generate reset states.",
    )
    parser.add_argument(
        "--grasp-env",
        default="GraspingEnvV2",
        help="For placement/insertion envs: grasping environment class name used by --grasp-model.",
    )
    parser.add_argument(
        "--grasp-xml-file",
        default=None,
        help="For placement/insertion envs: XML scene used by the grasping policy. Defaults to object_lift.xml.",
    )
    parser.add_argument(
        "--grasp-max-steps",
        type=int,
        default=300,
        help="For placement/insertion envs: max rollout steps per grasp-policy reset attempt.",
    )
    parser.add_argument(
        "--grasp-attempts",
        type=int,
        default=6,
        help="For placement/insertion envs: how many grasp-policy reset attempts to try before falling back to the best snapshot.",
    )
    parser.add_argument(
        "--grasp-min-lift",
        type=float,
        default=0.025,
        help="For placement/insertion envs: minimum object lift height required before a grasp snapshot is accepted.",
    )
    parser.add_argument(
        "--grasp-ee-obj-dist",
        type=float,
        default=0.035,
        help="For placement/insertion envs: max EE-object distance allowed for a grasp snapshot.",
    )
    parser.add_argument(
        "--grasp-hold-steps",
        type=int,
        default=3,
        help="For placement/insertion envs: required consecutive valid grasp steps before transferring the state.",
    )
    parser.add_argument(
        "--terminate-ee-obj-dist",
        type=float,
        default=0.08,
        help="For placement/insertion envs: terminate the episode when the object is this far or farther from the EE.",
    )
    parser.add_argument(
        "--place-above-model",
        default=None,
        help="For InsertTargetEnv: path to the trained PlaceAboveSite SAC .zip used to generate reset states.",
    )
    parser.add_argument(
        "--place-above-xml-file",
        default=None,
        help="For InsertTargetEnv: XML scene used by the place-above policy. Defaults to the insertion env XML.",
    )
    parser.add_argument(
        "--place-above-max-steps",
        type=int,
        default=150,
        help="For InsertTargetEnv: max rollout steps per place-above-policy reset attempt.",
    )
    parser.add_argument(
        "--place-above-attempts",
        type=int,
        default=4,
        help="For InsertTargetEnv: how many place-above-policy reset attempts to try before falling back to the best snapshot.",
    )
    parser.add_argument(
        "--place-above-hold-steps",
        type=int,
        default=10,
        help="For InsertTargetEnv: required consecutive valid place-above steps before transferring the state.",
    )
    parser.add_argument(
        "--place-above-target-height",
        type=float,
        default=0.02,
        help="For InsertTargetEnv: target height above the XML place used by the place-above policy.",
    )
    return parser.parse_args()


def resolve_env_names() -> list[str]:
    try:
        from source.envs import GraspingEnvV3
    except ModuleNotFoundError:
        GraspingEnvV3 = None

    env_names = [
        "GraspingEnv",
        "GraspingEnvIK",
        "GraspingEnvV1",
        "GraspingEnvV2",
        "InsertTargetEnv",
        "InsertTargetEnvIK",
        "PlaceAboveTargetEnv",
        "PlaceTargetEnv",
        "ReachingEnv",
        "PlaceAboveSiteEnv",
    ]
    if GraspingEnvV3 is not None:
        env_names.append("GraspingEnvV3")
    return sorted(env_names)


def resolve_default_xml_path(env_name: str) -> Path:
    if env_name in {
        "InsertTargetEnv",
        "InsertTargetEnvIK",
        "PlaceTargetEnv",
        "PlaceAboveTargetEnv",
        "PlaceAboveSiteEnv",
    }:
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


def sanitize_column_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "value"


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


def get_joint_qpos_size(model: Any, joint_id: int) -> int:
    import mujoco

    joint_type = int(model.jnt_type[joint_id])
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def extract_robot_joint_positions(env: Any) -> dict[str, Any]:
    import mujoco

    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    object_info = getattr(env, "object_info", None)
    if (
        model is None
        or data is None
        or not isinstance(object_info, dict)
        or not object_info
    ):
        return {}

    try:
        first_object_qposadr = min(
            int(info["qposadr"]) for info in object_info.values()
        )
    except (KeyError, TypeError, ValueError):
        return {}

    joint_positions: dict[str, Any] = {}
    for joint_id in range(int(model.njnt)):
        qposadr = int(model.jnt_qposadr[joint_id])
        if qposadr >= first_object_qposadr:
            continue

        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        safe_joint_name = sanitize_column_part(joint_name or f"joint_{joint_id}")
        column_name = f"robot_joint_pos_{safe_joint_name}"
        qpos_size = get_joint_qpos_size(model, joint_id)
        joint_qpos = data.qpos[qposadr : qposadr + qpos_size].copy()
        joint_positions[column_name] = (
            float(joint_qpos[0]) if qpos_size == 1 else joint_qpos
        )

    return joint_positions


def collect_debug_state(env: Any, debug_state_getter: Any) -> dict[str, Any]:
    debug_state = dict(debug_state_getter())
    debug_state.update(extract_robot_joint_positions(env))
    return debug_state


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
        self._rows: list[dict[str, Any]] = []

    def _open_writer(self) -> None:
        assert self._fieldnames is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        self._writer.writeheader()

    def _rewrite_file(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

        self._open_writer()
        assert self._fieldnames is not None
        assert self._writer is not None
        for existing_row in self._rows:
            normalized_row = {key: existing_row.get(key, "") for key in self._fieldnames}
            self._writer.writerow(normalized_row)
        self._file.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
            self._open_writer()

        assert self._fieldnames is not None
        extra_columns = [key for key in row if key not in self._fieldnames]
        if extra_columns:
            self._fieldnames.extend(extra_columns)
            self._rewrite_file()

        self._rows.append(dict(row))
        assert self._writer is not None
        normalized_row = {key: row.get(key, "") for key in self._fieldnames}
        self._writer.writerow(normalized_row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._fieldnames = None
            self._rows = []


def main() -> None:
    args = parse_args()
    if args.env in {
        "InsertTargetEnv",
        "PlaceTargetEnv",
        "PlaceAboveTargetEnv",
        "PlaceAboveSiteEnv",
    } and not args.grasp_model:
        raise ValueError(
            f"{args.env} requires --grasp-model so reset can start from the trained grasping policy state."
        )
    if args.env == "InsertTargetEnv" and not args.place_above_model:
        raise ValueError(
            f"{args.env} requires --place-above-model so reset can start from the trained PlaceAboveSite policy state."
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
            GraspingEnvIK,
            GraspingEnvV1,
            GraspingEnvV2,
            GraspingEnvV3,
            InsertTargetEnv,
            InsertTargetEnvIK,
            PlaceAboveSiteEnv,
            PlaceAboveTargetEnv,
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
        "GraspingEnvIK": GraspingEnvIK,
        "GraspingEnvV1": GraspingEnvV1,
        "GraspingEnvV2": GraspingEnvV2,
        "InsertTargetEnv": InsertTargetEnv,
        "InsertTargetEnvIK": InsertTargetEnvIK,
        "PlaceAboveTargetEnv": PlaceAboveTargetEnv,
        "PlaceAboveSiteEnv": PlaceAboveSiteEnv,
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
    if args.env in {
        "InsertTargetEnv",
        "InsertTargetEnvIK",
        "PlaceTargetEnv",
        "PlaceAboveTargetEnv",
        "PlaceAboveSiteEnv",
    }:
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
            }
        )
        if args.env != "PlaceAboveSiteEnv":
            env_kwargs["terminate_ee_obj_distance"] = args.terminate_ee_obj_dist
        if args.env == "InsertTargetEnv":
            env_kwargs.update(
                {
                    "place_above_model_path": args.place_above_model,
                    "place_above_xml_file": args.place_above_xml_file,
                    "place_above_max_steps": args.place_above_max_steps,
                    "place_above_attempts_per_reset": args.place_above_attempts,
                    "place_above_success_hold_steps": args.place_above_hold_steps,
                    "place_above_target_height_above_place": args.place_above_target_height,
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
                    debug_state=collect_debug_state(env.unwrapped, debug_state_getter),
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
                        debug_state=collect_debug_state(
                            env.unwrapped, debug_state_getter
                        ),
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
