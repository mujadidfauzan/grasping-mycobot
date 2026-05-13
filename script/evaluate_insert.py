from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OBJECT_PLACE_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
OBJECT_LIFT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
DEFAULT_INSERT_MODEL_PATH = (
    PROJECT_ROOT / "melogs" / "ik_models" / "insert-ik-model.zip"
)
DEFAULT_EVAL_LOG_DIR = PROJECT_ROOT / "logs_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an InsertTargetEnvIK SAC policy, optionally opening the "
            "gripper only during evaluation after the box reaches the target pose."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to the trained InsertTargetEnvIK .zip. If omitted, uses "
            "melogs/ik_models/insert-ik-model.zip when present, otherwise the "
            "newest checkpoint under logs/models/InsertTargetEnvIK/."
        ),
    )
    parser.add_argument(
        "--xml-file",
        default=None,
        help="Path to object_place.xml. Defaults to source/robot/object_place.xml.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max policy steps per episode. Defaults to env.max_episode_steps.",
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
        help="Seed for env.reset(). Episode N uses seed+N when provided.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic actions.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EVAL_LOG_DIR),
        help="Directory where debug CSV files will be saved.",
    )
    parser.add_argument(
        "--release-mode",
        choices=["never", "on-align", "on-success"],
        default="on-align",
        help=(
            "never: keep gripper closed; on-align: open as soon as target_pose_aligned "
            "is true; on-success: open after the env reports terminated_success."
        ),
    )
    parser.add_argument(
        "--pre-release-hold-steps",
        type=int,
        default=50,
        help=(
            "After the release condition is reached, apply zero action for this "
            "many steps before opening the gripper."
        ),
    )
    parser.add_argument(
        "--post-release-steps",
        type=int,
        default=40,
        help="Extra zero-action steps to simulate after evaluation gripper release.",
    )
    parser.add_argument(
        "--grasp-model",
        default=None,
        help=(
            "Path to the trained GraspingEnvIK .zip used for reset snapshots. "
            "If omitted, InsertTargetEnvIK uses its built-in default."
        ),
    )
    parser.add_argument(
        "--grasp-xml-file",
        default=None,
        help="XML scene used by the grasping reset policy. Defaults to object_lift.xml.",
    )
    parser.add_argument(
        "--grasp-max-steps",
        type=int,
        default=300,
        help="Max rollout steps per grasp-policy reset attempt.",
    )
    parser.add_argument(
        "--grasp-attempts",
        type=int,
        default=6,
        help="How many grasp-policy reset attempts to try before fallback.",
    )
    parser.add_argument(
        "--grasp-min-lift",
        type=float,
        default=0.035,
        help="Minimum lift height required for an accepted grasp snapshot.",
    )
    parser.add_argument(
        "--grasp-ee-obj-dist",
        type=float,
        default=0.04,
        help="Maximum EE-object distance allowed for an accepted grasp snapshot.",
    )
    parser.add_argument(
        "--grasp-hold-steps",
        type=int,
        default=3,
        help="Consecutive valid grasp steps required before transferring state.",
    )
    parser.add_argument(
        "--terminate-ee-obj-dist",
        type=float,
        default=0.08,
        help="Terminate when object is this far or farther from the end effector.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print a compact step summary every N steps. Use 0 to disable.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str | None, default_path: Path, label: str) -> Path:
    path = default_path if path_arg is None else Path(path_arg).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def resolve_latest_model() -> Path:
    models_root = PROJECT_ROOT / "logs" / "models" / "InsertTargetEnvIK"
    if not models_root.exists():
        raise FileNotFoundError(
            f"Model folder not found: {models_root}. Pass --model explicitly."
        )

    candidates = sorted(
        models_root.glob("**/*.zip"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .zip model found under {models_root}. Pass --model explicitly."
        )
    return candidates[-1]


def resolve_insert_model(model_arg: str | None) -> Path:
    if model_arg is not None:
        return resolve_path(model_arg, DEFAULT_INSERT_MODEL_PATH, "Insert model")
    if DEFAULT_INSERT_MODEL_PATH.exists():
        return DEFAULT_INSERT_MODEL_PATH.resolve()
    return resolve_latest_model()


def resolve_output_csv_path(output_dir_arg: str, model_path: Path) -> Path:
    output_dir = Path(output_dir_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model_label = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in model_path.stem
    ).strip("._")
    return (
        output_dir / "InsertTargetEnvIK" / f"{timestamp}_{model_label}_debug_state.csv"
    )


def format_metric(value: Any, precision: int = 4) -> str:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(scalar):
        return str(scalar)
    return f"{scalar:.{precision}f}"


def is_truthy_metric(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def get_metric(debug_state: dict[str, Any], info: dict[str, Any], key: str) -> Any:
    if key in info:
        return info[key]
    return debug_state.get(key)


def should_release(
    *,
    release_mode: str,
    debug_state: dict[str, Any],
    info: dict[str, Any],
) -> bool:
    if release_mode == "never":
        return False
    if release_mode == "on-align":
        return is_truthy_metric(get_metric(debug_state, info, "target_pose_aligned"))
    if release_mode == "on-success":
        return is_truthy_metric(get_metric(debug_state, info, "terminated_success"))
    raise ValueError(f"Unsupported release mode: {release_mode}")


def build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs: dict[str, Any] = {
        "grasp_xml_file": args.grasp_xml_file,
        "grasp_max_steps": args.grasp_max_steps,
        "grasp_attempts_per_reset": args.grasp_attempts,
        "grasp_success_min_lift": args.grasp_min_lift,
        "grasp_success_ee_obj_dist": args.grasp_ee_obj_dist,
        "grasp_success_hold_steps": args.grasp_hold_steps,
        "terminate_ee_obj_distance": args.terminate_ee_obj_dist,
        "allow_eval_gripper_release": args.release_mode != "never",
    }
    if args.grasp_model is not None:
        env_kwargs["grasp_model_path"] = str(
            resolve_path(args.grasp_model, OBJECT_LIFT_XML_PATH, "Grasp model")
        )
    return env_kwargs


def collect_debug_state(env: Any) -> dict[str, Any]:
    debug_state_getter = getattr(env.unwrapped, "get_debug_state", None)
    if not callable(debug_state_getter):
        raise AttributeError(
            f"{type(env.unwrapped).__name__} does not implement get_debug_state()."
        )
    return dict(debug_state_getter())


def flatten_debug_state(debug_state: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in debug_state.items():
        if isinstance(value, (str, bool, int, float)) or value is None:
            flattened[key] = value
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            flattened[key] = array.item()
            continue

        flat_values = array.reshape(-1)
        suffixes = infer_column_suffixes(key, int(flat_values.size))
        for suffix, item in zip(suffixes, flat_values, strict=True):
            flattened[f"{key}_{suffix}"] = item.item()
    return flattened


def infer_column_suffixes(name: str, size: int) -> list[str]:
    if size == 4 and "quat" in name:
        return ["w", "x", "y", "z"]
    if size == 3:
        return ["x", "y", "z"]
    if size == 2 and "gripper" in name:
        return ["left", "right"]
    return [str(index) for index in range(size)]


def normalize_csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8f}".rstrip("0").rstrip(".") or "0"
    if value is None:
        return ""
    return value


class DebugCsvWriter:
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self._file = None
        self._writer = None
        self._fieldnames: list[str] | None = None
        self._rows: list[dict[str, Any]] = []

    def _open_writer(self) -> None:
        import csv

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
        for row in self._rows:
            self._writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        self._file.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        normalized = {key: normalize_csv_value(value) for key, value in row.items()}
        if self._fieldnames is None:
            self._fieldnames = list(normalized.keys())
            self._open_writer()

        assert self._fieldnames is not None
        extra_columns = [key for key in normalized if key not in self._fieldnames]
        if extra_columns:
            self._fieldnames.extend(extra_columns)
            self._rewrite_file()

        self._rows.append(dict(normalized))
        assert self._writer is not None
        self._writer.writerow(
            {key: normalized.get(key, "") for key in self._fieldnames}
        )
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = None
        self._writer = None
        self._fieldnames = None
        self._rows = []


def build_row(
    *,
    episode: int,
    step: int,
    phase: str,
    reward: float | None,
    terminated: bool,
    truncated: bool,
    debug_state: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "episode": episode,
        "step": step,
        "phase": phase,
        "reward": "" if reward is None else float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    if info:
        for key, value in info.items():
            if isinstance(value, (str, bool, int, float, np.generic)) or value is None:
                row[f"info_{key}"] = value
    row.update(flatten_debug_state(debug_state))
    return row


def print_step_summary(
    *,
    episode: int,
    step: int,
    phase: str,
    reward: float | None,
    debug_state: dict[str, Any],
    info: dict[str, Any],
) -> None:
    target_dist = get_metric(debug_state, info, "object_target_dist")
    target_angle = get_metric(debug_state, info, "object_target_angle_rad")
    ee_obj_dist = get_metric(debug_state, info, "ee_object_dist")
    aligned = is_truthy_metric(get_metric(debug_state, info, "target_pose_aligned"))
    success = is_truthy_metric(get_metric(debug_state, info, "terminated_success"))
    too_far = is_truthy_metric(get_metric(debug_state, info, "terminated_ee_obj_far"))
    gripper_state = debug_state.get("gripper_state", "n/a")
    reward_text = "n/a" if reward is None else f"{reward:.3f}"
    print(
        " ".join(
            [
                f"ep={episode}",
                f"step={step}",
                f"phase={phase}",
                f"reward={reward_text}",
                f"obj_target={format_metric(target_dist)}m",
                f"angle={format_metric(target_angle)}rad",
                f"ee_obj={format_metric(ee_obj_dist)}m",
                f"aligned={int(aligned)}",
                f"success={int(success)}",
                f"too_far={int(too_far)}",
                f"gripper={gripper_state}",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be greater than 0.")
    if args.pre_release_hold_steps < 0:
        raise ValueError("--pre-release-hold-steps must be non-negative.")
    if args.post_release_steps < 0:
        raise ValueError("--post-release-steps must be non-negative.")

    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        from source.envs import InsertTargetEnvIK
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import InsertTargetEnvIK. Run this script from the project root "
            "or with: python -m script.evaluate_insert"
        ) from exc

    xml_path = resolve_path(args.xml_file, OBJECT_PLACE_XML_PATH, "Insert XML")
    model_path = resolve_insert_model(args.model)
    output_csv_path = resolve_output_csv_path(args.output_dir, model_path)
    render_mode = None if args.render == "none" else args.render
    deterministic = not args.stochastic

    env = InsertTargetEnvIK(
        xml_file=str(xml_path),
        render_mode=render_mode,
        **build_env_kwargs(args),
    )
    model = SAC.load(str(model_path), env=env, device="auto")
    csv_writer = DebugCsvWriter(output_csv_path)
    max_policy_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(getattr(env, "max_episode_steps", 150))
    )

    print(f"[OK] Model        : {model_path}")
    print(f"[OK] XML          : {xml_path}")
    print(f"[OK] CSV          : {output_csv_path}")
    print(f"[OK] Render       : {args.render}")
    print(f"[OK] Deterministic: {deterministic}")
    print(f"[OK] Release mode : {args.release_mode}")
    print(f"[OK] Hold before release: {args.pre_release_hold_steps} zero-action steps")

    try:
        for episode in range(1, args.episodes + 1):
            episode_seed = (args.seed + episode - 1) if args.seed is not None else None
            obs, reset_info = env.reset(seed=episode_seed)
            release_pending = False
            released = False
            pre_release_hold_count = 0
            post_release_count = 0
            last_info: dict[str, Any] = dict(reset_info or {})
            debug_state = collect_debug_state(env)
            csv_writer.write_row(
                build_row(
                    episode=episode,
                    step=0,
                    phase="reset",
                    reward=None,
                    terminated=False,
                    truncated=False,
                    debug_state=debug_state,
                    info=last_info,
                )
            )
            print_step_summary(
                episode=episode,
                step=0,
                phase="reset",
                reward=None,
                debug_state=debug_state,
                info=last_info,
            )

            max_steps = (
                max_policy_steps
                + args.pre_release_hold_steps
                + args.post_release_steps
                + 1
            )
            for step in range(1, max_steps + 1):
                if not released and not release_pending and step > max_policy_steps:
                    break
                if (
                    release_pending
                    and pre_release_hold_count >= args.pre_release_hold_steps
                ):
                    env.open_gripper_for_eval()
                    released = True
                    release_pending = False
                    debug_state = collect_debug_state(env)
                    csv_writer.write_row(
                        build_row(
                            episode=episode,
                            step=step - 1,
                            phase="release",
                            reward=None,
                            terminated=False,
                            truncated=False,
                            debug_state=debug_state,
                            info=last_info,
                        )
                    )
                    print_step_summary(
                        episode=episode,
                        step=step - 1,
                        phase="release",
                        reward=None,
                        debug_state=debug_state,
                        info=last_info,
                    )
                if released and post_release_count >= args.post_release_steps:
                    break

                if released:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    phase = "post_release"
                elif release_pending:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                    phase = "pre_release_hold"
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    phase = "step"

                obs, reward, terminated, truncated, step_info = env.step(action)
                last_info = dict(step_info or {})
                debug_state = collect_debug_state(env)
                csv_writer.write_row(
                    build_row(
                        episode=episode,
                        step=step,
                        phase=phase,
                        reward=float(reward),
                        terminated=terminated,
                        truncated=truncated,
                        debug_state=debug_state,
                        info=last_info,
                    )
                )

                if args.print_every > 0 and (
                    step == 1 or step % args.print_every == 0 or terminated or truncated
                ):
                    print_step_summary(
                        episode=episode,
                        step=step,
                        phase=phase,
                        reward=float(reward),
                        debug_state=debug_state,
                        info=last_info,
                    )

                if released:
                    post_release_count += 1
                    continue

                if release_pending:
                    pre_release_hold_count += 1
                    continue

                if should_release(
                    release_mode=args.release_mode,
                    debug_state=debug_state,
                    info=last_info,
                ):
                    release_pending = True
                    pre_release_hold_count = 0
                    csv_writer.write_row(
                        build_row(
                            episode=episode,
                            step=step,
                            phase="release_pending",
                            reward=float(reward),
                            terminated=terminated,
                            truncated=truncated,
                            debug_state=debug_state,
                            info=last_info,
                        )
                    )
                    print_step_summary(
                        episode=episode,
                        step=step,
                        phase="release_pending",
                        reward=float(reward),
                        debug_state=debug_state,
                        info=last_info,
                    )
                    continue

                if terminated or truncated:
                    break
    finally:
        csv_writer.close()
        env.close()

    print("[OK] Done.")


if __name__ == "__main__":
    main()
