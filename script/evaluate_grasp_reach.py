from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OBJECT_LIFT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
OBJECT_PLACE_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs_eval" / "pipeline"


def parse_args() -> argparse.Namespace:
    grasp_env_names, reach_env_names = resolve_env_names()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a two-stage pipeline that runs a grasping policy first, "
            "then transfers the simulator state directly to a reach/place policy."
        )
    )
    parser.add_argument(
        "--grasp-env",
        choices=grasp_env_names,
        default="GraspingEnv",
        help="Environment class used by the grasping policy.",
    )
    parser.add_argument(
        "--grasp-model",
        default=None,
        help="Path to the trained grasping SAC .zip. Defaults to the newest model for --grasp-env.",
    )
    parser.add_argument(
        "--grasp-xml-file",
        default=None,
        help="XML scene for the grasping policy. Defaults to object_lift.xml.",
    )
    parser.add_argument(
        "--reach-env",
        choices=reach_env_names,
        default="PlaceAboveSiteEnv",
        help="Environment class used by the reach/place policy.",
    )
    parser.add_argument(
        "--reach-model",
        default=None,
        help="Path to the trained reach/place SAC .zip. Defaults to the newest model for --reach-env.",
    )
    parser.add_argument(
        "--reach-xml-file",
        default=None,
        help="XML scene for the reach/place policy. Defaults to object_place.xml.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of pipeline episodes to run.",
    )
    parser.add_argument(
        "--grasp-max-steps",
        type=int,
        default=300,
        help="Maximum steps to run the grasping policy before attempting transfer.",
    )
    parser.add_argument(
        "--reach-max-steps",
        type=int,
        default=150,
        help="Maximum steps to run the reach/place policy after transfer.",
    )
    parser.add_argument(
        "--render",
        choices=["none", "human", "rgb_array"],
        default="human",
        help="Render mode for both environments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed. Episode i uses seed + i.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV outputs will be saved.",
    )
    parser.add_argument(
        "--transfer-min-lift",
        type=float,
        default=0.025,
        help="Minimum lift height required before the grasp snapshot can be transferred.",
    )
    parser.add_argument(
        "--transfer-ee-obj-dist",
        type=float,
        default=0.035,
        help="Maximum EE-object distance allowed for a transferable grasp snapshot.",
    )
    parser.add_argument(
        "--transfer-hold-steps",
        type=int,
        default=3,
        help="Required consecutive valid grasp steps before transfer.",
    )
    parser.add_argument(
        "--transfer-close-threshold",
        type=float,
        default=0.005,
        help="Minimum signed gripper control magnitude considered closed.",
    )
    parser.add_argument(
        "--allow-fallback-transfer",
        action="store_true",
        help="If grasp success is not reached, transfer the best scored snapshot instead of skipping reach.",
    )
    return parser.parse_args()


def resolve_env_names() -> tuple[list[str], list[str]]:
    grasp_env_names = ["GraspingEnv", "GraspingEnvV1", "GraspingEnvV2"]
    if (PROJECT_ROOT / "source" / "envs" / "grasping_env_v3.py").exists():
        grasp_env_names.append("GraspingEnvV3")

    reach_env_names = ["PlaceAboveSiteEnv", "PlaceAboveTargetEnv", "PlaceTargetEnv"]
    if (PROJECT_ROOT / "source" / "envs" / "reaching_env.py").exists():
        reach_env_names.append("ReachingEnv")

    return sorted(grasp_env_names), sorted(reach_env_names)


def build_env_registry() -> dict[str, Any]:
    from source.envs import (
        GraspingEnv,
        GraspingEnvV1,
        GraspingEnvV2,
        PlaceAboveSiteEnv,
        PlaceAboveTargetEnv,
        PlaceTargetEnv,
        ReachingEnv,
    )

    try:
        from source.envs import GraspingEnvV3
    except ModuleNotFoundError:
        GraspingEnvV3 = None

    registry = {
        "GraspingEnv": GraspingEnv,
        "GraspingEnvV1": GraspingEnvV1,
        "GraspingEnvV2": GraspingEnvV2,
        "PlaceAboveSiteEnv": PlaceAboveSiteEnv,
        "PlaceAboveTargetEnv": PlaceAboveTargetEnv,
        "PlaceTargetEnv": PlaceTargetEnv,
    }
    if ReachingEnv is not None:
        registry["ReachingEnv"] = ReachingEnv
    if GraspingEnvV3 is not None:
        registry["GraspingEnvV3"] = GraspingEnvV3
    return registry


def resolve_default_xml_path(env_name: str) -> Path:
    if env_name in {"PlaceAboveSiteEnv", "PlaceAboveTargetEnv", "PlaceTargetEnv"}:
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
            f"Model folder not found: {models_root}. Pass the model path explicitly."
        )
    candidates = sorted(models_root.glob("**/*.zip"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No .zip model found under {models_root}. Pass the model path explicitly."
        )
    return candidates[-1]


def resolve_model_path(model_arg: str | None, env_name: str) -> Path:
    model_path = (
        Path(model_arg).expanduser().resolve()
        if model_arg is not None
        else resolve_latest_model(env_name)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return model_path


def sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "evaluation"


def sanitize_column_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def build_output_paths(
    output_dir_arg: str,
    grasp_env_name: str,
    reach_env_name: str,
    grasp_model_path: Path,
    reach_model_path: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir_arg).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()

    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    grasp_label = sanitize_filename_part(grasp_model_path.stem)
    reach_label = sanitize_filename_part(reach_model_path.stem)
    stem = f"{timestamp}_{grasp_env_name}_{grasp_label}__to__{reach_env_name}_{reach_label}"

    return {
        "grasp": output_dir / f"{stem}_grasp_debug_state.csv",
        "reach": output_dir / f"{stem}_reach_debug_state.csv",
        "summary": output_dir / f"{stem}_summary.csv",
    }


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
                "Debug state columns changed during pipeline evaluation. "
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


def capture_grasp_snapshot(env: Any, debug_state: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    active_object = str(debug_state["active_object"])
    object_info = getattr(env, "object_info", {})
    dofadr = int(object_info[active_object]["dofadr"])
    object_speed = float(np.linalg.norm(env.data.qvel[dofadr : dofadr + 3]))

    return {
        "qpos": env.data.qpos.copy(),
        "qvel": env.data.qvel.copy(),
        "ctrl": env.data.ctrl.copy(),
        "active_object": active_object,
        "obj_pos": np.asarray(debug_state["obj_pos"], dtype=np.float64).copy(),
        "obj_quat": np.asarray(debug_state["obj_quat"], dtype=np.float64).copy(),
        "ee_pos": np.asarray(debug_state["ee_pos"], dtype=np.float64).copy(),
        "ee_quat": np.asarray(debug_state["ee_quat"], dtype=np.float64).copy(),
        "lift_height": float(debug_state["lift_height"]),
        "ee_obj_dist": float(debug_state["ee_obj_dist"]),
        "object_speed": object_speed,
        "gripper_ctrl": env.data.ctrl[-2:].copy(),
        "grasp_latched": bool(debug_state.get("grasp_latched", False)),
        "terminated_like": bool(getattr(env, "success_counter", 0) > 0),
    }


def copy_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    copied: dict[str, Any] = {}
    for key, value in snapshot.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = value
    return copied


def is_good_grasp_snapshot(snapshot: dict[str, Any], args: argparse.Namespace) -> bool:
    import numpy as np

    gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
    is_closed = bool(
        gripper_ctrl[0] < -args.transfer_close_threshold
        and gripper_ctrl[1] > args.transfer_close_threshold
    )
    return bool(
        is_closed
        and float(snapshot["ee_obj_dist"]) <= args.transfer_ee_obj_dist
        and float(snapshot["lift_height"]) >= args.transfer_min_lift
    )


def score_grasp_snapshot(snapshot: dict[str, Any], args: argparse.Namespace) -> float:
    import numpy as np

    gripper_ctrl = np.asarray(snapshot["gripper_ctrl"], dtype=np.float64)
    is_closed = float(
        gripper_ctrl[0] < -args.transfer_close_threshold
        and gripper_ctrl[1] > args.transfer_close_threshold
    )
    return (
        4.0 * float(snapshot["lift_height"])
        - 2.5 * float(snapshot["ee_obj_dist"])
        - 0.2 * float(snapshot["object_speed"])
        + 0.05 * is_closed
        + 0.02 * float(snapshot["terminated_like"])
        + 0.02 * float(snapshot["grasp_latched"])
    )


def update_reach_metrics(
    metrics: dict[str, Any],
    debug_state: dict[str, Any],
) -> None:
    target_dist = debug_state.get("obj_target_dist", debug_state.get("object_target_dist"))
    if isinstance(target_dist, (int, float)):
        metrics["best_target_dist"] = min(
            float(metrics["best_target_dist"]),
            float(target_dist),
        )
        metrics["final_target_dist"] = float(target_dist)

    success_counter = debug_state.get("success_counter")
    if isinstance(success_counter, (int, float)):
        metrics["max_success_counter"] = max(
            int(metrics["max_success_counter"]),
            int(success_counter),
        )


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    env_registry = build_env_registry()
    if args.grasp_env not in env_registry:
        raise ValueError(f"Unsupported grasp env: {args.grasp_env}")
    if args.reach_env not in env_registry:
        raise ValueError(f"Unsupported reach env: {args.reach_env}")

    grasp_xml_path = resolve_xml_path(args.grasp_env, args.grasp_xml_file)
    reach_xml_path = resolve_xml_path(args.reach_env, args.reach_xml_file)
    grasp_model_path = resolve_model_path(args.grasp_model, args.grasp_env)
    reach_model_path = resolve_model_path(args.reach_model, args.reach_env)
    output_paths = build_output_paths(
        args.output_dir,
        args.grasp_env,
        args.reach_env,
        grasp_model_path,
        reach_model_path,
    )

    render_mode = None if args.render == "none" else args.render
    deterministic = not args.stochastic

    grasp_env_cls = env_registry[args.grasp_env]
    reach_env_cls = env_registry[args.reach_env]

    grasp_env = grasp_env_cls(xml_file=str(grasp_xml_path), render_mode=render_mode)
    reach_env_kwargs = {"xml_file": str(reach_xml_path), "render_mode": render_mode}
    if args.reach_env in {"PlaceAboveSiteEnv", "PlaceAboveTargetEnv", "PlaceTargetEnv"}:
        reach_env_kwargs.update(
            {
                "grasp_model_path": str(grasp_model_path),
                "grasp_env_name": args.grasp_env,
                "grasp_xml_file": str(grasp_xml_path),
            }
        )
    reach_env = reach_env_cls(**reach_env_kwargs)

    reach_reset_from_snapshot = getattr(
        reach_env.unwrapped,
        "reset_from_grasp_snapshot",
        None,
    )
    if not callable(reach_reset_from_snapshot):
        raise AttributeError(
            f"{type(reach_env.unwrapped).__name__} does not implement reset_from_grasp_snapshot()."
        )

    grasp_debug_state_getter = getattr(grasp_env.unwrapped, "get_debug_state", None)
    reach_debug_state_getter = getattr(reach_env.unwrapped, "get_debug_state", None)
    if not callable(grasp_debug_state_getter):
        raise AttributeError(
            f"{type(grasp_env.unwrapped).__name__} does not implement get_debug_state()."
        )
    if not callable(reach_debug_state_getter):
        raise AttributeError(
            f"{type(reach_env.unwrapped).__name__} does not implement get_debug_state()."
        )

    grasp_model = SAC.load(str(grasp_model_path), env=grasp_env, device="auto")
    reach_model = SAC.load(str(reach_model_path), env=reach_env, device="auto")

    grasp_writer = DebugStateCsvWriter(output_paths["grasp"])
    reach_writer = DebugStateCsvWriter(output_paths["reach"])
    summary_rows: list[dict[str, Any]] = []

    print(f"[OK] Grasp model : {grasp_model_path}")
    print(f"[OK] Reach model : {reach_model_path}")
    print(f"[OK] Grasp XML   : {grasp_xml_path}")
    print(f"[OK] Reach XML   : {reach_xml_path}")
    print(f"[OK] Grasp CSV   : {output_paths['grasp']}")
    print(f"[OK] Reach CSV   : {output_paths['reach']}")
    print(f"[OK] Summary CSV : {output_paths['summary']}")

    try:
        for ep in range(args.episodes):
            episode_index = ep + 1
            episode_seed = (args.seed + ep) if args.seed is not None else None

            grasp_obs, _ = grasp_env.reset(seed=episode_seed)
            grasp_writer.write_row(
                build_debug_row(
                    episode=episode_index,
                    step=0,
                    phase="reset",
                    terminated=False,
                    truncated=False,
                    debug_state=collect_debug_state(
                        grasp_env.unwrapped,
                        grasp_debug_state_getter,
                    ),
                )
            )

            transfer_snapshot: dict[str, Any] | None = None
            transfer_source = "not_transferred"
            best_snapshot: dict[str, Any] | None = None
            best_snapshot_score = -math.inf
            consecutive_good_steps = 0
            grasp_steps_run = 0
            grasp_terminated = False
            grasp_truncated = False

            for step_idx in range(args.grasp_max_steps):
                action, _ = grasp_model.predict(grasp_obs, deterministic=deterministic)
                grasp_obs, _reward, grasp_terminated, grasp_truncated, _ = grasp_env.step(
                    action
                )
                grasp_steps_run = step_idx + 1

                grasp_debug_state = collect_debug_state(
                    grasp_env.unwrapped,
                    grasp_debug_state_getter,
                )
                grasp_writer.write_row(
                    build_debug_row(
                        episode=episode_index,
                        step=grasp_steps_run,
                        phase="step",
                        terminated=grasp_terminated,
                        truncated=grasp_truncated,
                        debug_state=grasp_debug_state,
                    )
                )

                snapshot = capture_grasp_snapshot(grasp_env.unwrapped, grasp_debug_state)
                snapshot_score = score_grasp_snapshot(snapshot, args)
                if snapshot_score > best_snapshot_score:
                    best_snapshot_score = snapshot_score
                    best_snapshot = copy_snapshot(snapshot)

                if is_good_grasp_snapshot(snapshot, args):
                    consecutive_good_steps += 1
                else:
                    consecutive_good_steps = 0

                if consecutive_good_steps >= args.transfer_hold_steps:
                    transfer_snapshot = copy_snapshot(snapshot)
                    transfer_source = "grasp_success"
                    break

                if grasp_terminated or grasp_truncated:
                    break

            if transfer_snapshot is None and args.allow_fallback_transfer and best_snapshot is not None:
                transfer_snapshot = best_snapshot
                transfer_source = "grasp_fallback_best_snapshot"

            reach_steps_run = 0
            reach_terminated = False
            reach_truncated = False
            reach_metrics = {
                "best_target_dist": math.inf,
                "final_target_dist": math.inf,
                "max_success_counter": 0,
            }

            if transfer_snapshot is not None:
                reach_obs = reach_reset_from_snapshot(
                    transfer_snapshot,
                    seed=episode_seed,
                    reset_source=transfer_source,
                    attempt_count=1,
                )
                reach_reset_state = collect_debug_state(
                    reach_env.unwrapped,
                    reach_debug_state_getter,
                )
                update_reach_metrics(reach_metrics, reach_reset_state)
                reach_writer.write_row(
                    build_debug_row(
                        episode=episode_index,
                        step=0,
                        phase="reset",
                        terminated=False,
                        truncated=False,
                        debug_state=reach_reset_state,
                    )
                )

                for step_idx in range(args.reach_max_steps):
                    action, _ = reach_model.predict(reach_obs, deterministic=deterministic)
                    reach_obs, _reward, reach_terminated, reach_truncated, _ = reach_env.step(
                        action
                    )
                    reach_steps_run = step_idx + 1

                    reach_debug_state = collect_debug_state(
                        reach_env.unwrapped,
                        reach_debug_state_getter,
                    )
                    update_reach_metrics(reach_metrics, reach_debug_state)
                    reach_writer.write_row(
                        build_debug_row(
                            episode=episode_index,
                            step=reach_steps_run,
                            phase="step",
                            terminated=reach_terminated,
                            truncated=reach_truncated,
                            debug_state=reach_debug_state,
                        )
                    )

                    if reach_terminated or reach_truncated:
                        break

            summary_rows.append(
                {
                    "episode": episode_index,
                    "seed": "" if episode_seed is None else episode_seed,
                    "grasp_steps_run": grasp_steps_run,
                    "grasp_terminated": grasp_terminated,
                    "grasp_truncated": grasp_truncated,
                    "transfer_success": transfer_snapshot is not None,
                    "transfer_source": transfer_source,
                    "transfer_lift_height": (
                        ""
                        if transfer_snapshot is None
                        else normalize_csv_value(transfer_snapshot["lift_height"])
                    ),
                    "transfer_ee_obj_dist": (
                        ""
                        if transfer_snapshot is None
                        else normalize_csv_value(transfer_snapshot["ee_obj_dist"])
                    ),
                    "reach_steps_run": reach_steps_run,
                    "reach_terminated": reach_terminated,
                    "reach_truncated": reach_truncated,
                    "reach_best_target_dist": normalize_csv_value(
                        reach_metrics["best_target_dist"]
                    ),
                    "reach_final_target_dist": normalize_csv_value(
                        reach_metrics["final_target_dist"]
                    ),
                    "reach_max_success_counter": reach_metrics["max_success_counter"],
                }
            )
    finally:
        grasp_writer.close()
        reach_writer.close()
        grasp_env.close()
        reach_env.close()

    output_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    with output_paths["summary"].open("w", newline="", encoding="utf-8") as summary_file:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else [
            "episode",
            "seed",
            "grasp_steps_run",
            "grasp_terminated",
            "grasp_truncated",
            "transfer_success",
            "transfer_source",
            "transfer_lift_height",
            "transfer_ee_obj_dist",
            "reach_steps_run",
            "reach_terminated",
            "reach_truncated",
            "reach_best_target_dist",
            "reach_final_target_dist",
            "reach_max_success_counter",
        ]
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    successful_transfers = sum(bool(row["transfer_success"]) for row in summary_rows)
    print(
        "[OK] Done. "
        f"Successful transfers: {successful_transfers}/{len(summary_rows)}."
    )


if __name__ == "__main__":
    main()
