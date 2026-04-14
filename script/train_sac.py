import argparse
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch import nn

from source.envs import (
    GraspingEnv,
    GraspingEnvV1,
    GraspingEnvV2,
    GraspingEnvV3,
    PlaceTargetEnv,
    ReachingEnv,
)

ENV_REGISTRY = {
    "GraspingEnv": GraspingEnv,
    "GraspingEnvV1": GraspingEnvV1,
    "GraspingEnvV2": GraspingEnvV2,
    "PlaceTargetEnv": PlaceTargetEnv,
    "ReachingEnv": ReachingEnv,
}

if GraspingEnvV3 is not None:
    ENV_REGISTRY["GraspingEnvV3"] = GraspingEnvV3

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OBJECT_LIFT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
OBJECT_PLACE_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
DEFAULT_XML_BY_ENV = {
    "GraspingEnv": OBJECT_LIFT_XML_PATH,
    "GraspingEnvV1": OBJECT_LIFT_XML_PATH,
    "GraspingEnvV2": OBJECT_LIFT_XML_PATH,
    "PlaceTargetEnv": OBJECT_PLACE_XML_PATH,
    "ReachingEnv": OBJECT_LIFT_XML_PATH,
}

if GraspingEnvV3 is not None:
    DEFAULT_XML_BY_ENV["GraspingEnvV3"] = OBJECT_LIFT_XML_PATH

POLICY_KWARGS = {
    "net_arch": {
        "pi": [512, 512, 256],
        "qf": [512, 512, 256],
    },
    "activation_fn": nn.ReLU,
}

INFO_LOG_EXCLUDE_KEYS = {
    "episode",
    "terminal_observation",
    "TimeLimit.truncated",
}


def _serialize_config_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serialize_config_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _format_yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return format(value, ".15g")
        return json.dumps(str(value))
    return json.dumps(str(value))


def _yaml_lines(value, indent: int = 0) -> list[str]:
    prefix = " " * indent

    if isinstance(value, Mapping):
        if not value:
            return [f"{prefix}{{}}"]

        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                if item:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {{}}")
                continue

            if isinstance(item, list):
                if item:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: []")
                continue

            lines.append(f"{prefix}{key}: {_format_yaml_scalar(item)}")

        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]

        lines = []
        for item in value:
            if isinstance(item, (Mapping, list)):
                if item:
                    lines.append(f"{prefix}-")
                    lines.extend(_yaml_lines(item, indent + 2))
                else:
                    empty_literal = "{}" if isinstance(item, Mapping) else "[]"
                    lines.append(f"{prefix}- {empty_literal}")
                continue

            lines.append(f"{prefix}- {_format_yaml_scalar(item)}")

        return lines

    return [f"{prefix}{_format_yaml_scalar(value)}"]


def _unwrap_env(env):
    current = env
    visited: set[int] = set()

    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "venv"):
            current = current.venv
            continue
        if hasattr(current, "envs") and getattr(current, "envs"):
            current = current.envs[0]
            continue
        if hasattr(current, "env"):
            current = current.env
            continue
        break

    return getattr(current, "unwrapped", current)


def _describe_space(space) -> dict:
    description = {
        "type": type(space).__name__,
        "shape": list(space.shape),
        "dtype": str(space.dtype),
    }
    if hasattr(space, "low"):
        description["low"] = _serialize_config_value(space.low)
    if hasattr(space, "high"):
        description["high"] = _serialize_config_value(space.high)
    return description


def save_env_config(env, args, run_name: str, models_dir: Path, xml_path: Path) -> Path:
    base_env = _unwrap_env(env)
    export_config = getattr(base_env, "export_config", None)
    if callable(export_config):
        env_config = export_config()
    else:
        env_config = {
            "env_name": type(base_env).__name__,
            "xml_file": str(xml_path),
            "action": {
                "space": _describe_space(base_env.action_space),
            },
            "observation": {
                "space": _describe_space(base_env.observation_space),
            },
        }

    config_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "env": _serialize_config_value(env_config),
        "training": {
            "mode": args.mode,
            "algorithm": "SAC",
            "env_registry_name": args.env,
            "xml_file": str(xml_path),
            "total_timesteps": int(args.total_timesteps),
            "save_freq": int(args.save_freq),
            "video_freq": int(args.video_freq),
            "video_length": int(args.video_length),
            "log_freq": int(args.log_freq),
            "policy_kwargs": _serialize_config_value(POLICY_KWARGS),
        },
    }

    config_path = models_dir / "env_config.yaml"
    config_path.write_text(
        "\n".join(_yaml_lines(config_payload)) + "\n",
        encoding="utf-8",
    )
    return config_path


def _coerce_scalar_metric(value) -> float | None:
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


def _flatten_info_metrics(info: Mapping, prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, value in info.items():
        if key in INFO_LOG_EXCLUDE_KEYS:
            continue

        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            metrics.update(_flatten_info_metrics(value, metric_key))
            continue

        scalar = _coerce_scalar_metric(value)
        if scalar is not None:
            metrics[metric_key] = scalar

    return metrics


class InfoTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        metrics: dict[str, list[float]] = {}
        for info in infos:
            if not isinstance(info, Mapping):
                continue

            for key, value in _flatten_info_metrics(info).items():
                metrics.setdefault(key, []).append(value)

        for key, values in metrics.items():
            self.logger.record(f"custom/{key}", float(np.mean(values)))

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or resume SAC training for a MuJoCo environment."
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_REGISTRY.keys()),
        default="GraspingEnv",
        help="Environment name from ENV_REGISTRY.",
    )
    parser.add_argument(
        "--mode",
        choices=["fresh", "resume"],
        default="fresh",
        help="fresh: train from scratch, resume: continue from a saved model.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name. For fresh, defaults to timestamp. For resume, defaults to checkpoint folder name.",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint path (.zip) or 'latest' when using --mode resume.",
    )
    parser.add_argument(
        "--xml-file",
        default=None,
        help="Path to MuJoCo XML model. Defaults to the scene associated with --env.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Additional timesteps to train.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50_000,
        help="Checkpoint save frequency.",
    )
    parser.add_argument(
        "--keep-replay-buffers",
        type=int,
        default=1,
        help="How many latest replay buffers to keep (older ones are deleted). Use 0 to disable pruning.",
    )
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50_000,
        help="Video recording frequency.",
    )
    parser.add_argument(
        "--video-length",
        type=int,
        default=1000,
        help="Recorded video length.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1000,
        help="TensorBoard custom info logging frequency.",
    )
    parser.add_argument(
        "--gripper-assist-steps",
        type=int,
        default=100_000,
        help="For GraspingEnvV2: linearly fade heuristic gripper assistance to zero over this many timesteps. Use 0 to disable.",
    )
    parser.add_argument(
        "--gripper-action-scale",
        type=float,
        default=0.003,
        help="For GraspingEnvV2: incremental action scale for the learned gripper control.",
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


def list_checkpoints(models_root: Path):
    checkpoints = []
    for zip_path in models_root.glob("**/*.zip"):
        if zip_path.name.endswith("_final.zip"):
            continue
        checkpoints.append(zip_path)
    return sorted(checkpoints, key=lambda path: path.stat().st_mtime)


def resolve_resume_checkpoint(env_name: str, checkpoint_arg: str) -> Path:
    if checkpoint_arg != "latest":
        checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    models_root = PROJECT_ROOT / "logs" / "models" / env_name
    checkpoints = list_checkpoints(models_root)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint found in {models_root}. Use --mode fresh first or pass --checkpoint explicitly."
        )
    return checkpoints[-1]


def resolve_xml_path(env_name: str, xml_file_arg: str | None) -> Path:
    default_xml_path = DEFAULT_XML_BY_ENV[env_name]

    if xml_file_arg is None:
        xml_path = default_xml_path
    else:
        xml_path = Path(xml_file_arg).expanduser()
        if not xml_path.is_absolute():
            xml_path = (PROJECT_ROOT / xml_path).resolve()

    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return xml_path


def build_run_dirs(env_name: str, run_name: str):
    models_dir = PROJECT_ROOT / "logs" / "models" / env_name / run_name
    videos_dir = PROJECT_ROOT / "logs" / "videos" / env_name / run_name
    tb_dir = PROJECT_ROOT / "logs" / "tensorboard" / env_name / run_name

    models_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, videos_dir, tb_dir


def make_env(
    env_name: str,
    xml_file: str,
    render_mode: str = "rgb_array",
    env_kwargs: dict | None = None,
):
    env_cls = ENV_REGISTRY[env_name]
    env_kwargs = env_kwargs or {}

    def _init():
        env = env_cls(xml_file=xml_file, render_mode=render_mode, **env_kwargs)
        return Monitor(env)

    return _init


def build_env(args, videos_dir: Path, name_prefix: str):
    env_kwargs = {}
    if args.env == "GraspingEnvV2":
        env_kwargs.update(
            {
                "gripper_assist_steps": args.gripper_assist_steps,
                "gripper_action_scale": args.gripper_action_scale,
            }
        )
    elif args.env == "PlaceTargetEnv":
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

    env = DummyVecEnv([make_env(args.env, args.xml_file, env_kwargs=env_kwargs)])
    env = VecVideoRecorder(
        env,
        video_folder=str(videos_dir),
        record_video_trigger=lambda step: step % args.video_freq == 0,
        video_length=args.video_length,
        name_prefix=name_prefix,
    )
    return env


def build_new_model(env, tensorboard_log: str):
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=256,
        ent_coef="auto_0.01",
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device="auto",
        policy_kwargs=POLICY_KWARGS,
    )


class PruneReplayBufferCallback(BaseCallback):
    def __init__(
        self, models_dir: Path, save_freq: int, keep_last: int = 1, verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.models_dir = Path(models_dir)
        self.save_freq = int(save_freq)
        self.keep_last = int(keep_last)

    def _on_step(self) -> bool:
        if self.keep_last <= 0:
            return True
        if self.save_freq <= 0:
            return True
        if self.n_calls % self.save_freq != 0:
            return True

        replay_buffers = sorted(
            self.models_dir.glob("*_replay_buffer_*_steps.pkl"),
            key=lambda p: p.stat().st_mtime,
        )
        if len(replay_buffers) <= self.keep_last:
            return True

        to_delete = replay_buffers[: -self.keep_last]
        for path in to_delete:
            try:
                path.unlink()  # permanent delete (no trash)
                if self.verbose:
                    print(f"Deleted old replay buffer: {path}")
            except FileNotFoundError:
                pass
        return True


class GripperAssistScheduleCallback(BaseCallback):
    def _safe_env_method(self, num_timesteps: int) -> None:
        try:
            self.training_env.env_method(
                "set_training_num_timesteps", int(num_timesteps)
            )
        except AttributeError:
            pass

    def _on_training_start(self) -> None:
        self._safe_env_method(self.model.num_timesteps)

    def _on_step(self) -> bool:
        self._safe_env_method(self.num_timesteps)
        return True


def maybe_load_replay_buffer(model: SAC, checkpoint_path: Path):
    # SB3 CheckpointCallback naming is typically:
    # model: <prefix>_<steps>_steps.zip
    # replay buffer: <prefix>_replay_buffer_<steps>_steps.pkl
    stem = checkpoint_path.stem
    m = re.match(r"^(?P<prefix>.+)_(?P<steps>\d+)_steps$", stem)
    candidates = []
    if m:
        prefix = m.group("prefix")
        steps = m.group("steps")
        candidates.append(
            checkpoint_path.parent / f"{prefix}_replay_buffer_{steps}_steps.pkl"
        )

    # Fallback: just take the newest replay buffer in the same folder.
    candidates.extend(
        sorted(
            checkpoint_path.parent.glob("*_replay_buffer_*_steps.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:1]
    )

    replay_buffer_path = next((p for p in candidates if p.exists()), None)
    if replay_buffer_path is None:
        print("Replay buffer not found; continuing without it.")
        return

    model.load_replay_buffer(str(replay_buffer_path))
    print(f"Loaded replay buffer: {replay_buffer_path}")


def main():
    args = parse_args()
    if args.env == "PlaceTargetEnv" and not args.grasp_model:
        raise ValueError(
            "PlaceTargetEnv requires --grasp-model so reset can start from the trained grasping policy state."
        )
    xml_path = resolve_xml_path(args.env, args.xml_file)
    args.xml_file = str(xml_path)

    if args.mode == "resume":
        checkpoint_path = resolve_resume_checkpoint(args.env, args.checkpoint)
        default_run_name = checkpoint_path.parent.name
        video_name_prefix = f"{args.env.lower()}_{checkpoint_path.stem}"
    else:
        checkpoint_path = None
        default_run_name = f"SAC_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
        video_name_prefix = f"{args.env.lower()}_{default_run_name}"

    run_name = args.run_name or default_run_name
    models_dir, videos_dir, tb_dir = build_run_dirs(args.env, run_name)
    env = build_env(args, videos_dir, video_name_prefix)
    config_path = save_env_config(env, args, run_name, models_dir, xml_path)

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(models_dir),
        name_prefix="sac_lift",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    if args.mode == "resume":
        model = SAC.load(
            str(checkpoint_path),
            env=env,
            tensorboard_log=str(tb_dir),
            device="auto",
        )
        maybe_load_replay_buffer(model, checkpoint_path)
        reset_num_timesteps = False
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        model = build_new_model(env, str(tb_dir))
        reset_num_timesteps = True
        print("Starting training from scratch")

    print(f"Using XML scene: {xml_path}")
    print(f"Saved env config to: {config_path}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[
            checkpoint_callback,
            GripperAssistScheduleCallback(),
            InfoTensorboardCallback(log_freq=args.log_freq),
            PruneReplayBufferCallback(
                models_dir=models_dir,
                save_freq=args.save_freq,
                keep_last=args.keep_replay_buffers,
                verbose=1,
            ),
        ],
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=run_name,
    )

    final_model_path = models_dir / "sac_lift_final"
    model.save(str(final_model_path))
    print(f"Saved final model to: {final_model_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
