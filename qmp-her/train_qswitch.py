from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from qmpher.utils import resolve_repo_path

DEFAULT_XML = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
DEFAULT_GRASP_XML = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
DEFAULT_GRASP_MODEL = PROJECT_ROOT / "melogs" / "ik_models" / "grasp-ik-model.zip"
DEFAULT_INSERT_MODEL = PROJECT_ROOT / "melogs" / "ik_models" / "insert-ik-model.zip"
DEFAULT_RUN_ROOT = THIS_DIR / "runs"


POLICY_KWARGS = {
    "net_arch": {
        "pi": [512, 512, 256],
        "qf": [512, 512, 256],
    },
    "activation_fn": nn.ReLU,
}


def _range_arg(values: list[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise argparse.ArgumentTypeError("range arguments require exactly two floats")
    return float(values[0]), float(values[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a SAC target policy with Q-switch critic selection over "
            "GraspingEnvIK and InsertTargetEnvIK primitive policies."
        )
    )
    parser.add_argument("--xml-file", default=str(DEFAULT_XML))
    parser.add_argument("--grasp-xml-file", default=str(DEFAULT_GRASP_XML))
    parser.add_argument("--grasp-model", default=str(DEFAULT_GRASP_MODEL))
    parser.add_argument("--insert-model", default=str(DEFAULT_INSERT_MODEL))
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume", default=None, help="Optional target QSwitchSAC checkpoint."
    )

    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-sampled-goal", type=int, default=4)
    parser.add_argument(
        "--goal-selection-strategy",
        choices=["future", "final", "episode"],
        default="future",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ent-coef", default="auto_0.01")
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--render-mode", default="none", choices=["none", "human", "rgb_array"]
    )

    parser.add_argument("--object-x-range", nargs=2, type=float, default=(0.15, 0.27))
    parser.add_argument("--object-y-range", nargs=2, type=float, default=(-0.12, 0.12))
    parser.add_argument("--object-z", type=float, default=0.025)
    parser.add_argument(
        "--object-yaw-range", nargs=2, type=float, default=(-np.pi / 4, np.pi / 4)
    )
    parser.add_argument("--reset-settle-steps", type=int, default=20)

    parser.add_argument("--manual-close-distance", type=float, default=0.01)
    parser.add_argument("--manual-close-angle-deg", type=float, default=None)
    parser.add_argument("--manual-release-distance", type=float, default=0.01)
    parser.add_argument("--manual-release-angle-deg", type=float, default=10.0)
    parser.add_argument(
        "--success-steps-required",
        type=int,
        default=5,
        help="Consecutive target-aligned steps required before ending as success.",
    )

    parser.add_argument("--epsilon-initial", type=float, default=0.20)
    parser.add_argument("--epsilon-final", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--include-target-during-warmup", action="store_true")
    parser.add_argument("--disable-target-after-warmup", action="store_true")
    parser.add_argument("--include-zero-action", action="store_true")
    parser.add_argument(
        "--q-aggregation",
        choices=["min", "mean", "q1"],
        default="min",
        help="How to aggregate SAC twin critic values for Q-switch selection.",
    )
    parser.add_argument("--stochastic-primitives", action="store_true")
    parser.add_argument(
        "--non-strict-primitives",
        action="store_true",
        help="Skip primitive candidates instead of raising when an adapter fails.",
    )

    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument(
        "--video-freq",
        type=int,
        default=50_000,
        help="Record one rollout video every N env steps. Use 0 to disable.",
    )
    parser.add_argument("--video-length", type=int, default=1_000)
    parser.add_argument("--log-freq", type=int, default=1_000)
    parser.add_argument("--print-freq", type=int, default=5_000)
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def build_env(args: argparse.Namespace, videos_dir: Path, name_prefix: str):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

    from qmpher.envs import QMPGraspInsertEnv

    xml_file = _require_file(resolve_repo_path(args.xml_file), "Insert XML")
    record_video = args.video_freq > 0
    render_mode = (
        "rgb_array"
        if record_video
        else None if args.render_mode == "none" else args.render_mode
    )

    def _make_env():
        env = QMPGraspInsertEnv(
            xml_file=str(xml_file),
            render_mode=render_mode,
            object_x_range=_range_arg(list(args.object_x_range)),
            object_y_range=_range_arg(list(args.object_y_range)),
            object_z=args.object_z,
            object_yaw_range=_range_arg(list(args.object_yaw_range)),
            reset_settle_steps=args.reset_settle_steps,
            close_distance=args.manual_close_distance,
            close_angle_deg=args.manual_close_angle_deg,
            release_distance=args.manual_release_distance,
            release_angle_deg=args.manual_release_angle_deg,
            success_steps_required=args.success_steps_required,
        )
        return Monitor(env)

    env = DummyVecEnv([_make_env])
    if record_video:
        env = VecVideoRecorder(
            env,
            video_folder=str(videos_dir),
            record_video_trigger=lambda step: step % args.video_freq == 0,
            video_length=args.video_length,
            name_prefix=name_prefix,
        )
    return env


def build_primitives(args: argparse.Namespace) -> PrimitiveEnsemble:
    from qmpher.primitives import (
        GraspPrimitiveAdapter,
        InsertPrimitiveAdapter,
        PrimitiveEnsemble,
    )

    grasp_model = _require_file(resolve_repo_path(args.grasp_model), "Grasp model")
    insert_model = _require_file(resolve_repo_path(args.insert_model), "Insert model")
    grasp_xml = _require_file(resolve_repo_path(args.grasp_xml_file), "Grasp XML")
    deterministic = not bool(args.stochastic_primitives)
    strict = not bool(args.non_strict_primitives)

    return PrimitiveEnsemble(
        [
            GraspPrimitiveAdapter(
                model_path=grasp_model,
                xml_file=grasp_xml,
                label="primitive_grasp",
                deterministic=deterministic,
                device=args.device,
                env_kwargs={"post_grasp_mode": "off"},
                strict=strict,
            ),
            InsertPrimitiveAdapter(
                model_path=insert_model,
                label="primitive_insert",
                deterministic=deterministic,
                device=args.device,
                strict=strict,
            ),
        ]
    )


def build_qswitch_config(args: argparse.Namespace):
    from qmpher.q_switch_sac import QSwitchConfig

    return QSwitchConfig(
        enabled=True,
        include_target_during_warmup=bool(args.include_target_during_warmup),
        include_target_after_learning_starts=not bool(args.disable_target_after_warmup),
        include_zero_action=bool(args.include_zero_action),
        epsilon_initial=args.epsilon_initial,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        # Penting: epsilon tidak boleh raw random action untuk IK 6D.
        # Epsilon hanya memilih random dari candidate yang sudah valid.
        epsilon_random_action=False,
        q_aggregation=args.q_aggregation,
    )


def save_run_config(args: argparse.Namespace, run_dir: Path) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "policy_kwargs": {
            "net_arch": POLICY_KWARGS["net_arch"],
            "activation_fn": POLICY_KWARGS["activation_fn"].__name__,
        },
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    from stable_baselines3.common.callbacks import CheckpointCallback

    try:
        from stable_baselines3 import HerReplayBuffer
    except ImportError:
        from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

    from qmpher.callbacks import PrintQSwitchCallback, QSwitchInfoCallback
    from qmpher.q_switch_sac import QSwitchSAC

    run_name = (
        args.run_name or f"q_switch_sac_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
    )
    run_root = resolve_repo_path(args.run_root)
    run_dir = run_root / run_name
    model_dir = run_dir / "models"
    tb_dir = run_dir / "tensorboard"
    videos_dir = run_dir / "videos"
    model_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(args, run_dir)

    env = build_env(args, videos_dir, run_name)
    primitive_ensemble = build_primitives(args)
    qswitch_config = build_qswitch_config(args)

    if args.resume:
        model = QSwitchSAC.load(
            str(resolve_repo_path(args.resume)),
            env=env,
            device=args.device,
            tensorboard_log=str(tb_dir),
        )
        model.set_qswitch_ensemble(primitive_ensemble, qswitch_config)
        reset_num_timesteps = False
        print(f"Resuming target policy from: {args.resume}")
    else:
        model = QSwitchSAC(
            "MultiInputPolicy",
            env,
            primitive_ensemble=primitive_ensemble,
            qswitch_config=qswitch_config,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": args.n_sampled_goal,
                "goal_selection_strategy": args.goal_selection_strategy,
            },
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            ent_coef=args.ent_coef,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            verbose=1,
            tensorboard_log=str(tb_dir),
            device=args.device,
            policy_kwargs=POLICY_KWARGS,
            seed=args.seed,
        )
        reset_num_timesteps = True
        print("Starting Q-switch SAC training from scratch")

    callbacks = [
        QSwitchInfoCallback(log_freq=args.log_freq),
        PrintQSwitchCallback(print_freq=args.print_freq),
    ]
    if args.save_freq > 0:
        callbacks.insert(
            0,
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(model_dir),
                name_prefix="q_switch_sac",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
        )

    print(f"Run directory: {run_dir}")
    print(f"Primitive grasp model: {resolve_repo_path(args.grasp_model)}")
    print(f"Primitive insert model: {resolve_repo_path(args.insert_model)}")
    if args.video_freq > 0:
        print(f"Recording videos every {args.video_freq} steps to: {videos_dir}")
    print("Manual gripper is active during training.")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=run_name,
        )
        final_path = model_dir / "q_switch_sac_final"
        model.save(str(final_path))
        print(f"Saved final model to: {final_path}.zip")
    finally:
        primitive_ensemble.close()
        env.close()


if __name__ == "__main__":
    main()
