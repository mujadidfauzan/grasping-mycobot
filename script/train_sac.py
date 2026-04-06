import argparse
import re
from datetime import datetime
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from torch import nn

from source.envs import GraspingEnv, GraspingEnvV1

ENV_REGISTRY = {
    "GraspingEnv": GraspingEnv,
    "GraspingEnvV1": GraspingEnvV1,
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"

POLICY_KWARGS = {
    "net_arch": {
        "pi": [512, 512, 256],
        "qf": [512, 512, 256],
    },
    "activation_fn": nn.ReLU,
}


class InfoTensorboardCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            infos = self.locals.get("infos", [])
            if infos:
                info0 = infos[0]
                for key in [
                    "dist",
                    "reward_dist",
                    "control_penalty",
                    "reward_target",
                    "reward_dist_tanh",
                    "reward_target_tanh",
                    "reward_orient",
                    "stay_bonus",
                    "reward_dist_bonus",
                    "reward_target_bonus",
                ]:
                    if key in info0:
                        self.logger.record(f"custom/{key}", float(info0[key]))
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
        default=str(DEFAULT_XML_PATH),
        help="Path to MuJoCo XML model.",
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


def build_run_dirs(env_name: str, run_name: str):
    models_dir = PROJECT_ROOT / "logs" / "models" / env_name / run_name
    videos_dir = PROJECT_ROOT / "logs" / "videos" / env_name / run_name
    tb_dir = PROJECT_ROOT / "logs" / "tensorboard" / env_name / run_name

    models_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return models_dir, videos_dir, tb_dir


def make_env(env_name: str, xml_file: str, render_mode: str = "rgb_array"):
    env_cls = ENV_REGISTRY[env_name]

    def _init():
        env = env_cls(xml_file=xml_file, render_mode=render_mode)
        return Monitor(env)

    return _init


def build_env(args, videos_dir: Path, name_prefix: str):
    env = DummyVecEnv([make_env(args.env, args.xml_file)])
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

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[
            checkpoint_callback,
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
