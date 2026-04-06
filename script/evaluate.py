from __future__ import annotations

import argparse
from pathlib import Path

ENV_NAMES = [
    "GraspingEnv",
    "GraspingEnvV1",
    "GraspingEnvV2",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a trained SAC policy in the MuJoCo environment."
    )
    parser.add_argument(
        "--env",
        choices=sorted(ENV_NAMES),
        default="GraspingEnv",
        help="Environment name.",
    )
    parser.add_argument(
        "--xml-file",
        default=str(DEFAULT_XML_PATH),
        help="Path to MuJoCo XML model.",
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import SAC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        from source.envs import GraspingEnv, GraspingEnvV1, GraspingEnvV2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to import environments from source.envs. Run this script from the project root "
            "or ensure the project root is on PYTHONPATH."
        ) from exc

    env_registry = {
        "GraspingEnv": GraspingEnv,
        "GraspingEnvV1": GraspingEnvV1,
        "GraspingEnvV2": GraspingEnvV2,
    }

    xml_path = Path(args.xml_file).expanduser()
    if not xml_path.is_absolute():
        xml_path = (PROJECT_ROOT / xml_path).resolve()
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model
        else resolve_latest_model(args.env)
    )
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    render_mode = None if args.render == "none" else args.render
    env_cls = env_registry[args.env]
    env = env_cls(xml_file=str(xml_path), render_mode=render_mode)

    # Attach env to ensure action/obs spaces match what we are evaluating.
    model = SAC.load(str(model_path), env=env, device="auto")

    deterministic = not args.stochastic
    max_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(getattr(env, "max_episode_steps", 500))
    )

    print(f"[OK] Model: {model_path}")
    print(f"[OK] XML  : {xml_path}")
    print(
        f"[OK] Env  : {args.env} (render={args.render}, deterministic={deterministic})"
    )

    for ep in range(args.episodes):
        episode_seed = (args.seed + ep) if args.seed is not None else None
        obs, _info = env.reset(seed=episode_seed)
        for _step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _reward, terminated, truncated, _step_info = env.step(action)
            # if terminated or truncated:
            #     print(
            #         f"Terminated episode {ep + 1} at step {_step + 1} (terminated={terminated}, truncated={truncated})"
            #     )
            #     break

    env.close()
    print("[OK] Done.")


if __name__ == "__main__":
    main()
