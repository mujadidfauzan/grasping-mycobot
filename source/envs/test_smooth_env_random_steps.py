"""Quick test script for GraspingEnvIK smooth control.

Cara pakai yang disarankan:
1. Ganti file env lama dengan grasping_env_ik_smooth.py, atau copy isi file smooth
   ke modul env kamu yang biasa diimport.
2. Jalankan script ini dari project root.
3. Sesuaikan import di bawah sesuai struktur project kamu.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# Sesuaikan import ini dengan struktur project kamu.
# Contoh kalau env berada di source/envs/grasping_env_ik.py:
try:
    from source.envs.grasping_env_ik_v2 import GraspingEnvIKV2
except ModuleNotFoundError:
    # Fallback kalau kamu meletakkan script ini satu folder dengan file env.
    from source.envs.grasping_env_ik_v2 import GraspingEnvIKV2  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-file", type=str, default=None)
    parser.add_argument("--steps", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--action-range", type=float, default=1.0)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--interpolation-steps", type=int, default=10)
    parser.add_argument("--cartesian-scale", type=float, default=0.03)
    parser.add_argument("--rotation-scale-deg", type=float, default=5.0)
    parser.add_argument("--max-joint-delta-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    env_kwargs = dict(
        render_mode=args.render_mode,
        max_episode_steps=args.steps,
        frame_skip=1,
        control_interpolation_steps=args.interpolation_steps,
        cartesian_action_scale=args.cartesian_scale,
        cartesian_rotation_scale_deg=args.rotation_scale_deg,
        ik_step_size=0.4,
        ik_max_delta_deg=5.0,
        max_joint_ctrl_delta_deg=args.max_joint_delta_deg,
        smooth_cartesian_target=True,
        debug_ik=False,
    )

    if args.xml_file is not None:
        env_kwargs["xml_file"] = str(Path(args.xml_file).expanduser().resolve())

    env = GraspingEnvIKV2(**env_kwargs)
    obs, info = env.reset(seed=args.seed)

    print("Initial obs shape:", obs.shape)
    print("Action space:", env.action_space)
    print("Testing random wide actions...")

    for step_idx in range(args.steps):
        # Sample wide normalized action. Env will clip to action_space if needed.
        action = rng.uniform(
            low=-args.action_range,
            high=args.action_range,
            size=env.action_space.shape,
        ).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        debug = env.get_debug_state()

        print(
            f"step={step_idx + 1:03d} "
            f"reward={reward: .4f} "
            f"ee_obj={info.get('ee_object_dist', float('nan')):.4f} "
            f"obj_target={info.get('object_target_dist', float('nan')):.4f} "
            f"ik_success={info.get('ik_success')} "
            f"ik_fail={info.get('ik_failure_count')} "
            f"gripper={debug.get('gripper_state')}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

        if terminated or truncated:
            print("Done:", {"terminated": terminated, "truncated": truncated})
            break

    env.close()


if __name__ == "__main__":
    main()
