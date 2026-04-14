from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import SAC

from sim2real.remote import MyCobotRemote
from sim2real.vision import AprilTagPose

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

DEFAULT_ROBOT_IP = "10.16.121.76"
DEFAULT_MODEL_PATH = (
    "/home/fauzan/Mujoco/Skripsi/logs/models/GraspingEnv/"
    "SAC_26_02_2026_14_27_49/sac_lift_800000_steps.zip"
)
DEFAULT_CAM_INDEX = 2
DEFAULT_BASE_TAG_ID = 12
DEFAULT_OBJ_TAG_ID = 1
DEFAULT_TARGET_POS = np.array([0.18, 0.0, 0.15], dtype=np.float64)

# MuJoCo actuator limits from source/robot/robot.xml
JOINT_LIMITS_DEG = np.rad2deg(
    np.array(
        [
            [-2.9321, 2.9321],
            [-2.3561, 2.3561],
            [-2.6179, 2.6179],
            [-2.5307, 2.5307],
            [-2.8797, 2.8797],
            [-3.1416, 3.1416],
        ],
        dtype=np.float64,
    )
)


@dataclass
class SafetyConfig:
    action_scale_rad: float = 0.01
    action_clip: float = 1.0
    move_speed: int = 20
    loop_dt: float = 0.05
    ack_timeout: float = 0.5
    settle_timeout: float = 4.0
    poll_dt: float = 0.05
    joint_tolerance_deg: float = 1.5
    stable_polls_required: int = 3
    min_command_delta_deg: float = 0.15
    max_step_deg: float = 3.0
    max_consecutive_failures: int = 5
    show_window: bool = True
    object_z_offset_m: float = 0.0


class StateTracker:
    def __init__(self):
        self.obj_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.obj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.has_object_pose = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safer sim2real controller for MyCobot with blocking action gate."
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cam-index", type=int, default=DEFAULT_CAM_INDEX)
    parser.add_argument("--base-tag-id", type=int, default=DEFAULT_BASE_TAG_ID)
    parser.add_argument("--obj-tag-id", type=int, default=DEFAULT_OBJ_TAG_ID)
    parser.add_argument("--move-speed", type=int, default=20)
    parser.add_argument("--ack-timeout", type=float, default=0.5)
    parser.add_argument("--settle-timeout", type=float, default=4.0)
    parser.add_argument("--poll-dt", type=float, default=0.05)
    parser.add_argument("--loop-dt", type=float, default=0.05)
    parser.add_argument("--joint-tolerance-deg", type=float, default=1.5)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--max-step-deg", type=float, default=3.0)
    parser.add_argument("--min-command-delta-deg", type=float, default=0.15)
    parser.add_argument("--action-clip", type=float, default=1.0)
    parser.add_argument("--object-z-offset-m", type=float, default=0.0)
    parser.add_argument(
        "--hide-window",
        action="store_true",
        help="Disable OpenCV preview window from AprilTag vision.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SafetyConfig:
    return SafetyConfig(
        move_speed=int(args.move_speed),
        loop_dt=float(args.loop_dt),
        ack_timeout=float(args.ack_timeout),
        settle_timeout=float(args.settle_timeout),
        poll_dt=float(args.poll_dt),
        joint_tolerance_deg=float(args.joint_tolerance_deg),
        stable_polls_required=max(1, int(args.stable_polls)),
        max_step_deg=float(args.max_step_deg),
        min_command_delta_deg=float(args.min_command_delta_deg),
        action_clip=float(args.action_clip),
        show_window=not bool(args.hide_window),
        object_z_offset_m=float(args.object_z_offset_m),
    )


def _scipy_quat_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64
    )


def wrap_joint_error_deg(current_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    return (current_deg - target_deg + 180.0) % 360.0 - 180.0


def clip_joint_targets_deg(target_deg: np.ndarray) -> np.ndarray:
    lower = JOINT_LIMITS_DEG[:, 0]
    upper = JOINT_LIMITS_DEG[:, 1]
    return np.clip(target_deg, lower, upper)


def is_valid_robot_state(values: np.ndarray, expected_size: int) -> bool:
    return values.shape == (expected_size,) and np.all(np.isfinite(values))


def compute_safe_target_angles_deg(
    current_angles_deg: np.ndarray, action: np.ndarray, cfg: SafetyConfig
) -> np.ndarray:
    action = np.asarray(action[:6], dtype=np.float64)
    action = np.clip(action, -cfg.action_clip, cfg.action_clip)

    delta_deg = np.rad2deg(action * cfg.action_scale_rad)
    delta_deg = np.clip(delta_deg, -cfg.max_step_deg, cfg.max_step_deg)

    target_deg = current_angles_deg + delta_deg
    return clip_joint_targets_deg(target_deg)


def build_observation(
    mc: MyCobotRemote,
    vision: AprilTagPose,
    target_pos: np.ndarray,
    state: StateTracker,
    obj_tag_id: int,
    cfg: SafetyConfig,
) -> np.ndarray | None:
    arm_qpos = np.deg2rad(np.asarray(mc.angles, dtype=np.float64))
    gripper_qpos = np.array([0.02, -0.02], dtype=np.float64)
    robot_qpos = np.concatenate([arm_qpos, gripper_qpos])
    robot_qvel = np.zeros(8, dtype=np.float64)

    tags, _ = vision.get_tag_poses(show_window=cfg.show_window)
    if obj_tag_id in tags:
        obj_pos = np.asarray(tags[obj_tag_id]["pos"], dtype=np.float64).copy()
        obj_pos[2] += cfg.object_z_offset_m
        obj_rpy = np.asarray(tags[obj_tag_id]["rpy"], dtype=np.float64)
        obj_quat = _scipy_quat_to_wxyz(
            R.from_euler("xyz", obj_rpy, degrees=True).as_quat()
        )
        state.obj_pos = obj_pos
        state.obj_quat = obj_quat
        state.has_object_pose = True
    elif state.has_object_pose:
        obj_pos = state.obj_pos
        obj_quat = state.obj_quat
    else:
        return None

    ee_pos = np.asarray(mc.coords[:3], dtype=np.float64) / 1000.0
    ee_rpy = np.deg2rad(np.asarray(mc.coords[3:], dtype=np.float64))
    ee_quat = _scipy_quat_to_wxyz(R.from_euler("xyz", ee_rpy).as_quat())

    rel_obj_ee = obj_pos - ee_pos
    rel_obj_tgt = obj_pos - target_pos

    obs = np.concatenate(
        [
            robot_qpos,
            robot_qvel,
            gripper_qpos,
            obj_pos,
            obj_quat,
            target_pos,
            rel_obj_ee,
            rel_obj_tgt,
            ee_pos,
            ee_quat,
        ]
    ).astype(np.float32)

    if not np.all(np.isfinite(obs)):
        return None
    return obs


def wait_until_target_stable(
    mc: MyCobotRemote, target_deg: np.ndarray, cfg: SafetyConfig
) -> tuple[bool, float]:
    deadline = time.monotonic() + cfg.settle_timeout
    stable_polls = 0
    last_max_error = float("inf")

    while time.monotonic() <= deadline:
        if not mc.update_state():
            stable_polls = 0
            time.sleep(cfg.poll_dt)
            continue

        current_deg = np.asarray(mc.angles, dtype=np.float64)
        if not is_valid_robot_state(current_deg, mc.NUM_JOINTS):
            stable_polls = 0
            time.sleep(cfg.poll_dt)
            continue

        joint_errors = wrap_joint_error_deg(current_deg, target_deg)
        last_max_error = float(np.max(np.abs(joint_errors)))

        if last_max_error <= cfg.joint_tolerance_deg:
            stable_polls += 1
            if stable_polls >= cfg.stable_polls_required:
                return True, last_max_error
        else:
            stable_polls = 0

        time.sleep(cfg.poll_dt)

    return False, last_max_error


def main():
    args = parse_args()
    cfg = build_config(args)
    target_pos = DEFAULT_TARGET_POS.copy()

    mc = MyCobotRemote(args.robot_ip)
    model = SAC.load(args.model_path)
    vision = AprilTagPose(base_id=args.base_tag_id, cam_index=args.cam_index)
    state = StateTracker()

    consecutive_failures = 0

    try:
        mc.power_on()
        time.sleep(2.0)
        print("Safer sim2real controller started. Press Ctrl-C to stop.")

        while True:
            if not mc.update_state():
                consecutive_failures += 1
                print(f"State update failed ({consecutive_failures}).")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many robot state failures.")
                time.sleep(cfg.loop_dt)
                continue

            current_angles_deg = np.asarray(mc.angles, dtype=np.float64)
            current_coords = np.asarray(mc.coords, dtype=np.float64)
            if not is_valid_robot_state(current_angles_deg, mc.NUM_JOINTS):
                consecutive_failures += 1
                print("Invalid joint state received. Skipping cycle.")
                time.sleep(cfg.loop_dt)
                continue
            if not is_valid_robot_state(current_coords, mc.NUM_JOINTS):
                consecutive_failures += 1
                print("Invalid Cartesian state received. Skipping cycle.")
                time.sleep(cfg.loop_dt)
                continue

            obs = build_observation(
                mc=mc,
                vision=vision,
                target_pos=target_pos,
                state=state,
                obj_tag_id=args.obj_tag_id,
                cfg=cfg,
            )
            if obs is None:
                consecutive_failures += 1
                print("Observation incomplete. Waiting for a valid object pose.")
                time.sleep(cfg.loop_dt)
                continue

            action, _ = model.predict(obs, deterministic=True)
            target_deg = compute_safe_target_angles_deg(current_angles_deg, action, cfg)
            command_delta = float(np.max(np.abs(target_deg - current_angles_deg)))

            if command_delta < cfg.min_command_delta_deg:
                consecutive_failures = 0
                time.sleep(cfg.loop_dt)
                continue

            ack_ok = mc.send_angles(
                target_deg.tolist(),
                cfg.move_speed,
                wait=False,
                ack_timeout=cfg.ack_timeout,
            )
            if not ack_ok:
                consecutive_failures += 1
                print(f"SET_ANGLES ACK failed ({consecutive_failures}).")
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many command ACK failures.")
                time.sleep(cfg.loop_dt)
                continue

            settled, max_error = wait_until_target_stable(mc, target_deg, cfg)
            if not settled:
                consecutive_failures += 1
                print(
                    "Target not settled before timeout. "
                    f"max_error_deg={max_error:.3f} failures={consecutive_failures}"
                )
                if consecutive_failures >= cfg.max_consecutive_failures:
                    raise RuntimeError("Too many settle failures.")
            else:
                consecutive_failures = 0
                print(
                    "Target settled. "
                    f"delta_deg={command_delta:.3f} max_error_deg={max_error:.3f}"
                )

            time.sleep(cfg.loop_dt)

    except KeyboardInterrupt:
        print("Stop signal received.")
    finally:
        mc.stop()
        vision.release()
        print("Safer sim2real controller stopped.")


if __name__ == "__main__":
    main()
