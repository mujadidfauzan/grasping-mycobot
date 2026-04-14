from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OBJECT_LIFT_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_lift.xml"
OBJECT_PLACE_XML_PATH = PROJECT_ROOT / "source" / "robot" / "object_place.xml"
ENV_NAMES = ["GraspingEnvV2", "PlaceTargetEnv", "ReachingEnv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug viewer for pose targets and frame alignment."
    )
    parser.add_argument(
        "--env",
        choices=ENV_NAMES,
        default="ReachingEnv",
        help="Environment name.",
    )
    parser.add_argument(
        "--xml-file",
        default=None,
        help="Path to MuJoCo XML model. Defaults to the scene associated with --env.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for env.reset().",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60.0,
        help="Viewer refresh rate.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=1.0,
        help="Print pose/frame debug info every N seconds.",
    )
    parser.add_argument(
        "--reset-every",
        type=float,
        default=0.0,
        help="Auto-reset episode every N seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--mode",
        choices=["hold", "zero", "random"],
        default="hold",
        help="hold: static after reset, zero: step zero action, random: step random action.",
    )
    parser.add_argument(
        "--frame",
        choices=["none", "site", "body", "geom"],
        default="site",
        help="Coordinate frames to draw in the MuJoCo viewer.",
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


def quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    w, x, y, z = quat

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def format_array(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float64), precision=4, suppress_small=True
    )


def print_debug_state(env) -> None:
    state = env.get_debug_state()
    ee_rpy_deg = np.rad2deg(quat_to_euler_xyz(state["ee_quat"]))
    print()

    if "obj_quat" in state:
        obj_rpy_deg = np.rad2deg(quat_to_euler_xyz(state["obj_quat"]))
        print(f"[Object] {state['active_object']}")
        print(
            f"EE     pos={format_array(state['ee_pos'])} quat={format_array(state['ee_quat'])} "
            f"rpy_deg={format_array(ee_rpy_deg)}"
        )
        print(
            f"Object pos={format_array(state['obj_pos'])} quat={format_array(state['obj_quat'])} "
            f"rpy_deg={format_array(obj_rpy_deg)}"
        )
        print(
            f"EE->Obj pos_err={format_array(state['ee_obj_pos_error'])} "
            f"dist={state['ee_obj_dist']:.4f} m "
            f"(limit={state['terminate_ee_obj_distance']:.4f}, too_far={state['ee_obj_too_far']})"
        )
        print(
            f"EE->Obj rot_err={format_array(state['ee_obj_rot_error'])} "
            f"angle={np.rad2deg(state['ee_obj_angle_rad']):.2f} deg"
        )
        print(
            f"Obj->Target pos_err={format_array(state['obj_target_pos_error'])} "
            f"dist={state['obj_target_dist']:.4f} m"
        )
        print(
            f"Obj->Target rot_err={format_array(state['obj_target_rot_error'])} "
            f"angle={np.rad2deg(state['obj_target_angle_rad']):.2f} deg"
        )
        print(
            f"Yaw sampled={np.rad2deg(state['sampled_object_yaw']):.2f} deg "
            f"applied={np.rad2deg(state['applied_object_yaw']):.2f} deg "
            f"current={np.rad2deg(state['object_yaw']):.2f} deg"
        )
        print(
            f"Gripper assist_mix={state['gripper_assist_mix']:.3f} "
            f"should_close={state['gripper_should_close']}"
        )
        if "target_place_pos" in state:
            print(
                f"Target place pos={format_array(state['target_place_pos'])} "
                f"yaw_sampled={np.rad2deg(state.get('sampled_target_place_yaw', 0.0)):.2f} deg "
                f"yaw_applied={np.rad2deg(state.get('applied_target_place_yaw', 0.0)):.2f} deg "
                f"yaw_current={np.rad2deg(state.get('target_place_yaw', 0.0)):.2f} deg "
                f"init_source={state.get('grasp_reset_source', 'n/a')} "
                f"attempts={state.get('grasp_reset_attempts', 'n/a')}"
            )
        return

    target_rpy_deg = np.rad2deg(quat_to_euler_xyz(state["target_quat"]))
    print("[Task] Reaching")
    print(
        f"EE     pos={format_array(state['ee_pos'])} quat={format_array(state['ee_quat'])} "
        f"rpy_deg={format_array(ee_rpy_deg)}"
    )
    print(
        f"Target pos={format_array(state['target_pos'])} quat={format_array(state['target_quat'])} "
        f"rpy_deg={format_array(target_rpy_deg)}"
    )
    print(
        f"EE->Target pos_err={format_array(state['ee_target_pos_error'])} "
        f"dist={state['ee_target_dist']:.4f} m"
    )
    print(
        f"EE->Target rot_err={format_array(state['ee_target_rot_error'])} "
        f"angle={np.rad2deg(state['ee_target_angle_rad']):.2f} deg"
    )
    print(
        f"Target delta rpy_deg={format_array(state['target_delta_euler_deg'])} "
        f"success_counter={state['success_counter']}"
    )
    print(f"Last action={format_array(state['last_action'])}")


def resolve_frame_option(mujoco, frame_name: str):
    if frame_name == "none":
        return mujoco.mjtFrame.mjFRAME_NONE
    if frame_name == "site":
        return mujoco.mjtFrame.mjFRAME_SITE
    if frame_name == "body":
        return mujoco.mjtFrame.mjFRAME_BODY
    if frame_name == "geom":
        return mujoco.mjtFrame.mjFRAME_GEOM
    raise ValueError(f"Unsupported frame option: {frame_name}")


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


def main() -> None:
    args = parse_args()
    if args.env == "PlaceTargetEnv" and not args.grasp_model:
        raise ValueError(
            "PlaceTargetEnv requires --grasp-model so reset can start from the trained grasping policy state."
        )

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mujoco is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    from source.envs import GraspingEnvV2, PlaceTargetEnv, ReachingEnv

    env_registry = {
        "GraspingEnvV2": GraspingEnvV2,
        "PlaceTargetEnv": PlaceTargetEnv,
        "ReachingEnv": ReachingEnv,
    }

    xml_path = resolve_xml_path(args.env, args.xml_file)

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
    env = env_registry[args.env](xml_file=str(xml_path), render_mode=None, **env_kwargs)
    env.reset(seed=args.seed)
    if hasattr(env, "sync_visual_frames"):
        env.sync_visual_frames()

    frame_option = resolve_frame_option(mujoco, args.frame)
    step_dt = 1.0 / max(args.fps, 1.0)
    last_print = 0.0
    last_reset = time.time()

    print(f"[OK] XML  : {xml_path}")
    print(f"[OK] Env  : {args.env}")
    print(f"[OK] Mode : {args.mode}")
    print(f"[OK] Frame: {args.frame}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.frame = frame_option

        while viewer.is_running():
            loop_start = time.time()

            if args.mode == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                _obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    env.reset()
                    last_reset = time.time()
            elif args.mode == "random":
                action = env.action_space.sample()
                _obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    env.reset()
                    last_reset = time.time()
            else:
                if hasattr(env, "sync_visual_frames"):
                    env.sync_visual_frames()
                else:
                    mujoco.mj_forward(env.model, env.data)

            now = time.time()
            if args.reset_every > 0.0 and (now - last_reset) >= args.reset_every:
                env.reset()
                if hasattr(env, "sync_visual_frames"):
                    env.sync_visual_frames()
                last_reset = now

            if (now - last_print) >= args.print_every:
                print_debug_state(env)
                last_print = now

            viewer.sync()

            elapsed = time.time() - loop_start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    env.close()


if __name__ == "__main__":
    main()
