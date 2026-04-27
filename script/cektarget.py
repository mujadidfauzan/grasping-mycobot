from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    import mujoco.viewer
except ImportError:  # pragma: no cover - viewer is optional in headless setups
    mujoco.viewer = None


MODEL_PATH = Path(__file__).resolve().parents[1] / "source" / "robot" / "object_place.xml"
OBJECT_REFERENCE_Z_OFFSET = -0.015
TARGET_PLACE_REFERENCE_Z_OFFSET = 0.015
DEFAULT_SUCCESS_DISTANCE = 0.01
DEFAULT_SUCCESS_ANGLE_DEG = 10.0
CENTER_PLACE_BODY_POS = np.array([0.23, 0.0, 0.0], dtype=np.float64)
SIDE_PLACE_BODY_POSITIONS = [
    np.array([0.34, 0.18, 0.0], dtype=np.float64),
    np.array([0.34, -0.18, 0.0], dtype=np.float64),
]
SIDE_OBJECT_BODY_POSES = [
    (
        np.array([1.10, 0.45, 0.08], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ),
    (
        np.array([1.10, -0.45, 0.08], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    ),
]
ACTIVE_OBJECT_STANDBY_BODY_POSE = (
    np.array([0.50, 0.0, 0.06], dtype=np.float64),
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
)


@dataclass(frozen=True)
class ObjectCheckConfig:
    body_name: str
    geom_name: str
    site_name: str
    place_body_name: str
    place_geom_name: str
    place_site_name: str


OBJECT_CONFIGS: dict[str, ObjectCheckConfig] = {
    "obj_box": ObjectCheckConfig(
        body_name="obj_box",
        geom_name="obj_box_geom",
        site_name="obj_box_ref",
        place_body_name="cube_place",
        place_geom_name="cube_place_geom",
        place_site_name="cube_place_site",
    ),
    "obj_triangle": ObjectCheckConfig(
        body_name="obj_triangle",
        geom_name="obj_triangle_geom",
        site_name="obj_triangle_ref",
        place_body_name="tri_place",
        place_geom_name="tri_place_geom",
        place_site_name="tri_place_site",
    ),
    "obj_cylinder": ObjectCheckConfig(
        body_name="obj_cylinder",
        geom_name="obj_cylinder_geom",
        site_name="obj_cylinder_ref",
        place_body_name="cyl_place",
        place_geom_name="cyl_place_geom",
        place_site_name="cyl_place_site",
    ),
}

MODEL = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
TARGET_SITE_ID = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_SITE, "target")
TARGET_BODY_ID = int(MODEL.site_bodyid[TARGET_SITE_ID])
TARGET_SITE_LOCAL_POS = MODEL.site_pos[TARGET_SITE_ID].copy()
TARGET_SITE_LOCAL_QUAT = MODEL.site_quat[TARGET_SITE_ID].copy()


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
    wa, xa, ya, za = quat_a
    wb, xb, yb, zb = quat_b
    return np.array(
        [
            wa * wb - xa * xb - ya * yb - za * zb,
            wa * xb + xa * wb + ya * zb - za * yb,
            wa * yb - xa * zb + ya * wb + za * xb,
            wa * zb + xa * yb - ya * xb + za * wb,
        ],
        dtype=np.float64,
    )


def quat_rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = normalize_quat(quat)
    vec_quat = np.array([0.0, *np.asarray(vec, dtype=np.float64)], dtype=np.float64)
    rotated = quat_multiply(quat_multiply(quat, vec_quat), quat_conjugate(quat))
    return rotated[1:]


def rotation_vector(source_quat: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
    source_quat = normalize_quat(source_quat)
    target_quat = normalize_quat(target_quat)
    delta = normalize_quat(quat_multiply(target_quat, quat_conjugate(source_quat)))
    if delta[0] < 0.0:
        delta = -delta

    xyz = delta[1:]
    sin_half = np.linalg.norm(xyz)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)

    angle = 2.0 * np.arctan2(sin_half, np.clip(delta[0], -1.0, 1.0))
    axis = xyz / sin_half
    return axis * angle


def get_pose_error(
    source_pos: np.ndarray,
    source_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pos_error = np.asarray(target_pos, dtype=np.float64) - np.asarray(
        source_pos, dtype=np.float64
    )
    rot_error = rotation_vector(source_quat, target_quat)
    return pos_error, rot_error


def get_site_quat(data: mujoco.MjData, site_name: str) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.site(site_name).xmat)
    return normalize_quat(quat)


def get_site_pose(data: mujoco.MjData, site_name: str) -> tuple[np.ndarray, np.ndarray]:
    return data.site(site_name).xpos.copy(), get_site_quat(data, site_name)


def offset_pose_along_local_z(
    pos: np.ndarray, quat: np.ndarray, z_offset: float
) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float64).copy()
    quat = normalize_quat(np.asarray(quat, dtype=np.float64))
    if z_offset != 0.0:
        pos += quat_rotate_vector(
            quat,
            np.array([0.0, 0.0, float(z_offset)], dtype=np.float64),
        )
    return pos, quat


def get_pose_in_body_frame(
    world_pos: np.ndarray,
    world_quat: np.ndarray,
    body_pos: np.ndarray,
    body_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    body_quat = normalize_quat(np.asarray(body_quat, dtype=np.float64))
    body_quat_conj = quat_conjugate(body_quat)
    local_pos = quat_rotate_vector(
        body_quat_conj,
        np.asarray(world_pos, dtype=np.float64) - np.asarray(body_pos, dtype=np.float64),
    )
    local_quat = normalize_quat(
        quat_multiply(body_quat_conj, np.asarray(world_quat, dtype=np.float64))
    )
    return local_pos, local_quat


def pose_to_body_transform(
    world_pos: np.ndarray,
    world_quat: np.ndarray,
    local_pos: np.ndarray,
    local_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_pos = np.asarray(world_pos, dtype=np.float64)
    world_quat = normalize_quat(np.asarray(world_quat, dtype=np.float64))
    local_pos = np.asarray(local_pos, dtype=np.float64)
    local_quat = normalize_quat(np.asarray(local_quat, dtype=np.float64))
    body_quat = normalize_quat(quat_multiply(world_quat, quat_conjugate(local_quat)))
    body_pos = world_pos - quat_rotate_vector(body_quat, local_pos)
    return body_pos, body_quat


def freejoint_qposadr(body_name: str) -> int:
    body_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = int(MODEL.body_jntadr[body_id])
    if joint_id < 0:
        raise ValueError(f"Body `{body_name}` does not have a joint.")
    return int(MODEL.jnt_qposadr[joint_id])


def set_free_body_pose(
    data: mujoco.MjData,
    body_name: str,
    pos: np.ndarray,
    quat: np.ndarray,
) -> None:
    qpos_adr = freejoint_qposadr(body_name)
    data.qpos[qpos_adr : qpos_adr + 3] = np.asarray(pos, dtype=np.float64)
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = normalize_quat(quat)


def set_static_body_pose(body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_BODY, body_name)
    MODEL.body_pos[body_id] = np.asarray(pos, dtype=np.float64)
    MODEL.body_quat[body_id] = normalize_quat(quat)


def set_geom_collision_enabled(geom_name: str, enabled: bool) -> None:
    geom_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if enabled:
        MODEL.geom_contype[geom_id] = 1
        MODEL.geom_conaffinity[geom_id] = 1
    else:
        MODEL.geom_contype[geom_id] = 0
        MODEL.geom_conaffinity[geom_id] = 0


def set_object_pose_from_reference_pose(
    data: mujoco.MjData,
    config: ObjectCheckConfig,
    ref_world_pos: np.ndarray,
    ref_world_quat: np.ndarray,
) -> None:
    site_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_SITE, config.site_name)
    local_site_pos = MODEL.site_pos[site_id].copy()
    local_site_quat = normalize_quat(MODEL.site_quat[site_id].copy())
    local_ref_pos, local_ref_quat = offset_pose_along_local_z(
        local_site_pos, local_site_quat, OBJECT_REFERENCE_Z_OFFSET
    )
    body_pos, body_quat = pose_to_body_transform(
        ref_world_pos,
        ref_world_quat,
        local_ref_pos,
        local_ref_quat,
    )
    set_free_body_pose(data, config.body_name, body_pos, body_quat)


def hide_inactive_objects(data: mujoco.MjData, active_body_name: str) -> None:
    inactive_body_names = [
        body_name for body_name in OBJECT_CONFIGS if body_name != active_body_name
    ]
    for body_name, (pos, quat) in zip(inactive_body_names, SIDE_OBJECT_BODY_POSES):
        set_free_body_pose(data, body_name, pos, quat)


def sync_target_site_to_active_place(
    data: mujoco.MjData, config: ObjectCheckConfig
) -> None:
    target_pos, target_quat = get_active_place_pose(data, config)
    target_body_pos, target_body_quat = pose_to_body_transform(
        target_pos,
        target_quat,
        TARGET_SITE_LOCAL_POS,
        TARGET_SITE_LOCAL_QUAT,
    )
    MODEL.body_pos[TARGET_BODY_ID] = target_body_pos
    MODEL.body_quat[TARGET_BODY_ID] = normalize_quat(target_body_quat)


def arrange_scene_layout(data: mujoco.MjData, active_body_name: str) -> None:
    active_config = OBJECT_CONFIGS[active_body_name]
    identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    set_static_body_pose(active_config.place_body_name, CENTER_PLACE_BODY_POS, identity_quat)

    inactive_place_body_names = [
        config.place_body_name
        for body_name, config in OBJECT_CONFIGS.items()
        if body_name != active_body_name
    ]
    for place_body_name, place_pos in zip(
        inactive_place_body_names, SIDE_PLACE_BODY_POSITIONS
    ):
        set_static_body_pose(place_body_name, place_pos, identity_quat)

    hide_inactive_objects(data, active_body_name)
    active_standby_pos, active_standby_quat = ACTIVE_OBJECT_STANDBY_BODY_POSE
    set_free_body_pose(data, active_body_name, active_standby_pos, active_standby_quat)

    mujoco.mj_forward(MODEL, data)
    sync_target_site_to_active_place(data, active_config)
    mujoco.mj_forward(MODEL, data)


def count_contacts_between_geoms(
    data: mujoco.MjData, geom_name_1: str, geom_name_2: str
) -> tuple[int, float | None]:
    geom_1_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_GEOM, geom_name_1)
    geom_2_id = mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_GEOM, geom_name_2)
    contact_count = 0
    min_dist: float | None = None

    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        pair = {int(contact.geom1), int(contact.geom2)}
        if pair != {geom_1_id, geom_2_id}:
            continue

        contact_count += 1
        contact_dist = float(contact.dist)
        if min_dist is None or contact_dist < min_dist:
            min_dist = contact_dist

    return contact_count, min_dist


def get_active_object_pose(
    data: mujoco.MjData, config: ObjectCheckConfig
) -> tuple[np.ndarray, np.ndarray]:
    obj_pos, obj_quat = get_site_pose(data, config.site_name)
    return offset_pose_along_local_z(obj_pos, obj_quat, OBJECT_REFERENCE_Z_OFFSET)


def get_active_place_pose(
    data: mujoco.MjData, config: ObjectCheckConfig
) -> tuple[np.ndarray, np.ndarray]:
    place_pos, place_quat = get_site_pose(data, config.place_site_name)
    return offset_pose_along_local_z(
        place_pos,
        place_quat,
        TARGET_PLACE_REFERENCE_Z_OFFSET,
    )


def collect_insertion_metrics(
    data: mujoco.MjData,
    config: ObjectCheckConfig,
    success_distance: float,
    success_angle_rad: float,
) -> dict[str, float | int | bool | np.ndarray]:
    obj_pos, obj_quat = get_active_object_pose(data, config)
    place_body = data.body(config.place_body_name)
    place_body_pos = place_body.xpos.copy()
    place_body_quat = normalize_quat(place_body.xquat.copy())

    obj_local_pos, obj_local_quat = get_pose_in_body_frame(
        obj_pos,
        obj_quat,
        place_body_pos,
        place_body_quat,
    )

    place_site_id = mujoco.mj_name2id(
        MODEL, mujoco.mjtObj.mjOBJ_SITE, config.place_site_name
    )
    target_local_pos = MODEL.site_pos[place_site_id].copy()
    target_local_quat = normalize_quat(MODEL.site_quat[place_site_id].copy())
    target_local_pos[2] += TARGET_PLACE_REFERENCE_Z_OFFSET

    local_pos_error, local_rot_error = get_pose_error(
        obj_local_pos,
        obj_local_quat,
        target_local_pos,
        target_local_quat,
    )

    radial_error = float(np.linalg.norm(local_pos_error[:2]))
    height_error = float(local_pos_error[2])
    angle_error = float(np.linalg.norm(local_rot_error))
    contact_count, min_contact_dist = count_contacts_between_geoms(
        data,
        config.geom_name,
        config.place_geom_name,
    )
    pose_aligned = bool(
        radial_error < success_distance
        and abs(height_error) < success_distance
        and angle_error < success_angle_rad
    )

    return {
        "object_local_pos": obj_local_pos,
        "target_local_pos": target_local_pos,
        "radial_error": radial_error,
        "height_error": height_error,
        "angle_error_rad": angle_error,
        "angle_error_deg": float(np.rad2deg(angle_error)),
        "contact_count": int(contact_count),
        "min_contact_dist": min_contact_dist,
        "pose_aligned": pose_aligned,
        "inserted_contact_candidate": bool(pose_aligned and contact_count > 0),
    }


def format_vec3(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:+.4f}" for value in np.asarray(vec)) + "]"


def print_report(
    label: str,
    metrics: dict[str, float | int | bool | np.ndarray],
) -> None:
    status = "MASUK" if bool(metrics["inserted_contact_candidate"]) else "BELUM"
    min_contact_dist = metrics["min_contact_dist"]
    min_contact_text = (
        f"{float(min_contact_dist):+.6f}" if min_contact_dist is not None else "None"
    )

    print(f"[{label}] status={status}")
    print(
        "  radial_error="
        f"{1000.0 * float(metrics['radial_error']):.2f} mm"
        f", height_error={1000.0 * float(metrics['height_error']):+.2f} mm"
        f", angle_error={float(metrics['angle_error_deg']):.2f} deg"
    )
    print(
        "  contact_count="
        f"{int(metrics['contact_count'])}, min_contact_dist={min_contact_text}"
        f", pose_aligned={bool(metrics['pose_aligned'])}"
    )
    print(
        "  object_local_pos="
        f"{format_vec3(np.asarray(metrics['object_local_pos']))}"
    )
    print(
        "  target_local_pos="
        f"{format_vec3(np.asarray(metrics['target_local_pos']))}"
    )


def print_contact_debug(data: mujoco.MjData, config: ObjectCheckConfig) -> None:
    print("  contacts:")
    found = False
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1_name = mujoco.mj_id2name(
            MODEL, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
        )
        geom2_name = mujoco.mj_id2name(
            MODEL, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
        )
        geom_pair = {geom1_name, geom2_name}
        if config.geom_name not in geom_pair and config.place_geom_name not in geom_pair:
            continue

        found = True
        print(
            f"    {geom1_name} <-> {geom2_name}"
            f" | dist={float(contact.dist):+.6f}"
            f" | pos={format_vec3(np.asarray(contact.pos))}"
        )
    if not found:
        print("    tidak ada contact yang melibatkan objek/place aktif")


def run_direct_fit_check(
    data: mujoco.MjData,
    config: ObjectCheckConfig,
    success_distance: float,
    success_angle_rad: float,
) -> dict[str, float | int | bool | np.ndarray]:
    target_pos, target_quat = get_active_place_pose(data, config)
    set_object_pose_from_reference_pose(data, config, target_pos, target_quat)
    mujoco.mj_forward(MODEL, data)
    return collect_insertion_metrics(data, config, success_distance, success_angle_rad)


def run_drop_check(
    data: mujoco.MjData,
    config: ObjectCheckConfig,
    start_height: float,
    settle_seconds: float,
    success_distance: float,
    success_angle_rad: float,
    viewer=None,
    pause_seconds: float = 0.0,
    dump_contacts: bool = False,
) -> dict[str, float | int | bool | np.ndarray]:
    target_pos, target_quat = get_active_place_pose(data, config)
    start_pos, start_quat = offset_pose_along_local_z(
        target_pos, target_quat, start_height
    )
    set_object_pose_from_reference_pose(data, config, start_pos, start_quat)
    mujoco.mj_forward(MODEL, data)

    if viewer is not None:
        viewer.sync()
        if pause_seconds > 0.0:
            time.sleep(pause_seconds)

    steps = max(1, int(settle_seconds / MODEL.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(MODEL, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(MODEL.opt.timestep)

    if viewer is not None and pause_seconds > 0.0:
        time.sleep(pause_seconds)

    if dump_contacts:
        print_contact_debug(data, config)

    return collect_insertion_metrics(data, config, success_distance, success_angle_rad)


def reset_scene(data: mujoco.MjData, active_body_name: str) -> None:
    mujoco.mj_resetData(MODEL, data)
    arrange_scene_layout(data, active_body_name)


def run_single_object_check(
    object_body: str,
    start_height: float,
    settle_seconds: float,
    success_distance: float,
    success_angle_deg: float,
    data: mujoco.MjData | None = None,
    viewer=None,
    pause_seconds: float = 0.0,
    disable_place_collision: bool = False,
    dump_contacts: bool = False,
) -> tuple[dict[str, float | int | bool | np.ndarray], dict[str, float | int | bool | np.ndarray]]:
    config = OBJECT_CONFIGS[object_body]
    success_angle_rad = np.deg2rad(success_angle_deg)
    if data is None:
        data = mujoco.MjData(MODEL)

    for place_config in OBJECT_CONFIGS.values():
        set_geom_collision_enabled(place_config.place_geom_name, enabled=True)
    set_geom_collision_enabled(config.place_geom_name, enabled=not disable_place_collision)

    reset_scene(data, object_body)
    direct_fit_metrics = run_direct_fit_check(
        data,
        config,
        success_distance,
        success_angle_rad,
    )
    print_report(f"{object_body} | snap-to-target", direct_fit_metrics)

    reset_scene(data, object_body)
    drop_metrics = run_drop_check(
        data,
        config,
        start_height,
        settle_seconds,
        success_distance,
        success_angle_rad,
        viewer=viewer,
        pause_seconds=pause_seconds,
        dump_contacts=dump_contacts,
    )
    print_report(f"{object_body} | drop-test", drop_metrics)

    return direct_fit_metrics, drop_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cek apakah objek benar-benar bisa masuk ke target place yang sesuai "
            "di object_place.xml."
        )
    )
    parser.add_argument(
        "--object",
        choices=["all", *OBJECT_CONFIGS.keys()],
        default="all",
        help="Objek yang ingin dicek. Default: all.",
    )
    parser.add_argument(
        "--start-height",
        type=float,
        default=0.05,
        help="Jarak awal di atas target pose referensi, dalam meter. Default: 0.05.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=3.0,
        help="Durasi simulasi setelah objek dilepas, dalam detik. Default: 3.0.",
    )
    parser.add_argument(
        "--success-distance",
        type=float,
        default=DEFAULT_SUCCESS_DISTANCE,
        help="Ambang error posisi untuk menganggap objek aligned, dalam meter.",
    )
    parser.add_argument(
        "--success-angle-deg",
        type=float,
        default=DEFAULT_SUCCESS_ANGLE_DEG,
        help="Ambang error orientasi untuk menganggap objek aligned, dalam derajat.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Tampilkan MuJoCo viewer saat drop-test.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.5,
        help="Pause singkat sebelum dan sesudah drop saat viewer aktif. Default: 0.5.",
    )
    parser.add_argument(
        "--disable-place-collision",
        action="store_true",
        help="Debug: matikan collision place aktif. Jika objek lalu bisa masuk, masalahnya ada di collision mesh place.",
    )
    parser.add_argument(
        "--dump-contacts",
        action="store_true",
        help="Cetak contact yang melibatkan objek/place aktif setelah drop-test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_names = list(OBJECT_CONFIGS) if args.object == "all" else [args.object]

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer tidak tersedia di environment ini.")

        data = mujoco.MjData(MODEL)
        with mujoco.viewer.launch_passive(MODEL, data) as viewer:
            for object_body in object_names:
                run_single_object_check(
                    object_body=object_body,
                    start_height=args.start_height,
                    settle_seconds=args.settle_seconds,
                    success_distance=args.success_distance,
                    success_angle_deg=args.success_angle_deg,
                    data=data,
                    viewer=viewer,
                    pause_seconds=args.pause_seconds,
                    disable_place_collision=args.disable_place_collision,
                    dump_contacts=args.dump_contacts,
                )
    else:
        for object_body in object_names:
            run_single_object_check(
                object_body=object_body,
                start_height=args.start_height,
                settle_seconds=args.settle_seconds,
                success_distance=args.success_distance,
                success_angle_deg=args.success_angle_deg,
                disable_place_collision=args.disable_place_collision,
                dump_contacts=args.dump_contacts,
            )


if __name__ == "__main__":
    main()
