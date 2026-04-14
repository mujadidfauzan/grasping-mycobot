from __future__ import annotations

import argparse
import json
import socket
import time
from dataclasses import dataclass

import numpy as np
from pymycobot import MyCobot280

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5005
DEFAULT_SERIAL_PORT = "/dev/ttyAMA0"
DEFAULT_BAUD = 115200

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
class BridgeConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    serial_port: str = DEFAULT_SERIAL_PORT
    baud: int = DEFAULT_BAUD
    max_speed: int = 100
    max_delta_deg: float = 12.0
    recv_buffer_size: int = 4096


@dataclass
class BridgeState:
    last_seq: int = -1
    last_applied_angles: list[float] | None = None
    last_command_time: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safer UDP bridge for receiving validated commands on MyCobot."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--serial-port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--max-speed", type=int, default=100)
    parser.add_argument("--max-delta-deg", type=float, default=12.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BridgeConfig:
    return BridgeConfig(
        host=str(args.host),
        port=int(args.port),
        serial_port=str(args.serial_port),
        baud=int(args.baud),
        max_speed=int(args.max_speed),
        max_delta_deg=float(args.max_delta_deg),
    )


def send_json(sock: socket.socket, addr, payload: dict) -> None:
    sock.sendto(json.dumps(payload).encode(), addr)


def sanitize_speed(speed: int | float, cfg: BridgeConfig) -> int:
    return int(np.clip(int(speed), 1, cfg.max_speed))


def get_robot_state(mc: MyCobot280) -> tuple[list[float], list[float]]:
    angles = mc.get_angles()
    coords = mc.get_coords()
    if not isinstance(angles, list) or len(angles) != 6:
        raise ValueError(f"Invalid robot joint state: {angles}")
    if not isinstance(coords, list) or len(coords) != 6:
        raise ValueError(f"Invalid robot Cartesian state: {coords}")
    return [float(v) for v in angles], [float(v) for v in coords]


def clip_angles_to_limits(angles_deg: np.ndarray) -> np.ndarray:
    return np.clip(angles_deg, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1])


def limit_step_from_current(
    current_deg: np.ndarray, target_deg: np.ndarray, max_delta_deg: float
) -> np.ndarray:
    delta = np.clip(target_deg - current_deg, -max_delta_deg, max_delta_deg)
    return current_deg + delta


def parse_angles_payload(payload: dict) -> tuple[list[float], int, int]:
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("SET_ANGLES data harus berupa object.")

    angles = data.get("angles")
    speed = data.get("speed", 20)
    seq = data.get("seq")

    if not isinstance(angles, list) or len(angles) != 6:
        raise ValueError("SET_ANGLES membutuhkan 6 joint angles.")
    if seq is None:
        raise ValueError("SET_ANGLES membutuhkan field seq.")

    numeric_angles = []
    for angle in angles:
        numeric_angles.append(float(angle))

    return numeric_angles, int(speed), int(seq)


def parse_gripper_payload(payload: dict) -> tuple[int, int]:
    data = payload.get("data")
    if isinstance(data, dict):
        state = int(data.get("state", 0))
        speed = int(data.get("speed", 50))
    else:
        state = int(data)
        speed = 50

    if state not in (0, 1):
        raise ValueError("SET_GRIPPER state harus 0 atau 1.")
    return state, speed


def handle_get_state(
    mc: MyCobot280, bridge_state: BridgeState, sock: socket.socket, addr
) -> None:
    angles, coords = get_robot_state(mc)
    send_json(
        sock,
        addr,
        {
            "response": "STATE",
            "ok": True,
            "angles": angles,
            "coords": coords,
            "last_seq": bridge_state.last_seq,
            "timestamp": time.time(),
        },
    )


def handle_set_angles(
    mc: MyCobot280,
    bridge_state: BridgeState,
    sock: socket.socket,
    addr,
    payload: dict,
    cfg: BridgeConfig,
) -> None:
    requested_angles, speed, seq = parse_angles_payload(payload)

    if seq <= bridge_state.last_seq:
        send_json(
            sock,
            addr,
            {
                "response": "SET_ANGLES_ACK",
                "ok": False,
                "seq": seq,
                "error": "stale sequence",
                "last_seq": bridge_state.last_seq,
            },
        )
        return

    current_angles, _ = get_robot_state(mc)
    current_angles_arr = np.asarray(current_angles, dtype=np.float64)
    requested_arr = np.asarray(requested_angles, dtype=np.float64)

    clipped_arr = clip_angles_to_limits(requested_arr)
    applied_arr = limit_step_from_current(
        current_angles_arr, clipped_arr, cfg.max_delta_deg
    )
    applied_arr = clip_angles_to_limits(applied_arr)

    speed = sanitize_speed(speed, cfg)
    applied_angles = applied_arr.astype(np.float64).tolist()

    mc.send_angles(applied_angles, speed)
    bridge_state.last_seq = seq
    bridge_state.last_applied_angles = applied_angles
    bridge_state.last_command_time = time.time()

    send_json(
        sock,
        addr,
        {
            "response": "SET_ANGLES_ACK",
            "ok": True,
            "seq": seq,
            "speed": speed,
            "requested_angles": requested_angles,
            "applied_angles": applied_angles,
            "clipped_to_limits": bool(not np.allclose(clipped_arr, requested_arr)),
            "step_limited": bool(not np.allclose(applied_arr, clipped_arr)),
        },
    )


def handle_set_gripper(
    mc: MyCobot280, sock: socket.socket, addr, payload: dict, cfg: BridgeConfig
) -> None:
    state, speed = parse_gripper_payload(payload)
    speed = sanitize_speed(speed, cfg)

    mc.set_gripper_state(state, speed)
    mc.set_gripper_state(state, speed)

    send_json(
        sock,
        addr,
        {
            "response": "SET_GRIPPER_ACK",
            "ok": True,
            "state": state,
            "speed": speed,
        },
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    bridge_state = BridgeState()

    mc = MyCobot280(cfg.serial_port, cfg.baud)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((cfg.host, cfg.port))

    print(
        "Safe bridge active "
        f"[udp={cfg.host}:{cfg.port} serial={cfg.serial_port} baud={cfg.baud}]"
    )

    try:
        while True:
            data, addr = sock.recvfrom(cfg.recv_buffer_size)
            cmd = None

            try:
                payload = json.loads(data.decode())
                cmd = payload.get("command")

                if cmd == "GET_STATE":
                    handle_get_state(mc, bridge_state, sock, addr)
                elif cmd == "SET_ANGLES":
                    handle_set_angles(mc, bridge_state, sock, addr, payload, cfg)
                elif cmd == "SET_GRIPPER":
                    handle_set_gripper(mc, sock, addr, payload, cfg)
                else:
                    raise ValueError(f"Unknown command: {cmd}")

            except Exception as exc:
                print(f"Safe bridge error ({cmd}): {exc}")
                response_type = {
                    "GET_STATE": "STATE",
                    "SET_ANGLES": "SET_ANGLES_ACK",
                    "SET_GRIPPER": "SET_GRIPPER_ACK",
                }.get(cmd, "ERROR")
                send_json(
                    sock,
                    addr,
                    {"response": response_type, "ok": False, "error": str(exc)},
                )
    finally:
        sock.close()


if __name__ == "__main__":
    main()
