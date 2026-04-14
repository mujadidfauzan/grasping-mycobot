import json
import socket
import time


class MyCobotRemote:
    """UDP client for communicating with a remote MyCobot robot arm."""

    DEFAULT_PORT = 5005
    DEFAULT_TIMEOUT = 0.1
    DEFAULT_ACK_TIMEOUT = 0.5
    DEFAULT_SETTLE_TIMEOUT = 4.0
    DEFAULT_POLL_INTERVAL = 0.05
    NUM_JOINTS = 6
    RESPONSE_TYPES = {
        "GET_STATE": "STATE",
        "SET_ANGLES": "SET_ANGLES_ACK",
        "SET_GRIPPER": "SET_GRIPPER_ACK",
    }

    def __init__(
        self, ip: str, port: int = DEFAULT_PORT, timeout: float = DEFAULT_TIMEOUT
    ):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)

        # Last known good state (fallback when communication fails)
        self._last_angles: list[float] = [0.0] * self.NUM_JOINTS
        self._last_coords: list[float] = [0.0] * self.NUM_JOINTS
        self._next_seq = 0

    def _send(
        self,
        command: str,
        data=None,
        *,
        expect_response: bool = False,
        timeout: float | None = None,
    ) -> dict | None:
        """Send a UDP command and return the parsed response (or None on failure)."""
        message = json.dumps({"command": command, "data": data}).encode()
        original_timeout = self.sock.gettimeout()
        try:
            self.sock.sendto(message, self.addr)
            if not expect_response and command != "GET_STATE":
                return {}

            deadline = None if timeout is None else time.monotonic() + float(timeout)
            expected_response = self.RESPONSE_TYPES.get(command)

            while True:
                if deadline is None:
                    self.sock.settimeout(original_timeout)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return None
                    self.sock.settimeout(remaining)

                raw, _ = self.sock.recvfrom(1024)
                response = json.loads(raw.decode())
                if (
                    expected_response is None
                    or response.get("response") == expected_response
                ):
                    return response
        except (OSError, socket.timeout, json.JSONDecodeError):
            return None
        finally:
            self.sock.settimeout(original_timeout)

    def update_state(self) -> bool:
        """
        Fetch angles + coords in ONE UDP round-trip and cache them.
        Call this ONCE per control loop, then read .angles and .coords directly
        to avoid multiple blocking network calls per iteration.
        """
        state = self._send("GET_STATE", expect_response=True) or {}
        updated = False

        if isinstance(state.get("angles"), list):
            self._last_angles = state["angles"]
            updated = True
        if isinstance(state.get("coords"), list):
            self._last_coords = state["coords"]
            updated = True

        return updated

    @property
    def angles(self) -> list[float]:
        """Last cached joint angles in degrees."""
        print(f"Current angles: {self._last_angles}")
        return self._last_angles

    @property
    def coords(self) -> list[float]:
        """Last cached end-effector coords [x, y, z, rx, ry, rz]."""
        return self._last_coords

    # Keep these for backward compatibility — but note each fires a UDP call
    def get_angles(self) -> list[float]:
        self.update_state()
        return self._last_angles

    def get_coords(self) -> list[float]:
        return self._last_coords  # reuse state already fetched by update_state()

    @staticmethod
    def _max_joint_error_deg(current: list[float], target: list[float]) -> float:
        errors = []
        for current_angle, target_angle in zip(current, target):
            delta = (float(current_angle) - float(target_angle) + 180.0) % 360.0 - 180.0
            errors.append(abs(delta))
        return max(errors, default=float("inf"))

    def wait_until_angles_reached(
        self,
        angles: list[float],
        *,
        tolerance_deg: float = 2.0,
        timeout: float = DEFAULT_SETTLE_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        """Block until the cached joint state is close to the requested target."""
        target = [float(angle) for angle in angles]
        deadline = time.monotonic() + float(timeout)

        while time.monotonic() <= deadline:
            self.update_state()
            if self._max_joint_error_deg(self._last_angles, target) <= tolerance_deg:
                return True
            time.sleep(poll_interval)

        return False

    def send_angles(
        self,
        angles: list[float],
        speed: int = 20,
        *,
        wait: bool = False,
        tolerance_deg: float = 2.0,
        ack_timeout: float = DEFAULT_ACK_TIMEOUT,
        settle_timeout: float = DEFAULT_SETTLE_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ) -> bool:
        """Send target joint angles and optionally block until the target is reached."""
        target = [float(angle) for angle in angles]
        seq = self._next_seq
        self._next_seq += 1

        response = self._send(
            "SET_ANGLES",
            {"angles": target, "speed": int(speed), "seq": seq},
            expect_response=True,
            timeout=ack_timeout,
        )
        if not response or not response.get("ok") or response.get("seq") != seq:
            return False

        if not wait:
            return True

        return self.wait_until_angles_reached(
            target,
            tolerance_deg=tolerance_deg,
            timeout=settle_timeout,
            poll_interval=poll_interval,
        )

    def set_gripper_state(self, state: int, speed: int = 50) -> bool:
        """Set gripper state: 0 = open, 1 = closed."""
        response = self._send(
            "SET_GRIPPER",
            {"state": int(state), "speed": int(speed)},
            expect_response=True,
            timeout=self.DEFAULT_ACK_TIMEOUT,
        )
        return bool(response and response.get("ok"))

    def power_on(self) -> None:
        print(f"Connected to robot at {self.addr}")

    def stop(self) -> None:
        print("Stopping remote connection...")
        self.sock.close()
