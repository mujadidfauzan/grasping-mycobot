import json
import socket
import time

import numpy as np

RASPI_IP = "10.16.120.250"
PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.5)


def talk_to_robot(command, data=None):
    message = {"command": command, "data": data}
    sock.sendto(json.dumps(message).encode(), (RASPI_IP, PORT))

    if command == "GET_STATE":
        try:
            raw_data, _ = sock.recvfrom(1024)
            return json.loads(raw_data.decode())
        except:
            return None
    return None


last_valid_state = None
try:
    print("Mulai Loop Kontrol...")
    while True:
        state = talk_to_robot("GET_STATE")
        print(f"Robot state: {state}")
        if state:
            last_valid_state = state
            current_angles = state["angles"]
            current_coords = state["coords"]
            print(f"Robot di: {current_coords}")
        else:
            # print("Gagal ambil data, gunakan data terakhir.")
            if last_valid_state:
                current_angles = last_valid_state["angles"]
                current_coords = last_valid_state["coords"]
                print(f"Robot di (last valid): {current_coords}")

        target_angles = [
            0.08137732920459544,
            -0.9487635330901536,
            -0.7979580716456075,
            -0.24846189464804908,
            0.38522823011051255,
            0.46248111674340486,
        ]

        target_angles = [np.rad2deg(a) for a in target_angles]
        print(f"Set target angles: {target_angles}")
        talk_to_robot("SET_ANGLES", target_angles)

        # talk_to_robot("SET_GRIPPER", 1)  # Tutup

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Berhenti.")
    print("Berhenti.")
