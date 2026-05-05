from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "source" / "robot" / "reaching.xml"
EE_SITE = "attachment_site"
TARGET_SITE = "target"
STEP_DEG = 2.0

selected_joint = 0


def main():
    global selected_joint

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = data.ctrl.copy()
    ctrl[:6] = data.qpos[:6]
    data.ctrl[:] = ctrl

    def print_pos():
        ee_pos = data.site(EE_SITE).xpos.copy()
        target_pos = data.site(TARGET_SITE).xpos.copy()
        print(
            f"\rJoint {selected_joint + 1} | EE {ee_pos} | Target {target_pos}",
            end="",
            flush=True,
        )

    def key_callback(keycode):
        global selected_joint

        if ord("1") <= keycode <= ord("6"):
            selected_joint = keycode - ord("1")
        elif keycode == 262:  # right arrow
            data.ctrl[selected_joint] += np.deg2rad(STEP_DEG)
        elif keycode == 263:  # left arrow
            data.ctrl[selected_joint] -= np.deg2rad(STEP_DEG)
        elif keycode in (ord("R"), ord("r")):
            mujoco.mj_resetData(model, data)
            data.ctrl[:6] = data.qpos[:6]

        data.ctrl[:] = np.clip(
            data.ctrl,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )

    print("Tekan 1-6 untuk pilih joint, panah kiri/kanan untuk gerak, R untuk reset.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            print_pos()
            viewer.sync()


if __name__ == "__main__":
    main()
