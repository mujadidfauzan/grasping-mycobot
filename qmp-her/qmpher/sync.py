from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .utils import unwrap_env


def _name_map(model: mujoco.MjModel, obj_type: mujoco.mjtObj) -> dict[str, int]:
    names: dict[str, int] = {}
    count = {
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    }[obj_type]
    for obj_id in range(int(count)):
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        if name:
            names[str(name)] = int(obj_id)
    return names


def _joint_qpos_size(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4
    return 1


def _joint_dof_size(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        return 6
    if joint_type == mujoco.mjtJoint.mjJNT_BALL:
        return 3
    return 1


def _active_object_info(env: Any) -> dict[str, Any]:
    env = unwrap_env(env)
    getter = getattr(env, "_get_active_obj_info", None)
    if callable(getter):
        return dict(getter())
    active_obj_name = str(getattr(env, "active_obj_name", "box"))
    object_info = getattr(env, "object_info")
    return dict(object_info[active_obj_name])


def _copy_common_joint_state(
    *,
    source_env: Any,
    target_env: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    skip_joint_names: set[str],
) -> None:
    source_joint_map = _name_map(source_env.model, mujoco.mjtObj.mjOBJ_JOINT)
    target_joint_map = _name_map(target_env.model, mujoco.mjtObj.mjOBJ_JOINT)
    for joint_name in sorted(set(source_joint_map).intersection(target_joint_map)):
        if joint_name in skip_joint_names:
            continue

        source_joint_id = source_joint_map[joint_name]
        target_joint_id = target_joint_map[joint_name]
        source_qposadr = int(source_env.model.jnt_qposadr[source_joint_id])
        source_dofadr = int(source_env.model.jnt_dofadr[source_joint_id])
        target_qposadr = int(target_env.model.jnt_qposadr[target_joint_id])
        target_dofadr = int(target_env.model.jnt_dofadr[target_joint_id])

        qpos_size = _joint_qpos_size(source_env.model, source_joint_id)
        dof_size = _joint_dof_size(source_env.model, source_joint_id)
        if qpos_size != _joint_qpos_size(target_env.model, target_joint_id):
            continue
        if dof_size != _joint_dof_size(target_env.model, target_joint_id):
            continue

        qpos[target_qposadr : target_qposadr + qpos_size] = source_env.data.qpos[
            source_qposadr : source_qposadr + qpos_size
        ]
        qvel[target_dofadr : target_dofadr + dof_size] = source_env.data.qvel[
            source_dofadr : source_dofadr + dof_size
        ]


def _copy_common_ctrl(source_env: Any, target_env: Any) -> np.ndarray:
    ctrl = target_env.data.ctrl.copy()
    source_act_map = _name_map(source_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR)
    target_act_map = _name_map(target_env.model, mujoco.mjtObj.mjOBJ_ACTUATOR)
    for act_name in sorted(set(source_act_map).intersection(target_act_map)):
        ctrl[target_act_map[act_name]] = source_env.data.ctrl[source_act_map[act_name]]
    return np.clip(ctrl, target_env.model.actuator_ctrlrange[:, 0], target_env.model.actuator_ctrlrange[:, 1])


def sync_grasp_env_from_target(
    *,
    target_env: Any,
    grasp_env: Any,
    target_object_name: str = "box",
) -> None:
    """Synchronize a hidden GraspingEnvIK with the live target env state."""
    source = unwrap_env(target_env)
    target = unwrap_env(grasp_env)

    qpos = target.init_qpos.copy()
    qvel = target.init_qvel.copy()

    source_object_info = _active_object_info(source)
    target.active_obj_name = target_object_name
    target_object_info = target.object_info[target_object_name]
    skip_joint_names = {
        str(source_object_info["joint_name"]),
        str(target_object_info["joint_name"]),
    }
    _copy_common_joint_state(
        source_env=source,
        target_env=target,
        qpos=qpos,
        qvel=qvel,
        skip_joint_names=skip_joint_names,
    )

    source_body = source.data.body(str(source_object_info["body_name"]))
    object_qposadr = int(target_object_info["qposadr"])
    object_dofadr = int(target_object_info["dofadr"])
    qpos[object_qposadr : object_qposadr + 3] = source_body.xpos.copy()
    qpos[object_qposadr + 3 : object_qposadr + 7] = source_body.xquat.copy()
    qvel[object_dofadr : object_dofadr + 6] = 0.0

    target.set_state(qpos, qvel)
    target.data.ctrl[:] = _copy_common_ctrl(source, target)

    reset_ik = getattr(target, "_reset_ik_state", None)
    if callable(reset_ik):
        reset_ik()
    sync_visual_frames = getattr(target, "sync_visual_frames", None)
    if callable(sync_visual_frames):
        sync_visual_frames()
    mujoco.mj_forward(target.model, target.data)
