import mujoco
import numpy as np

from .robot_spec import QuadrupedRobotSpec
from .types import CanonicalQuadrupedMotion, JointSpaceMotion


def _free_joint_id(model: mujoco.MjModel) -> int:
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError(
            f"quadruped MJCF must have one free joint, found {len(free_joints)}"
        )
    return int(free_joints[0])


def set_named_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...],
    values: np.ndarray,
) -> None:
    if len(names) != len(values):
        raise ValueError(
            f"joint names and values differ: {len(names)} != {len(values)}"
        )
    for name, value in zip(names, values, strict=True):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise ValueError(f"joint {name!r} does not exist in MJCF")
        data.qpos[model.jnt_qposadr[joint_id]] = value


def set_free_root(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
) -> None:
    qpos_address = model.jnt_qposadr[_free_joint_id(model)]
    data.qpos[qpos_address : qpos_address + 3] = root_pos
    data.qpos[qpos_address + 3 : qpos_address + 7] = root_rot_wxyz


def source_forward_kinematics(
    spec: QuadrupedRobotSpec,
    motion: JointSpaceMotion,
) -> CanonicalQuadrupedMotion:
    if motion.joint_names != spec.motion_joint_order:
        raise ValueError("motion joint names do not match source specification")

    model = spec.model
    data = mujoco.MjData(model)
    root_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.root_body
    )
    site_ids = [
        mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            spec.legs[leg].foot_site,
        )
        for leg in spec.leg_order
    ]
    feet = np.empty((len(motion.root_pos), 4, 3), dtype=float)

    for frame_index in range(len(motion.root_pos)):
        set_free_root(
            model,
            data,
            motion.root_pos[frame_index],
            motion.root_rot[frame_index],
        )
        set_named_joint_positions(
            model,
            data,
            motion.joint_names,
            motion.joint_pos[frame_index],
        )
        mujoco.mj_forward(model, data)

        root_position = data.xpos[root_body_id]
        root_rotation = data.xmat[root_body_id].reshape(3, 3)
        for leg_index, site_id in enumerate(site_ids):
            feet[frame_index, leg_index] = root_rotation.T @ (
                data.site_xpos[site_id] - root_position
            )

    return CanonicalQuadrupedMotion(
        fps=motion.fps,
        root_pos=motion.root_pos.copy(),
        root_rot=motion.root_rot.copy(),
        foot_pos_root=feet,
        leg_order=spec.leg_order,
        loop_mode=motion.loop_mode,
    )
