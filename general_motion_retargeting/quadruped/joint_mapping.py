import mujoco
import numpy as np

from .robot_spec import QuadrupedRobotSpec, SEGMENTS


def _joint_qpos_address(
    model: mujoco.MjModel,
    joint_name: str,
) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise ValueError(f"joint {joint_name!r} does not exist in MJCF")
    return joint_id, int(model.jnt_qposadr[joint_id])


def map_initial_configuration(
    source_joint_pos: np.ndarray,
    source_spec: QuadrupedRobotSpec,
    target_spec: QuadrupedRobotSpec,
) -> np.ndarray:
    if source_joint_pos.shape != (len(source_spec.motion_joint_order),):
        raise ValueError(
            "source joint position shape does not match source joint order"
        )
    if set(target_spec.joint_mapping) != set(SEGMENTS):
        raise ValueError("target joint_mapping must define hip, thigh, and calf")

    source_by_name = dict(
        zip(source_spec.motion_joint_order, source_joint_pos, strict=True)
    )
    qpos = target_spec.model.qpos0.copy()

    for leg in target_spec.leg_order:
        source_joints = source_spec.legs[leg].joints
        target_joints = target_spec.legs[leg].joints
        for segment, source_name, target_name in zip(
            SEGMENTS, source_joints, target_joints, strict=True
        ):
            mapping = target_spec.joint_mapping[segment]
            source_delta = (
                source_by_name[source_name]
                - source_spec.reference_pose[source_name]
            )
            value = (
                target_spec.reference_pose[target_name]
                + mapping.sign * mapping.scale * source_delta
            )
            joint_id, qpos_address = _joint_qpos_address(
                target_spec.model, target_name
            )
            if target_spec.model.jnt_limited[joint_id]:
                lower, upper = target_spec.model.jnt_range[joint_id]
                value = float(np.clip(value, lower, upper))
            qpos[qpos_address] = value

    return qpos
