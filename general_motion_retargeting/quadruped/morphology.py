from dataclasses import dataclass

import mujoco
import numpy as np

from .kinematics import source_forward_kinematics
from .robot_spec import QuadrupedRobotSpec
from .types import CanonicalQuadrupedMotion, JointSpaceMotion


@dataclass(frozen=True)
class Morphology:
    neutral_feet: np.ndarray
    hip_length: float
    hip_width: float
    leg_reach: float


def _default_joint_positions(spec: QuadrupedRobotSpec) -> np.ndarray:
    return np.asarray(
        [
            spec.model.qpos0[
                spec.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        spec.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                ]
            ]
            for name in spec.motion_joint_order
        ],
        dtype=float,
    )


def _reference_motion(spec: QuadrupedRobotSpec) -> JointSpaceMotion:
    joint_pos = _default_joint_positions(spec)[None]
    return JointSpaceMotion(
        fps=1.0,
        root_pos=np.zeros((1, 3), dtype=float),
        root_rot=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float),
        joint_pos=joint_pos,
        joint_names=spec.motion_joint_order,
        loop_mode="Clamp",
    )


def _reference_hips_root(spec: QuadrupedRobotSpec) -> np.ndarray:
    model = spec.model
    data = mujoco.MjData(model)
    from .kinematics import set_free_root, set_named_joint_positions

    set_free_root(
        model,
        data,
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    set_named_joint_positions(
        model,
        data,
        spec.motion_joint_order,
        _default_joint_positions(spec),
    )
    mujoco.mj_forward(model, data)
    root_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, spec.root_body
    )
    root_pos = data.xpos[root_id]
    root_rot = data.xmat[root_id].reshape(3, 3)
    hips = []
    for leg in spec.leg_order:
        first_joint = spec.legs[leg].joints[0]
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, first_joint
        )
        body_id = model.jnt_bodyid[joint_id]
        hips.append(root_rot.T @ (data.xpos[body_id] - root_pos))
    return np.asarray(hips)


def describe_morphology(spec: QuadrupedRobotSpec) -> Morphology:
    neutral_feet = source_forward_kinematics(
        spec, _reference_motion(spec)
    ).foot_pos_root[0]
    hips = _reference_hips_root(spec)

    front_x = np.mean(hips[:2, 0])
    rear_x = np.mean(hips[2:, 0])
    left_y = np.mean(hips[[0, 2], 1])
    right_y = np.mean(hips[[1, 3], 1])
    leg_reach = float(np.mean(np.linalg.norm(neutral_feet - hips, axis=1)))
    values = np.array([front_x - rear_x, left_y - right_y, leg_reach])
    if np.any(np.abs(values) < 1e-6):
        raise ValueError(f"degenerate quadruped morphology: {values}")

    return Morphology(
        neutral_feet=neutral_feet,
        hip_length=float(abs(values[0])),
        hip_width=float(abs(values[1])),
        leg_reach=leg_reach,
    )


def scale_foot_trajectories(
    motion: CanonicalQuadrupedMotion,
    source_spec: QuadrupedRobotSpec,
    target_spec: QuadrupedRobotSpec,
    trajectory_scale: np.ndarray | None = None,
    motion_center: str = "temporal_median",
    root_translation_scale: np.ndarray | None = None,
) -> CanonicalQuadrupedMotion:
    source = describe_morphology(source_spec)
    target = describe_morphology(target_spec)
    if motion_center == "temporal_median":
        source_motion_center = np.median(motion.foot_pos_root, axis=0)
    elif motion_center == "mjcf_default":
        source_motion_center = source.neutral_feet
    else:
        raise ValueError(f"unsupported motion center: {motion_center!r}")
    if trajectory_scale is None:
        scale = np.broadcast_to(
            (1.0, 1.0, target.leg_reach / source.leg_reach),
            (4, 3),
        )
    else:
        scale = np.asarray(trajectory_scale, dtype=float)
        if scale.shape != (4, 3):
            raise ValueError("trajectory_scale must have shape [4, 3]")
    feet = target.neutral_feet + (
        motion.foot_pos_root - source_motion_center
    ) * scale
    root_scale = np.asarray(
        root_translation_scale
        if root_translation_scale is not None
        else (1.0, 1.0, 1.0),
        dtype=float,
    )
    if root_scale.shape != (3,):
        raise ValueError("root_translation_scale must have shape [3]")
    root_pos = motion.root_pos[0] + (
        motion.root_pos - motion.root_pos[0]
    ) * root_scale
    return CanonicalQuadrupedMotion(
        fps=motion.fps,
        root_pos=root_pos,
        root_rot=motion.root_rot.copy(),
        foot_pos_root=feet,
        leg_order=motion.leg_order,
        loop_mode=motion.loop_mode,
    )
