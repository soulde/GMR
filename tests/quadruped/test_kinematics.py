from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from general_motion_retargeting.quadruped.kinematics import (
    source_forward_kinematics,
)
from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "general_motion_retargeting/quadruped/configs/laikago.yaml"
)
FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"
DOG_PACE = ROOT / "assets/quadrupeds/motions/dog_pace.txt"


def test_source_fk_returns_root_relative_feet():
    spec = load_robot_spec(CONFIG, ROOT)
    motion = load_motion_imitation(
        FIXTURE, spec.motion_joint_order, spec.quaternion_order
    )

    canonical = source_forward_kinematics(spec, motion)

    assert canonical.foot_pos_root.shape == (2, 4, 3)
    assert canonical.leg_order == ("FL", "FR", "RL", "RR")
    assert np.isfinite(canonical.foot_pos_root).all()


def test_source_fk_is_invariant_to_world_root_translation():
    spec = load_robot_spec(CONFIG, ROOT)
    motion = load_motion_imitation(
        FIXTURE, spec.motion_joint_order, spec.quaternion_order
    )
    shifted = type(motion)(
        fps=motion.fps,
        root_pos=motion.root_pos + np.array([4.0, -2.0, 1.0]),
        root_rot=motion.root_rot,
        joint_pos=motion.joint_pos,
        joint_names=motion.joint_names,
        loop_mode=motion.loop_mode,
    )

    original_feet = source_forward_kinematics(spec, motion).foot_pos_root
    shifted_feet = source_forward_kinematics(spec, shifted).foot_pos_root

    np.testing.assert_allclose(shifted_feet, original_feet, atol=1e-8)


def test_dog_pace_source_fk_has_consistent_ground_clearance():
    spec = load_robot_spec(CONFIG, ROOT)
    motion = load_motion_imitation(
        DOG_PACE,
        spec.motion_joint_order,
        spec.quaternion_order,
        spec.root_frame_rotation_wxyz,
    )

    canonical = source_forward_kinematics(spec, motion)
    root_rotation = Rotation.from_quat(
        motion.root_rot, scalar_first=True
    )
    foot_world = motion.root_pos[:, None, :] + np.stack(
        [
            root_rotation.apply(canonical.foot_pos_root[:, leg_index])
            for leg_index in range(4)
        ],
        axis=1,
    )
    minimum_height = foot_world[:, :, 2].min(axis=0)

    assert np.all((minimum_height > 0.03) & (minimum_height < 0.07))
    assert np.ptp(minimum_height) < 0.01
