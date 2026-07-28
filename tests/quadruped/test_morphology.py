from pathlib import Path

import numpy as np

from general_motion_retargeting.quadruped.kinematics import (
    source_forward_kinematics,
)
from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)
from general_motion_retargeting.quadruped.morphology import (
    describe_morphology,
    scale_foot_trajectories,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "general_motion_retargeting/quadruped/configs"
FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"


def test_morphology_centers_motion_on_target_stance_and_preserves_horizontal_delta():
    source_spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)
    target_spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)
    motion = load_motion_imitation(
        FIXTURE,
        source_spec.motion_joint_order,
        source_spec.quaternion_order,
    )
    canonical = source_forward_kinematics(source_spec, motion)
    target_shape = describe_morphology(target_spec)

    scaled = scale_foot_trajectories(
        canonical, source_spec, target_spec
    )

    source_shape = describe_morphology(source_spec)
    source_center = np.median(canonical.foot_pos_root, axis=0)
    expected_scale = np.array(
        [1.0, 1.0, target_shape.leg_reach / source_shape.leg_reach]
    )
    expected = target_shape.neutral_feet + (
        canonical.foot_pos_root - source_center
    ) * expected_scale

    np.testing.assert_allclose(scaled.foot_pos_root, expected)
    np.testing.assert_allclose(
        np.diff(scaled.foot_pos_root[:, :, :2], axis=0),
        np.diff(canonical.foot_pos_root[:, :, :2], axis=0),
    )


def test_morphology_scales_root_displacement_about_first_frame():
    source_spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)
    target_spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)
    motion = load_motion_imitation(
        FIXTURE,
        source_spec.motion_joint_order,
        source_spec.quaternion_order,
    )
    canonical = source_forward_kinematics(source_spec, motion)

    scaled = scale_foot_trajectories(
        canonical,
        source_spec,
        target_spec,
        root_translation_scale=np.array([2.0, 0.5, 1.0]),
    )

    np.testing.assert_allclose(scaled.root_pos[0], canonical.root_pos[0])
    np.testing.assert_allclose(
        scaled.root_pos[1] - scaled.root_pos[0],
        (canonical.root_pos[1] - canonical.root_pos[0])
        * np.array([2.0, 0.5, 1.0]),
    )
