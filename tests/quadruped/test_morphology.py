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


def test_morphology_maps_neutral_stance_and_scales_dynamic_delta():
    source_spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)
    target_spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)
    motion = load_motion_imitation(
        FIXTURE,
        source_spec.motion_joint_order,
        source_spec.quaternion_order,
    )
    canonical = source_forward_kinematics(source_spec, motion)
    source_shape = describe_morphology(source_spec)
    target_shape = describe_morphology(target_spec)

    scaled = scale_foot_trajectories(
        canonical, source_spec, target_spec
    )

    source_delta = canonical.foot_pos_root[1] - source_shape.neutral_feet
    target_delta = scaled.foot_pos_root[1] - target_shape.neutral_feet
    expected_scale = np.array(
        [
            target_shape.hip_length / source_shape.hip_length,
            target_shape.hip_width / source_shape.hip_width,
            target_shape.leg_reach / source_shape.leg_reach,
        ]
    )
    np.testing.assert_allclose(target_delta, source_delta * expected_scale)
