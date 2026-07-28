from pathlib import Path

import mujoco
import numpy as np
import pytest

from general_motion_retargeting.quadruped.joint_mapping import (
    map_initial_configuration,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "general_motion_retargeting/quadruped/configs"


def test_initial_map_operates_around_reference_pose():
    source_spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)
    target_spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)
    source = np.array(
        [
            source_spec.reference_pose[name]
            for name in source_spec.motion_joint_order
        ]
    )

    qpos = map_initial_configuration(source, source_spec, target_spec)

    for leg in target_spec.leg_order:
        for name in target_spec.legs[leg].joints:
            joint_id = mujoco.mj_name2id(
                target_spec.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            actual = qpos[target_spec.model.jnt_qposadr[joint_id]]
            assert actual == pytest.approx(target_spec.reference_pose[name])
