from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.quadruped.retarget import (
    QuadrupedRobotRetargeter,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec
from general_motion_retargeting.quadruped.types import JointSpaceMotion


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "general_motion_retargeting/quadruped/configs"


def test_go2_retargeter_solves_reference_stance():
    source_spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)
    target_spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)
    source_joints = np.array(
        [
            source_spec.reference_pose[name]
            for name in source_spec.motion_joint_order
        ]
    )
    motion = JointSpaceMotion(
        fps=50.0,
        root_pos=np.array([[0.0, 0.0, 0.45]]),
        root_rot=np.array([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=source_joints[None],
        joint_names=source_spec.motion_joint_order,
        loop_mode="Clamp",
    )
    retargeter = QuadrupedRobotRetargeter(
        source_spec=source_spec,
        target_spec=target_spec,
        solver="proxqp",
        damping=0.5,
        max_iterations=10,
        use_velocity_limit=False,
    )

    result = retargeter.retarget_motion(motion)

    assert result.qpos.shape == (1, target_spec.model.nq)
    assert np.isfinite(result.qpos).all()
    assert result.diagnostics[0].final_error < 0.05
    for joint_name in target_spec.motion_joint_order:
        joint_id = mujoco.mj_name2id(
            target_spec.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        value = result.qpos[0, target_spec.model.jnt_qposadr[joint_id]]
        lower, upper = target_spec.model.jnt_range[joint_id]
        assert lower <= value <= upper
