from pathlib import Path

import pytest

from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "general_motion_retargeting/quadruped/configs"


def test_load_laikago_spec_resolves_model_and_leg_order():
    spec = load_robot_spec(CONFIGS / "laikago.yaml", ROOT)

    assert spec.robot == "laikago"
    assert spec.leg_order == ("FL", "FR", "RL", "RR")
    assert len(spec.motion_joint_order) == 12
    assert spec.model.nq == 19
    assert spec.quaternion_order == "xyzw"
    assert spec.foot_contact_offset == pytest.approx(0.03)
    assert spec.root_frame_rotation_wxyz == pytest.approx(
        (0.5, 0.5, 0.5, 0.5)
    )


def test_load_go2_spec_has_legal_default_configuration():
    spec = load_robot_spec(CONFIGS / "unitree_go2.yaml", ROOT)

    assert spec.robot == "unitree_go2"
    assert spec.root_body == "base"
    assert spec.foot_contact_offset == pytest.approx(0.022)
    for joint_id in range(spec.model.njnt):
        if not spec.model.jnt_limited[joint_id]:
            continue
        value = spec.model.qpos0[spec.model.jnt_qposadr[joint_id]]
        lower, upper = spec.model.jnt_range[joint_id]
        assert lower <= value <= upper


def test_spec_rejects_missing_leg_before_loading_model(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "robot: bad\n"
        "model_type: quadruped\n"
        "mjcf_path: missing.xml\n"
        "root_body: trunk\n"
        "legs: {}\n"
        "motion: {joint_order: [], quaternion_order: xyzw}\n"
    )

    with pytest.raises(ValueError, match="missing legs: FL, FR, RL, RR"):
        load_robot_spec(path, tmp_path)
