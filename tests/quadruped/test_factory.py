import mujoco
import pytest

from general_motion_retargeting.factory import create_retargeter
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec
from general_motion_retargeting.params import (
    QUADRUPED_IK_CONFIG_DICT,
    QUADRUPED_ROBOT_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
)


def test_factory_defaults_to_humanoid(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "general_motion_retargeting.factory.GeneralMotionRetargeting",
        lambda **kwargs: sentinel,
    )

    actual = create_retargeter(src_human="smplx", tgt_robot="unitree_g1")

    assert actual is sentinel


def test_factory_selects_quadruped(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "general_motion_retargeting.factory.QuadrupedRobotRetargeter",
        lambda **kwargs: sentinel,
    )

    actual = create_retargeter(model_type="quadruped")

    assert actual is sentinel


def test_quadruped_retargeter_exposes_common_gmr_model_metadata():
    from general_motion_retargeting.quadruped.config import load_retarget_config

    root = QUADRUPED_ROBOT_CONFIG_DICT["unitree_go2"].parents[3]
    source = load_robot_spec(QUADRUPED_ROBOT_CONFIG_DICT["laikago"], root)
    target = load_robot_spec(
        QUADRUPED_ROBOT_CONFIG_DICT["unitree_go2"], root
    )
    config = load_retarget_config(
        QUADRUPED_IK_CONFIG_DICT["laikago"]["unitree_go2"]
    )

    retargeter = create_retargeter(
        model_type="quadruped",
        source_spec=source,
        target_spec=target,
        config=config,
    )

    assert retargeter.model is target.model
    assert retargeter.xml_file == str(target.mjcf_path)
    assert "base" in retargeter.robot_body_names
    assert "FL_thigh_joint" in retargeter.robot_dof_names
    assert "FL_thigh" in retargeter.robot_motor_names
    assert retargeter.ik_limits is retargeter.limits


def test_factory_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="model_type must be"):
        create_retargeter(model_type="hexapod")


def test_go2_registration_loads_and_resolves_base():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_go2"]))

    assert ROBOT_BASE_DICT["unitree_go2"] == "base"
    assert model.body(ROBOT_BASE_DICT["unitree_go2"]).id >= 0
    assert QUADRUPED_ROBOT_CONFIG_DICT["unitree_go2"].is_file()
    assert QUADRUPED_IK_CONFIG_DICT["laikago"]["unitree_go2"].is_file()
    spec = load_robot_spec(
        QUADRUPED_ROBOT_CONFIG_DICT["unitree_go2"],
        QUADRUPED_ROBOT_CONFIG_DICT["unitree_go2"].parents[3],
    )
    assert spec.mjcf_path.resolve() == ROBOT_XML_DICT["unitree_go2"].resolve()
