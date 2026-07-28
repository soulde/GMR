import mujoco
import pytest

from general_motion_retargeting.factory import create_retargeter
from general_motion_retargeting.params import ROBOT_BASE_DICT, ROBOT_XML_DICT


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


def test_factory_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="model_type must be"):
        create_retargeter(model_type="hexapod")


def test_go2_registration_loads_and_resolves_base():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_go2"]))

    assert ROBOT_BASE_DICT["unitree_go2"] == "base"
    assert model.body(ROBOT_BASE_DICT["unitree_go2"]).id >= 0
