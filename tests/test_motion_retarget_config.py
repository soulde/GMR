import pathlib

import pytest

from general_motion_retargeting.motion_retarget import (
    resolve_ik_config,
    resolve_robot_xml,
)
from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT


def test_explicit_paths_allow_an_unregistered_robot(tmp_path):
    xml = tmp_path / "robot.xml"
    config = tmp_path / "config.json"
    xml.write_text("<mujoco/>")
    config.write_text("{}")

    assert resolve_robot_xml("external_test_robot", xml) == xml
    assert resolve_ik_config("smplx", "external_test_robot", config) == config


def test_registered_robot_still_uses_registry():
    assert resolve_robot_xml("unitree_g1") == pathlib.Path(
        ROBOT_XML_DICT["unitree_g1"]
    )
    assert resolve_ik_config("smplx", "unitree_g1") == pathlib.Path(
        IK_CONFIG_DICT["smplx"]["unitree_g1"]
    )


def test_unknown_robot_requires_explicit_model_path():
    with pytest.raises(KeyError, match="unknown robot.*robot_xml_path|--mjcf"):
        resolve_robot_xml("external_test_robot")


def test_missing_explicit_paths_report_the_path(tmp_path):
    missing_xml = tmp_path / "missing.xml"
    missing_config = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match=str(missing_xml)):
        resolve_robot_xml("external_test_robot", missing_xml)
    with pytest.raises(FileNotFoundError, match=str(missing_config)):
        resolve_ik_config("smplx", "external_test_robot", missing_config)
