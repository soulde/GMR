import json
import pathlib

import numpy as np
import pytest

from general_motion_retargeting import GeneralMotionRetargeting
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


def external_site_robot(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text(
        """
        <mujoco>
          <compiler boundmass="0.001" boundinertia="0.001"/>
          <worldbody>
            <body name="root">
              <freejoint name="root"/>
              <geom type="sphere" size="0.05" mass="1"/>
              <body name="arm">
                <joint name="elbow" axis="0 1 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 .2 0 0"/>
                <site name="wrist" pos=".2 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "robot_root_name": "root",
                "human_root_name": "pelvis",
                "human_height_assumption": 1.8,
                "ground_height": 0.0,
                "human_scale_table": {"pelvis": 1.0, "hand": 1.0},
                "use_ik_match_table1": True,
                "use_ik_match_table2": False,
                "ik_match_table1": {
                    "root": [
                        "pelvis", 1.0, 0.0, [0, 0, 0], [1, 0, 0, 0]
                    ],
                    "wrist": [
                        "hand", 1.0, 0.0, [0, 0, 0], [1, 0, 0, 0]
                    ]
                },
                "ik_match_table2": {},
                "global_position_offsets": {"hand": [0.1, 0.2, 0.3]},
                "initialize_root_from_human": True,
                "initialization_retargets": 1,
                "initial_joint_positions": {"elbow": 0.25},
            }
        )
    )
    return xml, config


def test_external_config_can_target_body_or_site_frames(tmp_path):
    xml, config = external_site_robot(tmp_path)
    retarget = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="external",
        robot_xml_path=xml,
        ik_config_path=config,
        verbose=False,
    )

    assert retarget._resolve_frame_type("root") == "body"
    assert retarget._resolve_frame_type("wrist") == "site"
    with pytest.raises(ValueError, match="unknown MuJoCo body or site frame"):
        retarget._resolve_frame_type("missing")


def test_external_config_applies_global_offsets_and_initial_pose(tmp_path):
    xml, config = external_site_robot(tmp_path)
    retarget = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="external",
        robot_xml_path=xml,
        ik_config_path=config,
        verbose=False,
    )
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    root_position = np.array([1.0, 2.0, 3.0])
    retarget.update_targets(
        {
            "pelvis": (root_position, identity.copy()),
            "hand": (root_position, identity.copy()),
        }
    )

    np.testing.assert_allclose(
        retarget.scaled_human_data["hand"][0],
        root_position + np.array([0.1, 0.2, 0.3]),
    )
    retarget._initialize_root_from_human_target()
    np.testing.assert_allclose(retarget.configuration.data.qpos[:3], root_position)
    assert retarget.configuration.data.qpos[
        retarget.model.joint("elbow").qposadr[0]
    ] == pytest.approx(0.25)
