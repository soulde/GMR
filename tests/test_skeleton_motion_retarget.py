import json

import numpy as np
import pytest

from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.skeleton_motion_retarget import (
    SkeletonMotionRetargeting,
)


DR02_XML = "assets/dr02/mjcf/dr02_pos.xml"


def _entry(human_name, position_weight=1, orientation_weight=0, offset=None):
    return [
        human_name,
        position_weight,
        orientation_weight,
        [0, 0, 0] if offset is None else offset,
        [1, 0, 0, 0],
    ]


def _write_config(tmp_path, *, robot_child="body", offset=None, scale=1.0):
    config = {
        "algorithm": "skeleton",
        "robot_root_name": "base_link",
        "human_root_name": "Hips",
        "ground_height": 0,
        "human_height_assumption": 1.75,
        "initialize_root_from_human": False,
        "initialization_retargets": 1,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": {"Hips": 1.0, "Spine2": scale},
        "ik_match_table1": {
            "base_link": _entry("Hips", 1),
            "body": _entry("Spine2", 1, offset=offset),
        },
        "ik_match_table2": {
            "base_link": _entry("Hips", 1),
            "body": _entry("Spine2", 1, offset=offset),
        },
        "skeleton_chains": [
            {
                "name": "spine",
                "segments": [
                    {
                        "human_parent": "Hips",
                        "human_child": "Spine2",
                        "robot_parent_frame": "base_link",
                        "robot_child_frame": robot_child,
                        "target_length": 0.29,
                    }
                ],
            }
        ],
    }
    path = tmp_path / "skeleton.json"
    path.write_text(json.dumps(config))
    return path


def _frame():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "Hips": [np.array([0.0, 0.0, 1.0]), identity.copy()],
        "Spine2": [np.array([0.0, 0.0, 2.0]), identity.copy()],
    }


def _retargeter(config_path):
    return SkeletonMotionRetargeting(
        src_human="fixture",
        tgt_robot="dr02",
        skeleton_config_path=config_path,
        robot_xml_path=DR02_XML,
        verbose=False,
    )


def test_skeleton_retargeter_has_gmr_compatible_api(tmp_path):
    retargeter = _retargeter(_write_config(tmp_path))
    qpos = retargeter.retarget(_frame())
    assert qpos.shape == (retargeter.model.nq,)
    assert np.isfinite(qpos).all()


def test_skeleton_retargeter_has_independent_target_preparation():
    assert (
        GeneralMotionRetargeting.update_targets
        is not SkeletonMotionRetargeting.update_targets
    )


def test_reconstructed_segment_uses_configured_length(tmp_path):
    retargeter = _retargeter(_write_config(tmp_path))
    retargeter.update_targets(_frame())
    data = retargeter.reconstructed_human_data
    assert np.linalg.norm(data["Spine2"][0] - data["Hips"][0]) == pytest.approx(0.29)


def test_chain_scale_must_be_one(tmp_path):
    with pytest.raises(ValueError, match=r"Spine2.*1.0"):
        _retargeter(_write_config(tmp_path, scale=1.1))


def test_missing_robot_frame_fails_during_construction(tmp_path):
    with pytest.raises(ValueError, match="missing_frame"):
        _retargeter(_write_config(tmp_path, robot_child="missing_frame"))


def test_segment_diagnostic_preserves_configured_length(tmp_path):
    retargeter = _retargeter(_write_config(tmp_path))
    diagnostic = retargeter.segment_diagnostics[("Hips", "Spine2")]
    assert diagnostic["configured_length"] == pytest.approx(0.29)
    assert diagnostic["mjcf_reference_distance"] == pytest.approx(0.29)


def test_large_position_offset_warns(tmp_path):
    with pytest.warns(UserWarning, match=r"Spine2.*0.05"):
        _retargeter(_write_config(tmp_path, offset=[0, 0, 0.06]))
