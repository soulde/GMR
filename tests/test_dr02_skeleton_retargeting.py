import json
from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.params import SKELETON_CONFIG_DICT
from general_motion_retargeting.skeleton_retarget import (
    SkeletonTargetReconstructor,
    parse_skeleton_chains,
)


EXPECTED_LENGTHS = {
    ("Hips", "Spine2"): 0.29,
    ("Spine2", "LeftArm"): 0.3246858828159918,
    ("LeftArm", "LeftForeArm"): 0.265,
    ("LeftForeArm", "LeftHand"): 0.2495,
    ("Spine2", "RightArm"): 0.3246858828159918,
    ("RightArm", "RightForeArm"): 0.265,
    ("RightForeArm", "RightHand"): 0.2495,
    ("Hips", "LeftUpLeg"): 0.1614,
    ("LeftUpLeg", "LeftLeg"): 0.4194,
    ("LeftLeg", "LeftFoot"): 0.41,
    ("LeftFoot", "LeftToe"): 0.1110180165558726,
    ("Hips", "RightUpLeg"): 0.1614,
    ("RightUpLeg", "RightLeg"): 0.4194,
    ("RightLeg", "RightFoot"): 0.41,
    ("RightFoot", "RightToe"): 0.1110180165558726,
}


def _load_config():
    path = SKELETON_CONFIG_DICT["bvh_lafan1"]["dr02"]
    with path.open() as stream:
        return json.load(stream)


def _configured_lengths(config):
    return {
        (segment["human_parent"], segment["human_child"]): segment[
            "target_length"
        ]
        for chain in config["skeleton_chains"]
        for segment in chain["segments"]
    }


def test_dr02_skeleton_config_registered():
    path = SKELETON_CONFIG_DICT["bvh_lafan1"]["dr02"]
    assert path.name == "bvh_lafan1_to_dr02.json"
    assert path.parent.name == "skeleton_configs"
    assert path.is_file()


def test_dr02_skeleton_config_has_full_body_lengths():
    config = _load_config()
    assert config["algorithm"] == "skeleton"
    assert _configured_lengths(config) == pytest.approx(EXPECTED_LENGTHS)


def test_dr02_skeleton_chain_scales_are_one():
    config = _load_config()
    chain_joints = {
        joint
        for chain in config["skeleton_chains"]
        for segment in chain["segments"]
        for joint in (segment["human_parent"], segment["human_child"])
    }
    for joint in chain_joints:
        assert config["human_scale_table"][joint] == pytest.approx(1.0)


def test_dr02_skeleton_length_offsets_are_zero():
    config = _load_config()
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for entry in config[table_name].values():
            assert entry[3] == [0, 0, 0]


def test_dr02_skeleton_prioritizes_elbows_over_wrists_in_stage_two():
    config = _load_config()
    stage1 = config["ik_match_table1"]
    stage2 = config["ik_match_table2"]

    for side in ("left", "right"):
        assert stage1[f"{side}_elbow_link"][1] == pytest.approx(8)
        assert stage2[f"{side}_elbow_link"][1] == pytest.approx(12)
        assert stage2[f"{side}_wrist_x_link"][1] == pytest.approx(10)


def test_dr02_skeleton_uses_minimum_solver_iterations_for_large_motion():
    settings = _load_config()["solver_settings"]

    assert settings == {
        "minimum_iterations": 10,
        "maximum_iterations": 30,
        "position_error_threshold": pytest.approx(0.01),
    }


DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "LAFAN1"
DANCE_BVH = DATA_ROOT / "dance" / "dance1_subject1.bvh"


@pytest.mark.skipif(not DANCE_BVH.is_file(), reason="local LAFAN1 fixture unavailable")
def test_dr02_full_body_reconstruction_has_constant_lengths():
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    config = _load_config()
    reconstructor = SkeletonTargetReconstructor(parse_skeleton_chains(config))
    frames, _ = load_bvh_file(DANCE_BVH)
    for frame in frames[::30]:
        reconstructed = reconstructor.reconstruct(frame)
        for (parent, child), expected in EXPECTED_LENGTHS.items():
            actual = np.linalg.norm(
                reconstructed[child][0] - reconstructed[parent][0]
            )
            assert actual == pytest.approx(expected, abs=1e-6)


@pytest.mark.skipif(not DANCE_BVH.is_file(), reason="local LAFAN1 fixture unavailable")
def test_dr02_skeleton_retargeter_produces_finite_qpos():
    from general_motion_retargeting import SkeletonMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    frames, human_height = load_bvh_file(DANCE_BVH)
    retargeter = SkeletonMotionRetargeting(
        src_human="bvh_lafan1",
        tgt_robot="dr02",
        actual_human_height=human_height,
    )

    qpos = [retargeter.retarget(frame) for frame in frames[::30]]

    assert qpos
    assert all(np.isfinite(frame_qpos).all() for frame_qpos in qpos)
