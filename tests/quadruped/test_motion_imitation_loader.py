from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)


FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"
DOG_PACE = (
    Path(__file__).resolve().parents[2]
    / "assets/quadrupeds/motions/dog_pace.txt"
)
JOINTS = tuple(f"joint_{index}" for index in range(12))


def test_load_motion_imitation_normalizes_to_wxyz():
    motion = load_motion_imitation(FIXTURE, JOINTS, "wxyz")

    assert motion.fps == pytest.approx(50.0)
    assert motion.loop_mode == "Wrap"
    assert motion.root_pos.shape == (2, 3)
    assert motion.root_rot.shape == (2, 4)
    np.testing.assert_allclose(motion.root_rot[0], [1.0, 0.0, 0.0, 0.0])
    assert motion.joint_pos.shape == (2, 12)
    assert motion.joint_names == JOINTS


def test_load_motion_imitation_rejects_wrong_joint_count(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(
        '{"LoopMode":"Wrap","FrameDuration":0.02,'
        '"Frames":[[0,0,0,0,0,0,1,1]]}'
    )

    with pytest.raises(ValueError, match="expected 19 values"):
        load_motion_imitation(bad, JOINTS, "xyzw")


def test_load_motion_imitation_converts_xyzw_quaternion(tmp_path):
    path = tmp_path / "xyzw.txt"
    values = [0, 0, 0.45, 0, 0, 0, 1, *([0] * 12)]
    path.write_text(
        '{"LoopMode":"Wrap","FrameDuration":0.02,"Frames":['
        + str(values)
        + "]}"
    )

    motion = load_motion_imitation(path, JOINTS, "xyzw")

    np.testing.assert_allclose(motion.root_rot[0], [1, 0, 0, 0])


def test_load_dog_pace_aligns_source_root_to_first_frame():
    motion = load_motion_imitation(
        DOG_PACE,
        JOINTS,
        "wxyz",
        "relative_first_frame",
    )

    np.testing.assert_allclose(
        motion.root_rot[0],
        [1.0, 0.0, 0.0, 0.0],
        atol=1e-7,
    )
