from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)


FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"
JOINTS = tuple(f"joint_{index}" for index in range(12))


def test_load_motion_imitation_normalizes_to_wxyz():
    motion = load_motion_imitation(FIXTURE, JOINTS, "xyzw")

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
