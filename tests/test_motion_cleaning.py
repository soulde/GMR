import numpy as np
import pytest

from general_motion_retargeting.motion_cleaning import (
    clean_motion_segments,
    contiguous_valid_segments,
)


def test_bad_frame_splits_motion_without_joining_across_gap():
    errors = np.array([0.01, 0.02, 0.20, 0.03, 0.04])

    assert contiguous_valid_segments(errors, threshold=0.10) == ((0, 2), (3, 5))


def test_consecutive_bad_frames_form_one_removed_gap():
    errors = np.array([0.01, 0.20, 0.30, 0.02])

    assert contiguous_valid_segments(errors, threshold=0.10) == ((0, 1), (3, 4))


def test_short_valid_fragments_are_discarded():
    errors = np.array([0.01, 0.20, 0.01, 0.01, 0.20, 0.01])

    assert contiguous_valid_segments(
        errors, threshold=0.10, minimum_segment_frames=2
    ) == ((2, 4),)


def test_padding_removes_neighboring_frames_around_failure():
    errors = np.array([0.01, 0.01, 0.20, 0.01, 0.01])

    assert contiguous_valid_segments(
        errors, threshold=0.10, padding_frames=1
    ) == ((0, 1), (4, 5))


def test_clean_motion_slices_every_frame_aligned_array():
    motion = {
        "fps": 30,
        "root_pos": np.arange(15).reshape(5, 3),
        "root_rot": np.arange(20).reshape(5, 4),
        "dof_pos": np.arange(10).reshape(5, 2),
        "local_body_pos": None,
        "link_body_list": ["base"],
    }

    parts = clean_motion_segments(motion, ((0, 2), (3, 5)))

    assert len(parts) == 2
    assert parts[0]["fps"] == 30
    assert parts[0]["root_pos"].tolist() == motion["root_pos"][:2].tolist()
    assert parts[1]["dof_pos"].tolist() == motion["dof_pos"][3:5].tolist()
    assert parts[1]["link_body_list"] == ["base"]


def test_clean_motion_rejects_inconsistent_frame_counts():
    motion = {
        "fps": 30,
        "root_pos": np.zeros((3, 3)),
        "root_rot": np.zeros((2, 4)),
        "dof_pos": np.zeros((3, 2)),
    }

    with pytest.raises(ValueError, match="frame counts"):
        clean_motion_segments(motion, ((0, 2),))
