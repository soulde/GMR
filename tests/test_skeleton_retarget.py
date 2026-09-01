import numpy as np
import pytest

from general_motion_retargeting.skeleton_retarget import (
    SkeletonTargetReconstructor,
    parse_skeleton_chains,
)


def _segment(parent, child, length):
    return {
        "human_parent": parent,
        "human_child": child,
        "robot_parent_frame": parent.lower(),
        "robot_child_frame": child.lower(),
        "target_length": length,
    }


def _config(segments):
    return {"skeleton_chains": [{"name": "arm", "segments": segments}]}


def _pose(position):
    return [np.asarray(position, dtype=float), np.array([1.0, 0.0, 0.0, 0.0])]


def _frame(elbow=(1, 0, 0), wrist=(1, 1, 0)):
    return {
        "Shoulder": _pose([0, 0, 0]),
        "Elbow": _pose(elbow),
        "Wrist": _pose(wrist),
    }


def _arm_chains():
    return parse_skeleton_chains(
        _config(
            [
                _segment("Shoulder", "Elbow", 0.3),
                _segment("Elbow", "Wrist", 0.2),
            ]
        )
    )


@pytest.mark.parametrize("length", [0.0, -0.1, float("nan"), float("inf")])
def test_parse_skeleton_chains_rejects_invalid_length(length):
    with pytest.raises(ValueError, match=r"arm.*target_length"):
        parse_skeleton_chains(_config([_segment("Shoulder", "Elbow", length)]))


def test_parse_skeleton_chains_rejects_disconnected_segments():
    config = _config(
        [
            _segment("Shoulder", "Elbow", 0.3),
            _segment("Other", "Wrist", 0.2),
        ]
    )
    with pytest.raises(ValueError, match="disconnected"):
        parse_skeleton_chains(config)


def test_parse_skeleton_chains_rejects_duplicate_child_ownership():
    config = {
        "skeleton_chains": [
            {"name": "arm_a", "segments": [_segment("A", "Elbow", 0.3)]},
            {"name": "arm_b", "segments": [_segment("B", "Elbow", 0.3)]},
        ]
    }
    with pytest.raises(ValueError, match=r"Elbow.*more than one"):
        parse_skeleton_chains(config)


def test_reconstruct_uses_configured_lengths_at_every_angle():
    reconstructor = SkeletonTargetReconstructor(_arm_chains())
    frames = (
        _frame(elbow=[1, 0, 0], wrist=[2, 0, 0]),
        _frame(elbow=[0, 1, 0], wrist=[0, 2, 0]),
        _frame(elbow=[1, 1, 0], wrist=[2, 0, 0]),
    )
    for frame in frames:
        result = reconstructor.reconstruct(frame)
        assert np.linalg.norm(result["Elbow"][0] - result["Shoulder"][0]) == pytest.approx(0.3)
        assert np.linalg.norm(result["Wrist"][0] - result["Elbow"][0]) == pytest.approx(0.2)


def test_reconstruct_uses_reconstructed_parent_for_branch():
    config = {
        "skeleton_chains": [
            {"name": "spine", "segments": [_segment("Hips", "Spine", 0.4)]},
            {"name": "arm", "segments": [_segment("Spine", "Shoulder", 0.3)]},
        ]
    }
    frame = {
        "Hips": _pose([0, 0, 0]),
        "Spine": _pose([0, 0, 2]),
        "Shoulder": _pose([0, 1, 2]),
    }
    result = SkeletonTargetReconstructor(parse_skeleton_chains(config)).reconstruct(frame)
    np.testing.assert_allclose(result["Spine"][0], [0, 0, 0.4])
    np.testing.assert_allclose(result["Shoulder"][0], [0, 0.3, 0.4])


def test_reconstruct_does_not_mutate_input():
    frame = _frame()
    original = {name: [value[0].copy(), value[1].copy()] for name, value in frame.items()}
    SkeletonTargetReconstructor(_arm_chains()).reconstruct(frame)
    for name in frame:
        np.testing.assert_array_equal(frame[name][0], original[name][0])
        np.testing.assert_array_equal(frame[name][1], original[name][1])


def test_degenerate_segment_reuses_previous_direction():
    reconstructor = SkeletonTargetReconstructor(_arm_chains())
    reconstructor.reconstruct(_frame(elbow=[1, 0, 0], wrist=[2, 0, 0]))
    result = reconstructor.reconstruct(_frame(elbow=[0, 0, 0], wrist=[0, 0, 0]))
    np.testing.assert_allclose(result["Elbow"][0], [0.3, 0, 0])
    np.testing.assert_allclose(result["Wrist"][0], [0.5, 0, 0])


def test_first_frame_degenerate_segment_fails_clearly():
    with pytest.raises(ValueError, match=r"arm.*Shoulder.*Elbow"):
        SkeletonTargetReconstructor(_arm_chains()).reconstruct(
            _frame(elbow=[0, 0, 0], wrist=[0, 0, 0])
        )


def test_missing_joint_fails_clearly():
    frame = _frame()
    del frame["Wrist"]
    with pytest.raises(KeyError, match=r"arm.*Wrist"):
        SkeletonTargetReconstructor(_arm_chains()).reconstruct(frame)
