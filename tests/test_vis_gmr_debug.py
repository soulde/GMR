import pytest

from scripts.vis_gmr_debug import (
    align_reference_frames,
    build_parser,
    parse_height_offset,
)


def test_unified_parser_requires_motion_reference_mjcf_and_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--motion", "motion.pkl",
            "--reference", "reference.npz",
            "--mjcf", "robot.xml",
            "--ik-config", "config.json",
        ]
    )
    assert args.motion.name == "motion.pkl"
    assert args.height_offset == "auto"
    assert args.loop is True


def test_height_offset_accepts_auto_or_float_and_rejects_other_text():
    assert parse_height_offset("auto") == "auto"
    assert parse_height_offset("0.043") == pytest.approx(0.043)
    with pytest.raises(ValueError, match="auto or a number"):
        parse_height_offset("ground")


def test_reference_alignment_reproduces_saved_smplx_frame_offset():
    frames = list(range(145))
    assert align_reference_frames(frames, 144) == list(range(1, 145))
    assert align_reference_frames(list(range(144)), 144) == list(range(144))
    with pytest.raises(ValueError, match="frame count mismatch"):
        align_reference_frames(list(range(146)), 144)
