import pytest

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
from general_motion_retargeting.retarget_export import (
    export_paths,
    update_motion_manifest,
)
from scripts.vis_gmr_debug import (
    align_reference_frames,
    build_parser,
    parse_height_offset,
    resolve_viewer_inputs,
)


def test_unified_parser_requires_only_motion():
    parser = build_parser()
    args = parser.parse_args(["--motion", "motion.pkl"])
    assert args.motion.name == "motion.pkl"
    assert args.reference is None
    assert args.mjcf is None
    assert args.ik_config is None
    assert args.height_offset == "auto"
    assert args.loop is True


def manifest_fixture(tmp_path, robot="dr02", *, source_exists=True):
    repository = tmp_path / "repository"
    source = repository / "sources" / "walk.npz"
    source.parent.mkdir(parents=True)
    if source_exists:
        source.touch()
    paths = export_paths(robot, source, repository / "retarget_data")
    paths.motion.parent.mkdir(parents=True)
    paths.motion.touch()
    update_motion_manifest(paths, robot, source, repository_root=repository)
    return repository, source, paths


def test_viewer_inputs_are_inferred_from_motion_manifest(tmp_path):
    repository, source, paths = manifest_fixture(tmp_path)
    args = build_parser().parse_args(["--motion", str(paths.motion)])

    resolved = resolve_viewer_inputs(args, repository_root=repository)

    assert resolved.reference == source.resolve()
    assert resolved.mjcf == ROBOT_XML_DICT["dr02"].resolve()
    assert resolved.ik_config == IK_CONFIG_DICT["smplx"]["dr02"].resolve()


def test_each_explicit_viewer_input_overrides_manifest_value(tmp_path):
    repository, _, paths = manifest_fixture(tmp_path, source_exists=False)
    reference = tmp_path / "override.npz"
    mjcf = tmp_path / "override.xml"
    ik_config = tmp_path / "override.json"
    for path in (reference, mjcf, ik_config):
        path.touch()
    args = build_parser().parse_args(
        [
            "--motion",
            str(paths.motion),
            "--reference",
            str(reference),
            "--mjcf",
            str(mjcf),
            "--ik-config",
            str(ik_config),
        ]
    )

    resolved = resolve_viewer_inputs(args, repository_root=repository)

    assert resolved.reference == reference.resolve()
    assert resolved.mjcf == mjcf.resolve()
    assert resolved.ik_config == ik_config.resolve()


def test_fully_explicit_viewer_inputs_do_not_require_manifest(tmp_path):
    motion = tmp_path / "motion.pkl"
    reference = tmp_path / "reference.npz"
    mjcf = tmp_path / "robot.xml"
    ik_config = tmp_path / "config.json"
    for path in (motion, reference, mjcf, ik_config):
        path.touch()
    args = build_parser().parse_args(
        [
            "--motion",
            str(motion),
            "--reference",
            str(reference),
            "--mjcf",
            str(mjcf),
            "--ik-config",
            str(ik_config),
        ]
    )

    resolved = resolve_viewer_inputs(args, repository_root=tmp_path)

    assert resolved.reference == reference.resolve()
    assert resolved.mjcf == mjcf.resolve()
    assert resolved.ik_config == ik_config.resolve()


def test_viewer_inputs_reject_unknown_manifest_robot(tmp_path):
    repository, _, paths = manifest_fixture(tmp_path, robot="unknown")
    args = build_parser().parse_args(["--motion", str(paths.motion)])

    with pytest.raises(ValueError, match="unknown robot"):
        resolve_viewer_inputs(args, repository_root=repository)


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
