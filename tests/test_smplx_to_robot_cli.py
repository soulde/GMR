import numpy as np
import pytest

from general_motion_retargeting.retarget_export import ExportPaths
from scripts import smplx_to_robot


def test_parser_replaces_save_path_with_save():
    args = smplx_to_robot.build_parser().parse_args(
        ["--smplx_file", "walk.npz", "--robot", "dr02", "--save"]
    )

    assert args.save is True
    assert not hasattr(args, "save_path")


def test_main_rejects_looping_save():
    with pytest.raises(SystemExit):
        smplx_to_robot.main(
            [
                "--smplx_file",
                "walk.npz",
                "--robot",
                "dr02",
                "--save",
                "--loop",
            ]
        )


def test_main_exports_all_retargeted_frames(monkeypatch):
    qpos = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25])
    model = object()
    calls = []

    class FakeRetargeter:
        def __init__(self, **kwargs):
            assert kwargs == {
                "actual_human_height": 1.7,
                "src_human": "smplx",
                "tgt_robot": "dr02",
            }
            self.model = model
            self.scaled_human_data = {}

        def retarget(self, frame, offset_to_ground=False):
            assert frame == {"frame": 0}
            assert offset_to_ground is False
            return qpos.copy()

    monkeypatch.setattr(
        smplx_to_robot,
        "load_smplx_file",
        lambda *_: ("smplx", "body_model", "smplx_output", 1.7),
    )
    monkeypatch.setattr(
        smplx_to_robot,
        "get_smplx_data_offline_fast",
        lambda *_args, **_kwargs: ([{"frame": 0}], 30.0),
    )
    monkeypatch.setattr(smplx_to_robot, "GMR", FakeRetargeter)
    monkeypatch.setattr(
        smplx_to_robot,
        "export_retarget_motion",
        lambda *args, **kwargs: (
                calls.append((args, kwargs))
            or ExportPaths(
                "joints.json",
                "manifest.json",
                "walk.pkl",
                "walk.npz",
                "walk.csv",
            )
        ),
    )

    assert (
        smplx_to_robot.main(
            [
                "--smplx_file",
                "/data/walk.npz",
                "--robot",
                "dr02",
                "--no_viewer",
                "--save",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[:4] == (model, "dr02", "/data/walk.npz", 30.0)
    np.testing.assert_allclose(args[4], qpos[None, :])
    assert kwargs == {}
