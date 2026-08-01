from general_motion_retargeting.utils.smpl import _smplx_model_extension


def test_smplx_model_extension_detects_installed_format(tmp_path):
    model_dir = tmp_path / "smplx"
    model_dir.mkdir()
    (model_dir / "SMPLX_FEMALE.pkl").touch()

    assert _smplx_model_extension(tmp_path, "female") == "pkl"


def test_smplx_model_extension_defaults_to_npz(tmp_path):
    assert _smplx_model_extension(tmp_path, "neutral") == "npz"
