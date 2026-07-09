import pickle

import numpy as np


def fake_motion(path):
    T = 30
    data = {
        "fps": 30.0,
        "root_pos": np.tile(np.array([0.0, 0.0, 0.9]), (T, 1)),
        "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (T, 1)),
        "dof_pos": np.zeros((T, 21)),
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)


def test_dr02_xml_loads():
    import mujoco

    model = mujoco.MjModel.from_xml_path("assets/robots/dr02/dr02.xml")
    assert model.nq == 28
    assert model.nv == 27
    assert model.nu == 21


def test_dr02_motion_tools_fake_motion(tmp_path):
    import mujoco

    from general_motion_retargeting.dr02.motion_tools import compute_dataset_fields, compute_kinematic_metrics, load_motion

    path = tmp_path / "fake.pkl"
    fake_motion(path)
    motion = load_motion(path)
    model = mujoco.MjModel.from_xml_path("assets/robots/dr02/dr02.xml")
    metrics = compute_kinematic_metrics(model, motion)
    dataset = compute_dataset_fields(model, motion)

    assert metrics["joint_pos"].shape == (30, 21)
    assert dataset["joint_vel"].shape == (30, 21)
    for value in dataset.values():
        if isinstance(value, np.ndarray) and value.dtype != object:
            assert np.all(np.isfinite(value))
