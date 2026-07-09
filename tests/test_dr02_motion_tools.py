import pickle
import csv

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


def test_dr02_batch_quality_classification_handles_missing_contacts(tmp_path):
    from scripts.batch_dr02_retarget_pipeline import classify_quality, load_config

    quality_dir = tmp_path / "quality"
    quality_dir.mkdir()

    metrics_path = quality_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "left_foot_height",
                "right_foot_height",
                "left_foot_xy_speed",
                "right_foot_xy_speed",
                "left_contact",
                "right_contact",
            ],
        )
        writer.writeheader()
        for _ in range(3):
            writer.writerow(
                {
                    "left_foot_height": "0.05",
                    "right_foot_height": "0.05",
                    "left_foot_xy_speed": "0.0",
                    "right_foot_xy_speed": "0.0",
                    "left_contact": "0",
                    "right_contact": "0",
                }
            )

    joint_path = quality_dir / "joint_range.csv"
    with joint_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["violation_ratio"])
        writer.writeheader()
        writer.writerow({"violation_ratio": "0.0"})

    status, values = classify_quality(quality_dir, load_config(None))

    assert status == "FAIL"
    assert values["left_support_max_xy_vel"] == np.inf
    assert values["right_support_max_xy_vel"] == np.inf
