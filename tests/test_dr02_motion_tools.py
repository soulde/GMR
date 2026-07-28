import pickle
import csv
from argparse import Namespace

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

    from general_motion_retargeting.utils.robot_motion import (
        compute_dataset_fields,
        compute_kinematic_metrics,
        load_motion,
    )

    path = tmp_path / "fake.pkl"
    fake_motion(path)
    motion = load_motion(path)
    model = mujoco.MjModel.from_xml_path("assets/robots/dr02/dr02.xml")
    foot_sites = {"left_foot": "left_foot", "right_foot": "right_foot"}
    metrics = compute_kinematic_metrics(model, motion, foot_sites)
    dataset = compute_dataset_fields(model, motion, foot_sites)

    assert metrics["joint_pos"].shape == (30, 21)
    assert dataset["joint_vel"].shape == (30, 21)
    for value in dataset.values():
        if isinstance(value, np.ndarray) and value.dtype != object:
            assert np.all(np.isfinite(value))


def test_robot_motion_tools_support_configured_end_effectors(tmp_path):
    import mujoco

    from general_motion_retargeting.utils.robot_motion import (
        compute_kinematic_metrics,
        estimate_contacts,
        load_motion,
    )

    path = tmp_path / "fake.pkl"
    fake_motion(path)
    motion = load_motion(path)
    model = mujoco.MjModel.from_xml_path("assets/robots/dr02/dr02.xml")
    sites = {
        "left_foot": "left_foot",
        "right_foot": "right_foot",
    }

    metrics = compute_kinematic_metrics(model, motion, sites)
    contacts = estimate_contacts(metrics, sites)

    assert set(contacts) == set(sites)
    for label in sites:
        assert metrics[f"{label}_pos"].shape == (30, 3)
        assert metrics[f"{label}_vel"].shape == (30, 3)
        assert contacts[label].shape == (30,)


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


def test_dr02_batch_preserves_source_tree(tmp_path):
    from scripts.batch_dr02_retarget_pipeline import output_paths

    input_dir = tmp_path / "input"
    motion_path = input_dir / "subject" / "walk_stageii.npz"
    args = Namespace(
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        preserve_tree=True,
    )

    name, raw_pkl, pre_pkl, dataset_npz = output_paths(args, motion_path)

    assert name == "subject__walk_stageii"
    assert raw_pkl == tmp_path / "output/raw/subject/walk_stageii.pkl"
    assert pre_pkl == tmp_path / "output/preprocessed/subject/walk_stageii_preprocessed.pkl"
    assert dataset_npz == tmp_path / "output/dataset/subject/walk_stageii_dataset.npz"


def test_dr02_batch_excludes_by_relative_path_name_or_stem(tmp_path):
    from scripts.batch_dr02_retarget_pipeline import is_excluded, load_excludes

    input_dir = tmp_path / "input"
    exclude_file = tmp_path / "exclude.txt"
    exclude_file.write_text(
        "\n".join(
            [
                "# known bad motions",
                "subject/path_stageii.npz",
                "name_stageii.npz",
                "stem_stageii",
            ]
        )
    )
    excludes = load_excludes(exclude_file, input_dir)

    assert is_excluded(input_dir / "subject/path_stageii.npz", input_dir, excludes)
    assert is_excluded(input_dir / "other/name_stageii.npz", input_dir, excludes)
    assert is_excluded(input_dir / "third/stem_stageii.npz", input_dir, excludes)
    assert not is_excluded(input_dir / "subject/good_stageii.npz", input_dir, excludes)
