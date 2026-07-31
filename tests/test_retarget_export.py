import json
import pickle

import mujoco
import numpy as np
import pytest

from general_motion_retargeting.retarget_export import (
    ensure_joint_contract,
    export_retarget_motion,
    export_paths,
    scalar_joint_names,
)


def model_from_joints(joints: str, *, free_root: bool = True):
    root_joint = "<freejoint name='root'/>" if free_root else ""
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body name="base">
              {root_joint}
              <geom type="sphere" size="0.1"/>
              <body name="joint_body">
                {joints}
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def test_export_paths_use_robot_and_source_stem(tmp_path):
    paths = export_paths("dr02", "/data/walk_stageii.npz", tmp_path)

    assert paths.joints == tmp_path / "dr02" / "joints.json"
    assert paths.motion == tmp_path / "dr02" / "motions" / "walk_stageii.pkl"
    assert paths.dataset == tmp_path / "dr02" / "datasets" / "walk_stageii.npz"
    assert paths.csv == tmp_path / "dr02" / "beyondmimic" / "walk_stageii.csv"


def test_scalar_joint_names_follow_qpos_addresses():
    model = model_from_joints(
        """
        <joint name="hip" type="hinge"/>
        <joint name="knee" type="slide"/>
        """
    )

    assert scalar_joint_names(model) == ("hip", "knee")


def test_scalar_joint_names_requires_free_root():
    model = model_from_joints("<joint name='hip' type='hinge'/>", free_root=False)

    with pytest.raises(ValueError, match="free root"):
        scalar_joint_names(model)


def test_scalar_joint_names_rejects_multidof_joint():
    model = model_from_joints("<joint name='shoulder' type='ball'/>")

    with pytest.raises(ValueError, match="scalar"):
        scalar_joint_names(model)


def test_scalar_joint_names_rejects_unnamed_joint():
    model = model_from_joints("<joint type='hinge'/>")

    with pytest.raises(ValueError, match="named"):
        scalar_joint_names(model)


def test_joint_contract_is_created_and_reused(tmp_path):
    path = tmp_path / "dr02" / "joints.json"
    expected = {
        "format_version": 1,
        "robot": "dr02",
        "joint_names": ["hip", "knee"],
    }

    ensure_joint_contract(path, "dr02", ("hip", "knee"))
    first_contents = path.read_text()
    ensure_joint_contract(path, "dr02", ("hip", "knee"))

    assert json.loads(first_contents) == expected
    assert path.read_text() == first_contents


def test_joint_contract_rejects_existing_mismatch(tmp_path):
    path = tmp_path / "joints.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "robot": "dr02",
                "joint_names": ["knee", "hip"],
            }
        )
    )

    with pytest.raises(ValueError, match="does not match"):
        ensure_joint_contract(path, "dr02", ("hip", "knee"))


def two_joint_qpos():
    return np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2],
            [0.5, 0.0, 1.0, 2**-0.5, 0.0, 0.0, 2**-0.5, 0.3, 0.6],
        ]
    )


def test_export_retarget_motion_writes_all_artifacts(tmp_path):
    model = model_from_joints(
        "<joint name='hip' type='hinge'/><joint name='knee' type='slide'/>"
    )

    paths = export_retarget_motion(
        model,
        "dr02",
        "/data/walk.npz",
        10.0,
        two_joint_qpos(),
        output_root=tmp_path,
    )

    with paths.motion.open("rb") as stream:
        pkl = pickle.load(stream)
    assert set(pkl) == {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "local_body_pos",
        "link_body_list",
    }
    np.testing.assert_allclose(pkl["root_rot"][0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(pkl["dof_pos"], [[0.1, 0.2], [0.3, 0.6]])

    with np.load(paths.dataset) as dataset:
        assert set(dataset.files) == {
            "fps",
            "root_pos",
            "root_quat",
            "root_lin_vel",
            "root_ang_vel",
            "joint_pos",
            "joint_vel",
            "joint_names",
        }
        assert dataset["joint_names"].tolist() == ["hip", "knee"]
        np.testing.assert_allclose(dataset["root_quat"][0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(dataset["root_lin_vel"], [[5.0, 0.0, 0.0]] * 2)
        np.testing.assert_allclose(dataset["joint_vel"], [[2.0, 4.0]] * 2)
        np.testing.assert_allclose(
            dataset["root_ang_vel"], [[0.0, 0.0, 5.0 * np.pi]] * 2
        )

    csv = np.loadtxt(paths.csv, delimiter=",")
    assert csv.shape == (2, 9)
    np.testing.assert_allclose(csv[:, 7:], [[0.1, 0.2], [0.3, 0.6]])


def test_csv_downsamples_motion_above_30_fps(tmp_path):
    model = model_from_joints("<joint name='hip' type='hinge'/>")
    qpos = np.repeat(two_joint_qpos()[:1, :8], 4, axis=0)
    qpos[:, 0] = np.arange(4)

    paths = export_retarget_motion(
        model, "dr02", "run.npz", 60.0, qpos, output_root=tmp_path
    )

    csv = np.loadtxt(paths.csv, delimiter=",")
    np.testing.assert_allclose(csv[:, 0], [0.0, 2.0])


@pytest.mark.parametrize(
    ("fps", "qpos", "message"),
    [
        (0.0, two_joint_qpos(), "positive finite fps"),
        (30.0, np.empty((0, 9)), "at least one frame"),
        (30.0, np.zeros((2, 8)), "qpos"),
        (30.0, np.full((2, 9), np.nan), "finite"),
    ],
)
def test_export_retarget_motion_rejects_invalid_motion(
    tmp_path, fps, qpos, message
):
    model = model_from_joints(
        "<joint name='hip' type='hinge'/><joint name='knee' type='slide'/>"
    )

    with pytest.raises(ValueError, match=message):
        export_retarget_motion(
            model, "dr02", "bad.npz", fps, qpos, output_root=tmp_path
        )
