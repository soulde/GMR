import json
import pickle

import mujoco
import numpy as np
import pytest

from general_motion_retargeting.retarget_export import (
    encode_source_path,
    ensure_joint_contract,
    export_retarget_motion,
    export_paths,
    load_motion_manifest,
    scalar_joint_names,
    update_motion_manifest,
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


def test_encode_source_inside_repository_is_relative(tmp_path):
    source = tmp_path / "motion_data" / "walk.npz"
    source.parent.mkdir()
    source.touch()

    assert encode_source_path(source, tmp_path) == {
        "path": "motion_data/walk.npz",
        "base": "repository",
    }


def test_encode_external_source_is_absolute(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    source = tmp_path / "external" / "walk.npz"

    assert encode_source_path(source, repository) == {
        "path": source.resolve().as_posix(),
        "base": "absolute",
    }


def test_encode_relative_source_uses_working_directory(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "inputs" / "walk.npz"
    source.parent.mkdir(parents=True)
    source.touch()

    assert encode_source_path("inputs/walk.npz", repository, cwd=repository) == {
        "path": "inputs/walk.npz",
        "base": "repository",
    }


def test_manifest_preserves_motions_and_sorts_keys(tmp_path):
    repository = tmp_path / "repository"
    source_a = repository / "sources" / "a.npz"
    source_b = repository / "sources" / "b.npz"
    source_a.parent.mkdir(parents=True)
    source_a.touch()
    source_b.touch()
    paths_b = export_paths("dr02", source_b, repository / "retarget_data")
    paths_a = export_paths("dr02", source_a, repository / "retarget_data")

    update_motion_manifest(paths_b, "dr02", source_b, repository_root=repository)
    update_motion_manifest(paths_a, "dr02", source_a, repository_root=repository)

    manifest = json.loads(paths_a.manifest.read_text())
    assert manifest["format_version"] == 1
    assert manifest["robot"] == "dr02"
    assert list(manifest["motions"]) == ["motions/a.pkl", "motions/b.pkl"]
    assert manifest["motions"]["motions/a.pkl"] == {
        "source": {"path": "sources/a.npz", "base": "repository"},
        "dataset": "datasets/a.npz",
        "beyondmimic": "beyondmimic/a.csv",
    }


def test_manifest_replaces_same_motion_source(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    first = tmp_path / "first" / "walk.npz"
    second = tmp_path / "second" / "walk.npz"
    paths = export_paths("dr02", first, repository / "retarget_data")

    update_motion_manifest(paths, "dr02", first, repository_root=repository)
    update_motion_manifest(paths, "dr02", second, repository_root=repository)

    manifest = json.loads(paths.manifest.read_text())
    assert manifest["motions"]["motions/walk.pkl"]["source"] == {
        "path": second.resolve().as_posix(),
        "base": "absolute",
    }


@pytest.mark.parametrize(
    "header",
    [
        {"format_version": 2, "robot": "dr02", "motions": {}},
        {"format_version": 1, "robot": "unitree_g1", "motions": {}},
    ],
)
def test_manifest_rejects_incompatible_header(tmp_path, header):
    repository = tmp_path / "repository"
    source = repository / "walk.npz"
    paths = export_paths("dr02", source, repository / "retarget_data")
    paths.manifest.parent.mkdir(parents=True)
    paths.manifest.write_text(json.dumps(header))

    with pytest.raises(ValueError, match="manifest"):
        update_motion_manifest(paths, "dr02", source, repository_root=repository)


def test_load_motion_manifest_resolves_paths(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "sources" / "walk.npz"
    source.parent.mkdir(parents=True)
    source.touch()
    paths = export_paths("dr02", source, repository / "retarget_data")
    paths.motion.parent.mkdir(parents=True)
    paths.motion.touch()
    update_motion_manifest(paths, "dr02", source, repository_root=repository)

    entry = load_motion_manifest(paths.motion, repository_root=repository)

    assert entry.robot == "dr02"
    assert entry.reference == source.resolve()
    assert entry.dataset == paths.dataset.resolve()
    assert entry.beyondmimic == paths.csv.resolve()


def test_load_motion_manifest_can_skip_stale_reference(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "missing.npz"
    paths = export_paths("dr02", source, repository / "retarget_data")
    paths.motion.parent.mkdir(parents=True)
    paths.motion.touch()
    update_motion_manifest(paths, "dr02", source, repository_root=repository)

    entry = load_motion_manifest(
        paths.motion, repository_root=repository, require_reference=False
    )

    assert entry.reference is None


def test_load_motion_manifest_rejects_missing_exact_entry(tmp_path):
    repository = tmp_path / "repository"
    source = repository / "walk.npz"
    repository.mkdir()
    source.touch()
    paths = export_paths("dr02", source, repository / "retarget_data")
    paths.motion.parent.mkdir(parents=True)
    other_motion = paths.motion.with_name("other.pkl")
    other_motion.touch()
    update_motion_manifest(paths, "dr02", source, repository_root=repository)

    with pytest.raises(ValueError, match="motion entry"):
        load_motion_manifest(other_motion, repository_root=repository)


def test_load_motion_manifest_rejects_unsupported_source_base(tmp_path):
    repository = tmp_path / "repository"
    motion = repository / "retarget_data" / "dr02" / "motions" / "walk.pkl"
    motion.parent.mkdir(parents=True)
    motion.touch()
    manifest = motion.parent.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "format_version": 1,
                "robot": "dr02",
                "motions": {
                    "motions/walk.pkl": {
                        "source": {"path": "walk.npz", "base": "remote"},
                        "dataset": "datasets/walk.npz",
                        "beyondmimic": "beyondmimic/walk.csv",
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="source base"):
        load_motion_manifest(motion, repository_root=repository)


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
    source = tmp_path / "sources" / "walk.npz"
    source.parent.mkdir()
    source.touch()

    paths = export_retarget_motion(
        model,
        "dr02",
        source,
        10.0,
        two_joint_qpos(),
        output_root=tmp_path / "retarget_data",
        repository_root=tmp_path,
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

    entry = load_motion_manifest(paths.motion, repository_root=tmp_path)
    assert paths.manifest == tmp_path / "retarget_data" / "dr02" / "manifest.json"
    assert entry.reference == source.resolve()


def test_export_does_not_publish_manifest_when_artifact_write_fails(
    tmp_path, monkeypatch
):
    model = model_from_joints("<joint name='hip' type='hinge'/>")
    source = tmp_path / "walk.npz"
    source.touch()
    expected = export_paths("dr02", source, tmp_path / "retarget_data")
    qpos = two_joint_qpos()[:, :8]

    def fail_savez(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", fail_savez)
    with pytest.raises(OSError, match="disk full"):
        export_retarget_motion(
            model,
            "dr02",
            source,
            30.0,
            qpos,
            output_root=tmp_path / "retarget_data",
            repository_root=tmp_path,
        )

    assert not expected.manifest.exists()


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
