import json

import mujoco
import pytest

from general_motion_retargeting.retarget_export import (
    ensure_joint_contract,
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
