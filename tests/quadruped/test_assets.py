from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_laikago_mjcf_has_required_semantic_names():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/laikago/laikago.xml")
    )

    for name in ("trunk", "FR_foot", "FL_foot", "RR_foot", "RL_foot"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

    for leg in ("FR", "FL", "RR", "RL"):
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"
            )
            >= 0
        )
        for segment in ("hip", "thigh", "calf"):
            assert (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{leg}_{segment}_joint",
                )
                >= 0
            )


def test_laikago_qpos0_is_nominal_and_preserves_joint_angle_fk():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/laikago/laikago.xml")
    )
    data = mujoco.MjData(model)
    joint_names = tuple(
        f"{leg}_{segment}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for segment in ("hip", "thigh", "calf")
    )
    joint_values = (
        -0.1, 0.2, -0.8,
        0.1, 0.4, -1.0,
        -0.05, 0.1, -0.7,
        0.05, 0.3, -0.9,
    )
    expected_feet = np.array(
        [
            [0.1851379858, -0.0750227565, 0.0040072156],
            [0.1537898382, 0.1521711788, 0.0500201025],
            [-0.2336856532, -0.0939645467, -0.0129773053],
            [-0.2688123386, 0.1356156414, 0.0264688020],
        ]
    )

    data.qpos[:] = model.qpos0
    data.qpos[:3] = (0.0, 0.0, 0.4)
    for name, value in zip(joint_names, joint_values, strict=True):
        joint = model.joint(name)
        assert joint.range[0] <= model.qpos0[joint.qposadr] <= joint.range[1]
        data.qpos[joint.qposadr] = value
    mujoco.mj_forward(model, data)

    actual_feet = np.array(
        [
            data.site(f"{leg}_foot_site").xpos.copy()
            for leg in ("FR", "FL", "RR", "RL")
        ]
    )
    np.testing.assert_allclose(actual_feet, expected_feet, atol=1e-9)


def test_go2_mjcf_has_free_root_twelve_hinges_and_foot_sites():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/unitree_go2/go2.xml")
    )

    joint_types = model.jnt_type.tolist()
    assert joint_types.count(mujoco.mjtJoint.mjJNT_FREE) == 1
    assert joint_types.count(mujoco.mjtJoint.mjJNT_HINGE) == 12
    for leg in ("FL", "FR", "RL", "RR"):
        assert (
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"
            )
            >= 0
        )


def test_go2_scene_has_visual_meshes_floor_and_light():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/unitree_go2/scene.xml")
    )

    assert model.nmesh > 0
    assert model.nlight > 0
    floor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    assert floor_id >= 0
    assert model.geom_type[floor_id] == mujoco.mjtGeom.mjGEOM_PLANE


def test_go2_qpos0_is_legal_and_preserves_joint_angle_fk():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/unitree_go2/go2.xml")
    )
    data = mujoco.MjData(model)
    joint_names = (
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    )
    joint_values = (
        0.1, 0.7, -1.5,
        -0.1, 1.0, -2.0,
        0.05, 1.2, -2.2,
        -0.05, 0.8, -1.7,
    )
    expected_feet = np.array(
        [
            [0.207585067, 0.172745262, 0.098351885],
            [0.192319395, -0.164669448, 0.178840660],
            [-0.213771610, 0.151574086, 0.211065862],
            [-0.180591436, -0.155993173, 0.122757791],
        ]
    )

    data.qpos[:] = model.qpos0
    data.qpos[:3] = (0.0, 0.0, 0.4)
    for name, value in zip(joint_names, joint_values, strict=True):
        joint = model.joint(name)
        assert joint.range[0] <= model.qpos0[joint.qposadr] <= joint.range[1]
        data.qpos[joint.qposadr] = value
    mujoco.mj_forward(model, data)

    actual_feet = np.array(
        [
            data.site(f"{leg}_foot_site").xpos.copy()
            for leg in ("FL", "FR", "RL", "RR")
        ]
    )
    np.testing.assert_allclose(actual_feet, expected_feet, atol=1e-8)
