from pathlib import Path

import mujoco


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
