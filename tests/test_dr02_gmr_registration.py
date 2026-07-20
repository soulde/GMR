def test_dr02_registered_in_gmr_params():
    from general_motion_retargeting.params import ROBOT_BASE_DICT, IK_CONFIG_DICT, ROBOT_XML_DICT

    assert "dr02" in ROBOT_XML_DICT
    assert "dr02" in IK_CONFIG_DICT["smplx"]
    assert "dr02" in ROBOT_BASE_DICT


def test_dr02_mujoco_model_loads():
    import mujoco

    from general_motion_retargeting.params import ROBOT_XML_DICT

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["dr02"]))
    assert model.nq == 36
    assert model.nv == 35
    assert model.nu == 29
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot") >= 0

    joint_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    }
    actuator_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    }
    assert actuator_joints == joint_names


def test_dr02_ik_frames_exist():
    import json
    import mujoco

    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["dr02"]))
    with open(IK_CONFIG_DICT["smplx"]["dr02"]) as stream:
        config = json.load(stream)

    for table_name in ("ik_match_table1", "ik_match_table2"):
        for frame_name in config[table_name]:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, frame_name) >= 0
