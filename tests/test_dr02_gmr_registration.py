def test_dr02_registered_in_gmr_params():
    from general_motion_retargeting.params import ROBOT_BASE_DICT, IK_CONFIG_DICT, ROBOT_XML_DICT

    assert "dr02" in ROBOT_XML_DICT
    assert "dr02" in IK_CONFIG_DICT["smplx"]
    assert "dr02" in ROBOT_BASE_DICT


def test_dr02_mujoco_model_loads():
    import mujoco

    from general_motion_retargeting.params import ROBOT_XML_DICT

    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["dr02"]))
    assert model.nq == 28
    assert model.nv == 27
    assert model.nu == 21
