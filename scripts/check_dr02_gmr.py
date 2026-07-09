import mujoco

from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_BASE_DICT, ROBOT_XML_DICT


assert "dr02" in ROBOT_XML_DICT
assert "dr02" in IK_CONFIG_DICT["smplx"]
assert "dr02" in ROBOT_BASE_DICT

model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["dr02"]))
assert model.nq == 28
assert model.nv == 27
assert model.nu == 21

print("dr02 GMR registration checks passed")
