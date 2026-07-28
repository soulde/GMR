from .motion_retarget import GeneralMotionRetargeting
from .quadruped.retarget import QuadrupedRobotRetargeter


def create_retargeter(model_type: str = "humanoid", **kwargs):
    if model_type == "humanoid":
        return GeneralMotionRetargeting(**kwargs)
    if model_type == "quadruped":
        return QuadrupedRobotRetargeter(**kwargs)
    raise ValueError(
        "model_type must be 'humanoid' or 'quadruped', "
        f"got {model_type!r}"
    )
