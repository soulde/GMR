from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import mujoco
import yaml


LEG_ORDER = ("FL", "FR", "RL", "RR")
SEGMENTS = ("hip", "thigh", "calf")


@dataclass(frozen=True)
class LegSpec:
    name: str
    joints: tuple[str, str, str]
    foot_site: str


@dataclass(frozen=True)
class JointMapSpec:
    sign: float
    scale: float


@dataclass(frozen=True)
class QuadrupedRobotSpec:
    robot: str
    mjcf_path: Path
    model: mujoco.MjModel
    root_body: str
    legs: Mapping[str, LegSpec]
    motion_joint_order: tuple[str, ...]
    quaternion_order: str
    reference_pose: Mapping[str, float]
    joint_mapping: Mapping[str, JointMapSpec]

    @property
    def leg_order(self) -> tuple[str, str, str, str]:
        return LEG_ORDER


def _require_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
    label: str,
) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"{label} {name!r} does not exist in MJCF")
    return object_id


def _load_legs(payload: dict) -> dict[str, LegSpec]:
    raw_legs = payload.get("legs", {})
    missing = [leg for leg in LEG_ORDER if leg not in raw_legs]
    if missing:
        raise ValueError(f"missing legs: {', '.join(missing)}")
    extra = sorted(set(raw_legs) - set(LEG_ORDER))
    if extra:
        raise ValueError(f"unknown legs: {', '.join(extra)}")

    result = {}
    assigned_joints = set()
    for leg in LEG_ORDER:
        entry = raw_legs[leg]
        joints = tuple(entry.get("joints", ()))
        if len(joints) != 3:
            raise ValueError(f"leg {leg} must declare exactly three joints")
        duplicates = assigned_joints.intersection(joints)
        if duplicates:
            raise ValueError(
                f"joints assigned to multiple legs: {', '.join(sorted(duplicates))}"
            )
        assigned_joints.update(joints)
        result[leg] = LegSpec(
            name=leg,
            joints=joints,
            foot_site=str(entry["foot_site"]),
        )
    return result


def load_robot_spec(
    path: str | Path,
    repo_root: str | Path,
) -> QuadrupedRobotSpec:
    config_path = Path(path)
    with config_path.open() as stream:
        payload = yaml.safe_load(stream)

    if payload.get("model_type") != "quadruped":
        raise ValueError("model_type must be 'quadruped'")

    legs = _load_legs(payload)
    model_path = Path(repo_root) / payload["mjcf_path"]
    if not model_path.is_file():
        raise ValueError(f"MJCF file does not exist: {model_path}")
    try:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    except ValueError as error:
        raise ValueError(f"failed to compile MJCF {model_path}: {error}") from error

    root_body = str(payload["root_body"])
    _require_name(model, mujoco.mjtObj.mjOBJ_BODY, root_body, "root body")

    for leg in legs.values():
        _require_name(model, mujoco.mjtObj.mjOBJ_SITE, leg.foot_site, "foot site")
        for joint_name in leg.joints:
            joint_id = _require_name(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name, "leg joint"
            )
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"leg joint {joint_name!r} must be a hinge")

    motion = payload["motion"]
    motion_joint_order = tuple(motion["joint_order"])
    if len(set(motion_joint_order)) != len(motion_joint_order):
        raise ValueError("motion joint_order contains duplicate names")
    for joint_name in motion_joint_order:
        _require_name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name, "motion joint"
        )

    quaternion_order = str(motion["quaternion_order"])
    if quaternion_order not in ("wxyz", "xyzw"):
        raise ValueError(
            f"unsupported quaternion order: {quaternion_order!r}"
        )

    reference_pose = {
        str(name): float(value)
        for name, value in payload["reference_pose"].items()
    }
    for joint_name, value in reference_pose.items():
        joint_id = _require_name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name, "reference joint"
        )
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            if value < lower or value > upper:
                raise ValueError(
                    f"reference position {value} for {joint_name!r} "
                    f"is outside [{lower}, {upper}]"
                )

    joint_mapping = {
        str(segment): JointMapSpec(
            sign=float(entry["sign"]),
            scale=float(entry["scale"]),
        )
        for segment, entry in payload.get("joint_mapping", {}).items()
    }
    unknown_segments = sorted(set(joint_mapping) - set(SEGMENTS))
    if unknown_segments:
        raise ValueError(
            f"unknown joint mapping segments: {', '.join(unknown_segments)}"
        )

    return QuadrupedRobotSpec(
        robot=str(payload["robot"]),
        mjcf_path=model_path.resolve(),
        model=model,
        root_body=root_body,
        legs=MappingProxyType(legs),
        motion_joint_order=motion_joint_order,
        quaternion_order=quaternion_order,
        reference_pose=MappingProxyType(reference_pose),
        joint_mapping=MappingProxyType(joint_mapping),
    )
