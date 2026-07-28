from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import mujoco
import numpy as np
import yaml


LEG_ORDER = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class LegSpec:
    name: str
    joints: tuple[str, str, str]
    foot_site: str


@dataclass(frozen=True)
class QuadrupedRobotSpec:
    robot: str
    mjcf_path: Path
    model: mujoco.MjModel
    root_body: str
    legs: Mapping[str, LegSpec]
    foot_contact_offset: float
    motion_joint_order: tuple[str, ...]
    quaternion_order: str
    root_frame_rotation_wxyz: tuple[float, float, float, float] | None

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
            qpos0 = model.qpos0[model.jnt_qposadr[joint_id]]
            if model.jnt_limited[joint_id]:
                lower, upper = model.jnt_range[joint_id]
                if qpos0 < lower or qpos0 > upper:
                    raise ValueError(
                        f"MJCF qpos0 {qpos0} for {joint_name!r} "
                        f"is outside [{lower}, {upper}]"
                    )

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
    root_frame_rotation = motion.get("root_frame_rotation")
    root_frame_rotation_wxyz = None
    if root_frame_rotation is not None:
        root_frame_rotation = np.asarray(root_frame_rotation, dtype=float)
        if root_frame_rotation.shape != (4,):
            raise ValueError("root_frame_rotation must contain four values")
        if quaternion_order == "xyzw":
            root_frame_rotation = root_frame_rotation[[3, 0, 1, 2]]
        norm = np.linalg.norm(root_frame_rotation)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("root_frame_rotation must be a finite quaternion")
        root_frame_rotation_wxyz = tuple(root_frame_rotation / norm)

    foot_contact_offset = float(payload.get("foot_contact_offset", 0.0))
    if not np.isfinite(foot_contact_offset) or foot_contact_offset < 0.0:
        raise ValueError("foot_contact_offset must be finite and non-negative")

    return QuadrupedRobotSpec(
        robot=str(payload["robot"]),
        mjcf_path=model_path.resolve(),
        model=model,
        root_body=root_body,
        legs=MappingProxyType(legs),
        foot_contact_offset=foot_contact_offset,
        motion_joint_order=motion_joint_order,
        quaternion_order=quaternion_order,
        root_frame_rotation_wxyz=root_frame_rotation_wxyz,
    )
