from __future__ import annotations

import json
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Sequence

import mujoco


@dataclass(frozen=True)
class ExportPaths:
    joints: pathlib.Path
    motion: pathlib.Path
    dataset: pathlib.Path
    csv: pathlib.Path


def export_paths(
    robot: str,
    source_path: str | pathlib.Path,
    output_root: str | pathlib.Path = pathlib.Path("retarget_data"),
) -> ExportPaths:
    base = pathlib.Path(output_root) / robot
    stem = pathlib.Path(source_path).stem
    return ExportPaths(
        joints=base / "joints.json",
        motion=base / "motions" / f"{stem}.pkl",
        dataset=base / "datasets" / f"{stem}.npz",
        csv=base / "beyondmimic" / f"{stem}.csv",
    )


def scalar_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("GMR robot model must start with a free root joint")

    scalar_types = {
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE,
    }
    joints = []
    for joint_id in range(1, model.njnt):
        if model.jnt_type[joint_id] not in scalar_types:
            raise ValueError("GMR export supports only scalar hinge/slide joints")
        name = model.joint(joint_id).name
        if not name:
            raise ValueError("all exported robot joints must be named")
        joints.append((int(model.jnt_qposadr[joint_id]), name))
    joints.sort()
    return tuple(name for _, name in joints)


def ensure_joint_contract(
    path: str | pathlib.Path,
    robot: str,
    joint_names: Sequence[str],
) -> None:
    path = pathlib.Path(path)
    contract = {
        "format_version": 1,
        "robot": robot,
        "joint_names": list(joint_names),
    }
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != contract:
            raise ValueError(f"existing joint contract does not match {robot}: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = pathlib.Path(stream.name)
        json.dump(contract, stream, indent=2)
        stream.write("\n")
    temporary_path.replace(path)
