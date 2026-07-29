from __future__ import annotations

import pathlib
from dataclasses import dataclass

import mujoco
import numpy as np
import rerun as rr

from general_motion_retargeting import load_robot_motion


SUPPORTED_MOTION_SUFFIXES = {".pkl", ".npz"}


@dataclass(frozen=True)
class MotionSpec:
    path: pathlib.Path
    name: str


@dataclass(frozen=True)
class MotionFrames:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    dof_pos: np.ndarray


def discover_motions(input_path: str | pathlib.Path) -> list[MotionSpec]:
    path = pathlib.Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_MOTION_SUFFIXES:
            raise ValueError(f"unsupported motion file: {path}")
        return [MotionSpec(path, path.stem)]
    if not path.is_dir():
        raise FileNotFoundError(path)

    files = sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in SUPPORTED_MOTION_SUFFIXES
        ),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    if not files:
        raise ValueError(f"no .pkl or .npz motion files in: {path}")
    return [
        MotionSpec(candidate, candidate.relative_to(path).with_suffix("").as_posix())
        for candidate in files
    ]


def _load_npz_motion(path: pathlib.Path):
    with np.load(path, allow_pickle=True) as payload:
        fps = payload["fps"].item()
        root_pos = payload["root_pos"]
        root_rot = payload["root_rot"][:, [3, 0, 1, 2]]
        dof_pos = payload["dof_pos"]
    return {}, fps, root_pos, root_rot, dof_pos, None, None


def load_and_validate_motion(
    spec: MotionSpec, model: mujoco.MjModel
) -> MotionFrames:
    if spec.path.suffix.lower() == ".npz":
        payload = _load_npz_motion(spec.path)
    else:
        payload = load_robot_motion(spec.path)

    _, fps, root_pos, root_rot, dof_pos, _, _ = payload
    fps = float(fps)
    root_pos = np.asarray(root_pos, dtype=float)
    root_rot = np.asarray(root_rot, dtype=float)
    dof_pos = np.asarray(dof_pos, dtype=float)
    prefix = f"{spec.path}:"

    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{prefix} expected positive finite fps, got {fps}")
    if root_pos.ndim != 2 or root_pos.shape[1:] != (3,):
        raise ValueError(f"{prefix} root_pos must have shape (frames, 3), got {root_pos.shape}")
    if root_rot.ndim != 2 or root_rot.shape[1:] != (4,):
        raise ValueError(f"{prefix} root_rot must have shape (frames, 4), got {root_rot.shape}")
    if dof_pos.ndim != 2:
        raise ValueError(f"{prefix} dof_pos must have shape (frames, dofs), got {dof_pos.shape}")

    frame_count = root_pos.shape[0]
    if frame_count == 0:
        raise ValueError(f"{prefix} expected at least one frame")
    if root_rot.shape[0] != frame_count or dof_pos.shape[0] != frame_count:
        raise ValueError(
            f"{prefix} frame counts differ: root_pos={frame_count}, "
            f"root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
        )
    if not all(np.isfinite(values).all() for values in (root_pos, root_rot, dof_pos)):
        raise ValueError(f"{prefix} motion arrays must contain only finite values")

    if model.nq < 7 or model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"{prefix} MJCF must start with a free root joint")
    expected_dofs = model.nq - 7
    if dof_pos.shape[1] != expected_dofs:
        raise ValueError(
            f"{prefix} expected {expected_dofs} DoF values from MJCF nq={model.nq}, "
            f"got {dof_pos.shape[1]}"
        )

    return MotionFrames(
        fps=fps,
        root_pos=root_pos,
        root_rot=root_rot,
        dof_pos=dof_pos,
    )


def load_and_validate_urdf(
    urdf_path: str | pathlib.Path,
    model: mujoco.MjModel,
    *,
    entity_path_prefix: str,
    frame_prefix: str,
):
    path = pathlib.Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    tree = rr.urdf.UrdfTree.from_file_path(
        path,
        entity_path_prefix=entity_path_prefix,
        frame_prefix=frame_prefix,
    )
    movable_joints = [joint for joint in tree.joints() if joint.joint_type != "fixed"]
    mjcf_joint_names = [model.joint(index).name for index in range(1, model.njnt)]
    urdf_joint_names = [joint.name for joint in movable_joints]
    if urdf_joint_names != mjcf_joint_names:
        raise ValueError(
            "URDF movable joints do not match MJCF joint order: "
            f"URDF={urdf_joint_names}, MJCF={mjcf_joint_names}"
        )
    return tree, movable_joints


def log_motion(
    recording: rr.RecordingStream,
    model: mujoco.MjModel,
    urdf_path: str | pathlib.Path,
    spec: MotionSpec,
    frames: MotionFrames,
) -> None:
    root_path = f"motions/{spec.name}/robot"
    frame_prefix = f"{spec.name}/"
    tree, movable_joints = load_and_validate_urdf(
        urdf_path,
        model,
        entity_path_prefix=root_path,
        frame_prefix=frame_prefix,
    )
    recording.log(
        f"motions/{spec.name}",
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )
    tree.log_urdf_to_recording(recording)

    frame_timeline = f"{spec.name}/frame"
    time_timeline = f"{spec.name}/time"
    transforms_path = f"motions/{spec.name}/transforms"
    world_frame = f"{frame_prefix}world"
    root_frame = f"{frame_prefix}{tree.root_link().name}"
    for frame_index in range(frames.root_pos.shape[0]):
        recording.set_time(frame_timeline, sequence=frame_index)
        recording.set_time(
            time_timeline,
            duration=frame_index / frames.fps,
        )
        recording.log(
            transforms_path,
            rr.Transform3D(
                translation=frames.root_pos[frame_index],
                quaternion=frames.root_rot[frame_index][[1, 2, 3, 0]],
                parent_frame=world_frame,
                child_frame=root_frame,
            ),
        )
        for joint, value in zip(
            movable_joints, frames.dof_pos[frame_index], strict=True
        ):
            recording.log(
                transforms_path,
                joint.compute_transform(float(value), clamp=False),
            )

    recording.disable_timeline(frame_timeline)
    recording.disable_timeline(time_timeline)
    print(f"Recorded {spec.name}: {frames.root_pos.shape[0]} frames")


def record_motions(
    recording: rr.RecordingStream,
    model: mujoco.MjModel,
    urdf_path: str | pathlib.Path,
    specs: list[MotionSpec],
) -> None:
    for spec in specs:
        frames = load_and_validate_motion(spec, model)
        log_motion(recording, model, urdf_path, spec, frames)
