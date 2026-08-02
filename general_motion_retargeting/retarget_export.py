from __future__ import annotations

import json
import pathlib
import pickle
import tempfile
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExportPaths:
    joints: pathlib.Path
    manifest: pathlib.Path
    motion: pathlib.Path
    dataset: pathlib.Path
    csv: pathlib.Path
    bodies: pathlib.Path | None = None


@dataclass(frozen=True)
class ManifestEntry:
    robot: str
    reference: pathlib.Path | None
    dataset: pathlib.Path
    beyondmimic: pathlib.Path


def export_paths(
    robot: str,
    source_path: str | pathlib.Path,
    output_root: str | pathlib.Path = pathlib.Path("retarget_data"),
) -> ExportPaths:
    base = pathlib.Path(output_root) / robot
    stem = pathlib.Path(source_path).stem
    return ExportPaths(
        joints=base / "joints.json",
        manifest=base / "manifest.json",
        motion=base / "motions" / f"{stem}.pkl",
        dataset=base / "datasets" / f"{stem}.npz",
        csv=base / "beyondmimic" / f"{stem}.csv",
        bodies=base / "bodies.json",
    )


def encode_source_path(
    source_path: str | pathlib.Path,
    repository_root: str | pathlib.Path,
    *,
    cwd: str | pathlib.Path | None = None,
) -> dict[str, str]:
    source = pathlib.Path(source_path)
    if not source.is_absolute():
        source = pathlib.Path(cwd or pathlib.Path.cwd()) / source
    source = source.resolve()
    repository = pathlib.Path(repository_root).resolve()
    try:
        relative = source.relative_to(repository)
    except ValueError:
        return {"path": source.as_posix(), "base": "absolute"}
    return {"path": relative.as_posix(), "base": "repository"}


def _write_json_atomic(path: pathlib.Path, payload: dict) -> None:
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
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def update_motion_manifest(
    paths: ExportPaths,
    robot: str,
    source_path: str | pathlib.Path,
    *,
    repository_root: str | pathlib.Path,
    cwd: str | pathlib.Path | None = None,
) -> pathlib.Path:
    if paths.manifest.exists():
        try:
            with paths.manifest.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid motion manifest: {paths.manifest}") from error
        if manifest.get("format_version") != 1 or manifest.get("robot") != robot:
            raise ValueError(
                f"existing motion manifest does not match robot/version: {paths.manifest}"
            )
        motions = manifest.get("motions")
        if not isinstance(motions, dict):
            raise ValueError(f"invalid motions mapping in manifest: {paths.manifest}")
    else:
        motions = {}

    robot_root = paths.manifest.parent
    motion_key = paths.motion.relative_to(robot_root).as_posix()
    motions[motion_key] = {
        "source": encode_source_path(
            source_path, repository_root, cwd=cwd
        ),
        "dataset": paths.dataset.relative_to(robot_root).as_posix(),
        "beyondmimic": paths.csv.relative_to(robot_root).as_posix(),
    }
    payload = {
        "format_version": 1,
        "robot": robot,
        "motions": dict(sorted(motions.items())),
    }
    _write_json_atomic(paths.manifest, payload)
    return paths.manifest


def load_motion_manifest(
    motion_path: str | pathlib.Path,
    *,
    repository_root: str | pathlib.Path,
    require_reference: bool = True,
) -> ManifestEntry:
    motion = pathlib.Path(motion_path).resolve()
    robot_root = motion.parent.parent
    manifest_path = robot_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"motion manifest not found: {manifest_path}")
    try:
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid motion manifest: {manifest_path}") from error

    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported motion manifest version: {manifest_path}")
    robot = manifest.get("robot")
    motions = manifest.get("motions")
    if not isinstance(robot, str) or not robot or not isinstance(motions, dict):
        raise ValueError(f"invalid motion manifest schema: {manifest_path}")
    try:
        motion_key = motion.relative_to(robot_root).as_posix()
    except ValueError as error:
        raise ValueError(f"motion is outside manifest robot directory: {motion}") from error
    entry = motions.get(motion_key)
    if not isinstance(entry, dict):
        raise ValueError(f"motion entry not found in manifest: {motion_key}")

    source = entry.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"invalid source entry for motion: {motion_key}")
    source_path = source.get("path")
    source_base = source.get("base")
    if not isinstance(source_path, str) or source_base not in {
        "repository",
        "absolute",
    }:
        raise ValueError(f"unsupported source base for motion: {motion_key}")
    if source_base == "repository":
        resolved_source = pathlib.Path(repository_root).resolve() / source_path
    else:
        resolved_source = pathlib.Path(source_path)
        if not resolved_source.is_absolute():
            raise ValueError(f"absolute source path is not absolute: {source_path}")
    resolved_source = resolved_source.resolve()
    if require_reference and not resolved_source.is_file():
        raise FileNotFoundError(f"source motion not found: {resolved_source}")

    dataset = entry.get("dataset")
    beyondmimic = entry.get("beyondmimic")
    if not isinstance(dataset, str) or not isinstance(beyondmimic, str):
        raise ValueError(f"invalid artifact paths for motion: {motion_key}")
    return ManifestEntry(
        robot=robot,
        reference=resolved_source if require_reference else None,
        dataset=(robot_root / dataset).resolve(),
        beyondmimic=(robot_root / beyondmimic).resolve(),
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


def _scalar_joint_addresses(model: mujoco.MjModel) -> tuple[int, ...]:
    scalar_joint_names(model)
    addresses = [int(model.jnt_qposadr[joint_id]) for joint_id in range(1, model.njnt)]
    return tuple(sorted(addresses))


def articulation_body_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Return the root and movable-link bodies in articulation order."""
    scalar_joint_names(model)
    ordered = [(int(model.jnt_qposadr[0]), int(model.jnt_bodyid[0]))]
    ordered.extend(
        (int(model.jnt_qposadr[joint_id]), int(model.jnt_bodyid[joint_id]))
        for joint_id in range(1, model.njnt)
    )
    body_ids = []
    for _, body_id in sorted(ordered):
        if body_id not in body_ids:
            body_ids.append(body_id)

    names = []
    for body_id in body_ids:
        name = model.body(body_id).name
        if not name:
            raise ValueError("all exported articulation bodies must be named")
        names.append(name)
    return tuple(names)


def _resolve_body_ids(
    model: mujoco.MjModel,
    body_names: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    required = articulation_body_names(model)
    selected = tuple(body_names) if body_names is not None else required
    if not selected:
        raise ValueError("expected at least one exported body")
    if len(set(selected)) != len(selected):
        raise ValueError("exported body names must be unique")

    body_ids = []
    for name in selected:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"exported body does not exist in model: {name}")
        body_ids.append(int(body_id))

    root_body_id = int(model.jnt_bodyid[0])
    if body_ids[0] != root_body_id:
        raise ValueError("the free-root body must be first in the body contract")
    missing = set(required).difference(selected)
    if missing:
        raise ValueError(
            "body contract must include every movable body: "
            + ", ".join(sorted(missing))
        )
    return selected, tuple(body_ids)


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


def ensure_body_contract(
    path: str | pathlib.Path,
    robot: str,
    body_names: Sequence[str],
) -> None:
    path = pathlib.Path(path)
    contract = {
        "format_version": 1,
        "robot": robot,
        "body_names": list(body_names),
    }
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != contract:
            raise ValueError(f"existing body contract does not match {robot}: {path}")
        return
    _write_json_atomic(path, contract)


def _temporary_path(path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        return pathlib.Path(stream.name)


def _central_difference(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    return np.gradient(values, 1.0 / fps, axis=0)


def _angular_velocity(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.zeros(quat_wxyz.shape[:-1] + (3,), dtype=float)
    if len(quat_wxyz) < 2:
        return velocity
    rotations = Rotation.from_quat(quat_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4))
    matrices = rotations.as_matrix().reshape(quat_wxyz.shape[:-1] + (3, 3))
    if len(quat_wxyz) == 2:
        relative = Rotation.from_matrix(matrices[1] @ matrices[0].swapaxes(-1, -2))
        angular = relative.as_rotvec() * fps
        velocity[:] = angular
        return velocity

    relative = Rotation.from_matrix(
        matrices[2:] @ matrices[:-2].swapaxes(-1, -2)
    )
    velocity[1:-1] = relative.as_rotvec() * (0.5 * fps)
    velocity[0] = velocity[1]
    velocity[-1] = velocity[-2]
    return velocity


def _resample_qpos(
    qpos: np.ndarray,
    input_fps: float,
    output_fps: int,
) -> np.ndarray:
    if len(qpos) < 2:
        return qpos.copy()
    duration = (len(qpos) - 1) / input_fps
    output_times = np.arange(0.0, duration, 1.0 / output_fps)
    input_times = np.arange(len(qpos), dtype=float) / input_fps
    resampled = np.empty((len(output_times), qpos.shape[1]), dtype=float)
    linear_columns = np.r_[0:3, 7:qpos.shape[1]]
    for column in linear_columns:
        resampled[:, column] = np.interp(output_times, input_times, qpos[:, column])

    root_quat_xyzw = qpos[:, [4, 5, 6, 3]].copy()
    root_quat_xyzw /= np.linalg.norm(root_quat_xyzw, axis=1, keepdims=True)
    resampled_quat = Slerp(input_times, Rotation.from_quat(root_quat_xyzw))(
        output_times
    ).as_quat()
    resampled[:, 3:7] = resampled_quat[:, [3, 0, 1, 2]]
    return resampled


def _body_world_poses(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    body_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    body_index = np.asarray(body_ids, dtype=int)
    positions = np.empty((len(qpos), len(body_ids), 3), dtype=float)
    quaternions = np.empty((len(qpos), len(body_ids), 4), dtype=float)
    for frame_index, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        positions[frame_index] = data.xpos[body_index]
        quaternions[frame_index] = data.xquat[body_index]
    return positions, quaternions


def export_retarget_motion(
    model: mujoco.MjModel,
    robot: str,
    source_path: str | pathlib.Path,
    fps: float,
    qpos_frames: np.ndarray,
    *,
    output_root: str | pathlib.Path = pathlib.Path("retarget_data"),
    repository_root: str | pathlib.Path = REPOSITORY_ROOT,
    cwd: str | pathlib.Path | None = None,
    beyondmimic_fps: int = 50,
    body_names: Sequence[str] | None = None,
) -> ExportPaths:
    fps = float(fps)
    qpos = np.asarray(qpos_frames, dtype=float)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"expected positive finite fps, got {fps}")
    if qpos.ndim != 2 or qpos.shape[1:] != (model.nq,):
        raise ValueError(f"qpos must have shape (frames, {model.nq}), got {qpos.shape}")
    if qpos.shape[0] == 0:
        raise ValueError("expected at least one frame")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos frames must contain only finite values")
    if not isinstance(beyondmimic_fps, int) or beyondmimic_fps <= 0:
        raise ValueError(
            f"expected positive integer BeyondMimic fps, got {beyondmimic_fps}"
        )

    joint_names = scalar_joint_names(model)
    joint_addresses = _scalar_joint_addresses(model)
    exported_body_names, body_ids = _resolve_body_ids(model, body_names)
    root_quat_wxyz = qpos[:, 3:7].copy()
    norms = np.linalg.norm(root_quat_wxyz, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("root quaternions must have nonzero norm")
    root_quat_wxyz /= norms

    root_pos = qpos[:, :3].copy()
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
    joint_pos = qpos[:, joint_addresses].copy()
    paths = export_paths(robot, source_path, output_root)

    ensure_joint_contract(paths.joints, robot, joint_names)
    assert paths.bodies is not None
    ensure_body_contract(paths.bodies, robot, exported_body_names)

    motion_payload = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_quat_xyzw,
        "dof_pos": joint_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    temporary = _temporary_path(paths.motion)
    try:
        with temporary.open("wb") as stream:
            pickle.dump(motion_payload, stream)
        temporary.replace(paths.motion)
    finally:
        temporary.unlink(missing_ok=True)

    temporary = _temporary_path(paths.dataset)
    try:
        beyondmimic_qpos = _resample_qpos(qpos, fps, beyondmimic_fps)
        beyondmimic_joint_pos = beyondmimic_qpos[:, joint_addresses]
        body_pos_w, body_quat_w = _body_world_poses(
            model, beyondmimic_qpos, body_ids
        )
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                fps=np.asarray([beyondmimic_fps], dtype=np.int64),
                joint_pos=beyondmimic_joint_pos.astype(np.float32),
                joint_vel=_central_difference(
                    beyondmimic_joint_pos, beyondmimic_fps
                ).astype(np.float32),
                body_pos_w=body_pos_w.astype(np.float32),
                body_quat_w=body_quat_w.astype(np.float32),
                body_lin_vel_w=_central_difference(
                    body_pos_w, beyondmimic_fps
                ).astype(np.float32),
                body_ang_vel_w=_angular_velocity(
                    body_quat_w, beyondmimic_fps
                ).astype(np.float32),
            )
        temporary.replace(paths.dataset)
    finally:
        temporary.unlink(missing_ok=True)

    csv_motion = np.concatenate((root_pos, root_quat_xyzw, joint_pos), axis=1)
    if fps > 30.0:
        indices = np.arange(0, len(csv_motion), fps / 30.0).astype(int)
        csv_motion = csv_motion[indices]
    temporary = _temporary_path(paths.csv)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            np.savetxt(stream, csv_motion, delimiter=",")
        temporary.replace(paths.csv)
    finally:
        temporary.unlink(missing_ok=True)

    update_motion_manifest(
        paths,
        robot,
        source_path,
        repository_root=repository_root,
        cwd=cwd,
    )
    return paths
