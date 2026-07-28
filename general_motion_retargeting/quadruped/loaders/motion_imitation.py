import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..types import JointSpaceMotion


def load_motion_imitation(
    path: str | Path,
    joint_names: tuple[str, ...],
    quaternion_order: str,
    root_frame_rotation_wxyz: tuple[float, float, float, float] | None = None,
) -> JointSpaceMotion:
    with Path(path).open() as stream:
        payload = json.load(stream)

    duration = float(payload["FrameDuration"])
    if duration <= 0:
        raise ValueError("FrameDuration must be positive")

    frames = np.asarray(payload["Frames"], dtype=float)
    expected = 3 + 4 + len(joint_names)
    if frames.ndim != 2 or frames.shape[1] != expected:
        raise ValueError(
            f"motion frames expected {expected} values, got shape {frames.shape}"
        )

    root_rot = frames[:, 3:7].copy()
    if quaternion_order == "xyzw":
        root_rot = root_rot[:, [3, 0, 1, 2]]
    elif quaternion_order != "wxyz":
        raise ValueError(f"unsupported quaternion order: {quaternion_order}")

    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("root quaternion has zero norm")
    root_rot = root_rot / norms

    if root_frame_rotation_wxyz is not None:
        rotations = Rotation.from_quat(root_rot, scalar_first=True)
        root_frame_rotation = Rotation.from_quat(
            root_frame_rotation_wxyz, scalar_first=True
        )
        root_rot = (rotations * root_frame_rotation.inv()).as_quat(
            scalar_first=True
        )

    return JointSpaceMotion(
        fps=1.0 / duration,
        root_pos=frames[:, :3].copy(),
        root_rot=root_rot,
        joint_pos=frames[:, 7:].copy(),
        joint_names=joint_names,
        loop_mode=str(payload["LoopMode"]),
    )
