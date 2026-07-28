import argparse
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


CONFIG_ROOT = REPO_ROOT / "general_motion_retargeting" / "quadruped" / "configs"


def _load_motion(path: Path) -> dict:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    if not isinstance(motion, dict):
        raise ValueError("motion pickle must contain a dictionary")
    return motion


def _joint_addresses(spec) -> np.ndarray:
    return np.asarray(
        [
            spec.model.jnt_qposadr[
                mujoco.mj_name2id(
                    spec.model, mujoco.mjtObj.mjOBJ_JOINT, name
                )
            ]
            for name in spec.motion_joint_order
        ],
        dtype=int,
    )


def _reconstruct_qpos(motion: dict, spec) -> tuple[np.ndarray, np.ndarray]:
    root_pos = np.asarray(motion["root_pos"], dtype=float)
    root_xyzw = np.asarray(motion["root_rot"], dtype=float)
    joint_pos = np.asarray(motion["dof_pos"], dtype=float)
    frames = len(root_pos)
    if root_pos.shape != (frames, 3):
        raise ValueError(f"root_pos must have shape ({frames}, 3)")
    if root_xyzw.shape != (frames, 4):
        raise ValueError(f"root_rot must have shape ({frames}, 4)")
    if joint_pos.shape != (frames, len(spec.motion_joint_order)):
        raise ValueError(
            "dof_pos must have shape "
            f"({frames}, {len(spec.motion_joint_order)})"
        )

    qpos = np.zeros((frames, spec.model.nq), dtype=float)
    free_joint = int(
        np.flatnonzero(spec.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
    )
    root_address = spec.model.jnt_qposadr[free_joint]
    qpos[:, root_address : root_address + 3] = root_pos
    qpos[:, root_address + 3 : root_address + 7] = root_xyzw[:, [3, 0, 1, 2]]
    joint_addresses = _joint_addresses(spec)
    qpos[:, joint_addresses] = joint_pos
    return qpos, joint_addresses


def check_motion(
    motion_path,
    robot="unitree_go2",
    velocity_limit=3.0 * np.pi,
) -> dict:
    config_path = CONFIG_ROOT / f"{robot}.yaml"
    if not config_path.is_file():
        raise ValueError(f"unknown quadruped robot {robot!r}")
    spec = load_robot_spec(config_path, REPO_ROOT)
    motion = _load_motion(Path(motion_path))
    fps = float(motion["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")

    qpos, joint_addresses = _reconstruct_qpos(motion, spec)
    finite = bool(np.isfinite(qpos).all())
    joint_pos = qpos[:, joint_addresses]

    lower = np.empty(len(joint_addresses), dtype=float)
    upper = np.empty(len(joint_addresses), dtype=float)
    for index, joint_name in enumerate(spec.motion_joint_order):
        joint_id = mujoco.mj_name2id(
            spec.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        lower[index], upper[index] = spec.model.jnt_range[joint_id]
    violations = np.maximum(
        np.maximum(lower - joint_pos, joint_pos - upper), 0.0
    )
    max_joint_limit_violation = float(np.max(violations, initial=0.0))
    lower_margin = np.min(joint_pos - lower, axis=0)
    upper_margin = np.min(upper - joint_pos, axis=0)

    joint_velocity = np.diff(joint_pos, axis=0) * fps
    velocity_violation = np.maximum(np.abs(joint_velocity) - velocity_limit, 0.0)
    max_velocity = float(np.max(np.abs(joint_velocity), initial=0.0))
    max_velocity_violation = float(np.max(velocity_violation, initial=0.0))

    data = mujoco.MjData(spec.model)
    site_ids = np.asarray(
        [
            mujoco.mj_name2id(
                spec.model,
                mujoco.mjtObj.mjOBJ_SITE,
                spec.legs[leg].foot_site,
            )
            for leg in spec.leg_order
        ],
        dtype=int,
    )
    foot_heights = np.empty((len(qpos), len(site_ids)), dtype=float)
    for frame_index, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(spec.model, data)
        foot_heights[frame_index] = data.site_xpos[site_ids, 2]

    diagnostics = motion.get("retarget_diagnostics", ())
    non_converged = [
        row
        for row in diagnostics
        if not np.isfinite(float(row["final_error"]))
        or bool(row["reached_max_iterations"])
    ]
    non_converged_ratio = (
        len(non_converged) / len(diagnostics) if diagnostics else 0.0
    )
    max_final_error = max(
        (float(row["final_error"]) for row in diagnostics),
        default=0.0,
    )

    return {
        "motion": str(Path(motion_path).resolve()),
        "robot": robot,
        "frames": len(qpos),
        "fps": fps,
        "finite": finite,
        "max_joint_limit_violation": max_joint_limit_violation,
        "max_velocity": max_velocity,
        "velocity_limit": float(velocity_limit),
        "max_velocity_violation": max_velocity_violation,
        "min_foot_height": float(foot_heights.min()),
        "max_foot_height": float(foot_heights.max()),
        "non_converged_ratio": float(non_converged_ratio),
        "max_final_error": max_final_error,
        "joint_limit_margins": {
            name: {
                "lower": float(lower_margin[index]),
                "upper": float(upper_margin[index]),
            }
            for index, name in enumerate(spec.motion_joint_order)
        },
    }


def _accepted(report: dict, max_non_converged_ratio: float) -> bool:
    return bool(
        report["finite"]
        and report["max_joint_limit_violation"] <= 1e-8
        and report["max_velocity_violation"] <= 1e-8
        and report["non_converged_ratio"] <= max_non_converged_ratio
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a saved quadruped GMR motion."
    )
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--robot", default="unitree_go2")
    parser.add_argument("--velocity_limit", type=float, default=3.0 * np.pi)
    parser.add_argument("--max_non_converged_ratio", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = check_motion(
        args.motion,
        robot=args.robot,
        velocity_limit=args.velocity_limit,
    )
    report["accepted"] = _accepted(report, args.max_non_converged_ratio)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
