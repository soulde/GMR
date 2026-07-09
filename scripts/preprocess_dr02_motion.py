import argparse
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.dr02.motion_tools import (
    compute_dataset_fields,
    compute_kinematic_metrics,
    ensure_dir,
    finite_difference,
    load_motion,
    normalize_quat,
    quat_slerp,
    save_motion_pkl,
)


def smoothstep(alpha):
    return 3.0 * alpha**2 - 2.0 * alpha**3


def find_stable_start_frame(metrics):
    left_h = metrics["left_foot_height"]
    right_h = metrics["right_foot_height"]
    joint_speed = np.linalg.norm(metrics["joint_vel"], axis=1)
    base_height = metrics["base_height"]
    mean_height = np.median(base_height)
    score = (
        5.0 * np.abs(left_h - right_h)
        + 2.0 * np.abs(left_h)
        + 2.0 * np.abs(right_h)
        + 2.0 * np.abs(base_height - mean_height)
        + 1.0 * np.abs(metrics["base_roll"])
        + 1.0 * np.abs(metrics["base_pitch"])
        + 0.05 * joint_speed
    )
    return int(np.argmin(score))


def build_preprocessed_motion(model, motion, stand_time, blend_time, fps, start_frame=None, auto_start_frame=False):
    source = {
        "fps": motion["fps"],
        "root_pos": motion["root_pos"],
        "root_quat": motion["root_quat"],
        "joint_pos": motion["joint_pos"],
    }

    if auto_start_frame:
        metrics = compute_kinematic_metrics(model, source)
        start_frame = find_stable_start_frame(metrics)
    if start_frame is None:
        start_frame = 0

    root_pos = source["root_pos"][start_frame:]
    root_quat = source["root_quat"][start_frame:]
    joint_pos = source["joint_pos"][start_frame:]

    stand_frames = max(0, int(round(stand_time * fps)))
    blend_frames = max(0, int(round(blend_time * fps)))

    q_stand = np.zeros(model.nq - 7, dtype=float)
    root_stand = root_pos[0].copy()
    root_quat_stand = root_quat[0].copy()

    stand_root_pos = np.repeat(root_stand[None, :], stand_frames, axis=0)
    stand_root_quat = np.repeat(root_quat_stand[None, :], stand_frames, axis=0)
    stand_joint_pos = np.repeat(q_stand[None, :], stand_frames, axis=0)

    blend_root_pos = []
    blend_root_quat = []
    blend_joint_pos = []
    for i in range(blend_frames):
        alpha = (i + 1) / max(1, blend_frames)
        s = smoothstep(alpha)
        blend_root_pos.append((1.0 - s) * root_stand + s * root_pos[0])
        blend_root_quat.append(quat_slerp(root_quat_stand, root_quat[0], [s])[0])
        blend_joint_pos.append((1.0 - s) * q_stand + s * joint_pos[0])

    pieces_root_pos = [arr for arr in [stand_root_pos, np.asarray(blend_root_pos), root_pos] if len(arr)]
    pieces_root_quat = [arr for arr in [stand_root_quat, np.asarray(blend_root_quat), root_quat] if len(arr)]
    pieces_joint_pos = [arr for arr in [stand_joint_pos, np.asarray(blend_joint_pos), joint_pos] if len(arr)]

    out = {
        "fps": float(fps),
        "root_pos": np.vstack(pieces_root_pos),
        "root_quat": normalize_quat(np.vstack(pieces_root_quat)),
        "joint_pos": np.vstack(pieces_joint_pos),
        "recommended_start_frame": start_frame,
        "recommended_start_time": start_frame / motion["fps"],
    }
    out["joint_vel"] = finite_difference(out["joint_pos"], fps)
    out["root_lin_vel"] = finite_difference(out["root_pos"], fps)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/dr02.xml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stand-time", type=float, default=1.0)
    parser.add_argument("--blend-time", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--auto-start-frame", action="store_true")
    parser.add_argument("--export-dataset", action="store_true")
    parser.add_argument("--cycle-length", type=int, default=None)
    parser.add_argument("--contact-height-threshold", type=float, default=0.04)
    parser.add_argument("--contact-xy-speed-threshold", type=float, default=0.20)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    motion = load_motion(args.motion)
    fps = float(args.fps if args.fps is not None else motion["fps"])
    processed = build_preprocessed_motion(
        model,
        motion,
        args.stand_time,
        args.blend_time,
        fps,
        start_frame=args.start_frame,
        auto_start_frame=args.auto_start_frame,
    )

    dataset = compute_dataset_fields(
        model,
        processed,
        cycle_length=args.cycle_length,
        contact_height=args.contact_height_threshold,
        contact_xy_speed=args.contact_xy_speed_threshold,
    )
    processed.update(
        {
            "root_ang_vel": dataset["root_ang_vel"],
            "left_foot_contact": dataset["left_foot_contact"],
            "right_foot_contact": dataset["right_foot_contact"],
        }
    )

    ensure_dir(args.out.parent)
    if args.export_dataset or args.out.suffix == ".npz":
        np.savez(args.out, **dataset)
        print(f"Wrote DR02 motion dataset to {args.out}")
    else:
        save_motion_pkl(args.out, processed)
        print(
            f"Wrote preprocessed DR02 motion to {args.out}; "
            f"recommended_start_frame={processed['recommended_start_frame']}"
        )


if __name__ == "__main__":
    main()
