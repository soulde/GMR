import argparse
from pathlib import Path

import numpy as np


REQUIRED = [
    "root_pos",
    "root_quat",
    "root_lin_vel",
    "root_ang_vel",
    "joint_pos",
    "joint_vel",
    "left_foot_pos",
    "right_foot_pos",
    "left_foot_vel",
    "right_foot_vel",
    "left_foot_contact",
    "right_foot_contact",
    "phase",
    "vx_ref",
    "vy_ref",
    "yaw_rate_ref",
    "fps",
    "joint_names",
]


def status_line(level, message):
    print(f"{level}: {message}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    data = dict(np.load(args.dataset, allow_pickle=True))
    errors = 0
    warnings = 0

    for key in REQUIRED:
        if key not in data:
            status_line("ERROR", f"missing field {key}")
            errors += 1

    if errors:
        status_line("ERROR", "dataset is missing required fields")
        raise SystemExit(1)

    T = data["root_pos"].shape[0]
    shape_checks = {
        "root_pos": (T, 3),
        "root_quat": (T, 4),
        "root_lin_vel": (T, 3),
        "root_ang_vel": (T, 3),
        "joint_pos": (T, 29),
        "joint_vel": (T, 29),
        "left_foot_pos": (T, 3),
        "right_foot_pos": (T, 3),
        "left_foot_vel": (T, 3),
        "right_foot_vel": (T, 3),
        "left_foot_contact": (T,),
        "right_foot_contact": (T,),
        "phase": (T,),
        "vx_ref": (T,),
        "vy_ref": (T,),
        "yaw_rate_ref": (T,),
    }

    for key, expected in shape_checks.items():
        if data[key].shape != expected:
            status_line("ERROR", f"{key} shape {data[key].shape}, expected {expected}")
            errors += 1
        if not np.all(np.isfinite(data[key])):
            status_line("ERROR", f"{key} contains NaN or Inf")
            errors += 1

    quat_norm = np.linalg.norm(data["root_quat"], axis=1)
    if np.max(np.abs(quat_norm - 1.0)) > 1e-3:
        status_line("ERROR", "root_quat is not unit length")
        errors += 1

    for key in ["left_foot_contact", "right_foot_contact"]:
        values = np.unique(data[key])
        if not np.all(np.isin(values, [0, 1, False, True])):
            status_line("ERROR", f"{key} is not bool/0/1")
            errors += 1

    if np.min(data["phase"]) < -1e-8 or np.max(data["phase"]) > 1.0 + 1e-8:
        status_line("ERROR", "phase is outside [0, 1]")
        errors += 1

    if data["joint_names"].shape[0] != 29:
        status_line("ERROR", f"joint_names length {data['joint_names'].shape[0]}, expected 29")
        errors += 1

    if np.max(np.abs(data["vx_ref"])) > 5.0:
        status_line("WARNING", "vx_ref exceeds 5 m/s")
        warnings += 1
    if np.max(np.abs(data["yaw_rate_ref"])) > 10.0:
        status_line("WARNING", "yaw_rate_ref exceeds 10 rad/s")
        warnings += 1

    if errors:
        status_line("ERROR", f"dataset check failed with {errors} errors and {warnings} warnings")
        raise SystemExit(1)
    if warnings:
        status_line("WARNING", f"dataset check passed with {warnings} warnings")
    else:
        status_line("PASS", f"dataset check passed for {T} frames")


if __name__ == "__main__":
    main()
