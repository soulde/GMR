import argparse
from pathlib import Path

import mujoco
import numpy as np

from general_motion_retargeting.utils.robot_motion import (
    ensure_dir,
    estimate_contacts,
    joint_limit_report,
    load_motion,
    plot_series,
    write_array_csv,
    write_csv,
    compute_kinematic_metrics,
)

DR02_FOOT_SITES = {"left_foot": "left_foot", "right_foot": "right_foot"}


def contact_summary(left_contact, right_contact):
    left = left_contact.astype(bool)
    right = right_contact.astype(bool)
    double = left & right
    single = left ^ right
    flight = ~(left | right)
    return {
        "left_contact_ratio": float(left.mean()),
        "right_contact_ratio": float(right.mean()),
        "double_support_ratio": float(double.mean()),
        "single_support_ratio": float(single.mean()),
        "flight_ratio": float(flight.mean()),
    }


def mean_or_nan(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float("nan")


def max_or_nan(values):
    values = np.asarray(values)
    return float(values.max()) if values.size else float("nan")


def write_summary(path, metrics, joint_rows, left_contact, right_contact, height_threshold, xy_threshold):
    issues = []
    max_violation = max((row["violation_ratio"] for row in joint_rows), default=0.0)
    warn_joints = [row for row in joint_rows if row["violation_ratio"] > 0.01]
    error_joints = [row for row in joint_rows if row["violation_ratio"] > 0.10]

    contacts = contact_summary(left_contact, right_contact)
    left_support_xy = metrics["left_foot_xy_speed"][left_contact]
    right_support_xy = metrics["right_foot_xy_speed"][right_contact]
    left_support_h = metrics["left_foot_height"][left_contact]
    right_support_h = metrics["right_foot_height"][right_contact]

    left_max_contact_h = max_or_nan(left_support_h)
    right_max_contact_h = max_or_nan(right_support_h)
    left_min_h = float(metrics["left_foot_height"].min())
    right_min_h = float(metrics["right_foot_height"].min())
    left_support_mean_xy = mean_or_nan(left_support_xy)
    right_support_mean_xy = mean_or_nan(right_support_xy)
    left_support_max_xy = max_or_nan(left_support_xy)
    right_support_max_xy = max_or_nan(right_support_xy)

    if contacts["flight_ratio"] > 0.25:
        issues.append("high flight ratio")
    if left_min_h < -0.03 or right_min_h < -0.03:
        issues.append("foot penetration")
    if left_max_contact_h > 0.06 or right_max_contact_h > 0.06:
        issues.append("foot floating during estimated contact")
    if left_support_max_xy > xy_threshold or right_support_max_xy > xy_threshold:
        issues.append("support foot sliding")
    if max_violation > 0.0:
        issues.append("joint limit violation")

    lines = [
        "DR02 motion quality summary",
        f"frames: {len(metrics['time'])}",
        f"fps: {metrics['fps']:.6f}",
        f"duration: {metrics['time'][-1]:.3f}s",
        "",
        "Contact thresholds",
        f"contact_height_threshold: {height_threshold}",
        f"contact_xy_speed_threshold: {xy_threshold}",
        "",
        "Contact ratios",
    ]
    lines.extend(f"{k}: {v:.6f}" for k, v in contacts.items())
    lines.extend(
        [
            "",
            "Foot diagnostics",
            f"left_support_mean_xy_vel: {left_support_mean_xy:.6f}",
            f"right_support_mean_xy_vel: {right_support_mean_xy:.6f}",
            f"left_support_max_xy_vel: {left_support_max_xy:.6f}",
            f"right_support_max_xy_vel: {right_support_max_xy:.6f}",
            f"left_foot_min_height: {left_min_h:.6f}",
            f"right_foot_min_height: {right_min_h:.6f}",
            f"left_foot_max_contact_height: {left_max_contact_h:.6f}",
            f"right_foot_max_contact_height: {right_max_contact_h:.6f}",
            "",
            "Base diagnostics",
            f"base_height_min: {metrics['base_height'].min():.6f}",
            f"base_height_max: {metrics['base_height'].max():.6f}",
            f"base_roll_abs_max_rad: {np.max(np.abs(metrics['base_roll'])):.6f}",
            f"base_pitch_abs_max_rad: {np.max(np.abs(metrics['base_pitch'])):.6f}",
            "",
            "Joint limit diagnostics",
            f"max_violation_ratio: {max_violation:.6f}",
            f"warning_joints_gt_1pct: {len(warn_joints)}",
            f"error_joints_gt_10pct: {len(error_joints)}",
        ]
    )
    for row in error_joints:
        lines.append(f"ERROR joint_limit {row['joint_name']}: {row['violation_ratio']:.4f}")
    for row in warn_joints:
        if row not in error_joints:
            lines.append(f"WARNING joint_limit {row['joint_name']}: {row['violation_ratio']:.4f}")

    if issues:
        lines.extend(["", "Possible issue:"])
        for issue in issues:
            lines.append(f"- {issue}")
        if "foot penetration" in issues or "foot floating during estimated contact" in issues:
            lines.append("- wrong ground_height")
            lines.append("- wrong foot z offset")
    else:
        lines.extend(["", "No severe quality issue detected by first-pass checks."])

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/dr02.xml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact-height-threshold", type=float, default=0.04)
    parser.add_argument("--contact-xy-speed-threshold", type=float, default=0.20)
    args = parser.parse_args()

    out = ensure_dir(args.out)
    plots = ensure_dir(out / "plots")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    motion = load_motion(args.motion)
    metrics = compute_kinematic_metrics(model, motion, DR02_FOOT_SITES)
    contacts = estimate_contacts(
        metrics,
        DR02_FOOT_SITES,
        args.contact_height_threshold,
        args.contact_xy_speed_threshold,
    )
    left_contact = contacts["left_foot"]
    right_contact = contacts["right_foot"]
    joint_rows = joint_limit_report(model, metrics["joint_pos"])

    T = len(metrics["time"])
    rows = []
    for i in range(T):
        rows.append(
            {
                "frame": i,
                "time": metrics["time"][i],
                "base_height": metrics["base_height"][i],
                "base_roll": metrics["base_roll"][i],
                "base_pitch": metrics["base_pitch"][i],
                "left_foot_height": metrics["left_foot_height"][i],
                "right_foot_height": metrics["right_foot_height"][i],
                "left_foot_xy_speed": metrics["left_foot_xy_speed"][i],
                "right_foot_xy_speed": metrics["right_foot_xy_speed"][i],
                "left_contact": int(left_contact[i]),
                "right_contact": int(right_contact[i]),
            }
        )
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "joint_range.csv", joint_rows)
    write_csv(
        out / "contact_labels.csv",
        [
            {
                "frame": i,
                "time": metrics["time"][i],
                "left_contact": int(left_contact[i]),
                "right_contact": int(right_contact[i]),
                "left_foot_height": metrics["left_foot_height"][i],
                "right_foot_height": metrics["right_foot_height"][i],
                "left_foot_xy_speed": metrics["left_foot_xy_speed"][i],
                "right_foot_xy_speed": metrics["right_foot_xy_speed"][i],
            }
            for i in range(T)
        ],
    )
    write_array_csv(
        out / "foot_height.csv",
        ["frame", "time", "left_foot_height", "right_foot_height"],
        np.column_stack([np.arange(T), metrics["time"], metrics["left_foot_height"], metrics["right_foot_height"]]),
    )
    write_array_csv(
        out / "foot_xy_velocity.csv",
        ["frame", "time", "left_foot_xy_speed", "right_foot_xy_speed"],
        np.column_stack([np.arange(T), metrics["time"], metrics["left_foot_xy_speed"], metrics["right_foot_xy_speed"]]),
    )

    margins = np.asarray(
        [min(row["min_margin"], row["max_margin"]) for row in joint_rows], dtype=float
    )
    plot_series(plots / "base_height.png", metrics["time"], [metrics["base_height"]], ["base_height"], "Base height", "m")
    plot_series(
        plots / "base_rpy.png",
        metrics["time"],
        [metrics["base_roll"], metrics["base_pitch"], metrics["base_yaw"]],
        ["roll", "pitch", "yaw"],
        "Base RPY",
        "rad",
    )
    plot_series(plots / "left_foot_height.png", metrics["time"], [metrics["left_foot_height"]], ["left"], "Left foot height", "m")
    plot_series(plots / "right_foot_height.png", metrics["time"], [metrics["right_foot_height"]], ["right"], "Right foot height", "m")
    plot_series(plots / "left_foot_xy_vel.png", metrics["time"], [metrics["left_foot_xy_speed"]], ["left"], "Left foot XY speed", "m/s")
    plot_series(plots / "right_foot_xy_vel.png", metrics["time"], [metrics["right_foot_xy_speed"]], ["right"], "Right foot XY speed", "m/s")
    plot_series(plots / "joint_limit_margin.png", np.arange(len(margins)), [margins], ["min_margin"], "Joint limit margin", "rad")

    write_summary(
        out / "summary.txt",
        metrics,
        joint_rows,
        left_contact,
        right_contact,
        args.contact_height_threshold,
        args.contact_xy_speed_threshold,
    )
    print(f"Wrote DR02 motion quality report to {out}")


if __name__ == "__main__":
    main()
