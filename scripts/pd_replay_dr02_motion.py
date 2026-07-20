import argparse
import importlib
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.dr02.motion_tools import (
    FOOT_SITES,
    compute_kinematic_metrics,
    ensure_dir,
    finite_difference,
    load_motion,
    model_joint_info,
    plot_series,
    write_csv,
)


def sample_reference(motion, sim_time, speed):
    fps = motion["fps"]
    ref_time = sim_time * speed
    frame = int(np.clip(np.floor(ref_time * fps), 0, len(motion["joint_pos"]) - 1))
    return frame


def contact_forces(model, data, site_ids):
    forces = {name: 0.0 for name in site_ids}
    for i in range(data.ncon):
        contact = data.contact[i]
        c_array = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(model, data, i, c_array)
        normal_force = abs(c_array[0])
        for name, site_id in site_ids.items():
            body_id = model.site_bodyid[site_id]
            if contact.geom1 in model.body_geomadr[body_id] + np.arange(model.body_geomnum[body_id]):
                forces[name] += normal_force
            if contact.geom2 in model.body_geomadr[body_id] + np.arange(model.body_geomnum[body_id]):
                forces[name] += normal_force
    return forces


def write_summary(path, rows, torque_rows, saturation_rows, fall_time, possible):
    max_tracking = max(row["joint_error_norm"] for row in rows) if rows else 0.0
    max_roll = max(abs(row["base_roll"]) for row in rows) if rows else 0.0
    max_pitch = max(abs(row["base_pitch"]) for row in rows) if rows else 0.0
    min_height = min(row["base_height"] for row in rows) if rows else 0.0
    lines = [
        "DR02 PD replay summary",
        f"sim_steps: {len(rows)}",
        f"fall_time: {fall_time if fall_time is not None else 'not_detected'}",
        f"base_height_min: {min_height:.6f}",
        f"base_roll_abs_max: {max_roll:.6f}",
        f"base_pitch_abs_max: {max_pitch:.6f}",
        f"max_joint_tracking_error_norm: {max_tracking:.6f}",
        "",
        "Top saturated joints:",
    ]
    top = sorted(saturation_rows, key=lambda x: x["saturation_ratio"], reverse=True)[:5]
    for i, row in enumerate(top, 1):
        lines.append(
            f"{i}. {row['joint_name']} saturation_ratio={row['saturation_ratio']:.4f} "
            f"max_torque_ratio={row['max_torque_ratio']:.4f}"
        )
    if possible:
        lines.extend(["", "Possible issue:"])
        lines.extend(f"- {item}" for item in possible)
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/mjcf/dr02_pos.xml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--torque-limit-scale", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--disable-viewer", action="store_true")
    args = parser.parse_args()

    out = ensure_dir(args.out)
    plots = ensure_dir(out / "plots")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    motion = load_motion(args.motion)
    info = model_joint_info(model)
    metrics_ref = compute_kinematic_metrics(model, motion)

    joint_pos = motion["joint_pos"]
    joint_vel = finite_difference(joint_pos, motion["fps"]) * args.speed
    root_pos = motion["root_pos"]
    root_quat = motion["root_quat"]

    start = int(np.clip(args.start_frame, 0, len(joint_pos) - 1))
    data.qpos[:3] = root_pos[start]
    data.qpos[3:7] = root_quat[start]
    data.qpos[7:] = joint_pos[start]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    kp = model.actuator_gainprm[:, 0] * args.kp_scale
    kd = 2.0 * np.sqrt(np.maximum(kp, 1e-6)) * args.kd_scale
    torque_ranges = info["torque_ranges"] * args.torque_limit_scale
    torque_limit = np.where(info["torque_limited"], np.maximum(np.abs(torque_ranges).max(axis=1), 1e-6), np.inf)

    site_ids = {
        "left": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES["left"]),
        "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES["right"]),
    }

    duration = (len(joint_pos) - start) / motion["fps"] / max(args.speed, 1e-6)
    max_time = args.max_time if args.max_time is not None else duration
    steps = int(max_time / model.opt.timestep)

    rows = []
    torque_rows = []
    tracking_rows = []
    contact_rows = []
    fall_time = None
    possible = []

    viewer_ctx = None
    viewer = None
    if not args.disable_viewer:
        mujoco_viewer = importlib.import_module("mujoco.viewer")
        viewer_ctx = mujoco_viewer.launch_passive(model, data)
        viewer = viewer_ctx.__enter__()

    try:
        for step in range(steps):
            sim_time = step * model.opt.timestep
            frame = sample_reference(motion, sim_time + start / motion["fps"], args.speed)
            q = data.qpos[7:].copy()
            dq = data.qvel[6:].copy()
            q_ref = joint_pos[frame]
            dq_ref = joint_vel[frame]
            tau_cmd = kp * (q_ref - q) + kd * (dq_ref - dq)
            tau_applied = np.clip(tau_cmd, -torque_limit, torque_limit)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[6:] = tau_applied
            mujoco.mj_step(model, data)

            rpy = R.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")
            joint_error = q_ref - q
            ratio = np.divide(np.abs(tau_cmd), torque_limit, out=np.zeros_like(tau_cmd), where=np.isfinite(torque_limit))
            finite_ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            forces = contact_forces(model, data, site_ids)
            left_slip = metrics_ref["left_foot_xy_speed"][frame]
            right_slip = metrics_ref["right_foot_xy_speed"][frame]

            rows.append(
                {
                    "step": step,
                    "time": sim_time,
                    "ref_frame": frame,
                    "base_height": data.qpos[2],
                    "base_roll": rpy[0],
                    "base_pitch": rpy[1],
                    "joint_error_norm": np.linalg.norm(joint_error),
                    "max_torque_ratio": np.max(finite_ratio),
                    "left_contact_force": forces["left"],
                    "right_contact_force": forces["right"],
                    "left_foot_slip": left_slip,
                    "right_foot_slip": right_slip,
                }
            )
            tracking_rows.append({"step": step, "time": sim_time, "joint_error_norm": np.linalg.norm(joint_error)})
            contact_rows.append(
                {
                    "step": step,
                    "time": sim_time,
                    "left_contact_force": forces["left"],
                    "right_contact_force": forces["right"],
                    "left_foot_slip": left_slip,
                    "right_foot_slip": right_slip,
                }
            )
            torque_rows.append(
                {
                    "step": step,
                    "time": sim_time,
                    **{f"{name}_tau_cmd": tau_cmd[i] for i, name in enumerate(info["names"])},
                    **{f"{name}_tau_applied": tau_applied[i] for i, name in enumerate(info["names"])},
                    **{f"{name}_torque_ratio": finite_ratio[i] for i, name in enumerate(info["names"])},
                }
            )

            if fall_time is None and (data.qpos[2] < 0.45 or abs(rpy[0]) > 0.9 or abs(rpy[1]) > 0.9):
                fall_time = sim_time
                possible.append("base fell below height/tilt threshold")

            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    break
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

    torque_arr = np.asarray(
        [[row[f"{name}_torque_ratio"] for name in info["names"]] for row in torque_rows], dtype=float
    )
    saturation_rows = []
    for i, name in enumerate(info["names"]):
        ratios = torque_arr[:, i] if len(torque_arr) else np.zeros(1)
        saturation_rows.append(
            {
                "joint_name": name,
                "mean_torque_ratio": float(np.mean(ratios)),
                "max_torque_ratio": float(np.max(ratios)),
                "near_saturation_ratio": float(np.mean(ratios > 0.8)),
                "saturation_ratio": float(np.mean(ratios > 1.0)),
            }
        )
    if any(row["saturation_ratio"] > 0.05 for row in saturation_rows):
        possible.append("long torque saturation")
    if max((row["joint_error_norm"] for row in rows), default=0.0) > 1.0:
        possible.append("large joint tracking error")

    write_csv(out / "pd_replay_log.csv", rows)
    write_csv(out / "torque_log.csv", torque_rows)
    write_csv(out / "tracking_error.csv", tracking_rows)
    write_csv(out / "contact_log.csv", contact_rows)
    write_csv(out / "torque_saturation_summary.csv", saturation_rows)

    time = np.asarray([row["time"] for row in rows], dtype=float)
    plot_series(out / "plots" / "base_height.png", time, [[row["base_height"] for row in rows]], ["base_height"], "Base height", "m")
    plot_series(
        out / "plots" / "base_rpy.png",
        time,
        [[row["base_roll"] for row in rows], [row["base_pitch"] for row in rows]],
        ["roll", "pitch"],
        "Base roll/pitch",
        "rad",
    )
    plot_series(out / "plots" / "joint_tracking_error.png", time, [[row["joint_error_norm"] for row in rows]], ["norm"], "Joint tracking error", "rad")
    plot_series(out / "plots" / "torque_ratio.png", time, [[row["max_torque_ratio"] for row in rows]], ["max_ratio"], "Max torque ratio", "ratio")
    plot_series(
        out / "plots" / "foot_contact_force.png",
        time,
        [[row["left_contact_force"] for row in rows], [row["right_contact_force"] for row in rows]],
        ["left", "right"],
        "Foot contact force",
        "N",
    )
    plot_series(
        out / "plots" / "foot_slip.png",
        time,
        [[row["left_foot_slip"] for row in rows], [row["right_foot_slip"] for row in rows]],
        ["left", "right"],
        "Reference foot slip",
        "m/s",
    )
    write_summary(out / "summary.txt", rows, torque_rows, saturation_rows, fall_time, possible)
    print(f"Wrote DR02 PD replay report to {out}")


if __name__ == "__main__":
    main()
