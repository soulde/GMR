import csv
import os
import pickle
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


FOOT_BODIES = {
    "left": "left_ankle_x_link",
    "right": "right_ankle_x_link",
}

FOOT_SITES = {
    "left": "left_foot",
    "right": "right_foot",
}


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_motion(path):
    path = Path(path)
    if path.suffix == ".npz":
        data = dict(np.load(path, allow_pickle=True))
        fps = float(np.asarray(data["fps"]).item()) if np.asarray(data["fps"]).shape == () else float(data["fps"])
        root_pos = np.asarray(data["root_pos"], dtype=float)
        root_quat = np.asarray(data.get("root_quat", data.get("root_rot")), dtype=float)
        joint_pos = np.asarray(data.get("joint_pos", data.get("dof_pos")), dtype=float)
    else:
        with path.open("rb") as f:
            data = pickle.load(f)
        fps = float(data.get("fps", 30.0))
        root_pos = np.asarray(data["root_pos"], dtype=float)
        joint_pos = np.asarray(data.get("joint_pos", data.get("dof_pos")), dtype=float)
        if "root_quat" in data:
            root_quat = np.asarray(data["root_quat"], dtype=float)
        else:
            root_quat = np.asarray(data["root_rot"], dtype=float)
            # GMR pkl stores root_rot as xyzw.
            root_quat = root_quat[:, [3, 0, 1, 2]]

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T, 3), got {root_pos.shape}")
    if root_quat.ndim != 2 or root_quat.shape[1] != 4:
        raise ValueError(f"root_quat/root_rot must have shape (T, 4), got {root_quat.shape}")
    if joint_pos.ndim != 2:
        raise ValueError(f"joint_pos/dof_pos must have shape (T, N), got {joint_pos.shape}")

    root_quat = normalize_quat(root_quat)
    return {
        "fps": fps,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_pos": joint_pos,
        "raw": data,
    }


def save_motion_pkl(path, motion):
    path = Path(path)
    ensure_dir(path.parent)
    payload = {
        "fps": float(motion["fps"]),
        "root_pos": np.asarray(motion["root_pos"], dtype=float),
        "root_rot": np.asarray(motion["root_quat"], dtype=float)[:, [1, 2, 3, 0]],
        "root_quat": np.asarray(motion["root_quat"], dtype=float),
        "dof_pos": np.asarray(motion["joint_pos"], dtype=float),
        "joint_pos": np.asarray(motion["joint_pos"], dtype=float),
        "joint_vel": np.asarray(motion.get("joint_vel", []), dtype=float),
        "root_lin_vel": np.asarray(motion.get("root_lin_vel", []), dtype=float),
        "root_ang_vel": np.asarray(motion.get("root_ang_vel", []), dtype=float),
        "left_foot_contact": np.asarray(motion.get("left_foot_contact", [])),
        "right_foot_contact": np.asarray(motion.get("right_foot_contact", [])),
        "local_body_pos": None,
        "link_body_list": None,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return quat / norm


def unwrap_yaw(root_quat):
    yaw = R.from_quat(root_quat, scalar_first=True).as_euler("xyz")[:, 2]
    return np.unwrap(yaw)


def quat_slerp(q0, q1, alpha):
    key_rots = R.from_quat(np.stack([q0, q1]), scalar_first=True)
    slerp = Slerp([0.0, 1.0], key_rots)
    return slerp(alpha).as_quat(scalar_first=True)


def finite_difference(values, fps):
    values = np.asarray(values, dtype=float)
    vel = np.zeros_like(values)
    if len(values) < 2:
        return vel
    dt = 1.0 / fps
    vel[:-1] = (values[1:] - values[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def angular_velocity_from_quat(root_quat, fps):
    root_quat = normalize_quat(root_quat)
    ang = np.zeros((len(root_quat), 3), dtype=float)
    if len(root_quat) < 2:
        return ang
    dt = 1.0 / fps
    rots = R.from_quat(root_quat, scalar_first=True)
    for i in range(len(root_quat) - 1):
        delta = rots[i].inv() * rots[i + 1]
        ang[i] = delta.as_rotvec() / dt
    ang[-1] = ang[-2]
    return ang


def model_joint_info(model):
    names = []
    qpos_addrs = []
    dof_addrs = []
    ranges = []
    limited = []
    torque_ranges = []
    torque_limited = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
        dof_addrs.append(int(model.jnt_dofadr[jid]))
        ranges.append(model.jnt_range[jid].copy())
        limited.append(bool(model.jnt_limited[jid]))
        torque_ranges.append(model.jnt_actfrcrange[jid].copy())
        torque_limited.append(bool(model.jnt_actfrclimited[jid]))
    return {
        "names": names,
        "qpos_addrs": np.asarray(qpos_addrs, dtype=int),
        "dof_addrs": np.asarray(dof_addrs, dtype=int),
        "ranges": np.asarray(ranges, dtype=float),
        "limited": np.asarray(limited, dtype=bool),
        "torque_ranges": np.asarray(torque_ranges, dtype=float),
        "torque_limited": np.asarray(torque_limited, dtype=bool),
    }


def set_qpos(data, root_pos, root_quat, joint_pos):
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat
    data.qpos[7:] = joint_pos


def compute_kinematic_metrics(model, motion):
    root_pos = motion["root_pos"]
    root_quat = motion["root_quat"]
    joint_pos = motion["joint_pos"]
    fps = motion["fps"]

    if joint_pos.shape[1] != model.nq - 7:
        raise ValueError(f"joint_pos has {joint_pos.shape[1]} columns, expected {model.nq - 7}")

    data = mujoco.MjData(model)
    left_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES["left"])
    right_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, FOOT_SITES["right"])

    T = len(root_pos)
    left_foot_pos = np.zeros((T, 3), dtype=float)
    right_foot_pos = np.zeros((T, 3), dtype=float)
    base_rpy = R.from_quat(root_quat, scalar_first=True).as_euler("xyz")

    for t in range(T):
        set_qpos(data, root_pos[t], root_quat[t], joint_pos[t])
        mujoco.mj_forward(model, data)
        left_foot_pos[t] = data.site_xpos[left_site]
        right_foot_pos[t] = data.site_xpos[right_site]

    left_foot_vel = finite_difference(left_foot_pos, fps)
    right_foot_vel = finite_difference(right_foot_pos, fps)
    joint_vel = finite_difference(joint_pos, fps)
    root_lin_vel = finite_difference(root_pos, fps)
    root_ang_vel = angular_velocity_from_quat(root_quat, fps)

    left_xy_speed = np.linalg.norm(left_foot_vel[:, :2], axis=1)
    right_xy_speed = np.linalg.norm(right_foot_vel[:, :2], axis=1)

    return {
        "fps": fps,
        "time": np.arange(T) / fps,
        "root_pos": root_pos,
        "root_quat": root_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_lin_vel": root_lin_vel,
        "root_ang_vel": root_ang_vel,
        "left_foot_pos": left_foot_pos,
        "right_foot_pos": right_foot_pos,
        "left_foot_vel": left_foot_vel,
        "right_foot_vel": right_foot_vel,
        "left_foot_height": left_foot_pos[:, 2],
        "right_foot_height": right_foot_pos[:, 2],
        "left_foot_xy_speed": left_xy_speed,
        "right_foot_xy_speed": right_xy_speed,
        "base_height": root_pos[:, 2],
        "base_roll": base_rpy[:, 0],
        "base_pitch": base_rpy[:, 1],
        "base_yaw": np.unwrap(base_rpy[:, 2]),
    }


def estimate_contacts(metrics, height_threshold=0.04, xy_speed_threshold=0.20):
    left = (metrics["left_foot_height"] < height_threshold) & (
        metrics["left_foot_xy_speed"] < xy_speed_threshold
    )
    right = (metrics["right_foot_height"] < height_threshold) & (
        metrics["right_foot_xy_speed"] < xy_speed_threshold
    )
    return left.astype(bool), right.astype(bool)


def joint_limit_report(model, joint_pos):
    info = model_joint_info(model)
    q_ref_min = joint_pos.min(axis=0)
    q_ref_max = joint_pos.max(axis=0)
    rows = []
    for i, name in enumerate(info["names"]):
        q_min, q_max = info["ranges"][i]
        low_margin = q_ref_min[i] - q_min
        high_margin = q_max - q_ref_max[i]
        below = joint_pos[:, i] < q_min
        above = joint_pos[:, i] > q_max
        violation_count = int(np.count_nonzero(below | above))
        violation_ratio = violation_count / len(joint_pos)
        rows.append(
            {
                "joint_name": name,
                "q_min": q_min,
                "q_max": q_max,
                "q_ref_min": q_ref_min[i],
                "q_ref_max": q_ref_max[i],
                "min_margin": low_margin,
                "max_margin": high_margin,
                "violation_count": violation_count,
                "violation_ratio": violation_ratio,
            }
        )
    return rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_array_csv(path, header, array):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(np.asarray(array))


def plot_series(path, time, series, labels, title, ylabel):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(10, 4))
    for values, label in zip(series, labels):
        plt.plot(time, values, label=label)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    if labels:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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


def command_refs(root_pos, root_quat, fps):
    root_lin_vel = finite_difference(root_pos, fps)
    yaw = unwrap_yaw(root_quat)
    yaw_rate = finite_difference(yaw[:, None], fps)[:, 0]
    return root_lin_vel[:, 0], root_lin_vel[:, 1], yaw_rate


def compute_dataset_fields(model, motion, cycle_length=None, contact_height=0.04, contact_xy_speed=0.20):
    metrics = compute_kinematic_metrics(model, motion)
    left_contact, right_contact = estimate_contacts(metrics, contact_height, contact_xy_speed)
    T = len(metrics["root_pos"])
    if cycle_length is None or cycle_length <= 0:
        cycle_length = T
    phase = (np.arange(T) % cycle_length) / float(cycle_length)
    vx_ref, vy_ref, yaw_rate_ref = command_refs(metrics["root_pos"], metrics["root_quat"], metrics["fps"])
    joint_names = np.asarray(model_joint_info(model)["names"], dtype=object)
    return {
        "root_pos": metrics["root_pos"],
        "root_quat": metrics["root_quat"],
        "root_lin_vel": metrics["root_lin_vel"],
        "root_ang_vel": metrics["root_ang_vel"],
        "joint_pos": metrics["joint_pos"],
        "joint_vel": metrics["joint_vel"],
        "left_foot_pos": metrics["left_foot_pos"],
        "right_foot_pos": metrics["right_foot_pos"],
        "left_foot_vel": metrics["left_foot_vel"],
        "right_foot_vel": metrics["right_foot_vel"],
        "left_foot_contact": left_contact.astype(np.uint8),
        "right_foot_contact": right_contact.astype(np.uint8),
        "phase": phase,
        "phase_sin": np.sin(2 * np.pi * phase),
        "phase_cos": np.cos(2 * np.pi * phase),
        "vx_ref": vx_ref,
        "vy_ref": vy_ref,
        "yaw_rate_ref": yaw_rate_ref,
        "fps": np.asarray(metrics["fps"], dtype=float),
        "joint_names": joint_names,
    }
