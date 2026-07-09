import argparse
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def load_motion(path):
    with open(path, "rb") as f:
        motion = pickle.load(f)

    root_pos = np.asarray(motion["root_pos"], dtype=float)
    dof_pos = np.asarray(motion.get("dof_pos", motion.get("joint_pos")), dtype=float)
    if dof_pos is None:
        raise KeyError("Motion file must contain 'dof_pos' or 'joint_pos'.")

    if "root_quat" in motion:
        root_quat = np.asarray(motion["root_quat"], dtype=float)
    else:
        root_quat = np.asarray(motion["root_rot"], dtype=float)[:, [3, 0, 1, 2]]

    return root_pos, root_quat, dof_pos, float(motion.get("fps", 30.0))


def reset_to_frame(model, data, root_pos, root_quat, dof_pos, frame):
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat[frame]
    data.qpos[7:] = dof_pos[frame]
    data.qvel[:] = 0.0
    data.ctrl[:] = dof_pos[frame]
    mujoco.mj_forward(model, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/dr02.xml"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--root_mode", choices=["kinematic", "free"], default="kinematic")
    parser.add_argument("--warmup", type=float, default=0.5, help="Seconds to hold the first pose before playback.")
    parser.add_argument("--reset_each_loop", action="store_true")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    root_pos, root_quat, dof_pos, motion_fps = load_motion(args.motion)
    fps = args.fps if args.fps is not None else motion_fps

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    if dof_pos.shape[1] != model.nu:
        raise ValueError(f"dof_pos has {dof_pos.shape[1]} columns, expected model.nu={model.nu}.")

    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    targets = np.clip(dof_pos, ctrl_min, ctrl_max)

    frame_dt = 1.0 / (fps * args.speed)
    sim_steps_per_frame = max(1, int(round(frame_dt / model.opt.timestep)))
    wall_dt = sim_steps_per_frame * model.opt.timestep

    def step_frame(frame):
        data.ctrl[:] = targets[frame]
        for _ in range(sim_steps_per_frame):
            if args.root_mode == "kinematic":
                data.qpos[:3] = root_pos[frame]
                data.qpos[3:7] = root_quat[frame]
                data.qvel[:6] = 0.0
            mujoco.mj_step(model, data)

    reset_to_frame(model, data, root_pos, root_quat, targets, 0)

    for _ in range(max(0, int(args.warmup / model.opt.timestep))):
        data.ctrl[:] = targets[0]
        if args.root_mode == "kinematic":
            data.qpos[:3] = root_pos[0]
            data.qpos[3:7] = root_quat[0]
            data.qvel[:6] = 0.0
        mujoco.mj_step(model, data)

    if args.no_viewer:
        for frame in range(len(targets)):
            step_frame(frame)
        print(
            f"Simulated {len(targets)} frames with {sim_steps_per_frame} sim steps/frame "
            f"using root_mode={args.root_mode}"
        )
        return

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.time()
            step_frame(frame)
            viewer.sync()

            frame += 1
            if frame >= len(targets):
                frame = 0
                if args.reset_each_loop:
                    reset_to_frame(model, data, root_pos, root_quat, targets, 0)

            elapsed = time.time() - start
            if elapsed < wall_dt:
                time.sleep(wall_dt - elapsed)


if __name__ == "__main__":
    main()
