import argparse
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def _load_motion(path):
    with open(path, "rb") as f:
        motion = pickle.load(f)

    root_pos = np.asarray(motion["root_pos"])
    dof_pos = np.asarray(motion.get("dof_pos", motion.get("joint_pos")))
    if dof_pos is None:
        raise KeyError("Motion file must contain 'dof_pos' or 'joint_pos'.")

    if "root_quat" in motion:
        root_rot = np.asarray(motion["root_quat"])
    else:
        root_rot = np.asarray(motion["root_rot"])
        root_rot = root_rot[:, [3, 0, 1, 2]]

    fps = motion.get("fps", 30)
    return root_pos, root_rot, dof_pos, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/mjcf/dr02_pos.xml"))
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()

    root_pos, root_rot, dof_pos, motion_fps = _load_motion(args.motion)
    fps = args.fps if args.fps is not None else motion_fps

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)

    if dof_pos.shape[1] != model.nq - 7:
        raise ValueError(
            f"dof_pos has {dof_pos.shape[1]} columns, expected {model.nq - 7} for {args.xml}."
        )

    if args.no_viewer:
        for frame in range(len(root_pos)):
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_rot[frame]
            data.qpos[7:] = dof_pos[frame]
            mujoco.mj_forward(model, data)
        print(f"Validated {len(root_pos)} frames against {args.xml}")
        return

    frame_dt = 1.0 / (fps * args.speed)
    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_rot[frame]
            data.qpos[7:] = dof_pos[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(frame_dt)
            frame = (frame + 1) % len(root_pos)


if __name__ == "__main__":
    main()
