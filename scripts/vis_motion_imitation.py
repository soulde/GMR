import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.quadruped.kinematics import (
    set_free_root,
    set_named_joint_positions,
)
from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


CONFIG_ROOT = REPO_ROOT / "general_motion_retargeting/quadruped/configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Play motion_imitation data on its declared source MJCF."
    )
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--source_robot", default="laikago")
    parser.add_argument("--rate_limit", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_robot_spec(
        CONFIG_ROOT / f"{args.source_robot}.yaml",
        REPO_ROOT,
    )
    motion = load_motion_imitation(
        args.motion_file,
        spec.motion_joint_order,
        spec.quaternion_order,
        spec.root_frame_rotation_wxyz,
    )
    scene_path = spec.mjcf_path.with_name("scene.xml")
    model = mujoco.MjModel.from_xml_path(
        str(scene_path if scene_path.is_file() else spec.mjcf_path)
    )
    data = mujoco.MjData(model)
    limiter = RateLimiter(frequency=motion.fps, warn=False)

    with mujoco.viewer.launch_passive(
        model,
        data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        frame = 0
        while viewer.is_running():
            set_free_root(
                model,
                data,
                motion.root_pos[frame],
                motion.root_rot[frame],
            )
            set_named_joint_positions(
                model,
                data,
                motion.joint_names,
                motion.joint_pos[frame],
            )
            mujoco.mj_forward(model, data)
            viewer.cam.lookat = data.xpos[model.body(spec.root_body).id]
            viewer.cam.distance = 1.5
            viewer.sync()
            if args.rate_limit:
                limiter.sleep()
            frame = (frame + 1) % len(motion.root_pos)


if __name__ == "__main__":
    main()
