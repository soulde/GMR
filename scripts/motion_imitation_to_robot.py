import argparse
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.factory import create_retargeter
from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)
from general_motion_retargeting.quadruped.robot_spec import load_robot_spec

CONFIG_ROOT = REPO_ROOT / "general_motion_retargeting" / "quadruped" / "configs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retarget a motion_imitation quadruped motion to a robot."
    )
    parser.add_argument(
        "--model_type",
        choices=("quadruped",),
        default="quadruped",
    )
    parser.add_argument("--motion_file", type=Path, required=True)
    parser.add_argument("--source_robot", default="laikago")
    parser.add_argument("--robot", default="unitree_go2")
    parser.add_argument("--save_path", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=Path, default=None)
    parser.add_argument("--use_velocity_limit", action="store_true")
    return parser


def _load_named_spec(robot: str):
    config_path = CONFIG_ROOT / f"{robot}.yaml"
    if not config_path.is_file():
        available = ", ".join(path.stem for path in sorted(CONFIG_ROOT.glob("*.yaml")))
        raise ValueError(
            f"unknown quadruped robot {robot!r}; available configs: {available}"
        )
    return load_robot_spec(config_path, REPO_ROOT)


def _joint_qpos_addresses(spec) -> np.ndarray:
    addresses = []
    for joint_name in spec.motion_joint_order:
        joint_id = mujoco.mj_name2id(
            spec.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        addresses.append(spec.model.jnt_qposadr[joint_id])
    return np.asarray(addresses, dtype=int)


def _free_root_qpos_address(model: mujoco.MjModel) -> int:
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 1:
        raise ValueError(f"target MJCF must have one free root, found {len(free_joints)}")
    return int(model.jnt_qposadr[free_joints[0]])


def _serialize_result(result, target_spec, save_path: Path) -> None:
    root_address = _free_root_qpos_address(target_spec.model)
    joint_addresses = _joint_qpos_addresses(target_spec)
    root_wxyz = result.qpos[:, root_address + 3 : root_address + 7]
    payload = {
        "fps": float(result.fps),
        "root_pos": result.qpos[:, root_address : root_address + 3].copy(),
        "root_rot": root_wxyz[:, [1, 2, 3, 0]].copy(),
        "dof_pos": result.qpos[:, joint_addresses].copy(),
        "local_body_pos": None,
        "link_body_list": None,
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as stream:
        pickle.dump(payload, stream)


def run(
    motion_file,
    source_robot="laikago",
    robot="unitree_go2",
    save_path=None,
    headless=False,
    rate_limit=False,
    record_video=False,
    video_path=None,
    use_velocity_limit=False,
    model_type="quadruped",
):
    source_spec = _load_named_spec(source_robot)
    target_spec = _load_named_spec(robot)
    motion = load_motion_imitation(
        motion_file,
        source_spec.motion_joint_order,
        source_spec.quaternion_order,
    )
    retargeter = create_retargeter(
        model_type=model_type,
        source_spec=source_spec,
        target_spec=target_spec,
        use_velocity_limit=use_velocity_limit,
    )
    result = retargeter.retarget_motion(motion)

    if save_path is not None:
        _serialize_result(result, target_spec, Path(save_path))

    if not headless:
        from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer

        resolved_video_path = (
            Path(video_path)
            if video_path is not None
            else REPO_ROOT / "videos" / f"{robot}_{Path(motion_file).stem}.mp4"
        )
        viewer = RobotMotionViewer(
            robot_type=robot,
            motion_fps=result.fps,
            record_video=record_video,
            video_path=str(resolved_video_path),
        )
        joint_addresses = _joint_qpos_addresses(target_spec)
        root_address = _free_root_qpos_address(target_spec.model)
        try:
            for qpos in result.qpos:
                viewer.step(
                    root_pos=qpos[root_address : root_address + 3],
                    root_rot=qpos[root_address + 3 : root_address + 7],
                    dof_pos=qpos[joint_addresses],
                    rate_limit=rate_limit,
                )
        finally:
            viewer.close()

    return result


def main() -> None:
    args = build_parser().parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
