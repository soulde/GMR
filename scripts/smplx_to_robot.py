import argparse
import pathlib
import time

import numpy as np
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.retarget_export import export_retarget_motion
from general_motion_retargeting.utils.smpl import (
    get_smplx_data_offline_fast,
    load_smplx_file,
)


HERE = pathlib.Path(__file__).parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smplx_file",
        help="SMPLX motion file to load.",
        type=str,
        default=(
            "/home/yanjieze/projects/g1_wbc/GMR/motion_data/ACCAD/"
            "Male1General_c3d/General_A1_-_Stand_stageii.npz"
        ),
    )
    parser.add_argument("--robot", default="unitree_g1")
    parser.add_argument(
        "--mjcf",
        type=pathlib.Path,
        help="Explicit robot MJCF path for an external robot definition.",
    )
    parser.add_argument(
        "--ik-config",
        type=pathlib.Path,
        help="Explicit IK config path for an external robot definition.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write PKL, training NPZ, CSV, and joints.json under retarget_data/<robot>.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop the motion.")
    parser.add_argument(
        "--record_video", action="store_true", help="Record the video."
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Run retargeting without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Limit playback to the human motion frame rate.",
    )
    parser.add_argument(
        "--offset_to_ground",
        action="store_true",
        help="Align the lowest human foot target to the ground before solving IK.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.save and args.loop:
        parser.error("--save cannot be combined with --loop")
    if args.mjcf is not None and not args.no_viewer:
        parser.error(
            "external --mjcf currently requires --no_viewer; "
            "use vis_gmr_debug.py for playback"
        )

    smplx_folder = HERE / ".." / "assets" / "body_models"
    smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(
        args.smplx_file, smplx_folder
    )
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    retarget_kwargs = dict(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )
    if args.mjcf is not None:
        retarget_kwargs["robot_xml_path"] = args.mjcf
    if args.ik_config is not None:
        retarget_kwargs["ik_config_path"] = args.ik_config
    retarget = GMR(**retarget_kwargs)

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=f"videos/{args.robot}_{pathlib.Path(args.smplx_file).stem}.mp4",
        )

    qpos_frames = []
    frame_index = 0
    fps_counter = 0
    fps_start_time = time.time()
    try:
        while args.loop or frame_index < len(frames):
            frame = frames[frame_index % len(frames)]
            qpos = retarget.retarget(
                frame, offset_to_ground=args.offset_to_ground
            )
            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retarget.scaled_human_data,
                    human_pos_offset=np.array([0.0, 0.0, 0.0]),
                    show_human_body_name=False,
                    rate_limit=args.rate_limit,
                    follow_camera=False,
                )
            if args.save:
                qpos_frames.append(qpos.copy())

            frame_index += 1
            fps_counter += 1
            now = time.time()
            if now - fps_start_time >= 2.0:
                print(f"Actual rendering FPS: {fps_counter / (now - fps_start_time):.2f}")
                fps_counter = 0
                fps_start_time = now

        if args.save:
            paths = export_retarget_motion(
                retarget.model,
                args.robot,
                args.smplx_file,
                aligned_fps,
                np.asarray(qpos_frames),
            )
            print(f"Saved GMR motion to {paths.motion}")
            print(f"Saved training dataset to {paths.dataset}")
            print(f"Saved BeyondMimic CSV to {paths.csv}")
            print(f"Joint order contract: {paths.joints}")
            print(f"Body order contract: {paths.bodies}")
    finally:
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
