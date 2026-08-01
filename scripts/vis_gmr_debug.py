"""Unified MuJoCo playback and GMR position-target debugger."""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np

from general_motion_retargeting.gmr_debug_visualizer import (
    GMRDebugVisualizer,
    ReferencePoint,
    compose_scene_xml,
    compute_height_offset,
    load_effective_ik_config,
    transform_full_reference_frame,
    transform_reference_frame,
)


def parse_height_offset(value: str) -> str | float:
    if value == "auto":
        return value
    try:
        return float(value)
    except ValueError as error:
        raise ValueError("height offset must be auto or a number") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--ik-config", type=Path, required=True)
    parser.add_argument("--height-offset", default="auto")
    parser.add_argument("--actual-human-height", type=float)
    parser.add_argument(
        "--body-model-dir", type=Path, default=Path("assets/body_models")
    )
    parser.add_argument("--robot-alpha", type=float, default=0.3)
    parser.add_argument("--camera-body")
    parser.add_argument("--camera-distance", type=float)
    parser.add_argument("--camera-azimuth", type=float, default=180.0)
    parser.add_argument("--camera-elevation", type=float, default=-12.0)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--no-error-lines", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def load_motion(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    required = ("fps", "root_pos", "root_rot", "dof_pos")
    missing = [name for name in required if name not in motion]
    if missing:
        raise ValueError(f"motion is missing fields: {', '.join(missing)}")
    result = dict(motion)
    for name in ("root_pos", "root_rot", "dof_pos"):
        result[name] = np.asarray(result[name], dtype=float)
    frame_counts = {len(result[name]) for name in ("root_pos", "root_rot", "dof_pos")}
    if len(frame_counts) != 1:
        raise ValueError("motion arrays have different frame counts")
    return result


def load_reference_frames(
    path: Path, body_model_dir: Path
) -> tuple[list[dict], float, float | None, tuple[tuple[str, str], ...]]:
    """Load raw reference frames; SMPL-X remains the canonical input."""
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        from general_motion_retargeting.utils.smpl import (
            get_smplx_data_offline_fast,
            load_smplx_file,
        )

        smplx_data, body_model, output, height = load_smplx_file(
            path, body_model_dir
        )
        frames, fps = get_smplx_data_offline_fast(
            smplx_data, body_model, output, tgt_fps=30
        )
        from smplx.joint_names import JOINT_NAMES

        names = JOINT_NAMES[: len(body_model.parents)]
        parents = np.asarray(body_model.parents.cpu(), dtype=int)
        edges = tuple(
            (names[int(parents[index])], names[index])
            for index in range(1, len(names))
            if int(parents[index]) >= 0
        )
        return frames, float(fps), float(height), edges
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict):
            frames = payload.get("reference_frames", payload.get("frames"))
            fps = float(payload.get("fps", 30.0))
            height = payload.get("actual_human_height")
            names = payload.get("joint_names")
            parents = payload.get("parents")
        else:
            frames, fps, height, names, parents = payload, 30.0, None, None, None
        if not isinstance(frames, (list, tuple)):
            raise ValueError("reference pickle must contain a frame list")
        edges = ()
        if names is not None and parents is not None:
            edges = tuple(
                (names[int(parents[index])], names[index])
                for index in range(1, len(names))
                if int(parents[index]) >= 0
            )
        return list(frames), fps, None if height is None else float(height), edges
    raise ValueError("reference must be an SMPL-X .npz or frame .pkl")


def align_reference_frames(frames: list, motion_frame_count: int) -> list:
    if len(frames) == motion_frame_count:
        return frames
    if len(frames) == motion_frame_count + 1:
        return frames[1:]
    raise ValueError(
        f"frame count mismatch: motion={motion_frame_count}, reference={len(frames)}"
    )


def _translated_reference(
    points: dict[str, ReferencePoint], height_offset: float
) -> dict[str, ReferencePoint]:
    translation = np.array([0.0, 0.0, height_offset])
    return {
        name: ReferencePoint(
            reference_name=point.reference_name,
            world_position=point.world_position + translation,
            world_rotation=point.world_rotation,
        )
        for name, point in points.items()
    }


def _camera_body(model: mujoco.MjModel, config: dict, requested: str | None) -> str:
    name = requested or config.get("robot_root_name")
    if name and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
        return name
    if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        return model.body(model.jnt_bodyid[0]).name
    raise ValueError("cannot infer camera body; pass --camera-body")


def _set_frame(model, data, motion, index, height_offset):
    data.qpos[:3] = motion["root_pos"][index]
    data.qpos[2] += height_offset
    data.qpos[3:7] = motion["root_rot"][index][[3, 0, 1, 2]]
    if data.qpos[7:].shape != motion["dof_pos"][index].shape:
        raise ValueError(
            f"motion has {motion['dof_pos'].shape[1]} joints but MJCF expects "
            f"{data.qpos[7:].shape[0]}"
        )
    data.qpos[7:] = motion["dof_pos"][index]
    mujoco.mj_forward(model, data)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        height_option = parse_height_offset(args.height_offset)
        motion = load_motion(args.motion)
        frames, reference_fps, detected_height, full_reference_edges = load_reference_frames(
            args.reference, args.body_model_dir
        )
        frames = align_reference_frames(frames, len(motion["root_pos"]))
        actual_height = args.actual_human_height or detected_height
        config = load_effective_ik_config(args.ik_config, actual_height)
        model = mujoco.MjModel.from_xml_string(compose_scene_xml(args.mjcf))
        data = mujoco.MjData(model)
        height_offset = (
            compute_height_offset(model, motion)
            if height_option == "auto"
            else float(height_option)
        )
        camera_body = _camera_body(model, config, args.camera_body)
        visualizer = GMRDebugVisualizer(
            model,
            config,
            robot_alpha=args.robot_alpha,
            display_name=args.reference.stem,
        )
        fps = float(motion["fps"] or reference_fps)
        frame_limit = args.max_frames
        print(
            f"GMR debug viewer: frames={len(frames)}, mappings={len(visualizer.mappings)}, "
            f"height_offset={height_offset:.6f} m"
        )

        def update(index, viewer=None):
            _set_frame(model, data, motion, index, height_offset)
            reference = transform_reference_frame(frames[index], config)
            full_reference = transform_full_reference_frame(
                frames[index],
                config,
                hierarchy_edges=full_reference_edges,
                exact_targets=reference,
            )
            reference = _translated_reference(reference, height_offset)
            full_reference = _translated_reference(full_reference, height_offset)
            if viewer is None:
                from general_motion_retargeting.gmr_debug_visualizer import (
                    compute_correspondences,
                )

                rows = compute_correspondences(
                    model, data, reference, visualizer.mappings
                )
                return visualizer.statistics(rows, index)
            return visualizer.update(
                viewer,
                data,
                reference if not args.no_reference else {},
                index,
                show_error_lines=not args.no_error_lines,
                full_reference_skeleton=(
                    full_reference if not args.no_reference else {}
                ),
                full_reference_edges=full_reference_edges,
            )

        if args.headless:
            count = len(frames) if frame_limit is None else min(frame_limit, len(frames))
            last = None
            for index in range(count):
                last = update(index)
            if last is not None:
                print(
                    f"frame={last.frame_index} mean={last.mean:.6f} "
                    f"rms={last.rms:.6f} max={last.maximum:.6f}"
                )
            visualizer.restore_robot_alpha()
            return 0

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.distance = args.camera_distance or max(1.0, 2.0 * model.stat.extent)
            viewer.cam.azimuth = args.camera_azimuth
            viewer.cam.elevation = args.camera_elevation
            index = 0
            rendered = 0
            while viewer.is_running():
                started = time.monotonic()
                update(index, viewer)
                viewer.cam.lookat[:] = data.body(camera_body).xpos
                viewer.sync()
                rendered += 1
                if frame_limit is not None and rendered >= frame_limit:
                    break
                index += 1
                if index >= len(frames):
                    if not args.loop:
                        break
                    index = 0
                time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
        visualizer.restore_robot_alpha()
        return 0
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
