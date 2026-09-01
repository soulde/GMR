"""Remove high-error retargeting frames and export contiguous clean clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import mujoco
import numpy as np

from general_motion_retargeting.gmr_debug_visualizer import (
    build_position_correspondences,
    compute_correspondences,
    load_effective_ik_config,
    transform_reference_frame,
)
from general_motion_retargeting.motion_cleaning import (
    clean_motion_segments,
    contiguous_valid_segments,
)
from scripts.vis_gmr_debug import (
    align_reference_frames,
    build_reference_reconstructor,
    load_motion,
    load_reference_frames,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument("--ik-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--algorithm", choices=["gmr", "skeleton"], default="skeleton"
    )
    parser.add_argument(
        "--max-position-error",
        type=float,
        default=0.10,
        help="Remove frames whose largest active position-task error exceeds this many meters.",
    )
    parser.add_argument("--minimum-segment-frames", type=int, default=30)
    parser.add_argument("--padding-frames", type=int, default=0)
    parser.add_argument("--actual-human-height", type=float)
    parser.add_argument(
        "--bvh-format", choices=["lafan1", "nokov"], default="lafan1"
    )
    return parser


def _set_motion_frame(model, data, motion: dict, index: int) -> None:
    data.qpos[:3] = motion["root_pos"][index]
    data.qpos[3:7] = motion["root_rot"][index][[3, 0, 1, 2]]
    if data.qpos[7:].shape != motion["dof_pos"][index].shape:
        raise ValueError("motion DoF count does not match MJCF")
    data.qpos[7:] = motion["dof_pos"][index]
    mujoco.mj_forward(model, data)


def analyze_frame_errors(
    motion: dict,
    frames: list,
    model,
    config: dict,
    algorithm: str,
) -> tuple[np.ndarray, list[dict]]:
    mappings = build_position_correspondences(model, config)
    reconstructor = build_reference_reconstructor(algorithm, config)
    if reconstructor is not None:
        config["human_scale_table"] = {
            name: 1.0 for name in config["human_scale_table"]
        }
    data = mujoco.MjData(model)
    maximum_errors = []
    frame_rows = []
    for index, frame in enumerate(frames):
        _set_motion_frame(model, data, motion, index)
        if reconstructor is not None:
            frame = reconstructor.reconstruct(frame)
        reference = transform_reference_frame(frame, config)
        rows = compute_correspondences(model, data, reference, mappings)
        maximum_errors.append(max((row.error_norm for row in rows), default=0.0))
        frame_rows.append(
            {
                f"stage{row.stage}:{row.robot_name}->{row.reference_name}": row.error_norm
                for row in rows
            }
        )
    return np.asarray(maximum_errors), frame_rows


def _write_outputs(
    motion_path: Path,
    output_dir: Path,
    motion: dict,
    segments: tuple[tuple[int, int], ...],
    report: dict,
) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = motion_path.stem
    part_paths = tuple(
        output_dir / f"{stem}_part{index:03d}.pkl"
        for index in range(1, len(segments) + 1)
    )
    report_path = output_dir / f"{stem}_cleaning_report.json"
    existing = [path for path in (*part_paths, report_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")

    for path, part in zip(part_paths, clean_motion_segments(motion, segments)):
        with path.open("wb") as stream:
            pickle.dump(part, stream)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return (*part_paths, report_path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    motion = load_motion(args.motion)
    frames, _, detected_height, _ = load_reference_frames(
        args.reference, Path("assets/body_models"), args.bvh_format
    )
    frames = align_reference_frames(frames, len(motion["root_pos"]))
    config = load_effective_ik_config(
        args.ik_config, args.actual_human_height or detected_height
    )
    model = mujoco.MjModel.from_xml_path(str(args.mjcf.resolve()))
    maximum_errors, frame_rows = analyze_frame_errors(
        motion, frames, model, config, args.algorithm
    )
    segments = contiguous_valid_segments(
        maximum_errors,
        threshold=args.max_position_error,
        minimum_segment_frames=args.minimum_segment_frames,
        padding_frames=args.padding_frames,
    )
    invalid = maximum_errors > args.max_position_error
    bad_frames = []
    for index in np.flatnonzero(invalid):
        failures = {
            name: value
            for name, value in frame_rows[int(index)].items()
            if value > args.max_position_error
        }
        bad_frames.append(
            {
                "frame": int(index),
                "maximum_position_error": float(maximum_errors[index]),
                "failures": failures,
            }
        )
    report = {
        "motion": str(args.motion.resolve()),
        "reference": str(args.reference.resolve()),
        "algorithm": args.algorithm,
        "maximum_position_error_threshold": args.max_position_error,
        "padding_frames": args.padding_frames,
        "minimum_segment_frames": args.minimum_segment_frames,
        "input_frames": len(maximum_errors),
        "bad_frame_count": int(invalid.sum()),
        "kept_frame_count": int(sum(stop - start for start, stop in segments)),
        "discarded_frame_count": int(
            len(maximum_errors) - sum(stop - start for start, stop in segments)
        ),
        "short_fragment_frame_count": int(
            len(maximum_errors)
            - invalid.sum()
            - sum(stop - start for start, stop in segments)
        ),
        "segments": [
            {"part": index, "start": start, "stop": stop, "frames": stop - start}
            for index, (start, stop) in enumerate(segments, start=1)
        ],
        "bad_frames": bad_frames,
    }
    outputs = _write_outputs(args.motion, args.output_dir, motion, segments, report)
    print(
        f"cleaned {len(maximum_errors)} frames: removed {invalid.sum()}, "
        f"exported {len(segments)} clips to {args.output_dir}"
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
