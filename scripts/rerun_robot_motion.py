"""Visualize robot motion files with Rerun and an explicit MJCF model."""

from __future__ import annotations

import argparse
import pathlib

import mujoco
import rerun as rr

from general_motion_retargeting.rerun_motion import (
    discover_motions,
    record_motions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="A .pkl/.npz motion file or a folder searched recursively.",
    )
    parser.add_argument(
        "--mjcf",
        type=pathlib.Path,
        required=True,
        help="MJCF model used to validate the motion joint layout.",
    )
    parser.add_argument(
        "--urdf",
        type=pathlib.Path,
        required=True,
        help="URDF model imported by Rerun for geometry and transforms.",
    )
    parser.add_argument(
        "--recording-name",
        help="Rerun recording name (defaults to the input name).",
    )
    parser.add_argument(
        "--save",
        type=pathlib.Path,
        help="Optionally save the recording to an .rrd file.",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not open a Rerun Viewer; requires --save.",
    )
    return parser


def _configure_sinks(
    recording: rr.RecordingStream,
    *,
    spawn: bool,
    save_path: pathlib.Path | None,
) -> None:
    if spawn and save_path is not None:
        recording.spawn(connect=False)
        recording.set_sinks(rr.GrpcSink(), rr.FileSink(save_path))
    elif spawn:
        recording.spawn()
    elif save_path is not None:
        recording.save(save_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.no_spawn and args.save is None:
        parser.error("--no-spawn requires --save")

    try:
        specs = discover_motions(args.input)
        if not args.mjcf.is_file():
            raise FileNotFoundError(args.mjcf)
        if not args.urdf.is_file():
            raise FileNotFoundError(args.urdf)
        model = mujoco.MjModel.from_xml_path(str(args.mjcf))
        recording_name = args.recording_name or args.input.stem
        with rr.RecordingStream(recording_name) as recording:
            _configure_sinks(
                recording,
                spawn=not args.no_spawn,
                save_path=args.save,
            )
            record_motions(recording, model, args.urdf, specs)
            recording.flush()
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    if args.save is not None:
        print(f"Saved Rerun recording to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
