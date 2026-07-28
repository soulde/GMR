#!/usr/bin/env python3
"""Read-only checks for GMR retargeted robot motion files."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def iter_motion_files(path: Path):
    if path.is_file():
        yield path
        return
    for suffix in ("*.pkl", "*.npz"):
        yield from sorted(path.rglob(suffix))


def load_motion(path: Path):
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    with path.open("rb") as f:
        return pickle.load(f)


def array(data, *names):
    for name in names:
        if name in data:
            return name, np.asarray(data[name])
    return None, None


def check_file(path: Path, min_frames: int, expected_dof: int | None):
    result = {"path": str(path), "status": "PASS", "issues": []}
    try:
        data = load_motion(path)
    except Exception as exc:
        result["status"] = "FAIL"
        result["issues"].append(f"load_error: {exc}")
        return result

    fps = data.get("fps")
    try:
        fps = float(np.asarray(fps).item())
    except Exception:
        result["issues"].append("fps_missing_or_not_scalar")
        fps = None

    _, root_pos = array(data, "root_pos")
    quat_name, root_quat = array(data, "root_quat", "root_rot")
    joint_name, joint_pos = array(data, "joint_pos", "dof_pos")

    if root_pos is None:
        result["issues"].append("missing_root_pos")
    if root_quat is None:
        result["issues"].append("missing_root_quat_or_root_rot")
    if joint_pos is None:
        result["issues"].append("missing_joint_pos_or_dof_pos")

    frame_counts = []
    for name, value, width in [
        ("root_pos", root_pos, 3),
        (quat_name or "root_quat/root_rot", root_quat, 4),
    ]:
        if value is None:
            continue
        if value.ndim != 2 or value.shape[1] != width:
            result["issues"].append(f"{name}_bad_shape:{value.shape}")
        else:
            frame_counts.append(int(value.shape[0]))
        if value.size and not np.isfinite(value).all():
            result["issues"].append(f"{name}_nonfinite")

    if joint_pos is not None:
        if joint_pos.ndim != 2:
            result["issues"].append(f"{joint_name}_bad_shape:{joint_pos.shape}")
        else:
            frame_counts.append(int(joint_pos.shape[0]))
            if expected_dof is not None and joint_pos.shape[1] != expected_dof:
                result["issues"].append(f"{joint_name}_dof:{joint_pos.shape[1]}_expected:{expected_dof}")
        if joint_pos.size and not np.isfinite(joint_pos).all():
            result["issues"].append(f"{joint_name}_nonfinite")

    if frame_counts:
        unique_counts = sorted(set(frame_counts))
        result["frames"] = unique_counts[0] if len(unique_counts) == 1 else unique_counts
        if len(unique_counts) != 1:
            result["issues"].append(f"frame_count_mismatch:{unique_counts}")
        elif unique_counts[0] < min_frames:
            result["issues"].append(f"too_few_frames:{unique_counts[0]}_min:{min_frames}")

    if root_quat is not None and root_quat.ndim == 2 and root_quat.shape[1] == 4 and root_quat.size:
        norms = np.linalg.norm(root_quat, axis=1)
        result["quat_norm_min"] = float(norms.min())
        result["quat_norm_max"] = float(norms.max())
        if not np.allclose(norms, 1.0, atol=1e-3):
            result["issues"].append("quat_norm_not_unit")

    if root_pos is not None and root_pos.ndim == 2 and root_pos.shape[1] == 3 and len(root_pos):
        result["root_z_min"] = float(np.min(root_pos[:, 2]))
        result["root_z_max"] = float(np.max(root_pos[:, 2]))
        if len(root_pos) > 1:
            jumps = np.linalg.norm(np.diff(root_pos, axis=0), axis=1)
            result["root_jump_max"] = float(jumps.max())

    if fps is not None:
        result["fps"] = fps
        if not np.isfinite(fps) or fps <= 0:
            result["issues"].append(f"bad_fps:{fps}")

    if result["issues"]:
        fatal = [
            issue
            for issue in result["issues"]
            if issue.startswith(("load_error", "missing_", "frame_count_mismatch", "too_few_frames", "bad_fps"))
            or "_bad_shape" in issue
            or "_nonfinite" in issue
        ]
        result["status"] = "FAIL" if fatal else "WARN"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check retargeted GMR robot motion files.")
    parser.add_argument("path", type=Path, help="Motion file or directory")
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--expected-dof", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    path = args.path.resolve()
    if not path.exists():
        raise SystemExit(f"not found: {path}")

    rows = [check_file(p, args.min_frames, args.expected_dof) for p in iter_motion_files(path)]
    counts = Counter(row["status"] for row in rows)
    issue_counts = Counter(issue.split(":", 1)[0] for row in rows for issue in row["issues"])
    report = {
        "path": str(path),
        "total": len(rows),
        "status_counts": dict(counts),
        "issue_counts": dict(issue_counts.most_common()),
        "failures": [row for row in rows if row["status"] == "FAIL"][:50],
        "warnings": [row for row in rows if row["status"] == "WARN"][:50],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"path: {report['path']}")
        print(f"total: {report['total']}")
        print("status counts:")
        for status, count in counts.most_common():
            print(f"  {status}: {count}")
        if issue_counts:
            print("issue counts:")
            for issue, count in issue_counts.most_common():
                print(f"  {issue}: {count}")
        for label in ("failures", "warnings"):
            if report[label]:
                print(f"{label}:")
                for row in report[label][:10]:
                    print(f"  {row['path']}: {', '.join(row['issues'])}")
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
