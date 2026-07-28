import argparse
import csv
import math
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_CONFIG = {
    "foot_min_height_fail": -0.03,
    "foot_min_height_warn": -0.01,
    "support_xy_speed_warn": 0.15,
    "support_xy_speed_fail": 0.20,
    "joint_limit_warn_ratio": 0.01,
    "joint_limit_fail_ratio": 0.10,
    "flight_ratio_fail": 0.25,
    "flight_ratio_warn": 0.10,
    "pd_speed_smoke": 0.25,
    "pd_min_survival_time": 2.0,
    "pd_max_time_smoke": 2.0,
    "pd_disable_arm_torques": True,
    "contact_height_threshold": 0.04,
    "contact_xy_speed_threshold": 0.20,
    "stand_time": 1.0,
    "blend_time": 1.0,
}


SUMMARY_FIELDS = [
    "motion_path",
    "raw_pkl",
    "preprocessed_pkl",
    "dataset_npz",
    "quality_report",
    "pd_report",
    "quality_status",
    "pd_status",
    "dataset_status",
    "final_status",
    "error",
    "joint_limit_max_violation",
    "left_foot_min_height",
    "right_foot_min_height",
    "left_support_max_xy_vel",
    "right_support_max_xy_vel",
    "flight_ratio",
    "pd_fall_time",
    "pd_max_tracking_error",
    "pd_max_torque_ratio",
    "pd_top_saturated_joint",
]


def parse_scalar(value):
    value = value.strip()
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config(path):
    cfg = dict(DEFAULT_CONFIG)
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        cfg[key.strip()] = parse_scalar(value)
    return cfg


def run(cmd, cwd, dry_run=False):
    if dry_run:
        print("[DRY]", " ".join(map(str, cmd)))
        return
    subprocess.run([str(x) for x in cmd], cwd=cwd, check=True)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def read_summary_value(path, key):
    path = Path(path)
    if not path.exists():
        return None
    prefix = f"{key}:"
    for line in path.read_text().splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def parse_float(value, default=math.nan):
    if value is None:
        return default
    if isinstance(value, float):
        return value
    if value == "not_detected":
        return math.inf
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_name(input_dir, motion_path):
    rel = motion_path.relative_to(input_dir)
    return "__".join(rel.with_suffix("").parts)


def output_paths(args, motion_path):
    input_dir = args.input_dir.resolve()
    rel = motion_path.resolve().relative_to(input_dir)
    raw_dir = args.output_dir / "raw"
    pre_dir = args.output_dir / "preprocessed"
    dataset_dir = args.output_dir / "dataset"
    if args.preserve_tree:
        rel_no_suffix = rel.with_suffix("")
        raw_pkl = raw_dir / rel_no_suffix.with_suffix(".pkl")
        pre_pkl = pre_dir / rel_no_suffix.with_name(f"{rel_no_suffix.name}_preprocessed.pkl")
        dataset_npz = dataset_dir / rel_no_suffix.with_name(f"{rel_no_suffix.name}_dataset.npz")
        report_name = "__".join(rel_no_suffix.parts)
    else:
        report_name = safe_name(input_dir, motion_path.resolve())
        raw_pkl = raw_dir / f"{report_name}.pkl"
        pre_pkl = pre_dir / f"{report_name}_preprocessed.pkl"
        dataset_npz = dataset_dir / f"{report_name}_dataset.npz"
    return report_name, raw_pkl, pre_pkl, dataset_npz


def load_excludes(path, input_dir):
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    excludes = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        candidate = Path(line)
        if candidate.is_absolute():
            try:
                excludes.add(candidate.resolve().relative_to(input_dir.resolve()).as_posix())
            except ValueError:
                excludes.add(candidate.resolve().as_posix())
        else:
            excludes.add(candidate.as_posix())
    return excludes


def is_excluded(motion_path, input_dir, excludes):
    rel = motion_path.resolve().relative_to(input_dir.resolve()).as_posix()
    return rel in excludes or motion_path.name in excludes or motion_path.stem in excludes


def classify_quality(quality_dir, cfg):
    metrics_path = quality_dir / "metrics.csv"
    joint_path = quality_dir / "joint_range.csv"
    if not metrics_path.exists() or not joint_path.exists():
        return "ERROR", {}

    metrics = read_csv(metrics_path)
    joint_rows = read_csv(joint_path)

    left_contact_rows = [row for row in metrics if int(row["left_contact"])]
    right_contact_rows = [row for row in metrics if int(row["right_contact"])]

    left_min = min(float(row["left_foot_height"]) for row in metrics)
    right_min = min(float(row["right_foot_height"]) for row in metrics)
    left_max_xy = (
        max(float(row["left_foot_xy_speed"]) for row in left_contact_rows)
        if left_contact_rows
        else math.inf
    )
    right_max_xy = (
        max(float(row["right_foot_xy_speed"]) for row in right_contact_rows)
        if right_contact_rows
        else math.inf
    )
    flight_ratio = sum(
        1 for row in metrics if not int(row["left_contact"]) and not int(row["right_contact"])
    ) / max(1, len(metrics))
    max_violation = max(float(row["violation_ratio"]) for row in joint_rows)

    status = "PASS"
    if (
        max_violation > float(cfg["joint_limit_fail_ratio"])
        or left_min < float(cfg["foot_min_height_fail"])
        or right_min < float(cfg["foot_min_height_fail"])
        or left_max_xy > float(cfg["support_xy_speed_fail"])
        or right_max_xy > float(cfg["support_xy_speed_fail"])
        or flight_ratio > float(cfg["flight_ratio_fail"])
    ):
        status = "FAIL"
    elif (
        max_violation > float(cfg["joint_limit_warn_ratio"])
        or left_min < float(cfg["foot_min_height_warn"])
        or right_min < float(cfg["foot_min_height_warn"])
        or left_max_xy > float(cfg["support_xy_speed_warn"])
        or right_max_xy > float(cfg["support_xy_speed_warn"])
        or flight_ratio > float(cfg["flight_ratio_warn"])
    ):
        status = "WARN"

    return status, {
        "joint_limit_max_violation": max_violation,
        "left_foot_min_height": left_min,
        "right_foot_min_height": right_min,
        "left_support_max_xy_vel": left_max_xy,
        "right_support_max_xy_vel": right_max_xy,
        "flight_ratio": flight_ratio,
    }


def classify_pd(pd_dir, cfg):
    summary = pd_dir / "summary.txt"
    saturation = pd_dir / "torque_saturation_summary.csv"
    if not summary.exists():
        return "SKIP", {}
    fall_time = parse_float(read_summary_value(summary, "fall_time"))
    max_tracking = parse_float(read_summary_value(summary, "max_joint_tracking_error_norm"))
    top_joint = ""
    max_ratio = math.nan
    if saturation.exists():
        rows = read_csv(saturation)
        if rows:
            top = max(rows, key=lambda r: float(r["saturation_ratio"]))
            top_joint = top["joint_name"]
            max_ratio = max(float(r["max_torque_ratio"]) for r in rows)
    status = "PASS"
    if fall_time < float(cfg["pd_min_survival_time"]):
        status = "FAIL"
    elif not math.isinf(fall_time):
        status = "WARN"
    return status, {
        "pd_fall_time": fall_time,
        "pd_max_tracking_error": max_tracking,
        "pd_max_torque_ratio": max_ratio,
        "pd_top_saturated_joint": top_joint,
    }


def final_status(quality_status, pd_status, dataset_status):
    if dataset_status == "FAIL" or quality_status == "FAIL" or pd_status == "FAIL":
        return "FAIL"
    if quality_status == "WARN" or pd_status == "WARN":
        return "WARN"
    return "PASS"


def process_motion(motion_path, args, cfg):
    cwd = Path.cwd()
    input_dir = args.input_dir.resolve()
    name, raw_pkl, pre_pkl, dataset_npz = output_paths(args, motion_path)

    quality_dir = args.reports_dir / "motion_quality" / name
    pd_dir = args.reports_dir / "pd_replay" / name

    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "motion_path": str(motion_path),
            "raw_pkl": str(raw_pkl),
            "preprocessed_pkl": str(pre_pkl),
            "dataset_npz": str(dataset_npz),
            "quality_report": str(quality_dir),
            "pd_report": str(pd_dir if args.run_pd_smoke else ""),
        }
    )

    try:
        raw_pkl.parent.mkdir(parents=True, exist_ok=True)
        pre_pkl.parent.mkdir(parents=True, exist_ok=True)
        dataset_npz.parent.mkdir(parents=True, exist_ok=True)
        args.reports_dir.mkdir(parents=True, exist_ok=True)

        if args.force or not raw_pkl.exists():
            run(
                [
                    sys.executable,
                    "scripts/smplx_to_robot.py",
                    "--smplx_file",
                    motion_path,
                    "--robot",
                    "dr02",
                    "--save_path",
                    raw_pkl,
                    "--no_viewer",
                ],
                cwd,
                args.dry_run,
            )

        if args.force or not (quality_dir / "summary.txt").exists():
            run(
                [
                    sys.executable,
                    "scripts/check_dr02_motion_quality.py",
                    "--motion",
                    raw_pkl,
                    "--xml",
                    args.xml,
                    "--out",
                    quality_dir,
                    "--contact-height-threshold",
                    cfg["contact_height_threshold"],
                    "--contact-xy-speed-threshold",
                    cfg["contact_xy_speed_threshold"],
                ],
                cwd,
                args.dry_run,
            )
        quality_status, quality_values = classify_quality(quality_dir, cfg) if not args.dry_run else ("DRY", {})
        row.update(quality_values)
        row["quality_status"] = quality_status

        if args.force or not pre_pkl.exists():
            preprocess_cmd = [
                sys.executable,
                "scripts/preprocess_dr02_motion.py",
                "--motion",
                raw_pkl,
                "--xml",
                args.xml,
                "--out",
                pre_pkl,
                "--stand-time",
                cfg["stand_time"],
                "--blend-time",
                cfg["blend_time"],
            ]
            if args.auto_start_frame:
                preprocess_cmd.append("--auto-start-frame")
            run(preprocess_cmd, cwd, args.dry_run)

        if args.force or not dataset_npz.exists():
            dataset_cmd = [
                sys.executable,
                "scripts/preprocess_dr02_motion.py",
                "--motion",
                raw_pkl,
                "--xml",
                args.xml,
                "--out",
                dataset_npz,
                "--stand-time",
                cfg["stand_time"],
                "--blend-time",
                cfg["blend_time"],
                "--export-dataset",
            ]
            if args.auto_start_frame:
                dataset_cmd.append("--auto-start-frame")
            run(dataset_cmd, cwd, args.dry_run)

        try:
            if not args.dry_run:
                run([sys.executable, "scripts/check_dr02_motion_dataset.py", "--dataset", dataset_npz], cwd)
            row["dataset_status"] = "DRY" if args.dry_run else "PASS"
        except subprocess.CalledProcessError:
            row["dataset_status"] = "FAIL"

        if args.run_pd_smoke and (args.force or not (pd_dir / "summary.txt").exists()):
            pd_cmd = [
                sys.executable,
                "scripts/pd_replay_dr02_motion.py",
                "--motion",
                pre_pkl,
                "--xml",
                args.xml,
                "--out",
                pd_dir,
                "--speed",
                cfg["pd_speed_smoke"],
                "--max-time",
                cfg["pd_max_time_smoke"],
                "--disable-viewer",
            ]
            if bool(cfg["pd_disable_arm_torques"]):
                pd_cmd.append("--disable-arm-torques")
            run(pd_cmd, cwd, args.dry_run)

        pd_status, pd_values = classify_pd(pd_dir, cfg) if args.run_pd_smoke and not args.dry_run else ("SKIP", {})
        if args.dry_run and args.run_pd_smoke:
            pd_status = "DRY"
        row.update(pd_values)
        row["pd_status"] = pd_status
        row["final_status"] = (
            "DRY"
            if args.dry_run
            else final_status(row["quality_status"], row["pd_status"], row["dataset_status"])
        )
    except Exception as exc:
        row["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        row["quality_status"] = row["quality_status"] or "ERROR"
        row["dataset_status"] = row["dataset_status"] or "ERROR"
        row["pd_status"] = row["pd_status"] or "ERROR"
        row["final_status"] = "FAIL"
    return row


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("retargeting_data/dr02"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/dr02_batch"))
    parser.add_argument("--xml", type=Path, default=Path("assets/robots/dr02/dr02.xml"))
    parser.add_argument("--config", type=Path, default=Path("configs/dr02_motion_quality.yaml"))
    parser.add_argument("--pattern", type=str, default="*stageii.npz")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exclude-file", type=Path, default=None)
    parser.add_argument("--preserve-tree", action="store_true")
    parser.add_argument("--run-pd-smoke", action="store_true")
    parser.add_argument("--auto-start-frame", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.reports_dir = args.reports_dir.resolve()
    cfg = load_config(args.config)

    excludes = load_excludes(args.exclude_file, args.input_dir)
    motions = [
        motion
        for motion in sorted(args.input_dir.rglob(args.pattern))
        if not is_excluded(motion, args.input_dir, excludes)
    ]
    if args.limit is not None:
        motions = motions[: args.limit]
    if not motions:
        raise FileNotFoundError(f"No motions found under {args.input_dir} with pattern {args.pattern}")

    rows = []
    if args.jobs <= 1:
        for motion in motions:
            print(f"[{len(rows)+1}/{len(motions)}] {motion}")
            rows.append(process_motion(motion, args, cfg))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(process_motion, motion, args, cfg): motion for motion in motions}
            for i, future in enumerate(as_completed(futures), 1):
                motion = futures[future]
                print(f"[{i}/{len(motions)}] done {motion}")
                rows.append(future.result())

    summary_path = args.reports_dir / "batch_summary.csv"
    write_summary_csv(summary_path, rows)
    counts = {}
    for row in rows:
        counts[row["final_status"]] = counts.get(row["final_status"], 0) + 1
    print(f"Wrote batch summary to {summary_path}")
    print("Final status counts:", counts)


if __name__ == "__main__":
    main()
