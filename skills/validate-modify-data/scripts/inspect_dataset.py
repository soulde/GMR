#!/usr/bin/env python3
"""Read-only dataset inventory for the GMR workspace."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def human_size(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def iter_files(root: Path, limit: int | None = None) -> Iterable[Path]:
    count = 0
    for path in root.rglob("*"):
        if path.is_file():
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def sample_npz(path: Path) -> dict[str, object]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on local env
        return {"error": f"numpy unavailable: {exc}"}

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        return {"error": f"failed to load: {exc}"}

    result: dict[str, object] = {}
    for key in data.files:
        value = data[key]
        shape = tuple(int(x) for x in getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", "unknown"))
        item: dict[str, object] = {"shape": shape, "dtype": dtype}
        if shape == ():
            try:
                scalar = value.item()
                if isinstance(scalar, (str, int, float, bool)) or scalar is None:
                    item["value"] = scalar
            except Exception:
                pass
        elif dtype.startswith("float") or dtype.startswith("int"):
            finite = value.size == 0 or bool(np.isfinite(value).all())
            item["finite"] = finite
            if value.size and value.ndim >= 1:
                item["frames"] = int(shape[0])
        result[key] = item
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only inventory for local datasets.")
    parser.add_argument("root", type=Path, help="Dataset root to inspect")
    parser.add_argument("--sample-limit", type=int, default=8, help="Files per extension to sample")
    parser.add_argument("--scan-limit", type=int, default=None, help="Optional max files to scan")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    ext_counts: Counter[str] = Counter()
    top_counts: Counter[str] = Counter()
    total_size = 0
    samples: dict[str, list[str]] = defaultdict(list)
    largest: list[tuple[int, str]] = []

    for file_path in iter_files(root, args.scan_limit):
        rel = file_path.relative_to(root)
        ext = file_path.suffix.lower() or "<no-ext>"
        size = file_path.stat().st_size
        ext_counts[ext] += 1
        top_counts[rel.parts[0] if rel.parts else "."] += 1
        total_size += size
        if len(samples[ext]) < args.sample_limit:
            samples[ext].append(str(rel))
        largest.append((size, str(rel)))
        largest.sort(reverse=True)
        del largest[20:]

    npz_details = {}
    for sample in samples.get(".npz", [])[: min(3, args.sample_limit)]:
        npz_details[sample] = sample_npz(root / sample)

    report = {
        "root": str(root),
        "total_size": total_size,
        "total_size_human": human_size(total_size),
        "extension_counts": dict(ext_counts.most_common()),
        "top_level_file_counts": dict(top_counts.most_common(30)),
        "samples_by_extension": dict(samples),
        "largest_files": [{"size": size, "size_human": human_size(size), "path": path} for size, path in largest],
        "npz_sample_details": npz_details,
        "scan_limited": args.scan_limit is not None,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"root: {report['root']}")
    print(f"total size: {report['total_size_human']}")
    print("\nextensions:")
    for ext, count in ext_counts.most_common():
        print(f"  {ext}: {count}")
    print("\ntop-level file counts:")
    for name, count in top_counts.most_common(30):
        print(f"  {name}: {count}")
    print("\nsamples:")
    for ext, paths in samples.items():
        print(f"  {ext}:")
        for path in paths:
            print(f"    {path}")
    print("\nlargest files:")
    for size, path in largest[:10]:
        print(f"  {human_size(size)}  {path}")
    if npz_details:
        print("\nnpz sample details:")
        for path, details in npz_details.items():
            print(f"  {path}:")
            if "error" in details:
                print(f"    error: {details['error']}")
                continue
            for key, meta in details.items():
                shape = meta.get("shape")
                dtype = meta.get("dtype")
                finite = meta.get("finite")
                suffix = "" if finite is None else f", finite={finite}"
                print(f"    {key}: shape={shape}, dtype={dtype}{suffix}")
    if args.scan_limit is not None:
        print(f"\nscan limited to {args.scan_limit} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
