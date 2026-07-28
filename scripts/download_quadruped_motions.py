#!/usr/bin/env python3
"""Download quadruped reference motions from pinned upstream revisions."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REVISION = "d0e7b963c5a301984352d25a3ee0820266fa4218"
UPSTREAM_BASE_URL = (
    "https://raw.githubusercontent.com/erwincoumans/motion_imitation/"
    f"{UPSTREAM_REVISION}/motion_imitation/data/motions"
)
MOTION_HASHES = {
    "dog_backwards_pace.txt": "6f585f8f252d74e4744dfc656c4b5e1d793ad180a0101ba77641ec93bd04ebde",
    "dog_backwards_trot.txt": "b8cdc28bf7ed18825611ef316a2d2c8bd862b6367b7906293abcbe5ad2892b42",
    "dog_pace.txt": "2ebff49cc4991eae4ddd9f8b37514956ffcb7dc194a16dba15c5240857e6ed66",
    "dog_spin.txt": "fce7b861b34103ddffccd0fc31bd078d58ad0240a2740f56780d46a93308fe1f",
    "dog_trot.txt": "ec304dbbc4e5bfbfc4955074e27a7e0b3038485952ea9847e9ca42594c9e87a6",
    "hopturn.txt": "6be7f392fc6be81f919d0d56cbef3607cfbacb0e481ce6bf50719fc132e9f2a6",
    "inplace_steps.txt": "a9900d6617d22f7a41e314eb9fab204571999468f4a0932c8caf000cf5535454",
    "runningman.txt": "749e8bbd884ce941c2acd25287efc5be2a22ca97844abe93061b4babfe105fea",
    "sidesteps.txt": "4ec842bceb11df0c887c6ca17e0f5ec29cc9cd37c2544fb67949f092fd7cea4f",
    "turn.txt": "da0f6b04bc026f350f9f7ab2bdb50dea3d7fe9ed9d0c8f5ad492d615a5109efd",
}
MOTIONS = {
    Path(filename).stem: {
        "url": f"{UPSTREAM_BASE_URL}/{filename}",
        "sha256": sha256,
        "path": REPO_ROOT / "assets/quadrupeds/motions" / filename,
    }
    for filename, sha256 in MOTION_HASHES.items()
}


def download(name: str, force: bool = False) -> Path:
    spec = MOTIONS[name]
    destination = Path(spec["path"])
    expected_hash = str(spec["sha256"])

    if destination.exists() and not force:
        current_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        if current_hash == expected_hash:
            print(f"{destination.relative_to(REPO_ROOT)} is already up to date")
            return destination
        raise RuntimeError(
            f"{destination} exists with an unexpected checksum; "
            "rerun with --force to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(str(spec["url"])) as response:
        payload = response.read()

    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
        )

    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as temporary_file:
            temporary_file.write(payload)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    print(f"downloaded {name} to {destination.relative_to(REPO_ROOT)}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download pinned quadruped reference motions."
    )
    parser.add_argument(
        "motions",
        nargs="*",
        metavar="MOTION",
        help=(
            "motions to download; downloads all motions when omitted "
            f"(available: {', '.join(sorted(MOTIONS))})"
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing motion files"
    )
    args = parser.parse_args()

    requested_motions = args.motions or sorted(MOTIONS)
    unknown_motions = sorted(set(requested_motions) - MOTIONS.keys())
    if unknown_motions:
        parser.error(f"unknown motion: {', '.join(unknown_motions)}")

    for motion in requested_motions:
        download(motion, force=args.force)


if __name__ == "__main__":
    main()
