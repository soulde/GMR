"""Pure helpers for splitting retargeted motions around invalid frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy

import numpy as np


def contiguous_valid_segments(
    frame_errors: Sequence[float],
    *,
    threshold: float,
    minimum_segment_frames: int = 1,
    padding_frames: int = 0,
) -> tuple[tuple[int, int], ...]:
    """Return half-open ranges of valid frames without bridging invalid gaps."""
    errors = np.asarray(frame_errors, dtype=float)
    if errors.ndim != 1 or not np.isfinite(errors).all():
        raise ValueError("frame_errors must be a finite one-dimensional array")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if minimum_segment_frames < 1:
        raise ValueError("minimum_segment_frames must be at least one")
    if padding_frames < 0:
        raise ValueError("padding_frames must be non-negative")

    invalid = errors > threshold
    if padding_frames and invalid.any():
        padded = invalid.copy()
        for index in np.flatnonzero(invalid):
            start = max(0, int(index) - padding_frames)
            stop = min(len(invalid), int(index) + padding_frames + 1)
            padded[start:stop] = True
        invalid = padded

    segments = []
    start = None
    for index, is_invalid in enumerate(invalid):
        if not is_invalid and start is None:
            start = index
        if is_invalid and start is not None:
            if index - start >= minimum_segment_frames:
                segments.append((start, index))
            start = None
    if start is not None and len(invalid) - start >= minimum_segment_frames:
        segments.append((start, len(invalid)))
    return tuple(segments)


def _motion_frame_count(motion: Mapping[str, object]) -> int:
    counts = []
    for name in ("root_pos", "root_rot", "dof_pos"):
        if name not in motion:
            raise ValueError(f"motion is missing {name}")
        value = np.asarray(motion[name])
        if value.ndim == 0:
            raise ValueError(f"motion field {name} has no frame dimension")
        counts.append(len(value))
    if len(set(counts)) != 1:
        raise ValueError("motion frame counts are inconsistent")
    return counts[0]


def clean_motion_segments(
    motion: Mapping[str, object],
    segments: Sequence[tuple[int, int]],
) -> tuple[dict, ...]:
    """Slice frame-aligned motion arrays into independent motion dictionaries."""
    frame_count = _motion_frame_count(motion)
    parts = []
    for start, stop in segments:
        if not 0 <= start < stop <= frame_count:
            raise ValueError(f"invalid motion segment [{start}, {stop})")
        part = {}
        for name, value in motion.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == frame_count:
                part[name] = value[start:stop].copy()
            else:
                part[name] = copy.deepcopy(value)
        parts.append(part)
    return tuple(parts)
