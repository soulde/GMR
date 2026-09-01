"""Fixed-length, chain-structured target reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class SkeletonSegment:
    human_parent: str
    human_child: str
    robot_parent_frame: str
    robot_child_frame: str
    target_length: float


@dataclass(frozen=True)
class SkeletonChain:
    name: str
    segments: tuple[SkeletonSegment, ...]


def _required_string(entry: Mapping[str, object], key: str, chain_name: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"skeleton chain {chain_name!r} requires non-empty {key!r}")
    return value


def parse_skeleton_chains(config: Mapping[str, object]) -> tuple[SkeletonChain, ...]:
    """Parse and validate ordered, connected skeleton-chain segments."""
    raw_chains = config.get("skeleton_chains")
    if not isinstance(raw_chains, list) or not raw_chains:
        raise ValueError("skeleton_chains must be a non-empty list")

    chains: list[SkeletonChain] = []
    chain_names: set[str] = set()
    child_owners: dict[str, str] = {}
    for raw_chain in raw_chains:
        if not isinstance(raw_chain, Mapping):
            raise ValueError("each skeleton chain must be an object")
        name = _required_string(raw_chain, "name", "<unnamed>")
        if name in chain_names:
            raise ValueError(f"duplicate skeleton chain name {name!r}")
        chain_names.add(name)

        raw_segments = raw_chain.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError(f"skeleton chain {name!r} must contain segments")

        segments: list[SkeletonSegment] = []
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, Mapping):
                raise ValueError(f"skeleton chain {name!r} segment {index} must be an object")
            parent = _required_string(raw_segment, "human_parent", name)
            child = _required_string(raw_segment, "human_child", name)
            robot_parent = _required_string(raw_segment, "robot_parent_frame", name)
            robot_child = _required_string(raw_segment, "robot_child_frame", name)
            length = raw_segment.get("target_length")
            if (
                isinstance(length, bool)
                or not isinstance(length, (int, float))
                or not math.isfinite(float(length))
                or float(length) <= 0.0
            ):
                raise ValueError(
                    f"skeleton chain {name!r} segment {parent!r}->{child!r} "
                    "has invalid target_length"
                )
            if segments and segments[-1].human_child != parent:
                raise ValueError(
                    f"skeleton chain {name!r} is disconnected between "
                    f"{segments[-1].human_child!r} and {parent!r}"
                )
            if child in child_owners:
                raise ValueError(
                    f"human child {child!r} is owned by more than one segment: "
                    f"{child_owners[child]!r} and {name!r}"
                )
            child_owners[child] = name
            segments.append(
                SkeletonSegment(
                    human_parent=parent,
                    human_child=child,
                    robot_parent_frame=robot_parent,
                    robot_child_frame=robot_child,
                    target_length=float(length),
                )
            )
        chains.append(SkeletonChain(name=name, segments=tuple(segments)))
    return tuple(chains)


class SkeletonTargetReconstructor:
    """Rebuild source joint positions using fixed configured segment lengths."""

    def __init__(
        self,
        chains: Sequence[SkeletonChain],
        *,
        epsilon: float = 1e-8,
    ) -> None:
        if not chains:
            raise ValueError("at least one skeleton chain is required")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")
        self.chains = tuple(chains)
        self.epsilon = float(epsilon)
        self._previous_directions: dict[tuple[str, str], np.ndarray] = {}

    @staticmethod
    def _copy_frame(human_data: Mapping[str, Sequence[object]]):
        result = {}
        for name, transform in human_data.items():
            if len(transform) != 2:
                raise ValueError(f"human joint {name!r} must contain position and rotation")
            position = np.asarray(transform[0], dtype=float).copy()
            rotation = np.asarray(transform[1], dtype=float).copy()
            if position.shape != (3,) or rotation.shape != (4,):
                raise ValueError(f"human joint {name!r} has malformed transform")
            if not np.isfinite(position).all() or not np.isfinite(rotation).all():
                raise ValueError(f"human joint {name!r} contains non-finite values")
            result[name] = [position, rotation]
        return result

    def _source_position(self, human_data, chain_name: str, joint_name: str):
        try:
            transform = human_data[joint_name]
        except KeyError:
            raise KeyError(
                f"skeleton chain {chain_name!r} requires missing human joint {joint_name!r}"
            ) from None
        position = np.asarray(transform[0], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                f"skeleton chain {chain_name!r} human joint {joint_name!r} "
                "has invalid position"
            )
        return position

    def _resolve_direction(self, chain_name: str, segment, delta: np.ndarray):
        norm = float(np.linalg.norm(delta))
        key = (segment.human_parent, segment.human_child)
        if norm >= self.epsilon:
            direction = delta / norm
            self._previous_directions[key] = direction.copy()
            return direction
        if key in self._previous_directions:
            return self._previous_directions[key]
        raise ValueError(
            f"skeleton chain {chain_name!r} segment "
            f"{segment.human_parent!r}->{segment.human_child!r} is degenerate "
            "and has no previous direction"
        )

    def reconstruct(self, human_data: Mapping[str, Sequence[object]]):
        result = self._copy_frame(human_data)
        for chain in self.chains:
            for segment in chain.segments:
                parent_source = self._source_position(
                    human_data, chain.name, segment.human_parent
                )
                child_source = self._source_position(
                    human_data, chain.name, segment.human_child
                )
                if segment.human_parent not in result:
                    raise KeyError(
                        f"skeleton chain {chain.name!r} requires unavailable "
                        f"reconstructed parent {segment.human_parent!r}"
                    )
                direction = self._resolve_direction(
                    chain.name, segment, child_source - parent_source
                )
                result[segment.human_child][0] = (
                    result[segment.human_parent][0]
                    + segment.target_length * direction
                )
        return result
