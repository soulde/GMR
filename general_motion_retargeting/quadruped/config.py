import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .robot_spec import LEG_ORDER


@dataclass(frozen=True)
class RootTaskConfig:
    position_cost: float
    orientation_cost: float


@dataclass(frozen=True)
class FootTaskConfig:
    position_cost: float
    position_offset: tuple[float, float, float]


@dataclass(frozen=True)
class QuadrupedRetargetConfig:
    solver: str
    damping: float
    max_iterations: int
    velocity_limit: float
    ground_height: float
    motion_center: str
    root_task: RootTaskConfig
    foot_tasks: Mapping[str, FootTaskConfig]
    root_translation_scale: tuple[float, float, float]
    trajectory_scale: Mapping[str, tuple[float, float, float]]


def _positive(value, label: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _leg_table(payload: dict, label: str) -> dict:
    missing = [leg for leg in LEG_ORDER if leg not in payload]
    extra = sorted(set(payload) - set(LEG_ORDER))
    if missing or extra:
        raise ValueError(
            f"{label} must contain exactly {', '.join(LEG_ORDER)}"
        )
    return payload


def _vector3(value, label: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{label} must contain three finite values")
    return tuple(vector)


def load_retarget_config(path: str | Path) -> QuadrupedRetargetConfig:
    config_path = Path(path)
    with config_path.open() as stream:
        payload = json.load(stream)

    ik = payload["ik"]
    tasks = payload["tasks"]
    mapping = payload["motion_mapping"]
    root = tasks["root"]
    raw_feet = _leg_table(tasks["feet"], "tasks.feet")
    raw_scale_table = mapping["scale_table"]
    if set(raw_scale_table) != {"root", *LEG_ORDER}:
        raise ValueError(
            "motion_mapping.scale_table must contain exactly "
            "root, FL, FR, RL, RR"
        )
    raw_scale = {leg: raw_scale_table[leg] for leg in LEG_ORDER}
    motion_center = str(mapping["center"])
    if motion_center not in ("temporal_median", "mjcf_default"):
        raise ValueError(f"unsupported motion center: {motion_center!r}")

    foot_tasks = {
        leg: FootTaskConfig(
            position_cost=_positive(
                raw_feet[leg]["position_cost"],
                f"tasks.feet.{leg}.position_cost",
            ),
            position_offset=_vector3(
                raw_feet[leg]["position_offset"],
                f"tasks.feet.{leg}.position_offset",
            ),
        )
        for leg in LEG_ORDER
    }
    trajectory_scale = {
        leg: _vector3(
            raw_scale[leg],
            f"motion_mapping.scale_table.{leg}",
        )
        for leg in LEG_ORDER
    }
    if any(
        value <= 0.0
        for scale in trajectory_scale.values()
        for value in scale
    ):
        raise ValueError("trajectory scale values must be positive")
    root_translation_scale = _vector3(
        raw_scale_table["root"],
        "motion_mapping.scale_table.root",
    )
    if any(value <= 0.0 for value in root_translation_scale):
        raise ValueError("root translation scale values must be positive")

    max_iterations = int(ik["max_iterations"])
    if max_iterations <= 0:
        raise ValueError("ik.max_iterations must be positive")
    ground_height = float(payload["ground_height"])
    if not np.isfinite(ground_height):
        raise ValueError("ground_height must be finite")

    return QuadrupedRetargetConfig(
        solver=str(ik["solver"]),
        damping=_positive(ik["damping"], "ik.damping"),
        max_iterations=max_iterations,
        velocity_limit=_positive(
            ik["velocity_limit"], "ik.velocity_limit"
        ),
        ground_height=ground_height,
        motion_center=motion_center,
        root_task=RootTaskConfig(
            position_cost=_positive(
                root["position_cost"], "tasks.root.position_cost"
            ),
            orientation_cost=_positive(
                root["orientation_cost"], "tasks.root.orientation_cost"
            ),
        ),
        foot_tasks=MappingProxyType(foot_tasks),
        root_translation_scale=root_translation_scale,
        trajectory_scale=MappingProxyType(trajectory_scale),
    )
