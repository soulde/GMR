from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointSpaceMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    joint_pos: np.ndarray
    joint_names: tuple[str, ...]
    loop_mode: str

    def __post_init__(self) -> None:
        frames = self.root_pos.shape[0]
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.root_pos.shape != (frames, 3):
            raise ValueError("root_pos must have shape [T, 3]")
        if self.root_rot.shape != (frames, 4):
            raise ValueError("root_rot must have shape [T, 4]")
        if self.joint_pos.shape != (frames, len(self.joint_names)):
            raise ValueError("joint_pos shape does not match joint_names")
        for name, value in (
            ("root_pos", self.root_pos),
            ("root_rot", self.root_rot),
            ("joint_pos", self.joint_pos),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class CanonicalQuadrupedMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    foot_pos_root: np.ndarray
    leg_order: tuple[str, str, str, str]
    loop_mode: str


@dataclass(frozen=True)
class FrameDiagnostics:
    frame_index: int
    iterations: int
    initial_error: float
    final_error: float
    reached_max_iterations: bool
    joint_limit_hits: tuple[str, ...]


@dataclass(frozen=True)
class QuadrupedRetargetResult:
    qpos: np.ndarray
    fps: float
    loop_mode: str
    diagnostics: tuple[FrameDiagnostics, ...]
