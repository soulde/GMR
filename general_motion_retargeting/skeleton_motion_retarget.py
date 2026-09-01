"""Independent fixed-length skeleton-chain motion retargeter."""

from __future__ import annotations

import json
import pathlib
import warnings

import mink
import mujoco as mj
import numpy as np

from .motion_retarget import GeneralMotionRetargeting, _existing_path
from .skeleton_retarget import (
    SkeletonTargetReconstructor,
    parse_skeleton_chains,
)


def resolve_skeleton_config(src_human, tgt_robot, explicit=None) -> pathlib.Path:
    if explicit is not None:
        return _existing_path(explicit, label="skeleton retargeting config")
    from .params import SKELETON_CONFIG_DICT

    try:
        return pathlib.Path(SKELETON_CONFIG_DICT[src_human][tgt_robot])
    except KeyError as error:
        raise KeyError(
            f"no skeleton config for {src_human!r}/{tgt_robot!r}; "
            "pass skeleton_config_path"
        ) from error


class SkeletonMotionRetargeting(GeneralMotionRetargeting):
    """Retarget chain directions using explicit, fixed target lengths."""

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        skeleton_config_path: str | pathlib.Path | None = None,
        robot_xml_path: str | pathlib.Path | None = None,
        solver: str = "daqp",
        damping: float = 5e-1,
        verbose: bool = True,
        use_velocity_limit: bool = False,
    ) -> None:
        resolved_config = resolve_skeleton_config(
            src_human, tgt_robot, skeleton_config_path
        )
        with resolved_config.open() as stream:
            loaded_config = json.load(stream)
        config = loaded_config.get("config", loaded_config)
        if config.get("algorithm") != "skeleton":
            raise ValueError("skeleton config must set algorithm='skeleton'")
        chains = parse_skeleton_chains(config)
        self._validate_chain_scales(config, chains)
        solver_settings = config.get("solver_settings", {})
        self.minimum_iterations = int(
            solver_settings.get("minimum_iterations", 1)
        )
        self.maximum_iterations = int(
            solver_settings.get("maximum_iterations", 11)
        )
        self.position_error_threshold = float(
            solver_settings.get("position_error_threshold", 0.01)
        )
        if not 1 <= self.minimum_iterations <= self.maximum_iterations:
            raise ValueError(
                "solver_settings must satisfy 1 <= minimum_iterations "
                "<= maximum_iterations"
            )
        if self.position_error_threshold <= 0.0:
            raise ValueError(
                "position_error_threshold must be positive"
            )

        super().__init__(
            src_human=src_human,
            tgt_robot=tgt_robot,
            actual_human_height=actual_human_height,
            ik_config_path=resolved_config,
            robot_xml_path=robot_xml_path,
            solver=solver,
            damping=damping,
            verbose=verbose,
            use_velocity_limit=use_velocity_limit,
        )
        self.skeleton_config_path = resolved_config
        self.skeleton_chains = chains
        self.skeleton_reconstructor = SkeletonTargetReconstructor(chains)
        self.reconstructed_human_data = None
        self.segment_diagnostics = self._build_segment_diagnostics()
        self._warn_large_position_offsets(config)

    @staticmethod
    def _validate_chain_scales(config, chains) -> None:
        scale_table = config.get("human_scale_table", {})
        chain_joints = {
            joint
            for chain in chains
            for segment in chain.segments
            for joint in (segment.human_parent, segment.human_child)
        }
        for joint in sorted(chain_joints):
            scale = scale_table.get(joint)
            if scale != 1.0:
                raise ValueError(
                    f"skeleton-chain joint {joint!r} must have scale 1.0, "
                    f"got {scale!r}"
                )

    def _frame_position(self, frame_name: str) -> np.ndarray:
        frame_type = self._resolve_frame_type(frame_name)
        if frame_type == "body":
            frame_id = mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_BODY, frame_name
            )
            return self.configuration.data.xpos[frame_id].copy()
        frame_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, frame_name
        )
        return self.configuration.data.site_xpos[frame_id].copy()

    def _build_segment_diagnostics(self):
        mj.mj_forward(self.model, self.configuration.data)
        diagnostics = {}
        for chain in self.skeleton_chains:
            for segment in chain.segments:
                parent = self._frame_position(segment.robot_parent_frame)
                child = self._frame_position(segment.robot_child_frame)
                diagnostics[(segment.human_parent, segment.human_child)] = {
                    "configured_length": segment.target_length,
                    "mjcf_reference_distance": float(np.linalg.norm(child - parent)),
                }
        return diagnostics

    @staticmethod
    def _warn_large_position_offsets(config) -> None:
        warned = set()
        for table_name in ("ik_match_table1", "ik_match_table2"):
            for _, entry in config.get(table_name, {}).items():
                human_name, position_weight, rotation_weight, offset, _ = entry
                if position_weight == 0 and rotation_weight == 0:
                    continue
                magnitude = float(np.linalg.norm(np.asarray(offset, dtype=float)))
                if magnitude > 0.05 and human_name not in warned:
                    warnings.warn(
                        f"position offset for {human_name!r} is {magnitude:.6f} m, "
                        "above the 0.05 m frame-calibration threshold",
                        UserWarning,
                        stacklevel=3,
                    )
                    warned.add(human_name)

    def update_targets(self, human_data, offset_to_ground=False):
        reconstructed = self.skeleton_reconstructor.reconstruct(human_data)
        self.reconstructed_human_data = {
            name: [transform[0].copy(), transform[1].copy()]
            for name, transform in reconstructed.items()
        }

        calibrated = self.offset_human_data(
            reconstructed, self.pos_offsets1, self.rot_offsets1
        )
        table2_only = self.pos_offsets2.keys() - self.pos_offsets1.keys()
        calibrated = self.offset_human_data(
            calibrated,
            {name: self.pos_offsets2[name] for name in table2_only},
            {name: self.rot_offsets2[name] for name in table2_only},
        )
        calibrated = self.apply_ground_offset(calibrated)
        if offset_to_ground:
            calibrated = self.offset_human_data_to_ground(calibrated)
        self.scaled_human_data = calibrated

        if self.use_ik_match_table1:
            for body_name, task in self.human_body_to_task1.items():
                pos, rot = calibrated[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )
        if self.use_ik_match_table2:
            for body_name, task in self.human_body_to_task2.items():
                pos, rot = calibrated[body_name]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def _maximum_position_error(self, tasks) -> float:
        position_errors = [
            float(np.linalg.norm(task.compute_error(self.configuration)[:3]))
            for task in tasks
            if np.any(task.cost[:3] > 0.0)
        ]
        return max(position_errors, default=0.0)

    def _solve_stage(self, tasks) -> None:
        dt = self.configuration.model.opt.timestep
        for iteration in range(1, self.maximum_iterations + 1):
            velocity = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.solver,
                self.damping,
                limits=self.ik_limits,
                safety_break=False,
            )
            self.configuration.integrate_inplace(velocity, dt)
            if (
                iteration >= self.minimum_iterations
                and self._maximum_position_error(tasks)
                <= self.position_error_threshold
            ):
                break

    def retarget(self, human_data, offset_to_ground=False):
        """Solve Skeleton targets with its independent convergence settings."""
        self.update_targets(human_data, offset_to_ground)
        initializing_root = (
            self.initialize_root_from_human and not self.root_initialized
        )
        if initializing_root:
            self._initialize_root_from_human_target()
            self.root_initialized = True

        if self.use_ik_match_table1:
            self._solve_stage(self.tasks1)
        if self.use_ik_match_table2:
            self._solve_stage(self.tasks2)

        qpos = self.configuration.data.qpos.copy()
        if initializing_root:
            for _ in range(self.initialization_retargets - 1):
                qpos = self.retarget(human_data, offset_to_ground)
        return qpos
