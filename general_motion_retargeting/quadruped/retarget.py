from dataclasses import replace

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .joint_mapping import map_initial_configuration
from .kinematics import source_forward_kinematics
from .morphology import scale_foot_trajectories
from .robot_spec import QuadrupedRobotSpec
from .types import (
    FrameDiagnostics,
    JointSpaceMotion,
    QuadrupedRetargetResult,
)


class QuadrupedRobotRetargeter:
    def __init__(
        self,
        source_spec: QuadrupedRobotSpec,
        target_spec: QuadrupedRobotSpec,
        solver: str = "daqp",
        damping: float = 0.5,
        max_iterations: int = 10,
        use_velocity_limit: bool = False,
        velocity_limit: float = 3.0 * np.pi,
        root_position_cost: float = 100.0,
        root_orientation_cost: float = 10.0,
        foot_position_cost: float = 100.0,
    ) -> None:
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.solver = solver
        self.damping = damping
        self.max_iterations = max_iterations
        self.use_velocity_limit = use_velocity_limit
        self.velocity_limit = velocity_limit
        self.configuration = mink.Configuration(target_spec.model)

        self.root_task = mink.FrameTask(
            frame_name=target_spec.root_body,
            frame_type="body",
            position_cost=root_position_cost,
            orientation_cost=root_orientation_cost,
            lm_damping=1.0,
        )
        self.foot_tasks = {
            leg: mink.FrameTask(
                frame_name=target_spec.legs[leg].foot_site,
                frame_type="site",
                position_cost=foot_position_cost,
                orientation_cost=0.0,
                lm_damping=1.0,
            )
            for leg in target_spec.leg_order
        }
        self.tasks = [self.root_task, *self.foot_tasks.values()]
        self.limits = [mink.ConfigurationLimit(target_spec.model)]
        if use_velocity_limit:
            self.limits.append(
                mink.VelocityLimit(
                    target_spec.model,
                    {
                        name: velocity_limit
                        for name in target_spec.motion_joint_order
                    },
                )
            )

    def _task_error(self) -> float:
        errors = [
            task.compute_error(self.configuration)[task.cost > 0]
            for task in self.tasks
        ]
        return float(np.linalg.norm(np.concatenate(errors)))

    def _set_targets(
        self,
        root_pos: np.ndarray,
        root_rot: np.ndarray,
        foot_pos_root: np.ndarray,
    ) -> None:
        root_rotation = Rotation.from_quat(
            root_rot, scalar_first=True
        )
        self.root_task.set_target(
            mink.SE3.from_rotation_and_translation(
                mink.SO3(root_rot), root_pos
            )
        )
        identity = mink.SO3.identity()
        for leg_index, leg in enumerate(self.target_spec.leg_order):
            foot_world = root_pos + root_rotation.apply(
                foot_pos_root[leg_index]
            )
            self.foot_tasks[leg].set_target(
                mink.SE3.from_rotation_and_translation(identity, foot_world)
            )

    def _joint_limit_hits(self, qpos: np.ndarray) -> tuple[str, ...]:
        hits = []
        model = self.target_spec.model
        for name in self.target_spec.motion_joint_order:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if not model.jnt_limited[joint_id]:
                continue
            value = qpos[model.jnt_qposadr[joint_id]]
            lower, upper = model.jnt_range[joint_id]
            if np.isclose(value, lower, atol=1e-6) or np.isclose(
                value, upper, atol=1e-6
            ):
                hits.append(name)
        return tuple(hits)

    def _solve_frame(self, frame_index: int) -> FrameDiagnostics:
        initial_error = self._task_error()
        best_error = initial_error
        best_qpos = self.configuration.q.copy()
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            velocity = mink.solve_ik(
                self.configuration,
                self.tasks,
                self.configuration.model.opt.timestep,
                self.solver,
                self.damping,
                self.limits,
            )
            if not np.isfinite(velocity).all():
                raise ValueError(
                    f"IK returned non-finite velocity at frame {frame_index}"
                )
            self.configuration.integrate_inplace(
                velocity, self.configuration.model.opt.timestep
            )
            error = self._task_error()
            if error < best_error:
                improvement = best_error - error
                best_error = error
                best_qpos = self.configuration.q.copy()
                if improvement <= 1e-3:
                    break
            else:
                break

        self.configuration.update(best_qpos)
        return FrameDiagnostics(
            frame_index=frame_index,
            iterations=iterations,
            initial_error=initial_error,
            final_error=best_error,
            reached_max_iterations=iterations == self.max_iterations,
            joint_limit_hits=self._joint_limit_hits(best_qpos),
        )

    def _apply_ground_offset(self, qpos: np.ndarray) -> None:
        model = self.target_spec.model
        data = mujoco.MjData(model)
        site_ids = [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                self.target_spec.legs[leg].foot_site,
            )
            for leg in self.target_spec.leg_order
        ]
        lowest = np.inf
        for frame in qpos:
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            lowest = min(lowest, *(data.site_xpos[site_ids, 2]))
        free_joint = int(
            np.flatnonzero(
                model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
            )[0]
        )
        root_qpos_address = model.jnt_qposadr[free_joint]
        qpos[:, root_qpos_address + 2] -= lowest

    def _enforce_frame_velocity(
        self, previous_qpos: np.ndarray, fps: float
    ) -> None:
        max_step = self.velocity_limit / fps
        qpos = self.configuration.q.copy()
        for joint_name in self.target_spec.motion_joint_order:
            joint_id = mujoco.mj_name2id(
                self.target_spec.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            address = self.target_spec.model.jnt_qposadr[joint_id]
            qpos[address] = np.clip(
                qpos[address],
                previous_qpos[address] - max_step,
                previous_qpos[address] + max_step,
            )
        self.configuration.update(qpos)

    def retarget_motion(
        self,
        motion: JointSpaceMotion,
    ) -> QuadrupedRetargetResult:
        canonical = source_forward_kinematics(self.source_spec, motion)
        scaled = scale_foot_trajectories(
            canonical, self.source_spec, self.target_spec
        )
        initial_qpos = map_initial_configuration(
            motion.joint_pos[0], self.source_spec, self.target_spec
        )
        self.configuration.update(initial_qpos)

        qpos = np.empty(
            (len(motion.root_pos), self.target_spec.model.nq), dtype=float
        )
        diagnostics = []
        for frame_index in range(len(motion.root_pos)):
            if frame_index == 0:
                mapped = map_initial_configuration(
                    motion.joint_pos[0],
                    self.source_spec,
                    self.target_spec,
                )
                mapped[:7] = np.concatenate(
                    [motion.root_pos[0], motion.root_rot[0]]
                )
                self.configuration.update(mapped)
            self._set_targets(
                motion.root_pos[frame_index],
                motion.root_rot[frame_index],
                scaled.foot_pos_root[frame_index],
            )
            diagnostic = self._solve_frame(frame_index)
            if self.use_velocity_limit and frame_index > 0:
                self._enforce_frame_velocity(qpos[frame_index - 1], motion.fps)
                diagnostic = replace(
                    diagnostic,
                    final_error=self._task_error(),
                    joint_limit_hits=self._joint_limit_hits(
                        self.configuration.q
                    ),
                )
            diagnostics.append(diagnostic)
            qpos[frame_index] = self.configuration.q

        if not np.isfinite(qpos).all():
            raise ValueError("retargeted qpos contains non-finite values")
        self._apply_ground_offset(qpos)
        return QuadrupedRetargetResult(
            qpos=qpos,
            fps=motion.fps,
            loop_mode=motion.loop_mode,
            diagnostics=tuple(diagnostics),
        )
