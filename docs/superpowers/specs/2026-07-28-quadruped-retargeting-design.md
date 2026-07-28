# Quadruped Robot-to-Robot Retargeting Design

## Objective

Extend GMR with a quadruped robot-to-robot retargeting path on the
`dev/quadruped` branch. The first supported input is a
`motion_imitation` JSON-formatted `.txt` motion and its source robot MJCF.
The first target is Unitree Go2. Wheeled-legged robots are out of scope.

The implementation must preserve the existing humanoid retargeting behavior.
Quadruped support is selected explicitly with `model_type="quadruped"`;
existing callers continue to use the humanoid path by default.

## Scope

Phase 1 includes:

- Source and target robot models in MJCF.
- A small YAML semantic specification for each quadruped robot.
- Parsing `motion_imitation` root pose and source joint positions.
- Source-model forward kinematics in MuJoCo.
- Morphology-aware transfer of trunk-relative foot trajectories.
- A target joint mapping used as the Go2 IK initial state.
- Go2 trunk and foot `mink.FrameTask` targets.
- `mink.solve_ik`, `mink.ConfigurationLimit`, and optional
  `mink.VelocityLimit`.
- The same ground-offset capability as the current GMR main path.
- Existing GMR-compatible robot motion output and MuJoCo visualization.

Phase 1 does not include:

- URDF loading or URDF-to-MJCF conversion.
- Contact inference or contact-aware optimization.
- Support-foot anchoring or foot-slip correction.
- Dynamics, torque, or stability optimization.
- Wheeled-legged or non-quadruped robots.
- Automatic semantic inference from body or joint names.

## Architecture

The humanoid `GeneralMotionRetargeting` algorithm remains unchanged.
Quadruped support is implemented by a separate
`QuadrupedRobotRetargeter`. A public factory selects the implementation:

```python
def create_retargeter(
    model_type: str = "humanoid",
    **kwargs,
) -> GeneralMotionRetargeting | QuadrupedRobotRetargeter:
    ...
```

The quadruped data flow is:

```text
motion_imitation txt + source MJCF + source YAML
    -> JointSpaceMotion
    -> source MuJoCo forward kinematics
    -> CanonicalQuadrupedMotion
    -> morphology scaling
    -> Go2 semantic joint-map initial state
    -> Go2 Mink trunk and foot IK
    -> limits, optional velocity limits, and ground offset
    -> GMR-compatible RobotMotion
```

All kinematic models are loaded with:

```python
mujoco.MjModel.from_xml_path(str(mjcf_path))
```

This keeps the implementation on GMR's existing MuJoCo and Mink stack.

## Package Layout

Create the following focused modules:

```text
general_motion_retargeting/quadruped/
├── __init__.py
├── types.py
├── robot_spec.py
├── kinematics.py
├── morphology.py
├── joint_mapping.py
├── retarget.py
└── loaders/
    ├── __init__.py
    └── motion_imitation.py
```

Responsibilities:

- `types.py`: immutable motion and diagnostic data structures.
- `robot_spec.py`: YAML loading, semantic validation, and MuJoCo name lookup.
- `kinematics.py`: source MJCF state assignment and batched foot FK.
- `morphology.py`: trunk-relative foot trajectory normalization and scaling.
- `joint_mapping.py`: semantic source-to-target joint initial-state mapping.
- `retarget.py`: Go2 Mink task setup, frame solving, limits, and diagnostics.
- `loaders/motion_imitation.py`: input format parsing and normalization.

Robot configuration and assets are organized as:

```text
general_motion_retargeting/quadruped/configs/
├── laikago.yaml
└── unitree_go2.yaml

assets/quadrupeds/
├── laikago/
│   └── laikago.xml
└── unitree_go2/
    └── go2.xml
```

Asset licenses and origins must be documented next to each imported model.

## Motion Types

The parsed source motion retains its original joint-space representation:

```python
@dataclass(frozen=True)
class JointSpaceMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    joint_pos: np.ndarray
    joint_names: tuple[str, ...]
    loop_mode: str
```

Shape requirements are:

- `root_pos`: `[T, 3]`.
- `root_rot`: `[T, 4]`, normalized scalar-first `wxyz`.
- `joint_pos`: `[T, J]`.
- `len(joint_names) == J`.
- All arrays have finite values and the same frame count.

Source FK produces the robot-independent quadruped representation:

```python
@dataclass(frozen=True)
class CanonicalQuadrupedMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    foot_pos_root: np.ndarray
    leg_order: tuple[str, str, str, str]
    loop_mode: str
```

`foot_pos_root` has shape `[T, 4, 3]`; `leg_order` is always
`("FL", "FR", "RL", "RR")`. Foot targets are positions only in Phase 1.

The saved result remains compatible with existing GMR tooling:

```python
{
    "root_pos": np.ndarray,
    "root_rot": np.ndarray,
    "dof_pos": np.ndarray,
    "fps": float,
}
```

Saved `root_rot` follows the existing output convention. Conversion between
MuJoCo `wxyz` and saved `xyzw` must occur at the existing serialization
boundary rather than inside the retargeter.

## Robot Semantic Specification

MJCF defines kinematics; YAML defines semantics that cannot be inferred
reliably. Each source and target specification contains:

```yaml
robot: laikago
model_type: quadruped
mjcf_path: assets/quadrupeds/laikago/laikago.xml
root_body: trunk

legs:
  FL:
    joints: [FL_hip_joint, FL_thigh_joint, FL_calf_joint]
    foot_body: FL_foot
    foot_site: FL_foot_site
  FR:
    joints: [FR_hip_joint, FR_thigh_joint, FR_calf_joint]
    foot_body: FR_foot
    foot_site: FR_foot_site
  RL:
    joints: [RL_hip_joint, RL_thigh_joint, RL_calf_joint]
    foot_body: RL_foot
    foot_site: RL_foot_site
  RR:
    joints: [RR_hip_joint, RR_thigh_joint, RR_calf_joint]
    foot_body: RR_foot
    foot_site: RR_foot_site

motion:
  joint_order:
    - FR_hip_joint
    - FR_thigh_joint
    - FR_calf_joint
    - FL_hip_joint
    - FL_thigh_joint
    - FL_calf_joint
    - RR_hip_joint
    - RR_thigh_joint
    - RR_calf_joint
    - RL_hip_joint
    - RL_thigh_joint
    - RL_calf_joint
  quaternion_order: xyzw

reference_pose:
  FL_hip_joint: 0.0
  FL_thigh_joint: 0.8
  FL_calf_joint: -1.5
```

Exact names and source file order are populated from the selected MJCF and
motion file during implementation. They are not guessed from this example.

The Go2 specification additionally provides the semantic joint-map
coefficients:

```yaml
joint_mapping:
  FL:
    hip: {sign: 1.0, scale: 1.0}
    thigh: {sign: 1.0, scale: 1.0}
    calf: {sign: 1.0, scale: 1.0}
```

The mapping operates around source and target reference poses:

```python
q_target = target_reference + sign * scale * (
    q_source - source_reference
)
```

It supplies an IK initial state, not the final retargeted result.

## Source Motion Parsing

The `motion_imitation` loader reads the JSON-formatted `.txt` fields:

- `LoopMode`.
- `FrameDuration`.
- `Frames`.

Each frame is split according to the source specification into root
translation, root quaternion, and source joint positions. The loader:

- Converts quaternion order to internal `wxyz`.
- Normalizes quaternions and rejects zero-norm rotations.
- Computes `fps = 1.0 / FrameDuration`.
- Validates frame width against `motion.joint_order`.
- Rejects empty motions and non-finite values.
- Preserves the declared loop mode without adding loop optimization.

The loader does not treat the biological dog skeleton as an intermediate
model. The txt file is interpreted as joint-space motion for its declared
source robot MJCF.

## Source Forward Kinematics

For each frame:

1. Assign source root pose and named source joint positions to `MjData.qpos`.
2. Call `mujoco.mj_forward`.
3. Read each configured foot site position.
4. Transform each foot position from world coordinates into the source root
   body frame.

The qpos address for every joint is obtained from MuJoCo model metadata.
Array offsets are never assumed. The source root can be free or fixed, but
the specification and motion must agree.

## Morphology Transfer

Morphology transfer scales trunk-relative foot trajectories rather than
copying source joint angles:

- Longitudinal coordinates use source and target front/rear hip spacing and
  nominal leg reach.
- Lateral coordinates use source and target left/right hip spacing.
- Vertical coordinates use source and target nominal leg reach.
- Source and target reference poses define the neutral foot positions.
- Deviations from the neutral feet are scaled separately from neutral stance.

The transfer preserves:

- FL/FR/RL/RR phase relationships.
- Relative step length.
- Relative stance width.
- Relative foot lift.

It does not infer contact or pin support feet. Unreachable targets are handled
by the target IK diagnostics and failure policy.

## Target Go2 IK

The Go2 retargeter creates:

- One trunk `mink.FrameTask` for root position and orientation.
- Four foot `mink.FrameTask` instances for position only.
- `mink.ConfigurationLimit`.
- Optional `mink.VelocityLimit` using configured Go2 motor limits.

Each frame:

1. Initialize from the semantic joint map for the first frame.
2. Use the previous solved Go2 qpos for subsequent frames.
3. Set the trunk and four scaled foot targets.
4. Call `mink.solve_ik`.
5. Integrate the solved velocity.
6. Repeat until the error improvement threshold or maximum iteration count.
7. Retain the lowest-error configuration encountered.

The solver remains configurable, with GMR-compatible defaults for solver and
damping. Phase 1 does not add a trajectory-wide optimizer.

## Ground Offset

Ground handling matches the current GMR main flow:

- Determine the lowest configured foot point.
- Apply a vertical offset to the root or all frame targets.
- Preserve relative body and foot positions.

No contact state is inferred. No foot is anchored in world coordinates, and
no support-foot sliding correction is performed.

## Validation and Failure Policy

Model and configuration validation fails before retargeting when:

- Either MJCF cannot be compiled.
- `model_type` is not `quadruped`.
- Any FL/FR/RL/RR leg is missing.
- A leg does not declare exactly three supported hinge joints.
- A configured joint, body, or site does not exist.
- A joint is assigned to more than one leg.
- Motion width differs from the configured source joint order.
- Reference positions violate model joint limits.
- Source motion root layout conflicts with the MJCF root joint.

Per-frame diagnostics are represented as:

```python
@dataclass(frozen=True)
class FrameDiagnostics:
    frame_index: int
    iterations: int
    initial_error: float
    final_error: float
    reached_max_iterations: bool
    joint_limit_hits: tuple[str, ...]
```

NaN or infinite input/output fails immediately. A non-converged frame retains
its best finite, in-limit configuration and is reported. The complete run
fails when the configured maximum non-converged frame ratio is exceeded.

The final summary includes:

- Mean and maximum foot-position error.
- Mean and maximum trunk pose error.
- Non-converged frame count and ratio.
- Joint-limit hit counts.
- Maximum joint velocity.
- Minimum and maximum foot height.

## CLI and Integration

Add a dedicated first-stage CLI:

```bash
python scripts/motion_imitation_to_robot.py \
  --model_type quadruped \
  --motion_file motion_imitation/data/motions/dog_pace.txt \
  --source_robot laikago \
  --robot unitree_go2 \
  --save_path retargeting_data/go2/dog_pace.pkl
```

Supported options include existing GMR conventions for headless operation,
rate limiting, recording, and saving. The existing `RobotMotionViewer` is
extended only as needed to register Go2's MJCF, base body, and camera distance.

## Testing

Tests are divided by responsibility:

1. Source and target YAML validation, including missing legs, unknown names,
   duplicate joints, and invalid reference poses.
2. `motion_imitation` parsing, frame width, FPS, loop mode, quaternion order,
   quaternion normalization, and finite-value validation.
3. Source FK at the reference pose and equivalence of batched and frame-wise
   evaluation.
4. Morphology scaling of neutral stance, stride, stance width, and foot lift.
5. Go2 Mink IK for a static stance, small single-foot displacement, joint
   limits, and an unreachable target.
6. Full `dog_pace.txt` to Go2 regression, including shape, finite values,
   limits, velocity limits, ground offset, save/load, and viewer model loading.
7. All existing humanoid tests to demonstrate no behavior regression.

Phase 1 acceptance criteria:

- A Laikago MJCF and `dog_pace.txt` produce a Go2 12-DoF motion.
- Output frame count and FPS match the input.
- Output has no NaN or infinite values.
- All target joints stay within Go2 limits.
- Configured velocity limits are respected.
- At least 95 percent of frames meet the configured foot-position error
  threshold.
- The saved motion plays continuously in the existing MuJoCo viewer.
- Existing humanoid tests pass unchanged.

## Phase 2 TODO: Contact-Aware Retargeting

The following features are intentionally deferred:

- Four-foot contact inference.
- Contact enter/exit hysteresis and minimum duration.
- Support-foot world-position anchoring.
- Support-foot slip correction.
- Contact-driven root height and orientation correction.
- Flight-phase handling.
- Contact continuity across loop boundaries.

Phase 2 requires its own design and implementation plan before development.
