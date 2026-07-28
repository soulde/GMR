# Quadruped Robot Retargeting

GMR provides a separate robot-to-robot path for quadrupeds. The first
supported source is a Laikago `motion_imitation` motion and the first target is
Unitree Go2. The existing humanoid `GeneralMotionRetargeting` flow is
unchanged.

All adapters and saved outputs must follow
[GMR Motion Coordinate Conventions](motion_coordinate_conventions.md).
See [Quadruped Configuration Reference](quadruped_configuration.md) for the
complete robot YAML and source-to-target IK JSON schema.

## Pipeline

```text
motion_imitation txt + source MJCF + source YAML
  + source-to-target IK config
  -> source joint-space motion
  -> MuJoCo source forward kinematics
  -> trunk-relative FL/FR/RL/RR foot trajectories
  -> morphology scaling
  -> target Mink trunk and foot IK
  -> joint and optional velocity limits
  -> lowest-foot ground offset
  -> GMR-compatible pickle
```

Internal quaternions are scalar-first `wxyz`. Saved `root_rot` values use the
existing GMR `xyzw` convention.

## Usage

```bash
python scripts/download_quadruped_motions.py

python scripts/motion_imitation_to_robot.py \
  --motion_file assets/quadrupeds/motions/dog_pace.txt \
  --source_robot laikago \
  --robot unitree_go2 \
  --save_path retargeting_data/go2/dog_pace.pkl \
  --headless \
  --use_velocity_limit

python scripts/check_quadruped_motion.py \
  --motion retargeting_data/go2/dog_pace.pkl \
  --robot unitree_go2
```

Remove `--headless` to play the result in `RobotMotionViewer`. Add
`--rate_limit` for real-time playback.

The output pickle contains:

- `fps`
- `root_pos`, shape `[T, 3]`
- `root_rot`, shape `[T, 4]`, ordered `xyzw`
- `dof_pos`, shape `[T, 12]`, ordered by the target YAML
- `retarget_diagnostics`, one IK diagnostic record per frame

## Adding A Robot

Phase 1 requires an MJCF and a semantic YAML file. Add the MJCF under
`assets/quadrupeds/<robot>/` and the YAML under
`general_motion_retargeting/quadruped/configs/<robot>.yaml`.

As in the humanoid GMR path, solver and task parameters live in a
source-to-target IK configuration. Add
`general_motion_retargeting/quadruped/ik_configs/<source>_to_<target>.json`.
The conversion CLI selects this file automatically; `--retarget_config`
overrides it.

The quadruped-specific configuration contains:

- `ik`: solver, damping, iteration count, and joint velocity limit.
- `tasks.root`: root position and orientation costs.
- `tasks.feet`: per-leg position costs and root-frame XYZ target offsets.
- `motion_mapping.center`: `temporal_median` or `mjcf_default`.
- `motion_mapping.scale_table`: root and per-leg XYZ dynamic scales.
- `ground_height`: final physical foot-contact surface height.

This is the quadruped equivalent of the humanoid `ik_match_table`: robot model
semantics remain in the robot YAML, while source-to-target tuning remains in
the pair configuration. Contact inference and support-foot anchoring are not
implicit configuration features.

The detailed field reference, units, coordinate semantics, tuning order, and
complete examples are maintained in
[Quadruped Configuration Reference](quadruped_configuration.md).

Each model must declare one root and four legs in this fixed semantic order:

```yaml
robot: example
model_type: quadruped
mjcf_path: assets/quadrupeds/example/example.xml
root_body: trunk
foot_contact_offset: 0.0
legs:
  FL:
    joints: [FL_hip_joint, FL_thigh_joint, FL_calf_joint]
    foot_site: FL_foot_site
  FR:
    joints: [FR_hip_joint, FR_thigh_joint, FR_calf_joint]
    foot_site: FR_foot_site
  RL:
    joints: [RL_hip_joint, RL_thigh_joint, RL_calf_joint]
    foot_site: RL_foot_site
  RR:
    joints: [RR_hip_joint, RR_thigh_joint, RR_calf_joint]
    foot_site: RR_foot_site
motion:
  quaternion_order: xyzw
  root_frame_rotation: [0.5, 0.5, 0.5, 0.5]
  joint_order: [...]
```

Names are validated against the MJCF. The implementation does not infer
semantics from naming conventions. Source `motion.joint_order` must exactly
match the values after the root position and quaternion in each source frame.
As in humanoid GMR, target IK starts from MJCF `model.qpos0`; every declared
leg joint must have a legal default value.

`foot_contact_offset` is the vertical distance from each configured foot site
to the lowest physical contact surface. For example, Go2 places each foot site
at the center of a spherical collision geom with radius `0.022 m`, so its
offset is `0.022`. Ground alignment and foot-height quality checks use
`site_z - foot_contact_offset`; this prevents aligning the collision-geom
center to the floor.

## Input And Scaling

A `motion_imitation` frame contains three root-position values, four root
quaternion values, and the configured source joint values. `FrameDuration`
defines FPS and `LoopMode` is preserved.

Optional `root_frame_rotation` removes a fixed source-model basis rotation
before transfer. It is independent of motion frames. Omit it when the source
motion and source MJCF already share the GMR root/world convention.

Source MuJoCo FK converts joint motion into foot positions relative to the
source trunk. Morphology transfer centers each leg trajectory on its temporal
median and maps that center to the target reference stance. Horizontal
trajectory deltas remain unscaled so a stationary source support foot does not
gain velocity merely because the robots have different hip spacing. Vertical
deltas are scaled by the target/source nominal leg-reach ratio. This avoids
copying source joint angles between robots with different link geometry while
keeping the dynamic trajectory in the target's reachable neighborhood.

The target solver uses a Mink body task for the trunk and position-only site
tasks for all four feet. `ConfigurationLimit` always applies. With
`--use_velocity_limit`, Mink limits each IK update and the final frame-to-frame
joint delta is clipped to the configured velocity at the motion FPS.

Ground handling only shifts all root heights until the global lowest solved
foot contact surface reaches zero height. It does not infer support state or
move individual feet.

## Quality Checks

`check_quadruped_motion.py` reconstructs target qpos from the saved file and
checks:

- required shapes and finite values
- named target joint limits and margins
- finite-difference joint velocity
- minimum and maximum foot-site heights from Go2 FK
- maximum IK task error and frames that exhausted IK iterations

It exits nonzero for invalid values, joint-limit violations, configured
velocity violations, or an excessive non-converged-frame ratio.

## Phase 1 Limits

- Source and target models must be MJCF.
- Robots must have four legs with three hinge joints per leg.
- Wheel joints and wheeled-legged robots are unsupported.
- No dynamics, torque, stability, or loop-closure optimization is performed.
- Semantic names and source columns must be declared explicitly.

## Phase 2 TODO

Contact-aware retargeting requires a separate design and implementation:

- infer four-foot contacts from source motion
- distinguish stance, swing, and flight phases
- anchor support feet in world coordinates
- correct support-foot sliding
- resolve conflicting multi-foot constraints
- add contact-aware quality metrics and thresholds
- evaluate dynamics and stability constraints
