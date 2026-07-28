# Quadruped Robot Retargeting

GMR provides a separate robot-to-robot path for quadrupeds. The first
supported source is a Laikago `motion_imitation` motion and the first target is
Unitree Go2. The existing humanoid `GeneralMotionRetargeting` flow is
unchanged.

All adapters and saved outputs must follow
[GMR Motion Coordinate Conventions](motion_coordinate_conventions.md).

## Pipeline

```text
motion_imitation txt + source MJCF + source YAML
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

Each model must declare one root and four legs in this fixed semantic order:

```yaml
robot: example
model_type: quadruped
mjcf_path: assets/quadrupeds/example/example.xml
root_body: trunk
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
reference_pose:
  FL_hip_joint: 0.0
joint_mapping:
  hip: {sign: 1.0, scale: 1.0}
  thigh: {sign: 1.0, scale: 1.0}
  calf: {sign: 1.0, scale: 1.0}
```

Names are validated against the MJCF. The implementation does not infer
semantics from naming conventions. Source `motion.joint_order` must exactly
match the values after the root position and quaternion in each source frame.

## Input And Scaling

A `motion_imitation` frame contains three root-position values, four root
quaternion values, and the configured source joint values. `FrameDuration`
defines FPS and `LoopMode` is preserved.

Optional `root_frame_rotation` removes a fixed source-model basis rotation
before transfer. It is independent of motion frames. Omit it when the source
motion and source MJCF already share the GMR root/world convention.

Source MuJoCo FK converts joint motion into foot positions relative to the
source trunk. Morphology transfer scales the neutral stance and trajectory
deltas using source and target hip spacing and nominal leg reach. This avoids
copying source joint angles between robots with different link geometry.

The target solver uses a Mink body task for the trunk and position-only site
tasks for all four feet. `ConfigurationLimit` always applies. With
`--use_velocity_limit`, Mink limits each IK update and the final frame-to-frame
joint delta is clipped to the configured velocity at the motion FPS.

Ground handling only shifts all root heights by the global lowest solved foot
height. It does not infer support state or move individual feet.

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
