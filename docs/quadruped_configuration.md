# Quadruped Configuration Reference

Quadruped retargeting uses two configuration layers:

1. A robot semantic YAML describes one MJCF model.
2. A source-to-target JSON describes one retargeting pair.

Keep geometry and naming in the robot YAML. Keep solver parameters, task
weights, offsets, and trajectory tuning in the pair JSON. This follows the
same separation as humanoid robot assets and `ik_configs`.

## Relationship To Humanoid GMR

The quadruped path follows the humanoid architecture where the source and
target domains are equivalent, but does not copy humanoid field names that do
not fit robot-to-robot motion.

| Concern | Humanoid GMR | Quadruped GMR |
| --- | --- | --- |
| Target model | `ROBOT_XML_DICT` MJCF | Robot semantic YAML `mjcf_path` |
| Target initial state | MJCF `model.qpos0` | MJCF `model.qpos0` |
| Pair tuning | `<source>_to_<robot>.json` | `<source_robot>_to_<target_robot>.json` |
| Source scale | Per-body `human_scale_table` | Root/leg XYZ `scale_table` |
| IK correspondence | Body `ik_match_table` | Root plus `FL/FR/RL/RR` foot tasks |
| Task offsets | Per-body position/rotation offsets | Per-foot root-frame position offsets |
| Ground parameter | `ground_height` plus script offset | Physical contact-surface `ground_height` |
| Joint initialization map | None | None |
| Explicit reference pose | None | None |

Humanoid GMR initializes `Mink.Configuration(model)` directly and expects the
MJCF default to be usable. Quadruped GMR now enforces the same contract and
adds validation because quadruped knee ranges commonly exclude zero.

Quadruped-specific additions are:

- explicit four-leg semantics, because source and target are both robots
- `foot_contact_offset`, because foot sites may be collision-sphere centers
- temporal-median foot trajectory centering for robot clips
- independent per-leg XYZ trajectory scales

Solver, damping, iteration count, and velocity magnitude are configurable in
the quadruped pair file. In the current humanoid implementation, solver and
damping are constructor arguments, iteration count is fixed in code, and the
velocity magnitude is fixed when velocity limiting is enabled.

### Intentional Runtime Differences

These differences are deliberate and should not be removed merely to match
field names:

| Behavior | Why quadruped differs |
| --- | --- |
| One IK task set | Root pose plus four position-only feet already form the complete task set. Humanoid table 2 adds more articulated body targets. |
| Physical foot ground alignment | Quadruped sites may be spherical-foot centers, so final alignment uses `site_z - foot_contact_offset`. |
| Final frame velocity clip | Multiple Mink iterations can satisfy a per-iteration limit while the saved frame-to-frame delta exceeds it. Quadruped output enforces the configured saved-motion limit. |
| Temporal-median center | Robot source reference poses are often unrelated to a specific clip. Median centering keeps targets near the target MJCF default. |
| XYZ scales | Fore/aft, lateral, and vertical morphology do not generally share one scalar ratio. |
| Position-only foot tasks | A 3-DoF quadruped leg cannot independently satisfy arbitrary foot position and orientation. |

Neither path performs contact inference or support-foot anchoring in core GMR.

## File Locations

```text
general_motion_retargeting/quadruped/
├── configs/
│   ├── laikago.yaml
│   └── unitree_go2.yaml
└── ik_configs/
    └── laikago_to_unitree_go2.json
```

Robots and source-target pairs are explicitly registered in
`general_motion_retargeting/params.py`, matching humanoid GMR's
`IK_CONFIG_DICT` pattern. Pair filenames should still follow
`<source_robot>_to_<target_robot>.json`. Use `--retarget_config PATH` to load a
temporary or experimental pair config without changing the registry.

## Coordinate Contract

All values after source loading use the GMR convention:

- right-handed coordinates
- `+X` forward, `+Y` left, and `+Z` up
- internal quaternions ordered `wxyz`
- saved GMR `root_rot` ordered `xyzw`
- distances in meters
- joint angles in radians
- angular velocities in radians per second

See [Motion Coordinate Conventions](motion_coordinate_conventions.md) for the
source adapter contract.

## Robot Semantic YAML

Complete shape:

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
  joint_order:
    - FL_hip_joint
    - FL_thigh_joint
    - FL_calf_joint
    - FR_hip_joint
    - FR_thigh_joint
    - FR_calf_joint
    - RL_hip_joint
    - RL_thigh_joint
    - RL_calf_joint
    - RR_hip_joint
    - RR_thigh_joint
    - RR_calf_joint

```

### Top-Level Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `robot` | yes | Stable robot identifier used by CLI and config lookup. |
| `model_type` | yes | Must be `quadruped`. |
| `mjcf_path` | yes | MJCF path relative to the repository root. |
| `root_body` | yes | Floating-base body used for root tasks and root-relative FK. |
| `foot_contact_offset` | no | Vertical distance from foot site to the lowest physical contact surface. Default `0`. |
| `legs` | yes | Exact `FL`, `FR`, `RL`, and `RR` semantic table. |
| `motion` | yes | Source/serialized joint and quaternion conventions. |

### `legs`

Each leg must declare exactly three unique hinge joints in kinematic order:
hip/abduction, thigh/hip-pitch, and calf/knee. The implementation does not
infer leg semantics from names.

`foot_site` identifies the point tracked by the position-only foot IK task.
Place it consistently across source and target models, normally at the center
of a spherical toe or at the sole reference point.

### `foot_contact_offset`

Foot quality checks and final ground alignment use:

```text
physical_contact_height = foot_site_z - foot_contact_offset
```

Examples:

- Go2 uses `0.022` because its site is at the center of a `0.022 m` spherical
  foot.
- Laikago uses `0.03` because its site is at the center of a `0.03 m`
  spherical toe.
- A site already placed on a flat sole uses `0.0`.

This is a geometric offset, not artificial clearance. A wrong value creates
systematic penetration or floating even when site trajectories look correct.

### `motion.quaternion_order`

Accepted values are `wxyz` and `xyzw`. This declares the order used by the
source format and by `root_frame_rotation`. The loader converts values to
internal `wxyz`. Saved GMR files always use `xyzw`.

### `motion.root_frame_rotation`

Optional fixed source-model basis rotation. It removes a static mismatch
between the source motion root frame and source MJCF root frame.

It must contain four finite values, use `motion.quaternion_order`, and describe
model/frame calibration rather than a specific motion frame. Omit it when the
source data and source MJCF already share the GMR axes. Do not use it to force
the first frame to identity.

### `motion.joint_order`

For a Motion Imitation text source, this list must exactly match the joint
columns following root position and quaternion. For a target robot, it defines
the serialized `dof_pos` order.

All names must exist in the MJCF. Duplicate names are rejected.

### MJCF Default Configuration

As in humanoid GMR, `Mink.Configuration(model)` starts from MuJoCo
`model.qpos0`. Quadruped YAML does not define a second reference pose and does
not copy source joint angles to the target.

Every declared target leg joint must have a `qpos0` value inside its MJCF
range. The loader rejects an invalid default before retargeting. The default
configuration also defines target neutral feet for morphology transfer.

An MJCF `<keyframe>` does not change `model.qpos0`, and GMR does not
automatically load a keyframe. Robot assets must therefore provide a legal
default directly. If joint `ref` values are introduced, preserve the model's
existing joint-angle and FK semantics; changing `ref` alone changes the
physical meaning of serialized joint positions.

## Pair IK JSON

Complete Laikago-to-Go2 example:

```json
{
  "ground_height": 0.0,
  "ik": {
    "solver": "daqp",
    "damping": 0.5,
    "max_iterations": 10,
    "velocity_limit": 9.42477796076938
  },
  "motion_mapping": {
    "center": "temporal_median",
    "scale_table": {
      "root": [1.0, 1.0, 1.0],
      "FL": [1.0, 1.0, 0.878280151975182],
      "FR": [1.0, 1.0, 0.878280151975182],
      "RL": [1.0, 1.0, 0.878280151975182],
      "RR": [1.0, 1.0, 0.878280151975182]
    }
  },
  "tasks": {
    "root": {
      "position_cost": 100.0,
      "orientation_cost": 100.0
    },
    "feet": {
      "FL": {
        "position_cost": 100.0,
        "position_offset": [0.0, 0.0, 0.0]
      },
      "FR": {
        "position_cost": 100.0,
        "position_offset": [0.0, 0.0, 0.0]
      },
      "RL": {
        "position_cost": 100.0,
        "position_offset": [0.0, 0.0, 0.0]
      },
      "RR": {
        "position_cost": 100.0,
        "position_offset": [0.0, 0.0, 0.0]
      }
    }
  }
}
```

Every leg table must contain exactly `FL`, `FR`, `RL`, and `RR`. Unknown or
missing leg keys are rejected.

### `ground_height`

Final physical foot-contact surface height in world meters. After all frames
are solved, GMR shifts the complete target root trajectory so:

```text
global minimum(site_z - foot_contact_offset) = ground_height
```

This is a single global root-height translation. It does not infer contact,
move individual feet, or correct support-foot sliding.

### `ik`

| Field | Unit | Meaning |
| --- | --- | --- |
| `solver` | name | QP solver passed to Mink. The supplied config uses `daqp`. |
| `damping` | dimensionless | Larger values reduce aggressive updates but increase tracking error. |
| `max_iterations` | count | Maximum solve iterations per frame. |
| `velocity_limit` | rad/s | Per-joint hinge velocity limit when velocity limiting is enabled. |

`velocity_limit` only affects conversion when `--use_velocity_limit` is
passed. The retargeter applies both Mink's velocity limit and a final
frame-to-frame joint delta clip.

### `motion_mapping.center`

`temporal_median` computes a separate temporal median foot position for each
source leg and maps those centers to the target reference feet. It is robust
when the source YAML reference pose differs from the clip's actual nominal
pose and is the recommended robot-to-robot default.

`mjcf_default` uses source feet computed from `model.qpos0`. It provides
clip-independent calibration but requires an MJCF default representative of
every processed clip.

Changing the center alters the reachable neighborhood but does not modify
source root or joint data.

The supplied Laikago MJCF has a nominal `qpos0` with thigh `0.67 rad` and calf
`-1.25 rad`, baked into the model without changing raw Motion Imitation joint
FK semantics. For `dog_pace`, temporal centering still performs better:

| Metric | `temporal_median` | `mjcf_default` |
| --- | ---: | ---: |
| Maximum IK error | `0.0419` | `0.0500` |
| Maximum physical foot height | `0.069 m` | `0.079 m` |
| Inferred flight ratio | `5.1%` | `5.1%` |
| Source-contact sliding frames | `7` | `9` |
| Calf upper-limit hits | none | all four legs |

Use `mjcf_default` only when the source MJCF default is intentionally authored
as the nominal pose for the processed motion family.

### `motion_mapping.scale_table`

The table must contain exactly `root`, `FL`, `FR`, `RL`, and `RR`. Every value
is a positive `[x, y, z]` scale vector.

`root` scales target root displacement relative to the first source frame:

```text
target_root = source_root_first
              + (source_root - source_root_first) * scale_table.root
```

It is the robot-to-robot counterpart of humanoid root scaling. Anchoring the
first frame preserves the clip's initial world placement. Values must be
finite and greater than zero. Start with `[1, 1, 1]`; change it only when root
travel or vertical oscillation should scale with target morphology.

Each leg entry scales that leg's root-frame dynamic displacement:

```text
target_foot_root =
    target_reference_foot
    + (source_foot_root - source_center) * scale_table.<leg>
```

Interpretation:

- `x`: fore/aft step displacement
- `y`: lateral step displacement
- `z`: lift and compression displacement

Values must be finite and greater than zero.

Start with horizontal scales of `1.0`. Scaling horizontal dynamics by
hip-length or hip-width ratios can create world-frame support-foot velocity
even when the source support foot is nearly stationary. A useful initial
vertical value is:

```text
target nominal leg reach / source nominal leg reach
```

Per-leg values support real model asymmetry or repeatable source calibration
differences. Do not use them as implicit contact correction.

### `tasks.root`

`position_cost` and `orientation_cost` are positive Mink task weights.

Four foot tasks compete with one root task. A root orientation weight much
smaller than each foot position weight can let unreachable foot targets rotate
the trunk away from the source orientation. Equal root and per-foot weights
are a conservative Go2 starting point.

### `tasks.feet`

`position_cost` controls each position-only foot-site task. Larger values
reduce foot tracking error but can increase root error or push joints toward
limits when a target is unreachable.

`position_offset` is an XYZ offset in the target root frame, in meters. It is
added to the mapped root-relative foot target before root orientation rotates
the target into the world frame.

Use offsets only for consistent geometric calibration errors. Do not use Z
offsets to compensate for spherical foot radius; use the robot YAML's
`foot_contact_offset`.

## Recommended Tuning Order

1. Validate both MJCF models, axes, quaternion order, joint order, and foot
   sites.
2. Set `foot_contact_offset` from collision geometry.
3. Author a legal target MJCF `qpos0` with comfortable joint-limit margins.
4. Use `temporal_median` and start trajectory scales at `[1, 1, reach_ratio]`.
5. Inspect desired and solved hip-to-foot reach before changing task weights.
6. Tune root and foot costs while monitoring orientation and foot error.
7. Tune damping and iteration count only after targets are reachable.
8. Enable velocity limiting and inspect frames clipped at the configured value.
9. Apply per-leg scales or offsets only for repeatable leg-specific errors.

Do not start by increasing foot weights. If calf joints hit extension limits,
reduce or recenter the requested trajectory first.

## Quality Gates

Run:

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

Check at minimum:

- finite root, quaternion, and joint values
- zero joint-range violation
- velocity at or below the configured limit
- physical foot-contact minimum at `ground_height`
- no repeated zero joint-limit margins
- low non-converged-frame ratio
- acceptable maximum IK task error

The current checker does not infer four-foot contacts. Support-foot sliding,
stance/swing consistency, dynamics, torque, and stability require separate
diagnostics.

## Adding A Source-Target Pair

1. Add or validate both robot YAML files.
2. Confirm source text joint columns match source `motion.joint_order`.
3. Add `<source>_to_<target>.json` and register it in
   `QUADRUPED_IK_CONFIG_DICT`.
4. Start from a reachable target MJCF default configuration.
5. Measure source and target nominal leg reach.
6. Set motion center and per-leg scales.
7. Run the quadruped tests and full motion checker.
8. Inspect the result in the target MJCF scene.

The pair config is required by the conversion CLI. Missing registry entries or
files fail explicitly; no unrelated robot parameters are used as fallback.
