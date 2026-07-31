# SMPL-X Retarget Export Layout

## Goal

Make `scripts/smplx_to_robot.py` write deterministic, robot-scoped artifacts
without requiring `--save_path`. Multiple motions for the same robot share one
joint-order contract.

## Output layout

For input `/data/walk_stageii.npz` and `--robot dr02`, saving produces:

```text
retarget_data/dr02/
├── joints.json
├── motions/
│   └── walk_stageii.pkl
├── datasets/
│   └── walk_stageii.npz
└── beyondmimic/
    └── walk_stageii.csv
```

The repository-relative `retarget_data` directory is the fixed output root.
The source basename supplies the artifact basename. Existing artifacts with the
same robot and basename are replaced by a successful rerun.

## Joint-order contract

Joint order comes from the MuJoCo model used by GMR, sorted by each joint's
`jnt_qposadr`, excluding the free-root joint. The exporter supports the current
GMR convention of a free root followed by scalar hinge/slide joints and rejects
models whose non-root joints occupy more than one `qpos` value.

`joints.json` contains a versioned, explicit contract:

```json
{
  "format_version": 1,
  "robot": "dr02",
  "joint_names": ["joint_a", "joint_b"]
}
```

If `joints.json` already exists, its robot and ordered names must match the
current model. A mismatch is an error; the exporter never silently rewrites a
different joint contract.

## Artifact formats

The PKL remains the authoritative GMR motion format and preserves the existing
keys and conventions:

- `fps`
- `root_pos`
- `root_rot` in XYZW order
- `dof_pos` in `joints.json` order
- `local_body_pos`
- `link_body_list`

The generic training NPZ contains only robot-independent fields:

- `fps`
- `root_pos`
- `root_quat` in XYZW order
- `root_lin_vel`
- `root_ang_vel`
- `joint_pos`
- `joint_vel`
- `joint_names`

Linear and joint velocities use finite differences at `fps`. Angular velocity
uses relative quaternion rotation divided by the frame interval. The first
sample uses the first forward difference, matching the output frame count.
Robot-specific contacts, foot states, commands, and phase are outside this
generic format.

The BeyondMimic CSV preserves the existing converter's column convention:

```text
root_pos(3), root_rot_xyzw(4), dof_pos(N)
```

Motions above 30 FPS are sampled to 30 FPS using the existing index-selection
behavior. Other frame rates are preserved.

## Integration

A reusable export module owns path construction, joint extraction and
validation, serialization, velocity calculation, and CSV conversion.
`smplx_to_robot.py` collects frames whenever saving is enabled and calls this
module once after retargeting.

The old `--save_path` option is removed. A new `--save` flag enables export;
without it, the script retains its current view-only behavior. `--loop` and
`--save` are rejected together because a looping conversion never reaches the
export step.

## Failure behavior

Inputs are validated before artifact writes: positive finite FPS, matching
frame counts, finite arrays, correct root shapes, and a DoF count equal to the
joint contract. Parent directories are created as needed. Each individual file
is written via a temporary sibling and atomically replaced, minimizing partial
files if serialization fails.

## Tests

Unit tests cover:

- MuJoCo qpos-address joint ordering and unsupported joint layouts;
- new and matching `joints.json` contracts plus mismatch rejection;
- deterministic paths from robot and input basename;
- PKL, generic NPZ, and CSV fields, shapes, ordering, and downsampling;
- invalid motion validation;
- `smplx_to_robot.py` CLI save/loop behavior and export invocation.

