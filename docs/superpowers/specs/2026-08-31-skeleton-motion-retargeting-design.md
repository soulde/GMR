# Skeleton Motion Retargeting Design

## Objective

Add a second, independent full-body retargeting algorithm to GMR. The new
algorithm reconstructs human targets chain by chain using human motion
directions and explicitly configured target segment lengths, then solves the resulting targets with
the existing Mink inverse-kinematics infrastructure.

The original `GeneralMotionRetargeting` behavior and configuration semantics
remain unchanged. The new public API must preserve the existing call shape:

```python
qpos = retargeter.retarget(human_frame)
```

The first supported pair is LAFAN1 to DR02.

## Motivation

The existing GMR position scaling applies an independent scale to every human
joint relative to the human root:

```text
scaled_joint = scaled_root + scale[joint] * (joint - root)
```

If adjacent joints use different scales, their distance depends on their pose
relative to the root. A nominally rigid upper arm or forearm can therefore
change target length as the character moves. Large child-local position
offsets have the same problem because they rotate with the child joint.

Chain retargeting avoids this by transferring source bone directions while
using configured target segment lengths. This follows the chain-oriented principle of
Unreal Engine IK retargeting, adapted for a rigid robot whose links cannot be
scaled.

## Architecture

Create an independent `SkeletonMotionRetargeting` class. It owns chain target
reconstruction and reuses the established robot model, Mink tasks, joint
limits, damping, and iterative IK solve behavior. It must not add a mode flag
to `GeneralMotionRetargeting` or change the original algorithm path.

The frame data flow is:

```text
LAFAN1 global joint transforms
  -> validate and normalize source chain directions
  -> align source and target retarget-pose semantics
  -> reconstruct full-body targets using configured target segment lengths
  -> apply only explicit frame-origin calibration offsets
  -> set Mink position and orientation tasks
  -> run the existing two-stage IK solve
  -> return qpos
```

The implementation consists of three focused units:

1. A chain configuration loader that validates explicit target segment lengths
   against resolved robot frames and can report the corresponding MJCF reference
   distances for calibration.
2. A stateless chain target reconstructor with a small amount of previous-frame
   direction state for degenerate source segments.
3. `SkeletonMotionRetargeting`, which connects reconstructed targets to the
   established Mink IK solve and exposes the compatible `retarget()` API.

## Full-Body Chain Model

The initial DR02/LAFAN1 configuration covers:

- root and pelvis;
- pelvis to torso/spine anchor;
- left and right shoulder, elbow, and wrist chains;
- left and right hip, knee, ankle, and toe chains.

For each directed segment, the source unit direction is:

```text
d = normalize(source_child_position - source_parent_position)
```

The reconstructed child position is:

```text
target_child_position = target_parent_position + target_length * d
```

Reconstruction proceeds from each chain root toward its end effector, so every
child depends on the already reconstructed parent rather than the human root.
Robot segment lengths are explicit, user-tunable configuration values. Their
initial values should be measured from the MJCF reference configuration, but the
configuration remains authoritative because the desired retargeting anchor
distance can differ from the distance between two body-frame origins.

The pelvis/root policy remains explicit. Root translation and orientation are
transferred according to the configuration, while articulated child chains use
fixed target lengths. Ground handling remains compatible with the existing GMR
workflow.

## Rotation Retargeting

Position reconstruction and orientation transfer are separate concerns.
Rotation offsets align source bone-local axes with robot frame-local axes in a
retarget pose. Each offset must satisfy both:

- the source bone direction maps to the robot link direction; and
- the source flexion/twist axes map to the corresponding robot joint axes with
  the correct sign.

Orientation tasks are optional per target. Position-only tasks still receive a
valid aligned rotation so small frame-origin offsets have defined local-frame
semantics. Large offsets must not be used to emulate a different link length.

## Configuration

Add a separate configuration file while preserving the established GMR shape:

```text
general_motion_retargeting/skeleton_configs/
  bvh_lafan1_to_dr02.json
```

The existing top-level root, ground, initialization, and two-stage IK fields
remain recognizable. `ik_match_table1` and `ik_match_table2` retain their
current entry layout:

```text
[human_joint, position_weight, orientation_weight,
 position_offset, rotation_offset]
```

This keeps task weights and calibration editable with the same workflow as the
original retargeter. The skeleton configuration adds only:

```json
{
  "algorithm": "skeleton",
  "skeleton_chains": [
    {
      "name": "left_arm",
      "segments": [
        {
          "human_parent": "LeftArm",
          "human_child": "LeftForeArm",
          "robot_parent_frame": "left_shoulder_y_link",
          "robot_child_frame": "left_elbow_link",
          "target_length": 0.265
        },
        {
          "human_parent": "LeftForeArm",
          "human_child": "LeftHand",
          "robot_parent_frame": "left_elbow_link",
          "robot_child_frame": "left_wrist_x_link",
          "target_length": 0.2395
        }
      ]
    }
  ]
}
```

Segments are ordered from chain root to end effector. A segment child must be
the next segment's parent. `target_length` is expressed in metres and must be
positive. Existing IK tables decide which reconstructed joints become tasks in
each solve stage. A diagnostic compares configured lengths with MJCF reference
frame distances without silently overriding the configured values.

`human_scale_table` may remain in the file for visual/configuration familiarity,
but skeleton-chain joints must all be `1.0` and the chain reconstructor does not
use those values. An optional root translation scale is a separate scalar and
defaults to `1.0`.

## Retarget Base Pose and Offsets

The source and target skeletons must first share a consistent retarget base
pose. For LAFAN1 and DR02 this means aligning the corresponding torso, arm, leg,
hand, and foot chains to a common reference pose before transferring animation.
The base-pose correction is represented by the existing per-task
`rotation_offset` quaternion.

`rotation_offset` has one purpose: compensate for source/target rest-pose and
local-axis differences. It must align the bone direction and the relevant
flexion/twist axes. It must not be tuned to change a segment length.

`position_offset` has one purpose: compensate for a small difference between a
semantic joint location and the selected robot frame origin. It is applied in
the rotation-corrected local frame. It must not be used to shorten or lengthen a
chain; `target_length` owns that behavior.

The configuration validator reports position offsets whose magnitude exceeds a
conservative calibration threshold of `0.05 m`. This is a diagnostic warning,
not a hard failure, because some robot frame origins may legitimately require a
larger correction.

Configuration validation occurs during construction. It rejects missing human
chain declarations, disconnected chains, duplicate child ownership, missing
robot frames, non-positive configured segment lengths, malformed quaternions,
and invalid task weights.

## Degenerate Input Handling

If a source segment length is below a configured epsilon, reconstruction uses
the most recent valid direction for that segment. If the first processed frame
is degenerate and no previous direction exists, `retarget()` raises a clear
error identifying the chain and source joints. Missing source joints fail with
an equally specific error.

All reconstructed positions and rotations are checked for finite values before
they are passed to Mink.

## Compatibility

`SkeletonMotionRetargeting` returns the same qpos layout as
`GeneralMotionRetargeting`. Existing exporters and the debug visualizer should
select the algorithm explicitly while keeping their output formats unchanged.

The original retargeter, its public API, and its existing configuration files
must continue to work without behavioral changes.

## Verification

Automated tests must prove:

- reconstructed segment lengths equal the configured target lengths within
  `1e-6 m` for every tested frame;
- segment lengths remain constant across straight, flexed, and rotated poses;
- left and right chains use their independently configured target lengths;
- configured target lengths can differ from MJCF frame-origin distances and
  remain authoritative;
- rotation offsets preserve segment length while correcting a deliberately
  mismatched retarget base pose;
- full-body chain order and parent-child connectivity are validated;
- degenerate segments reuse a prior valid direction and fail clearly when no
  prior direction exists;
- missing robot frames fail during retargeter construction, and missing human
  joints fail before reconstruction of the first affected frame;
- retargeting a representative LAFAN1 sequence produces only finite qpos;
- the new class preserves `retarget(frame) -> qpos` compatibility;
- all existing private and public GMR tests continue to pass.

Visual verification uses the current LAFAN1 dance motion in
`vis_gmr_debug.py`, comparing the original GMR output with skeleton-retargeting
output. Review focuses on constant arm and leg lengths, elbow and knee bending,
hand orientation, pelvis motion, foot contact, and absence of sudden solve
jumps.

## Non-Goals

- Do not replace or modify the original GMR algorithm.
- Do not stretch robot links or modify the MJCF geometry.
- Do not implement Unreal Engine assets or depend on Unreal Engine.
- Do not add learned retargeting, dynamics, collision avoidance, or contact
  optimization in the first implementation.
- Do not generalize configuration generation to every GMR robot before the
  DR02/LAFAN1 implementation is validated.
