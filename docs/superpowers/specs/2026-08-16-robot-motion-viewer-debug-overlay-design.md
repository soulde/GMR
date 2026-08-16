# RobotMotionViewer Debug Overlay Design

## Goal

Add the position-target diagnostics from `vis_gmr_debug.py` to
`RobotMotionViewer` as an opt-in overlay. Existing callers and rendering remain
unchanged unless debug mode is explicitly enabled.

## Public API

`RobotMotionViewer.__init__` gains these keyword arguments:

- `debug=False` controls the overlay and leaves it disabled by default.
- `ik_config_path=None` supplies the bare or calibration-wrapped IK config
  needed to map human targets to MuJoCo bodies and sites.
- `debug_robot_alpha=0.3` controls robot transparency while debugging.

Enabling debug mode without an IK config raises `ValueError` during
construction. The config is used only to identify active position mappings;
the viewer consumes `human_motion_data` that GMR has already scaled and offset,
so it must not repeat target preprocessing.

`RobotMotionViewer.step` gains an optional `human_skeleton_edges` argument.
Each edge is a `(parent_name, child_name)` pair. Explicit edges take precedence
over automatic inference.

## Rendering And Data Flow

`GMRDebugVisualizer` remains the single owner of correspondence computation,
debug geometry, robot transparency, and error text. `RobotMotionViewer`
composes an optional instance instead of duplicating or inheriting its logic.

On each debug frame, `RobotMotionViewer`:

1. writes the robot pose and runs MuJoCo forward kinematics;
2. converts the already-processed human positions and rotations into
   `ReferencePoint` values without applying scale or IK offsets again;
3. resolves explicit or inferred human skeleton edges;
4. calls `GMRDebugVisualizer.update` to draw blue human targets and bones,
   orange robot targets and bones, red error connectors, and frame error text;
5. synchronizes and rate-limits exactly as before.

The existing axis-frame visualization remains unchanged when debug mode is
off. In debug mode it is replaced by the diagnostic overlay, avoiding duplicate
geometry at each human target.

## Skeleton Inference

Skeleton inference uses a small registry of canonical parent relationships:

- SMPL-X names use the existing sparse SMPL-X debug hierarchy.
- LAFAN1 names connect the processed targets through hips, legs, spine, arms,
  and hands. Modified foot targets attach to their corresponding lower legs.

Only edges whose endpoints exist in the current human frame are returned.
Unknown input formats safely degrade to unconnected target points. Callers can
provide `human_skeleton_edges` for any other format without changing the
viewer or global registry.

## Lifecycle And Errors

- Missing individual human targets are skipped by the existing correspondence
  computation and do not stop playback.
- A debug frame with `human_motion_data=None` clears stale debug geometry and
  text while continuing to show the robot.
- `close()` restores the robot's original geometry alpha before closing the
  MuJoCo viewer and video resources.
- Debug rendering uses the existing robot MJCF. It does not require the
  composed floor scene used by the standalone debug script.

## Compatibility

All new constructor and step parameters are optional. Existing BVH, SMPL-X,
GVHMR, FBX, Xsens, OptiTrack, and saved-motion playback calls continue to use
the current visualization. A caller opts in by passing `debug=True`, an IK
config path, and the existing `retargeter.scaled_human_data` to `step`.

The standalone `vis_gmr_debug.py` remains supported and continues to own
offline reference loading, height correction, playback controls, and complete
source-skeleton loading. This change shares its renderer; it does not replace
the standalone workflow.

## Verification

Unit tests will cover processed-frame conversion without a second transform,
SMPL-X and LAFAN1 edge inference, explicit-edge precedence, constructor
validation, default-mode compatibility, overlay geometry and statistics,
missing-target handling, stale-overlay clearing, and alpha restoration.

Run the focused public tests for `RobotMotionViewer` and
`GMRDebugVisualizer`, then the complete public suite. Finally, run a MuJoCo GUI
smoke test using the LAFAN1-to-G1 sample and confirm the overlay, error text,
playback, and close lifecycle visually.
