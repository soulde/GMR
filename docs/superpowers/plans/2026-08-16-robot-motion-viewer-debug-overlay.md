# RobotMotionViewer Debug Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the standalone GMR target/error visualization to `RobotMotionViewer` as an opt-in, backward-compatible overlay.

**Architecture:** Add pure helpers for converting already-processed GMR targets and inferring source skeleton edges. Compose `GMRDebugVisualizer` inside `RobotMotionViewer` only in debug mode; preserve the current axis-frame path otherwise.

**Tech Stack:** Python, NumPy, SciPy Rotation, MuJoCo, pytest

## Global Constraints

- Debug mode is disabled by default.
- Debug input is `retargeter.scaled_human_data`; never scale or offset it again.
- `debug=True` requires `ik_config_path` and fails before GUI launch if absent.
- Explicit `human_skeleton_edges` override inferred edges.
- SMPL-X and LAFAN1 infer edges; unknown formats degrade to points.
- Missing targets do not stop playback.
- `close()` restores original robot alpha.
- `vis_gmr_debug.py` remains supported.

---

### Task 1: Processed Targets And Skeleton Edges

**Files:**
- Modify: `general_motion_retargeting/gmr_debug_visualizer.py`
- Modify: `tests/test_gmr_debug_visualizer.py`

**Interfaces:**
- Produces: `processed_reference_frame(frame) -> dict[str, ReferencePoint]`
- Extends: `reference_edges(names) -> tuple[tuple[str, str], ...]`

- [ ] **Step 1: Write a failing processed-target test**

Import `processed_reference_frame`. Pass position `[1.25, -0.5, 0.8]` and the
WXYZ quaternion for a 90-degree Z rotation. Assert the returned position is
unchanged and the rotation equals `Rotation.from_euler("z", 90,
degrees=True).as_matrix()`.

- [ ] **Step 2: Verify RED**

```bash
cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_gmr_debug_visualizer.py::test_processed_reference_frame_preserves_final_gmr_target_positions -q
```

Expected: import failure because the helper does not exist.

- [ ] **Step 3: Implement the conversion helper**

```python
def processed_reference_frame(frame):
    return {
        name: ReferencePoint(
            reference_name=name,
            world_position=np.asarray(value[0], dtype=float).copy(),
            world_rotation=Rotation.from_quat(
                np.asarray(value[1], dtype=float), scalar_first=True
            ).as_matrix(),
        )
        for name, value in frame.items()
    }
```

- [ ] **Step 4: Verify GREEN with the command from Step 2**

Expected: one passing test.

- [ ] **Step 5: Write failing hierarchy tests**

Assert SMPL-X returns `("pelvis", "left_knee")`, LAFAN1 returns
`("Hips", "LeftUpLeg")` and `("LeftLeg", "LeftFootMod")`, missing endpoints
are filtered, and unknown names return `()`.

- [ ] **Step 6: Verify the LAFAN1 hierarchy test is RED**

```bash
cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_gmr_debug_visualizer.py::test_reference_edges_infers_supported_skeletons_and_filters_missing_names -q
```

Expected: LAFAN1 assertions fail.

- [ ] **Step 7: Add the LAFAN1 parent registry**

Define edges for hips to spine and upper legs; upper leg to lower leg; lower
leg to modified foot; spine to upper arms; upper arm to forearm; and forearm
to hand. Combine this registry with `_SMPLX_DEBUG_PARENTS` in
`reference_edges`, filtering against available names.

- [ ] **Step 8: Run focused tests and commit**

```bash
(cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_gmr_debug_visualizer.py -q)
git -C GMR add general_motion_retargeting/gmr_debug_visualizer.py tests/test_gmr_debug_visualizer.py
git -C GMR commit -m "feat: support processed GMR debug targets"
```

Expected: all focused tests pass before the commit.

---

### Task 2: RobotMotionViewer Composition

**Files:**
- Modify: `general_motion_retargeting/robot_motion_viewer.py`
- Create: `tests/test_robot_motion_viewer_debug.py`

**Interfaces:**
- Produces: `RobotMotionViewer(..., debug=False, ik_config_path=None, debug_robot_alpha=0.3)`
- Produces: `step(..., human_skeleton_edges=None) -> None`

- [ ] **Step 1: Build a headless test fixture**

Create a minimal free-root MuJoCo XML and fake passive viewer with
`mujoco.MjvScene`, `sync`, `close`, and `set_texts`. Patch robot parameter
dictionaries and `mujoco.viewer.launch_passive` so tests open no GUI.

- [ ] **Step 2: Write failing constructor tests**

```python
def test_debug_mode_requires_ik_config_path(minimal_robot):
    with pytest.raises(ValueError, match="ik_config_path"):
        RobotMotionViewer("test_robot", debug=True)

def test_default_mode_does_not_create_debug_visualizer(minimal_robot):
    viewer = RobotMotionViewer("test_robot")
    assert viewer.debug_visualizer is None
```

- [ ] **Step 3: Verify RED**

```bash
cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_robot_motion_viewer_debug.py -k 'requires_ik_config or default_mode' -q
```

Expected: `debug` is an unknown constructor keyword.

- [ ] **Step 4: Implement opt-in construction**

Validate `ik_config_path` before launching MuJoCo. When enabled, load the config
with `load_effective_ik_config` and construct `GMRDebugVisualizer` with
`debug_robot_alpha`. Otherwise set `self.debug_visualizer = None`. Initialize
`self.debug_frame_index = 0`.

- [ ] **Step 5: Write failing overlay tests**

Cover nonzero debug geometry and statistics text; inferred LAFAN1 edges;
explicit-edge precedence; missing-target skipping; clearing stale geometry and
text when human data is absent; and alpha restoration on close. Spy on
`GMRDebugVisualizer.update` to inspect `full_reference_edges`.

- [ ] **Step 6: Verify overlay tests are RED**

```bash
cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_robot_motion_viewer_debug.py -q
```

Expected: `step` lacks the debug path and `human_skeleton_edges` argument.

- [ ] **Step 7: Implement the debug branch in `step`**

After forward kinematics, clear stale debug geometry/text. If human data is
present, convert it with `processed_reference_frame`, resolve explicit or
inferred edges, and call `debug_visualizer.update` with both
`reference_skeleton` and `full_reference_skeleton` set to those processed
points. Increment the debug frame index. Preserve the existing `draw_frame`
loop exactly when debug is disabled.

- [ ] **Step 8: Restore alpha in `close` and run focused tests**

```bash
(cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests/test_robot_motion_viewer_debug.py tests/test_gmr_debug_visualizer.py -q)
```

Expected: all focused tests pass.

- [ ] **Step 9: Commit the viewer integration**

```bash
git -C GMR add general_motion_retargeting/robot_motion_viewer.py tests/test_robot_motion_viewer_debug.py
git -C GMR commit -m "feat: add RobotMotionViewer debug overlay"
```

---

### Task 3: Regression And GUI Verification

**Files:**
- Verify: `tests/test_robot_motion_viewer_debug.py`
- Verify: `tests/test_gmr_debug_visualizer.py`
- Verify: `tests/test_vis_gmr_debug.py`

**Interfaces:**
- Consumes: the completed opt-in overlay
- Produces: automated and visual evidence for compatibility and debug rendering

- [ ] **Step 1: Run the public suite**

```bash
cd GMR && PYTHONPATH=. ../.venv/bin/pytest tests -q
```

Expected: all public tests pass, apart from documented dependency skips.

- [ ] **Step 2: Run private regression tests**

```bash
PYTHONPATH=GMR:. .venv/bin/pytest robots/chocolate/tests tests -q
```

Expected: no new failures relative to the known `initialize_root_from_human`
production-config contract failure.

- [ ] **Step 3: Run a bounded LAFAN1-to-G1 GUI smoke test in tmux**

Construct `GeneralMotionRetargeting` for `bvh_lafan1` and `unitree_g1`, then a
debug-enabled `RobotMotionViewer` using
`IK_CONFIG_DICT["bvh_lafan1"]["unitree_g1"]`. Retarget a bounded slice of
`data/LAFAN1/walk/walk1_subject1.bvh` and pass `scaled_human_data` to `step`.

Expected: translucent G1, blue target points/bones, orange robot points/bones,
red error lines, live error text, and clean shutdown.

- [ ] **Step 4: Inspect the final submodule state**

```bash
git -C GMR status --short
git -C GMR diff HEAD~2 --check
git -C GMR log -3 --oneline
```

Expected: only planned source, tests, design, and plan changes; scoped commits.
