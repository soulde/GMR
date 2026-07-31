# SMPL-X Retarget Exports Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SMPL-X retarget command emit a robot-scoped joint contract, GMR PKL, generic training NPZ, and BeyondMimic CSV under `retarget_data/<robot>/`.

**Architecture:** Add a focused `retarget_export` module that derives the canonical scalar-joint order from the same MuJoCo model used by GMR, validates one normalized motion payload, and writes all formats. Keep `smplx_to_robot.py` responsible for retargeting and CLI behavior only; it hands its accumulated `qpos` frames to the exporter once.

**Tech Stack:** Python 3.10+, NumPy, SciPy Rotation, MuJoCo, pickle, JSON, pytest.

## Global Constraints

- The fixed output root is repository-relative `retarget_data`.
- Each robot has exactly one versioned `joints.json`; mismatches are errors.
- PKL retains the existing GMR keys and XYZW `root_rot` convention.
- Generic NPZ excludes contacts, foot states, commands, and phase.
- CSV columns are `root_pos(3), root_rot_xyzw(4), dof_pos(N)` and motions above 30 FPS use the existing sampling behavior.
- The old `--save_path` option is replaced by `--save`; `--loop --save` is invalid.

---

### Task 1: Canonical joint contract and artifact paths

**Files:**
- Create: `general_motion_retargeting/retarget_export.py`
- Create: `tests/test_retarget_export.py`

**Interfaces:**
- Consumes: `mujoco.MjModel`, robot name, source path, and optional output root.
- Produces: `ExportPaths`, `scalar_joint_names(model)`, and `ensure_joint_contract(path, robot, joint_names)`.

- [ ] **Step 1: Write failing path and joint-order tests**

```python
def test_export_paths_use_robot_and_source_stem(tmp_path):
    paths = export_paths("dr02", "/data/walk_stageii.npz", tmp_path)
    assert paths.joints == tmp_path / "dr02/joints.json"
    assert paths.motion == tmp_path / "dr02/motions/walk_stageii.pkl"
    assert paths.dataset == tmp_path / "dr02/datasets/walk_stageii.npz"
    assert paths.csv == tmp_path / "dr02/beyondmimic/walk_stageii.csv"

def test_scalar_joint_names_follow_qpos_addresses():
    model = free_hinge_model(("hip", "knee"))
    assert scalar_joint_names(model) == ("hip", "knee")
```

Also test rejection of a non-free root, ball joints, and unnamed scalar joints.

- [ ] **Step 2: Run tests and verify the new module is missing**

Run: `pytest tests/test_retarget_export.py -v`

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement paths and canonical joint extraction**

```python
@dataclass(frozen=True)
class ExportPaths:
    joints: pathlib.Path
    motion: pathlib.Path
    dataset: pathlib.Path
    csv: pathlib.Path

def export_paths(robot, source_path, output_root=pathlib.Path("retarget_data")):
    base = pathlib.Path(output_root) / robot
    stem = pathlib.Path(source_path).stem
    return ExportPaths(base / "joints.json", base / "motions" / f"{stem}.pkl",
                       base / "datasets" / f"{stem}.npz",
                       base / "beyondmimic" / f"{stem}.csv")
```

Use `model.jnt_qposadr` to sort non-root joints and verify every retained joint is hinge/slide and occupies exactly one qpos slot.

- [ ] **Step 4: Implement and test the versioned contract**

Write JSON as `{"format_version": 1, "robot": robot, "joint_names": list(joint_names)}`. If the path exists, load it and require exact dictionary equality. Write a new contract through a temporary sibling followed by `Path.replace()`.

Run: `pytest tests/test_retarget_export.py -v`

Expected: all Task 1 tests pass, including exact-match reuse and mismatch rejection.

- [ ] **Step 5: Commit Task 1**

```bash
git add general_motion_retargeting/retarget_export.py tests/test_retarget_export.py
git commit -m "feat: define retarget joint export contract"
```

### Task 2: Validate and serialize all motion artifacts

**Files:**
- Modify: `general_motion_retargeting/retarget_export.py`
- Modify: `tests/test_retarget_export.py`

**Interfaces:**
- Consumes: `export_retarget_motion(model, robot, source_path, fps, qpos_frames, output_root=Path("retarget_data"))`.
- Produces: the four paths in `ExportPaths` and atomically written PKL/NPZ/CSV artifacts.

- [ ] **Step 1: Write failing validation and serialization tests**

Use two frames of a free-root, two-hinge model. Assert PKL keys are exactly the existing GMR keys, root quaternion is converted WXYZ to XYZW, and `dof_pos` follows `joints.json`. Assert NPZ has exactly:

```python
{
    "fps", "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
    "joint_pos", "joint_vel", "joint_names",
}
```

Assert CSV has `7 + N` columns. Add failures for non-positive FPS, nonfinite qpos, wrong qpos width, and zero frames.

- [ ] **Step 2: Run focused tests and verify missing exporter failure**

Run: `pytest tests/test_retarget_export.py -k 'motion or artifact or invalid' -v`

Expected: FAIL because `export_retarget_motion` is not defined.

- [ ] **Step 3: Implement normalized motion validation and velocity calculation**

Validate `qpos_frames.shape == (T, model.nq)`, `T > 0`, finite arrays, and positive finite FPS. Split `qpos[:, :3]`, normalize `qpos[:, 3:7]`, and take scalar joints using their `jnt_qposadr` rather than assuming contiguous model storage. Compute finite differences with the last sample copied from the previous velocity. Compute angular velocity from consecutive relative SciPy rotations and retain `T` samples.

- [ ] **Step 4: Implement atomic PKL and NPZ serialization**

PKL payload:

```python
{
    "fps": float(fps), "root_pos": root_pos,
    "root_rot": root_quat_wxyz[:, [1, 2, 3, 0]],
    "dof_pos": joint_pos, "local_body_pos": None, "link_body_list": None,
}
```

NPZ stores the eight fields specified above, with `root_quat` in XYZW to match the public artifact convention. Use temporary siblings and replacement for both files.

- [ ] **Step 5: Implement CSV conversion and downsampling**

Build columns from PKL-convention arrays. When `fps > 30`, select `np.arange(0, T, fps / 30.0).astype(int)`; otherwise retain all rows. Write with `np.savetxt(..., delimiter=",")` through a temporary sibling.

Run: `pytest tests/test_retarget_export.py -v`

Expected: all exporter tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add general_motion_retargeting/retarget_export.py tests/test_retarget_export.py
git commit -m "feat: export retarget motion artifacts"
```

### Task 3: Integrate fixed exports into the SMPL-X command

**Files:**
- Modify: `scripts/smplx_to_robot.py`
- Create: `tests/test_smplx_to_robot_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 `export_retarget_motion(...)`.
- Produces: `build_parser()` and `main(argv=None)` returning `0`, with exports enabled by `--save`.

- [ ] **Step 1: Write failing CLI tests**

```python
def test_parser_replaces_save_path_with_save():
    args = build_parser().parse_args(["--smplx_file", "walk.npz", "--robot", "dr02", "--save"])
    assert args.save is True
    assert not hasattr(args, "save_path")

def test_main_rejects_looping_save(capsys):
    with pytest.raises(SystemExit):
        main(["--smplx_file", "walk.npz", "--robot", "dr02", "--save", "--loop"])
```

Add a monkeypatched one-frame conversion test asserting the exporter receives the retargeter's exact `model`, robot, source path, aligned FPS, and stacked qpos frames.

- [ ] **Step 2: Run CLI tests and verify import/interface failure**

Run: `pytest tests/test_smplx_to_robot_cli.py -v`

Expected: FAIL because the script has no import-safe parser or `main(argv)`.

- [ ] **Step 3: Refactor the command into import-safe functions**

Move parser construction to `build_parser()` and runtime code to `main(argv=None)`. Remove `--save_path`, add `--save`, reject `--save --loop` with `parser.error`, collect `qpos.copy()` only when saving, and call:

```python
export_retarget_motion(
    retarget.model, args.robot, args.smplx_file, aligned_fps,
    np.asarray(qpos_list),
)
```

Close the viewer in `finally` so export or validation failures do not leak it.

- [ ] **Step 4: Document the command and output tree**

Add a concise README example using `--smplx_file`, `--robot`, `--no_viewer`, and `--save`, followed by the four output paths and the statement that `joints.json` is shared by all motions for that robot.

- [ ] **Step 5: Run focused and regression tests**

Run: `pytest tests/test_smplx_to_robot_cli.py tests/test_retarget_export.py tests/test_rerun_motion.py -v`

Expected: all pass.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 6: Commit Task 3**

```bash
git add scripts/smplx_to_robot.py tests/test_smplx_to_robot_cli.py README.md
git commit -m "feat: add fixed SMPL-X retarget exports"
```

### Task 4: Final verification

**Files:**
- Verify only; modify earlier files only if a test exposes a defect.

**Interfaces:**
- Consumes: completed Tasks 1–3.
- Produces: evidence that the feature and existing suite pass.

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Inspect repository state and artifact scope**

Run: `git status --short && git log -4 --oneline`

Expected: only intentional changes remain and the design/implementation commits are visible.

