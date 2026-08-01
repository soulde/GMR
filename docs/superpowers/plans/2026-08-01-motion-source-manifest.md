# Motion Source Manifest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every exported robot motion's source SMPL-X file and let the GMR debug viewer infer reference, MJCF, and IK config from only `--motion`.

**Architecture:** Extend `retarget_export.py` with a versioned manifest boundary that owns source-path encoding, atomic updates, and exact motion lookup. The exporter publishes the manifest only after all artifacts succeed. `vis_gmr_debug.py` resolves optional CLI inputs through that boundary and existing robot configuration maps before entering its unchanged visualization pipeline.

**Tech Stack:** Python 3.10+, pathlib, JSON, tempfile, argparse, pytest, existing MuJoCo/SMPL-X viewer stack.

## Global Constraints

- One `retarget_data/<robot>/manifest.json` is shared by all motions for that robot.
- Manifest format version is exactly `1`; robot and version mismatches are errors.
- Repository sources are repository-relative POSIX paths; external sources are normalized absolute paths.
- Motion entries use the exact PKL path relative to the robot export directory.
- Existing `--reference`, `--mjcf`, and `--ik-config` remain optional per-field overrides.
- The fully explicit legacy viewer invocation does not require a manifest.
- Manifest publication happens only after PKL, NPZ, and CSV artifact writes succeed.
- Preserve the user's untracked `gmr_debug_visualizer.py`, `vis_gmr_debug.py`, and their two test files; do not overwrite them wholesale.

---

### Task 1: Manifest source encoding, update, and lookup

**Files:**
- Modify: `general_motion_retargeting/retarget_export.py`
- Modify: `tests/test_retarget_export.py`

**Interfaces:**
- Consumes: repository root, robot export root, source path, `ExportPaths`.
- Produces: `ManifestEntry`, `encode_source_path(source_path, repository_root, cwd=None)`, `update_motion_manifest(...)`, and `load_motion_manifest(motion_path, repository_root, require_reference=True)`.

- [ ] **Step 1: Write failing source-path encoding tests**

```python
def test_encode_source_inside_repository_is_relative(tmp_path):
    source = tmp_path / "motion_data/walk.npz"
    source.parent.mkdir()
    source.touch()
    assert encode_source_path(source, tmp_path) == {
        "path": "motion_data/walk.npz",
        "base": "repository",
    }

def test_encode_external_source_is_absolute(tmp_path):
    source = tmp_path.parent / "external/walk.npz"
    assert encode_source_path(source, tmp_path) == {
        "path": source.resolve().as_posix(),
        "base": "absolute",
    }
```

Add a relative-argument test proving `cwd` is used before repository containment is decided.

- [ ] **Step 2: Run encoding tests and verify RED**

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py -k encode_source -v`

Expected: FAIL because `encode_source_path` is not defined.

- [ ] **Step 3: Implement source encoding**

Resolve `source_path` against `cwd or Path.cwd()`. Attempt `resolved.relative_to(repository_root.resolve())`; on success return repository form, otherwise absolute form. Always use `as_posix()`.

- [ ] **Step 4: Write failing atomic update tests**

Test initial creation, a second motion preserving the first, same-motion source replacement, lexicographically sorted serialized keys, and exact rejection of these existing headers:

```python
{"format_version": 2, "robot": "dr02", "motions": {}}
{"format_version": 1, "robot": "unitree_g1", "motions": {}}
```

Expected entries contain `source`, `dataset`, and `beyondmimic` exactly as specified in the design.

- [ ] **Step 5: Run update tests and verify RED**

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py -k manifest -v`

Expected: FAIL because manifest update behavior is absent.

- [ ] **Step 6: Implement atomic manifest update**

```python
def update_motion_manifest(
    paths: ExportPaths,
    robot: str,
    source_path: str | Path,
    *,
    repository_root: str | Path,
    cwd: str | Path | None = None,
) -> Path:
    ...
```

Set `manifest_path = paths.motion.parent.parent / "manifest.json"`; require an exact version/robot header when it exists, replace `motions[paths.motion.relative_to(robot_root).as_posix()]`, sort the motion mapping, and atomically replace JSON through a temporary sibling.

- [ ] **Step 7: Write failing lookup tests**

Define the wished-for result:

```python
entry = load_motion_manifest(motion_path, repository_root=repo)
assert entry.robot == "dr02"
assert entry.reference == repo / "motion_data/walk.npz"
assert entry.dataset == robot_root / "datasets/walk.npz"
assert entry.beyondmimic == robot_root / "beyondmimic/walk.csv"
```

Add failures for missing/malformed manifest, unsupported version, exact motion entry missing, unsupported source base, and missing resolved source.

- [ ] **Step 8: Implement strict lookup and run Task 1 tests**

```python
@dataclass(frozen=True)
class ManifestEntry:
    robot: str
    reference: Path | None
    dataset: Path
    beyondmimic: Path
```

Load only `motion.parent.parent / "manifest.json"`, validate schema types and values, require the exact relative motion key, and resolve source according to `base`. When `require_reference=True`, require that source file and return its path; when false, return `reference=None` so an explicit Viewer reference can recover from a stale manifest source. Always return the artifact paths relative to the robot root.

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py -v`

Expected: all exporter and manifest tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add general_motion_retargeting/retarget_export.py tests/test_retarget_export.py
git commit -m "feat: add motion source manifest"
```

### Task 2: Publish manifest after successful artifact export

**Files:**
- Modify: `general_motion_retargeting/retarget_export.py`
- Modify: `tests/test_retarget_export.py`

**Interfaces:**
- Consumes: Task 1 `update_motion_manifest` and existing `export_retarget_motion`.
- Produces: `ExportPaths.manifest` and automatic manifest publication from every successful export.

- [ ] **Step 1: Write failing export integration tests**

Extend the artifact test to assert:

```python
assert paths.manifest == tmp_path / "dr02/manifest.json"
entry = load_motion_manifest(paths.motion, repository_root=repository_root)
assert entry.robot == "dr02"
assert entry.reference == source.resolve()
```

Add a serialization-failure test by monkeypatching `np.savez` to raise and assert `paths.manifest` does not exist.

- [ ] **Step 2: Run integration tests and verify RED**

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py -k 'writes_all_artifacts or publish' -v`

Expected: FAIL because `ExportPaths` has no manifest and export does not update it.

- [ ] **Step 3: Add manifest path and publish last**

Add `manifest: pathlib.Path` to `ExportPaths`, set it to `base / "manifest.json"`, add `repository_root` and `cwd` keyword parameters to `export_retarget_motion`, and call `update_motion_manifest` only after the CSV temporary file has been replaced.

- [ ] **Step 4: Update SMPL-X CLI test expectations**

Update the `ExportPaths` fixture in `tests/test_smplx_to_robot_cli.py` with its manifest path. Do not assert on mock internals; keep asserting the real CLI return and captured exporter boundary arguments.

- [ ] **Step 5: Run export and CLI tests**

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py tests/test_smplx_to_robot_cli.py -v`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add general_motion_retargeting/retarget_export.py tests/test_retarget_export.py tests/test_smplx_to_robot_cli.py
git commit -m "feat: publish manifests with retarget exports"
```

### Task 3: Infer debug Viewer inputs from motion manifest

**Files:**
- Modify carefully: `scripts/vis_gmr_debug.py`
- Modify carefully: `tests/test_vis_gmr_debug.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 `load_motion_manifest`, `ROBOT_XML_DICT`, and `IK_CONFIG_DICT`.
- Produces: `ResolvedViewerInputs` and `resolve_viewer_inputs(args, repository_root)`; parser requires only `--motion`.

- [ ] **Step 1: Characterize the user's current untracked Viewer baseline**

Run: `PYTHONPATH=. uv run pytest tests/test_vis_gmr_debug.py tests/test_gmr_debug_visualizer.py -v`

Expected: all existing debug-viewer tests pass before edits. Save `git diff --no-index /dev/null <file>` output only for inspection; do not replace or stage unrelated user content.

- [ ] **Step 2: Write failing parser and inference tests**

Change the parser test to prove only `--motion` is required. Add a real manifest fixture and assert:

```python
resolved = resolve_viewer_inputs(args, repository_root=repo)
assert resolved.reference == source
assert resolved.mjcf == ROBOT_XML_DICT["dr02"]
assert resolved.ik_config == IK_CONFIG_DICT["smplx"]["dr02"]
```

Add independent tests that each explicit `--reference`, `--mjcf`, or `--ik-config` wins over its inferred value, and that all three explicit values work with no manifest.

- [ ] **Step 3: Run Viewer resolution tests and verify RED**

Run: `PYTHONPATH=. uv run pytest tests/test_vis_gmr_debug.py -v`

Expected: FAIL because parser arguments are required and `resolve_viewer_inputs` is absent.

- [ ] **Step 4: Implement input resolution without changing rendering**

```python
@dataclass(frozen=True)
class ResolvedViewerInputs:
    reference: Path
    mjcf: Path
    ik_config: Path
```

Make the three parser arguments optional. If all are explicit, return them after file validation. Otherwise load the manifest once with `require_reference=args.reference is None`, use explicit values where present, and infer remaining values from its robot and project maps. Raise focused `ValueError`/`FileNotFoundError` messages for missing robot keys or files.

In `main`, call `resolve_viewer_inputs` before loading frames and replace only `args.reference`, `args.mjcf`, and `args.ik_config` reads with the resolved values. Use `inputs.reference.stem` for the display name. Leave visualization math and update behavior intact.

- [ ] **Step 5: Add failure tests and verify compatibility**

Add cases for absent manifest, absent motion entry, missing inferred source, unknown robot, missing inferred MJCF, and missing inferred IK file. Preserve the existing complete-argument parser test as the legacy compatibility proof.

Run: `PYTHONPATH=. uv run pytest tests/test_vis_gmr_debug.py tests/test_gmr_debug_visualizer.py -v`

Expected: all debug Viewer tests pass.

- [ ] **Step 6: Document the simplified command**

Add this example near the existing visualization documentation:

```bash
PYTHONPATH=. uv run python scripts/vis_gmr_debug.py \
  --motion retarget_data/dr02/motions/walk.pkl
```

Document that the three old parameters are optional overrides and that newly exported motions populate `manifest.json` automatically.

- [ ] **Step 7: Run focused regression suite and inspect changes**

Run: `PYTHONPATH=. uv run pytest tests/test_retarget_export.py tests/test_smplx_to_robot_cli.py tests/test_vis_gmr_debug.py tests/test_gmr_debug_visualizer.py -v`

Expected: all pass.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only planned files plus the user's pre-existing untracked files appear.

- [ ] **Step 8: Commit Task 3 without absorbing unrelated user files**

After explicit user consent to version the previously untracked Viewer files, stage only the targeted files and preserve all of their baseline content:

```bash
git add scripts/vis_gmr_debug.py tests/test_vis_gmr_debug.py README.md
git commit -m "feat: infer debug viewer inputs from manifest"
```

### Task 4: Final verification

**Files:**
- Verify only; modify an earlier planned file only if a failing test demonstrates a defect.

**Interfaces:**
- Consumes: completed Tasks 1–3.
- Produces: final evidence for the merged feature.

- [ ] **Step 1: Run the complete test suite**

Run: `PYTHONPATH=. uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Verify format and repository scope**

Run: `git diff --check && git status --short && git log -5 --oneline`

Expected: no whitespace errors, implementation commits are present, and no unrelated user file has been added accidentally.
