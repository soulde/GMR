---
name: validate-modify-data
description: Use when working in this repository to validate, inspect, modify, retarget, post-check, or judge usability of local motion/robotics datasets such as AMASS, SMPL-X, BVH, NPZ, GMR PKL, retargeted motions, or project-specific data before loaders, converters, robot retargeting, training, simulation, or visualization.
---

# Validate Modify Data

## Overview

Treat dataset work as an evidence chain: identify what exists, compare it with what this repo expects, prove a minimal path works, retarget with traceable inputs/outputs, validate generated robot motion, then modify data only with a reversible plan.

This skill is project-bound. Prefer repository code, configs, docs, and existing scripts over generic dataset assumptions.

## Workflow

### 1. Locate the Data Boundary

Find the dataset root, archive source, generated directories, and any existing outputs.

Use read-only inventory first:

```bash
./.venv/bin/python skills/validate-modify-data/scripts/inspect_dataset.py <data-root>
```

If the project virtualenv is unavailable, use `python3`; `.npz` shape inspection requires numpy.

Also inspect:

- top-level directories and file extensions
- total size and largest subdirectories
- sample file names
- recent files when data was just extracted or generated

Do not classify `stagei`, calibration, model, metadata, or body-shape files as motion samples until verified.

### 2. Learn the Repo's Expectations

Search before modifying:

```bash
rg -n "(data|dataset|motion|npz|bvh|amass|smpl|retarget|loader|fps|frame|pose|trans|root_orient|qpos|dof)"
```

Read the relevant loader, converter, config, README, docs, and test scripts. Extract concrete expectations:

- path conventions
- file extensions
- required keys or columns
- array shapes and dtypes
- frame rate, units, coordinate frame, and skeleton/body model
- naming rules for subjects, actions, splits, robots, or output files

### 3. Inspect Representative Samples

Open a small, intentional sample: first file, largest file, shortest file if discoverable, and one file per important subtype.

For `.npz`, print keys, shapes, dtypes, scalar metadata, and frame counts. For BVH/text/JSON/YAML/CSV, inspect headers and a few rows. Prefer structured parsers over string guessing when available.

Check for:

- missing required keys
- inconsistent shapes or lengths across fields
- NaN/Inf values
- empty files or zero-frame motions
- unexpected coordinate/unit/frame-rate changes
- corrupt files that fail to load

### 4. Prove Source Usability

Run the smallest existing repo path that exercises the data:

- data loader import or direct load
- converter dry-run
- one-file retargeting attempt
- viewer on one known sample
- project test that covers the format

Record the exact command and whether failure is caused by data, config, dependency, path, or code behavior.

### 5. Retarget With a Small Gate First

Retarget one representative source file before running a folder job. Use `--no_viewer` when available and save to `/tmp` or a scratch output path:

```bash
./.venv/bin/python scripts/smplx_to_robot.py \
  --smplx_file <source.npz> \
  --robot <robot_name> \
  --save_path /tmp/<name>.pkl \
  --no_viewer
```

For folder retargeting, create or reuse an exclude list for known-bad source files. Check the batch script's actual filtering behavior; do not assume a manifest is honored unless the code reads it.

Before a long batch run, state:

- source folder and target folder
- robot name and IK config path
- include/exclude rules
- CPU/GPU assumptions and memory risk
- expected output extension and folder mapping
- resume/override behavior
- log path and failure collection method

Use existing commands where possible:

```bash
./.venv/bin/python scripts/smplx_to_robot_dataset.py \
  --src_folder <source-dir> \
  --tgt_folder <target-dir> \
  --robot <robot_name> \
  --num_cpus <n>
```

For DR02 training-data generation, prefer the project batch pipeline because it retargets, quality-checks, preprocesses, exports training `.npz`, and writes a summary. Preserve the source tree when the user wants outputs organized like the raw data:

```bash
./.venv/bin/python scripts/batch_dr02_retarget_pipeline.py \
  --input-dir <source-dir> \
  --output-dir <workspace-output-dir> \
  --reports-dir <workspace-reports-dir> \
  --exclude-file <exclude-list.txt> \
  --preserve-tree \
  --jobs <n> \
  --auto-start-frame
```

Prefer small subsets first when a full run would be expensive. Preserve raw retarget outputs; put adjusted or preprocessed outputs in a separate directory.

### 6. Check Retargeted Outputs

Validate generated robot motions before using them for training, control, or visualization:

```bash
./.venv/bin/python skills/validate-modify-data/scripts/check_robot_motion.py <retarget-output-root-or-file>
```

Check at least:

- output count vs expected input count after exclusions
- load errors
- required keys: `fps`, `root_pos`, `root_rot` or `root_quat`, `dof_pos` or `joint_pos`
- finite numeric arrays
- consistent frame counts
- minimum frame count after downsampling
- quaternion norms near 1
- DoF count for the target robot
- root height and large jumps

For robot-specific checks, use repo tools when present:

- `scripts/vis_robot_motion.py` for a visual spot check
- `scripts/check_dr02_motion_quality.py` for DR02 kinematic metrics
- `scripts/preprocess_dr02_motion.py` and `scripts/check_dr02_motion_dataset.py` for DR02 dataset packaging
- `scripts/pd_replay_dr02_motion.py` for DR02 PD smoke tests
- `scripts/batch_dr02_retarget_pipeline.py` when the task is a DR02 end-to-end batch

Classify each output as `PASS`, `WARN`, or `FAIL`. Failures should feed back into the exclude list or a documented data adjustment.

### 7. Adjust Data Deliberately

Data adjustment is allowed only after a failing check names the problem. Common adjustments in this repo include:

- exclude zero-frame or too-short source motions
- normalize or add missing `root_quat`/`joint_pos` aliases for downstream tools
- height-offset retargeted root positions so the lowest body/foot is not below ground
- reset XY origin to the first frame
- trim bad leading frames
- resample to expected FPS
- clamp or flag joint-limit violations
- add velocities/contact fields for dataset consumers

For every adjustment, write to a new path such as `preprocessed/`, `dataset/`, or `<name>_fixed.pkl` unless the user explicitly requests in-place edits. Record the transform, input, output, skipped files, and validation command.

Do not hide retargeting failures by silently deleting outputs. Keep a failure manifest with source path, target path, error, and proposed action.

### 8. Plan Modifications Before Writing

Before changing data, state:

- source path and output path
- transformation rules
- whether edits are in-place or copied
- validation command to run afterward
- rollback strategy

Default to writing a new output directory or manifest. Only edit in place when the user explicitly asks and the change is small enough to audit.

### 9. Modify Data Reproducibly

Use scripts for bulk changes. Keep transformations deterministic and logged.

For every modified file or generated artifact, preserve enough information to answer:

- what input produced it
- what fields changed
- what files were skipped and why
- what command can reproduce it

Avoid destructive cleanup until after post-change validation succeeds.

### 10. Revalidate After Changes

Repeat the same inventory, sample inspection, retarget smoke test, output check, and robot-specific quality checks used before modification. Compare before/after counts, schemas, frame counts, DoF counts, and failure manifests.

End with one of these conclusions:

- `usable`: source and retargeted data match the repo path tested
- `partially usable`: some subsets or generated outputs work and blockers are named
- `not usable yet`: blockers prevent the intended path
- `unknown`: a required validation command could not be run

Include the exact next command the user should run or the next fix to make.

## Reporting Format

Keep the final report concise:

- dataset root and size
- sample count by type
- expected schema or loader path
- source validation command and result
- retarget command and output path, when run
- post-retarget validation command and result
- modifications made, if any
- usability conclusion
- remaining risks

## Common Mistakes

- Counting every `.npz` as a motion without excluding body/model/calibration files.
- Trusting filename action labels without checking the actual arrays.
- Modifying data before proving the existing loader path.
- Running a full retarget batch before a one-file smoke test.
- Assuming a generated exclude list is used by batch code without checking the script.
- Treating a saved `.pkl` as usable without checking frame count, DoF, finite arrays, and quaternion convention.
- Fixing one sample and assuming all subjects/actions share the same schema.
- Re-running a different validation after modification, making before/after comparisons meaningless.
