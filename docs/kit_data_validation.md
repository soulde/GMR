# KIT AMASS SMPL-X Data Validation

Dataset root: `data/AMASS_SMPLX/KIT`

Validation date: 2026-07-10

## Inventory

- Total size: 26.9 GiB
- `.npz` files: 4287
- Motion files: 4232 `*_stageii.npz`
- Subject/body files: 55 `*_stagei.npz`
- Other `.npz` files: 0
- License file: `LICENSE.txt`

Largest subject folders include `3` (3.4 GiB), `675` (2.3 GiB), `572` (1.6 GiB), `969` (1.4 GiB), and `674` (1.2 GiB).

## Schema Checks

All 4232 `*_stageii.npz` files were opened with numpy. Required SMPL-X fields were present:

- `gender`
- `surface_model_type`
- `mocap_frame_rate`
- `mocap_time_length`
- `trans`
- `poses`
- `betas`
- `root_orient`
- `pose_body`
- `pose_hand`
- `pose_jaw`
- `pose_eye`

Results:

- Load errors: 0
- Missing required fields: 0
- NaN/Inf in required numeric arrays: 0
- Frame-length mismatches across required per-frame arrays: 0
- FPS: all 4232 files are 120.0
- Surface model: all 4232 files are `smplx`
- Gender split: 2973 male, 1259 female
- Frame count range: 1 to 26363
- Mean frame count: 938.34

## Usability Check

The repo expects raw AMASS SMPL-X files and reads them through:

- `general_motion_retargeting.utils.smpl.load_smplx_file`
- `general_motion_retargeting.utils.smpl.get_smplx_data_offline_fast`
- `scripts/smplx_to_robot.py`
- `scripts/smplx_to_robot_dataset.py`

Single-motion smoke test:

```bash
.venv/bin/python scripts/smplx_to_robot.py \
  --smplx_file data/AMASS_SMPLX/KIT/883/wipe_arm_bigcircle01_stageii.npz \
  --robot unitree_g1 \
  --save_path /tmp/gmr_kit_smoke_wipe_arm_bigcircle01.pkl \
  --no_viewer
```

Result: success. Output pickle contained `fps`, `root_pos`, `root_rot`, `dof_pos`, `local_body_pos`, and `link_body_list`; the generated motion had 174 frames and 29 robot DoF.

## Required Exclusion

One file is structurally valid but not usable by the current downsampling path:

```text
data/AMASS_SMPLX/KIT/9/WalkingStraightBackwards08_stageii.npz
```

It has only 1 source frame at 120 FPS. `get_smplx_data_offline_fast(..., tgt_fps=30)` computes zero target frames and fails with:

```text
ValueError: need at least one array to stack
```

The non-destructive exclusion list is:

```text
data/AMASS_SMPLX/KIT_EXCLUDE.txt
```

## Conclusion

Status: partially usable.

4231 of 4232 motion files pass schema checks and the representative end-to-end SMPL-X retarget path works. The dataset should be used with `9/WalkingStraightBackwards08_stageii.npz` excluded, or the downsampling function should be patched to handle very short sequences.

No raw dataset files were modified.
