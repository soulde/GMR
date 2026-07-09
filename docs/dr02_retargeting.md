# DR02 Retargeting

DR02 assets are copied from `soulde/mjlabplusplus` into:

```text
assets/robots/dr02/
├── dr02.xml
├── dr02_std.urdf
└── assets/
```

The MJCF mesh directory is set to `assets`, relative to `assets/robots/dr02/dr02.xml`.

## Install

Create and activate the GMR environment as described by the upstream project:

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
```

If rendering has C++ runtime issues on Ubuntu:

```bash
conda install -c conda-forge libstdcxx-ng -y
```

## SMPL-X Body Models

Download SMPL-X body models from the SMPL-X website and place them under:

```text
assets/body_models/smplx/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_FEMALE.pkl
├── SMPLX_MALE.pkl
├── SMPLX_NEUTRAL.npz
├── SMPLX_FEMALE.npz
└── SMPLX_MALE.npz
```

The default `smplx` package looks for `.npz`. If using only SMPL-X `.pkl` files, follow the upstream GMR note and set `ext` in `smplx/body_models.py` from `npz` to `pkl`.

## AMASS SMPL-X Data

Download AMASS SMPL-X motion files and place them anywhere convenient, for example:

```text
data/AMASS_SMPLX/KIT/
```

Use raw SMPL-X AMASS data, not SMPL+H data.

## Inspect DR02

Run:

```bash
python scripts/inspect_dr02.py
```

Expected model dimensions:

```text
nq = 28
nv = 27
nu = 21
```

The script also prints MuJoCo body, joint, actuator, site, and geom names. The first DR02 retargeting config uses `base_link`, `body`, `left_knee_link`, `right_knee_link`, `left_ankle_x_link`, and `right_ankle_x_link`.

## Retarget One Motion

```bash
python scripts/smplx_to_robot.py \
  --smplx_file data/AMASS_SMPLX/KIT/<some_motion>.npz \
  --robot dr02 \
  --save_path retargeting_data/dr02/test_walk.pkl \
  --rate_limit
```

The first DR02 IK config intentionally tracks only pelvis/root, torso, knees, and feet. This keeps the first pass stable before adding low-weight arm constraints.

On a headless machine without `DISPLAY`, use:

```bash
python scripts/smplx_to_robot.py \
  --smplx_file data/AMASS_SMPLX/KIT/<some_motion>.npz \
  --robot dr02 \
  --save_path retargeting_data/dr02/test_walk.pkl \
  --rate_limit \
  --no_viewer
```

## Replay

Use the DR02 kinematic replay script:

```bash
python scripts/vis_dr02_retargeted_motion.py \
  --motion retargeting_data/dr02/test_walk.pkl \
  --xml assets/robots/dr02/dr02.xml
```

The replay script directly writes `qpos[:3]`, `qpos[3:7]`, and `qpos[7:]`, then calls `mujoco.mj_forward`.

On a headless machine:

```bash
python scripts/vis_dr02_retargeted_motion.py \
  --motion retargeting_data/dr02/test_walk.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --no_viewer
```

## Common Issues

If `python` is not available, use the Python executable from your GMR environment, for example `python3` or `.venv/bin/python`.

If mesh loading fails, verify that `assets/robots/dr02/dr02.xml` contains:

```xml
<compiler angle="radian" meshdir="assets" />
```

If the robot is rotated, mirrored, or facing the wrong way, adjust the quaternion offsets in `general_motion_retargeting/ik_configs/smplx_to_dr02.json`.

If the feet penetrate or float above the ground, adjust `ground_height` and the foot position offsets in `smplx_to_dr02.json`.

If joints jump or the torso twists strongly, reduce torso orientation weight first and keep foot tracking as the priority.
