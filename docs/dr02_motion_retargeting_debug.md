# DR02 Motion Retargeting Debug

Kinematic replay and PD replay answer different questions.

Kinematic replay directly writes `root_pos`, `root_quat`, and `joint_pos` into MuJoCo `qpos`, then calls `mj_forward`. It checks coordinate frames, body mapping, joint order, and whether the reference pose looks reasonable.

PD replay is a diagnostic tool. It is not the final controller. A motion can look correct in kinematic replay and still fall in free-base PD replay because the trajectory may have foot slip, inconsistent root dynamics, bad initial phase, contact mismatch, torque saturation, or no stabilizing controller.

## Recommended Flow

1. Run GMR and inspect kinematic replay.
2. Run motion quality checks.
3. Preprocess the clip with stand and blend-in frames.
4. Run PD replay diagnostics at slow speed.
5. Export an imitation / motion tracking dataset.
6. Use motion tracking RL for final stabilization.

## Motion Quality

```bash
python scripts/check_dr02_motion_quality.py \
  --motion retargeting_data/dr02/test_walk.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --out reports/dr02_motion_quality/test_walk
```

Outputs include:

```text
summary.txt
metrics.csv
joint_range.csv
foot_height.csv
foot_xy_velocity.csv
contact_labels.csv
plots/
```

Read `summary.txt` first. Joint limit warnings above 1 percent are worth inspecting. Errors above 10 percent mean the motion should not be treated as a valid PD replay reference.

## Preprocess

```bash
python scripts/preprocess_dr02_motion.py \
  --motion retargeting_data/dr02/test_walk.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --out retargeting_data/dr02/test_walk_preprocessed.pkl \
  --stand-time 1.0 \
  --blend-time 1.0
```

Use `--auto-start-frame` to choose a lower-velocity, more stable start frame based on foot height symmetry, base roll/pitch, base height, and joint velocity.

## PD Replay

```bash
python scripts/pd_replay_dr02_motion.py \
  --motion retargeting_data/dr02/test_walk_preprocessed.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --out reports/dr02_pd_replay/test_walk \
  --speed 0.5
```

The script writes torque PD into `data.qfrc_applied[6:]`:

```text
tau = kp * (q_ref - q) + kd * (dq_ref - dq)
```

It logs fall time, base height, roll/pitch, joint tracking error, torque saturation, contact force, and reference foot slip.

For first-pass leg/base diagnosis, disable arm torques so passive arm gravity holding does not dominate the saturation report:

```bash
python scripts/pd_replay_dr02_motion.py \
  --motion retargeting_data/dr02/test_walk_preprocessed.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --out reports/dr02_pd_replay/test_walk_no_arms \
  --speed 0.5 \
  --disable-arm-torques
```

Try:

```bash
--speed 0.25
--speed 0.5
--speed 1.0
```

If 0.25x fails immediately, inspect mapping, limits, foot height, and initial state. If slow speeds work longer than 1.0x, the trajectory is probably too aggressive or dynamically inconsistent.

## Dataset Export

```bash
python scripts/preprocess_dr02_motion.py \
  --motion retargeting_data/dr02/test_walk.pkl \
  --xml assets/robots/dr02/dr02.xml \
  --out retargeting_data/dr02/test_walk_dataset.npz \
  --export-dataset
```

Then validate:

```bash
python scripts/check_dr02_motion_dataset.py \
  --dataset retargeting_data/dr02/test_walk_dataset.npz
```

## Batch Pipeline

For large AMASS batches, use the fixed DR02 pipeline entry instead of running each script by hand:

```bash
python scripts/batch_dr02_retarget_pipeline.py \
  --input-dir data/AMASS_SMPLX/KIT \
  --output-dir retargeting_data/dr02 \
  --reports-dir reports/dr02_batch \
  --run-pd-smoke \
  --jobs 1
```

The default quality thresholds live in:

```text
configs/dr02_motion_quality.yaml
```

The batch script writes resumable outputs:

```text
retargeting_data/dr02/raw/
retargeting_data/dr02/preprocessed/
retargeting_data/dr02/dataset/
reports/dr02_batch/motion_quality/
reports/dr02_batch/pd_replay/
reports/dr02_batch/batch_summary.csv
```

`batch_summary.csv` is the main file for filtering clips. It includes:

```text
quality_status
pd_status
dataset_status
final_status
joint_limit_max_violation
left_foot_min_height
right_foot_min_height
left_support_max_xy_vel
right_support_max_xy_vel
flight_ratio
pd_fall_time
pd_max_torque_ratio
```

Recommended first pass:

```bash
python scripts/batch_dr02_retarget_pipeline.py \
  --input-dir data/AMASS_SMPLX/KIT \
  --output-dir retargeting_data/dr02 \
  --reports-dir reports/dr02_batch \
  --limit 20 \
  --run-pd-smoke \
  --jobs 1
```

Then inspect `reports/dr02_batch/batch_summary.csv`. Use `PASS` clips first for imitation / motion tracking. Keep `WARN` clips for later review. Drop or repair `FAIL` clips before training.

The script skips existing outputs by default, so interrupted batch runs can be resumed. Use `--force` only when retargeting settings, quality thresholds, or preprocessing settings changed and outputs must be regenerated.

For faster non-PD data generation:

```bash
python scripts/batch_dr02_retarget_pipeline.py \
  --input-dir data/AMASS_SMPLX/KIT \
  --output-dir retargeting_data/dr02 \
  --reports-dir reports/dr02_batch \
  --jobs 4
```

Keep `--run-pd-smoke` at `--jobs 1` or a small job count first. MuJoCo simulation and SMPL-X loading are CPU-heavy, and parallel viewer-free PD smoke tests can make failures harder to read.

## What To Fix Where

Fix in GMR retargeting or preprocessing:

```text
wrong direction
left/right swap
foot penetration/floating
joint limit violation
large foot slip during support
large joint velocity spikes
bad start phase
```

Leave for motion tracking RL or a controller:

```text
free-base balance
push recovery
contact force regulation
torque allocation under disturbance
stable command-conditioned walking
```

PD replay falling does not mean GMR mapping failed. If kinematic replay is normal, joint limits are clean, and contact geometry is plausible, the clip can move to motion tracking RL.
