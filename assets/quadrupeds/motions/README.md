# motion_imitation Reference Motion

Download the reference trajectory before running the quadruped example:

```bash
python scripts/download_quadruped_motions.py
```

With no arguments, the script downloads every `.txt` trajectory in the
upstream `motion_imitation/data/motions` directory. A single named motion can
also be selected, for example:

```bash
python scripts/download_quadruped_motions.py dog_trot
```

The upstream revision is pinned and every file is verified against its
SHA-256 checksum. Trajectory files are intentionally not tracked by this
repository. The source project is Apache-2.0 licensed; see `LICENSE`.
