# Laikago MJCF

This MJCF represents the Laikago kinematic structure used by the
`motion_imitation` reference motions. Its visual OBJ meshes are copied from
PyBullet's `pybullet_data/laikago` package and retain their upstream license
in `LICENSE`.

Sources:

- `motion_imitation` commit
  `d0e7b963c5a301984352d25a3ee0820266fa4218`
- `motion_imitation/robots/laikago.py`
- PyBullet `laikago/laikago_toes_limits.urdf`

MJCF link transforms, joint axes, limits, and toe offsets are the official
URDF values transformed by the upstream fixed root rotation into GMR's
`+X`-forward, `+Y`-left, `+Z`-up frame.

Motion reference columns are raw URDF joint states: upstream
`imitation_task.py` writes them with `resetJointStateMultiDof`. The
`JOINT_DIRECTIONS` and `JOINT_OFFSETS` constants in `laikago.py` belong to the
motor command API and must not be applied again during reference-motion FK.
The nominal thigh/calf pose (`0.67`, `-1.25` radians) is represented as MJCF
`qpos0`. Matching static body rotations and joint `ref` values preserve the
original raw-joint FK for every input angle.
Transparent primitive geometry keeps source forward kinematics independent
of the visual mesh assets. The visible OBJ geometry is transformed from the
URDF's native frame into GMR's canonical frame.

The Motion Imitation source project is Apache-2.0 licensed. The Laikago URDF
and meshes were created by Erwin Coumans from Unitree CAD data used by
permission; see `LICENSE`.
