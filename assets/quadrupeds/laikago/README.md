# Laikago Kinematic MJCF

This collision-only MJCF represents the Laikago kinematic structure used by
the `motion_imitation` reference motions.

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
Primitive geometry keeps source forward kinematics independent of external
mesh assets.

The source project is Apache-2.0 licensed; see `LICENSE`.
