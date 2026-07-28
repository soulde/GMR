# Laikago Kinematic MJCF

This collision-only MJCF represents the Laikago kinematic structure used by
the `motion_imitation` reference motions.

Sources:

- https://github.com/erwincoumans/motion_imitation
- `motion_imitation/robots/laikago_constants.py`
- `retarget_motion/retarget_config_laikago.py`

The joint order, SDK directions, offsets, neutral angles, and hip positions
come from those files. MJCF joint axes and `ref` values encode the SDK-to-model
joint conversion. Primitive geometry keeps source forward kinematics
independent of external mesh assets.

The source project is Apache-2.0 licensed; see `LICENSE`.
