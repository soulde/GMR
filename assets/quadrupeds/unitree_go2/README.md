# Unitree Go2 MJCF

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2

The model is adapted from the MuJoCo Menagerie Go2 model. Mesh visuals were
removed so the vendored model remains compact; collision geometry, inertials,
joint limits, actuators, and the `home` keyframe are retained. Stable
`FL/FR/RL/RR_foot_site` sites were added for retargeting.

License: BSD-3-Clause; see `LICENSE`.
