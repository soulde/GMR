# GMR Motion Coordinate Conventions

Every source adapter must convert its input into the GMR internal convention
before retargeting. Source arrays must not be passed directly to a retargeter
based only on matching shapes.

## Internal Coordinate Contract

GMR's MuJoCo-facing internal motion representation uses:

- right-handed Cartesian coordinates
- `+X`: robot forward
- `+Y`: robot left
- `+Z`: up
- position unit: metres
- joint-angle unit: radians
- time unit: seconds
- root position shape: `[T, 3]`
- root rotation shape: `[T, 4]`
- joint position shape: `[T, J]`

Internal root quaternions are scalar-first `wxyz`. They are active
local-to-world rotations suitable for direct assignment to a MuJoCo free
joint:

```python
data.qpos[root_qpos_address : root_qpos_address + 3] = root_pos
data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = root_rot_wxyz
```

Quaternions must be finite, normalized, and temporally sign-continuous when
interpolation or finite differencing is required. Joint columns must follow
the explicit source specification; array offsets must not be inferred from
names or assumed to match target qpos order.

## Source Adapter Responsibilities

An adapter must explicitly define and verify:

1. Position units and scale.
2. World handedness and up axis.
3. Which horizontal axis is forward.
4. Quaternion component order.
5. Whether the quaternion is active or passive.
6. Whether it maps local-to-world or world-to-local.
7. The source model's fixed root-frame orientation.
8. Joint column order, angle units, axes, signs, and reference offsets.
9. Frame duration or FPS.

The adapter must transform root translation and orientation into the internal
contract. A quaternion reorder alone is insufficient when source and target
root frames differ.

For robot-to-robot data whose first root orientation contains a fixed source
model frame, use an explicit source configuration such as:

```yaml
motion:
  quaternion_order: wxyz
  root_orientation_mode: relative_first_frame
```

`relative_first_frame` computes:

```text
R_gmr(t) = inverse(R_source(0)) * R_source(t)
```

This makes the first root orientation identity while preserving subsequent
orientation changes in the initial source-root frame. Use `absolute` only
when the source already satisfies the GMR world and root-frame contract.

If the source world axes differ from GMR, apply a declared basis transform to
both positions and rotations. Do not repair this later in the viewer or at
the output serialization boundary.

## Saved GMR Motion Format

The standard saved pickle uses:

```python
{
    "fps": float,
    "root_pos": np.ndarray,  # [T, 3], metres, GMR world axes
    "root_rot": np.ndarray,  # [T, 4], xyzw
    "dof_pos": np.ndarray,   # [T, J], radians, target order
}
```

Saved `root_rot` is vector-first `xyzw` for compatibility with existing GMR
tools. Serialization performs only the component reorder:

```text
[w, x, y, z] -> [x, y, z, w]
```

Coordinate or root-frame correction must already be complete before this
step. Loading for MuJoCo converts `xyzw` back to internal `wxyz`.

## Required Acceptance Checks

Every new source format or robot configuration needs a representative
end-to-end test that verifies:

- the known identity quaternion decodes as identity
- the first aligned frame has expected roll, pitch, and heading
- a known forward trajectory moves along `+X`
- standing feet are below the trunk along `-Z`
- left and right feet have the expected signs on `Y`
- saved and reloaded root rotations represent the same orientation
- target joint columns resolve by MuJoCo joint metadata
- all values are finite and quaternions are normalized
- visual playback shows an upright robot moving in the expected direction

For periodic motions, also inspect the last-to-first position and orientation
behavior declared by the source loop mode.

## Motion Imitation Laikago

The vendored `dog_pace.txt` uses:

- metres and radians
- root quaternion order `wxyz`
- a fixed initial Laikago root-frame orientation
- motion joint order declared in `laikago.yaml`

It therefore requires `root_orientation_mode: relative_first_frame`. Reading
it as `xyzw` or transferring its absolute root quaternion directly produces
an approximately 90-degree rolled Go2.
