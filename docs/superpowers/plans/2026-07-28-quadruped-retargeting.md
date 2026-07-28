# Quadruped Robot-to-Robot Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget `motion_imitation` quadruped joint-space motions from a declared source MJCF to Unitree Go2 while preserving the existing humanoid GMR path.

**Architecture:** Add an independent `QuadrupedRobotRetargeter` that converts source joint motion to trunk-relative foot trajectories with MuJoCo FK, scales those trajectories to Go2 morphology, and solves Go2 trunk/foot targets with Mink. Both source and target models are MJCF; YAML supplies explicit leg semantics and motion column order.

**Tech Stack:** Python 3.10, NumPy, SciPy, MuJoCo, Mink, PyYAML, pytest.

## Global Constraints

- Work only on `dev/quadruped` in `/tmp/GMR-dev-quadruped`.
- Preserve `GeneralMotionRetargeting` behavior and its public constructor.
- Select quadruped behavior with `model_type="quadruped"`; default remains `"humanoid"`.
- Phase 1 supports four legs with three hinge joints per leg.
- Phase 1 source and target models must be MJCF.
- First source format is `motion_imitation` JSON-formatted `.txt`.
- First target is `unitree_go2`.
- Do not implement wheel joints, contact inference, support-foot anchoring, foot-slip correction, dynamics optimization, or URDF conversion.
- Ground handling must not exceed the current GMR main flow: apply a vertical offset based on the lowest foot point.
- Internal quaternions use scalar-first `wxyz`; saved GMR motion continues to use the repository's existing serialization convention.
- Use TDD, run focused tests after every implementation step, and commit after each task.

---

### Task 1: Add Verified Quadruped MJCF Assets and Semantic Configurations

**Files:**
- Create: `assets/quadrupeds/laikago/laikago.xml`
- Create: `assets/quadrupeds/laikago/LICENSE`
- Create: `assets/quadrupeds/laikago/README.md`
- Create: `assets/quadrupeds/unitree_go2/go2.xml`
- Create directory: `assets/quadrupeds/unitree_go2/assets/`
- Create: `assets/quadrupeds/unitree_go2/LICENSE`
- Create: `assets/quadrupeds/unitree_go2/README.md`
- Create: `assets/quadrupeds/motions/dog_pace.txt`
- Create: `assets/quadrupeds/motions/LICENSE`
- Create: `assets/quadrupeds/motions/README.md`
- Create: `general_motion_retargeting/quadruped/configs/laikago.yaml`
- Create: `general_motion_retargeting/quadruped/configs/unitree_go2.yaml`
- Create: `tests/quadruped/test_assets.py`

**Interfaces:**
- Consumes: MuJoCo MJCF files and the `motion_imitation` Laikago 12-joint frame convention.
- Produces: Loadable source and target `mujoco.MjModel` files plus explicit FL/FR/RL/RR semantic YAML consumed by Task 3.

- [ ] **Step 1: Import the Go2 model with its license**

Copy `unitree_go2/go2.xml`, its referenced `assets/`, `LICENSE`, and model
`README.md` from MuJoCo Menagerie into
`assets/quadrupeds/unitree_go2/`. Keep upstream filenames unchanged inside the
asset directory. Add sites named `FL_foot_site`, `FR_foot_site`,
`RL_foot_site`, and `RR_foot_site` at the corresponding foot bodies when the
upstream model does not provide those stable names. Record the upstream
repository and imported commit SHA in the local README:

```markdown
# Unitree Go2 MJCF

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2
License: BSD-3-Clause; see `LICENSE`.

The model is vendored so GMR retargeting and visualization do not depend on a
runtime network checkout.
```

- [ ] **Step 2: Add the Laikago kinematic MJCF and provenance**

Create a free-base, 12-hinge Laikago MJCF whose joint transforms, axes, limits,
and neutral pose match the `motion_imitation` Laikago model. The required
runtime names are:

```text
root body: trunk
FR: FR_hip_joint, FR_thigh_joint, FR_calf_joint, FR_foot
FL: FL_hip_joint, FL_thigh_joint, FL_calf_joint, FL_foot
RR: RR_hip_joint, RR_thigh_joint, RR_calf_joint, RR_foot
RL: RL_hip_joint, RL_thigh_joint, RL_calf_joint, RL_foot
```

Each foot body must contain a zero-radius-independent site named
`<LEG>_foot_site`. Use primitive collision geometry so mesh files are not
required for source FK. Document that the kinematic parameters are adapted
from the Apache-2.0 `motion_imitation` Laikago model and include that license.

- [ ] **Step 3: Vendor the reference motion with its license**

Copy `motion_imitation/data/motions/dog_pace.txt` to
`assets/quadrupeds/motions/dog_pace.txt`. Copy the upstream Apache-2.0 license
to `assets/quadrupeds/motions/LICENSE` and record the upstream repository and
imported commit SHA in `assets/quadrupeds/motions/README.md`. Tests and example
commands must use this offline path.

- [ ] **Step 4: Write failing asset-load tests**

```python
# tests/quadruped/test_assets.py
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[2]


def test_laikago_mjcf_has_required_semantic_names():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/laikago/laikago.xml")
    )
    for name in ("trunk", "FR_foot", "FL_foot", "RR_foot", "RL_foot"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    for leg in ("FR", "FL", "RR", "RL"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"
        ) >= 0
        for segment in ("hip", "thigh", "calf"):
            assert mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{leg}_{segment}_joint",
            ) >= 0


def test_go2_mjcf_has_free_root_and_twelve_hinge_joints():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets/quadrupeds/unitree_go2/go2.xml")
    )
    joint_types = model.jnt_type.tolist()
    assert joint_types.count(mujoco.mjtJoint.mjJNT_FREE) == 1
    assert joint_types.count(mujoco.mjtJoint.mjJNT_HINGE) == 12
    for leg in ("FL", "FR", "RL", "RR"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site"
        ) >= 0
```

- [ ] **Step 5: Run tests and verify the assets fail until complete**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_assets.py -v
```

Expected before both assets are complete: FAIL with an MJCF load or missing-name
error. Expected after completing the assets: `2 passed`.

- [ ] **Step 6: Add explicit semantic YAML**

Use the stable names required by the asset test. The Laikago motion order must
match the 12 values stored after root position and quaternion:

```yaml
# general_motion_retargeting/quadruped/configs/laikago.yaml
robot: laikago
model_type: quadruped
mjcf_path: assets/quadrupeds/laikago/laikago.xml
root_body: trunk
legs:
  FL:
    joints: [FL_hip_joint, FL_thigh_joint, FL_calf_joint]
    foot_site: FL_foot_site
  FR:
    joints: [FR_hip_joint, FR_thigh_joint, FR_calf_joint]
    foot_site: FR_foot_site
  RL:
    joints: [RL_hip_joint, RL_thigh_joint, RL_calf_joint]
    foot_site: RL_foot_site
  RR:
    joints: [RR_hip_joint, RR_thigh_joint, RR_calf_joint]
    foot_site: RR_foot_site
motion:
  quaternion_order: xyzw
  joint_order:
    - FR_hip_joint
    - FR_thigh_joint
    - FR_calf_joint
    - FL_hip_joint
    - FL_thigh_joint
    - FL_calf_joint
    - RR_hip_joint
    - RR_thigh_joint
    - RR_calf_joint
    - RL_hip_joint
    - RL_thigh_joint
    - RL_calf_joint
reference_pose:
  FR_hip_joint: 0.0
  FR_thigh_joint: 0.67
  FR_calf_joint: -1.25
  FL_hip_joint: 0.0
  FL_thigh_joint: 0.67
  FL_calf_joint: -1.25
  RR_hip_joint: 0.0
  RR_thigh_joint: 0.67
  RR_calf_joint: -1.25
  RL_hip_joint: 0.0
  RL_thigh_joint: 0.67
  RL_calf_joint: -1.25
```

Create the Go2 YAML with root body `body`, joints
`<LEG>_hip_joint`, `<LEG>_thigh_joint`, `<LEG>_calf_joint`, the stable
`<LEG>_foot_site` sites added in Step 1, neutral pose
`hip=0.0, thigh=0.9, calf=-1.8`, and per-semantic-joint initial mapping:

```yaml
joint_mapping:
  hip: {sign: 1.0, scale: 1.0}
  thigh: {sign: 1.0, scale: 1.0}
  calf: {sign: 1.0, scale: 1.0}
```

- [ ] **Step 7: Verify and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_assets.py -v
```

Expected: `2 passed`.

Commit:

```bash
git add assets/quadrupeds general_motion_retargeting/quadruped/configs tests/quadruped/test_assets.py
git commit -m "feat: add quadruped MJCF assets and semantics"
```

---

### Task 2: Define Motion Types and Parse motion_imitation Files

**Files:**
- Create: `general_motion_retargeting/quadruped/__init__.py`
- Create: `general_motion_retargeting/quadruped/types.py`
- Create: `general_motion_retargeting/quadruped/loaders/__init__.py`
- Create: `general_motion_retargeting/quadruped/loaders/motion_imitation.py`
- Create: `tests/quadruped/fixtures/motion_imitation_two_frames.txt`
- Create: `tests/quadruped/test_motion_imitation_loader.py`

**Interfaces:**
- Consumes: `load_motion_imitation(path, joint_names, quaternion_order)`.
- Produces: `JointSpaceMotion` and `CanonicalQuadrupedMotion` dataclasses used by all later tasks.

- [ ] **Step 1: Write the fixture**

```json
{
  "LoopMode": "Wrap",
  "FrameDuration": 0.02,
  "Frames": [
    [0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0, 0.0, 0.67, -1.25, 0.0, 0.67, -1.25, 0.0, 0.67, -1.25, 0.0, 0.67, -1.25],
    [0.01, 0.0, 0.45, 0.0, 0.0, 0.04997917, 0.99875026, 0.1, 0.60, -1.20, -0.1, 0.74, -1.30, -0.1, 0.60, -1.20, 0.1, 0.74, -1.30]
  ]
}
```

- [ ] **Step 2: Write failing loader tests**

```python
from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.quadruped.loaders.motion_imitation import (
    load_motion_imitation,
)


FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"
JOINTS = tuple(f"joint_{i}" for i in range(12))


def test_load_motion_imitation_normalizes_to_wxyz():
    motion = load_motion_imitation(FIXTURE, JOINTS, "xyzw")
    assert motion.fps == pytest.approx(50.0)
    assert motion.loop_mode == "Wrap"
    assert motion.root_pos.shape == (2, 3)
    assert motion.root_rot.shape == (2, 4)
    np.testing.assert_allclose(motion.root_rot[0], [1.0, 0.0, 0.0, 0.0])
    assert motion.joint_pos.shape == (2, 12)
    assert motion.joint_names == JOINTS


def test_load_motion_imitation_rejects_wrong_joint_count(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text(
        '{"LoopMode":"Wrap","FrameDuration":0.02,"Frames":[[0,0,0,0,0,0,1,1]]}'
    )
    with pytest.raises(ValueError, match="expected 19 values"):
        load_motion_imitation(bad, JOINTS, "xyzw")
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_motion_imitation_loader.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement immutable validated motion types**

```python
# general_motion_retargeting/quadruped/types.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointSpaceMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    joint_pos: np.ndarray
    joint_names: tuple[str, ...]
    loop_mode: str

    def __post_init__(self) -> None:
        frames = self.root_pos.shape[0]
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.root_pos.shape != (frames, 3):
            raise ValueError("root_pos must have shape [T, 3]")
        if self.root_rot.shape != (frames, 4):
            raise ValueError("root_rot must have shape [T, 4]")
        if self.joint_pos.shape != (frames, len(self.joint_names)):
            raise ValueError("joint_pos shape does not match joint_names")
        for name, value in (
            ("root_pos", self.root_pos),
            ("root_rot", self.root_rot),
            ("joint_pos", self.joint_pos),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


@dataclass(frozen=True)
class CanonicalQuadrupedMotion:
    fps: float
    root_pos: np.ndarray
    root_rot: np.ndarray
    foot_pos_root: np.ndarray
    leg_order: tuple[str, str, str, str]
    loop_mode: str
```

- [ ] **Step 5: Implement the JSON loader**

Implement `load_motion_imitation` with `json.load`, exact frame-width
validation, quaternion conversion, and normalization:

```python
def load_motion_imitation(
    path: str | Path,
    joint_names: tuple[str, ...],
    quaternion_order: str,
) -> JointSpaceMotion:
    with Path(path).open() as stream:
        payload = json.load(stream)
    duration = float(payload["FrameDuration"])
    frames = np.asarray(payload["Frames"], dtype=float)
    expected = 3 + 4 + len(joint_names)
    if frames.ndim != 2 or frames.shape[1] != expected:
        raise ValueError(
            f"motion frames expected {expected} values, got shape {frames.shape}"
        )
    root_rot = frames[:, 3:7].copy()
    if quaternion_order == "xyzw":
        root_rot = root_rot[:, [3, 0, 1, 2]]
    elif quaternion_order != "wxyz":
        raise ValueError(f"unsupported quaternion order: {quaternion_order}")
    norms = np.linalg.norm(root_rot, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("root quaternion has zero norm")
    return JointSpaceMotion(
        fps=1.0 / duration,
        root_pos=frames[:, :3].copy(),
        root_rot=root_rot / norms,
        joint_pos=frames[:, 7:].copy(),
        joint_names=joint_names,
        loop_mode=str(payload["LoopMode"]),
    )
```

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_motion_imitation_loader.py -v
```

Expected: `2 passed`.

Commit:

```bash
git add general_motion_retargeting/quadruped tests/quadruped
git commit -m "feat: parse motion imitation quadruped motions"
```

---

### Task 3: Load and Validate Quadruped Robot Specifications

**Files:**
- Modify: `pyproject.toml`
- Modify: `setup.py`
- Modify: `uv.lock`
- Create: `general_motion_retargeting/quadruped/robot_spec.py`
- Create: `tests/quadruped/test_robot_spec.py`

**Interfaces:**
- Consumes: `load_robot_spec(path, repo_root)`.
- Produces: `QuadrupedRobotSpec`, `LegSpec`, `JointMapSpec`, and validated `mujoco.MjModel`.

- [ ] **Step 1: Add PyYAML**

Add `"pyyaml>=6.0.2"` to `pyproject.toml` and `"pyyaml"` to `setup.py`, then run:

```bash
uv lock
```

- [ ] **Step 2: Write failing validation tests**

```python
from pathlib import Path

import pytest

from general_motion_retargeting.quadruped.robot_spec import load_robot_spec


ROOT = Path(__file__).resolve().parents[2]


def test_load_laikago_spec_resolves_model_and_leg_order():
    spec = load_robot_spec(
        ROOT / "general_motion_retargeting/quadruped/configs/laikago.yaml",
        ROOT,
    )
    assert spec.robot == "laikago"
    assert spec.leg_order == ("FL", "FR", "RL", "RR")
    assert len(spec.motion_joint_order) == 12
    assert spec.model.nq == 19


def test_spec_rejects_missing_leg(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "robot: bad\nmodel_type: quadruped\nmjcf_path: missing.xml\n"
        "root_body: trunk\nlegs: {}\nmotion: {joint_order: [], quaternion_order: xyzw}\n"
        "reference_pose: {}\n"
    )
    with pytest.raises(ValueError, match="missing legs: FL, FR, RL, RR"):
        load_robot_spec(path, tmp_path)
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_robot_spec.py -v
```

Expected: FAIL because `robot_spec` does not exist.

- [ ] **Step 4: Implement spec dataclasses and validation**

Define:

```python
@dataclass(frozen=True)
class LegSpec:
    name: str
    joints: tuple[str, str, str]
    foot_site: str


@dataclass(frozen=True)
class JointMapSpec:
    sign: float
    scale: float


@dataclass(frozen=True)
class QuadrupedRobotSpec:
    robot: str
    mjcf_path: Path
    model: mujoco.MjModel
    root_body: str
    legs: Mapping[str, LegSpec]
    motion_joint_order: tuple[str, ...]
    quaternion_order: str
    reference_pose: Mapping[str, float]
    joint_mapping: Mapping[str, JointMapSpec]

    @property
    def leg_order(self) -> tuple[str, str, str, str]:
        return ("FL", "FR", "RL", "RR")
```

`load_robot_spec` must validate in this order:

1. `model_type == "quadruped"`.
2. Exactly FL/FR/RL/RR are present.
3. Every leg has exactly three distinct joints.
4. MJCF compiles.
5. Root body, all joints, and all foot sites exist.
6. Every configured leg joint is a hinge.
7. Every motion joint is unique and exists.
8. Every reference-pose joint exists and is within its MJCF range.

Use `mujoco.mj_name2id` and `model.jnt_qposadr`; do not infer qpos offsets.

- [ ] **Step 5: Run focused tests and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_robot_spec.py tests/quadruped/test_assets.py -v
```

Expected: all tests pass.

Commit:

```bash
git add pyproject.toml setup.py uv.lock general_motion_retargeting/quadruped/robot_spec.py tests/quadruped/test_robot_spec.py
git commit -m "feat: validate quadruped robot specifications"
```

---

### Task 4: Compute Source Foot Trajectories and Morphology Scaling

**Files:**
- Create: `general_motion_retargeting/quadruped/kinematics.py`
- Create: `general_motion_retargeting/quadruped/morphology.py`
- Create: `tests/quadruped/test_kinematics.py`
- Create: `tests/quadruped/test_morphology.py`

**Interfaces:**
- Consumes: `source_forward_kinematics(spec, motion)`.
- Produces: `CanonicalQuadrupedMotion`.
- Consumes: `scale_foot_trajectories(source_motion, source_spec, target_spec)`.
- Produces: target-morphology `CanonicalQuadrupedMotion`.

- [ ] **Step 1: Write failing FK tests**

```python
def test_source_fk_returns_root_relative_feet(laikago_spec, two_frame_motion):
    canonical = source_forward_kinematics(laikago_spec, two_frame_motion)
    assert canonical.foot_pos_root.shape == (2, 4, 3)
    assert canonical.leg_order == ("FL", "FR", "RL", "RR")
    assert np.isfinite(canonical.foot_pos_root).all()
    np.testing.assert_allclose(
        canonical.foot_pos_root[0, 0, 1],
        -canonical.foot_pos_root[0, 1, 1],
        atol=1e-5,
    )
```

- [ ] **Step 2: Run FK test to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_kinematics.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement named-qpos assignment and FK**

Implement helpers:

```python
def set_named_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...],
    values: np.ndarray,
) -> None:
    for name, value in zip(names, values, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
```

For each frame, assign the free root at its `jnt_qposadr`, call
`mujoco.mj_forward`, read `data.site_xpos`, and transform world foot positions
into the configured root body's frame:

```python
foot_root = root_rotation_world.T @ (foot_world - root_position_world)
```

- [ ] **Step 4: Write failing morphology tests**

```python
def test_morphology_preserves_neutral_stance_and_scales_delta(
    laikago_spec, go2_spec, canonical_motion
):
    scaled = scale_foot_trajectories(
        canonical_motion, laikago_spec, go2_spec
    )
    assert scaled.foot_pos_root.shape == canonical_motion.foot_pos_root.shape
    source_delta = (
        canonical_motion.foot_pos_root[1] - canonical_motion.foot_pos_root[0]
    )
    target_delta = scaled.foot_pos_root[1] - scaled.foot_pos_root[0]
    assert np.sign(target_delta[..., 0]).tolist() == np.sign(
        source_delta[..., 0]
    ).tolist()
```

- [ ] **Step 5: Implement morphology descriptors and scaling**

Compute neutral feet by running each model at its configured reference pose.
Derive:

```python
@dataclass(frozen=True)
class Morphology:
    neutral_feet: np.ndarray
    hip_length: float
    hip_width: float
    leg_reach: float
```

Scale neutral-relative deltas:

```python
scale = np.array(
    [
        target.hip_length / source.hip_length,
        target.hip_width / source.hip_width,
        target.leg_reach / source.leg_reach,
    ]
)
target_feet = target.neutral_feet + (
    source_feet - source.neutral_feet
) * scale
```

Reject morphology denominators below `1e-6`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_kinematics.py tests/quadruped/test_morphology.py -v
```

Expected: all tests pass.

Commit:

```bash
git add general_motion_retargeting/quadruped/kinematics.py general_motion_retargeting/quadruped/morphology.py tests/quadruped
git commit -m "feat: derive and scale quadruped foot trajectories"
```

---

### Task 5: Implement Go2 Initial Mapping and Mink IK

**Files:**
- Create: `general_motion_retargeting/quadruped/joint_mapping.py`
- Create: `general_motion_retargeting/quadruped/retarget.py`
- Create: `tests/quadruped/test_joint_mapping.py`
- Create: `tests/quadruped/test_retarget.py`

**Interfaces:**
- Consumes: `map_initial_configuration(source_q, source_spec, target_spec)`.
- Produces: a complete target `qpos`.
- Consumes: `QuadrupedRobotRetargeter.retarget_motion(JointSpaceMotion)`.
- Produces: `QuadrupedRetargetResult`.

- [ ] **Step 1: Write failing initial-map tests**

```python
def test_initial_map_operates_around_reference_pose(
    laikago_spec, go2_spec
):
    source = np.array(
        [laikago_spec.reference_pose[name] for name in laikago_spec.motion_joint_order]
    )
    qpos = map_initial_configuration(source, laikago_spec, go2_spec)
    for leg in go2_spec.leg_order:
        for name in go2_spec.legs[leg].joints:
            joint_id = mujoco.mj_name2id(
                go2_spec.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            actual = qpos[go2_spec.model.jnt_qposadr[joint_id]]
            assert actual == pytest.approx(go2_spec.reference_pose[name])
```

- [ ] **Step 2: Implement semantic initial mapping**

Map semantic indices `hip=0`, `thigh=1`, and `calf=2` independently for each
leg. Initialize the complete Go2 qpos from `mujoco.MjData(model).qpos`, write
the target reference pose, then apply:

```python
target_ref + mapping.sign * mapping.scale * (source_value - source_ref)
```

Clamp only hinge values to their MJCF ranges.

- [ ] **Step 3: Write failing static IK tests**

```python
def test_go2_retargeter_solves_reference_stance(
    laikago_spec, go2_spec, reference_motion
):
    retargeter = QuadrupedRobotRetargeter(
        source_spec=laikago_spec,
        target_spec=go2_spec,
        solver="daqp",
        damping=0.5,
        max_iterations=10,
        use_velocity_limit=False,
    )
    result = retargeter.retarget_motion(reference_motion)
    assert result.qpos.shape == (1, go2_spec.model.nq)
    assert np.isfinite(result.qpos).all()
    assert result.diagnostics[0].final_error < 0.02
```

- [ ] **Step 4: Run tests to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_joint_mapping.py tests/quadruped/test_retarget.py -v
```

Expected: FAIL because mapping and retarget classes do not exist.

- [ ] **Step 5: Implement diagnostics and result types**

Add to `types.py`:

```python
@dataclass(frozen=True)
class FrameDiagnostics:
    frame_index: int
    iterations: int
    initial_error: float
    final_error: float
    reached_max_iterations: bool
    joint_limit_hits: tuple[str, ...]


@dataclass(frozen=True)
class QuadrupedRetargetResult:
    qpos: np.ndarray
    fps: float
    loop_mode: str
    diagnostics: tuple[FrameDiagnostics, ...]
```

- [ ] **Step 6: Implement Mink task setup**

`QuadrupedRobotRetargeter.__init__` must:

1. Create `mink.Configuration(target_spec.model)`.
2. Create one root-body `mink.FrameTask` with configurable position and
   orientation costs.
3. Create four foot-site `mink.FrameTask` instances with position cost and zero
   orientation cost.
4. Add `mink.ConfigurationLimit`.
5. Optionally add `mink.VelocityLimit` for the target's 12 configured joints.

Use the target foot sites and `frame_type="site"`. Use `frame_type="body"` for
the root.

- [ ] **Step 7: Implement frame solving and best-state retention**

For each frame:

1. Convert the scaled trunk-relative feet to world positions with the desired
   target root pose.
2. Set the root and foot task targets with `mink.SE3`.
3. Compute concatenated task error.
4. Solve, integrate, and retain the lowest-error qpos.
5. Stop at `max_iterations` or when improvement is at most `1e-3`.
6. Restore the best qpos before returning the frame.
7. Reject non-finite solver output immediately.

Use the mapped initial state for frame zero and the previous best qpos for
later frames.

- [ ] **Step 8: Implement Phase 1 ground offset**

After solving the complete motion, run Go2 FK for all frames, find the minimum
configured foot-site `z`, and subtract it from every root `qpos[:, 2]`.
Do not infer support or alter individual feet.

- [ ] **Step 9: Run focused tests and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_joint_mapping.py tests/quadruped/test_retarget.py -v
```

Expected: all tests pass.

Commit:

```bash
git add general_motion_retargeting/quadruped tests/quadruped
git commit -m "feat: retarget quadruped foot trajectories with mink"
```

---

### Task 6: Register Go2 and Add the Retargeter Factory

**Files:**
- Modify: `general_motion_retargeting/params.py`
- Modify: `general_motion_retargeting/__init__.py`
- Modify: `general_motion_retargeting/robot_motion_viewer.py`
- Create: `general_motion_retargeting/factory.py`
- Create: `tests/quadruped/test_factory.py`

**Interfaces:**
- Consumes: `create_retargeter(model_type, **kwargs)`.
- Produces: existing `GeneralMotionRetargeting` or new `QuadrupedRobotRetargeter`.

- [ ] **Step 1: Write failing factory tests**

```python
def test_factory_defaults_to_humanoid(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "general_motion_retargeting.factory.GeneralMotionRetargeting",
        lambda **kwargs: sentinel,
    )
    assert create_retargeter(src_human="smplx", tgt_robot="unitree_g1") is sentinel


def test_factory_selects_quadruped(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "general_motion_retargeting.factory.QuadrupedRobotRetargeter",
        lambda **kwargs: sentinel,
    )
    assert create_retargeter(model_type="quadruped") is sentinel


def test_factory_rejects_unknown_model_type():
    with pytest.raises(ValueError, match="model_type must be"):
        create_retargeter(model_type="hexapod")
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_factory.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the factory without changing the humanoid constructor**

```python
def create_retargeter(model_type: str = "humanoid", **kwargs):
    if model_type == "humanoid":
        return GeneralMotionRetargeting(**kwargs)
    if model_type == "quadruped":
        return QuadrupedRobotRetargeter(**kwargs)
    raise ValueError(
        "model_type must be 'humanoid' or 'quadruped', "
        f"got {model_type!r}"
    )
```

Export `create_retargeter` and `QuadrupedRobotRetargeter` from the package
without replacing the existing `GeneralMotionRetargeting` export.

- [ ] **Step 4: Register Go2 viewer metadata**

Add:

```python
ROBOT_XML_DICT["unitree_go2"] = (
    ASSET_ROOT / "quadrupeds" / "unitree_go2" / "go2.xml"
)
ROBOT_BASE_DICT["unitree_go2"] = "body"
VIEWER_CAM_DISTANCE_DICT["unitree_go2"] = 1.5
```

Add a test that loads the registered path and resolves
`ROBOT_BASE_DICT["unitree_go2"]` to body `body`.

- [ ] **Step 5: Run factory and humanoid regression tests**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_factory.py tests/test_dr02_gmr_registration.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add general_motion_retargeting tests/quadruped/test_factory.py
git commit -m "feat: register go2 and quadruped retargeter factory"
```

---

### Task 7: Add the motion_imitation-to-Go2 CLI and Serialization

**Files:**
- Create: `scripts/motion_imitation_to_robot.py`
- Create: `tests/quadruped/test_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: command-line source motion, source robot name, and target robot.
- Produces: GMR-compatible pickle and optional viewer playback.

- [ ] **Step 1: Write failing CLI parser and smoke tests**

```python
from scripts.motion_imitation_to_robot import build_parser, run


def test_parser_requires_motion_and_defaults_to_go2():
    args = build_parser().parse_args(["--motion_file", "dog_pace.txt"])
    assert args.model_type == "quadruped"
    assert args.source_robot == "laikago"
    assert args.robot == "unitree_go2"


def test_headless_fixture_writes_gmr_motion(tmp_path):
    output = tmp_path / "motion.pkl"
    run(
        motion_file=FIXTURE,
        source_robot="laikago",
        robot="unitree_go2",
        save_path=output,
        headless=True,
        rate_limit=False,
    )
    with output.open("rb") as stream:
        motion = pickle.load(stream)
    assert set(("root_pos", "root_rot", "dof_pos", "fps")) <= motion.keys()
    assert motion["root_pos"].shape == (2, 3)
    assert motion["root_rot"].shape == (2, 4)
    assert motion["dof_pos"].shape == (2, 12)
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_cli.py -v
```

Expected: FAIL because the script does not exist.

- [ ] **Step 3: Implement parser and orchestration**

The parser exposes:

```text
--model_type quadruped
--motion_file PATH
--source_robot laikago
--robot unitree_go2
--save_path PATH
--headless
--rate_limit
--record_video
--video_path PATH
--use_velocity_limit
```

`run` loads source and target specs, parses the motion, calls
`QuadrupedRobotRetargeter.retarget_motion`, splits qpos into root and 12 target
joints using model metadata, converts root quaternion from internal `wxyz` to
saved `xyzw`, and writes with `pickle.dump`.

- [ ] **Step 4: Add viewer playback**

When not headless, construct `RobotMotionViewer(robot_type="unitree_go2",
motion_fps=result.fps, ...)` and call:

```python
viewer.step(
    root_pos=qpos[:3],
    root_rot=qpos[3:7],
    dof_pos=target_joint_values,
    rate_limit=args.rate_limit,
)
```

Keep visualization outside `QuadrupedRobotRetargeter`.

- [ ] **Step 5: Document usage and Phase 2 boundary**

Add a README section with:

```bash
python scripts/motion_imitation_to_robot.py \
  --motion_file assets/quadrupeds/motions/dog_pace.txt \
  --source_robot laikago \
  --robot unitree_go2 \
  --save_path retargeting_data/go2/dog_pace.pkl \
  --rate_limit
```

State that Phase 1 requires MJCF for both robots and does not infer contacts or
correct support-foot slip.

- [ ] **Step 6: Run CLI tests and commit**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped/test_cli.py -v
```

Expected: all tests pass.

Commit:

```bash
git add scripts/motion_imitation_to_robot.py tests/quadruped/test_cli.py README.md
git commit -m "feat: add motion imitation to go2 command"
```

---

### Task 8: Add End-to-End Quality Checks and Complete Regression Verification

**Files:**
- Create: `tests/quadruped/test_end_to_end.py`
- Create: `scripts/check_quadruped_motion.py`
- Create: `docs/quadruped_retargeting.md`

**Interfaces:**
- Consumes: a saved Go2 GMR pickle and Go2 MJCF.
- Produces: deterministic quality metrics and nonzero exit status on acceptance failure.

- [ ] **Step 1: Write failing end-to-end assertions**

```python
def test_dog_pace_retargets_to_valid_go2_motion(pipeline):
    result = pipeline("assets/quadrupeds/motions/dog_pace.txt")
    assert result.qpos.shape[0] > 1
    assert result.qpos.shape[1] == 19
    assert np.isfinite(result.qpos).all()
    assert result.non_converged_ratio <= 0.05
    assert result.max_joint_limit_violation <= 1e-8
    assert result.max_velocity_violation <= 1e-8
    assert result.min_foot_height == pytest.approx(0.0, abs=1e-5)
```

Mark the test `pytest.mark.integration`; the vendored reference motion keeps
the test deterministic and offline.

- [ ] **Step 2: Implement reusable quality metrics**

`scripts/check_quadruped_motion.py` must:

1. Load the Go2 MJCF and saved pickle.
2. Validate array shape and finite values.
3. Resolve the 12 target joints by config name.
4. Report lower/upper limit margins.
5. Compute finite-difference joint velocity at saved FPS.
6. Run Go2 FK and report min/max foot heights.
7. Report retarget diagnostics when embedded by the CLI.
8. Exit nonzero for NaN, limit violation, configured velocity violation, or
   excessive non-convergence.

- [ ] **Step 3: Run quadruped tests**

Run:

```bash
env PYTHONPATH=. uv run pytest tests/quadruped -v
```

Expected: all quadruped tests pass.

- [ ] **Step 4: Run the full repository suite**

Run:

```bash
env PYTHONPATH=. uv run pytest -q
```

Expected: all existing humanoid and new quadruped tests pass.

- [ ] **Step 5: Run a real headless conversion and checker**

Run:

```bash
env PYTHONPATH=. uv run python scripts/motion_imitation_to_robot.py \
  --motion_file assets/quadrupeds/motions/dog_pace.txt \
  --source_robot laikago \
  --robot unitree_go2 \
  --save_path /tmp/go2_dog_pace.pkl \
  --headless \
  --use_velocity_limit

env PYTHONPATH=. uv run python scripts/check_quadruped_motion.py \
  --motion /tmp/go2_dog_pace.pkl \
  --robot unitree_go2
```

Expected: both commands exit zero; checker reports finite motion, zero joint
limit violations, zero configured velocity violations, and at most 5 percent
non-converged frames.

- [ ] **Step 6: Document architecture, configuration, and diagnostics**

`docs/quadruped_retargeting.md` must cover:

- MJCF and YAML source robot onboarding.
- Required FL/FR/RL/RR semantics.
- `motion_imitation` frame interpretation.
- Morphology scaling.
- Mink task and solver parameters.
- Output format.
- Quality checker usage.
- Phase 1 limitations.
- Phase 2 contact-aware TODO list copied from the approved design.

- [ ] **Step 7: Final diff and regression review**

Run:

```bash
git diff --check master...HEAD
git status --short
env PYTHONPATH=. uv run pytest -q
```

Expected: no whitespace errors, no unexpected untracked files, and all tests
pass.

- [ ] **Step 8: Commit**

```bash
git add tests/quadruped/test_end_to_end.py scripts/check_quadruped_motion.py docs/quadruped_retargeting.md
git commit -m "test: verify quadruped retargeting end to end"
```

---

## Final Acceptance Checklist

- [ ] `dev/quadruped` contains only intentional quadruped changes and the approved documentation.
- [ ] Laikago and Go2 MJCF files compile with the locked MuJoCo version.
- [ ] Source and target semantic configurations validate without inferred names.
- [ ] `motion_imitation` input is normalized to internal `wxyz`.
- [ ] Source FK produces FL/FR/RL/RR trunk-relative foot trajectories.
- [ ] Morphology scaling preserves neutral stance and gait deltas.
- [ ] Go2 Mink IK uses configuration and optional velocity limits.
- [ ] Ground offset matches GMR main-flow capability and does not infer contact.
- [ ] Saved output loads in the existing Go2 MuJoCo viewer.
- [ ] Full `dog_pace` conversion passes the checker.
- [ ] Existing humanoid tests pass unchanged.
- [ ] Contact-aware retargeting remains documented as Phase 2 work and is not partially implemented.
