# Skeleton Motion Retargeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add an independent full-body chain retargeting algorithm that transfers LAFAN1 directions onto configured DR02 segment lengths and returns the same qpos format as GMR.

**Architecture:** A pure SkeletonTargetReconstructor validates and rebuilds ordered chains. SkeletonMotionRetargeting subclasses the existing retargeter only to reuse MuJoCo/Mink setup and the two-stage solve; it overrides target preparation without changing GeneralMotionRetargeting. A separate GMR-shaped JSON configuration retains the current IK tables and adds explicit chain segments.

**Tech Stack:** Python 3.11, NumPy, SciPy Rotation, MuJoCo, Mink, pytest, JSON.

**Spec:** docs/superpowers/specs/2026-08-31-skeleton-motion-retargeting-design.md

## Global Constraints

- Do not change GeneralMotionRetargeting behavior.
- Preserve retarget(human_frame, offset_to_ground=False) -> np.ndarray.
- Configured target_length values in metres are authoritative.
- rotation_offset corrects base-pose/local-axis differences only.
- position_offset corrects frame-origin differences only.
- Skeleton-chain scale entries are 1.0 and ignored by reconstruction.
- Cover spine, both arms, and both legs through the toes.
- Never commit generated motion, datasets, videos, or machine paths.

---

### Task 1: Pure chain parsing and reconstruction

**Files:**
- Create: general_motion_retargeting/skeleton_retarget.py
- Create: tests/test_skeleton_retarget.py

**Interfaces:**
- SkeletonSegment(human_parent, human_child, robot_parent_frame, robot_child_frame, target_length)
- SkeletonChain(name, segments)
- parse_skeleton_chains(config) -> tuple[SkeletonChain, ...]
- SkeletonTargetReconstructor(chains, epsilon=1e-8)
- SkeletonTargetReconstructor.reconstruct(human_data) -> copied human-data dictionary

- [ ] **Step 1: Write failing validation tests**

    def test_rejects_non_positive_length():
        config = chain_config(target_length=0.0)
        with pytest.raises(ValueError, match="target_length"):
            parse_skeleton_chains(config)

    def test_rejects_disconnected_chain():
        config = chain_config(segments=[
            segment("Shoulder", "Elbow", 0.3),
            segment("Other", "Wrist", 0.2),
        ])
        with pytest.raises(ValueError, match="disconnected"):
            parse_skeleton_chains(config)

Also test empty chains, duplicate names, duplicate child ownership, and non-finite lengths.

- [ ] **Step 2: Run RED**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_retarget.py -q

Expected: import failure because skeleton_retarget.py does not exist.

- [ ] **Step 3: Implement immutable parsing**

Use frozen dataclasses SkeletonSegment and SkeletonChain. Validate ordered connectivity: each segment child equals the next segment parent. Reject malformed strings and target_length <= 0.

- [ ] **Step 4: Write failing fixed-length tests**

    def test_lengths_are_constant_at_every_angle():
        reconstructor = SkeletonTargetReconstructor(chains)
        for elbow in ([1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]):
            result = reconstructor.reconstruct(frame(elbow))
            assert distance(result, "Shoulder", "Elbow") == pytest.approx(0.3)
            assert distance(result, "Elbow", "Wrist") == pytest.approx(0.2)

    def test_branches_use_reconstructed_shared_root():
        result = full_body_reconstructor.reconstruct(full_body_frame())
        assert distance(result, "Spine", "LeftShoulder") == pytest.approx(0.32)
        assert distance(result, "Spine", "RightShoulder") == pytest.approx(0.32)

- [ ] **Step 5: Implement reconstruction**

For each ordered segment:
1. Read direction from original source child minus original source parent.
2. Normalize it or resolve a cached previous direction.
3. Set reconstructed child to reconstructed parent plus target_length times direction.
4. Preserve the source quaternion.
5. Never mutate the caller frame.
6. Reject missing and non-finite data with chain/joint names.

- [ ] **Step 6: Test degenerate handling**

    def test_degenerate_reuses_previous_direction():
        r = SkeletonTargetReconstructor(chains)
        r.reconstruct(valid_frame())
        result = r.reconstruct(zero_length_frame())
        np.testing.assert_allclose(result["Elbow"][0], [0.3, 0, 0])

    def test_first_degenerate_frame_fails():
        with pytest.raises(ValueError, match="left_arm.*Shoulder.*Elbow"):
            SkeletonTargetReconstructor(chains).reconstruct(zero_length_frame())

- [ ] **Step 7: Run and commit**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_retarget.py -q
    git add general_motion_retargeting/skeleton_retarget.py tests/test_skeleton_retarget.py
    git commit -m "feat: add fixed-length skeleton target reconstruction"

---

### Task 2: Independent Mink retargeter

**Files:**
- Create: general_motion_retargeting/skeleton_motion_retarget.py
- Modify: general_motion_retargeting/__init__.py
- Create: tests/test_skeleton_motion_retarget.py

**Interfaces:**
- SkeletonMotionRetargeting(GeneralMotionRetargeting)
- Constructor mirrors GMR and accepts skeleton_config_path.
- retarget() remains inherited and returns qpos.

- [ ] **Step 1: Write failing API isolation tests**

    def test_compatible_api(fixture_config, fixture_xml):
        r = SkeletonMotionRetargeting(
            src_human="fixture", tgt_robot="fixture",
            skeleton_config_path=fixture_config,
            robot_xml_path=fixture_xml, verbose=False,
        )
        qpos = r.retarget(fixture_human_frame())
        assert qpos.shape == (r.model.nq,)
        assert np.isfinite(qpos).all()

    def test_original_update_path_is_untouched():
        assert GeneralMotionRetargeting.update_targets is not SkeletonMotionRetargeting.update_targets

- [ ] **Step 2: Run RED**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_motion_retarget.py -q

- [ ] **Step 3: Implement the subclass**

Constructor resolution:
1. Resolve the skeleton config.
2. Pass it as ik_config_path to super().__init__ so existing task setup is reused.
3. Load the same JSON, require algorithm == "skeleton", parse chains, and create the reconstructor.
4. Validate skeleton-chain human_scale_table entries are exactly 1.0.

Override update_targets:
1. Copy/convert input.
2. Reconstruct chain positions without scale_human_data.
3. Apply table-1 offsets and table-2-only offsets using existing semantics.
4. Apply ground handling.
5. Set existing Mink task targets.
6. Store reconstructed/calibrated data in scaled_human_data for compatibility.

Do not edit motion_retarget.py to share helpers.

- [ ] **Step 4: Test semantic isolation**

    def test_chain_reconstruction_ignores_root_relative_scale():
        # Invalid non-1 scale must fail construction rather than alter lengths.

    def test_rotation_offset_preserves_segment_length():
        assert reconstructed_length(identity_pose) == pytest.approx(
            reconstructed_length(rotated_base_pose)
        )

    def test_position_offset_does_not_change_raw_reconstructed_length():
        # Inspect reconstructor output before task-frame calibration.

- [ ] **Step 5: Add robot-frame diagnostics**

Resolve every robot parent/child frame at construction. Record configured_length and MJCF reference_distance in segment_diagnostics. Missing frames fail. Warn when an active position offset norm exceeds 0.05 m; never override target_length.

- [ ] **Step 6: Run and commit**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_retarget.py tests/test_skeleton_motion_retarget.py -q
    git add general_motion_retargeting/skeleton_motion_retarget.py general_motion_retargeting/__init__.py tests/test_skeleton_motion_retarget.py
    git commit -m "feat: add skeleton motion retargeter"

---

### Task 3: Full-body DR02/LAFAN1 configuration

**Files:**
- Create: general_motion_retargeting/skeleton_configs/bvh_lafan1_to_dr02.json
- Modify: general_motion_retargeting/params.py
- Create: tests/test_dr02_skeleton_retargeting.py

**Interfaces:**
- SKELETON_CONFIG_ROOT
- SKELETON_CONFIG_DICT["bvh_lafan1"]["dr02"]

- [ ] **Step 1: Write failing registration and exact-length tests**

Assert the registered file exists and its segment map exactly equals:

    Hips -> Spine2: 0.29
    Spine2 -> LeftArm: 0.3246858828159918
    LeftArm -> LeftForeArm: 0.265
    LeftForeArm -> LeftHand: 0.2395
    Spine2 -> RightArm: 0.3246858828159918
    RightArm -> RightForeArm: 0.265
    RightForeArm -> RightHand: 0.2395
    Hips -> LeftUpLeg: 0.1614
    LeftUpLeg -> LeftLeg: 0.4194
    LeftLeg -> LeftFoot: 0.41
    LeftFoot -> LeftToe: 0.1110180165558726
    Hips -> RightUpLeg: 0.1614
    RightUpLeg -> RightLeg: 0.4194
    RightLeg -> RightFoot: 0.41
    RightFoot -> RightToe: 0.1110180165558726

- [ ] **Step 2: Run RED**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_dr02_skeleton_retargeting.py -q

- [ ] **Step 3: Register config root**

Add SKELETON_CONFIG_ROOT = HERE / "skeleton_configs" and a dictionary containing only bvh_lafan1/dr02 initially.

- [ ] **Step 4: Create configuration**

Keep the familiar GMR roots, initialization fields, human_scale_table, ik_match_table1, and ik_match_table2 shapes. Add algorithm="skeleton" and five chains:

    spine: Hips -> Spine2
    left_arm: Spine2 -> LeftArm -> LeftForeArm -> LeftHand
    right_arm: Spine2 -> RightArm -> RightForeArm -> RightHand
    left_leg: Hips -> LeftUpLeg -> LeftLeg -> LeftFoot -> LeftToe
    right_leg: Hips -> RightUpLeg -> RightLeg -> RightFoot -> RightToe

Set chain-node scales to 1.0. Use the exact lengths above. Start from current task weights and Retarget Base Pose rotations. Reset offsets used only to fake bone length; retain only defensible frame-origin corrections.

- [ ] **Step 5: Add all-frame geometry test**

Load local dance1_subject1.bvh when available and skip otherwise. For every 30th frame, reconstruct and assert every configured segment length within 1e-6 m.

- [ ] **Step 6: Run and commit**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_retarget.py tests/test_skeleton_motion_retarget.py tests/test_dr02_skeleton_retargeting.py -q
    git add general_motion_retargeting/skeleton_configs/bvh_lafan1_to_dr02.json general_motion_retargeting/params.py tests/test_dr02_skeleton_retargeting.py
    git commit -m "feat: configure full-body DR02 skeleton retargeting"

---

### Task 4: Algorithm selection in BVH tools

**Files:**
- Modify: scripts/bvh_to_robot.py
- Modify: scripts/vis_gmr_debug.py
- Modify: tests/test_vis_gmr_debug.py
- Create: tests/test_bvh_to_robot_algorithm.py

**Interfaces:**
- --algorithm {gmr,skeleton}, default gmr.
- Skeleton selection constructs SkeletonMotionRetargeting.
- Existing commands without the flag remain unchanged.

- [ ] **Step 1: Write failing parser tests**

    def test_parser_accepts_skeleton():
        args = build_parser().parse_args(required + ["--algorithm", "skeleton"])
        assert args.algorithm == "skeleton"

    def test_default_is_gmr():
        assert build_parser().parse_args(required).algorithm == "gmr"

For bvh_to_robot, monkeypatch both constructors and assert only the selected one is called.

- [ ] **Step 2: Run RED**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_vis_gmr_debug.py tests/test_bvh_to_robot_algorithm.py -q

- [ ] **Step 3: Implement selection**

Add a small class selector mapping gmr to GeneralMotionRetargeting and skeleton to SkeletonMotionRetargeting. In skeleton mode resolve SKELETON_CONFIG_DICT unless an explicit config path is supplied.

- [ ] **Step 4: Identify algorithm in debug output**

Startup output must contain algorithm=gmr or algorithm=skeleton.

- [ ] **Step 5: Run and commit**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_vis_gmr_debug.py tests/test_bvh_to_robot_algorithm.py -q
    git add scripts/bvh_to_robot.py scripts/vis_gmr_debug.py tests/test_vis_gmr_debug.py tests/test_bvh_to_robot_algorithm.py
    git commit -m "feat: expose skeleton retargeting in BVH tools"

---

### Task 5: Integration, documentation, and visual verification

**Files:**
- Modify: README.md
- Modify: tests/test_dr02_skeleton_retargeting.py

- [ ] **Step 1: Add finite-qpos integration test**

When the local dance fixture exists, construct SkeletonMotionRetargeting for bvh_lafan1/dr02, retarget every 30th frame, and assert qpos shape is model.nq and all values are finite.

- [ ] **Step 2: Run focused and full suites**

    PYTHONPATH=. ../.venv/bin/pytest tests/test_skeleton_retarget.py tests/test_skeleton_motion_retarget.py tests/test_dr02_skeleton_retargeting.py -q
    PYTHONPATH=. ../.venv/bin/pytest tests -q

Record pass/skip counts. Original GMR tests may not regress.

- [ ] **Step 3: Export temporary comparison motion**

    PYTHONPATH=. ../.venv/bin/python scripts/bvh_to_robot.py \
      --bvh_file /home/jvwei/GMR-private/data/LAFAN1/dance/dance1_subject1.bvh \
      --robot dr02 \
      --algorithm skeleton \
      --save_path /tmp/dr02_lafan1_skeleton_retarget.pkl

Write only to /tmp/dr02_lafan1_skeleton_retarget.pkl.

- [ ] **Step 4: Open debug playback**

    PYTHONPATH=. ../.venv/bin/python scripts/vis_gmr_debug.py \
      --algorithm skeleton \
      --motion /tmp/dr02_lafan1_skeleton_retarget.pkl \
      --reference /home/jvwei/GMR-private/data/LAFAN1/dance/dance1_subject1.bvh \
      --mjcf assets/dr02/mjcf/dr02_pos.xml \
      --ik-config general_motion_retargeting/skeleton_configs/bvh_lafan1_to_dr02.json \
      --actual-human-height 1.75 \
      --loop

Inspect constant arm/leg lengths, elbow/knee direction, palms, pelvis motion, foot contact, and solve jumps.

- [ ] **Step 5: Document usage**

README examples must show Python construction and CLI selection. State that target_length owns proportions; rotation_offset owns Retarget Base Pose/axis correction; position_offset owns frame-origin calibration.

- [ ] **Step 6: Commit**

    git add README.md tests/test_dr02_skeleton_retargeting.py
    git commit -m "docs: document skeleton motion retargeting"

- [ ] **Step 7: Final verification**

    git status --short --branch
    git log --oneline -6

Confirm /tmp artifacts are untracked and no unrelated dirty files were staged.
