import json

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from general_motion_retargeting.gmr_debug_visualizer import (
    GMRDebugVisualizer,
    build_position_correspondences,
    compose_scene_xml,
    compute_correspondences,
    compute_height_offset,
    load_effective_ik_config,
    processed_reference_frame,
    reference_edges,
    transform_full_reference_frame,
    transform_reference_frame,
)


def minimal_model():
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="pelvis" pos="0 0 1">
              <freejoint/>
              <geom name="robot_visual" type="sphere" size=".1"/>
              <body name="hand" pos="0 0 .5">
                <site name="wrist" pos=".1 0 0"/>
              </body>
              <body name="left_foot" pos="0 .1 -.8">
                <site name="left_sole" pos="0 0 -.1"/>
              </body>
              <body name="right_foot" pos="0 -.1 -.8">
                <site name="right_sole" pos="0 0 -.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )


def config_payload():
    return {
        "human_root_name": "pelvis",
        "human_height_assumption": 2.0,
        "ground_height": 0.1,
        "human_scale_table": {"pelvis": 1.0, "hand": 0.5},
        "global_position_offsets": {"hand": [0.0, 0.2, 0.0]},
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "ik_match_table1": {
            "pelvis": ["pelvis", 10, 0, [0, 0, 0], [1, 0, 0, 0]],
            "wrist": ["hand", 5, 0, [1, 0, 0], [1, 0, 0, 0]],
        },
        "ik_match_table2": {
            "wrist": ["hand", 7, 0, [9, 9, 9], [1, 0, 0, 0]],
            "hand": ["unused", 0, 10, [0, 0, 0], [1, 0, 0, 0]],
        },
    }


def test_processed_reference_frame_preserves_final_gmr_target_positions():
    frame = {
        "Hips": (
            np.array([1.25, -0.5, 0.8]),
            np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        )
    }

    points = processed_reference_frame(frame)

    np.testing.assert_allclose(
        points["Hips"].world_position, [1.25, -0.5, 0.8]
    )
    np.testing.assert_allclose(
        points["Hips"].world_rotation,
        Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        atol=1e-12,
    )


def test_reference_edges_infers_supported_skeletons_and_filters_missing_names():
    assert ("pelvis", "left_knee") in reference_edges(
        ["pelvis", "left_knee", "left_foot"]
    )
    lafan_edges = reference_edges(
        ["Hips", "LeftUpLeg", "LeftLeg", "LeftFootMod"]
    )
    assert ("Hips", "LeftUpLeg") in lafan_edges
    assert ("LeftLeg", "LeftFootMod") in lafan_edges
    assert reference_edges(["custom_root", "custom_hand"]) == ()


def test_load_config_unwraps_and_applies_height_ratio_without_mutating(tmp_path):
    path = tmp_path / "candidate.json"
    original = config_payload()
    path.write_text(json.dumps({"config": original}))

    loaded = load_effective_ik_config(path, actual_human_height=1.0)

    assert loaded["human_scale_table"] == {"pelvis": 0.5, "hand": 0.25}
    assert original["human_scale_table"]["pelvis"] == 1.0


def test_reference_transform_matches_gmr_table1_transform_order():
    config = config_payload()
    frame = {
        "pelvis": (np.array([2.0, 0.0, 1.0]), np.array([1, 0, 0, 0])),
        "hand": (np.array([4.0, 0.0, 2.0]), np.array([1, 0, 0, 0])),
    }

    points = transform_reference_frame(frame, config)

    # Root scales globally. Hand first scales pelvis-relative, then receives
    # table-1 local offset minus ground height, then its global offset.
    np.testing.assert_allclose(points["pelvis"].world_position, [2, 0, 0.9])
    np.testing.assert_allclose(points["hand"].world_position, [4, 0.2, 1.4])


def test_reference_transform_honors_disabled_global_offsets():
    config = config_payload()
    config["use_global_position_offsets"] = False
    frame = {
        "pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0, 0])),
        "hand": (np.array([0.0, 0.0, 1.5]), np.array([1, 0, 0, 0])),
    }

    points = transform_reference_frame(frame, config)

    np.testing.assert_allclose(points["hand"].world_position, [1.0, 0.0, 1.15])


def test_full_reference_uses_nearest_mapped_descendant_scale_and_exact_targets():
    config = config_payload()
    frame = {
        "pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0, 0])),
        "mid": (np.array([0.0, 0.0, 2.0]), np.array([1, 0, 0, 0])),
        "hand": (np.array([0.0, 0.0, 3.0]), np.array([1, 0, 0, 0])),
    }
    exact = transform_reference_frame(frame, config)

    full = transform_full_reference_frame(
        frame,
        config,
        hierarchy_edges=(("pelvis", "mid"), ("mid", "hand")),
        exact_targets=exact,
    )

    assert full["mid"].world_position[2] == pytest.approx(1.5)
    np.testing.assert_allclose(
        full["hand"].world_position, exact["hand"].world_position
    )


def test_position_mappings_include_both_stages_and_resolve_body_or_site():
    mappings = build_position_correspondences(minimal_model(), config_payload())

    assert [(m.stage, m.robot_name, m.frame_type, m.weight) for m in mappings] == [
        (1, "pelvis", "body", 10.0),
        (1, "wrist", "site", 5.0),
        (2, "wrist", "site", 7.0),
    ]


def test_correspondence_uses_live_body_and_site_positions():
    model = minimal_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mappings = build_position_correspondences(model, config_payload())
    refs = transform_reference_frame(
        {
            "pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0, 0])),
            "hand": (np.array([0.0, 0.0, 1.5]), np.array([1, 0, 0, 0])),
        },
        config_payload(),
    )

    rows = compute_correspondences(model, data, refs, mappings)

    wrist = next(row for row in rows if row.robot_name == "wrist")
    np.testing.assert_allclose(wrist.robot_world_position, data.site("wrist").xpos)
    assert wrist.error_norm == pytest.approx(
        np.linalg.norm(wrist.robot_world_position - wrist.reference_world_position)
    )


def test_scene_xml_is_composed_from_arbitrary_mjcf_path(tmp_path):
    robot = tmp_path / "robot.xml"
    robot.write_text("<mujoco model='robot'><worldbody><body name='base'/></worldbody></mujoco>")

    xml = compose_scene_xml(robot)
    model = mujoco.MjModel.from_xml_string(xml)

    assert model.body("base").id > 0
    assert model.geom("gmr_debug_floor").id >= 0
    assert model.texture("gmr_debug_skybox").id >= 0


def test_height_offset_grounds_low_percentile_of_named_sole_sites():
    model = minimal_model()
    motion = {
        "root_pos": np.array([[0, 0, 0.0], [0, 0, -0.1]]),
        "root_rot": np.tile([0, 0, 0, 1.0], (2, 1)),
        "dof_pos": np.empty((2, 0)),
    }

    offset = compute_height_offset(
        model, motion, foot_frames=("left_sole", "right_sole"), percentile=0
    )

    assert offset == pytest.approx(1.0)


def test_visualizer_changes_only_robot_alpha_and_restores_it():
    model = minimal_model()
    original = model.geom_rgba.copy()
    visualizer = GMRDebugVisualizer(model, config_payload(), robot_alpha=0.3)

    floor_id = model.geom("floor").id
    robot_id = model.geom("robot_visual").id
    assert model.geom_rgba[floor_id, 3] == original[floor_id, 3]
    assert model.geom_rgba[robot_id, 3] == pytest.approx(0.3)

    visualizer.restore_robot_alpha()
    np.testing.assert_allclose(model.geom_rgba, original)


def test_visualizer_draws_reference_robot_bones_errors_and_text():
    model = minimal_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    config = config_payload()
    visualizer = GMRDebugVisualizer(
        model,
        config,
        error_threshold=0.0,
        display_name="walking_medium07_stageii",
    )
    references = transform_reference_frame(
        {
            "pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1, 0, 0, 0])),
            "hand": (np.array([0.0, 0.0, 1.5]), np.array([1, 0, 0, 0])),
        },
        config,
    )

    class Viewer:
        user_scn = mujoco.MjvScene(model, maxgeom=100)
        texts = None

        def set_texts(self, texts):
            self.texts = texts

    viewer = Viewer()
    stats = visualizer.update(viewer, data, references, frame_index=3)

    assert viewer.user_scn.ngeom >= 7
    assert viewer.texts is not None
    assert viewer.texts[0][2].startswith("motion\n")
    assert viewer.texts[0][3].startswith("walking_medium07_stageii\n")
    assert stats.frame_index == 3
    assert stats.maximum >= stats.mean > 0.0
