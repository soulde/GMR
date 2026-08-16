import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

import general_motion_retargeting.robot_motion_viewer as viewer_module
from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer


@pytest.fixture
def minimal_robot(tmp_path, monkeypatch):
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="2 2 .1"/>
            <body name="pelvis" pos="0 0 1">
              <freejoint/>
              <geom name="robot_visual" type="sphere" size=".1"/>
              <body name="left_knee_link" pos="0 0 -.5">
                <geom type="sphere" size=".05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    config_path = tmp_path / "ik.json"
    config_path.write_text(
        json.dumps(
            {
                "human_root_name": "Hips",
                "human_height_assumption": 1.75,
                "ground_height": 0.0,
                "human_scale_table": {"Hips": 1.0, "LeftLeg": 1.0},
                "use_ik_match_table1": True,
                "use_ik_match_table2": False,
                "ik_match_table1": {
                    "pelvis": ["Hips", 1, 0, [0, 0, 0], [1, 0, 0, 0]],
                    "left_knee_link": [
                        "LeftLeg",
                        1,
                        0,
                        [0, 0, 0],
                        [1, 0, 0, 0],
                    ],
                },
                "ik_match_table2": {},
            }
        )
    )

    class FakeViewer:
        def __init__(self, model):
            self.user_scn = mujoco.MjvScene(model, maxgeom=100)
            self.opt = mujoco.MjvOption()
            self.cam = mujoco.MjvCamera()
            self.texts = None
            self.sync_calls = 0
            self.closed = False

        def set_texts(self, texts):
            self.texts = texts

        def sync(self):
            self.sync_calls += 1

        def close(self):
            self.closed = True

    launched = SimpleNamespace(viewer=None)

    def launch_passive(*, model, data, **kwargs):
        launched.viewer = FakeViewer(model)
        return launched.viewer

    monkeypatch.setitem(viewer_module.ROBOT_XML_DICT, "test_robot", xml_path)
    monkeypatch.setitem(viewer_module.ROBOT_BASE_DICT, "test_robot", "pelvis")
    monkeypatch.setitem(
        viewer_module.VIEWER_CAM_DISTANCE_DICT, "test_robot", 2.0
    )
    monkeypatch.setattr(viewer_module.mjv, "launch_passive", launch_passive)
    return SimpleNamespace(config_path=config_path, launched=launched)


def human_targets():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "Hips": (np.array([0.2, 0.0, 1.0]), identity),
        "LeftUpLeg": (np.array([0.2, 0.0, 0.75]), identity),
        "LeftLeg": (np.array([0.2, 0.0, 0.5]), identity),
    }


def step_once(viewer, human_motion_data):
    viewer.step(
        root_pos=np.array([0.0, 0.0, 0.0]),
        root_rot=np.array([1.0, 0.0, 0.0, 0.0]),
        dof_pos=np.empty(0),
        human_motion_data=human_motion_data,
        rate_limit=False,
        follow_camera=False,
    )


def test_debug_mode_requires_ik_config_path(minimal_robot):
    with pytest.raises(ValueError, match="ik_config_path"):
        RobotMotionViewer("test_robot", debug=True)


def test_default_mode_does_not_create_debug_visualizer(minimal_robot):
    viewer = RobotMotionViewer("test_robot")

    assert viewer.debug_visualizer is None

    viewer.close()


def test_debug_step_draws_overlay_and_error_statistics(minimal_robot):
    viewer = RobotMotionViewer(
        "test_robot", debug=True, ik_config_path=minimal_robot.config_path
    )

    step_once(viewer, human_targets())

    assert viewer.viewer.user_scn.ngeom > 0
    assert viewer.viewer.texts
    assert viewer.viewer.texts[0][2].startswith("frame\n")
    viewer.close()


def test_explicit_human_edges_override_inferred_edges(minimal_robot, monkeypatch):
    viewer = RobotMotionViewer(
        "test_robot", debug=True, ik_config_path=minimal_robot.config_path
    )
    captured = {}

    def update(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(viewer.debug_visualizer, "update", update)
    explicit = (("Hips", "LeftLeg"),)

    viewer.step(
        root_pos=np.array([0.0, 0.0, 0.0]),
        root_rot=np.array([1.0, 0.0, 0.0, 0.0]),
        dof_pos=np.empty(0),
        human_motion_data=human_targets(),
        human_skeleton_edges=explicit,
        rate_limit=False,
        follow_camera=False,
    )

    assert captured["full_reference_edges"] == explicit
    viewer.close()


def test_debug_step_skips_missing_targets_and_clears_stale_overlay(minimal_robot):
    viewer = RobotMotionViewer(
        "test_robot", debug=True, ik_config_path=minimal_robot.config_path
    )

    step_once(viewer, {"Hips": human_targets()["Hips"]})
    assert viewer.viewer.user_scn.ngeom > 0

    step_once(viewer, None)

    assert viewer.viewer.user_scn.ngeom == 0
    assert viewer.viewer.texts == []
    viewer.close()


def test_close_restores_robot_alpha(minimal_robot):
    viewer = RobotMotionViewer(
        "test_robot",
        debug=True,
        ik_config_path=minimal_robot.config_path,
        debug_robot_alpha=0.2,
    )
    original_rgba = viewer.debug_visualizer._original_rgba.copy()

    viewer.close()

    np.testing.assert_allclose(viewer.model.geom_rgba, original_rgba)
    assert viewer.viewer.closed
