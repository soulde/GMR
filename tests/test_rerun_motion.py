import mujoco
import numpy as np
import pytest
import rerun as rr

from general_motion_retargeting.rerun_motion import (
    MotionFrames,
    MotionSpec,
    discover_motions,
    log_motion,
    load_and_validate_motion,
    load_and_validate_urdf,
    record_motions,
)


def free_model(dofs=1):
    children = "\n".join(
        f"""
        <body name="link_{index}" pos="0 0 0.1">
          <joint name="joint_{index}" type="hinge"/>
          <geom type="sphere" size="0.05"/>
        </body>
        """
        for index in range(dofs)
    )
    return mujoco.MjModel.from_xml_string(
        f"""
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.1"/>
              {children}
            </body>
          </worldbody>
        </mujoco>
        """
    )


def motion_tuple(frames=2, dofs=1, fps=30.0):
    return (
        {},
        fps,
        np.zeros((frames, 3)),
        np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        np.zeros((frames, dofs)),
        None,
        None,
    )


def test_discover_motions_recursively_with_stable_relative_names(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "walk.npz").touch()
    (tmp_path / "a" / "walk.pkl").touch()
    (tmp_path / "notes.txt").touch()

    specs = discover_motions(tmp_path)

    assert [(s.path.relative_to(tmp_path).as_posix(), s.name) for s in specs] == [
        ("a/walk.pkl", "a/walk"),
        ("b/walk.npz", "b/walk"),
    ]


def test_discover_single_file_uses_stem(tmp_path):
    path = tmp_path / "walk.pkl"
    path.touch()

    assert discover_motions(path) == [MotionSpec(path, "walk")]


def test_discover_rejects_unsupported_file(tmp_path):
    path = tmp_path / "walk.txt"
    path.touch()

    with pytest.raises(ValueError, match="unsupported motion file"):
        discover_motions(path)


def test_discover_rejects_empty_folder(tmp_path):
    with pytest.raises(ValueError, match=r"no \.pkl or \.npz motion files"):
        discover_motions(tmp_path)


def test_load_and_validate_motion_accepts_normalized_motion(tmp_path, monkeypatch):
    spec = MotionSpec(tmp_path / "walk.pkl", "walk")
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_robot_motion",
        lambda _: motion_tuple(),
    )

    frames = load_and_validate_motion(spec, free_model())

    assert frames.fps == 30.0
    assert frames.root_pos.shape == (2, 3)
    assert frames.root_rot.shape == (2, 4)
    assert frames.dof_pos.shape == (2, 1)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (motion_tuple(frames=0), "at least one frame"),
        (motion_tuple(fps=0.0), "positive finite fps"),
        (motion_tuple(fps=np.nan), "positive finite fps"),
        (
            ({}, 30.0, np.zeros((2, 2)), np.zeros((2, 4)), np.zeros((2, 1)), None, None),
            "root_pos",
        ),
        (
            ({}, 30.0, np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)), None, None),
            "root_rot",
        ),
        (
            ({}, 30.0, np.zeros((2, 3)), np.zeros((1, 4)), np.zeros((2, 1)), None, None),
            "frame counts",
        ),
    ],
)
def test_load_and_validate_motion_rejects_invalid_arrays(
    tmp_path, monkeypatch, payload, message
):
    spec = MotionSpec(tmp_path / "bad.pkl", "bad")
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_robot_motion",
        lambda _: payload,
    )

    with pytest.raises(ValueError, match=message) as error:
        load_and_validate_motion(spec, free_model())

    assert str(spec.path) in str(error.value)


def test_load_and_validate_motion_rejects_nonfinite_values(tmp_path, monkeypatch):
    spec = MotionSpec(tmp_path / "bad.pkl", "bad")
    payload = list(motion_tuple())
    payload[2][0, 0] = np.nan
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_robot_motion",
        lambda _: tuple(payload),
    )

    with pytest.raises(ValueError, match="finite values"):
        load_and_validate_motion(spec, free_model())


def test_load_and_validate_motion_rejects_nonfree_model(tmp_path, monkeypatch):
    spec = MotionSpec(tmp_path / "bad.pkl", "bad")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody><body><joint type="hinge"/><geom type="sphere" size="0.1"/></body></worldbody>
        </mujoco>
        """
    )
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_robot_motion",
        lambda _: motion_tuple(dofs=0),
    )

    with pytest.raises(ValueError, match="free root joint"):
        load_and_validate_motion(spec, model)


def test_load_and_validate_motion_rejects_wrong_dof_count(tmp_path, monkeypatch):
    spec = MotionSpec(tmp_path / "bad.pkl", "bad")
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_robot_motion",
        lambda _: motion_tuple(dofs=2),
    )

    with pytest.raises(ValueError, match="expected 1 DoF"):
        load_and_validate_motion(spec, free_model(dofs=1))


def write_urdf(tmp_path, joint_name="joint_0"):
    path = tmp_path / "robot.urdf"
    path.write_text(
        f"""
        <robot name="test">
          <link name="base"/>
          <link name="link_0"/>
          <joint name="{joint_name}" type="revolute">
            <parent link="base"/>
            <child link="link_0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1" upper="1" effort="1" velocity="1"/>
          </joint>
        </robot>
        """
    )
    return path


def test_load_and_validate_urdf_matches_mjcf_joint_order(tmp_path):
    tree, joints = load_and_validate_urdf(
        write_urdf(tmp_path),
        free_model(),
        entity_path_prefix="motions/walk/robot",
        frame_prefix="walk/",
    )

    assert tree.root_link().name == "base"
    assert [joint.name for joint in joints] == ["joint_0"]
    assert joints[0].compute_transform(0.25) is not None


def test_load_and_validate_urdf_rejects_joint_name_mismatch(tmp_path):
    with pytest.raises(ValueError, match="URDF movable joints do not match MJCF"):
        load_and_validate_urdf(
            write_urdf(tmp_path, joint_name="wrong_joint"),
            free_model(),
            entity_path_prefix="motions/walk/robot",
            frame_prefix="walk/",
        )


class FakeRecording:
    def __init__(self):
        self.logs = []
        self.times = []
        self.disabled = []

    def log(self, entity_path, archetype, *, static=False):
        self.logs.append((entity_path, archetype, static))

    def set_time(self, timeline, **time):
        self.times.append((timeline, time))

    def disable_timeline(self, timeline):
        self.disabled.append(timeline)


class FakeJoint:
    name = "joint_0"

    def __init__(self):
        self.clamp_values = []

    def compute_transform(self, value, clamp=True):
        self.clamp_values.append(clamp)
        return rr.Transform3D(
            rotation_axis_angle=rr.RotationAxisAngle(
                axis=[0.0, 0.0, 1.0],
                radians=value,
            ),
            parent_frame="walk/base",
            child_frame="walk/link_0",
        )


class FakeTree:
    def log_urdf_to_recording(self, recording):
        recording.log("motions/walk/robot/model", rr.Points3D([[0, 0, 0]]), static=True)

    def root_link(self):
        return type("Root", (), {"name": "base"})()


def test_log_motion_uses_urdf_root_and_joint_transforms(monkeypatch):
    model = free_model()
    frames = MotionFrames(
        fps=20.0,
        root_pos=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        root_rot=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        dof_pos=np.zeros((2, 1)),
    )
    recording = FakeRecording()
    joint = FakeJoint()
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_and_validate_urdf",
        lambda *args, **kwargs: (FakeTree(), [joint]),
    )

    log_motion(
        recording,
        model,
        "robot.urdf",
        MotionSpec(None, "turn_left/a"),
        frames,
    )

    assert recording.times == [
        ("turn_left/a/frame", {"sequence": 0}),
        ("turn_left/a/time", {"duration": 0.0}),
        ("turn_left/a/frame", {"sequence": 1}),
        ("turn_left/a/time", {"duration": 0.05}),
    ]
    assert recording.disabled == ["turn_left/a/frame", "turn_left/a/time"]
    static_logs = [entry for entry in recording.logs if entry[2]]
    dynamic_logs = [entry for entry in recording.logs if not entry[2]]
    assert len(static_logs) == 2
    assert len(dynamic_logs) == 4
    assert all(isinstance(entry[1], rr.Transform3D) for entry in dynamic_logs)
    assert joint.clamp_values == [False, False]


def test_record_motions_loads_and_logs_every_spec(monkeypatch):
    specs = [MotionSpec(None, "a"), MotionSpec(None, "b")]
    frames = MotionFrames(
        fps=30.0,
        root_pos=np.zeros((1, 3)),
        root_rot=np.array([[1.0, 0.0, 0.0, 0.0]]),
        dof_pos=np.zeros((1, 1)),
    )
    loaded = []
    logged = []
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.load_and_validate_motion",
        lambda spec, model: loaded.append(spec.name) or frames,
    )
    monkeypatch.setattr(
        "general_motion_retargeting.rerun_motion.log_motion",
        lambda recording, model, urdf, spec, motion: logged.append(spec.name),
    )

    record_motions(FakeRecording(), free_model(), "robot.urdf", specs)

    assert loaded == ["a", "b"]
    assert logged == ["a", "b"]


def test_cli_rejects_no_spawn_without_save(tmp_path):
    from scripts.rerun_robot_motion import main

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--input",
                str(tmp_path),
                "--mjcf",
                "robot.xml",
                "--urdf",
                "robot.urdf",
                "--no-spawn",
            ]
        )

    assert error.value.code == 2


def test_cli_requires_urdf(tmp_path):
    from scripts.rerun_robot_motion import main

    with pytest.raises(SystemExit) as error:
        main(["--input", str(tmp_path), "--mjcf", "robot.xml"])

    assert error.value.code == 2


def test_cli_headless_export_writes_rrd(tmp_path):
    import pickle

    from scripts.rerun_robot_motion import main

    xml = tmp_path / "robot.xml"
    xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    urdf = write_urdf(tmp_path, joint_name="unused")
    xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body>
              <freejoint/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    urdf.write_text('<robot name="test"><link name="base"/></robot>')
    motion = tmp_path / "motion.pkl"
    with motion.open("wb") as output:
        pickle.dump(
            {
                "fps": 30.0,
                "root_pos": np.zeros((2, 3)),
                "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
                "dof_pos": np.zeros((2, 0)),
                "local_body_pos": None,
                "link_body_list": None,
            },
            output,
        )
    rrd = tmp_path / "motion.rrd"

    assert (
        main(
            [
                "--input",
                str(motion),
                "--mjcf",
                str(xml),
                "--urdf",
                str(urdf),
                "--save",
                str(rrd),
                "--no-spawn",
            ]
        )
        == 0
    )
    assert rrd.stat().st_size > 0
