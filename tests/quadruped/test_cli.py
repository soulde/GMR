import pickle
from pathlib import Path

import numpy as np

from scripts.motion_imitation_to_robot import build_parser, run


FIXTURE = Path(__file__).parent / "fixtures/motion_imitation_two_frames.txt"


def test_parser_requires_motion_and_defaults_to_go2():
    args = build_parser().parse_args(["--motion_file", "dog_pace.txt"])

    assert args.model_type == "quadruped"
    assert args.source_robot == "laikago"
    assert args.robot == "unitree_go2"


def test_headless_fixture_writes_gmr_motion(tmp_path):
    output = tmp_path / "motion.pkl"

    result = run(
        motion_file=FIXTURE,
        source_robot="laikago",
        robot="unitree_go2",
        save_path=output,
        headless=True,
        rate_limit=False,
    )

    with output.open("rb") as stream:
        motion = pickle.load(stream)
    assert {"root_pos", "root_rot", "dof_pos", "fps"} <= motion.keys()
    assert motion["root_pos"].shape == (2, 3)
    assert motion["root_rot"].shape == (2, 4)
    assert motion["dof_pos"].shape == (2, 12)
    assert np.isfinite(motion["root_pos"]).all()
    assert np.isfinite(motion["root_rot"]).all()
    assert np.isfinite(motion["dof_pos"]).all()
    np.testing.assert_allclose(
        motion["root_rot"],
        result.qpos[:, [4, 5, 6, 3]],
    )
