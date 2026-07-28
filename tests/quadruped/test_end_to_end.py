from pathlib import Path

import numpy as np
import pytest

from scripts.check_quadruped_motion import check_motion
from scripts.motion_imitation_to_robot import run


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_dog_pace_retargets_to_valid_go2_motion(tmp_path):
    output = tmp_path / "go2_dog_pace.pkl"
    result = run(
        motion_file=ROOT / "assets/quadrupeds/motions/dog_pace.txt",
        source_robot="laikago",
        robot="unitree_go2",
        save_path=output,
        headless=True,
        rate_limit=False,
        use_velocity_limit=True,
    )

    report = check_motion(output, robot="unitree_go2")

    assert result.qpos.shape == (39, 19)
    np.testing.assert_allclose(
        result.qpos[0, 3:7],
        [1.0, 0.0, 0.0, 0.0],
        atol=2e-5,
    )
    assert result.qpos[-1, 0] > result.qpos[0, 0]
    assert report["finite"]
    assert report["max_joint_limit_violation"] <= 1e-8
    assert report["max_velocity_violation"] <= 1e-8
    assert report["min_foot_height"] == pytest.approx(0.0, abs=1e-5)
    assert report["non_converged_ratio"] <= 0.05
