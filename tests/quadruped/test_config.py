from pathlib import Path

import pytest

from general_motion_retargeting.quadruped.config import load_retarget_config


ROOT = Path(__file__).resolve().parents[2]


def test_load_laikago_to_go2_retarget_config():
    config = load_retarget_config(
        ROOT
        / "general_motion_retargeting/quadruped/ik_configs"
        / "laikago_to_unitree_go2.json"
    )

    assert config.root_task.orientation_cost == pytest.approx(100.0)
    assert config.velocity_limit == pytest.approx(3.0 * 3.141592653589793)
    assert config.trajectory_scale["FL"][:2] == pytest.approx((1.0, 1.0))
    assert config.root_translation_scale == pytest.approx((1.0, 1.0, 1.0))
    assert config.foot_tasks["RR"].position_offset == pytest.approx(
        (0.0, 0.0, 0.0)
    )
