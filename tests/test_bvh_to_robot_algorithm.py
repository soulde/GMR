from general_motion_retargeting import (
    GeneralMotionRetargeting,
    SkeletonMotionRetargeting,
)
from scripts.bvh_to_robot import build_parser, retargeter_class


def test_bvh_cli_defaults_to_original_gmr():
    args = build_parser().parse_args(["--bvh_file", "walk.bvh"])

    assert args.algorithm == "gmr"
    assert retargeter_class(args.algorithm) is GeneralMotionRetargeting


def test_bvh_cli_selects_skeleton_retargeter():
    args = build_parser().parse_args(
        ["--bvh_file", "walk.bvh", "--algorithm", "skeleton"]
    )

    assert retargeter_class(args.algorithm) is SkeletonMotionRetargeting
