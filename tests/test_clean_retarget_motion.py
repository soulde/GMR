from scripts.clean_retarget_motion import build_parser


def test_cleaner_cli_defaults_to_skeleton_and_ten_centimeters():
    args = build_parser().parse_args(
        [
            "--motion", "motion.pkl",
            "--reference", "motion.bvh",
            "--mjcf", "robot.xml",
            "--ik-config", "skeleton.json",
            "--output-dir", "cleaned",
        ]
    )

    assert args.algorithm == "skeleton"
    assert args.max_position_error == 0.10
    assert args.minimum_segment_frames == 30
    assert args.padding_frames == 0
