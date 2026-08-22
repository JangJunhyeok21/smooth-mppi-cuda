#!/usr/bin/env python3
"""Step 4: identify steering mapping and servo lag after classic fitting."""
from pathlib import Path

import steering_actuator_regression as regression


# ---------------------------------------------------------------------------
# User-configurable Step 4 settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "model_tuning/data/0821"
OUTPUT_DIR = ROOT / "model_tuning/results/steering_actuator_regression"
STEER_SCALE_MIN = 0.15
STEER_SCALE_MAX = 1.2
STEER_BIAS_MIN_RAD = -0.15
STEER_BIAS_MAX_RAD = 0.15
STEER_TIME_CONSTANT_MIN_S = 0.01
STEER_TIME_CONSTANT_MAX_S = 0.6
RANDOM_SEED = 31
USE_PLOT = True
GT_CONSISTENCY_MODE = "none" # "adjust_pose_to_states", "adjust_states_to_pose", or "none"
POSE_DERIVATIVE_SMOOTH_WINDOW_S = 0.20

# Step 4 open-loop loss/model-selection weights. These are independent user
# settings, but normally should match Step 3 for a consistent deployment gate.
VX_LOSS_WEIGHT = 0.1
VY_LOSS_WEIGHT = 0.1
YAW_RATE_LOSS_WEIGHT = 1.5
POSITION_LOSS_WEIGHT = 8.0
YAW_TRAJECTORY_LOSS_WEIGHT = 1.5


def main():
    """Apply the visible Step 4 settings and run the implementation."""
    regression.DATA = Path(DATA_PATH).expanduser().resolve()
    regression.OUT = Path(OUTPUT_DIR).expanduser().resolve()
    regression.STATIC_BOUNDS = (
        (STEER_SCALE_MIN, STEER_SCALE_MAX),
        (STEER_BIAS_MIN_RAD, STEER_BIAS_MAX_RAD),
    )
    regression.TAU_BOUNDS = (
        STEER_TIME_CONSTANT_MIN_S, STEER_TIME_CONSTANT_MAX_S)
    regression.SEED = RANDOM_SEED
    regression.SHOW_PLOTS = USE_PLOT
    regression.classic.GT_CONSISTENCY_MODE = GT_CONSISTENCY_MODE
    regression.classic.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = POSE_DERIVATIVE_SMOOTH_WINDOW_S
    regression.classic.VX_LOSS_WEIGHT = VX_LOSS_WEIGHT
    regression.classic.VY_LOSS_WEIGHT = VY_LOSS_WEIGHT
    regression.classic.YAW_RATE_LOSS_WEIGHT = YAW_RATE_LOSS_WEIGHT
    regression.classic.POSITION_LOSS_WEIGHT = POSITION_LOSS_WEIGHT
    regression.classic.YAW_TRAJECTORY_LOSS_WEIGHT = YAW_TRAJECTORY_LOSS_WEIGHT
    regression.main()

if __name__ == "__main__":
    main()
