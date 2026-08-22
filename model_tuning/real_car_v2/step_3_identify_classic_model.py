#!/usr/bin/env python3
"""Step 3: identify Pacejka front/rear parameters and yaw inertia I_z."""
from pathlib import Path

import numpy as np

import classic_model_regression as regression


# ---------------------------------------------------------------------------
# User-configurable Step 3 settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
OUTPUT_DIR = ROOT / "model_tuning/results/dynamic_40ms_regression"
ROLLOUT_HORIZON_STEPS = 60       # 60 * 40 ms = 2.4 s
MAX_WINDOWS_PER_BAG = 80
ACTUATOR_WARMUP_SAMPLES = 40     # 40 * 20 ms = 0.8 s
RANDOM_SEED = 31
USE_PLOT = True
INTERACTIVE_BAG_INSPECTOR = True  # p + time click opens detailed open-loop plots
TRAJECTORY_TIME_LABEL_INTERVAL_S = 1.0
EVALUATE_ONLY = False  # True: load saved params and only regenerate diagnostics
EVALUATION_PARAMS_PATH = OUTPUT_DIR / "params.json"
# Apply a gate-passing candidate to config/params.yaml for the next numbered
# stage. A rejected/boundary candidate remains isolated in params.json.
APPLY_ACCEPTED_PARAMS_TO_YAML = True

# False: fit with every split and reuse one train bag for performance metrics
# and plots. This is an in-sample diag01.nostic, not held-out generalization.
USE_VALIDATION_TEST_SPLIT = False
TRAIN_EVALUATION_BAG_INDEX = -1   # -1 selects the last usable train bag

# Reject only unmistakable MCL discontinuities. Step 1 manual review owns
# collision removal; legitimate high-acceleration/high-slip data remain here.
MAX_POSITION_STEP_20MS = 0.5 # Reject windows with >0.5 m position change in 20 ms (50 m/s)
MAX_YAW_STEP_20MS = 0.5 # Reject windows with >0.5 rad yaw change in 20 ms (25 rad/s)
AUTO_FIT_POSITION_SPEED_SCALE = False
POSITION_SPEED_SCALE_BOUNDS = (0.70, 1.20)

# Resolve the pose/state contradiction in exactly one of three ways:
#   "adjust_pose_to_states": keep KF vx/vy/yaw_rate; integrate them into pose GT
#   "adjust_states_to_pose": keep raw MCL pose; differentiate it into state GT
#   "none": keep original KF states and raw MCL pose even when inconsistent
GT_CONSISTENCY_MODE = "adjust_states_to_pose"  # "adjust_pose_to_states", "adjust_states_to_pose", or "none"

# Open-loop optimization/model-selection weights. Increase POSITION_LOSS_WEIGHT
# when MPPI path placement is more important than matching individual states.
VX_LOSS_WEIGHT = 0.0
VY_LOSS_WEIGHT = 0.1
YAW_RATE_LOSS_WEIGHT = 1.5
POSITION_LOSS_WEIGHT = 1.5
YAW_TRAJECTORY_LOSS_WEIGHT = 1.5

# Smoothing used only by "adjust_states_to_pose" before pose differentiation.
VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = 0.20

# Pacejka bounds: B, C, D, E for front and rear tires.
PACEJKA_B_F_BOUNDS = (0.2, 30.0)
PACEJKA_C_F_BOUNDS = (0.0, 2.5)
PACEJKA_D_F_BOUNDS = (0.05, 3.5)
PACEJKA_E_F_BOUNDS = (-2.0, 1.0)
PACEJKA_B_R_BOUNDS = (0.2, 30.0)
PACEJKA_C_R_BOUNDS = (0.0, 2.5)
PACEJKA_D_R_BOUNDS = (0.05, 3.5)
PACEJKA_E_R_BOUNDS = (-2.0, 1.0)
YAW_INERTIA_MIN = 0.005
YAW_INERTIA_MAX = 0.5

ADAM_RESTARTS = 3
ADAM_STEPS = 600
SURROGATE_SAMPLES = 400
SURROGATE_PROPOSALS = 40000


def main():
    """Apply the visible Step 3 settings and run the implementation."""
    regression.DATA = Path(DATA_PATH).expanduser().resolve()
    regression.OUT = Path(OUTPUT_DIR).expanduser().resolve()
    regression.HORIZON = ROLLOUT_HORIZON_STEPS
    regression.MAX_PER_BAG = MAX_WINDOWS_PER_BAG
    regression.WARMUP_SAMPLES = ACTUATOR_WARMUP_SAMPLES
    regression.SEED = RANDOM_SEED
    regression.SHOW_PLOTS = USE_PLOT
    regression.INTERACTIVE_BAG_INSPECTOR = INTERACTIVE_BAG_INSPECTOR
    regression.TRAJECTORY_TIME_LABEL_INTERVAL_S = TRAJECTORY_TIME_LABEL_INTERVAL_S
    regression.EVALUATE_ONLY = EVALUATE_ONLY
    regression.EVALUATION_PARAMS_PATH = Path(EVALUATION_PARAMS_PATH).expanduser().resolve()
    regression.APPLY_ACCEPTED_PARAMS_TO_YAML = APPLY_ACCEPTED_PARAMS_TO_YAML
    regression.USE_VALIDATION_TEST_SPLIT = USE_VALIDATION_TEST_SPLIT
    regression.TRAIN_EVALUATION_BAG_INDEX = TRAIN_EVALUATION_BAG_INDEX
    regression.MAX_POSITION_STEP_20MS = MAX_POSITION_STEP_20MS
    regression.MAX_YAW_STEP_20MS = MAX_YAW_STEP_20MS
    regression.AUTO_FIT_POSITION_SPEED_SCALE = AUTO_FIT_POSITION_SPEED_SCALE
    regression.POSITION_SPEED_SCALE_BOUNDS = POSITION_SPEED_SCALE_BOUNDS
    regression.GT_CONSISTENCY_MODE = GT_CONSISTENCY_MODE
    regression.VX_LOSS_WEIGHT = VX_LOSS_WEIGHT
    regression.VY_LOSS_WEIGHT = VY_LOSS_WEIGHT
    regression.YAW_RATE_LOSS_WEIGHT = YAW_RATE_LOSS_WEIGHT
    regression.POSITION_LOSS_WEIGHT = POSITION_LOSS_WEIGHT
    regression.YAW_TRAJECTORY_LOSS_WEIGHT = YAW_TRAJECTORY_LOSS_WEIGHT
    regression.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S
    regression.BOUNDS = np.asarray((
        PACEJKA_B_F_BOUNDS, PACEJKA_C_F_BOUNDS,
        PACEJKA_D_F_BOUNDS, PACEJKA_E_F_BOUNDS,
        PACEJKA_B_R_BOUNDS, PACEJKA_C_R_BOUNDS,
        PACEJKA_D_R_BOUNDS, PACEJKA_E_R_BOUNDS,
    ), dtype=np.float64)
    regression.I_Z_MIN = YAW_INERTIA_MIN
    regression.I_Z_MAX = YAW_INERTIA_MAX
    regression.ADAM_RESTARTS = ADAM_RESTARTS
    regression.ADAM_STEPS = ADAM_STEPS
    regression.SURROGATE_SAMPLES = SURROGATE_SAMPLES
    regression.SURROGATE_PROPOSALS = SURROGATE_PROPOSALS
    regression.main()

if __name__ == "__main__":
    main()
