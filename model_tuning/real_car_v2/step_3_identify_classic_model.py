#!/usr/bin/env python3
"""Step 3: identify Pacejka front/rear parameters and yaw inertia I_z."""
from pathlib import Path
import os

import numpy as np

import classic_model_regression as regression


# ---------------------------------------------------------------------------
# User-configurable Step 3 settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
# 변경사항
DATA_PATH = ROOT / "model_tuning/data/ifac2026"
YAML_EVALUATION_MODE = False # os.environ.get("STEP3_YAML_EVALUATION_MODE", "0") != "0"
REGRESSION_METHODS = ("de_robust_ls",) # adam, de_robust_ls, mlp_surrogate

OUTPUT_DIR = Path(os.environ.get("DYNAMIC_REGRESSION_OUT",
    ROOT / "model_tuning/results/dynamic_40ms_regression_collision_refined"))
ROLLOUT_HORIZON_STEPS = 40       # 40 * 40 ms = 1.6 s
MAX_WINDOWS_PER_BAG = 0           # 0 = keep every eligible window in each bag
WINDOW_START_STRIDE = 1           # 1 = do not discard every second/third window
V_MIN = 0.1                      # [m/s] reject rollouts containing lower GT vx
ACTUATOR_WARMUP_SAMPLES = 40     # 40 * 20 ms = 0.8 s
RANDOM_SEED = 31
USE_PLOT = os.environ.get("STEP3_USE_PLOT","1")!="0"
INTERACTIVE_BAG_INSPECTOR = os.environ.get("STEP3_INTERACTIVE_PLOT","1")!="0"  # p + time click opens detailed open-loop plots
TRAJECTORY_TIME_LABEL_INTERVAL_S = 1.0 # Traj 라벨 몇초간격으로 표시할지. 0이면 라벨 없음.
EVALUATE_ONLY = False # os.environ.get("STEP3_EVALUATE_ONLY", "0") != "0"
# STEP3_EVALUATE_ONLY=1: load saved params and only regenerate diagnostics.
# The default performs regression, matching this script's identify step name.
EVALUATION_PARAMS_PATH = OUTPUT_DIR / "params.json"
# True: do not regress or write YAML. Load every Step-1 NPZ in DATA_PATH and
# evaluate the parameters currently stored in config/params.yaml interactively.
# Use Left/Right to change bag, or type a bag number and Enter to jump;
# press p and click a time-series panel to start an open-loop prediction.
# Apply a gate-passing candidate to config/params.yaml for the next numbered
# stage. A rejected/boundary candidate remains isolated in params.json.
APPLY_ACCEPTED_PARAMS_TO_YAML = os.environ.get("STEP3_APPLY_TO_YAML","1")!="0"

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
VY_LOSS_WEIGHT = 0.0
# Global weight search (2026-08-25): keep the original total weight 28.5,
# while using the validation/test Pareto-best ratio yaw-rate:position:yaw =
# 2.5:1.0:0.1.
YAW_RATE_LOSS_WEIGHT = 19.7916666667
POSITION_LOSS_WEIGHT = 7.9166666667
YAW_TRAJECTORY_LOSS_WEIGHT = 0.7916666667

# Smoothing used only by "adjust_states_to_pose" before pose differentiation.
VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = 0.20
# 0 disables longitudinal load transfer. Set a measured/selected CG height to
# use Fzf=(m*g*l_r-m*ax*h_cg)/L and Fzr=(m*g*l_f+m*ax*h_cg)/L.
LOAD_TRANSFER_H_CG_M = float(os.environ.get("LOAD_TRANSFER_H_CG_M","0.0"))

# Pacejka bounds: B, C, D, E for front and rear tires.
PACEJKA_B_F_BOUNDS = (0.2, 30.0)
PACEJKA_C_F_BOUNDS = (0.0, 2.5)
PACEJKA_D_F_BOUNDS = (0.05, 3.5)
PACEJKA_E_F_BOUNDS = (-10.0, 1.0)
PACEJKA_B_R_BOUNDS = (0.2, 30.0)
PACEJKA_C_R_BOUNDS = (0.0, 2.5)
PACEJKA_D_R_BOUNDS = (0.05, 3.5)
PACEJKA_E_R_BOUNDS = (-10.0, 1.0)
# True: E_f=E_r=0으로 고정하고 B/C/D만 회귀한다.
# False: 위의 E bounds를 사용해 B/C/D/E를 모두 회귀한다.
FIX_PACEJKA_E_ZERO = False
YAW_INERTIA_MIN = 0.005
YAW_INERTIA_MAX = 0.5

# Regression backends to execute.  The best validation result among these and
# (optionally) the current YAML model is selected.
# Available modes:
#   "adam"          : differentiable recursive rollout optimized by AdamW
#   "de_robust_ls"  : differential evolution followed by robust least squares
#   "mlp_surrogate": learn an objective surrogate, then search the surrogate
# Adam is the default so Step 3 no longer has to run every expensive backend.
INCLUDE_CURRENT_MODEL_AS_CANDIDATE = True
# This is a separate DE-based Pacejka/I_z coordinate-descent refinement. Keep
# it disabled for a genuinely Adam-only experiment; otherwise it can drive E
# back to a bound even when Adam itself returned an interior solution.
RUN_ALTERNATING_PACEJKA_IZ = False

ADAM_RESTARTS = 3
ADAM_STEPS = 600
DE_POPULATION_SIZE = 6
DE_MAX_ITERATIONS = 35
SURROGATE_SAMPLES = 400
SURROGATE_PROPOSALS = 40000


def main():
    """Apply the visible Step 3 settings and run the implementation."""
    regression.DATA = Path(DATA_PATH).expanduser().resolve()
    regression.OUT = Path(OUTPUT_DIR).expanduser().resolve()
    regression.HORIZON = ROLLOUT_HORIZON_STEPS
    regression.MAX_PER_BAG = MAX_WINDOWS_PER_BAG
    regression.WINDOW_START_STRIDE = WINDOW_START_STRIDE
    if V_MIN < 0.0:
        raise ValueError(f"V_MIN must be non-negative, got {V_MIN}")
    regression.V_MIN = float(V_MIN)
    regression.WARMUP_SAMPLES = ACTUATOR_WARMUP_SAMPLES
    regression.SEED = RANDOM_SEED
    regression.SHOW_PLOTS = USE_PLOT
    regression.INTERACTIVE_BAG_INSPECTOR = INTERACTIVE_BAG_INSPECTOR
    regression.TRAJECTORY_TIME_LABEL_INTERVAL_S = TRAJECTORY_TIME_LABEL_INTERVAL_S
    regression.EVALUATE_ONLY = EVALUATE_ONLY
    regression.YAML_EVALUATION_MODE = YAML_EVALUATION_MODE
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
    regression.LOAD_TRANSFER_H_CG_M = LOAD_TRANSFER_H_CG_M
    regression.BOUNDS = np.asarray((
        PACEJKA_B_F_BOUNDS, PACEJKA_C_F_BOUNDS,
        PACEJKA_D_F_BOUNDS, PACEJKA_E_F_BOUNDS,
        PACEJKA_B_R_BOUNDS, PACEJKA_C_R_BOUNDS,
        PACEJKA_D_R_BOUNDS, PACEJKA_E_R_BOUNDS,
    ), dtype=np.float64)
    regression.FIX_PACEJKA_E_ZERO = bool(FIX_PACEJKA_E_ZERO)
    if FIX_PACEJKA_E_ZERO:
        regression.BOUNDS[[3, 7]] = 0.0
    regression.I_Z_MIN = YAW_INERTIA_MIN
    regression.I_Z_MAX = YAW_INERTIA_MAX
    regression.REGRESSION_METHODS = REGRESSION_METHODS
    regression.INCLUDE_CURRENT_MODEL_AS_CANDIDATE = INCLUDE_CURRENT_MODEL_AS_CANDIDATE
    regression.RUN_ALTERNATING_PACEJKA_IZ = RUN_ALTERNATING_PACEJKA_IZ
    regression.ADAM_RESTARTS = ADAM_RESTARTS
    regression.ADAM_STEPS = ADAM_STEPS
    regression.DE_POPSIZE = DE_POPULATION_SIZE
    regression.DE_MAXITER = DE_MAX_ITERATIONS
    regression.SURROGATE_SAMPLES = SURROGATE_SAMPLES
    regression.SURROGATE_PROPOSALS = SURROGATE_PROPOSALS
    regression.main()

if __name__ == "__main__":
    main()
