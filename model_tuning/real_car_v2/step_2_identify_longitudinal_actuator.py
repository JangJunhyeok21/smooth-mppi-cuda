#!/usr/bin/env python3
"""Step 2: identify longitudinal actuator K_v, tau_acc and tau_brake.

This calibration is invoked by the alternating refinement loop. If accepted,
Evaluate all tuned parameters in Step 6 before regenerating causal KF states.
The v_ref slew-rate limit is a fixed runtime constraint read from params.yaml;
it is deliberately not an identification variable.
"""
from pathlib import Path

import visualize_and_regress_longitudinal_actuator as regression


# ---------------------------------------------------------------------------
# User-configurable Step 2 settings
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIRS = (ROOT / "model_tuning/data/ifac2026",)
OUTPUT_DIR = ROOT / "model_tuning/results/longitudinal_actuator_regression"

# Identified parameter bounds. Units: Kp [1/s], time constants [s].
SPEED_SERVO_KP_MIN = 0.05
SPEED_SERVO_KP_MAX = 60.0
ACCEL_TIME_CONSTANT_MIN = 0.002
ACCEL_TIME_CONSTANT_MAX = 0.8
BRAKE_TIME_CONSTANT_MIN = 0.002
BRAKE_TIME_CONSTANT_MAX = 0.8

# Match the MPPI/Step-3/Step-6 horizon while retaining the 20 ms source grid.
# 60 model knots * 40 ms = 2.4 s = 120 source samples * 20 ms.
MODEL_DT_S = 0.04
HORIZON_STEPS = 40
WARMUP_DURATION_S = 0.8
START_STRIDE_SAMPLES = 5
MAX_ROLLOUTS_PER_SESSION = 800
OPTIMIZER_POPULATION_SIZE = 36
OPTIMIZER_MAX_ITERATIONS = 80
OPTIMIZER_LOCAL_MAX_ITERATIONS = 600
RANDOM_SEED = 31
UPDATE_PARAMS_YAML = True
USE_PLOT = True

# False: use every bag for fitting and reuse one train bag only for the
# performance plots/metrics (in-sample diagnostic, not generalization).
USE_VALIDATION_TEST_SPLIT = False
TRAIN_EVALUATION_BAG_INDEX = -1   # -1 selects the last usable train bag


def main():
    """Apply the visible Step 2 settings and run the implementation."""
    regression.SOURCE_DIRS = tuple(Path(path).expanduser().resolve()
                                   for path in SOURCE_DIRS)
    regression.OUTPUT_DIR = Path(OUTPUT_DIR).expanduser().resolve()
    regression.SPEED_SERVO_KP_BOUNDS = (
        SPEED_SERVO_KP_MIN, SPEED_SERVO_KP_MAX)
    regression.ACCEL_TIME_CONSTANT_BOUNDS = (
        ACCEL_TIME_CONSTANT_MIN, ACCEL_TIME_CONSTANT_MAX)
    regression.BRAKE_TIME_CONSTANT_BOUNDS = (
        BRAKE_TIME_CONSTANT_MIN, BRAKE_TIME_CONSTANT_MAX)
    regression.BOUNDS = (regression.SPEED_SERVO_KP_BOUNDS,
                         regression.ACCEL_TIME_CONSTANT_BOUNDS,
                         regression.BRAKE_TIME_CONSTANT_BOUNDS)
    rollout_duration_s = HORIZON_STEPS * MODEL_DT_S
    regression.ROLLOUT_STEPS = max(
        1, int(round(rollout_duration_s / regression.SOURCE_DT_S)))
    regression.MODEL_DT_S = MODEL_DT_S
    regression.HORIZON_STEPS = HORIZON_STEPS
    regression.WARMUP_S = WARMUP_DURATION_S
    regression.START_STRIDE = START_STRIDE_SAMPLES
    regression.MAX_ROLLOUTS_PER_SESSION = MAX_ROLLOUTS_PER_SESSION
    regression.OPTIMIZER_POPULATION_SIZE = OPTIMIZER_POPULATION_SIZE
    regression.OPTIMIZER_MAX_ITERATIONS = OPTIMIZER_MAX_ITERATIONS
    regression.OPTIMIZER_LOCAL_MAX_ITERATIONS = OPTIMIZER_LOCAL_MAX_ITERATIONS
    regression.RANDOM_SEED = RANDOM_SEED
    regression.UPDATE_CONFIG = UPDATE_PARAMS_YAML
    regression.SHOW_PLOTS = USE_PLOT
    regression.USE_VALIDATION_TEST_SPLIT = USE_VALIDATION_TEST_SPLIT
    regression.TRAIN_EVALUATION_BAG_INDEX = TRAIN_EVALUATION_BAG_INDEX
    regression.main()

if __name__ == "__main__":
    main()
