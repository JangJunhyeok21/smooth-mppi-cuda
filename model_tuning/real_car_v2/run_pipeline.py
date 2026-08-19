#!/usr/bin/env python3
"""Run the numbered real-car training pipeline from extraction products onward."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

# User-editable switches. The default performs the complete reproducible run.
RUN_BUILD_DATASET = True
RUN_CLASSIC_REGRESSION = True
RUN_BUILD_RESIDUAL_DATASET = True
RUN_ONE_STEP_TRAINING = True
RUN_RECURSIVE_STAGE1 = True
RUN_RECURSIVE_STAGE2 = True
RUN_EVALUATION = True
RUN_PLOT = True
RUN_DEPLOY_TO_MPPI = True

ONE_STEP = ROOT / "model_tuning/results/dynamic_40ms_residual_seed31"
STAGE1 = ROOT / "model_tuning/results/dynamic_40ms_yaw_preserved_stage1"
STAGE2 = ROOT / "model_tuning/results/dynamic_40ms_yaw_preserved_stage2"


def run(script, *arguments, environment=None):
    command = [PYTHON, str(HERE / script), *map(str, arguments)]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT, env=environment)


def main():
    if RUN_BUILD_DATASET:
        run("step_2_build_20ms_dataset.py")
    if RUN_CLASSIC_REGRESSION:
        run("step_3_regress_classic_model.py")
    if RUN_BUILD_RESIDUAL_DATASET:
        run("step_4_build_40ms_dataset.py")
    if RUN_ONE_STEP_TRAINING:
        run("step_5_train_residual_mlp.py")

    # Preserve the high-speed yaw solution by applying the low-speed gate to
    # all three residual heads during recursive optimization. At runtime/eval,
    # delta_ax is intentionally ungated so launches remain observable.
    yaw_environment = os.environ.copy()
    yaw_environment["GATE_AX_RESIDUAL"] = "1"
    if RUN_RECURSIVE_STAGE1:
        run("step_6_finetune_recursive.py", ONE_STEP,
            "--out", STAGE1, "--epochs", "100", "--seed", "31",
            environment=yaw_environment)
    if RUN_RECURSIVE_STAGE2:
        run("step_6_finetune_recursive.py", STAGE1,
            "--out", STAGE2, "--epochs", "100", "--seed", "31",
            environment=yaw_environment)
    if RUN_EVALUATION:
        run("step_7_evaluate_rollout.py", STAGE2,
            "--out", STAGE2 / "rollout_ax_ungated_metrics.json")
    if RUN_PLOT:
        run("visualize_highspeed_tail_comparison.py")
    if RUN_DEPLOY_TO_MPPI:
        run("step_8_deploy_to_mppi.py")

    print("\nCompleted: dynamic_40ms_yaw_preserved_stage2")
    print("weight:", STAGE2 / "dynamic_40ms_residual.bin")
    print("runtime: config/dynamic_40ms_residual_servo_lag.bin")
    print("contract: control_dt=0.02 s, model_dt=0.04 s")


if __name__ == "__main__":
    main()
