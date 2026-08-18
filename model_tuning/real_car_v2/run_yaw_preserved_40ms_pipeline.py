#!/usr/bin/env python3
"""One-command reproduction/deployment of dynamic_40ms_yaw_preserved_stage2."""
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
        run("build_dataset.py")
    if RUN_CLASSIC_REGRESSION:
        run("regress_dynamic_40ms_advanced.py")
    if RUN_BUILD_RESIDUAL_DATASET:
        run("build_dynamic_40ms_dataset.py")
    if RUN_ONE_STEP_TRAINING:
        run("train_dynamic_40ms.py")

    # Preserve the high-speed yaw solution by applying the low-speed gate to
    # all three residual heads during recursive optimization. At runtime/eval,
    # delta_ax is intentionally ungated so launches remain observable.
    yaw_environment = os.environ.copy()
    yaw_environment["GATE_AX_RESIDUAL"] = "1"
    if RUN_RECURSIVE_STAGE1:
        run("finetune_dynamic_40ms_recursive.py", ONE_STEP,
            "--out", STAGE1, "--epochs", "100", "--seed", "31",
            environment=yaw_environment)
    if RUN_RECURSIVE_STAGE2:
        run("finetune_dynamic_40ms_recursive.py", STAGE1,
            "--out", STAGE2, "--epochs", "100", "--seed", "31",
            environment=yaw_environment)
    if RUN_EVALUATION:
        run("evaluate_dynamic_40ms.py", STAGE2,
            "--out", STAGE2 / "rollout_ax_ungated_metrics.json")
    if RUN_PLOT:
        run("plot_highspeed_tail_comparison.py")
    if RUN_DEPLOY_TO_MPPI:
        run("deploy_dynamic_40ms_to_mppi.py")

    print("\nCompleted: dynamic_40ms_yaw_preserved_stage2")
    print("weight:", STAGE2 / "dynamic_40ms_residual.bin")
    print("runtime: config/dynamic_40ms_residual_servo_lag.bin")
    print("contract: control_dt=0.02 s, model_dt=0.04 s")


if __name__ == "__main__":
    main()
