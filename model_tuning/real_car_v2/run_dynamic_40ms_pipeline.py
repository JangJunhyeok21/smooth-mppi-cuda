#!/usr/bin/env python3
"""Run the complete 40 ms dynamic + servo-lag residual training pipeline.

Edit only the switches below. No command-line arguments are required.
"""
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

RUN_BUILD_COMBINED_DATASET = True
RUN_ACTUATOR_REGRESSION = True
RUN_CLASSIC_REGRESSION = True
RUN_BUILD_RESIDUAL_DATASET = True
RUN_ONE_STEP_TRAINING = True
RUN_RECURSIVE_STAGE1 = True
RUN_RECURSIVE_STAGE2 = True
RUN_EVALUATION = True
RUN_EXPORT_TO_MPPI = True

ONE_STEP = HERE.parents[1] / "model_tuning/results/dynamic_40ms_residual_seed31"
STAGE1 = HERE.parents[1] / "model_tuning/results/dynamic_40ms_recursive_seed31"
STAGE2 = HERE.parents[1] / "model_tuning/results/dynamic_40ms_recursive_stage2_seed31"


def run(*args):
    command = [PYTHON, *map(str, args)]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=HERE.parents[1])


def main():
    if RUN_BUILD_COMBINED_DATASET:
        run(HERE / "build_dataset.py")
    if RUN_ACTUATOR_REGRESSION:
        run(HERE / "regress_longitudinal_actuator.py")
    if RUN_CLASSIC_REGRESSION:
        run(HERE / "regress_dynamic_40ms.py")
    if RUN_BUILD_RESIDUAL_DATASET:
        run(HERE / "build_dynamic_40ms_dataset.py")
    if RUN_ONE_STEP_TRAINING:
        run(HERE / "train_dynamic_40ms.py")
    if RUN_RECURSIVE_STAGE1:
        run(HERE / "finetune_dynamic_40ms_recursive.py")
    if RUN_RECURSIVE_STAGE2:
        run(HERE / "finetune_dynamic_40ms_recursive.py", STAGE1,
            "--out", STAGE2, "--epochs", "100", "--seed", "31")
    if RUN_EVALUATION:
        run(HERE / "evaluate_dynamic_40ms.py")
    if RUN_EXPORT_TO_MPPI:
        run(HERE / "deploy_dynamic_40ms_to_mppi.py")


if __name__ == "__main__":
    main()
