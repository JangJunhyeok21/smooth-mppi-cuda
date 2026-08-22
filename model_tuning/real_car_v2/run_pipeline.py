#!/usr/bin/env python3
"""Run the complete identification pipeline followed by unified Step 6."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def run(script, *arguments):
    command=[sys.executable,str(HERE/script),*map(str,arguments)]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=2,choices=(1,2,3))
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--kf-threshold",type=float,default=.05)
    parser.add_argument("--skip-step1",action="store_true")
    parser.add_argument("--update-simulator",action="store_true")
    args=parser.parse_args()

    alternating=["--iterations",args.iterations,"--epochs",args.epochs,
                 "--kf-threshold",args.kf_threshold]
    if args.skip_step1:alternating.append("--skip-step1")
    run("run_alternating_refinement.py",*alternating)
    step6=[]
    if args.update_simulator:step6.append("--update-simulator")
    run("step_6_train_evaluate_deploy_residual.py",*step6)
    print("\nCompleted Step 1-6. See model_tuning/results/step_6_residual")


if __name__ == "__main__":
    main()
