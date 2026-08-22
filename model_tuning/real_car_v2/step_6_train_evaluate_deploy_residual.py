#!/usr/bin/env python3
"""Step 6: train, recursively fine-tune, evaluate, and deploy residual MLP."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STEP1_DATA=ROOT/"model_tuning/data/ifac0810_0819_autonomous_physics_clean"
CLASSIC_PARAMS=ROOT/"model_tuning/results/dynamic_40ms_regression/params.json"
OUTPUT=ROOT/"model_tuning/results/step_6_residual"
ONE_STEP_EPOCHS=300
RECURSIVE_EPOCHS=100
SEED=31


def run(script,*arguments,env=None):
    command=[sys.executable,str(HERE/script),*map(str,arguments)]
    print("\n+"," ".join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=STEP1_DATA,
                        help="Step-1 bag NPZ directory")
    parser.add_argument("--classic-params",type=Path,default=CLASSIC_PARAMS)
    parser.add_argument("--out",type=Path,default=OUTPUT)
    parser.add_argument("--one-step-epochs",type=int,default=ONE_STEP_EPOCHS)
    parser.add_argument("--recursive-epochs",type=int,default=RECURSIVE_EPOCHS)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--update-simulator",action="store_true")
    parser.add_argument("--no-deploy",action="store_true",
                        help="train/evaluate only; do not update runtime config")
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["DYNAMIC_CLASSIC_PARAMS"]=str(args.classic_params.resolve())
    one_step=args.out/"one_step";recursive=args.out/"recursive"
    run("residual_mlp_training.py",args.data,"--out",one_step,"--epochs",
        args.one_step_epochs,"--seed",args.seed,env=env)
    run("residual_recursive_finetuning.py",one_step,"--out",recursive,
        "--epochs",args.recursive_epochs,"--seed",args.seed,env=env)
    run("residual_rollout_evaluation.py",recursive,"--out",args.out/"residual.json",
        "--classic-params",args.classic_params,env=env)
    run("residual_rollout_evaluation.py",recursive,"--out",args.out/"classic_only.json",
        "--classic-params",args.classic_params,"--disable-mlp",env=env)
    if not args.no_deploy:
        deploy=[recursive,"--regression",args.classic_params,
                "--alternating-summary",args.out/"no_alternating_summary.json"]
        if args.update_simulator:deploy.append("--update-simulator")
        run("deploy_residual_model.py",*deploy,env=env)
    print(f"\nStep 6 complete: {args.out}")


if __name__=="__main__":main()
