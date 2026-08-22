#!/usr/bin/env python3
"""Step 6: train, recursively fine-tune, evaluate, and deploy residual MLP."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

import numpy as np


ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
STEP1_DATA=ROOT/"model_tuning/data/0821"
CLASSIC_PARAMS=ROOT/"model_tuning/results/dynamic_40ms_regression/params.json"
OUTPUT=ROOT/"model_tuning/results/step_6_residual"
ONE_STEP_EPOCHS=300
RECURSIVE_EPOCHS=100
SEED=31
# ---------------------------------------------------------------------------
# User-configurable Step 6 settings (F5 실행 시 사용)
# ---------------------------------------------------------------------------
USE_PLOT=True
EVALUATE_ONLY=False  # F5: one-step -> recursive -> evaluation 순서로 실행
EVALUATION_MODEL_PATH=OUTPUT/"recursive"
DEPLOY_AFTER_TRAINING=True
ROLLOUT_DT_S=0.04
HORIZON_STEPS=60  # Step 1 currently stores 1.2 s = HORIZON_STEPS * 40 ms
INSPECT_BAG_ID=None  # None: INSPECT_RANDOM_SEED로 가능한 bag 하나를 선택
INSPECT_RANDOM_SEED=31


def run(script,*arguments,env=None):
    command=[sys.executable,str(HERE/script),*map(str,arguments)]
    print("\n+"," ".join(command),flush=True)
    subprocess.run(command,cwd=ROOT,env=env,check=True)


def print_evaluation_change(residual_path,classic_path,horizon_steps):
    residual=json.loads(Path(residual_path).read_text())
    classic=json.loads(Path(classic_path).read_text())
    horizon_s=horizon_steps*ROLLOUT_DT_S
    print(f"\nStep 6 classic-only -> residual {horizon_s:g} s "
          "open-loop RMSE:")
    for split in sorted(set(residual)&set(classic)):
        if split=="evaluation_contract":continue
        print(f"  [{split}]")
        for metric in ("trajectory_m","yaw_rad","vx_mps","vy_mps","yaw_rate_rps"):
            before=float(classic[split]["final_horizon"][metric]["rmse"])
            after=float(residual[split]["final_horizon"][metric]["rmse"])
            reduction=100.0*(before-after)/max(abs(before),1e-12)
            print(f"    {metric}: {before:.6g} -> {after:.6g} ({reduction:+.2f}%)")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",type=Path,default=STEP1_DATA,
                        help="Step-1 bag NPZ directory")
    parser.add_argument("--classic-params",type=Path,default=CLASSIC_PARAMS)
    parser.add_argument("--out",type=Path,default=OUTPUT)
    parser.add_argument("--one-step-epochs",type=int,default=ONE_STEP_EPOCHS)
    parser.add_argument("--recursive-epochs",type=int,default=RECURSIVE_EPOCHS)
    parser.add_argument("--seed",type=int,default=SEED)
    parser.add_argument("--horizon-steps",type=int,default=HORIZON_STEPS,
                        help="number of 40 ms recursive rollout steps")
    parser.add_argument("--evaluate-only",action="store_true",default=EVALUATE_ONLY,
                        help="load --model and only regenerate evaluation/plots")
    parser.add_argument("--model",type=Path,default=EVALUATION_MODEL_PATH,
                        help="existing recursive model directory for evaluation-only mode")
    parser.add_argument("--no-plot",action="store_true",default=not USE_PLOT)
    parser.add_argument("--update-simulator",action="store_true")
    parser.add_argument("--no-deploy",action="store_true",
                        help="train/evaluate only; do not update runtime config")
    args=parser.parse_args()
    if args.horizon_steps < 1:
        parser.error("--horizon-steps must be at least 1")
    args.out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env["DYNAMIC_CLASSIC_PARAMS"]=str(args.classic_params.resolve())
    one_step=args.out/"one_step"
    if args.evaluate_only:
        recursive=args.model.expanduser().resolve()
        binary=recursive/"dynamic_40ms_residual.bin"
        if not binary.is_file():
            raise FileNotFoundError(f"evaluation model binary does not exist: {binary}")
        print(f"Step 6 evaluation-only mode: {recursive}")
    else:
        recursive=args.out/"recursive"
        run("residual_mlp_training.py",args.data,"--out",one_step,"--epochs",
            args.one_step_epochs,"--seed",args.seed,
            "--horizon-steps",args.horizon_steps,env=env)
        run("residual_recursive_finetuning.py",one_step,"--out",recursive,
            "--epochs",args.recursive_epochs,"--seed",args.seed,
            "--horizon-steps",args.horizon_steps,"--data",args.data,env=env)
    residual_report=args.out/"residual.json"
    classic_report=args.out/"classic_only.json"
    run("residual_rollout_evaluation.py",recursive,"--out",residual_report,
        "--classic-params",args.classic_params,"--data",args.data,
        "--horizon-steps",args.horizon_steps,env=env)
    run("residual_rollout_evaluation.py",recursive,"--out",classic_report,
        "--classic-params",args.classic_params,"--data",args.data,"--disable-mlp",
        "--horizon-steps",args.horizon_steps,env=env)
    print_evaluation_change(residual_report,classic_report,args.horizon_steps)
    if not args.no_plot:
        from callback_training_data import load_callback_archives
        inspection_data=load_callback_archives(
            args.data,model_dt=ROLLOUT_DT_S,horizon=args.horizon_steps)
        bag_names=np.unique(inspection_data["bag_name"])
        eligible=list(range(len(bag_names)))
        if not eligible:
            raise RuntimeError("configured horizon을 만족하는 inspection bag이 없습니다")
        if INSPECT_BAG_ID is None:
            inspect_bag=int(np.random.default_rng(
                INSPECT_RANDOM_SEED).choice(eligible))
        else:
            inspect_bag=int(INSPECT_BAG_ID)
            if inspect_bag not in eligible:
                raise ValueError(f"INSPECT_BAG_ID={inspect_bag} is not eligible; "
                                 f"choose one of {eligible}")
        print(f"\nStep 6 interactive inspection bag_id={inspect_bag}")
        bag_residual=args.out/"inspection_bag_residual.json"
        bag_classic=args.out/"inspection_bag_classic.json"
        common=("--classic-params",args.classic_params,"--data",args.data,
                "--horizon-steps",args.horizon_steps,"--bag-id",inspect_bag)
        run("residual_rollout_evaluation.py",recursive,"--out",bag_residual,
            *common,env=env)
        run("residual_rollout_evaluation.py",recursive,"--out",bag_classic,
            *common,"--disable-mlp",env=env)
        # Keep plotting in this debugpy/F5 process. A child Python process is
        # detached from VS Code's Matplotlib integration and may fall back to
        # the non-interactive Agg backend even when TkAgg is requested.
        from visualize_best_p95_worst_rollouts import main as visualize
        visualize([
            "--result-dir",str(args.out),
            "--new-file",bag_residual.with_suffix(".npz").name,
            "--baseline-file",bag_classic.with_suffix(".npz").name,
            "--data",str(args.data),
            "--bag-id",str(inspect_bag),
            "--new-label","recursive residual MLP",
            "--baseline-label","classic only",
        ])
    if DEPLOY_AFTER_TRAINING and not args.evaluate_only and not args.no_deploy:
        deploy=[recursive,"--regression",args.classic_params,
                "--alternating-summary",args.out/"no_alternating_summary.json"]
        if args.update_simulator:deploy.append("--update-simulator")
        run("deploy_residual_model.py",*deploy,env=env)
    mode="evaluation only" if args.evaluate_only else "training/evaluation"
    print(f"\nStep 6 {mode} complete: {args.out}")


if __name__=="__main__":main()
