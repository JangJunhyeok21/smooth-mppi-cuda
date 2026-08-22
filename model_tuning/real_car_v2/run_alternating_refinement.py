#!/usr/bin/env python3
"""Repeat numbered calibration/training stages with validation checkpointing."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from contract import ClassicModelParameters
from deploy_residual_model import replace_scalar

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
YAML = ROOT / "config/params.yaml"
RESULTS = ROOT / "model_tuning/results/alternating"
DATA20 = ROOT / "model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz"
DATA40 = ROOT / "model_tuning/data/dynamic_40ms_residual.npz"


def run(script, *arguments, env=None):
    command = [sys.executable, str(HERE/script), *map(str, arguments)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def rerun_kf():
    source=os.environ.get("KF_RECOMPUTE_NPZ_SOURCE")
    output=os.environ.get("DYNAMIC_SOURCE_DIRS")
    if source and output and os.pathsep not in output:
        run("recompute_kf_npz_dataset.py","--source",source,"--out",output)
    else:
        run("step_1_extract_data.py")


def apply_lateral_to_yaml(report_path):
    report=json.loads(report_path.read_text())
    if not report.get("deployment_gate_passed",False):
        print(f"Classic candidate rejected; keeping current YAML: {report_path}",flush=True)
        return False
    fitted = report["expanded_fitted"]
    text = YAML.read_text()
    for name in ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r"):
        text = replace_scalar(text, f"dynamic_mlp_{name}", fitted[name])
    text = replace_scalar(text, "dynamic_mlp_I_z", fitted["I_z"])
    YAML.write_text(text)
    return True


def relative_movement(previous, current):
    values=[]
    for name, old in previous.identified_dict().items():
        new=getattr(current,name)
        if name == "steer_bias":
            values.append(abs(new-old)/.05)
        else:
            values.append(abs(new-old)/(abs(old)+1e-8))
    return float(max(values))


def validation_score(report):
    value=report.get("validation", {})
    return (value.get("trajectory_m", {}).get("mean", 1e9)
            +.5*value.get("trajectory_m", {}).get("p95", 1e9)
            +.2*value.get("yaw_rate_rps", {}).get("p95", 1e9))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2, choices=(1,2,3))
    parser.add_argument("--kf-threshold", type=float, default=.05)
    parser.add_argument("--convergence", type=float, default=.01)
    parser.add_argument("--skip-step1", action="store_true",
                        help="reuse already extracted KF archive for the first round")
    parser.add_argument("--epochs", type=int, default=100)
    args=parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    iterations=[];previous_params=ClassicModelParameters.from_yaml(YAML)
    previous_model=None;previous_score=None

    for iteration in range(args.iterations):
        directory=RESULTS/f"iteration_{iteration:02d}";directory.mkdir(parents=True,exist_ok=True)
        if not args.skip_step1 or iteration:
            rerun_kf()

        # Round A, block A/B/C.  If a model exists it is passed as an immutable
        # binary into every classic rollout; only classic parameters are optimized.
        actuator_env=os.environ.copy()
        if previous_model:actuator_env["IDENTIFICATION_LOCAL_FRACTION"]="0.2"
        run("step_2_identify_longitudinal_actuator.py",env=actuator_env)
        # Identify the lateral plant first from the current rough steering
        # calibration. The next alternating round will revisit classic after
        # any accepted Step-4 steering update.
        classic_dir=directory/"classic"
        env=os.environ.copy();env["DYNAMIC_REGRESSION_OUT"]=str(classic_dir)
        if previous_model:
            env["FROZEN_MLP_BIN"]=str(previous_model/"dynamic_40ms_residual.bin")
            env["CLASSIC_RESIDUAL_PENALTY"]="0.0001"
            env["CLASSIC_LOCAL_FRACTION"]="0.2"
        run("step_3_identify_classic_model.py",env=env)
        apply_lateral_to_yaml(classic_dir/"params.json")
        run("step_4_identify_steering_actuator.py",env=actuator_env)
        current_params=ClassicModelParameters.from_yaml(YAML)
        movement=relative_movement(previous_params,current_params)

        # A/B evaluation happens before regeneration: NPZ contains the exact
        # Step-1 parameter snapshot and YAML now contains the tuned candidate.
        kf_data=os.environ.get("DYNAMIC_SOURCE_DIRS",
            str(ROOT/"model_tuning/data/ifac0817_0820_classic_kf"))
        if os.pathsep in kf_data:
            raise RuntimeError("Step 6 A/B evaluation currently requires one KF source directory")
        run("step_5_evaluate_velocity_observer.py","--data",kf_data,
            "--out",directory/"observer_evaluation","--no-plot")

        # Any accepted process-model change invalidates KF states and the
        # temporary Step-3 dataset. Regenerate only after the A/B report.
        kf_rerun=movement>args.kf_threshold
        if kf_rerun:
            rerun_kf()

        # Round B: new classic/KF hashes force pseudo-label regeneration.  The
        # previous MLP initializes one-step training, but classic is frozen.
        env=os.environ.copy();env.update({"PIPELINE_ITERATION":str(iteration),
            "DYNAMIC_CLASSIC_PARAMS":str(classic_dir/"params.json"),
            "DYNAMIC_RESIDUAL_DATA":str(DATA40)})
        one_step=directory/"one_step";arguments=[kf_data,"--out",one_step,"--epochs",args.epochs]
        if previous_model:arguments.extend(("--initialize-from",previous_model))
        run("residual_mlp_training.py",*arguments,env=env)
        recursive=directory/"recursive"
        run("residual_recursive_finetuning.py",one_step,"--out",recursive,
            "--epochs",args.epochs,env=env)
        evaluation=directory/"evaluation.json"
        run("residual_rollout_evaluation.py",recursive,"--out",evaluation,
            "--data",DATA40,"--classic-params",classic_dir/"params.json")
        classic_evaluation=directory/"classic_only_evaluation.json"
        run("residual_rollout_evaluation.py",recursive,"--out",classic_evaluation,
            "--data",DATA40,"--classic-params",classic_dir/"params.json","--disable-mlp")
        report=json.loads(evaluation.read_text());score=validation_score(report)
        dataset=np.load(DATA40);valid=dataset["valid"]
        residual=dataset["targets"][valid]
        residual_stats={name:{"rms":float(np.sqrt(np.mean(residual[:,i]**2))),
                              "p95":float(np.quantile(np.abs(residual[:,i]),.95))}
                        for i,name in enumerate(("delta_ax","delta_ay","delta_yaw_accel"))}
        relative_improvement=(None if previous_score is None else
                              (previous_score-score)/max(abs(previous_score),1e-9))
        record={"iteration":iteration,"classic_parameters":asdict(current_params),
                "classic_parameter_hash":current_params.digest(),
                "classic_parameter_movement":movement,"kf_rerun":kf_rerun,
                "validation_score":score,"relative_improvement":relative_improvement,
                "residual_pseudo_label_stats":residual_stats,
                "classic_params":str(classic_dir/"params.json"),"model":str(recursive),
                "evaluation":str(evaluation)}
        (directory/"iteration.json").write_text(json.dumps(record,indent=2)+"\n")
        iterations.append(record);previous_params=current_params
        previous_model=recursive;previous_score=score
        if relative_improvement is not None and 0 <= relative_improvement < args.convergence:
            break

    best=min(iterations,key=lambda item:item["validation_score"])
    summary={"selection":"minimum held-out validation rollout score, not last iteration",
             "best_iteration":best["iteration"],"best":best,"iterations":iterations}
    (RESULTS/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
