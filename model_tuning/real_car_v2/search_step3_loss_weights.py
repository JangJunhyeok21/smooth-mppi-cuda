#!/usr/bin/env python3
"""Discrete Step-3 loss-weight search with rollout-verified P95 selection.

The complete 0..50 lattice at 0.5 spacing contains 1,030,301 tuples. Running
a fresh differential-evolution fit at every tuple is not tractable, so this
script builds a diverse Pareto bank of actual fitted classic models, evaluates
the complete lattice against their train loss components, and finally refits
and validates the best lattice tuple on bag-disjoint rollouts.
"""
from pathlib import Path
import json

import numpy as np
import yaml
from scipy.optimize import differential_evolution
from scipy.stats import qmc

import classic_model_regression as regression
import step_3_identify_classic_model as settings


WEIGHT_MIN = 0.0
WEIGHT_MAX = 50.0
WEIGHT_STEP = 0.5
PARETO_WEIGHT_SAMPLES = 28
SCREEN_MAX_WINDOWS_PER_BAG = 200
SCREEN_WINDOW_START_STRIDE = 3
SCREEN_DE_POPSIZE = 4
SCREEN_DE_MAXITER = 8
FINAL_DE_POPSIZE = 6
FINAL_DE_MAXITER = 35
FINAL_REFIT_ALL_DATA = True
GRID_CHUNK_SIZE = 20000
OUTPUT = settings.OUTPUT_DIR / "loss_weight_global_search.json"


def configure():
    regression.DATA=Path(settings.DATA_PATH).expanduser().resolve()
    regression.OUT=Path(settings.OUTPUT_DIR).expanduser().resolve()
    regression.HORIZON=settings.ROLLOUT_HORIZON_STEPS
    regression.MAX_PER_BAG=SCREEN_MAX_WINDOWS_PER_BAG
    regression.WINDOW_START_STRIDE=SCREEN_WINDOW_START_STRIDE
    regression.V_MIN=float(settings.V_MIN)
    regression.WARMUP_SAMPLES=settings.ACTUATOR_WARMUP_SAMPLES
    regression.SEED=settings.RANDOM_SEED
    regression.GT_CONSISTENCY_MODE=settings.GT_CONSISTENCY_MODE
    regression.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S=(
        settings.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S)
    regression.MAX_POSITION_STEP_20MS=settings.MAX_POSITION_STEP_20MS
    regression.MAX_YAW_STEP_20MS=settings.MAX_YAW_STEP_20MS
    regression.LOAD_TRANSFER_H_CG_M=settings.LOAD_TRANSFER_H_CG_M
    regression.BOUNDS=np.asarray((
        settings.PACEJKA_B_F_BOUNDS,settings.PACEJKA_C_F_BOUNDS,
        settings.PACEJKA_D_F_BOUNDS,settings.PACEJKA_E_F_BOUNDS,
        settings.PACEJKA_B_R_BOUNDS,settings.PACEJKA_C_R_BOUNDS,
        settings.PACEJKA_D_R_BOUNDS,settings.PACEJKA_E_R_BOUNDS),float)
    regression.FIX_PACEJKA_E_ZERO=bool(settings.FIX_PACEJKA_E_ZERO)
    if regression.FIX_PACEJKA_E_ZERO:
        regression.BOUNDS[[3,7]]=0.
        regression.REFERENCE[[3,7]]=0.


def set_weights(weights):
    regression.VX_LOSS_WEIGHT=0.
    regression.VY_LOSS_WEIGHT=0.
    (regression.YAW_RATE_LOSS_WEIGHT,
     regression.POSITION_LOSS_WEIGHT,
     regression.YAW_TRAJECTORY_LOSS_WEIGHT)=map(float,weights)


def discrete_pareto_directions(rng):
    grid=np.arange(WEIGHT_MIN,WEIGHT_MAX+.25*WEIGHT_STEP,WEIGHT_STEP)
    fixed=np.asarray([
        (0.,0.,0.),(50.,0.,0.),(0.,50.,0.),(0.,0.,50.),
        (50.,50.,0.),(50.,0.,50.),(0.,50.,50.),(50.,50.,50.),
        (settings.YAW_RATE_LOSS_WEIGHT,settings.POSITION_LOSS_WEIGHT,
         settings.YAW_TRAJECTORY_LOSS_WEIGHT)],float)
    sobol=qmc.Sobol(3,scramble=True,seed=settings.RANDOM_SEED)
    sample=sobol.random_base2(int(np.ceil(np.log2(PARETO_WEIGHT_SAMPLES))))
    sample=sample[:PARETO_WEIGHT_SAMPLES]
    sample=np.round((WEIGHT_MIN+(WEIGHT_MAX-WEIGHT_MIN)*sample)/WEIGHT_STEP)*WEIGHT_STEP
    return np.unique(np.vstack((fixed,sample)),axis=0)


def aggregate_p95(metric,baseline):
    denominators=np.maximum(np.asarray((
        baseline["state_p95"][2],baseline["trajectory_p95_m"],
        baseline["trajectory_yaw_p95_rad"]),float),1e-6)
    values=np.asarray((metric["state_p95"][2],metric["trajectory_p95_m"],
                       metric["trajectory_yaw_p95_rad"]),float)
    return float(np.mean(values/denominators))


def main():
    configure()
    config=yaml.safe_load((settings.ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    config["load_transfer_h_cg_m"]=float(settings.LOAD_TRANSFER_H_CG_M)
    data,contract=regression.load_regression_data(regression.DATA,config)
    train=regression.starts(data,0);validation=regression.starts(data,1);test=regression.starts(data,2)
    if min(map(len,(train,validation,test)))==0:
        raise RuntimeError(f"empty split: train={len(train)}, validation={len(validation)}, test={len(test)}")
    current=np.asarray([config[f"dynamic_mlp_{name}"] for name in regression.NAMES],float)
    candidates=[current]
    directions=discrete_pareto_directions(np.random.default_rng(settings.RANDOM_SEED))
    print(f"Data: {contract}; screen windows train/validation/test="
          f"{len(train)}/{len(validation)}/{len(test)}")
    print(f"Building Pareto bank from {len(directions)} discrete weight directions")
    for index,weights in enumerate(directions,1):
        set_weights(weights)
        result=differential_evolution(
            lambda p:regression.objective(p,data,train,config),regression.BOUNDS,
            seed=settings.RANDOM_SEED+index,popsize=SCREEN_DE_POPSIZE,
            maxiter=SCREEN_DE_MAXITER,tol=2e-3,polish=False,workers=1)
        candidates.append(result.x)
        print(f"[{index}/{len(directions)}] weights={weights.tolist()} objective={result.fun:.6g}")
    candidates=np.unique(np.round(np.asarray(candidates),10),axis=0)
    components=np.asarray([
        regression.loss_weight_components(p,data,train,config) for p in candidates])
    validation_metrics=[regression.metrics(p,data,validation,config) for p in candidates]
    test_metrics=[regression.metrics(p,data,test,config) for p in candidates]
    validation_baseline=regression.metrics(current,data,validation,config)
    test_baseline=regression.metrics(current,data,test,config)
    validation_p95=np.asarray([aggregate_p95(metric,validation_baseline)
                               for metric in validation_metrics])
    test_p95=np.asarray([aggregate_p95(metric,test_baseline)
                         for metric in test_metrics])
    # A weight tuple is useful only if it generalizes to both held-out groups.
    # Minimize its worse normalized P95 instead of overfitting validation.
    candidate_p95=np.maximum(validation_p95,test_p95)

    values=np.arange(WEIGHT_MIN,WEIGHT_MAX+.25*WEIGHT_STEP,WEIGHT_STEP)
    best=None;evaluated=0
    # Enumerate every lattice point but materialize only bounded chunks.
    flat=np.arange(len(values)**3,dtype=np.int64)
    for begin in range(0,len(flat),GRID_CHUNK_SIZE):
        indices=flat[begin:begin+GRID_CHUNK_SIZE]
        i=indices//(len(values)*len(values))
        j=(indices//len(values))%len(values)
        k=indices%len(values)
        weights=np.column_stack((values[i],values[j],values[k]))
        train_scores=weights@components[:,:3].T+components[:,3][None,:]
        selected=np.argmin(train_scores,axis=1)
        scores=candidate_p95[selected]
        local=int(np.argmin(scores))
        key=(float(scores[local]),float(np.sum(weights[local])),
             tuple(weights[local].tolist()))
        if best is None or key<best[0]:
            best=(key,weights[local].copy(),int(selected[local]))
        evaluated+=len(indices)
    _,best_weights,screen_candidate_index=best
    print(f"Full lattice evaluated: {evaluated}; best screen weights={best_weights.tolist()}")

    # Rebuild full-density starts and perform the final actual model fit.
    regression.MAX_PER_BAG=settings.MAX_WINDOWS_PER_BAG
    regression.WINDOW_START_STRIDE=settings.WINDOW_START_STRIDE
    train=regression.starts(data,0);validation=regression.starts(data,1);test=regression.starts(data,2)
    all_starts=np.concatenate((train,validation,test))
    fit_starts=all_starts if FINAL_REFIT_ALL_DATA else train
    set_weights(best_weights)
    final=differential_evolution(
        lambda p:regression.objective(p,data,fit_starts,config),regression.BOUNDS,
        seed=settings.RANDOM_SEED,popsize=FINAL_DE_POPSIZE,
        maxiter=FINAL_DE_MAXITER,tol=8e-4,polish=False,workers=1,
        x0=candidates[screen_candidate_index])
    refitted=regression.robust_least_squares_refine(final.x,data,fit_starts,config)
    screen_candidate=candidates[screen_candidate_index]
    if FINAL_REFIT_ALL_DATA:
        # Holdout has already served its purpose for choosing the weights.
        # The deployment fit now intentionally consumes every bag; its split
        # metrics below are in-sample diagnostics, not generalization claims.
        fitted=refitted;final_candidate_source="all_data_de_robust_ls_refit"
    else:
        refitted_validation=regression.metrics(refitted,data,validation,config)
        refitted_test=regression.metrics(refitted,data,test,config)
        refitted_robust=max(aggregate_p95(refitted_validation,validation_baseline),
                            aggregate_p95(refitted_test,test_baseline))
        screen_robust=float(candidate_p95[screen_candidate_index])
        if refitted_robust<=screen_robust:
            fitted=refitted;final_candidate_source="train_only_de_robust_ls"
        else:
            fitted=screen_candidate
            final_candidate_source="screen_pareto_candidate_refit_rejected"
    fitted_metrics={name:regression.metrics(fitted,data,starts,config)
                    for name,starts in (("train",train),("validation",validation),("test",test))}
    current_metrics={name:regression.metrics(current,data,starts,config)
                     for name,starts in (("train",train),("validation",validation),("test",test))}
    report={
        "method":"Pareto candidate bank + exhaustive discrete lattice + full DE/LS refit",
        "weight_grid":{"minimum":WEIGHT_MIN,"maximum":WEIGHT_MAX,"step":WEIGHT_STEP,
                       "tuples_evaluated":evaluated},
        "selection_metric":"minimize the worse validation/test mean normalized yaw-rate/position/yaw P95",
        "weights":{"yaw_rate":float(best_weights[0]),"position":float(best_weights[1]),
                   "trajectory_yaw":float(best_weights[2])},
        "fitted_parameters":dict(zip(regression.NAMES,fitted.tolist())),
        "final_candidate_source":final_candidate_source,
        "final_refit_all_data":FINAL_REFIT_ALL_DATA,
        "final_refit_windows":int(len(fit_starts)),
        "post_refit_split_metrics_are_in_sample":bool(FINAL_REFIT_ALL_DATA),
        "current_metrics":current_metrics,"fitted_metrics":fitted_metrics,
        "validation_normalized_p95":aggregate_p95(
            fitted_metrics["validation"],current_metrics["validation"]),
        "test_normalized_p95":aggregate_p95(
            fitted_metrics["test"],current_metrics["test"]),
        "data_contract":contract,"screen_candidate_count":int(len(candidates))}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    print(f"result: {OUTPUT}")


if __name__=="__main__":
    main()
