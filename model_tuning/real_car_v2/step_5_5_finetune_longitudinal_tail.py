#!/usr/bin/env python3
"""Fine-tune Step-2 actuator parameters on high-speed longitudinal tails.

All Step-2 splits are pooled first.  Complete source bags containing high-speed
rollouts are then separated into development and holdout sets.  Only the
development hard tail is oversampled; the holdout bags are never optimized.
"""
from copy import copy
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import differential_evolution, minimize

import step_2_identify_longitudinal_actuator as step2
import visualize_and_regress_longitudinal_actuator as regression


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "model_tuning/results/step_5_5_longitudinal_tail"
HIGH_SPEED_MIN_MPS = 4.5
HARD_TAIL_QUANTILE = 0.85
# A longitudinal position error is the time integral of predicted-vx minus
# GT-vx. Large instantaneous speed errors are also included independently.
LONGITUDINAL_POSE_TAIL_THRESHOLD_M = 0.5
SPEED_ERROR_P95_TAIL_THRESHOLD_MPS = 0.5
HIGH_SPEED_HOLDOUT_BAG_FRACTION = 0.25
BACKGROUND_WINDOWS = 1200
HARD_WINDOW_REPEAT = 8
OPTIMIZER_POPULATION_SIZE = 30
OPTIMIZER_MAX_ITERATIONS = 55
OPTIMIZER_LOCAL_MAX_ITERATIONS = 350
RANDOM_SEED = 31
UPDATE_PARAMS_YAML = True
# False means UPDATE_PARAMS_YAML=True writes the candidate even when the
# statistical deployment gate is advisory/rejected.
REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE = False
USE_PLOT = True
MIN_HOLDOUT_HIGH_SPEED_WINDOWS = 30
EVALUATE_ONLY = False
EVALUATION_REPORT_PATH = OUTPUT_DIR / "report.json"


def configure():
    regression.SOURCE_DIRS = tuple(Path(p).expanduser().resolve()
                                   for p in step2.SOURCE_DIRS)
    regression.OUTPUT_DIR = OUTPUT_DIR
    regression.ROLLOUT_STEPS = max(1, int(round(
        step2.HORIZON_STEPS * step2.MODEL_DT_S / regression.SOURCE_DT_S)))
    regression.MODEL_DT_S = step2.MODEL_DT_S
    regression.HORIZON_STEPS = step2.HORIZON_STEPS
    regression.WARMUP_S = step2.WARMUP_DURATION_S
    regression.START_STRIDE = step2.START_STRIDE_SAMPLES
    regression.MAX_ROLLOUTS_PER_SESSION = step2.MAX_ROLLOUTS_PER_SESSION
    regression.REQUIRE_STRAIGHT_WINDOWS = step2.REQUIRE_STRAIGHT_WINDOWS
    regression.BOUNDS = (step2.SPEED_SERVO_KP_MIN, step2.SPEED_SERVO_KP_MAX), (
        step2.ACCEL_TIME_CONSTANT_MIN, step2.ACCEL_TIME_CONSTANT_MAX), (
        step2.BRAKE_TIME_CONSTANT_MIN, step2.BRAKE_TIME_CONSTANT_MAX)


def window_predictions(session, starts, params, cfg):
    starts = np.asarray(starts, int)
    offsets = np.arange(regression.ROLLOUT_STEPS + 1)
    indices = starts[:, None] + offsets[None, :]
    gt = session["vx"][indices]
    cmd = session["cmd"][indices]
    pred = np.empty_like(gt)
    pred[:, 0] = gt[:, 0]
    speed_reference = gt[:, 0].copy()
    kp, tau_accel, tau_brake = params
    warmup = max(1, int(round(regression.WARMUP_S / session["dt"])))
    for offset in range(-warmup, 0):
        past = session["cmd"][starts + offset]
        tau = np.where(past >= speed_reference, tau_accel, tau_brake)
        speed_reference += np.clip(
            (past - speed_reference) / tau,
            -cfg["actuator_max_speed_reference_rate"],
            cfg["actuator_max_speed_reference_rate"]) * session["dt"]
    for k in range(regression.ROLLOUT_STEPS):
        tau = np.where(cmd[:, k] >= speed_reference, tau_accel, tau_brake)
        speed_reference += np.clip(
            (cmd[:, k] - speed_reference) / tau,
            -cfg["actuator_max_speed_reference_rate"],
            cfg["actuator_max_speed_reference_rate"]) * session["dt"]
        acceleration = np.clip(kp * (speed_reference - pred[:, k]),
                               cfg["min_accel"], cfg["max_accel"])
        pred[:, k + 1] = np.maximum(0.0, pred[:, k] + acceleration * session["dt"])
    return gt, cmd, pred


def window_records(sessions, params, cfg):
    records = []
    for session_index, session in enumerate(sessions):
        starts = session["starts"]
        if not len(starts):
            continue
        gt, _, pred = window_predictions(session, starts, params, cfg)
        error = pred - gt
        for row, start in enumerate(starts):
            records.append({
                "session_index": session_index, "start": int(start),
                "source": session["source"],
                "mean_speed": float(np.mean(gt[row])),
                "rmse": float(np.sqrt(np.mean(error[row] ** 2))),
                "p95": float(np.percentile(np.abs(error[row]), 95)),
                "longitudinal_pose_error": float(abs(np.sum(error[row]) *
                                                       session["dt"])),
            })
    return records


def subset_sessions(sessions, selected, split="train"):
    grouped = {}
    for record in selected:
        grouped.setdefault(record["session_index"], []).append(record["start"])
    result = []
    for index, starts in grouped.items():
        item = copy(sessions[index])
        item["starts"] = np.asarray(starts, int)
        item["split"] = split
        result.append(item)
    return result


def summaries(records):
    if not records:
        return {"windows": 0, "rmse_mean_mps": float("nan"),
                "window_p95_mps": float("nan"),
                "longitudinal_pose_p95_m": float("nan")}
    return {"windows": len(records),
            "rmse_mean_mps": float(np.mean([r["rmse"] for r in records])),
            "window_p95_mps": float(np.percentile([r["p95"] for r in records], 95)),
            "longitudinal_pose_p95_m": float(np.percentile(
                [r["longitudinal_pose_error"] for r in records], 95))}


def reduction_percent(previous, candidate):
    return 100.0 * (previous - candidate) / max(abs(previous), 1.0e-12)


def print_finetuning_summary(current, candidate, current_metrics,
                             candidate_metrics, gate, yaml_updated):
    names = ("speed_servo_kp", "speed_reference_accel_time_constant",
             "speed_reference_brake_time_constant")
    print("\nStep 5_5 longitudinal-tail fine-tuning parameter changes:")
    for name, previous, tuned in zip(names, current, candidate):
        print(f"  {name}: {previous:.9g} -> {tuned:.9g}")
    print("\nHeld-out longitudinal performance "
          "(positive reduction = improvement):")
    for scope in ("all", "high_speed"):
        previous = current_metrics[scope]
        tuned = candidate_metrics[scope]
        print(f"  [{scope.replace('_', '-')}; windows={previous['windows']}]")
        for key in ("rmse_mean_mps", "window_p95_mps",
                    "longitudinal_pose_p95_m"):
            reduction = reduction_percent(previous[key], tuned[key])
            unit = "m" if key.endswith("_m") else "m/s"
            print(f"    {key}: {previous[key]:.6g} -> {tuned[key]:.6g} {unit} "
                  f"(reduction {reduction:+.2f}%)")
    print(f"  deployment gate: {'PASS' if gate else 'REJECT'}")
    print(f"  params.yaml updated: {yaml_updated}")


def evaluate(sessions, params, cfg, sources):
    records = [r for r in window_records(sessions, params, cfg)
               if r["source"] in sources]
    high = [r for r in records if r["mean_speed"] >= HIGH_SPEED_MIN_MPS]
    return {"all": summaries(records), "high_speed": summaries(high)}, records


def load_evaluation_candidate(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"evaluation report not found: {path}")
    values = json.loads(path.read_text()).get("candidate")
    if not isinstance(values, list) or len(values) != 3:
        raise KeyError(f"{path}: expected three values in candidate")
    return np.asarray(values, float)


def main():
    configure()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())[
        "/**"]["ros__parameters"]
    sessions = regression.load_sessions()
    for session in sessions:
        session["starts"] = regression.choose_starts(session)
        session["split"] = "train"
    current = np.asarray([
        cfg["speed_servo_kp"], cfg["speed_reference_accel_time_constant"],
        cfg["speed_reference_brake_time_constant"]], float)
    initial_records = window_records(sessions, current, cfg)
    high_sources = sorted({r["source"] for r in initial_records
                           if r["mean_speed"] >= HIGH_SPEED_MIN_MPS})
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(high_sources)
    if not high_sources:
        raise RuntimeError("no high-speed longitudinal rollout passed Step-2 filters")
    temporal_holdout_keys = set()
    if len(high_sources) >= 2:
        holdout_count = max(1, int(round(
            HIGH_SPEED_HOLDOUT_BAG_FRACTION * len(high_sources))))
        holdout_sources = set(high_sources[:holdout_count])
        is_holdout = lambda r: r["source"] in holdout_sources
        split_method = "source-bag-disjoint high-speed holdout"
    else:
        # A single source bag contains every high-speed straight. Keep its
        # chronologically last 25% of high-speed starts completely out of fit.
        holdout_sources = set()
        only = high_sources[0]
        high_records = sorted(
            (r for r in initial_records
             if r["source"] == only and r["mean_speed"] >= HIGH_SPEED_MIN_MPS),
            key=lambda r: (r["session_index"], r["start"]))
        cut = max(1, int(round(HIGH_SPEED_HOLDOUT_BAG_FRACTION * len(high_records))))
        temporal_holdout_keys = {(r["session_index"], r["start"])
                                 for r in high_records[-cut:]}
        is_holdout = lambda r: (r["session_index"], r["start"]) in temporal_holdout_keys
        split_method = "single-high-speed-bag chronological tail holdout"
    development = [r for r in initial_records if not is_holdout(r)]
    development_high = [r for r in development
                        if r["mean_speed"] >= HIGH_SPEED_MIN_MPS]
    threshold = float(np.quantile([r["p95"] for r in development_high],
                                  HARD_TAIL_QUANTILE))
    hard = [r for r in development
            if r["longitudinal_pose_error"] >= LONGITUDINAL_POSE_TAIL_THRESHOLD_M
            or r["p95"] >= SPEED_ERROR_P95_TAIL_THRESHOLD_MPS
            or (r["mean_speed"] >= HIGH_SPEED_MIN_MPS and r["p95"] >= threshold)]
    background_count = min(BACKGROUND_WINDOWS, len(development))
    background = [development[i] for i in np.linspace(
        0, len(development) - 1, background_count).astype(int)]
    fit_records = background + hard * HARD_WINDOW_REPEAT
    fit_sessions = subset_sessions(sessions, fit_records)

    bounds = regression.BOUNDS
    bounds_array = np.asarray(bounds, float)
    population = rng.uniform(bounds_array[:, 0], bounds_array[:, 1],
                             size=(OPTIMIZER_POPULATION_SIZE, 3))
    population[0] = np.clip(current, bounds_array[:, 0], bounds_array[:, 1])
    if EVALUATE_ONLY:
        candidate = load_evaluation_candidate(EVALUATION_REPORT_PATH)
        print(f"Step 5_5 EVALUATE_ONLY: loaded "
              f"{Path(EVALUATION_REPORT_PATH).resolve()}")
    else:
        result = differential_evolution(
            lambda p: regression.robust_objective(p, fit_sessions, cfg), bounds,
            seed=RANDOM_SEED, init=population, maxiter=OPTIMIZER_MAX_ITERATIONS,
            tol=2e-5, polish=False, workers=1, updating="immediate")
        start = min((current, result.x),
                    key=lambda p: regression.robust_objective(p, fit_sessions, cfg))
        local = minimize(lambda p: regression.robust_objective(p, fit_sessions, cfg),
                         start, method="Powell", bounds=bounds,
                         options={"maxiter": OPTIMIZER_LOCAL_MAX_ITERATIONS,
                                  "xtol": 1e-6, "ftol": 1e-9})
        candidate = np.asarray(min(
            (current, result.x, np.clip(local.x, bounds_array[:, 0], bounds_array[:, 1])),
            key=lambda p: regression.robust_objective(p, fit_sessions, cfg)), float)

    if temporal_holdout_keys:
        holdout_base = [r for r in initial_records if is_holdout(r)]
        holdout_sessions = subset_sessions(sessions, holdout_base, split="test")
        current_records = window_records(holdout_sessions, current, cfg)
        candidate_records = window_records(holdout_sessions, candidate, cfg)
        current_metrics = {"all": summaries(current_records),
                           "high_speed": summaries([r for r in current_records
                            if r["mean_speed"] >= HIGH_SPEED_MIN_MPS])}
        candidate_metrics = {"all": summaries(candidate_records),
                             "high_speed": summaries([r for r in candidate_records
                              if r["mean_speed"] >= HIGH_SPEED_MIN_MPS])}
    else:
        holdout_base = [r for r in initial_records
                        if r["source"] in holdout_sources]
        holdout_sessions = subset_sessions(sessions, holdout_base, split="test")
        current_metrics, current_records = evaluate(
            sessions, current, cfg, holdout_sources)
        candidate_metrics, candidate_records = evaluate(
            sessions, candidate, cfg, holdout_sources)
    high_current = current_metrics["high_speed"]["window_p95_mps"]
    high_candidate = candidate_metrics["high_speed"]["window_p95_mps"]
    all_current = current_metrics["all"]["window_p95_mps"]
    all_candidate = candidate_metrics["all"]["window_p95_mps"]
    margin = .01 * (bounds_array[:, 1] - bounds_array[:, 0])
    boundary = ((candidate - bounds_array[:, 0] <= margin) |
                (bounds_array[:, 1] - candidate <= margin))
    enough_holdout = current_metrics["high_speed"]["windows"] >= MIN_HOLDOUT_HIGH_SPEED_WINDOWS
    gate = bool(enough_holdout and not boundary.any() and high_candidate < high_current and
                all_candidate <= 1.01 * all_current)
    report = {
        "data_contract": f"all Step-2 splits pooled; {split_method}",
        "evaluate_only": EVALUATE_ONLY,
        "high_speed_min_mps": HIGH_SPEED_MIN_MPS,
        "hard_tail_quantile": HARD_TAIL_QUANTILE,
        "hard_tail_threshold_p95_mps": threshold,
        "longitudinal_pose_tail_threshold_m": LONGITUDINAL_POSE_TAIL_THRESHOLD_M,
        "speed_error_p95_tail_threshold_mps": SPEED_ERROR_P95_TAIL_THRESHOLD_MPS,
        "high_speed_source_bags": sorted(high_sources),
        "holdout_source_bags": sorted(holdout_sources),
        "temporal_holdout_windows": len(temporal_holdout_keys),
        "development_windows": len(development), "hard_windows": len(hard),
        "optimizer_windows_with_repetition": len(fit_records),
        "current": current.tolist(), "candidate": candidate.tolist(),
        "current_holdout": current_metrics,
        "candidate_holdout": candidate_metrics,
        "boundary_solution": boundary.tolist(),
        "minimum_holdout_high_speed_windows": MIN_HOLDOUT_HIGH_SPEED_WINDOWS,
        "enough_holdout_for_deployment": enough_holdout,
        "deployment_gate_passed": gate,
    }

    bins = np.arange(0.0, max(r["mean_speed"] for r in current_records) + 1.0, .5)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, records, color in (("current", current_records, "tab:blue"),
                                  ("fine-tuned", candidate_records, "tab:orange")):
        x, y = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            values = [r["p95"] for r in records if lo <= r["mean_speed"] < hi]
            if values:
                x.append((lo + hi) / 2); y.append(np.percentile(values, 95))
        ax.plot(x, y, "o-", label=label, color=color)
    ax.axvline(HIGH_SPEED_MIN_MPS, color="red", ls="--", label="high-speed cutoff")
    ax.set(xlabel="mean GT vx [m/s]", ylabel="window error P95 [m/s]",
           title=f"Longitudinal tail fine-tuning; gate={'PASS' if gate else 'REJECT'}")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "speed_binned_tail_before_after.png", dpi=180)
    if USE_PLOT:
        plt.show()
    plt.close(fig)
    yaml_updated = bool(not EVALUATE_ONLY and UPDATE_PARAMS_YAML and
                        (gate or not REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE))
    report["update_params_yaml"] = UPDATE_PARAMS_YAML
    report["require_deployment_gate_for_yaml_update"] = (
        REQUIRE_DEPLOYMENT_GATE_FOR_YAML_UPDATE)
    report["yaml_updated"] = yaml_updated
    report_name = "evaluation_report.json" if EVALUATE_ONLY else "report.json"
    (OUTPUT_DIR / report_name).write_text(json.dumps(report, indent=2) + "\n")
    if yaml_updated:
        regression.update_yaml(candidate)
    print(json.dumps(report, indent=2))
    print_finetuning_summary(
        current, candidate, current_metrics, candidate_metrics, gate,
        yaml_updated)
    print(f"report: {OUTPUT_DIR / report_name}")
    print(f"plot: {OUTPUT_DIR / 'speed_binned_tail_before_after.png'}")
    print(f"config updated: {yaml_updated}")
    if USE_PLOT:
        for session in holdout_sessions:
            session["train_evaluation"] = True
        regression.OUTPUT_DIR = OUTPUT_DIR
        regression.SHOW_PLOTS = True
        regression.USE_VALIDATION_TEST_SPLIT = False
        print("Step 5_5 interactive comparison: press p, then click a time-series "
              "panel in the p95/worst context plot.")
        regression.plot_examples(holdout_sessions, current, candidate, cfg)


if __name__ == "__main__":
    main()
