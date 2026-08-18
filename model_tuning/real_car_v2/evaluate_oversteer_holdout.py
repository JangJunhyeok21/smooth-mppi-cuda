#!/usr/bin/env python3
"""Evaluate old/new 40 ms residual models on unseen oversteer windows."""
from pathlib import Path
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("OVERSTEER_DATA", ROOT / "model_tuning/data/dynamic_40ms_residual.npz"))
SOURCE_REPORT = ROOT / "model_tuning/data/dynamic_40ms_all_drive_source_20ms.json"
BASELINE = ROOT / "model_tuning/results/retrain_0817_0818_baseline/evaluation_on_augmented_test.npz"
UPDATED = ROOT / "model_tuning/results/dynamic_40ms_0817_0818_oversteer_peak/evaluation.npz"
OUTPUT = ROOT / "model_tuning/results/dynamic_40ms_0817_0818_oversteer_peak/oversteer_holdout"
COMPARISON_TRACE = os.environ.get("OVERSTEER_COMPARISON_TRACE")
OVERSTEER_EXCESS_RPS = 0.35
MIN_SPEED_MPS = 1.5


def angle_error(a, b):
    return np.abs((a-b+np.pi) % (2*np.pi)-np.pi)


def statistics(values):
    values = np.asarray(values)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def model_metrics(prediction, ground_truth, selected, event_masks, r_kinematic):
    prediction, ground_truth = prediction[selected], ground_truth[selected]
    trajectory = np.linalg.norm(prediction[:, -1, :2]-ground_truth[:, -1, :2], axis=1)
    yaw = angle_error(prediction[:, -1, 2], ground_truth[:, -1, 2])
    state_error = np.abs(prediction[:, -1, 3:6]-ground_truth[:, -1, 3:6])
    sequence_r_error = np.abs(prediction[:, :, 5]-ground_truth[:, :, 5])
    gt_peak = np.max(np.abs(ground_truth[:, :, 5]), axis=1)
    predicted_peak = np.max(np.abs(prediction[:, :, 5]), axis=1)
    recalls = []
    direction = []
    for local, source_index in enumerate(selected):
        event = event_masks[source_index]
        # trace includes the initial sample; event/r_kinematic are 30 knots.
        predicted_r = prediction[local, 1:, 5]
        gt_r = ground_truth[local, 1:, 5]
        threshold = np.abs(r_kinematic[source_index])+OVERSTEER_EXCESS_RPS
        predicted_event = np.abs(predicted_r) > threshold
        recalls.append(np.mean(predicted_event[event]) if event.any() else np.nan)
        direction.append(np.mean(np.sign(predicted_r[event]) == np.sign(gt_r[event]))
                         if event.any() else np.nan)
    return {
        "trajectory_m": statistics(trajectory), "yaw_rad": statistics(yaw),
        "vx_mps": statistics(state_error[:, 0]), "vy_mps": statistics(state_error[:, 1]),
        "yaw_rate_rps": statistics(state_error[:, 2]),
        "yaw_rate_sequence_mae_rps": statistics(sequence_r_error.mean(axis=1)),
        "yaw_rate_peak_abs_error_rps": statistics(np.abs(predicted_peak-gt_peak)),
        "oversteer_event_recall": float(np.nanmean(recalls)),
        "oversteer_yaw_direction_accuracy": float(np.nanmean(direction)),
    }, trajectory


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA)
    if COMPARISON_TRACE:
        trace = np.load(COMPARISON_TRACE)
        starts = trace["starts"].astype(int)
        baseline_prediction = trace["baseline_20d"]
        updated_prediction = trace["vx_delta_history_24d"]
        ground_truth = trace["ground_truth"]
        output = Path(os.environ.get("OVERSTEER_COMPARISON_OUT", OUTPUT))
        baseline_label = os.environ.get("OVERSTEER_BASELINE_LABEL",
                                        "classic_current_8d + residual_mlp_20d")
        updated_label = os.environ.get("OVERSTEER_UPDATED_LABEL",
                                       "classic_adam_recursive_8d + vx_delta_history_24d")
    else:
        baseline = np.load(BASELINE); updated = np.load(UPDATED)
        if not np.array_equal(baseline["starts"], updated["starts"]):
            raise RuntimeError("baseline and updated evaluation windows differ")
        starts = updated["starts"].astype(int)
        baseline_prediction, updated_prediction = baseline["predicted"], updated["predicted"]
        ground_truth = updated["ground_truth"]; output = OUTPUT
        baseline_label = "baseline_pre_0817_0818"
        updated_label = "updated_with_0817_0818"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SOURCE_REPORT.read_text())["sources"]
    new_test_bags = {int(entry["bag_id"]) for entry in manifest
                     if entry["split"] == "test" and "ifac0817_0818" in entry["source"]}
    window_bags = data["source_bag_id"][starts].astype(int)
    new_data_window = np.asarray([bag in new_test_bags for bag in window_bags])
    feature = data["source_features"].astype(float)
    config = yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
    wheelbase = float(config["l_f"])+float(config["l_r"])
    event_masks, r_kinematic, scores = [], [], []
    for start in starts:
        rows = start+2*np.arange(30)
        vx = feature[rows, 0]; applied_steer = feature[rows, 5]
        gt_r = ground_truth[len(scores), 1:, 5]
        expected_r = vx*np.tan(applied_steer)/wheelbase
        active = (vx > MIN_SPEED_MPS) & (np.abs(applied_steer) > .04)
        excess = np.abs(gt_r)-np.abs(expected_r)
        event = active & (excess > OVERSTEER_EXCESS_RPS)
        event_masks.append(event); r_kinematic.append(expected_r)
        scores.append(float(np.max(np.where(active, excess, -np.inf))))
    scores = np.asarray(scores); event_masks = np.asarray(event_masks); r_kinematic = np.asarray(r_kinematic)
    selected = np.flatnonzero(new_data_window & np.isfinite(scores) &
                              (scores > OVERSTEER_EXCESS_RPS))
    if not len(selected): raise RuntimeError("no holdout oversteer windows")
    report = {"definition": {"minimum_speed_mps": MIN_SPEED_MPS,
              "yaw_rate_excess_over_kinematic_rps": OVERSTEER_EXCESS_RPS},
              "all_test_windows": int(len(starts)),
              "new_0817_0818_holdout_windows": int(new_data_window.sum()),
              "oversteer_windows": int(len(selected))}
    old_metrics, old_traj = model_metrics(baseline_prediction, ground_truth,
                                          selected, event_masks, r_kinematic)
    new_metrics, new_traj = model_metrics(updated_prediction, ground_truth,
                                          selected, event_masks, r_kinematic)
    report[baseline_label] = old_metrics
    report[updated_label] = new_metrics
    report["relative_change_percent"] = {
        key: 100*(new_metrics[key]["mean"]/old_metrics[key]["mean"]-1)
        for key in ("trajectory_m", "yaw_rad", "vx_mps", "vy_mps", "yaw_rate_rps",
                    "yaw_rate_sequence_mae_rps", "yaw_rate_peak_abs_error_rps")}
    (output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")

    # Rank by the updated model endpoint trajectory error; show actual tail cases.
    ranks = np.argsort(new_traj)
    cases = (("best", ranks[0]), ("median", ranks[len(ranks)//2]), ("worst", ranks[-1]))
    fig, axes = plt.subplots(3, 6, figsize=(22, 11), constrained_layout=True)
    for row, (label, local) in enumerate(cases):
        source = selected[local]; gt = ground_truth[source]
        old = baseline_prediction[source]; new = updated_prediction[source]
        time = np.arange(len(gt))*.04
        axes[row, 0].plot(gt[:, 0], gt[:, 1], "k-", label="GT")
        axes[row, 0].plot(old[:, 0], old[:, 1], "C1--", label=baseline_label)
        axes[row, 0].plot(new[:, 0], new[:, 1], "C0-", label=updated_label)
        axes[row, 0].axis("equal"); axes[row, 0].set_title(f"{label}: trajectory")
        for column, state_column, title, unit in ((1, 3, "vx", "m/s"), (2, 4, "vy", "m/s"),
                                                   (3, 5, "yaw rate", "rad/s"),
                                                   (4, 2, "yaw", "rad")):
            axes[row, column].plot(time, gt[:, state_column], "k-")
            axes[row, column].plot(time, old[:, state_column], "C1--")
            axes[row, column].plot(time, new[:, state_column], "C0-")
            axes[row, column].set_title(title); axes[row, column].set_ylabel(unit)
        gt_ax = np.gradient(gt[:, 3], .04)-gt[:, 4]*gt[:, 5]
        old_ax = np.gradient(old[:, 3], .04)-old[:, 4]*old[:, 5]
        new_ax = np.gradient(new[:, 3], .04)-new[:, 4]*new[:, 5]
        gt_ay = np.gradient(gt[:, 4], .04)+gt[:, 3]*gt[:, 5]
        old_ay = np.gradient(old[:, 4], .04)+old[:, 3]*old[:, 5]
        new_ay = np.gradient(new[:, 4], .04)+new[:, 3]*new[:, 5]
        axes[row, 5].plot(time, gt_ax, "k-", label="GT ax")
        axes[row, 5].plot(time, old_ax, "C1--", label="old ax")
        axes[row, 5].plot(time, new_ax, "C0-", label="new ax")
        axes[row, 5].plot(time, gt_ay, "k:", label="GT ay")
        axes[row, 5].plot(time, old_ay, "C3--", label="old ay")
        axes[row, 5].plot(time, new_ay, "C2-", label="new ay")
        axes[row, 5].set_title("body acceleration"); axes[row, 5].set_ylabel("m/s²")
        for axis in axes[row]: axis.grid(alpha=.25); axis.set_xlabel("time [s]" if axis is not axes[row,0] else "x [m]")
    axes[0, 0].legend(fontsize=8); axes[0, 5].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Unseen oversteer holdout: {baseline_label} vs {updated_label}", fontsize=15)
    fig.savefig(output/"best_median_worst.png", dpi=180); plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
