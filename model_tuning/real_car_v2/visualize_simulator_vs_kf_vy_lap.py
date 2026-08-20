#!/usr/bin/env python3
"""Compare simulator ground-truth vy with the vy actually supplied to MPPI."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "model_tuning/results/simulator_kf_vy_lap_20260820/map1_lap_data.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input.with_name("visualize_simulator_vs_kf_vy_lap.png")

    data = np.load(args.input, allow_pickle=True)
    odom = np.asarray(data["odom"], dtype=float)
    mlp = np.asarray(data["mlp_input"], dtype=float)
    if odom.ndim != 2 or odom.shape[1] < 7 or mlp.ndim != 2 or mlp.shape[1] < 23:
        raise ValueError("recording does not contain odom and 22-D mppi_mlp_input")

    # Recorder layouts:
    # odom = [t,x,y,yaw,vx,simulator_true_vy,yaw_rate]
    # mlp_input = [t,feature_0(vx),feature_1(KF vy),...,feature_21,...]
    t = mlp[:, 0]
    valid = (t >= odom[0, 0]) & (t <= odom[-1, 0]) & np.isfinite(mlp[:, 2])
    t = t[valid]
    kf_vy = mlp[valid, 2]
    true_vy = np.interp(t, odom[:, 0], odom[:, 5])
    x = np.interp(t, odom[:, 0], odom[:, 1])
    y = np.interp(t, odom[:, 0], odom[:, 2])
    error = kf_vy - true_vy
    abs_error = np.abs(error)
    high_threshold = max(0.10, float(np.quantile(abs_error, 0.90)))
    high = abs_error >= high_threshold
    candidate_lags = np.linspace(-0.30, 0.15, 451)
    lag_mae = []
    for lag in candidate_lags:
        overlap = (t + lag >= odom[0, 0]) & (t + lag <= odom[-1, 0])
        lag_true = np.interp(t[overlap] + lag, odom[:, 0], odom[:, 5])
        lag_mae.append(np.mean(np.abs(kf_vy[overlap] - lag_true)))
    best_lag_index = int(np.argmin(lag_mae))

    metrics = {
        "samples": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "mae_mps": float(np.mean(abs_error)),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "p95_abs_error_mps": float(np.quantile(abs_error, 0.95)),
        "max_abs_error_mps": float(np.max(abs_error)),
        "signed_bias_kf_minus_true_mps": float(np.mean(error)),
        "mean_abs_magnitude_bias_mps": float(np.mean(np.abs(kf_vy) - np.abs(true_vy))),
        "correlation": float(np.corrcoef(true_vy, kf_vy)[0, 1]),
        "fraction_over_0_10_mps": float(np.mean(abs_error > 0.10)),
        "fraction_over_0_20_mps": float(np.mean(abs_error > 0.20)),
        "red_high_error_threshold_mps": high_threshold,
        "best_fit_kf_delay_s": float(-candidate_lags[best_lag_index]),
        "delay_compensated_mae_mps": float(lag_mae[best_lag_index]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    vmax = max(0.10, float(np.quantile(abs_error, 0.99)))
    scatter = ax.scatter(x, y, c=np.minimum(abs_error, vmax), cmap="YlOrRd",
                         vmin=0.0, vmax=vmax, s=13, linewidths=0)
    ax.scatter(x[high], y[high], facecolors="none", edgecolors="red", s=38,
               linewidths=0.9, label=f"large error ≥ {high_threshold:.3f} m/s")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.25); ax.legend(loc="best")
    ax.set_xlabel("map x [m]"); ax.set_ylabel("map y [m]")
    ax.set_title("Lap path colored by |simulator vy − KF vy|")
    fig.colorbar(scatter, ax=ax, label="absolute vy error [m/s]")

    ax = axes[0, 1]
    ax.plot(t, true_vy, color="black", lw=1.6, label="simulator true vy")
    ax.plot(t, kf_vy, color="tab:blue", lw=1.1, label="MPPI input: KF vy")
    ax.fill_between(t, true_vy, kf_vy, where=high, color="red", alpha=.28,
                    label="large-error interval")
    ax.grid(alpha=.3); ax.legend(); ax.set_xlabel("lap time [s]")
    ax.set_ylabel("vy [m/s]"); ax.set_title("Lateral velocity")

    ax = axes[1, 0]
    ax.plot(t, error, color="tab:purple", lw=1.0)
    ax.axhline(0, color="black", lw=.8)
    ax.fill_between(t, error, 0, where=high, color="red", alpha=.4)
    ax.grid(alpha=.3); ax.set_xlabel("lap time [s]")
    ax.set_ylabel("KF vy − true vy [m/s]"); ax.set_title("Signed estimation error")

    ax = axes[1, 1]
    ax.hist(abs_error, bins=35, color="tab:orange", alpha=.82)
    ax.axvline(metrics["mae_mps"], color="black", ls="--", label=f"MAE {metrics['mae_mps']:.3f}")
    ax.axvline(metrics["p95_abs_error_mps"], color="red", ls="--", label=f"P95 {metrics['p95_abs_error_mps']:.3f}")
    ax.grid(alpha=.3); ax.legend(); ax.set_xlabel("absolute vy error [m/s]")
    ax.set_ylabel("samples"); ax.set_title(f"max {metrics['max_abs_error_mps']:.3f} m/s")

    fig.suptitle("Map1 one-lap simulator true vy vs MPPI KF vy", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(output)


if __name__ == "__main__":
    main()
