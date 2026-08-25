#!/usr/bin/env python3
"""Plot a GG diagram from the exact dataset/configuration used by Step 3."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import classic_model_regression as regression
import step_3_identify_classic_model as step3


def configure_regression():
    regression.HORIZON = step3.ROLLOUT_HORIZON_STEPS
    regression.MAX_PER_BAG = step3.MAX_WINDOWS_PER_BAG
    regression.V_MIN = step3.V_MIN
    regression.WARMUP_SAMPLES = step3.ACTUATOR_WARMUP_SAMPLES
    regression.GT_CONSISTENCY_MODE = step3.GT_CONSISTENCY_MODE
    regression.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S = (
        step3.VY_POSE_DERIVATIVE_SMOOTH_WINDOW_S)
    regression.MAX_POSITION_STEP_20MS = step3.MAX_POSITION_STEP_20MS
    regression.MAX_YAW_STEP_20MS = step3.MAX_YAW_STEP_20MS


def rollout_sample_mask(data, window_starts):
    mask = np.zeros(len(data["features"]), dtype=bool)
    offsets = np.arange(0, 2 * regression.HORIZON + 1)
    indices = (window_starts[:, None] + offsets[None, :]).ravel()
    mask[np.unique(indices)] = True
    return mask


def draw_gg(axis, ax_mps2, ay_mps2, title, extent):
    g = 9.81
    longitudinal = ax_mps2 / g
    lateral = ay_mps2 / g
    image = axis.hexbin(lateral, longitudinal, gridsize=85, mincnt=1,
                        bins="log", cmap="turbo", extent=extent)
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius, style in ((0.5, "--"), (1.0, "-")):
        axis.plot(radius * np.cos(angle), radius * np.sin(angle),
                  style, color="black", linewidth=0.8, alpha=0.55,
                  label=f"{radius:g} g")
    axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.4)
    axis.set_title(f"{title}\nN={len(longitudinal):,}")
    axis.set_xlabel("lateral acceleration ay [g]")
    axis.set_ylabel("longitudinal acceleration ax [g]")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.15)
    axis.legend(loc="upper right", fontsize=8)
    return image


def metrics(ax_mps2, ay_mps2):
    resultant = np.hypot(ax_mps2, ay_mps2)
    return {
        "samples": int(len(ax_mps2)),
        "ax_min_mps2": float(np.min(ax_mps2)),
        "ax_max_mps2": float(np.max(ax_mps2)),
        "ay_abs_p95_mps2": float(np.quantile(np.abs(ay_mps2), .95)),
        "resultant_p95_mps2": float(np.quantile(resultant, .95)),
        "resultant_p99_mps2": float(np.quantile(resultant, .99)),
        "resultant_max_mps2": float(np.max(resultant)),
    }


def main():
    configure_regression()
    root = step3.ROOT
    config = yaml.safe_load((root / "config/params.yaml").read_text())[
        "/**"]["ros__parameters"]
    data, contract = regression.load_regression_data(step3.DATA_PATH, config)
    split_starts = [regression.starts(data, split) for split in range(3)]
    nonempty = [values for values in split_starts if len(values)]
    if not nonempty:
        raise RuntimeError("No Step 3 rollout window passes the current filters")
    window_starts = np.concatenate(nonempty)
    used_mask = rollout_sample_mask(data, window_starts)

    acceleration = np.asarray(data["observations"][:, :2], dtype=float)
    finite = np.isfinite(acceleration).all(axis=1)
    all_acceleration = acceleration[finite]
    used_acceleration = acceleration[finite & used_mask]
    if not len(used_acceleration):
        raise RuntimeError("No finite acceleration sample is used by Step 3")

    values_g = np.abs(all_acceleration / 9.81)
    limit = max(1.0, float(np.quantile(values_g, .997)))
    limit = np.ceil(limit * 4.0) / 4.0
    extent = (-limit, limit, -limit, limit)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True)
    image0 = draw_gg(axes[0], all_acceleration[:, 0], all_acceleration[:, 1],
                     "All raw IMU samples in Step 3 NPZ files", extent)
    image1 = draw_gg(axes[1], used_acceleration[:, 0], used_acceleration[:, 1],
                     f"Samples used by recursive regression (V_MIN={step3.V_MIN:g} m/s)",
                     extent)
    figure.colorbar(image0, ax=axes[0], label="log sample density")
    figure.colorbar(image1, ax=axes[1], label="log sample density")
    figure.suptitle("Step 3 GG diagram (bias/sign-corrected raw IMU)", fontsize=14)

    output_dir = Path(step3.OUTPUT_DIR) / "gg_diagram"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "step3_raw_gg_diagram.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "data_path": str(Path(step3.DATA_PATH).resolve()),
        "data_contract": contract,
        "acceleration_source": (
            "Step-1 imu_ax/imu_ay after current params.yaml axis sign and bias correction"),
        "v_min_mps": float(step3.V_MIN),
        "horizon_steps": int(step3.ROLLOUT_HORIZON_STEPS),
        "window_counts": {
            "train": int(len(split_starts[0])),
            "validation": int(len(split_starts[1])),
            "test": int(len(split_starts[2])),
        },
        "all_raw": metrics(all_acceleration[:, 0], all_acceleration[:, 1]),
        "regression_used": metrics(used_acceleration[:, 0], used_acceleration[:, 1]),
    }
    report_path = output_dir / "step3_raw_gg_diagram.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"GG diagram: {plot_path}")
    print(f"summary: {report_path}")


if __name__ == "__main__":
    main()
