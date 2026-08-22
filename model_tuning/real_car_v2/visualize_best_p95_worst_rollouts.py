#!/usr/bin/env python3
"""Interactive single-bag open-loop inspector used by unified Step 6."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from callback_training_data import load_callback_archives


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = ROOT / "model_tuning/results/step_6_residual"
DEFAULT_DATA = ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"
SOURCE_DT_S = 0.02
ROLLOUT_DT_S = 0.04


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--new-file", default="inspection_bag_residual.npz")
    parser.add_argument("--baseline-file", default="inspection_bag_classic.npz")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bag-id", type=int, required=True)
    parser.add_argument("--new-label", default="residual MLP")
    parser.add_argument("--baseline-label", default="classic only")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args(argv)


def relative_to_global(relative_pose, initial_pose):
    """Convert body-frame rollout pose to global MCL coordinates."""
    relative_pose = np.asarray(relative_pose, dtype=float)
    cosine, sine = np.cos(initial_pose[2]), np.sin(initial_pose[2])
    result = relative_pose.copy()
    result[:, 0] = initial_pose[0] + relative_pose[:, 0] * cosine - relative_pose[:, 1] * sine
    result[:, 1] = initial_pose[1] + relative_pose[:, 0] * sine + relative_pose[:, 1] * cosine
    result[:, 2] = initial_pose[2] + np.unwrap(relative_pose[:, 2])
    return result


def show_clicked_rollout(index, data, residual, classic, args, result_dir, bag_first):
    start = int(residual["starts"][index])
    residual_global = residual["predicted"][index]
    classic_global = classic["predicted"][index]
    gt = residual["ground_truth"][index].copy()
    horizon_steps = residual_global.shape[0] - 1
    initial_pose = gt[0, :3]
    residual_accel = residual["predicted_acceleration"][index]
    classic_accel = classic["predicted_acceleration"][index]
    gt_accel = residual["ground_truth_acceleration"][index]
    time = ROLLOUT_DT_S * np.arange(horizon_steps + 1)

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(gt[:, 0], gt[:, 1], "k-o", ms=2.3, lw=2.2, label="MCL pose GT")
    axis.plot(classic_global[:, 0], classic_global[:, 1], "--", lw=2,
              label=args.baseline_label)
    axis.plot(residual_global[:, 0], residual_global[:, 1], lw=2,
              label=args.new_label)
    axis.scatter([initial_pose[0]], [initial_pose[1]], color="tab:green", s=50,
                 zorder=5, label="clicked start")
    axis.set(title="global open-loop trajectory", xlabel="global x [m]",
             ylabel="global y [m]")
    axis.axis("equal"); axis.grid(alpha=.3); axis.legend()

    state_panels = (
        (axes[0, 1], 0, "global x", "m"),
        (axes[0, 2], 1, "global y", "m"),
        (axes[1, 0], 2, "global yaw", "rad"),
        (axes[1, 1], 3, "vx", "m/s"),
        (axes[1, 2], 4, "vy", "m/s"),
        (axes[2, 0], 5, "yaw rate", "rad/s"),
    )
    for axis, column, title, unit in state_panels:
        axis.plot(time, gt[:, column], "k", lw=2.2, label="GT")
        axis.plot(time, classic_global[:, column], "--", lw=1.9,
                  label=args.baseline_label)
        axis.plot(time, residual_global[:, column], lw=1.9, label=args.new_label)
        axis.set(title=title, xlabel="rollout time [s]", ylabel=unit)
        axis.grid(alpha=.3); axis.legend(fontsize=8)

    for axis, column, title in (
            (axes[2, 1], 0, "longitudinal acceleration ax"),
            (axes[2, 2], 1, "lateral acceleration ay")):
        axis.plot(time, gt_accel[:, column], "k", lw=2.2, label="IMU GT")
        axis.plot(time, classic_accel[:, column], "--", lw=1.9,
                  label=args.baseline_label)
        axis.plot(time, residual_accel[:, column], lw=1.9, label=args.new_label)
        axis.set(title=title, xlabel="rollout time [s]", ylabel="m/s²")
        axis.grid(alpha=.3); axis.legend(fontsize=8)

    start_time = float(data["anchor_time"][start] - bag_first)
    fig.suptitle(f"Clicked Step 6 open loop | bag_id={args.bag_id} | "
                 f"start={start_time:.3f} s | horizon={horizon_steps * ROLLOUT_DT_S:g} s")
    output = result_dir / f"clicked_step6_bag{args.bag_id}_{start_time:.3f}s.png"
    fig.savefig(output, dpi=180)
    fig.show(); fig.canvas.draw_idle()
    print(f"clicked Step 6 open-loop plot: {output}")


def create_bag_inspector(data, residual, classic, args, result_dir):
    bag_names = np.unique(data["bag_name"])
    bag_id_by_name = {name: index for index, name in enumerate(bag_names)}
    source_bag = np.asarray([bag_id_by_name[name] for name in data["bag_name"]])
    rows = np.flatnonzero(source_bag == args.bag_id)
    if not len(rows):
        raise ValueError(f"bag_id={args.bag_id} does not exist")
    first = float(np.min(data["anchor_time"][rows]))
    time = data["anchor_time"][rows] - first
    states = data["initial_state"]
    observations = data["imu"]
    mcl_pose = data["initial_pose"]
    starts = residual["starts"]
    if not np.array_equal(starts, classic["starts"]):
        raise RuntimeError("classic/residual rollout starts differ")
    selectable = np.flatnonzero(residual["bag_ids"] == args.bag_id)
    if not len(selectable):
        raise RuntimeError(f"bag_id={args.bag_id} has no valid rollout starts")

    fig, axes = plt.subplots(4, 2, figsize=(18, 18), constrained_layout=True)
    trajectory_axis = axes[0, 0]
    trajectory_axis.plot(mcl_pose[rows, 0], mcl_pose[rows, 1], "k", lw=1.8,
                         label="MCL trajectory")
    label_step = max(1, int(round(2.0 / SOURCE_DT_S)))
    for local in range(0, len(rows), label_step):
        trajectory_axis.annotate(f"{time[local]:.0f}s", mcl_pose[rows[local], :2],
                                 xytext=(4, 4), textcoords="offset points", fontsize=8)
    trajectory_axis.set(title="complete bag global MCL trajectory",
                        xlabel="global x [m]", ylabel="global y [m]")
    trajectory_axis.axis("equal"); trajectory_axis.grid(alpha=.3)
    trajectory_axis.legend()

    panels = (
        (axes[0, 1], states[rows, 0], "vx", "m/s"),
        (axes[1, 0], states[rows, 1], "vy", "m/s"),
        (axes[1, 1], np.unwrap(mcl_pose[rows, 2]), "yaw", "rad"),
        (axes[2, 0], states[rows, 2], "yaw rate", "rad/s"),
        (axes[2, 1], observations[rows, 0], "IMU ax", "m/s²"),
        (axes[3, 0], observations[rows, 1], "IMU ay", "m/s²"),
    )
    time_axes = []
    for axis, values, title, unit in panels:
        axis.plot(time, values, "k", lw=1.5)
        axis.set(title=title, xlabel="bag time [s]", ylabel=unit)
        axis.grid(alpha=.3); time_axes.append(axis)
    command_axis = axes[3, 1]
    command_axis.plot(time, data["commands"][rows, 0, 0], label="steer command")
    command_axis.plot(time, data["commands"][rows, 0, 1], label="speed command")
    command_axis.set(title="commands", xlabel="bag time [s]")
    command_axis.grid(alpha=.3); command_axis.legend(); time_axes.append(command_axis)

    armed = {"value": False}
    manager = getattr(fig.canvas, "manager", None)
    handler = getattr(manager, "key_press_handler_id", None)
    if handler is not None:
        fig.canvas.mpl_disconnect(handler)

    def on_key(event):
        if event.key and event.key.lower() == "p":
            armed["value"] = True
            fig.suptitle("PREDICTION ARMED: click a time panel", color="tab:red")
            fig.canvas.draw_idle()
            print("Step 6 prediction armed: click a time-series panel.")

    def on_click(event):
        if not armed["value"] or event.inaxes not in time_axes or event.xdata is None:
            return
        armed["value"] = False
        requested_time = first + float(event.xdata)
        candidates = starts[selectable]
        local = int(np.argmin(np.abs(data["anchor_time"][candidates] - requested_time)))
        selected_index = int(selectable[local])
        selected_time = float(data["anchor_time"][int(starts[selected_index])] - first)
        fig.suptitle(f"bag_id={args.bag_id} | selected start={selected_time:.3f} s",
                     color="black")
        fig.canvas.draw_idle()
        show_clicked_rollout(selected_index, data, residual, classic, args,
                             result_dir, first)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    horizon = (residual["predicted"].shape[1] - 1) * ROLLOUT_DT_S
    fig.suptitle(f"Step 6 bag inspector | bag_id={args.bag_id} | "
                 f"horizon={horizon:g} s | press p, then click a time panel")
    output = result_dir / "step6_bag_inspector.png"
    fig.savefig(output, dpi=180)
    print(f"Step 6 bag inspector: {output}")
    return fig


def main(argv=None):
    args = parse_args(argv)
    print(f"Step 6 visualization backend: {matplotlib.get_backend()}")
    result_dir = args.result_dir.resolve()
    residual = np.load(result_dir / args.new_file)
    classic = np.load(result_dir / args.baseline_file)
    horizon_steps = residual["predicted"].shape[1] - 1
    data = load_callback_archives(args.data, model_dt=ROLLOUT_DT_S,
                                  horizon=horizon_steps)
    inspector = create_bag_inspector(data, residual, classic, args, result_dir)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
