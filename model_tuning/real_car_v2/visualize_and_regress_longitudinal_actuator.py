#!/usr/bin/env python3
"""Identify the deployed speed-reference actuator parameters from bag data.

No CLI arguments are required: edit the constants below, then run this file.
The objective is recursive vx rollout error, not residual-MLP training loss.
Complete source sessions are kept in train/validation/test partitions.
"""
from pathlib import Path
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import differential_evolution, minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_source_dirs = os.environ.get(
    "DYNAMIC_SOURCE_DIRS",
    str(ROOT / "model_tuning/data/ifac0810_0819_autonomous_physics_clean"),
)
SOURCE_DIRS = tuple(Path(value).expanduser().resolve()
                    for value in _source_dirs.split(os.pathsep) if value)
OUTPUT_DIR = ROOT / "model_tuning/results/longitudinal_actuator_regression"
CONFIG_PATH = ROOT / "config/params.yaml"
SOURCE_DT_S = 0.02
ROLLOUT_STEPS = 50                 # 1.0 s at 50 Hz
START_STRIDE = 5                   # overlapping causal rollouts
MAX_ROLLOUTS_PER_SESSION = 800
RANDOM_SEED = 31
OPTIMIZER_POPULATION_SIZE = 36
OPTIMIZER_MAX_ITERATIONS = 80
OPTIMIZER_LOCAL_MAX_ITERATIONS = 600
# A fitted candidate is written only when it is interior to the bounds and
# improves both validation and test MAE. Boundary solutions are non-identifiable.
UPDATE_CONFIG = True
SHOW_PLOTS = True
USE_VALIDATION_TEST_SPLIT = False
TRAIN_EVALUATION_BAG_INDEX = -1

# These are source-session splits, identical to step_3_build_20ms_dataset.py.
VALIDATION_FILES = {"rosbag2_2026_08_08-16_54_33.npz",
    "rosbag2_2026_08_18-14_55_28.npz","rosbag2_2026_08_18-15_26_00.npz",
    "rosbag2_2026_08_19-19_53_54.npz","rosbag2_2026_08_19-20_02_26.npz"}
TEST_FILES = {
    "rosbag2_2026_08_10-21_45_57.npz",
    "rosbag2_2026_08_10-21_52_23.npz",
    "rosbag2_2026_08_17-17_31_57.npz",
    "rosbag2_2026_08_18-14_39_19.npz",
    "rosbag2_2026_08_19-20_23_43.npz",
}

# Identified vector: K_v [1/s], accel tau [s], brake tau [s].  The reference
# slew limit is a fixed runtime/safety setting, not an identifiable parameter.
# Identified-parameter limits. Units: Kp [1/s], time constants [s].
# ``speed_servo_kp`` is not a normalized coefficient; deployed values can be
# much larger than 4 (the current YAML is about 28).
SPEED_SERVO_KP_BOUNDS = (0.05, 60.0)
ACCEL_TIME_CONSTANT_BOUNDS = (0.002, 0.8)
BRAKE_TIME_CONSTANT_BOUNDS = (0.002, 0.8)
BOUNDS = (SPEED_SERVO_KP_BOUNDS,
          ACCEL_TIME_CONSTANT_BOUNDS,
          BRAKE_TIME_CONSTANT_BOUNDS)
WARMUP_S = 0.8 # seconds of warmup before each rollout; 40 samples at 50 Hz = 0.8 s


def load_sessions():
    sessions = []
    paths=sorted({path for directory in SOURCE_DIRS for path in directory.glob("*.npz")})
    if not paths:
        raise RuntimeError(f"no Step-1 NPZ files found in {SOURCE_DIRS}")
    for path in paths:
        with np.load(path) as z:
            if "samples" not in z.files or "columns" not in z.files:
                print(f"Skipping non-Step-1 NPZ without samples/columns: {path}",file=sys.stderr)
                continue
            names = {str(v): i for i, v in enumerate(z["columns"])}
            required=("bag_id","kf_vx","speed_cmd","steer","kf_yaw_rate")
            missing=[name for name in required if name not in names]
            if missing:
                raise RuntimeError(f"{path}: rerun Step 1; missing fields {missing}")
            samples = np.asarray(z["samples"], float)
            sample_dt=float(z["dt"])
        for segment in np.unique(samples[:, names["bag_id"]].astype(int)):
            a = samples[samples[:, names["bag_id"]].astype(int) == segment]
            if len(a) < ROLLOUT_STEPS + 6:
                continue
            split = "test" if path.name in TEST_FILES else "validation" if path.name in VALIDATION_FILES else "train"
            sessions.append({
                "name": f"{path.stem}:segment{segment}", "source": path.name,
                "source_path": str(path),"segment":int(segment),"sample_offset":0,
                "split": split,"split_origin":"predefined bag split",
                "vx": a[:, names["kf_vx"]].astype(float),
                "cmd": a[:, names["speed_cmd"]].astype(float),
                "steer": a[:, names.get("steer_cmd",names["steer"])].astype(float),
                "yaw_rate": a[:, names["kf_yaw_rate"]].astype(float),
                "dt": sample_dt,
            })
    if not sessions:
        raise RuntimeError(f"no segment with at least {ROLLOUT_STEPS+6} samples in {SOURCE_DIRS}")
    present={session["split"] for session in sessions}
    if not {"validation","test"}.issubset(present):
        if len(sessions)>=3:
            # Deterministic bag/segment-disjoint fallback for a new directory
            # whose filenames are not yet listed in VALIDATION_FILES/TEST_FILES.
            ordered=sorted(sessions,key=lambda item:(item["source_path"],item["segment"]))
            ordered[-1]["split"]="test";ordered[-1]["split_origin"]="fallback held-out segment"
            ordered[-2]["split"]="validation";ordered[-2]["split_origin"]="fallback held-out segment"
        elif len(sessions)==1:
            # A freshly extracted directory may temporarily contain one bag.
            # Keep temporal blocks disjoint so the script remains runnable;
            # replace this with bag-disjoint splits once >=3 bags are present.
            source=sessions[0];n=len(source["vx"]);cuts=(0,int(.6*n),int(.8*n),n)
            if min(np.diff(cuts))<ROLLOUT_STEPS+int(round(WARMUP_S/source["dt"]))+1:
                raise RuntimeError("the only available bag is too short for temporal train/validation/test")
            divided=[]
            for split,lo,hi in zip(("train","validation","test"),cuts[:-1],cuts[1:]):
                item={**source,"name":source["name"]+f":temporal_{split}",
                      "split":split,"split_origin":"single-bag temporal fallback",
                      "sample_offset":lo}
                for field in ("vx","cmd","steer","yaw_rate"):
                    item[field]=source[field][lo:hi]
                divided.append(item)
            sessions=divided
        else:
            sessions[0]["split"]="train"
            source=sessions[1];middle=len(source["vx"])//2
            minimum=ROLLOUT_STEPS+int(round(WARMUP_S/source["dt"]))+1
            if min(middle,len(source["vx"])-middle)<minimum:
                raise RuntimeError("two available bags are too short for held-out temporal validation/test")
            heldout=[]
            for split,lo,hi in (("validation",0,middle),("test",middle,len(source["vx"]))):
                item={**source,"name":source["name"]+f":temporal_{split}",
                    "split":split,"split_origin":"two-bag held-out temporal fallback",
                    "sample_offset":lo}
                for field in ("vx","cmd","steer","yaw_rate"):
                    item[field]=source[field][lo:hi]
                heldout.append(item)
            sessions=[sessions[0],*heldout]
    # Verify that each split actually contains informative straight rollouts.
    # Collision-trimmed directories can have nominal validation bags but zero
    # usable windows. Fall back to non-overlapping excitation clusters rather
    # than failing later inside percentile/plot code.
    usable={split:sum(len(choose_starts(item)) for item in sessions if item["split"]==split)
            for split in ("train","validation","test")}
    if any(count==0 for count in usable.values()):
        clusters=[]
        for item in sessions:
            starts=choose_starts(item)
            for cluster in np.split(starts,np.flatnonzero(np.diff(starts)>ROLLOUT_STEPS)+1):
                if len(cluster):clusters.append((item,cluster))
        if len(clusters)<3:
            raise RuntimeError(f"need three disjoint informative rollout groups; found {len(clusters)}")
        clusters.sort(key=lambda pair:len(pair[1]),reverse=True)
        rebuilt=[]
        for index,(item,cluster) in enumerate(clusters):
            split=("train","validation","test")[index] if index<3 else "train"
            rebuilt.append({**item,"name":item["name"]+f":excitation_{index}",
                "split":split,"split_origin":"disjoint excitation-cluster fallback",
                "starts_override":cluster.copy()})
        sessions=rebuilt
    return sessions


def choose_starts(session):
    if "starts_override" in session:
        return np.asarray(session["starts_override"],int).copy()
    n = len(session["vx"])
    warmup = max(1, int(round(WARMUP_S / session["dt"])))
    starts = np.arange(warmup, n - ROLLOUT_STEPS, START_STRIDE)
    if not len(starts):return starts
    # Exclude nearly stationary windows: they mostly identify the acceleration
    # clamp and provide little information about tau/rate.
    informative = np.array([
        np.ptp(session["cmd"][i:i + ROLLOUT_STEPS]) > 0.08 or
        np.mean(np.abs(session["cmd"][i:i + ROLLOUT_STEPS] - session["vx"][i:i + ROLLOUT_STEPS])) > 0.12
        for i in starts
    ],dtype=bool)
    straight = np.array([
        np.mean(np.abs(session["steer"][i:i + ROLLOUT_STEPS])) < 0.12 and
        np.mean(np.abs(session["yaw_rate"][i:i + ROLLOUT_STEPS])) < 0.35
        for i in starts],dtype=bool)
    starts = starts[informative & straight]
    if len(starts) > MAX_ROLLOUTS_PER_SESSION:
        starts = starts[np.linspace(0, len(starts) - 1, MAX_ROLLOUTS_PER_SESSION).astype(int)]
    return starts


def predict_window(vx_gt, cmd, dt, params, max_rate, min_accel, max_accel,
                   warmup_cmd=()):
    kp, tau_accel, tau_brake = params
    predicted = np.empty(len(cmd), float)
    predicted[0] = vx_gt[0]
    speed_reference = vx_gt[0]
    for past_cmd in warmup_cmd:
        tau = tau_accel if past_cmd >= speed_reference else tau_brake
        speed_reference += np.clip((past_cmd-speed_reference)/tau,
                                   -max_rate, max_rate)*dt
    for k in range(len(cmd) - 1):
        tau = tau_accel if cmd[k] >= speed_reference else tau_brake
        reference_rate = np.clip((cmd[k] - speed_reference) / tau, -max_rate, max_rate)
        speed_reference += reference_rate * dt
        acceleration = np.clip(kp * (speed_reference - predicted[k]), min_accel, max_accel)
        predicted[k + 1] = max(0.0, predicted[k] + acceleration * dt)
    return predicted


def residual_vector(params, sessions, split, cfg):
    errors = []
    for session in sessions:
        if split == "train_evaluation":
            if not session.get("train_evaluation", False):
                continue
        elif session["split"] != split:
            continue
        starts = session["starts"]
        if not len(starts):
            continue
        indices = starts[:, None] + np.arange(ROLLOUT_STEPS + 1)[None, :]
        gt = session["vx"][indices]
        cmd = session["cmd"][indices]
        pred = np.empty_like(gt)
        pred[:, 0] = gt[:, 0]
        speed_reference = gt[:, 0].copy()
        kp, tau_accel, tau_brake = params
        warmup_count = max(1, int(round(WARMUP_S/session["dt"])))
        speed_reference = gt[:, 0].copy()
        for offset in range(-warmup_count, 0):
            past = session["cmd"][starts+offset]
            tau = np.where(past >= speed_reference, tau_accel, tau_brake)
            speed_reference += np.clip((past-speed_reference)/tau,
                                       -cfg["actuator_max_speed_reference_rate"],
                                       cfg["actuator_max_speed_reference_rate"])*session["dt"]
        for k in range(ROLLOUT_STEPS):
            tau = np.where(cmd[:, k] >= speed_reference, tau_accel, tau_brake)
            reference_rate = np.clip((cmd[:, k] - speed_reference) / tau,
                                     -cfg["actuator_max_speed_reference_rate"],
                                     cfg["actuator_max_speed_reference_rate"])
            speed_reference += reference_rate * session["dt"]
            acceleration = np.clip(kp * (speed_reference - pred[:, k]),
                                   cfg["min_accel"], cfg["max_accel"])
            pred[:, k + 1] = np.maximum(0.0, pred[:, k] + acceleration * session["dt"])
        # Weight later samples more: the parameter values must remain good
        # recursively, rather than only matching the first transition.
        weights = np.linspace(0.35, 1.0, gt.shape[1])[None, :]
        errors.append((weights * (pred - gt)).ravel())
    return np.concatenate(errors) if errors else np.empty(0)


def robust_objective(params, sessions, cfg):
    e = residual_vector(params, sessions, "train", cfg)
    delta = 0.35
    ae = np.abs(e)
    huber = np.where(ae <= delta, 0.5 * e * e, delta * (ae - 0.5 * delta))
    return float(np.mean(huber))


def metrics(params, sessions, split, cfg):
    e = residual_vector(params, sessions, split, cfg)
    ae = np.abs(e)
    return {"samples": int(len(e)), "mae_mps": float(np.mean(ae)),
            "rmse_mps": float(np.sqrt(np.mean(e * e))),
            "p95_abs_mps": float(np.percentile(ae, 95)),
            "max_abs_mps": float(np.max(ae))}


def update_yaml(params):
    lines = CONFIG_PATH.read_text().splitlines()
    values = dict(zip(("speed_servo_kp", "speed_reference_accel_time_constant",
                       "speed_reference_brake_time_constant"), params))
    found = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        for key, value in values.items():
            if stripped.startswith(key + ":"):
                indent = line[:len(line) - len(line.lstrip())]
                lines[i] = f"{indent}{key}: {value:.9g}  # bag/session-disjoint vx rollout regression"
                found.add(key)
    if found != set(values):
        raise RuntimeError(f"could not update YAML keys: {set(values) - found}")
    CONFIG_PATH.write_text("\n".join(lines) + "\n")


def plot_clicked_open_loop(session, start_time_s, old, fitted, cfg):
    """Predict from an interactively selected time on a SOURCE_DT_S grid."""
    source_time = np.arange(len(session["vx"]), dtype=float) * session["dt"]
    duration_s = ROLLOUT_STEPS * SOURCE_DT_S
    if start_time_s < 0.0 or start_time_s + duration_s > source_time[-1]:
        print(f"Cannot predict at {start_time_s:.3f} s: need {duration_s:.3f} s "
              "of continuous future data.")
        return

    prediction_time = (start_time_s +
                       np.arange(ROLLOUT_STEPS + 1) * SOURCE_DT_S)
    gt = np.interp(prediction_time, source_time, session["vx"])
    cmd = np.interp(prediction_time, source_time, session["cmd"])
    warmup_time = np.arange(max(0.0, start_time_s - WARMUP_S),
                            start_time_s, SOURCE_DT_S)
    warmup_cmd = np.interp(warmup_time, source_time, session["cmd"])
    fixed_rate = cfg["actuator_max_speed_reference_rate"]
    current_prediction = predict_window(
        gt, cmd, SOURCE_DT_S, old, fixed_rate,
        cfg["min_accel"], cfg["max_accel"], warmup_cmd)
    fitted_prediction = predict_window(
        gt, cmd, SOURCE_DT_S, fitted, fixed_rate,
        cfg["min_accel"], cfg["max_accel"], warmup_cmd)
    current_rmse = float(np.sqrt(np.mean((current_prediction - gt) ** 2)))
    fitted_rmse = float(np.sqrt(np.mean((fitted_prediction - gt) ** 2)))

    relative_time = prediction_time - start_time_s
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(relative_time, gt, "k-", lw=2.3, label="GT speed (interpolated KF vx)")
    ax.plot(relative_time, cmd, color="0.45", ls=":", lw=2,
            label="interpolated speed command")
    ax.plot(relative_time, current_prediction, color="tab:blue", ls="--", lw=2,
            label=f"current parameters (RMSE {current_rmse:.4f} m/s)")
    ax.plot(relative_time, fitted_prediction, color="tab:red", lw=2,
            label=f"newly fitted Kp/tau (RMSE {fitted_rmse:.4f} m/s)")
    ax.scatter([0.0], [gt[0]], color="black", s=45, zorder=5,
               label=f"clicked start: {start_time_s:.3f} s")
    ax.set(title=(f"Clicked open-loop prediction | {session['name']} | "
                  f"dt={SOURCE_DT_S:.3f} s"),
           xlabel="time from clicked start [s]", ylabel="speed [m/s]")
    ax.grid(alpha=.3); ax.legend(ncol=2)
    fig.tight_layout()
    safe_name = session["name"].replace(":", "_").replace("/", "_")
    output = OUTPUT_DIR / f"clicked_open_loop_{safe_name}_{start_time_s:.3f}s.png"
    fig.savefig(output, dpi=180)
    fig.show()
    fig.canvas.draw_idle()
    print(f"Clicked open-loop: start={start_time_s:.3f} s, "
          f"current RMSE={current_rmse:.6f} m/s, "
          f"fitted RMSE={fitted_rmse:.6f} m/s")
    print(f"clicked open-loop plot: {output}")


def enable_clicked_open_loop(fig, time_axes, session, old, fitted, cfg):
    """Arm with ``p`` and use the next time-axis click as rollout start."""
    state = {"armed": False}

    # Matplotlib normally maps ``p`` to toolbar pan. Disable that default key
    # handler on this diagnostic figure so ``p`` unambiguously means predict.
    manager = getattr(fig.canvas, "manager", None)
    default_handler = getattr(manager, "key_press_handler_id", None)
    if default_handler is not None:
        fig.canvas.mpl_disconnect(default_handler)

    def on_key(event):
        if event.key and event.key.lower() == "p":
            state["armed"] = True
            print("Prediction selection armed: click a time-series panel.")
            fig.suptitle(
                f"{session['name']} | PREDICTION ARMED: click a start time",
                fontsize=15, color="tab:red")
            fig.canvas.draw_idle()

    def on_click(event):
        if not state["armed"] or event.inaxes not in time_axes or event.xdata is None:
            return
        state["armed"] = False
        start_time_s = float(event.xdata)
        fig.suptitle(
            f"Longitudinal outlier context | {session['name']} | "
            f"selected {start_time_s:.3f} s", fontsize=15, color="black")
        fig.canvas.draw_idle()
        plot_clicked_open_loop(session, start_time_s, old, fitted, cfg)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)


def plot_focus_session(selected_cases, old, fitted, cfg):
    """Plot the full state/trajectory context around p95 and worst windows."""
    focus_cases = {label: case for label, case in selected_cases.items()
                   if label in {"p95", "worst"}}
    # p95/worst currently come from one segment.  Keep this generic in case a
    # later fit selects two segments by creating one diagnostic per segment.
    grouped = {}
    for label, case in focus_cases.items():
        key = (case["session"]["source"], case["session"]["name"])
        grouped.setdefault(key, {})[label] = case
    diagnostics = []
    colors = {"p95": "tab:orange", "worst": "tab:red"}
    for diagnostic_index, ((source_name, session_name), cases) in enumerate(grouped.items()):
        source_path = Path(next(iter(cases.values()))["session"]["source_path"])
        archive = np.load(source_path)
        names = {str(value): index for index, value in enumerate(archive["columns"])}
        samples = archive["samples"].astype(float)
        representative=next(iter(cases.values()))["session"]
        segment = representative["segment"]
        sample_offset=representative.get("sample_offset",0)
        segment_samples = samples[samples[:, names["bag_id"]].astype(int) == segment]
        sample_dt = float(archive["dt"])
        time = np.arange(len(segment_samples)) * sample_dt

        fig, axes = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
        ax = axes[0, 0]
        ax.plot(segment_samples[:, names["kf_x"]], segment_samples[:, names["kf_y"]],
                color="0.65", lw=1.5, label="full KF trajectory")
        for label, case in cases.items():
            global_start=case["session"].get("sample_offset",0)+case["start"]
            rows = slice(global_start,global_start + ROLLOUT_STEPS + 1)
            ax.plot(segment_samples[rows, names["kf_x"]],
                    segment_samples[rows, names["kf_y"]], color=colors[label],
                    lw=3, label=f"{label}: {global_start*sample_dt:.2f}-"
                                   f"{(global_start+ROLLOUT_STEPS)*sample_dt:.2f} s")
            ax.scatter(segment_samples[global_start, names["kf_x"]],
                       segment_samples[global_start, names["kf_y"]],
                       color=colors[label], marker="o", s=45)
        ax.set(title="KF trajectory and ranked longitudinal windows",
               xlabel="x [m]", ylabel="y [m]")
        ax.axis("equal"); ax.grid(alpha=.3); ax.legend()

        def shade(axis):
            for label, case in cases.items():
                global_start=case["session"].get("sample_offset",0)+case["start"]
                begin = global_start * sample_dt
                end = (global_start + ROLLOUT_STEPS) * sample_dt
                axis.axvspan(begin, end, color=colors[label], alpha=.13,
                             label=f"{label} window")

        ax = axes[0, 1]
        ax.plot(time, segment_samples[:, names["kf_vx"]], "k-", lw=2, label="KF vx GT")
        ax.plot(time, segment_samples[:, names["speed_cmd"]], color="0.45", ls=":",
                lw=2, label="speed command")
        for label, case in cases.items():
            global_start=case["session"].get("sample_offset",0)+case["start"]
            local_time = (global_start + np.arange(ROLLOUT_STEPS + 1)) * sample_dt
            ax.plot(local_time, case["current_prediction"], "--", color=colors[label],
                    lw=2, label=f"current prediction ({label})")
        shade(ax); ax.set(title="Longitudinal speed", xlabel="segment time [s]",
                          ylabel="m/s"); ax.grid(alpha=.3); ax.legend(ncol=2)

        panels = (
            (axes[1, 0], "kf_vy", "KF lateral velocity", "m/s"),
            (axes[1, 1], "kf_yaw", "KF yaw", "rad"),
            (axes[2, 0], "kf_yaw_rate", "KF yaw rate", "rad/s"),
        )
        for ax, field, title, unit in panels:
            ax.plot(time, segment_samples[:, names[field]], "k-")
            shade(ax); ax.set(title=title, xlabel="segment time [s]", ylabel=unit)
            ax.grid(alpha=.3)

        ax = axes[2, 1]
        steer_field = "steer_cmd" if "steer_cmd" in names else "steer"
        ax.plot(time, segment_samples[:, names[steer_field]], label="steer command")
        if "imu_ax" in names:
            ax.plot(time, segment_samples[:, names["imu_ax"]], label="IMU ax")
        if "imu_ay" in names:
            ax.plot(time, segment_samples[:, names["imu_ay"]], label="IMU ay")
        shade(ax); ax.set(title="Commands and IMU context", xlabel="segment time [s]")
        ax.grid(alpha=.3); ax.legend(ncol=2)
        fig.suptitle(f"Longitudinal outlier context | {session_name} | "
                     "press p, then click a time panel", fontsize=15)
        enable_clicked_open_loop(
            fig, tuple(axes.flat[1:]), representative, old, fitted, cfg)
        suffix = "" if len(grouped) == 1 else f"_{diagnostic_index}"
        output = OUTPUT_DIR / f"focus_session_diagnostic{suffix}.png"
        fig.savefig(output, dpi=180)
        if SHOW_PLOTS: plt.show()
        plt.close(fig)
        diagnostics.append({
            "session": session_name, "source": str(source_path), "plot": str(output),
            "windows": {label: {
                "start_sample": case["session"].get("sample_offset",0)+case["start"],
                "start_s": (case["session"].get("sample_offset",0)+case["start"])*sample_dt,
                "end_s": (case["session"].get("sample_offset",0)+case["start"]+ROLLOUT_STEPS)*sample_dt,
                "current_rmse_mps": case["current_rmse"],
                "candidate_rmse_mps": case["candidate_rmse"],
            } for label, case in cases.items()},
        })
    return diagnostics


def plot_examples(sessions, old, fitted, cfg):
    fixed_rate = cfg["actuator_max_speed_reference_rate"]
    cases = []
    # Select only held-out windows.  Ranking training windows would make the
    # representative plot look better than genuine generalization performance.
    for session in sessions:
        selected_for_evaluation = (
            session["split"] in {"validation", "test"}
            if USE_VALIDATION_TEST_SPLIT
            else session.get("train_evaluation", False))
        if not selected_for_evaluation:
            continue
        warmup_count = max(1, int(round(WARMUP_S / session["dt"])))
        for start_value in session["starts"]:
            start = int(start_value)
            sl = slice(start, start + ROLLOUT_STEPS + 1)
            gt, cmd = session["vx"][sl], session["cmd"][sl]
            past = session["cmd"][start-warmup_count:start]
            current_prediction = predict_window(
                gt, cmd, session["dt"], old, fixed_rate,
                cfg["min_accel"], cfg["max_accel"], past)
            candidate_prediction = predict_window(
                gt, cmd, session["dt"], fitted, fixed_rate,
                cfg["min_accel"], cfg["max_accel"], past)
            cases.append({
                "session": session, "start": start, "gt": gt, "cmd": cmd,
                "current_prediction": current_prediction,
                "candidate_prediction": candidate_prediction,
                "current_rmse": float(np.sqrt(np.mean((current_prediction-gt)**2))),
                "candidate_rmse": float(np.sqrt(np.mean((candidate_prediction-gt)**2))),
            })
    if not cases:
        raise RuntimeError("no held-out longitudinal rollout available for plotting")
    cases.sort(key=lambda case: case["current_rmse"])
    selected_indices = (0, int(round(0.95 * (len(cases)-1))), len(cases)-1)
    selected = [cases[index] for index in selected_indices]
    labels = ("best", "p95", "worst")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12.5), squeeze=False)
    case_report = {"ranking_metric": "current-parameter holdout rollout RMSE [m/s]",
                   "total_holdout_rollouts": len(cases), "cases": {}}
    selected_cases = dict(zip(labels, selected))
    for label, ax, case in zip(labels, axes[:, 0], selected):
        session, start = case["session"], case["start"]
        gt, cmd = case["gt"], case["cmd"]
        sl = slice(start, start + ROLLOUT_STEPS + 1)
        t = np.arange(len(gt)) * session["dt"]
        ax.plot(t, gt, color="black", lw=2.2, label="GT speed (causal KF vx)")
        ax.plot(t, cmd, color="tab:gray", ls=":", lw=2, label="vehicle speed command")
        ax.plot(t, case["current_prediction"], color="tab:blue", ls="--", lw=2,
                label="prediction (current/accepted parameters)")
        ax.plot(t, case["candidate_prediction"], color="tab:red", lw=1.8,
                label="prediction (newly fitted Kp, accel tau, brake tau)")
        ax.set(title=(f"{label.upper()} holdout | current RMSE={case['current_rmse']:.4f} m/s, "
                      f"candidate RMSE={case['candidate_rmse']:.4f} m/s | "
                      f"{session['split']}: {session['name']}"),
               xlabel="rollout time [s]", ylabel="speed [m/s]")
        ax.grid(alpha=.3); ax.legend(ncol=2)
        case_report["cases"][label] = {
            "session": session["name"], "split": session["split"], "start": start,
            "current_rmse_mps": case["current_rmse"],
            "candidate_rmse_mps": case["candidate_rmse"],
        }
    fig.suptitle("Longitudinal actuator holdout rollouts: best / p95 / worst", fontsize=15)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "rollout_comparison.png", dpi=180)
    case_report["focus_session_diagnostics"] = plot_focus_session(
        selected_cases, old, fitted, cfg)
    (OUTPUT_DIR / "representative_rollouts.json").write_text(
        json.dumps(case_report, indent=2) + "\n")
    if SHOW_PLOTS: plt.show()
    plt.close(fig)


def main():
    global BOUNDS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG_PATH.read_text())["/**"]["ros__parameters"]
    sessions = load_sessions()
    for session in sessions:
        session["starts"] = choose_starts(session)
    if not USE_VALIDATION_TEST_SPLIT:
        for session in sessions:
            session["split"] = "train"
            session["train_evaluation"] = False
        usable_sources = list(dict.fromkeys(
            session["source"] for session in sessions if len(session["starts"])))
        if not usable_sources:
            raise RuntimeError("no usable train bag is available for evaluation")
        evaluation_source = usable_sources[TRAIN_EVALUATION_BAG_INDEX]
        for session in sessions:
            session["train_evaluation"] = session["source"] == evaluation_source
        evaluation_splits = ("train_evaluation", "train_evaluation")
        evaluation_contract = (
            f"in-sample train-bag diagnostic: {evaluation_source}")
    else:
        evaluation_splits = ("validation", "test")
        evaluation_contract = "bag-disjoint validation/test"
    print(f"Step 2 evaluation mode: {evaluation_contract}")
    old = np.array([cfg["speed_servo_kp"],
                    cfg["speed_reference_accel_time_constant"],
                    cfg["speed_reference_brake_time_constant"],
                    ], float)
    local=float(os.environ.get("IDENTIFICATION_LOCAL_FRACTION","0"))
    if local>0:
        original=np.asarray(BOUNDS,float);candidate=np.column_stack((old*(1-local),old*(1+local)))
        candidate[:,0]=np.maximum(candidate[:,0],original[:,0]);candidate[:,1]=np.minimum(candidate[:,1],original[:,1]);BOUNDS=tuple(map(tuple,candidate))
    # Differential evolution otherwise samples only random points.  Include the
    # deployed YAML parameters explicitly so regression can never overlook a
    # strong existing solution merely because the random population missed it.
    bounds_array = np.asarray(BOUNDS, float)
    rng = np.random.default_rng(RANDOM_SEED)
    population = rng.uniform(bounds_array[:, 0], bounds_array[:, 1],
                             size=(OPTIMIZER_POPULATION_SIZE, 3))
    population[0] = np.clip(old, bounds_array[:, 0], bounds_array[:, 1])
    result = differential_evolution(lambda p: robust_objective(p, sessions, cfg), BOUNDS,
                                    seed=RANDOM_SEED, init=population,
                                    maxiter=OPTIMIZER_MAX_ITERATIONS, tol=2e-5,
                                    polish=False, workers=1, updating="immediate")
    start = min((old, result.x), key=lambda p: robust_objective(p, sessions, cfg))
    refined = minimize(lambda p: robust_objective(p, sessions, cfg), start,
                       method="Powell", bounds=BOUNDS,
                       options={"maxiter": OPTIMIZER_LOCAL_MAX_ITERATIONS,
                                "xtol": 1e-6, "ftol": 1e-9})
    refined_candidate = np.clip(refined.x, bounds_array[:, 0], bounds_array[:, 1])
    fitted = np.asarray(min((old, result.x, refined_candidate),
                            key=lambda p: robust_objective(p, sessions, cfg)), float)
    validation_split, test_split = evaluation_splits
    report = {
        "parameter_order": ["speed_servo_kp", "speed_reference_accel_time_constant", "speed_reference_brake_time_constant"],
        "previous": old.tolist(), "fitted": fitted.tolist(),
        "fixed_v_ref_slew_rate_max": float(cfg["actuator_max_speed_reference_rate"]),
        "objective": "1.0 s recursive KF-vx rollout with candidate-dependent 0.8 s v_ref warm-up, straight-window train Huber",
        "target_source": "causal classic-model KF vx (sensor-fused estimate, not physical GT)",
        "evaluation_contract": evaluation_contract,
        "use_validation_test_split": USE_VALIDATION_TEST_SPLIT,
        "metrics_previous": {
            "train": metrics(old, sessions, "train", cfg),
            "validation": metrics(old, sessions, validation_split, cfg),
            "test": metrics(old, sessions, test_split, cfg)},
        "metrics_fitted": {
            "train": metrics(fitted, sessions, "train", cfg),
            "validation": metrics(fitted, sessions, validation_split, cfg),
            "test": metrics(fitted, sessions, test_split, cfg)},
        "sessions": [{"name": s["name"], "split": s["split"], "rollouts": int(len(s["starts"]))} for s in sessions],
    }
    margin = 0.01 * (np.array(BOUNDS)[:, 1] - np.array(BOUNDS)[:, 0])
    boundary = ((fitted-np.array(BOUNDS)[:, 0] <= margin) |
                (np.array(BOUNDS)[:, 1]-fitted <= margin))
    gate = (not boundary.any() and
            report["metrics_fitted"]["validation"]["mae_mps"] < report["metrics_previous"]["validation"]["mae_mps"] and
            report["metrics_fitted"]["test"]["mae_mps"] < report["metrics_previous"]["test"]["mae_mps"])
    report["boundary_solution"] = dict(zip(report["parameter_order"], boundary.tolist()))
    report["deployment_gate_passed"] = bool(gate)
    (OUTPUT_DIR / "regression.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_examples(sessions, old, fitted, cfg)
    if UPDATE_CONFIG and gate:
        update_yaml(fitted)
    print(json.dumps(report, indent=2))
    print("\nTuned longitudinal-actuator parameter candidates:")
    for name, previous, tuned in zip(report["parameter_order"], old, fitted):
        print(f"  {name}: {previous:.9g} -> {tuned:.9g}")
    if UPDATE_CONFIG and gate:
        print(f"Applied the tuned parameters to: {CONFIG_PATH}")
    elif not UPDATE_CONFIG:
        print("Tuned parameters were not applied to params.yaml "
              "(UPDATE_CONFIG is disabled).")
    else:
        print("Tuned parameters were not applied to params.yaml "
              "(deployment gate did not pass).")
    print(f"plot: {OUTPUT_DIR / 'rollout_comparison.png'}")
    print(f"config updated: {UPDATE_CONFIG and gate}")


if __name__ == "__main__":
    main()
