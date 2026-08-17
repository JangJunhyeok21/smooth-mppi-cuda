#!/usr/bin/env python3
"""Cold-start tests for an obstacle first observed during cornering."""
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

RUNS = int(os.environ.get("SUDDEN_OBSTACLE_RUNS", "5"))
BASE_SEED = int(os.environ.get("SUDDEN_OBSTACLE_BASE_SEED", "20260817"))
ROOT = Path("/home/a/smooth-mppi-cuda")
SIM_ROOT = Path("/home/a/f1tenth_gym_ros")
SOURCE = ROOT / "model_tuning/map1_closed_loop_no_imu"
OUTPUT = ROOT / "model_tuning/results/sudden_obstacle_stress"


def plot_results(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [row for row in rows if "minimum_distance_after_detection_m" in row]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    runs = [row["run"] for row in valid]
    clearance = [row["minimum_distance_after_detection_m"] for row in valid]
    axes[0].plot(runs, clearance, "o-", label="Measured minimum distance")
    axes[0].axhline(0.65, color="red", linestyle="--", label="Required 0.65 m")
    axes[0].set(xlabel="Run", ylabel="Center distance [m]",
                title="Sudden reveal during cornering")
    axes[0].grid(True, alpha=.3); axes[0].legend()
    reference = np.genfromtxt(
        ROOT/"data/map1/map1_centerline.csv", delimiter=",", names=True)
    axes[1].plot(reference["x_m"], reference["y_m"], color="0.75",
                 label="Map1 centerline")
    for row in valid:
        archive = OUTPUT/f"run_{row['run']:02d}"/"map1_lap_data.npz"
        if archive.exists():
            odom = np.load(archive, allow_pickle=True)["odom"].astype(float)
            axes[1].plot(odom[:, 1], odom[:, 2], alpha=.35)
    if valid:
        axes[1].scatter(valid[0]["target_x_m"], valid[0]["target_y_m"],
                        marker="x", s=100, color="red", label="Injected obstacle")
    axes[1].axis("equal"); axes[1].grid(True, alpha=.3); axes[1].legend()
    axes[1].set(title="Ten closed-loop trajectories", xlabel="x [m]", ylabel="y [m]")
    figure.tight_layout()
    figure.savefig(OUTPUT/"sudden_obstacle_summary.png", dpi=180)
    plt.close(figure)


def launch(command, cwd, log, environment=None):
    stream = log.open("w")
    process = subprocess.Popen(
        command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True, env=environment)
    return process, stream


def stop(process, stream):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    if stream is not None:
        stream.close()


def evaluate(run, event_path):
    event = json.loads(event_path.read_text())
    data = np.load(SOURCE / "map1_lap_data.npz", allow_pickle=True)
    odom = np.asarray(data["odom"], dtype=float)
    obstacle = np.asarray(data["obstacle"], dtype=float)
    target = np.array([event["target_x_m"], event["target_y_m"]])
    observed = np.hypot(obstacle[:, 1]-target[0], obstacle[:, 2]-target[1]) < 0.1
    detection_time = float(obstacle[np.flatnonzero(observed)[0], 0])
    after = odom[:, 0] >= detection_time
    distance = np.hypot(odom[after, 1]-target[0], odom[after, 2]-target[1])
    row = dict(event)
    row.update({
        "run": run, "status": str(data["status"]),
        "detection_time_s": detection_time,
        "minimum_distance_after_detection_m": float(distance.min()),
        "passed": bool(str(data["status"]) == "lap_complete" and distance.min() >= 0.65),
    })
    archive = OUTPUT / f"run_{run:02d}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png", "summary.txt"):
        shutil.copy2(SOURCE / name, archive / name)
    shutil.copy2(event_path, archive / "event.json")
    return row


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in range(1, RUNS+1):
        event_path = OUTPUT / f"event_{run:02d}.json"
        if event_path.exists(): event_path.unlink()
        for name in ("map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png", "summary.txt"):
            path = SOURCE / name
            if path.exists(): path.unlink()
        environment = os.environ.copy()
        environment.update({
            "F1TENTH_OBSTACLE_X": "20.0", "F1TENTH_OBSTACLE_Y": "20.0",
            "F1TENTH_OBSTACLE_YAW": "0.0",
            "SUDDEN_OBSTACLE_EVENT_PATH": str(event_path),
            # A different deterministic scenario per cold start makes failures
            # reproducible while covering multiple Map1 corners.
            "SUDDEN_OBSTACLE_SEED": str(BASE_SEED + run),
        })
        processes = []
        try:
            processes.append((*launch(
                ["ros2", "launch", "f1tenth_gym_ros", "gym_bridge_launch.py"],
                SIM_ROOT, OUTPUT/f"run_{run:02d}_sim.log", environment),))
            time.sleep(3)
            processes.append((*launch(
                ["/usr/bin/python3", "scripts/record_map1_lap.py"], ROOT,
                OUTPUT/f"run_{run:02d}_recorder.log", environment),))
            processes.append((*launch(
                ["/usr/bin/python3", "scripts/sudden_obstacle_injector.py"], ROOT,
                OUTPUT/f"run_{run:02d}_injector.log", environment),))
            time.sleep(.5)
            processes.append((*launch(
                ["ros2", "launch", "smppi_cuda_controller", "cuda_mppi.launch.py"], ROOT,
                OUTPUT/f"run_{run:02d}_mppi.log", environment),))
            deadline = time.monotonic()+40
            expected = [SOURCE/name for name in (
                "map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png", "summary.txt")]
            while time.monotonic() < deadline and not all(path.exists() for path in expected):
                time.sleep(.1)
            if not event_path.exists():
                raise RuntimeError("turning trigger did not inject an obstacle")
            if not all(path.exists() for path in expected):
                raise TimeoutError("recorder did not finish")
            row = evaluate(run, event_path)
        except Exception as error:
            row = {"run": run, "status": "harness_error", "passed": False,
                   "error": repr(error)}
        finally:
            for process, stream in reversed(processes):
                stop(process, stream)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (OUTPUT/"results.json").write_text(json.dumps(rows, indent=2))
        # Continue after a controller failure so randomized testing reports
        # the full spatial failure distribution instead of only the first
        # selected corner.
    summary = {"requested": RUNS, "completed": len(rows),
               "passed": sum(bool(row.get("passed")) for row in rows)}
    (OUTPUT/"summary.json").write_text(json.dumps(summary, indent=2))
    plot_results(rows)
    print(json.dumps(summary), flush=True)
    raise SystemExit(0 if summary["passed"] == RUNS else 1)


if __name__ == "__main__":
    main()
