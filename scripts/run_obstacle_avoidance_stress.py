#!/usr/bin/env python3
"""Run independent Map1 obstacle-avoidance cold starts and archive every run."""
import csv
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

RUNS = int(os.environ.get("F1TENTH_STRESS_RUNS", "30"))
STARTUP_SECONDS = 3.0
RUN_TIMEOUT_SECONDS = 35.0
ROOT = Path("/home/a/smooth-mppi-cuda")
SIM_ROOT = Path("/home/a/f1tenth_gym_ros")
SOURCE_RESULT = ROOT / "model_tuning/map1_closed_loop_no_imu"
OUTPUT = ROOT / "model_tuning/results/obstacle_avoidance_stress_30"


def plot_results(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = np.asarray([row["run"] for row in rows])
    lap_times = np.asarray([row["duration_s"] for row in rows])
    clearances = np.asarray([row["minimum_obstacle_distance_m"] for row in rows])
    obstacle_x = np.asarray([row["obstacle_x_m"] for row in rows])
    obstacle_y = np.asarray([row["obstacle_y_m"] for row in rows])
    reference = np.genfromtxt(ROOT / "data/map1/map1_centerline.csv",
                              delimiter=",", names=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(runs, lap_times, "o-")
    axes[0].set(xlabel="Run", ylabel="Lap time [s]", title="Randomized runs")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(runs, clearances, "o-", label="Measured center distance")
    axes[1].axhline(0.65, color="red", linestyle="--", label="Required 0.65 m")
    axes[1].set(xlabel="Run", ylabel="Minimum distance [m]", title="Obstacle clearance")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(reference["x_m"], reference["y_m"], color="0.7", label="Map1 centerline")
    scatter = axes[2].scatter(obstacle_x, obstacle_y, c=lap_times, cmap="viridis", s=55)
    axes[2].set_aspect("equal"); axes[2].set_title("Randomized obstacle stations")
    axes[2].legend(); figure.colorbar(scatter, ax=axes[2], label="Lap time [s]")
    figure.tight_layout()
    figure.savefig(OUTPUT / "randomized_obstacle_stress_summary.png", dpi=180)
    plt.close(figure)


def launch(command, cwd, log, environment=None):
    stream = log.open("w")
    process = subprocess.Popen(
        command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True, text=True, env=environment)
    return process, stream


def stop(process, stream):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    stream.close()


def evaluate(run_number, requested_obstacle):
    data = np.load(SOURCE_RESULT / "map1_lap_data.npz", allow_pickle=True)
    odom = data["odom"]
    obstacle = data["obstacle"]
    status = str(data["status"])
    center = np.median(obstacle[:, 1:3], axis=0)
    distance = np.hypot(odom[:, 1] - center[0], odom[:, 2] - center[1])
    pose_steps = np.hypot(*np.diff(odom[:, 1:3], axis=0).T)
    row = {
        "run": run_number,
        "status": status,
        "duration_s": float(odom[-1, 0]),
        "minimum_obstacle_distance_m": float(distance.min()),
        "clearance_over_0_65_m": float(distance.min() - 0.65),
        "maximum_speed_mps": float(np.max(odom[:, 4])),
        "maximum_pose_step_m": float(np.max(pose_steps)),
        "obstacle_x_m": float(center[0]),
        "obstacle_y_m": float(center[1]),
        "obstacle_position_error_m": float(np.hypot(*(center-requested_obstacle))),
    }
    row["passed"] = bool(
        status == "lap_complete"
        and row["minimum_obstacle_distance_m"] >= 0.65
        and row["obstacle_position_error_m"] < 0.02
        and row["maximum_pose_step_m"] < 0.5)
    archive = OUTPUT / f"run_{run_number:02d}"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png", "summary.txt"):
        shutil.copy2(SOURCE_RESULT / name, archive / name)
    return row


def build_obstacle_candidates():
    """Select randomized centerline stations with room on both passing sides."""
    reference = np.genfromtxt(ROOT / "data/map1/map1_centerline.csv",
                              delimiter=",", names=True)
    centerline = np.column_stack((reference["x_m"], reference["y_m"]))
    left = np.column_stack((reference["left_x_m"], reference["left_y_m"]))
    right = np.column_stack((reference["right_x_m"], reference["right_y_m"]))
    spawn = np.array([-1.796, -5.478])
    spawn_index = int(np.argmin(np.sum((centerline-spawn)**2, axis=1)))
    segment_lengths = np.linalg.norm(
        np.roll(centerline, -1, axis=0)-centerline, axis=1)
    forward_distance = np.zeros(len(centerline))
    distance = 0.0
    for offset in range(1, len(centerline)):
        previous = (spawn_index + offset - 1) % len(centerline)
        current = (spawn_index + offset) % len(centerline)
        distance += segment_lengths[previous]
        forward_distance[current] = distance
    candidates = []
    last = None
    for index in range(0, len(centerline), 3):
        point, left_point, right_point = centerline[index], left[index], right[index]
        # Do not place an obstacle in the launch transient. At least 4 m is
        # required for the initially stationary car, steering actuator state,
        # and shifted MPPI nominal horizon to reach normal closed-loop state.
        if forward_distance[index] < 4.0 or np.linalg.norm(point-spawn) < 1.5:
            continue
        left_room = np.linalg.norm(point-left_point)
        right_room = np.linalg.norm(point-right_point)
        # A centerline obstacle must leave a physically feasible passing lane
        # on either side. This deliberately excludes narrow bottlenecks rather
        # than weakening collision clearance until an impossible gap passes.
        if min(left_room, right_room) < 1.00:
            continue
        if last is not None and np.linalg.norm(point-last) < 0.55:
            continue
        candidates.append(point.copy())
        last = point.copy()
    if len(candidates) < 5:
        raise RuntimeError(f"only {len(candidates)} feasible obstacle locations")
    return np.asarray(candidates)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixed_obstacle = os.environ.get("F1TENTH_STRESS_FIXED_OBSTACLE")
    if fixed_obstacle:
        fixed = np.asarray([float(value) for value in fixed_obstacle.split(",")])
        if fixed.shape != (2,):
            raise ValueError("F1TENTH_STRESS_FIXED_OBSTACLE must be 'x,y'")
        candidates = fixed[None, :]
    else:
        candidates = build_obstacle_candidates()
    rng = np.random.default_rng(20260816)
    order = rng.permutation(len(candidates))
    rows = []
    for run in range(1, RUNS + 1):
        if (run-1) % len(order) == 0 and run > 1:
            order = rng.permutation(len(candidates))
        obstacle = candidates[order[(run-1) % len(order)]]
        sim_environment = os.environ.copy()
        sim_environment.update({
            "F1TENTH_OBSTACLE_X": f"{obstacle[0]:.9f}",
            "F1TENTH_OBSTACLE_Y": f"{obstacle[1]:.9f}",
            "F1TENTH_OBSTACLE_YAW": "0.0",
        })
        for name in ("map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png", "summary.txt"):
            path = SOURCE_RESULT / name
            if path.exists():
                path.unlink()
        sim, sim_log = launch(
            ["ros2", "launch", "f1tenth_gym_ros", "gym_bridge_launch.py"],
            SIM_ROOT, OUTPUT / f"run_{run:02d}_sim.log", sim_environment)
        recorder = controller = recorder_log = controller_log = None
        try:
            time.sleep(STARTUP_SECONDS)
            recorder, recorder_log = launch(
                ["/usr/bin/python3", "scripts/record_map1_lap.py"], ROOT,
                OUTPUT / f"run_{run:02d}_recorder.log")
            time.sleep(0.5)
            controller, controller_log = launch(
                ["ros2", "launch", "smppi_cuda_controller", "cuda_mppi.launch.py"],
                ROOT, OUTPUT / f"run_{run:02d}_mppi.log")
            deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
            expected_outputs = [SOURCE_RESULT / name for name in (
                "summary.txt", "map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png")]
            while not all(path.exists() for path in expected_outputs) and time.monotonic() < deadline:
                time.sleep(0.1)
            if not all(path.exists() for path in expected_outputs):
                raise TimeoutError("recorder did not produce all result files")
            # The ROS Python process can remain alive after rclpy.shutdown;
            # the atomic test result is the saved summary/NPZ, not process exit.
            stop(recorder, recorder_log)
            recorder_log = None
            row = evaluate(run, obstacle)
        except Exception as error:
            row = {"run": run, "status": "harness_error", "passed": False,
                   "error": repr(error)}
        finally:
            if recorder is not None and recorder_log is not None:
                stop(recorder, recorder_log)
            if controller is not None:
                stop(controller, controller_log)
            stop(sim, sim_log)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (OUTPUT / "results.json").write_text(json.dumps(rows, indent=2))
        if not row["passed"]:
            print(f"FAIL at run {run}; stopping for diagnosis", flush=True)
            break

    keys = sorted({key for row in rows for key in row})
    with (OUTPUT / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    passed = sum(bool(row.get("passed")) for row in rows)
    summary = {"requested_runs": RUNS, "completed_runs": len(rows),
               "passed_runs": passed, "failed_runs": len(rows)-passed,
               "all_passed": passed == RUNS}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_results(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    raise SystemExit(0 if summary["all_passed"] else 1)


if __name__ == "__main__":
    main()
