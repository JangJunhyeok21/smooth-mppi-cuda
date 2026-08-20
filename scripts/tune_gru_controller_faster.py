#!/usr/bin/env python3
"""GRU simulator에서 충돌 없이 safe25보다 빠른 MPPI 설정을 탐색한다."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "f1tenth_gym_ros"
OUT = ROOT / "model_tuning/results/gru_5mps_curvature_tuning"
VARIANTS = (
    ("curve25", {"max_speed": "5.0", "curve_lateral_accel_limit": "2.5",
        "heading_speed_limit_gain": "0.0", "contour_speed_limit_gain": "0.0",
        "q_progress": "10.0", "q_rear_slip": "1000.0",
        "rear_slip_soft_limit_deg": "5.0", "q_lat_g": "60.0",
        "q_boundary_slack": "10000.0", "q_boundary_terminal_slack": "20000.0"}),
    ("curve35", {"max_speed": "5.0", "curve_lateral_accel_limit": "3.5",
        "heading_speed_limit_gain": "0.0", "contour_speed_limit_gain": "0.0",
        "q_progress": "10.0", "q_rear_slip": "1000.0",
        "rear_slip_soft_limit_deg": "5.0", "q_lat_g": "60.0",
        "q_boundary_slack": "10000.0", "q_boundary_terminal_slack": "20000.0"}),
    ("curve45", {"max_speed": "5.0", "curve_lateral_accel_limit": "4.5",
        "heading_speed_limit_gain": "0.0", "contour_speed_limit_gain": "0.0",
        "q_progress": "10.0", "q_rear_slip": "1000.0",
        "rear_slip_soft_limit_deg": "5.0", "q_lat_g": "60.0",
        "q_boundary_slack": "10000.0", "q_boundary_terminal_slack": "20000.0"}),
    ("curve55", {"max_speed": "5.0", "curve_lateral_accel_limit": "5.5",
        "heading_speed_limit_gain": "0.0", "contour_speed_limit_gain": "0.0",
        "q_progress": "10.0", "q_rear_slip": "1000.0",
        "rear_slip_soft_limit_deg": "5.0", "q_lat_g": "60.0",
        "q_boundary_slack": "10000.0", "q_boundary_terminal_slack": "20000.0"}),
)


def start(command, cwd, log):
    stream = log.open("w")
    process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True, text=True)
    return process, stream


def stop(item):
    if item is None:
        return
    process, stream = item
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4)
    stream.close()


def run_variant(name, overrides, laps, output):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.txt", "map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png"):
        path = directory / filename
        if path.exists():
            path.unlink()
    simulator = recorder = controller = None
    try:
        simulator = start(["ros2", "launch", "f1tenth_gym_ros", "gym_bridge_launch.py"],
                          SIM_ROOT, directory / "sim.log")
        time.sleep(3)
        subprocess.run(["ros2", "topic", "pub", "--once", "/drive",
            "ackermann_msgs/msg/AckermannDriveStamped",
            "{drive: {speed: 0.0, steering_angle: 0.0}}"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["ros2", "topic", "pub", "--once", "/initialpose",
            "geometry_msgs/msg/PoseWithCovarianceStamped",
            "{header: {frame_id: map}, pose: {pose: {position: {x: -1.796, y: -5.478}, orientation: {z: 0.6965, w: 0.7176}}}}"],
            check=True, stdout=subprocess.DEVNULL)
        time.sleep(1)
        timeout = 35 if laps == 1 else 55
        recorder = start(["/usr/bin/python3", "scripts/record_map1_lap.py", "--laps", str(laps),
                          "--timeout", str(timeout), "--output", str(directory)],
                         ROOT, directory / "recorder.log")
        time.sleep(.4)
        command = ["ros2", "run", "smppi_cuda_controller", "smppi_node", "--ros-args",
                   "--params-file", str(ROOT / "config/params.yaml"),
                   "-p", "is_simulation:=true", "-p", "obstacle_avoidance_enabled:=false"]
        for key, value in overrides.items():
            command.extend(("-p", f"{key}:={value}"))
        controller = start(command, ROOT, directory / "controller.log")
        deadline = time.monotonic() + timeout + 6
        while not (directory / "summary.txt").exists() and time.monotonic() < deadline:
            time.sleep(.1)
        if not (directory / "summary.txt").exists():
            raise TimeoutError("recorder timeout")
        fields = dict(line.split("=", 1) for line in (directory / "summary.txt").read_text().splitlines() if "=" in line)
        ratio = float(fields["lap_ratio"]); duration = float(fields["duration_s"])
        return {"variant": name, "overrides": overrides, "status": fields["status"],
                "duration_s": duration, "lap_ratio": ratio,
                "seconds_per_lap": duration / ratio if ratio else 999.0}
    except Exception as error:
        return {"variant": name, "overrides": overrides, "status": "harness_error",
                "seconds_per_lap": 999.0, "error": repr(error)}
    finally:
        stop(controller); stop(recorder); stop(simulator)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scan = []
    for name, overrides in VARIANTS:
        row = run_variant(name, overrides, 1, OUT / "scan")
        scan.append(row); print(json.dumps(row), flush=True)
        (OUT / "scan_results.json").write_text(json.dumps(scan, indent=2) + "\n")
    completed = [row for row in scan if row["status"] == "laps_complete"]
    if not completed:
        raise RuntimeError("no collision-free candidate")
    winner = min(completed, key=lambda row: row["seconds_per_lap"])
    confirmation = run_variant(winner["variant"], winner["overrides"], 2, OUT / "confirmation")
    report = {"scan": scan, "winner": winner, "confirmation": confirmation}
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
