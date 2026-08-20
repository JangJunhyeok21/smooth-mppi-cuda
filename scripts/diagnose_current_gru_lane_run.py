#!/usr/bin/env python3
"""현재 params.yaml 그대로 GRU simulator에서 1 lap 진단 데이터를 기록한다."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = ROOT / "f1tenth_gym_ros"
OUT = ROOT / "model_tuning/results/current_gru_lane_cost_diagnosis"


def start(command, cwd, log):
    stream = log.open("w")
    process = subprocess.Popen(command, cwd=cwd, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True, text=True)
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("summary.txt", "map1_lap_data.npz", "map1_mppi_prediction_vs_simulator.png"):
        path = OUT / name
        if path.exists():
            path.unlink()
    simulator = recorder = controller = None
    try:
        simulator = start(["ros2", "launch", "f1tenth_gym_ros", "gym_bridge_launch.py"],
                          SIM_ROOT, OUT / "sim.log")
        time.sleep(3.0)
        subprocess.run(["ros2", "topic", "pub", "--once", "/drive",
            "ackermann_msgs/msg/AckermannDriveStamped",
            "{drive: {speed: 0.0, steering_angle: 0.0}}"], check=True,
            stdout=subprocess.DEVNULL)
        subprocess.run(["ros2", "topic", "pub", "--once", "/initialpose",
            "geometry_msgs/msg/PoseWithCovarianceStamped",
            "{header: {frame_id: map}, pose: {pose: {position: {x: -1.796, y: -5.478}, orientation: {z: 0.6965, w: 0.7176}}}}"],
            check=True, stdout=subprocess.DEVNULL)
        time.sleep(1.0)
        recorder = start(["/usr/bin/python3", "scripts/record_map1_lap.py", "--laps", "1",
                          "--timeout", "35", "--output", str(OUT)], ROOT, OUT / "recorder.log")
        time.sleep(0.4)
        controller = start(["ros2", "run", "smppi_cuda_controller", "smppi_node", "--ros-args",
                            "--params-file", str(ROOT / "config/params.yaml"),
                            "-p", "is_simulation:=true",
                            "-p", "obstacle_avoidance_enabled:=false"], ROOT, OUT / "controller.log")
        deadline = time.monotonic() + 40
        while not (OUT / "summary.txt").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not (OUT / "summary.txt").exists():
            raise TimeoutError("recorder timeout")
        print((OUT / "summary.txt").read_text())
    finally:
        stop(controller)
        stop(recorder)
        stop(simulator)
    print(json.dumps({"output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
