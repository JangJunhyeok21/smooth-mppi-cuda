#!/usr/bin/env python3
"""Run isolated IFAC2026 MPPI candidates without cross-domain ROS traffic."""

import argparse
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACK = ROOT / "data/ifac2026/ifac2026_mppi_track_optimal.csv"
MAP = ROOT / "data/ifac2026/ifac2026"
PARAMS = ROOT / "config/params.yaml"
DEFAULT_OUT = ROOT / "model_tuning/results/ifac2026_30lap_tuning/domain_isolated_search"

CANDIDATES = (
    ("baseline_current", 71, {}),
    ("rate8_current", 72,
     {"actuator_max_speed_reference_rate": 8.0}),
    ("rate8_slew15", 73,
     {"actuator_max_speed_reference_rate": 8.0,
      "max_accel_rate": 15.0, "q_du": 0.08}),
    ("rate8_slew20", 74,
     {"actuator_max_speed_reference_rate": 8.0,
      "max_accel_rate": 20.0, "q_du": 0.05, "noise_accel_std": 0.9}),
)


def start(command, env, log_path):
    stream = log_path.open("w")
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True)
    return process, stream


def stop(item):
    if item is None:
        return
    process, stream = item
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
    stream.close()


def run_candidate(candidate, output_root, laps, timeout):
    name, domain, overrides = candidate
    out = output_root / name
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.txt", "map1_lap_data.npz",
                     "map1_mppi_prediction_vs_simulator.png"):
        path = out / filename
        if path.exists():
            path.unlink()

    env = os.environ.copy()
    env.update({
        "ROS_DOMAIN_ID": str(domain),
        "ROS_LOCALHOST_ONLY": "1",
        "F1TENTH_SIM_MAP_PATH": str(MAP),
        "F1TENTH_SIM_TRACK_CSV": str(TRACK),
        "F1TENTH_SIM_NUM_AGENTS": "1",
        "F1TENTH_SIM_ENABLE_RVIZ": "false",
    })
    simulator = controller = recorder = None
    try:
        simulator = start(
            ["ros2", "launch", "f1tenth_gym_ros", "gym_bridge_launch.py"],
            env, out / "simulator.log")
        time.sleep(6)
        recorder = start(
            ["/usr/bin/python3", str(ROOT / "scripts/record_map1_lap.py"),
             "--laps", str(laps), "--timeout", str(timeout),
             "--track", str(TRACK), "--output", str(out)],
            env, out / "recorder.log")
        time.sleep(0.5)
        command = [
            "ros2", "run", "smppi_cuda_controller", "smppi_node", "--ros-args",
            "--params-file", str(PARAMS),
            "-p", "csv_file_path:=data/ifac2026/ifac2026_mppi_track_optimal.csv",
            "-p", "is_simulation:=true",
            "-p", "obstacle_avoidance_enabled:=false",
            "-p", "max_speed:=10.0",
            "-p", "actuator_max_speed_reference_rate:=6.0",
            "-p", "q_progress:=140.0",
            "-p", "q_contour:=0.0",
            "-p", "q_heading:=1.0",
            "-p", "collision_radius:=0.3",
            "-p", "q_boundary_slack:=10000.0",
            "-p", "q_boundary_terminal_slack:=50000.0",
        ]
        for key, value in overrides.items():
            command.extend(("-p", f"{key}:={value}"))
        controller = start(command, env, out / "mppi.log")
        deadline = time.monotonic() + timeout + 20
        summary = out / "summary.txt"
        while not summary.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not summary.exists():
            raise TimeoutError("recorder did not produce summary.txt")
    except Exception as error:
        return {"name": name, "domain": domain, "error": repr(error),
                "overrides": overrides}
    finally:
        stop(controller)
        stop(recorder)
        stop(simulator)

    fields = {}
    summary = out / "summary.txt"
    if summary.exists():
        fields = dict(line.split("=", 1) for line in summary.read_text().splitlines()
                      if "=" in line)
    return {"name": name, "domain": domain, "overrides": overrides, **fields}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--laps", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=110.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate", choices=[item[0] for item in CANDIDATES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = (next(item for item in CANDIDATES if item[0] == args.candidate),) \
        if args.candidate else CANDIDATES
    results = []
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        futures = [pool.submit(run_candidate, item, args.output, args.laps, args.timeout)
                   for item in candidates]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
