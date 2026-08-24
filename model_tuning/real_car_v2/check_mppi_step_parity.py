#!/usr/bin/env python3
"""Compare the deployed simulator and CUDA MPPI 40 ms transition exactly."""
from pathlib import Path
import importlib.util
import json
import subprocess
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CUDA_EXE = ROOT / "build/smppi_cuda_controller/mppi_step_parity"
SIMULATOR_MODEL = (ROOT / "f1tenth_gym_ros/src/f1tenth_gym/f1tenth_gym/envs"
                   / "dynamic_models/dynamic_mlp_residual.py")
STEPS = 30
TOLERANCE = 2.0e-3
LABELS = ["x", "y", "yaw", "vx", "vy", "yaw_rate", "ax", "ay", "beta",
          "steer[-4]", "speed[-4]", "steer[-3]", "speed[-3]", "steer[-2]",
          "speed[-2]", "steer[-1]", "speed[-1]", "steer[0]", "speed[0]",
          "applied_steer", "speed_reference"]


def resolve_repo_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def simulator_rollout(cfg, binary):
    spec = importlib.util.spec_from_file_location("simulator_dynamic_mlp_residual",
                                                  SIMULATOR_MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weights, mean, std = module.load_weights(binary)
    state = np.array([1., -.5, 0., 2.2, .3, -.4,
                      np.arctan2(.15, 2.2), .15], np.float32)
    current_accel = np.array([.2, -.1], np.float32)
    history = np.array([-.10, 2., -.05, 2.2, 0., 2.5, .08, 2.8, .12, 3.],
                       np.float32)
    applied = np.float32(.07)
    speed_reference = np.float32(2.)
    vx_history = np.array([1.95, 2., 2.08, 2.14, 2.2], np.float32)
    kwargs = dict(
        dt=cfg["model_dt"], lf=cfg["l_f"], lr=cfg["l_r"], mass=cfg["mass"],
        min_speed=cfg["min_speed"], max_speed=cfg["max_speed"],
        min_accel=cfg["min_accel"], max_accel=cfg["max_accel"],
        speed_servo_kp=cfg["speed_servo_kp"],
        speed_accel_tau=cfg["speed_reference_accel_time_constant"],
        speed_brake_tau=cfg["speed_reference_brake_time_constant"],
        max_speed_reference_rate=cfg["actuator_max_speed_reference_rate"],
        steer_scale=cfg["kinematic_steer_scale"],
        steer_bias=cfg["kinematic_steer_bias"],
        steer_time_constant=cfg["steer_servo_time_constant"],
        max_steer=cfg["max_steer"], max_steer_rate=cfg["actuator_max_steer_rate"],
        position_speed_scale=cfg["kinematic_position_speed_scale"],
        Bf=cfg["dynamic_mlp_B_f"], Cf=cfg["dynamic_mlp_C_f"],
        Df=cfg["dynamic_mlp_D_f"], Ef=cfg["dynamic_mlp_E_f"],
        Br=cfg["dynamic_mlp_B_r"], Cr=cfg["dynamic_mlp_C_r"],
        Dr=cfg["dynamic_mlp_D_r"], Er=cfg["dynamic_mlp_E_r"],
        Iz=cfg["dynamic_mlp_I_z"],
        mlp_max_residual_ax=cfg["mlp_max_residual_ax"],
        mlp_max_residual_ay=cfg["mlp_max_residual_ay"],
        mlp_max_residual_yaw_accel=cfg["mlp_max_residual_yaw_accel"])
    rows = []
    for k in range(STEPS):
        steer = .25 - .012 * k
        speed = np.clip(3.5 - .025 * k, cfg["min_speed"], cfg["max_speed"])
        state, history, applied, speed_reference, imu, next_vx_history = module.step(
            state, steer, speed, history, applied, speed_reference,
            weights, mean, std, current_accel=current_accel,
            vx_history=vx_history, **kwargs)
        if next_vx_history is not None:
            vx_history = next_vx_history
        canonical = np.array([state[0], state[1], state[4], state[3], state[7],
                              state[5], imu[1], imu[2], state[6]])
        current_accel = imu[1:3]
        rows.append(np.r_[canonical, history, applied, speed_reference])
    return np.asarray(rows)


def main():
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text())["/**"]["ros__parameters"]
    binary = resolve_repo_path(cfg["dynamic_mlp_servo_lag_weights_path"])
    if not CUDA_EXE.exists():
        raise RuntimeError(f"build the parity executable first: {CUDA_EXE}")
    keys = ["dynamic_mlp_B_f", "dynamic_mlp_C_f", "dynamic_mlp_D_f", "dynamic_mlp_E_f",
            "dynamic_mlp_B_r", "dynamic_mlp_C_r", "dynamic_mlp_D_r", "dynamic_mlp_E_r",
            "dynamic_mlp_I_z", "min_speed", "max_speed", "min_accel", "max_accel",
            "mass", "l_f", "l_r", "kinematic_steer_scale", "kinematic_steer_bias",
            "steer_servo_time_constant", "max_steer", "actuator_max_steer_rate",
            "speed_servo_kp", "speed_reference_accel_time_constant",
            "speed_reference_brake_time_constant", "actuator_max_speed_reference_rate",
            "kinematic_position_speed_scale", "mlp_max_residual_ax",
            "mlp_max_residual_ay", "mlp_max_residual_yaw_accel"]
    command = [str(CUDA_EXE), str(binary), str(STEPS), *[str(cfg[key]) for key in keys]]
    run = subprocess.run(command, text=True, capture_output=True)
    if run.returncode:
        print(run.stderr, file=sys.stderr)
        raise SystemExit(run.returncode)
    cuda = np.loadtxt(run.stdout.splitlines())
    simulator = simulator_rollout(cfg, binary)
    error = np.abs(cuda - simulator)
    worst = np.unravel_index(np.argmax(error), error.shape)
    report = {
        "status": "PASS" if float(error.max()) < TOLERANCE else "FAIL",
        "model": cfg["dynamics_model"], "weights": str(binary),
        "model_dt_s": cfg["model_dt"], "steps": STEPS,
        "one_step_max_abs_error": float(error[0].max()),
        "recursive_max_abs_error": float(error.max()),
        "worst_step": int(worst[0] + 1), "worst_state": LABELS[worst[1]],
        "worst_simulator_value": float(simulator[worst]),
        "worst_mppi_value": float(cuda[worst]), "tolerance": TOLERANCE}
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
