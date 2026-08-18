#!/usr/bin/env python3
"""5/5: Export a trained MLP to the runtime binary loaded by CUDA MPPI.

The result directory must contain model.pt, normalization.npz and metrics.json
created by the archived multi-model training pipeline. This converter remains
for reproducing legacy kinematic/E2E experiments; the current 40 ms residual
is deployed by real_car_v2/deploy_dynamic_40ms_to_mppi.py.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "model_tuning/results/ifac0807_0808_kf_cf12p72_cr75p09_yaw_curriculum"
OUTPUT_NAME = "slip_kinematic_MLP"
ACTIVATE_MODEL = False


MODEL_LAYOUT = {
    "kinematic_noslip_noimu": {
        "dynamics_model": "kinematic_noslip_noimu_direct_speed",
        "weight_key": "kinematic_noslip_noimu_weights_path",
        "binary": "kinematic_noslip_noimu_direct_speed.bin",
    },
    "slip_kinematic_with_imu": {
        "dynamics_model": "slip_kinematic_with_imu_direct_speed",
        "weight_key": "slip_kinematic_with_imu_weights_path",
        "binary": "slip_kinematic_with_imu_direct_speed.bin",
    },
    "dynamic_residual": {
        "dynamics_model": "dynamic_mlp_residual",
        "weight_key": "dynamic_mlp_weights_path",
        "binary": "dynamic_MLP.bin",
    },
    "e2e_mlp": {
        "dynamics_model": "e2e_mlp",
        "weight_key": "e2e_weights_path",
        "binary": "E2E.bin",
    },
    "dynamic_imu": {
        "dynamics_model": "dynamic_imu_recursive",
        "weight_key": "dynamic_imu_weights_path",
        "binary": "dynamic_imu_recursive.bin",
    },
}


def activate_yaml(path, dynamics_model, weight_key, binary, metrics):
    text = path.read_text()
    replacements = {
        "dynamics_model": dynamics_model,
        weight_key: str(binary.resolve()),
    }
    actuator=metrics.get("actuator_model") or {}
    steering=metrics.get("steering_command_mapping") or {}
    # Direct previous-command steering does not use servo parameters. Preserve
    # the separately selectable servo-lag model's tau/rate in the shared YAML.
    if actuator and not actuator.get("direct_steer", False):
        if actuator.get("servo_time_constant_s") is not None:
            replacements["steer_servo_time_constant"]=actuator["servo_time_constant_s"]
        if actuator.get("max_steering_rate_rad_s") is not None:
            replacements["actuator_max_steer_rate"]=actuator["max_steering_rate_rad_s"]
    yaw_actuator=metrics.get("yaw_rate_actuator_model") or {}
    if yaw_actuator:
        replacements["kinematic_yaw_rate_time_constant"]=yaw_actuator["time_constant_s"]
        replacements["kinematic_max_yaw_accel"]=yaw_actuator["max_yaw_accel_radps2"]
    if steering:
        replacements["kinematic_steer_scale"]=steering["scale"]
        replacements["kinematic_steer_bias"]=steering["bias_rad"]
    if metrics.get("position_speed_scale") is not None:
        replacements["kinematic_position_speed_scale"]=metrics["position_speed_scale"]
    if metrics.get("kp_speed") is not None:
        replacements["speed_servo_kp"]=metrics["kp_speed"]
    stiffness=metrics.get("kf_cornering_stiffness_N_per_rad") or {}
    if stiffness:
        replacements["kf_cornering_stiffness_front"]=stiffness["front"]
        replacements["kf_cornering_stiffness_rear"]=stiffness["rear"]
    signs=metrics.get("imu_axis_signs") or {}
    for short,key in (("wz","imu_wz_sign"),("ax","imu_ax_sign"),("ay","imu_ay_sign")):
        if short in signs:replacements[key]=signs[short]
    speed_limits=metrics.get("runtime_speed_limits_mps")
    if speed_limits:
        replacements["min_speed"],replacements["max_speed"]=speed_limits
    dynamic=metrics.get("dynamic_classic_params") or {}
    for short,key in (("Bf","dynamic_mlp_B_f"),("Cf","dynamic_mlp_C_f"),
                      ("Df","dynamic_mlp_D_f"),("Ef","dynamic_mlp_E_f"),
                      ("Br","dynamic_mlp_B_r"),("Cr","dynamic_mlp_C_r"),
                      ("Dr","dynamic_mlp_D_r"),("Er","dynamic_mlp_E_r"),
                      ("Iz","dynamic_mlp_I_z")):
        if short in dynamic:replacements[key]=dynamic[short]
    for key, value in replacements.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}\s*:\s*).*$"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise RuntimeError(f"{key!r} was not found exactly once in {path}")
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", nargs="?", default=str(RESULT_PATH), help="training result directory")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--name", default=OUTPUT_NAME, help="output basename; defaults to the result directory name")
    parser.add_argument("--activate", action="store_true", default=ACTIVATE_MODEL,
                        help="also select the exported model in config/params.yaml")
    args = parser.parse_args()

    result = Path(args.result).resolve()
    root = Path(args.project_root).resolve()
    metrics = json.loads((result / "metrics.json").read_text())
    model = metrics.get("model")
    if model not in MODEL_LAYOUT:
        raise SystemExit(f"unsupported model {model!r}; expected one of {tuple(MODEL_LAYOUT)}")
    layout = MODEL_LAYOUT[model]
    name = args.name or result.name
    binary = root / "config" / f"{name}.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)

    state = torch.load(result / "model.pt", map_location="cpu", weights_only=True)
    keys = ("net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias",
            "net.4.weight", "net.4.bias")
    norm = np.load(result / "normalization.npz")
    mean = np.r_[norm["base_mean"], np.tile(norm["command_mean"], 5)].astype(np.float32)
    std = np.r_[norm["base_std"], np.tile(norm["command_std"], 5)].astype(np.float32)
    # Binary layout: layer weights/biases, then feature mean and std. Both the
    # network and its normalization are therefore replaced at node startup.
    with binary.open("wb") as stream:
        for key in keys:
            np.asarray(state[key], dtype=np.float32).ravel().tofile(stream)
        mean.tofile(stream)
        std.tofile(stream)

    if args.activate:
        activate_yaml(root / "config" / "params.yaml", layout["dynamics_model"],
                      layout["weight_key"], binary, metrics)

    summary = {
        "model": model,
        "dynamics_model": layout["dynamics_model"],
        "binary": str(binary),
        "input_size": int(len(mean)),
        "normalization": "embedded in runtime binary",
        "params_yaml_activated": args.activate,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
