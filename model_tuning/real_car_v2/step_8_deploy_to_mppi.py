#!/usr/bin/env python3
"""Step 8: validate and deploy the selected model to the MPPI runtime."""
from pathlib import Path
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]

# User-editable deployment settings. No command-line arguments are required.
RESULT_PATH = ROOT / "model_tuning/results/dynamic_40ms_yaw_preserved_0815_stage2"
REGRESSION_PATH = ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"
YAML_PATH = ROOT / "config/params.yaml"
RUNTIME_BINARY_PATH = ROOT / "config/dynamic_40ms_residual_servo_lag.bin"
SIMULATOR_YAML_PATHS = (
    ROOT / "f1tenth_gym_ros/src/f1tenth_gym_ros/config/sim.yaml",
    Path("/home/a/f1tenth_gym_ros/src/f1tenth_gym_ros/config/sim.yaml"),
)
ACTIVATE_MODEL = True
ALLOW_BOUNDARY_REGRESSION = True  # explicit real-car deployment selection
EXPECTED_BINARY_BYTES = 14252


def replace_scalar(text, key, value):
    pattern = rf"(?m)^(\s*{re.escape(key)}\s*:\s*)[^#\n]*(.*)$"
    def replacement(match):
        suffix = match.group(2)
        separator = "  " if suffix.startswith("#") else ""
        return f"{match.group(1)}{value}{separator}{suffix}"
    text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"YAML key {key!r} was not found exactly once")
    return text


def main():
    source = RESULT_PATH / "dynamic_40ms_residual.bin"
    if source.stat().st_size != EXPECTED_BINARY_BYTES:
        raise RuntimeError(
            f"invalid CUDA binary size: {source.stat().st_size}, "
            f"expected {EXPECTED_BINARY_BYTES}")
    regression_report = json.loads(REGRESSION_PATH.read_text())
    boundary_override = not regression_report.get("deployment_gate_passed", False)
    if boundary_override and not ALLOW_BOUNDARY_REGRESSION:
        raise RuntimeError("deployment blocked: classic regression has a boundary solution")
    regression = regression_report["expanded_fitted"]
    required = ("B_f", "C_f", "D_f", "E_f", "B_r", "C_r", "D_r", "E_r")
    if not all(k in regression for k in required):
        raise RuntimeError("classic regression JSON is incomplete")

    RUNTIME_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, RUNTIME_BINARY_PATH)
    text = YAML_PATH.read_text()
    updates = {
        "dynamic_mlp_servo_lag_weights_path":
            str(RUNTIME_BINARY_PATH),
        "dynamic_mlp_B_f": regression["B_f"],
        "dynamic_mlp_C_f": regression["C_f"],
        "dynamic_mlp_D_f": regression["D_f"],
        "dynamic_mlp_E_f": regression["E_f"],
        "dynamic_mlp_B_r": regression["B_r"],
        "dynamic_mlp_C_r": regression["C_r"],
        "dynamic_mlp_D_r": regression["D_r"],
        "dynamic_mlp_E_r": regression["E_r"],
        "model_dt": 0.04,
    }
    if ACTIVATE_MODEL:
        updates["dynamics_model"] = "dynamic_mlp_residual_servo_lag"
    for key, value in updates.items():
        text = replace_scalar(text, key, value)
    YAML_PATH.write_text(text)

    simulator_updates = {
        "dynamics_model": "dynamic_mlp_residual_servo_lag",
        "dynamic_mlp_weights_path": str(RUNTIME_BINARY_PATH),
        "dynamic_mlp_model_dt": 0.04,
        "dynamic_mlp_B_f": regression["B_f"],
        "dynamic_mlp_C_f": regression["C_f"],
        "dynamic_mlp_D_f": regression["D_f"],
        "dynamic_mlp_E_f": regression["E_f"],
        "dynamic_mlp_B_r": regression["B_r"],
        "dynamic_mlp_C_r": regression["C_r"],
        "dynamic_mlp_D_r": regression["D_r"],
        "dynamic_mlp_E_r": regression["E_r"],
    }
    updated_simulator_yamls = []
    for simulator_yaml_path in dict.fromkeys(SIMULATOR_YAML_PATHS):
        if not simulator_yaml_path.exists():
            continue
        simulator_text = simulator_yaml_path.read_text()
        for key, value in simulator_updates.items():
            simulator_text = replace_scalar(simulator_text, key, value)
        simulator_yaml_path.write_text(simulator_text)
        updated_simulator_yamls.append(str(simulator_yaml_path))

    digest = hashlib.sha256(RUNTIME_BINARY_PATH.read_bytes()).hexdigest()
    print(json.dumps({
        "source": str(source),
        "runtime_binary": str(RUNTIME_BINARY_PATH),
        "bytes": RUNTIME_BINARY_PATH.stat().st_size,
        "sha256": digest,
        "yaml": str(YAML_PATH),
        "simulator_yamls": updated_simulator_yamls,
        "activated": ACTIVATE_MODEL,
        "boundary_regression_override": boundary_override,
        "updates": updates,
        "rebuild_required": False,
        "restart_node_required": True,
    }, indent=2))


if __name__ == "__main__":
    main()
