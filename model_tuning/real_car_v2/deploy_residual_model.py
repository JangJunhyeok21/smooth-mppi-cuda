#!/usr/bin/env python3
"""Validate and deploy the selected residual model for unified Step 6."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import ClassicModelParameters

ROOT = Path(__file__).resolve().parents[2]

# User-editable deployment settings. No command-line arguments are required.
RESULT_PATH = ROOT / "model_tuning/results/dynamic_40ms_yaw_preserved_0815_stage2"
ALTERNATING_SUMMARY = ROOT / "model_tuning/results/alternating/summary.json"
REGRESSION_PATH = ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"
YAML_PATH = ROOT / "config/params.yaml"
RUNTIME_BINARY_PATH = ROOT / "config/dynamic_40ms_residual_servo_lag.bin"
RUNTIME_BINARY_CONFIG_PATH = "config/dynamic_40ms_residual_servo_lag.bin"
SIMULATOR_YAML_PATHS = (
    ROOT / "simulator_ws/src/f1tenth_gym_ros/config/sim.yaml",
)
ACTIVATE_MODEL = True
ALLOW_BOUNDARY_REGRESSION = True  # explicit real-car deployment selection
EXPECTED_BINARY_BYTES = 14780  # 22-D: 기존 20-D + causal IMU ax/ay


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
    parser = argparse.ArgumentParser()
    parser.add_argument("result", nargs="?", default=str(RESULT_PATH),
                        help="배포할 Step 5/6 결과 폴더")
    parser.add_argument("--regression", default=str(REGRESSION_PATH),
                        help="Step 3 classic regression params.json")
    parser.add_argument("--update-simulator", action="store_true",
                        help="시뮬레이터 YAML도 함께 갱신")
    parser.add_argument("--alternating-summary", default=str(ALTERNATING_SUMMARY),
                        help="Step 7 summary; 존재하면 held-out best iteration을 배포")
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    regression_path = Path(args.regression).resolve()
    summary_path=Path(args.alternating_summary)
    selected_iteration=None;unified=None
    if summary_path.exists():
        summary=json.loads(summary_path.read_text());best=summary["best"]
        selected_iteration=best["iteration"];result_path=Path(best["model"]).resolve()
        regression_path=Path(best["classic_params"]).resolve()
        unified=ClassicModelParameters.from_mapping(best["classic_parameters"])
    source = result_path / "dynamic_40ms_residual.bin"
    if source.stat().st_size != EXPECTED_BINARY_BYTES:
        raise RuntimeError(
            f"invalid CUDA binary size: {source.stat().st_size}, "
            f"expected {EXPECTED_BINARY_BYTES}")
    regression_report = json.loads(regression_path.read_text())
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
            RUNTIME_BINARY_CONFIG_PATH,
        "dynamic_mlp_B_f": regression["B_f"],
        "dynamic_mlp_C_f": regression["C_f"],
        "dynamic_mlp_D_f": regression["D_f"],
        "dynamic_mlp_E_f": regression["E_f"],
        "dynamic_mlp_B_r": regression["B_r"],
        "dynamic_mlp_C_r": regression["C_r"],
        "dynamic_mlp_D_r": regression["D_r"],
        "dynamic_mlp_E_r": regression["E_r"],
        "dynamic_mlp_I_z": regression.get("I_z", unified.Iz if unified else
                                             ClassicModelParameters.from_yaml(YAML_PATH).Iz),
        "model_dt": 0.04,
    }
    if unified is not None:
        updates.update(unified.runtime_updates())
    if ACTIVATE_MODEL:
        updates["dynamics_model"] = "dynamic_mlp_residual_servo_lag"
    for key, value in updates.items():
        text = replace_scalar(text, key, value)
    YAML_PATH.write_text(text)

    simulator_updates = {
        "dynamics_model": "dynamic_mlp_residual_servo_lag",
        "dynamic_mlp_weights_path": RUNTIME_BINARY_CONFIG_PATH,
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
    simulator_paths = SIMULATOR_YAML_PATHS if args.update_simulator else ()
    for simulator_yaml_path in dict.fromkeys(simulator_paths):
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
        "selected_iteration": selected_iteration,
        "selection": "best held-out validation iteration" if selected_iteration is not None else "explicit result",
        "boundary_regression_override": boundary_override,
        "updates": updates,
        "rebuild_required": False,
        "restart_node_required": True,
    }, indent=2))


if __name__ == "__main__":
    main()
