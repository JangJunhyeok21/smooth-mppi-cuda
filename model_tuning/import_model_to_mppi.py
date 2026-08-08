#!/usr/bin/env python3
"""5/5: Export a trained MLP to the runtime binary loaded by CUDA MPPI.

The result directory must contain model.pt, normalization.npz and metrics.json
created by train_model.py.  Use --activate only when this model should become
the active model in config/params.yaml.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


MODEL_LAYOUT = {
    "kinematic_noslip_noimu": {
        "dynamics_model": "kinematic_noslip_noimu_direct_speed",
        "weight_key": "kinematic_noslip_noimu_weights_path",
        "binary": "kinematic_noslip_noimu_direct_speed.bin",
    },
    "kinematic_slip_noimu": {
        "dynamics_model": "kinematic_slip_noimu_direct_speed",
        "weight_key": "kinematic_slip_noimu_weights_path",
        "binary": "kinematic_slip_noimu_direct_speed.bin",
    },
    "dynamic_imu": {
        "dynamics_model": "dynamic_imu_recursive",
        "weight_key": "dynamic_imu_weights_path",
        "binary": "dynamic_imu_recursive.bin",
    },
}


def activate_yaml(path, dynamics_model, weight_key, binary):
    text = path.read_text()
    replacements = {
        "dynamics_model": dynamics_model,
        weight_key: str(binary.resolve()),
    }
    for key, value in replacements.items():
        pattern = rf"(?m)^(\s*{re.escape(key)}\s*:\s*).*$"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise RuntimeError(f"{key!r} was not found exactly once in {path}")
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="training result directory")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--name", help="output basename; defaults to the result directory name")
    parser.add_argument("--activate", action="store_true",
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
                      layout["weight_key"], binary)

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
