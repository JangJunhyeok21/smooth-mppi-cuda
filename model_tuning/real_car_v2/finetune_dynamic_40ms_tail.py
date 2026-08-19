#!/usr/bin/env python3
"""Tail-risk fine-tuning for high-speed yaw-rate recovery windows."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "step_6_finetune_recursive.py"
spec = importlib.util.spec_from_file_location("recursive", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# Start from the dense-yaw recursive model, not the one-step model.
m.INITIAL_MODEL_PATH = m.ROOT / "model_tuning/results/dynamic_40ms_recursive_stage2_seed31"
m.OUTPUT_PATH = m.ROOT / "model_tuning/results/dynamic_40ms_tail_cvar_seed31"
m.EPOCHS = 140
m.DENSE_YAW_RATE_LOSS_WEIGHT = 8.0
m.EARLY_YAW_EXTRA_WEIGHT = 5.0
m.EARLY_YAW_DECAY_SECONDS = 0.24
m.TAIL_CVAR_FRACTION = 0.20
m.TAIL_CVAR_WEIGHT = 3.0
m.HIGH_SPEED_SAMPLE_WEIGHT = 8.0
m.YAW_RECOVERY_SAMPLE_WEIGHT = 6.0
m.main()
