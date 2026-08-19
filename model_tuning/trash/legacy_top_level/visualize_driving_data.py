#!/usr/bin/env python3
"""2/4: Visualize the GT trajectories of every extracted bag/segment.

Run without arguments:
    python model_tuning/visualize_driving_data.py
or, from model_tuning/:
    python visualize_driving_data.py
"""
import os
import sys
from pathlib import Path

# =============================================================================
# USER SETTINGS
# Edit only this block. CLEANING_REPORT_PATH may be None.
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "model_tuning/data/ifac0807_mppi_observation.npz"
OUTPUT_PATH = PROJECT_ROOT / "model_tuning/results/ifac0807_driving_data"
CLEANING_REPORT_PATH = None
# True opens interactive plot windows after saving PNG files.
# Set False when running through SSH/headless mode.
SHOW_PLOTS = False  # Save plots without requiring a desktop/display server.
# Interactive backend used when SHOW_PLOTS=True. TkAgg is installed locally.
INTERACTIVE_BACKEND = "TkAgg"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
if SHOW_PLOTS:
    # Must be selected before importing pyplot through generate_plots.
    os.environ["MPLBACKEND"] = INTERACTIVE_BACKEND
    import matplotlib
    matplotlib.use(INTERACTIVE_BACKEND, force=True)

sys.path.insert(0, str(PROJECT_ROOT))
from model_tuning_utils.plot_extracted_bag_gt_trajectories import generate_plots

if __name__ == "__main__":
    generate_plots(DATASET_PATH, OUTPUT_PATH, CLEANING_REPORT_PATH, SHOW_PLOTS)
