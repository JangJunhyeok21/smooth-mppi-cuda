#!/usr/bin/env python3
"""4/4: Report and plot trajectory, velocity, and wrapped yaw-angle errors."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_tuning_utils.plot_single_model_results import main

if __name__ == "__main__":
    main()
