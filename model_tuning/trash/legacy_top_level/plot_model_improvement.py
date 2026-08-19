#!/usr/bin/env python3
import json,os
from pathlib import Path

SHOW_PLOTS = True  # True: save PNGs and open both figures; False: save only.
INTERACTIVE_BACKEND = "TkAgg"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smppi")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/smppi-cache")
HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
import matplotlib
matplotlib.use(INTERACTIVE_BACKEND if SHOW_PLOTS and HAS_DISPLAY else "Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[1];os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-smppi')
OLD=ROOT/'model_tuning/results/ifac0808_221108_fixed_model/metrics.json'
NEW=ROOT/'model_tuning/results/ifac0807_0808_actuator_regressed_yaw_curriculum/visualization/metrics.json'
OUT=ROOT/'model_tuning/results/ifac0807_0808_actuator_regressed_yaw_curriculum/visualization'
old=json.loads(OLD.read_text())['fixed_preserve_measured_overspeed']['1.00s'];new=json.loads(NEW.read_text())
items=[('Trajectory mean [m]',old['trajectory_mean_m'],new['final_trajectory_mean_m']),('Trajectory P95 [m]',old['trajectory_p95_m'],new['final_trajectory_p95_m']),('Speed MAE [m/s]',old['speed_mae_mps'],new['final_speed_mae_mps']),('Yaw-rate MAE [rad/s]',old['yaw_rate_mae_radps'],new['final_yaw_rate_mae_radps']),('Yaw MAE [deg]',old['yaw_mae_deg'],np.degrees(new['final_yaw_mae_rad']))]
fig,axes=plt.subplots(1,len(items),figsize=(17,4))
for ax,(title,a,b) in zip(axes,items):ax.bar(['Previous','Actuator+MLP'],[a,b],color=['0.55','tab:blue']);ax.set_title(title);ax.grid(axis='y',alpha=.25)
plot_path=OUT/'previous_vs_actuator_model.png'
fig.suptitle('Held-out 0808 (22:11:08), 1.0 s open-loop');fig.tight_layout();fig.savefig(plot_path,dpi=180)

if SHOW_PLOTS:
    if HAS_DISPLAY:
        plt.show(block=True)
        plt.close("all")
    else:
        plt.close(fig)
        print(f"SHOW_PLOTS=True, but no display is available; PNG saved to {plot_path}")
else:
    plt.close(fig)

print(json.dumps({n:{'previous':float(a),'new':float(b),'change_percent':float((b/a-1)*100)} for n,a,b in items},indent=2))
