#!/usr/bin/env python3
"""Plot kinematic-slip MLP velocity-residual ablation metrics."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
ENABLED_RESULT=ROOT/'model_tuning/results/kinematic_slip_noimu'
DISABLED_RESULT=ROOT/'model_tuning/results/kinematic_slip_noimu_no_vx_residual'
OUTPUT_PATH=ROOT/'model_tuning/results/kinematic_slip_vx_residual_ablation.png'

def main():
    p=argparse.ArgumentParser();p.add_argument('enabled',nargs='?',default=str(ENABLED_RESULT));p.add_argument('disabled',nargs='?',default=str(DISABLED_RESULT));p.add_argument('-o','--output',default=str(OUTPUT_PATH));a=p.parse_args()
    rows=[json.loads((Path(x)/'visualization/metrics.json').read_text()) for x in (a.enabled,a.disabled)]
    labels=('vx residual ON','vx residual OFF')
    keys=('test_1s_mean_m','test_1s_median_m','test_1s_p95_m','final_speed_mae_mps','final_yaw_rate_mae_radps','final_yaw_mae_rad')
    titles=('Trajectory mean [m]','Trajectory median [m]','Trajectory p95 [m]','Speed MAE [m/s]','Yaw-rate MAE [rad/s]','Yaw MAE [rad]')
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,key,title in zip(axes.flat,keys,titles):
        vals=[r[key] for r in rows];bars=ax.bar(labels,vals,color=('#2878b5','#e67e22'));ax.set_title(title);ax.grid(axis='y',alpha=.25)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.4f}',ha='center',va='bottom',fontsize=9)
    fig.suptitle('kinematic_slip_noimu: MLP vx-derivative residual ablation');fig.tight_layout()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=180);plt.close(fig)
if __name__=='__main__':main()
