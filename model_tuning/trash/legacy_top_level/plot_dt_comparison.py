#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DT002_RESULT=ROOT/'model_tuning/results/ifac0807_kinematic_slip_noimu_balanced'
DT0035_RESULT=ROOT/'model_tuning/results/ifac0807_kinematic_slip_noimu_dt0035'
OUTPUT_PATH=ROOT/'model_tuning/results/ifac0807_slip_dt_comparison.png'
def main():
 p=argparse.ArgumentParser();p.add_argument('dt002',nargs='?',default=str(DT002_RESULT));p.add_argument('dt0035',nargs='?',default=str(DT0035_RESULT));p.add_argument('-o','--output',default=str(OUTPUT_PATH));a=p.parse_args()
 rows=[json.loads((Path(x)/'visualization/metrics.json').read_text()) for x in (a.dt002,a.dt0035)];labels=('dt=0.020 s','dt=0.035 s')
 keys=('test_1s_mean_m','test_1s_median_m','test_1s_p95_m','final_speed_mae_mps','final_yaw_rate_mae_radps','final_yaw_mae_rad');titles=('Trajectory mean [m]','Trajectory median [m]','Trajectory p95 [m]','Speed MAE [m/s]','Yaw-rate MAE [rad/s]','Yaw MAE [rad]')
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for ax,key,title in zip(axes.flat,keys,titles):
  vals=[r[key] for r in rows];bars=ax.bar(labels,vals,color=('#2878b5','#e67e22'));ax.set_title(title);ax.grid(axis='y',alpha=.25)
  for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v,f'{v:.4f}',ha='center',va='bottom')
 fig.suptitle('IFAC0807 kinematic_slip_noimu dt comparison');fig.tight_layout();fig.savefig(a.output,dpi=180);plt.close(fig)
if __name__=='__main__':main()
