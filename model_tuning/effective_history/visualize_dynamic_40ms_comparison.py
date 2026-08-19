#!/usr/bin/env python3
"""Plot recorded-command 1.2 s comparison for effective, old/new dynamic models."""
from pathlib import Path
import json,numpy as np,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'model_tuning/results/effective_vs_dynamic_0813';OUT=ROOT/'model_tuning/results/dynamic_40ms_comparison.png'
NAMES=('effective_speed20_run2','effective_speed25_run1','effective_speed30_run1','aggressive_boundary_run1','aggressive_boundary_run2');LABELS=('speed20','speed25','speed30','aggressive1','aggressive2')
def main():
 values={k:[] for k in ('effective','dynamic20','dynamic40')}
 for n in NAMES:
  e=json.loads((R/f'{n}_effective_recorded.json').read_text());o=json.loads((R/f'{n}_dynamic_v2.json').read_text());q=next(iter(json.loads((R/f'{n}_dynamic40.json').read_text()).values()))
  values['effective'].append((e['trajectory_mean_p95_max_m'][0],e['vx_mae_p95_max_mps'][0],e['yaw_rate_mae_p95_max_rps'][0]));values['dynamic20'].append((o['trajectory_mean_p95_max_m'][0],o['vx_mae_p95_max_mps'][0],o['yaw_rate_mae_p95_max_rps'][0]));values['dynamic40'].append((q['trajectory_m']['mean'],q['vx_mps']['mean'],q['yaw_rate_rps']['mean']))
 fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True);x=np.arange(len(NAMES));width=.25;titles=('1.2 s trajectory final error [m]','1.2 s final vx absolute error [m/s]','1.2 s final yaw-rate absolute error [rad/s]')
 for j,(name,color) in enumerate((('effective','tab:orange'),('dynamic20','tab:blue'),('dynamic40','tab:green'))):
  z=np.asarray(values[name]);
  for k,ax in enumerate(axes):ax.bar(x+(j-1)*width,z[:,k],width,label=name,color=color)
 for ax,title in zip(axes,titles):ax.set_ylabel('MAE');ax.set_title(title);ax.grid(axis='y',alpha=.25);ax.legend()
 axes[-1].set_xticks(x,LABELS);fig.tight_layout();fig.savefig(OUT,dpi=180);print(OUT)
if __name__=='__main__':main()
