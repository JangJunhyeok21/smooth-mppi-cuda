#!/usr/bin/env python3
"""F5로 동일 held-out windows에서 simulator GRU와 기존 모델을 비교한다."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];C=ROOT/'model_tuning/results/compare_0817_0820_inertial_ekf_bias';G=ROOT/'model_tuning/results/simulator_gru_0817_0820_seed31';E=ROOT/'model_tuning/results/e2e_0817_0820_inertial_ekf_bias_seed31'
def stat(v):return {'mean':float(np.mean(v)),'p95':float(np.quantile(v,.95)),'max':float(np.max(v))}
def main():
 files={'deployed residual':C/'deployed_lateral_only.npz','new EKF residual':C/'new_lateral_only.npz','small E2E':E/'rollout_60step_metrics.npz','simulator GRU':G/'rollout_60step_metrics.npz'};data={k:np.load(v) for k,v in files.items()};common=set(data['simulator GRU']['starts'].tolist())
 for z in data.values():common&=set(z['starts'].tolist())
 common=np.asarray(sorted(common));report={'common_windows':len(common),'horizon_s':1.2,'models':{}}
 for name,z in data.items():
  lookup={int(v):i for i,v in enumerate(z['starts'])};idx=np.asarray([lookup[int(v)] for v in common]);p=z['predicted'][idx,-1];g=z['ground_truth'][idx,-1];report['models'][name]={'trajectory_m':stat(np.linalg.norm(p[:,:2]-g[:,:2],axis=1)),'yaw_rad':stat(np.abs((p[:,2]-g[:,2]+np.pi)%(2*np.pi)-np.pi)),'vx_mps':stat(abs(p[:,3]-g[:,3])),'vy_mps':stat(abs(p[:,4]-g[:,4])),'yaw_rate_rps':stat(abs(p[:,5]-g[:,5]))}
 out=C/'simulator_gru_common_window_comparison.json';out.write_text(json.dumps(report,indent=2)+'\n');names=list(report['models']);x=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(15,5.5));width=.24
 for j,q in enumerate(('mean','p95','max')):axes[0].bar(x+(j-1)*width,[report['models'][n]['trajectory_m'][q] for n in names],width,label=q)
 axes[0].set_xticks(x,names,rotation=12);axes[0].set_ylabel('endpoint trajectory error [m]');axes[0].grid(axis='y',alpha=.3);axes[0].legend()
 keys=('vx_mps','vy_mps','yaw_rate_rps');sx=np.arange(3)
 for j,n in enumerate(names):axes[1].bar(sx+(j-1.5)*.19,[report['models'][n][k]['mean'] for k in keys],.19,label=n)
 axes[1].set_xticks(sx,('vx','vy','yaw-rate'));axes[1].set_ylabel('endpoint mean absolute error');axes[1].grid(axis='y',alpha=.3);axes[1].legend(fontsize=8);fig.suptitle(f'Same {len(common)} held-out windows, 1.2 s');fig.tight_layout();png=C/'simulator_gru_common_window_comparison.png';fig.savefig(png,dpi=180);print(png);print(out);plt.show()
if __name__=='__main__':main()
