#!/usr/bin/env python3
"""Plot baseline versus early/dense-yaw recursive fine-tuning."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'model_tuning/results/dynamic_40ms_recursive_stage2_seed31'
NEW=ROOT/'model_tuning/results/dynamic_40ms_yawrate_dense_seed31'

def main():
 old=json.loads((OLD/'rollout_30step_metrics.json').read_text());new=json.loads((NEW/'rollout_30step_metrics.json').read_text())
 keys=('trajectory_m','yaw_rad','vx_mps','vy_mps','yaw_rate_rps');labels=('trajectory [m]','yaw [deg]','vx [m/s]','vy [m/s]','yaw-rate [rad/s]')
 fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
 for j,(key,label) in enumerate(zip(keys,labels)):
  vals=[]
  for q in (old,new):
   vals.append([q[s][key]['mean']*(180/np.pi if key=='yaw_rad' else 1) for s in ('validation','test_aggressive')])
  x=np.arange(2)
  for i,(name,color) in enumerate((('baseline','#4e79a7'),('early dense yaw-rate','#e15759'))):
   bars=ax.flat[j].bar(x+(i-.5)*.36,vals[i],.36,label=name,color=color);ax.flat[j].bar_label(bars,fmt='%.3f',fontsize=8)
  ax.flat[j].set_xticks(x,('validation','aggressive test'));ax.flat[j].set_title(label);ax.flat[j].grid(axis='y',alpha=.3);ax.flat[j].legend()
 ax.flat[5].axis('off');fig.suptitle('40 ms dynamic residual: early dense yaw-rate retraining (no explicit yaw loss)')
 fig.savefig(NEW/'yaw_dense_metrics_comparison.png',dpi=180);plt.close(fig)

 z0=np.load(ROOT/'model_tuning/results/effective_vs_dynamic_0813/aggressive_boundary_run1_dynamic40.npz');z1=np.load(NEW/'aggressive_run1.npz');g=z0['ground_truth'];p0=z0['predicted'];p1=z1['predicted'];idx=int(np.argmax(np.linalg.norm(p0[:,-1,:2]-g[:,-1,:2],axis=1)));t=np.arange(31)*.04
 fig,ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
 ax[0,0].plot(g[idx,:,0],g[idx,:,1],'k',lw=2,label='GT');ax[0,0].plot(p0[idx,:,0],p0[idx,:,1],'--',label='baseline');ax[0,0].plot(p1[idx,:,0],p1[idx,:,1],'--',label='early dense yaw-rate');ax[0,0].set_aspect('equal',adjustable='datalim');ax[0,0].set_title('Worst trajectory')
 for a,state,title,scale in ((ax[0,1],5,'yaw-rate [rad/s]',1),(ax[1,0],2,'yaw [deg]',180/np.pi),(ax[1,1],3,'vx [m/s]',1)):
  a.plot(t,g[idx,:,state]*scale,'k',lw=2,label='GT');a.plot(t,p0[idx,:,state]*scale,'--',label='baseline');a.plot(t,p1[idx,:,state]*scale,'--',label='early dense yaw-rate');a.set_title(title);a.set_xlabel('time [s]')
 for a in ax.flat:a.grid(alpha=.3);a.legend()
 fig.savefig(NEW/'yaw_dense_worst_comparison.png',dpi=180);plt.close(fig)
 print(NEW/'yaw_dense_metrics_comparison.png');print(NEW/'yaw_dense_worst_comparison.png')
if __name__=='__main__':main()
