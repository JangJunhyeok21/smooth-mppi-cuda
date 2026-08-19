#!/usr/bin/env python3
"""Plot held-out aggressive run2: effective, old lag, and retrained lag."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'model_tuning/results'
BASE=R/'effective_vs_dynamic_0813'
NEW=R/'dynamic_40ms_yaw_preserved_stage2'
OUT=NEW/'heldout_run2_effective_old_refined.png'

def load(path):
 z=np.load(path);return z['predicted'],z['ground_truth'],z['starts']
def final_error(p,g):return np.linalg.norm(p[:,-1,:2]-g[:,-1,:2],axis=1)
def main():
 ep,eg,es=load(BASE/'aggressive_boundary_run2_effective_recorded.npz')
 op,og,os=load(BASE/'aggressive_boundary_run2_latest40.npz')
 npred,ngt,ns=load(NEW/'rollout_ax_ungated_metrics.npz')
 models={'Effective':(ep,eg,es,24342),'Old lag residual':(op,og,os,0),'Retrained lag residual':(npred,ngt,ns,0)}
 err=final_error(npred,ngt);chosen=[np.argmin(err),np.argsort(err)[len(err)//2],np.argmax(err)]
 fig,axes=plt.subplots(4,3,figsize=(15,14),constrained_layout=True)
 for col,(idx,title) in enumerate(zip(chosen,('Best','Median','Worst'))):
  target_start=ns[idx]
  for label,(p,g,starts,offset) in models.items():
   j=int(np.argmin(np.abs(starts+offset-target_start)));axes[0,col].plot(p[j,:,0],p[j,:,1],label=label)
  axes[0,col].plot(ngt[idx,:,0],ngt[idx,:,1],'k--',lw=2,label='GT');axes[0,col].set_title(f'{title}: new error {err[idx]:.3f} m');axes[0,col].axis('equal');axes[0,col].grid(alpha=.25)
  t=np.arange(31)*.04
  for row,(state,name) in enumerate(((3,'vx [m/s]'),(4,'vy [m/s]'),(5,'yaw rate [rad/s]')),1):
   axes[row,col].plot(t,ngt[idx,:,state],'k--',lw=2,label='GT')
   for label,(p,g,starts,offset) in models.items():
    j=int(np.argmin(np.abs(starts+offset-target_start)));axes[row,col].plot(t,p[j,:,state],label=label)
   axes[row,col].set_ylabel(name);axes[row,col].set_xlabel('time [s]');axes[row,col].grid(alpha=.25)
 axes[0,0].legend(fontsize=8);fig.suptitle('Held-out aggressive run2: 1.2 s open-loop tail comparison',fontsize=15);fig.savefig(OUT,dpi=190);print(OUT)
if __name__=='__main__':main()
