#!/usr/bin/env python3
from pathlib import Path
import numpy as np,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'model_tuning/results/real_car_v2_dynamic_residual_all6_seed31/rollout_60step_predictions_kf_0p5.npz';NEW=ROOT/'model_tuning/results/dynamic_stable_rollout_residual_seed31/rollout_60step_predictions_kf_0p5.npz';OUT=ROOT/'model_tuning/results/dynamic_rollout_regression/model_comparison_best_median_worst.png';dt=.02;scale=.8633491306389823
def pose(q):
 p=np.zeros((len(q),q.shape[1],3))
 for k in range(q.shape[1]):
  z=p[:,k-1] if k else np.zeros((len(q),3));vx,vy,r=q[:,k].T;p[:,k,0]=z[:,0]+scale*(vx*np.cos(z[:,2])-vy*np.sin(z[:,2]))*dt;p[:,k,1]=z[:,1]+scale*(vx*np.sin(z[:,2])+vy*np.cos(z[:,2]))*dt;p[:,k,2]=z[:,2]+r*dt
 return p
def main():
 o=np.load(OLD);n=np.load(NEW);gt=n['gt'];gp=pose(gt);series=[('Old classic',o['physics'],'tab:gray'),('Old classic+MLP',o['residual'],'tab:blue'),('Rollout-fit classic',n['physics'],'tab:green'),('Rollout-fit classic+MLP',n['residual'],'tab:red')];poses=[(a,pose(q),c) for a,q,c in series];err=np.linalg.norm(poses[-1][1][:,-1,:2]-gp[:,-1,:2],axis=1);order=np.argsort(err);ids=[order[0],order[len(order)//2],order[-1]];t=np.arange(1,61)*dt;fig,ax=plt.subplots(3,4,figsize=(20,13),constrained_layout=True)
 for row,(idx,title) in enumerate(zip(ids,['Best','Median','Worst'])):
  ax[row,0].plot(gp[idx,:,0],gp[idx,:,1],'k',lw=2.5,label='GT')
  for name,p,c in poses:ax[row,0].plot(p[idx,:,0],p[idx,:,1],'--',color=c,label=name)
  ax[row,0].set_title(title+' trajectory');ax[row,0].axis('equal');ax[row,0].grid(alpha=.3);ax[row,0].legend(fontsize=7)
  for col,(s,label) in enumerate([(0,'vx [m/s]'),(1,'vy [m/s]'),(2,'yaw rate [rad/s]')],1):
   ax[row,col].plot(t,gt[idx,:,s],'k',lw=2,label='GT')
   for name,q,c in series:ax[row,col].plot(t,q[idx,:,s],'--',color=c,label=name)
   ax[row,col].set(xlabel='time [s]',ylabel=label);ax[row,col].grid(alpha=.3);ax[row,col].legend(fontsize=7)
 fig.suptitle('1.2 s test: one-step-fit vs recursive-rollout-fit Pacejka and residual MLP',fontsize=15);fig.savefig(OUT,dpi=180);print(OUT)
if __name__=='__main__':main()
