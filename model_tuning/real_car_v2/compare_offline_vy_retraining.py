#!/usr/bin/env python3
"""Compare KF-vy-trained and offline-vy-trained Adam8D+vx-delta24D models."""
from pathlib import Path
import json, os, sys
import matplotlib.pyplot as plt
import numpy as np
import torch, yaml

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from compare_recursive_20d_vx_delta_24d import replay_new, metrics

DATA=Path(os.environ.get('OFFLINE_VY_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual_offline_vy_adam8d.npz'))
OLD_MODEL=Path(os.environ.get('OLD_VY_MODEL',ROOT/'model_tuning/results/adam8d_vx_delta24d_stage2/model.pt'))
OLD_PARAMS=Path(os.environ.get('OLD_VY_PARAMS',ROOT/'model_tuning/results/classic_adam_recursive_8d_0817_0818/params.json'))
NEW_MODEL=Path(os.environ.get('OFFLINE_VY_MODEL',ROOT/'model_tuning/results/offline_vy_adam8d_vxdelta24_stage2/model.pt'))
NEW_PARAMS=Path(os.environ.get('OFFLINE_VY_PARAMS',ROOT/'model_tuning/results/dynamic_regression_offline_vy/adam_params.json'))
OUT=Path(os.environ.get('OFFLINE_VY_COMPARISON_OUT',ROOT/'model_tuning/results/offline_vy_adam8d_vxdelta24_stage2/comparison'));DT=.04;H=30
NEW_LABEL=os.environ.get('OFFLINE_VY_MODEL_LABEL','offline_vy_trained_adam8d_vxdelta24d')
OLD_LABEL=os.environ.get('OLD_VY_MODEL_LABEL','previous_model')

def truth_trace(starts,source,teacher_vy,position_scale):
    result=[]
    for start in starts:
        pose=np.zeros(3);initial=source[start,:3].copy();initial[1]=teacher_vy[start];trace=[np.r_[pose,initial]]
        for row in range(start+2,start+2*H+1,2):
            state=source[row,:3].copy();state[1]=teacher_vy[row]
            yaw=pose[2];pose=np.array((pose[0]+position_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*DT,
                pose[1]+position_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*DT,yaw+state[2]*DT));trace.append(np.r_[pose,state])
        result.append(trace)
    return np.asarray(result)

def main():
    data=np.load(DATA);x=data['source_features'].astype(float);b=data['source_bag_id'];sp=data['source_split'];valid=data['source_valid']
    starts=np.asarray([i for i in range(10,len(x)-60) if sp[i]==2 and sp[i+60]==2 and valid[i-8:i+61].all() and np.all(b[i-8:i+61]==b[i])])[::5]
    cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];teacher=data['source_teacher_vy'] if 'source_teacher_vy' in data.files else x[:,1];truth=truth_trace(starts,x,teacher,float(cfg['kinematic_position_speed_scale']))
    old,old_acc=replay_new(starts,data,cfg,json.loads(OLD_PARAMS.read_text())['expanded_fitted'],torch.load(OLD_MODEL,map_location='cpu',weights_only=False))
    new,new_acc=replay_new(starts,data,cfg,json.loads(NEW_PARAMS.read_text())['expanded_fitted'],torch.load(NEW_MODEL,map_location='cpu',weights_only=False))
    obs=data['source_observations'];gt_acc=np.asarray([obs[start+2*np.arange(H),:2] for start in starts])
    report={'windows':len(starts),'ground_truth_vy':'offline MCL+IMU robust smoother','models':{
        OLD_LABEL:metrics(old,truth,old_acc,gt_acc),
        NEW_LABEL:metrics(new,truth,new_acc,gt_acc)}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    error=np.linalg.norm(new[:,-1,:2]-truth[:,-1,:2],axis=1);order=np.argsort(error);cases=(order[0],order[len(order)//2],order[-1]);t=np.arange(H+1)*DT
    fig,axes=plt.subplots(7,3,figsize=(17,24),constrained_layout=True);ta=np.arange(H)*DT
    for col,(label,index) in enumerate(zip(('best','median','worst'),cases)):
        axes[0,col].plot(truth[index,:,0],truth[index,:,1],'k-',label='offline-vy GT');axes[0,col].plot(old[index,:,0],old[index,:,1],'C1--',label=OLD_LABEL);axes[0,col].plot(new[index,:,0],new[index,:,1],'C0-',label=NEW_LABEL);axes[0,col].axis('equal');axes[0,col].set_title(f'{label} trajectory')
        for row,(channel,title,unit) in enumerate(((3,'vx','m/s'),(4,'vy','m/s'),(5,'yaw rate','rad/s'),(2,'yaw','rad')),1):
            axes[row,col].plot(t,truth[index,:,channel],'k-',label='GT');axes[row,col].plot(t,old[index,:,channel],'C1--',label=OLD_LABEL);axes[row,col].plot(t,new[index,:,channel],'C0-',label=NEW_LABEL);axes[row,col].set_title(title);axes[row,col].set_ylabel(unit)
        for row,(channel,title) in enumerate(((0,'ax'),(1,'ay')),5):
            axes[row,col].plot(ta,gt_acc[index,:,channel],'k-',label='GT IMU');axes[row,col].plot(ta,old_acc[index,:,channel],'C1--',label=OLD_LABEL);axes[row,col].plot(ta,new_acc[index,:,channel],'C0-',label=NEW_LABEL);axes[row,col].set_title(title);axes[row,col].set_ylabel('m/s²')
        for axis in axes[:,col]:axis.grid(alpha=.25);axis.legend(fontsize=7)
    fig.suptitle(f'{OLD_LABEL} vs {NEW_LABEL}: causal-KF input, offline-smoother teacher')
    fig.savefig(OUT/'best_median_worst.png',dpi=180);plt.close(fig)
    np.savez_compressed(OUT/'traces.npz',starts=starts,ground_truth=truth,
        kf_vy_trained=old,offline_vy_trained=new,baseline_20d=old,
        vx_delta_history_24d=new)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
