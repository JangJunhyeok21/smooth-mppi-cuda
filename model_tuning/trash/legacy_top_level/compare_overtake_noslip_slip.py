#!/usr/bin/env python3
"""Compare overtake legacy dynamics, no-slip MLP and slip MLP on identical windows."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from model_tuning_utils.train_rollout import prepare

DT=.02; LF=.163; LR=.161; WB=LF+LR
SA=.8954927921295166; SB=-.003672674298286438
ROOT=Path(__file__).resolve().parents[1]
DATASET_PATH=ROOT/'model_tuning/data/ifac0807_mppi_observation.npz'
OUTPUT_PATH=ROOT/'model_tuning/results/default_overtake_comparison'
NOSLIP_RESULT=ROOT/'model_tuning/results/ifac0807_kinematic_noimu_16d'
SLIP_RESULT=ROOT/'model_tuning/results/kinematic_slip_noimu';BATCH_SIZE=1024

class MLP(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)

def load_model(path,n,device):
    net=MLP(n).to(device);net.load_state_dict(torch.load(path/'model.pt',map_location=device,weights_only=True));net.eval()
    z=np.load(path/'normalization.npz')
    mean=np.r_[z['base_mean'],np.tile(z['command_mean'],5)].astype(np.float32)
    std=np.r_[z['base_std'],np.tile(z['command_std'],5)].astype(np.float32)
    return net,torch.tensor(mean,device=device),torch.tensor(std,device=device)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',nargs='?',default=str(DATASET_PATH));ap.add_argument('-o','--output',default=str(OUTPUT_PATH))
    ap.add_argument('--noslip-result',default=str(NOSLIP_RESULT))
    ap.add_argument('--slip-result',default=str(SLIP_RESULT))
    ap.add_argument('--batch-size',type=int,default=BATCH_SIZE);a=ap.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);z=np.load(a.dataset);raw=z['samples'];split=raw[:,10].astype(int)
    cfg=argparse.Namespace(pose_window=21,horizon=1.,history=50,max_pose_step=.25,min_speed=.7,max_speed=10.,max_beta=.7,max_omega=8.,impact_decel=-10.,impact_margin=.5,strict_no_imu=False)
    pose,polar,_,_,starts,dt,horizon=prepare(a.dataset,cfg);starts=starts[split[starts]==1]
    body=np.c_[polar[:,0]*np.cos(polar[:,1]),polar[:,0]*np.sin(polar[:,1]),polar[:,2]].astype(np.float32)
    cmd=raw[:,[7,9]].astype(np.float32);accel=raw[:,8].astype(np.float32);dev=torch.device('cpu')
    noslip,nm,ns=load_model(Path(a.noslip_result),16,dev);slip,sm,ss=load_model(Path(a.slip_result),18,dev)
    scale=torch.tensor([8.,8.,30.]);pred={k:[] for k in ('overtake_legacy','kinematic_noslip_noimu','kinematic_slip_noimu')}
    for q0 in range(0,len(starts),a.batch_size):
        ids=starts[q0:q0+a.batch_size];j0=ids+49;b=len(ids)
        gt0=torch.tensor(pose[j0],dtype=torch.float32);uall=torch.tensor(cmd,dtype=torch.float32)
        hist=torch.stack([uall[j0-d] for d in range(5,0,-1)],1)
        states={
          'overtake_legacy':torch.tensor(np.c_[polar[j0,0],polar[j0,1],polar[j0,2]],dtype=torch.float32),
          'kinematic_noslip_noimu':torch.tensor(np.c_[raw[j0,4],np.zeros(b),raw[j0,6]],dtype=torch.float32),
          'kinematic_slip_noimu':torch.tensor(body[j0],dtype=torch.float32)}
        poses={k:gt0.clone() for k in states};trajs={k:[gt0.clone()] for k in states}
        for k in range(horizon):
            ix=j0+k;u=uall[ix];steer=torch.clamp(SA*u[:,0]+SB,-.55,.55)
            # Exact high-speed update_dynamics from origin/feature/overtake-integration.
            s=states['overtake_legacy'];v,beta,w=s.T;vx=v*torch.cos(beta);vy=v*torch.sin(beta);rawsteer=u[:,0]
            af=rawsteer-torch.atan2(vy+LF*w,vx);ar=-torch.atan2(vy-LR*w,vx)
            fzf=3.74*9.81*LR/WB;fzr=3.74*9.81*LF/WB
            ff=fzf*2.0843*torch.sin(1.5*torch.atan(1.5*af));fr=fzr*1.9233*torch.sin(1.5*torch.atan(1.5*ar))
            dv=torch.tensor(accel[ix])*(1-.04*v);dw=(LF*ff*torch.cos(rawsteer)-LR*fr)/.04712
            db=(ff+fr)/(3.74*v)-w;sn=torch.stack((v+dv*dt,beta+db*dt,w+dw*dt),1)
            p=poses['overtake_legacy'];pn=torch.stack((p[:,0]+v*torch.cos(p[:,2]+beta)*dt,p[:,1]+v*torch.sin(p[:,2]+beta)*dt,p[:,2]+w*dt),1)
            states['overtake_legacy']=sn;poses['overtake_legacy']=pn;trajs['overtake_legacy'].append(pn)
            s=states['kinematic_noslip_noimu'];baseax=torch.clamp(8*(u[:,1]-s[:,0]),-8.,8.5);basev=s[:,0]+baseax*dt;basew=s[:,0]*torch.tan(steer)/WB
            f=torch.stack((s[:,0],s[:,2],u[:,0],u[:,1],basev,basew),1);corr=torch.tanh(noslip((torch.cat((f,hist.reshape(b,-1)),1)-nm)/ns))*scale
            sn=torch.stack((basev+corr[:,0]*dt,torch.zeros(b),basew+corr[:,2]*dt),1);p=poses['kinematic_noslip_noimu']
            pn=torch.stack((p[:,0]+sn[:,0]*torch.cos(p[:,2])*dt,p[:,1]+sn[:,0]*torch.sin(p[:,2])*dt,p[:,2]+sn[:,2]*dt),1)
            states['kinematic_noslip_noimu']=sn;poses['kinematic_noslip_noimu']=pn;trajs['kinematic_noslip_noimu'].append(pn)
            s=states['kinematic_slip_noimu'];speed=torch.hypot(s[:,0],s[:,1]);beta=torch.atan2(s[:,1],s[:,0]);baseax=torch.clamp(8*(u[:,1]-speed),-8.,8.5);basev=torch.clamp(speed+baseax*dt,0.,10.)
            classic=torch.stack((basev*torch.cos(beta),basev*torch.sin(beta),basev*torch.cos(beta)*torch.tan(steer)/WB),1)
            f=torch.cat((s,u,classic),1);corr=torch.tanh(slip((torch.cat((f,hist.reshape(b,-1)),1)-sm)/ss))*scale;sn=classic+corr*dt
            speedn=torch.hypot(sn[:,0],sn[:,1]);betan=torch.atan2(sn[:,1],sn[:,0]);p=poses['kinematic_slip_noimu']
            pn=torch.stack((p[:,0]+speedn*torch.cos(p[:,2]+betan)*dt,p[:,1]+speedn*torch.sin(p[:,2]+betan)*dt,p[:,2]+sn[:,2]*dt),1)
            states['kinematic_slip_noimu']=sn;poses['kinematic_slip_noimu']=pn;trajs['kinematic_slip_noimu'].append(pn)
            hist=torch.cat((hist[:,1:],u[:,None]),1)
        for name in pred:pred[name].append((torch.stack(trajs[name],1).detach().numpy(),states[name].detach().numpy()))
    gt=np.stack([pose[starts+49+k] for k in range(horizon+1)],1);gt_speed=polar[starts+49+horizon,0];gt_w=polar[starts+49+horizon,2]
    metrics={};arrays={"starts":starts,"gt_pose":gt}
    for name,chunks in pred.items():
        tr=np.concatenate([x[0] for x in chunks]);st=np.concatenate([x[1] for x in chunks]);e=np.linalg.norm(tr[:,-1,:2]-gt[:,-1,:2],axis=1)
        ps=st[:,0] if name=='overtake_legacy' else np.hypot(st[:,0],st[:,1]);pe=np.abs(ps-gt_speed);we=np.abs(st[:,2]-gt_w)
        yaw_delta=np.arctan2(np.sin(tr[:,-1,2]-gt[:,-1,2]),np.cos(tr[:,-1,2]-gt[:,-1,2]))
        metrics[name]={"windows":len(e),"trajectory_1s_mean_m":float(e.mean()),"trajectory_1s_median_m":float(np.median(e)),"trajectory_1s_p95_m":float(np.quantile(e,.95)),"trajectory_1s_worst_m":float(e.max()),"final_speed_mae_mps":float(pe.mean()),"final_yaw_rate_mae_radps":float(we.mean()),"final_yaw_mae_deg":float(np.degrees(np.mean(np.abs(yaw_delta))))}
        arrays[name+'_trajectory']=tr;arrays[name+'_position_error']=e
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez_compressed(out/'predictions.npz',**arrays)
    import matplotlib.pyplot as plt
    names=list(pred);labels=['Overtake legacy','No-slip kinematic+MLP','Slip kinematic+MLP'];x=np.arange(3)
    keys=('trajectory_1s_mean_m','trajectory_1s_p95_m','final_speed_mae_mps','final_yaw_rate_mae_radps','final_yaw_mae_deg')
    ylabels=('1 s trajectory mean [m]','1 s trajectory p95 [m]','Final speed MAE [m/s]','Final yaw-rate MAE [rad/s]','Final yaw MAE [deg]')
    fig,ax=plt.subplots(2,3,figsize=(16,8));ax.flat[-1].axis('off')
    for axis,key,ylabel in zip(ax.flat,keys,ylabels):
        axis.bar(x,[metrics[n][key] for n in names]);axis.set_xticks(x,labels,rotation=12);axis.set_ylabel(ylabel);axis.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'quantitative_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,12));titles=('Best','Median','Worst')
    for row,(name,label) in enumerate(zip(names,labels)):
        tr=arrays[name+'_trajectory'];err=arrays[name+'_position_error'];order=np.argsort(err);chosen=(order[0],order[len(order)//2],order[-1])
        for col,index in enumerate(chosen):
            axis=axes[row,col];axis.plot(gt[index,:,0],gt[index,:,1],'k-',lw=2,label='GT');axis.plot(tr[index,:,0],tr[index,:,1],'--',lw=2,label='Prediction')
            axis.set_title(f'{label} — {titles[col]} ({err[index]:.3f} m)');axis.axis('equal');axis.grid(alpha=.25)
            if row==0 and col==0:axis.legend()
    fig.tight_layout();fig.savefig(out/'best_median_worst_trajectories.png',dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
