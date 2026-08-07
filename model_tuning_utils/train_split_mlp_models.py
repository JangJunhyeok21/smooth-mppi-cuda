#!/usr/bin/env python3
"""Train deployment-oriented dynamic+IMU and kinematic+no-IMU MLP models.

Both models consume /drive speed as a speed setpoint.  No
(speed_command - measured_vx) / dt action reconstruction is used.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
try:
    from model_tuning_utils.train_rollout import prepare
except ModuleNotFoundError:
    from train_rollout import prepare


DT = .02
LF, LR = .163, .161
SA, SB = .8954927921295166, -.003672674298286438
ACCEL_SCALE = torch.tensor([8., 8., 30.])


class MLP(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 64), nn.SiLU(),
                                 nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def ema_by_segment(values, bag_id, alpha=.25):
    out = values.astype(np.float32).copy()
    for bid in np.unique(bag_id):
        ii = np.flatnonzero(bag_id == bid)
        for k in range(1, len(ii)):
            out[ii[k]] = alpha*out[ii[k]] + (1-alpha)*out[ii[k-1]]
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset');p.add_argument('-o','--output',required=True)
    p.add_argument('--model',choices=('dynamic_imu','kinematic_noimu'),required=True)
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--patience',type=int,default=30)
    p.add_argument('--seed',type=int,default=71);p.add_argument('--device',default='cuda')
    p.add_argument('--kp-speed',type=float,default=8.0)
    p.add_argument('--normalization',choices=('zscore','none'),default='zscore',
                   help='input feature transform; none feeds physical raw values directly')
    p.add_argument('--yaw-target',choices=('imu','odom'),default='imu',
                   help='kinematic_noimu supervision only; never changes inference inputs')
    p.add_argument('--imu-wz-sign',type=float,choices=(-1.,1.),default=1.)
    p.add_argument('--imu-ax-sign',type=float,choices=(-1.,1.),default=1.)
    p.add_argument('--imu-ay-sign',type=float,choices=(-1.,1.),default=1.)
    a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
    device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=argparse.Namespace(pose_window=21,horizon=1.,history=50,max_pose_step=.25,
        min_speed=.3,max_speed=10.,max_beta=.7,max_omega=8.,impact_decel=-10.,impact_margin=.5,
        strict_no_imu=(a.model=='kinematic_noimu' and a.yaw_target=='odom'))
    pose,polar,_,_,starts,dt,horizon=prepare(a.dataset,cfg)
    raw=np.load(a.dataset)['samples']; split=raw[:,10].astype(int);bag=raw[:,11].astype(int)
    body=np.c_[polar[:,0]*np.cos(polar[:,1]),polar[:,0]*np.sin(polar[:,1]),polar[:,2]].astype(np.float32)
    command=raw[:,[7,9]].astype(np.float32) # steer, /drive.speed setpoint
    if raw.shape[1]>=15:
        imu_raw=raw[:,12:15].astype(np.float32).copy()
    elif a.model=='kinematic_noimu' and a.yaw_target=='odom':
        imu_raw=np.zeros((len(raw),3),np.float32)
    else:
        raise SystemExit('selected model/target requires appended IMU columns')
    imu_raw*=np.array([a.imu_wz_sign,a.imu_ax_sign,a.imu_ay_sign],np.float32)
    imu=ema_by_segment(imu_raw,bag) # signed [wz, ax, ay]
    observed_body=body.copy();target_body=body.copy()
    if a.model=='kinematic_noimu':
        # Deployment input has no IMU/KF: odom vx and odom yaw-rate only.
        observed_body=np.c_[raw[:,4],np.zeros(len(raw)),raw[:,6]].astype(np.float32)
        # IMU is supervision only, never an input.  This is not inference leakage.
        target_yaw_rate=imu[:,0] if a.yaw_target=='imu' else raw[:,6]
        target_body=np.c_[raw[:,4],np.zeros(len(raw)),target_yaw_rate].astype(np.float32)
        if np.max(np.abs(observed_body[:,1]))>0 or np.max(np.abs(target_body[:,1]))>0:
            raise RuntimeError('kinematic_noimu vy input/target must be exactly zero')
        # Rebuild valid windows from odom/command/pose only.  The shared
        # prepare() function uses a KF for dynamic-model validity checks; a
        # strict no-IMU experiment must not let that KF select its samples.
        jump=np.r_[False,(bag[1:]!=bag[:-1]) |
                   (np.abs(np.diff(raw[:,0])-dt)>.5*dt) |
                   (np.hypot(np.diff(raw[:,1]),np.diff(raw[:,2]))>.25)]
        valid=(np.all(np.isfinite(np.c_[pose,observed_body,command]),axis=1) &
               (observed_body[:,0]>=.3) & (observed_body[:,0]<=10.) &
               (np.abs(observed_body[:,2])<8.) &
               (np.abs(command[:,0])<=.55) & (command[:,1]>=0.) &
               (command[:,1]<=12.) & ~jump)
        total=50+horizon+1
        good=np.convolve(valid.astype(np.int16),np.ones(total,dtype=np.int16),mode='valid')==total
        good &= bag[:len(good)]==bag[total-1:]
        starts=np.flatnonzero(good)
        if len(starts)<100:raise SystemExit('too few strict no-IMU windows')
    train_starts=starts[split[starts]==0];test_starts=starts[split[starts]==1]
    split_policy='source-bag-disjoint'
    if not len(train_starts) or not len(test_starts):
        # Explicit user-requested exception for a one-bag dataset. Metrics are
        # fit/reconstruction numbers, not unseen-bag generalization.
        train_starts=starts.copy();test_starts=starts.copy();split_policy='single-bag-identical-train-test'
    rng=np.random.default_rng(a.seed)
    if split_policy.startswith('single-bag'):
        # User-requested exception: fitting, validation/checkpoint selection,
        # and reported test all see the exact same valid-window set.
        fit_starts=train_starts.copy();val_starts=train_starts.copy()
    else:
        sh=rng.permutation(train_starts);nv=max(1,len(sh)//10)
        val_starts,fit_starts=sh[:nv],sh[nv:]
    # Normalization uses measured train states and commands only.  Repeated command
    # history has the same normalization as the current command.
    train_rows=(np.arange(len(raw)) if split_policy.startswith('single-bag')
                else np.flatnonzero(split==0))
    if a.model=='kinematic_noimu':
        # Initial vy is unobserved and forced to zero, but use the train-only
        # physical vy spread for normalization because predicted vy is fed back.
        base=np.c_[target_body,command,target_body[:,0],np.zeros(len(body)),target_body[:,2]]
    else:
        # state vx,vy,r,ax,ay; command; base ax,ay,r
        base_ax=np.clip(a.kp_speed*(command[:,1]-body[:,0]),-8,8.5)
        steer=np.clip(SA*command[:,0]+SB,-.55,.55)
        base_r=body[:,0]*np.tan(steer)/(LF+LR)
        base=np.c_[body,imu[:,1:3],command,base_ax,body[:,0]*base_r,base_r]
    mean=base[train_rows].mean(0).astype(np.float32);std=base[train_rows].std(0).clip(1e-4).astype(np.float32)
    cmd_mean=command[train_rows].mean(0).astype(np.float32);cmd_std=command[train_rows].std(0).clip(1e-4).astype(np.float32)
    if a.normalization=='none':
        mean=np.zeros_like(mean);std=np.ones_like(std)
        cmd_mean=np.zeros_like(cmd_mean);cmd_std=np.ones_like(cmd_std)
    nbase=8 if a.model=='kinematic_noimu' else 10
    net=MLP(nbase+10).to(device);opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
    mean_t=torch.tensor(mean,device=device);std_t=torch.tensor(std,device=device)
    cm_t=torch.tensor(cmd_mean,device=device);cs_t=torch.tensor(cmd_std,device=device)
    body_t=torch.tensor(target_body,device=device);observed_body_t=torch.tensor(observed_body,device=device)
    cmd_t=torch.tensor(command,device=device);imu_t=torch.tensor(imu,device=device)
    pose_t=torch.tensor(pose,dtype=torch.float32,device=device);scale=ACCEL_SCALE.to(device)

    def rollout(ids,training=False,collect=False):
        ids=np.asarray(ids);j0=ids+49;b=len(ids)
        s=(observed_body_t[j0].clone() if a.model=='kinematic_noimu'
           else body_t[j0].clone());q=pose_t[j0].clone()
        # Dynamic IMU is an initial condition. Future ax/ay are predicted below.
        axay=imu_t[j0,1:3].clone()
        hist=torch.stack([cmd_t[j0-d] for d in range(5,0,-1)],1)
        losses=[];traj=[torch.cat((q,s,axay),1)] if collect else None
        for k in range(horizon):
            ix=j0+k;u=cmd_t[ix];steer=torch.clamp(SA*u[:,0]+SB,-.55,.55)
            base_ax=torch.clamp(a.kp_speed*(u[:,1]-s[:,0]),-8.,8.5)
            base_r=s[:,0]*torch.tan(steer)/(LF+LR)
            if a.model=='kinematic_noimu':
                classic=torch.stack((s[:,0]+base_ax*dt,torch.zeros(b,device=device),base_r),1)
                fbase=torch.cat((s,u,classic),1)
                pred=torch.tanh(net(torch.cat(((fbase-mean_t)/std_t,
                    ((hist-cm_t)/cs_t).reshape(b,-1)),1)))*scale
                # Strict no-slip/no-IMU deployment: vy is neither measured nor
                # estimated, so do not feed a learned pseudo-vy back.
                ns=torch.stack((classic[:,0]+pred[:,0]*dt,
                                torch.zeros(b,device=device),
                                classic[:,2]+pred[:,2]*dt),1)
                next_axay=torch.stack(((ns[:,0]-s[:,0])/dt,
                    (ns[:,1]-s[:,1])/dt+ns[:,0]*ns[:,2]),1)
            else:
                base_ay=s[:,0]*base_r
                fbase=torch.cat((s,axay,u,base_ax[:,None],base_ay[:,None],base_r[:,None]),1)
                # Outputs next [ax, ay, yaw_rate], recursively reused at k+1.
                pred=torch.tanh(net(torch.cat(((fbase-mean_t)/std_t,
                    ((hist-cm_t)/cs_t).reshape(b,-1)),1)))*scale
                next_axay=pred[:,:2];nr=pred[:,2]
                nvx=s[:,0]+(next_axay[:,0]+s[:,1]*s[:,2])*dt
                nvy=s[:,1]+(next_axay[:,1]-s[:,0]*s[:,2])*dt
                ns=torch.stack((nvx,nvy,nr),1)
            nq=torch.stack((q[:,0]+(ns[:,0]*torch.cos(q[:,2])-ns[:,1]*torch.sin(q[:,2]))*dt,
                            q[:,1]+(ns[:,0]*torch.sin(q[:,2])+ns[:,1]*torch.cos(q[:,2]))*dt,
                            q[:,2]+ns[:,2]*dt),1)
            gt_s=body_t[ix+1];gt_q=pose_t[ix+1];gt_imu=imu_t[ix+1,1:3]
            state_loss=.8*torch.nn.functional.smooth_l1_loss(ns,gt_s,reduction='none').mean(1)
            yaw_loss=.4*torch.nn.functional.smooth_l1_loss(nq[:,2],gt_q[:,2],reduction='none')
            pos_loss=.15*torch.linalg.vector_norm(nq[:,:2]-gt_q[:,:2],dim=1)
            imu_loss=(.04*torch.nn.functional.smooth_l1_loss(next_axay,gt_imu,reduction='none').mean(1)
                      if a.model=='dynamic_imu' else 0.)
            losses.append(state_loss+yaw_loss+pos_loss+imu_loss)
            s,q,axay=ns,nq,next_axay;hist=torch.cat((hist[:,1:],u[:,None]),1)
            if collect:traj.append(torch.cat((q,s,axay),1))
        loss=torch.stack(losses,1).mean()
        return loss,(torch.stack(traj,1) if collect else None)

    best=float('inf');bad=0;history=[]
    for epoch in range(1,a.epochs+1):
        net.train();order=rng.permutation(fit_starts);total=0.;count=0;t0=time.time()
        for begin in range(0,len(order),a.batch_size):
            ids=order[begin:begin+a.batch_size];opt.zero_grad(set_to_none=True)
            loss,_=rollout(ids,True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(ids);count+=len(ids)
        net.eval()
        vloss=0.;vpos=0
        with torch.no_grad():
            for i in range(0,len(val_starts),a.batch_size):
                vi=val_starts[i:i+a.batch_size];vl,vt=rollout(vi,collect=True)
                jj=vi+49+horizon
                vloss+=float(vl)*len(vi)
                vpos+=float(torch.linalg.vector_norm(vt[:,-1,:2]-pose_t[jj,:2],dim=1).sum())
        vl=vloss/len(val_starts);vp=vpos/len(val_starts)
        tr=total/count;history.append((epoch,tr,vl,vp,time.time()-t0));print(epoch,tr,vl,vp,flush=True)
        # Select without touching held-out test bags.  The deployment objective
        # requested by this project is final 1 s trajectory distance.
        if vp<best-1e-5:best=vp;bad=0;torch.save(net.state_dict(),out/'model.pt')
        else:bad+=1
        if bad>=a.patience:break
    net.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True));net.eval()
    alltraj=[]
    with torch.no_grad():
        for begin in range(0,len(test_starts),a.batch_size):alltraj.append(rollout(test_starts[begin:begin+a.batch_size],collect=True)[1].cpu().numpy())
    traj=np.concatenate(alltraj);j0=test_starts+49
    gtpose=np.stack([pose[j0+k] for k in range(horizon+1)],1);gtstate=np.stack([target_body[j0+k] for k in range(horizon+1)],1)
    pe=np.linalg.norm(traj[:,:,:2]-gtpose[:,:,:2],axis=2);se=np.abs(traj[:,:,3]-gtstate[:,:,0]);we=np.abs(traj[:,:,5]-gtstate[:,:,2])
    metrics={'model':a.model,'action':'[steer_cmd, /drive.speed setpoint]','test_windows':len(test_starts),
      'split_policy':split_policy,'test_is_unseen_data':not split_policy.startswith('single-bag'),
      'yaw_rate_target':(a.yaw_target if a.model=='kinematic_noimu' else 'recursive_imu'),
      'imu_axis_signs':{'wz':a.imu_wz_sign,'ax':a.imu_ax_sign,'ay':a.imu_ay_sign},
      'kinematic_input_vy_max_abs':(float(np.max(np.abs(observed_body[:,1]))) if a.model=='kinematic_noimu' else None),
      'input_normalization':a.normalization,
      'test_1s_mean_m':float(pe[:,-1].mean()),'test_1s_median_m':float(np.median(pe[:,-1])),'test_1s_p95_m':float(np.quantile(pe[:,-1],.95)),
      'test_1s_worst_m':float(pe[:,-1].max()),'final_speed_mae_mps':float(se[:,-1].mean()),'final_yaw_rate_mae_radps':float(we[:,-1].mean()),
      'kp_speed':a.kp_speed,'dt':dt,'epochs':len(history),
      'best_validation_loss':(best if np.isfinite(best) else None),
      'dynamic_imu_policy':('measured EMA ax/ay only at horizon start; predicted ax/ay recursively thereafter' if a.model=='dynamic_imu' else None)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez_compressed(out/'test_predictions.npz',starts=test_starts,prediction=traj,gt_pose=gtpose,gt_state=gtstate,position_error=pe,speed_error=se,yaw_rate_error=we)
    np.savez(out/'normalization.npz',base_mean=mean,base_std=std,command_mean=cmd_mean,command_std=cmd_std)
    with (out/'history.csv').open('w',newline='') as f: w=csv.writer(f);w.writerow(('epoch','train','validation','validation_final_position_m','seconds'));w.writerows(history)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
