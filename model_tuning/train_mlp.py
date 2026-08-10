#!/usr/bin/env python3
"""Train deployment-oriented dynamic+IMU and kinematic+no-slip+no-IMU MLP models.

Both models consume /drive speed as a speed setpoint.  No
(speed_command - measured_vx) / dt action reconstruction is used.
"""
import argparse, csv, json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from model_tuning_utils.train_rollout import prepare


DT = .02
LF, LR = .163, .161
SA, SB = .8954927921295166, -.003672674298286438
ACCEL_SCALE = torch.tensor([8., 8., 30.])
# 0807 /newmcl_pose offline dynamic regression (N, SI units).
DYNAMIC_PARAMS = dict(Bf=3.7417837566480756,Cf=1.57976538150357,Df=.24887250653725695,Ef=-1.,
                      Br=3.302764006817267,Cr=1.9942488896242827,Dr=.4480617877807337,Er=-1.,
                      Iz=.4999999998525169,mass=3.74)

# USER SETTINGS — `python3 model_tuning/train_mlp.py` uses these values.
DEFAULTS = dict(
    dataset="model_tuning/data/ifac0807_0808_hardcase_train_test.npz",
    output="model_tuning/results/ifac0807_0808_kf_cf12p72_cr75p09_yaw_curriculum",
    model="slip_kinematic_with_imu", epochs=160, batch_size=512, lr=2e-3,
    patience=30, seed=71, device="cuda", kp_speed=8.0, rollout_horizon=1.0,
    runtime_min_speed=.5, runtime_max_speed=4.0, position_speed_scale=1.0,
    # Set to a JSON path only when explicitly desired; otherwise the scalar
    # above is the single source of truth shared with params.yaml.
    position_speed_scale_json=None,
    state_loss_weight=2.0, vx_loss_weight=2.0, vy_loss_weight=2.0,
    state_yaw_rate_loss_weight=8.0, position_loss_weight=1.0,
    yaw_loss_weight=8.0, yaw_rate_loss_weight=8.0,
    checkpoint_objective="position", resume=None, normalization="zscore",
    disable_velocity_residual=False, yaw_target="imu", slip_yaw_source="imu",
    imu_wz_sign=-1.0, imu_ax_sign=1.0, imu_ay_sign=-1.0,
    kf_cf=12.7222491, kf_cr=75.0944752, history=6,
    steer_time_constant=.18397091, max_steer_rate=.8791163,
    steer_scale=.50927964, steer_bias=.01015773,
    balanced_sampling=True, curriculum=True,
)


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
    p.add_argument('dataset',nargs='?',default=DEFAULTS['dataset']);p.add_argument('-o','--output',default=DEFAULTS['output'])
    p.add_argument('--model',choices=('dynamic_imu','e2e_mlp','dynamic_residual','kinematic_noslip_noimu','slip_kinematic_with_imu'),default=DEFAULTS['model'])
    p.add_argument('--epochs',type=int,default=160);p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--lr',type=float,default=2e-3);p.add_argument('--patience',type=int,default=30)
    p.add_argument('--seed',type=int,default=71);p.add_argument('--device',default='cuda')
    p.add_argument('--kp-speed',type=float,default=8.0)
    p.add_argument('--rollout-horizon',type=float,default=1.0,
                   help='recursive training/evaluation horizon in seconds')
    p.add_argument('--runtime-min-speed',type=float,default=.5)
    p.add_argument('--runtime-max-speed',type=float,default=3.0)
    p.add_argument('--position-speed-scale',type=float,default=1.0,
                   help='odom/VESC speed to physical map displacement scale')
    p.add_argument('--position-speed-scale-json',
                   help='JSON from regress_position_speed_scale.py; overrides --position-speed-scale')
    p.add_argument('--state-loss-weight',type=float,default=.8)
    p.add_argument('--vx-loss-weight',type=float,default=1.0,
                   help='relative vx component weight inside state loss')
    p.add_argument('--vy-loss-weight',type=float,default=1.0,
                   help='relative vy component weight inside state loss')
    p.add_argument('--state-yaw-rate-loss-weight',type=float,default=1.0,
                   help='relative yaw-rate component weight inside state loss')
    p.add_argument('--position-loss-weight',type=float,default=.15)
    p.add_argument('--yaw-loss-weight',type=float,default=.4)
    p.add_argument('--yaw-rate-loss-weight',type=float,default=0.0,
                   help='extra direct supervision for recursive yaw-rate state')
    p.add_argument('--checkpoint-objective',choices=('position','position_p95','loss'),default='position')
    p.add_argument('--resume',help='optional model.pt used to continue training')
    p.add_argument('--normalization',choices=('zscore','none'),default='zscore',
                   help='input feature transform; none feeds physical raw values directly')
    p.add_argument('--disable-velocity-residual',action='store_true',
                   help='kinematic models: use classic base_vx without the first MLP correction')
    p.add_argument('--yaw-target',choices=('imu','odom'),default='imu',
                   help='kinematic_noslip_noimu supervision only; never changes inference inputs')
    p.add_argument('--slip-yaw-source',choices=('imu','kf','odom'),default='imu',
                   help='kinematic_slip state yaw-rate; imu matches the runtime observable measurement')
    p.add_argument('--imu-wz-sign',type=float,choices=(-1.,1.),default=1.)
    p.add_argument('--imu-ax-sign',type=float,choices=(-1.,1.),default=1.)
    p.add_argument('--imu-ay-sign',type=float,choices=(-1.,1.),default=1.)
    p.add_argument('--kf-cf',type=float,default=12.7222491,
                   help='linear-KF front cornering stiffness [N/rad]')
    p.add_argument('--kf-cr',type=float,default=75.0944752,
                   help='linear-KF rear cornering stiffness [N/rad]')
    p.add_argument('--history',type=int,default=6,help='contiguous samples required before rollout')
    p.add_argument('--steer-time-constant',type=float,default=.08)
    p.add_argument('--max-steer-rate',type=float,default=6.0)
    p.add_argument('--steer-scale',type=float,default=SA)
    p.add_argument('--steer-bias',type=float,default=SB)
    p.add_argument('--balanced-sampling',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--curriculum',action=argparse.BooleanOptionalAction,default=True)
    p.set_defaults(**DEFAULTS)
    a=p.parse_args()
    if a.position_speed_scale_json:
        scale_result=json.loads(Path(a.position_speed_scale_json).read_text())
        a.position_speed_scale=float(scale_result['position_speed_scale'])
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    is_noslip=a.model=='kinematic_noslip_noimu';is_slip=a.model=='slip_kinematic_with_imu';is_dynamic_residual=a.model=='dynamic_residual';is_e2e=a.model in ('dynamic_imu','e2e_mlp')
    is_kinematic=is_noslip or is_slip
    device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    cfg=argparse.Namespace(pose_window=21,horizon=a.rollout_horizon,history=a.history,max_pose_step=.25,
        min_speed=.3,max_speed=10.,max_beta=.7,max_omega=8.,impact_decel=-10.,impact_margin=.5,
        strict_no_imu=(is_noslip and a.yaw_target=='odom'),
        imu_wz_sign=a.imu_wz_sign,imu_ay_sign=a.imu_ay_sign,
        kf_cornering_stiffness_front=a.kf_cf,
        kf_cornering_stiffness_rear=a.kf_cr)
    pose,polar,_,_,starts,dt,horizon=prepare(a.dataset,cfg)
    raw=np.load(a.dataset)['samples']; split=raw[:,10].astype(int);bag=raw[:,11].astype(int)
    body=np.c_[polar[:,0]*np.cos(polar[:,1]),polar[:,0]*np.sin(polar[:,1]),polar[:,2]].astype(np.float32)
    command=raw[:,[7,9]].astype(np.float32) # steer, /drive.speed setpoint
    command_delta=np.r_[0.,np.diff(command[:,0])].astype(np.float32)
    effective_steer=np.empty(len(raw),np.float32)
    for bid in np.unique(bag):
        ii=np.flatnonzero(bag==bid);effective_steer[ii[0]]=a.steer_scale*command[ii[0],0]+a.steer_bias
        for k in range(1,len(ii)):
            target=np.clip(a.steer_scale*command[ii[k],0]+a.steer_bias,-.55,.55)
            rate=np.clip((target-effective_steer[ii[k-1]])/a.steer_time_constant,
                         -a.max_steer_rate,a.max_steer_rate)
            effective_steer[ii[k]]=effective_steer[ii[k-1]]+rate*dt
    if raw.shape[1]>=15:
        imu_raw=raw[:,12:15].astype(np.float32).copy()
    elif is_noslip and a.yaw_target=='odom':
        imu_raw=np.zeros((len(raw),3),np.float32)
    else:
        raise SystemExit('selected model/target requires appended IMU columns')
    imu_raw*=np.array([a.imu_wz_sign,a.imu_ax_sign,a.imu_ay_sign],np.float32)
    imu=ema_by_segment(imu_raw,bag) # signed [wz, ax, ay]
    observed_body=body.copy();target_body=body.copy()
    if is_noslip:
        # Deployment input has no IMU/KF: odom vx and odom yaw-rate only.
        observed_body=np.c_[raw[:,4],np.zeros(len(raw)),raw[:,6]].astype(np.float32)
        # IMU is supervision only, never an input.  This is not inference leakage.
        target_yaw_rate=imu[:,0] if a.yaw_target=='imu' else raw[:,6]
        target_body=np.c_[raw[:,4],np.zeros(len(raw)),target_yaw_rate].astype(np.float32)
        if np.max(np.abs(observed_body[:,1]))>0 or np.max(np.abs(target_body[:,1]))>0:
            raise RuntimeError('kinematic_noslip_noimu vy input/target must be exactly zero')
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
        total=a.history+horizon+1
        good=np.convolve(valid.astype(np.int16),np.ones(total,dtype=np.int16),mode='valid')==total
        good &= bag[:len(good)]==bag[total-1:]
        starts=np.flatnonzero(good)
        if len(starts)<100:raise SystemExit('too few strict no-IMU windows')
    elif is_slip:
        # KF is needed because vy is not measured.  Do not also replace the
        # observable yaw-rate with the KF's coupled state: in real turns that
        # estimate was about 2x too small. Runtime uses the same signed/causal
        # EMA IMU yaw-rate, with odom available as an explicit fallback.
        slip_w={'imu':imu[:,0], 'kf':body[:,2], 'odom':raw[:,6]}[a.slip_yaw_source]
        observed_body=np.c_[body[:,0],body[:,1],slip_w].astype(np.float32)
        target_body=observed_body.copy()
    elif is_dynamic_residual or is_e2e:
        # Runtime state uses KF vy and directly observable signed IMU yaw-rate.
        observed_body=np.c_[body[:,0],body[:,1],imu[:,0]].astype(np.float32)
        target_body=observed_body.copy()
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
    if is_noslip:
        # Initial vy is unobserved and forced to zero, but use the train-only
        # physical vy spread for normalization because predicted vy is fed back.
        # 16-D no-IMU feature base: [v, yaw_rate, steer_cmd, speed_cmd,
        # classic_next_v, classic_next_yaw_rate]. Current/classic vy were
        # constant zeros and are intentionally removed.
        base=np.c_[target_body[:,[0,2]],command,target_body[:,0],target_body[:,2]]
    elif is_slip or is_dynamic_residual:
        steer=effective_steer
        measured_speed=np.hypot(target_body[:,0],target_body[:,1])
        measured_beta=np.arctan2(target_body[:,1],target_body[:,0])
        base_ax=np.clip(a.kp_speed*(command[:,1]-measured_speed),-8,8.5)
        base_speed=np.clip(measured_speed+base_ax*dt,
                           np.minimum(a.runtime_min_speed,measured_speed),
                           np.maximum(a.runtime_max_speed,measured_speed))
        base_vx=base_speed*np.cos(measured_beta);base_vy=base_speed*np.sin(measured_beta)
        if is_dynamic_residual:
            vx_safe=np.maximum(np.abs(target_body[:,0]),.5);vy0=target_body[:,1];w0=target_body[:,2]
            af=steer-np.arctan2(vy0+LF*w0,vx_safe);ar=-np.arctan2(vy0-LR*w0,vx_safe)
            dp=DYNAMIC_PARAMS;fzf=dp['mass']*9.81*LR/(LF+LR);fzr=dp['mass']*9.81*LF/(LF+LR)
            fyf=fzf*dp['Df']*np.sin(dp['Cf']*np.arctan(dp['Bf']*af-dp['Ef']*(dp['Bf']*af-np.arctan(dp['Bf']*af))))
            fyr=fzr*dp['Dr']*np.sin(dp['Cr']*np.arctan(dp['Br']*ar-dp['Er']*(dp['Br']*ar-np.arctan(dp['Br']*ar))))
            beta_dot=(fyf*np.cos(steer)+fyr)/(dp['mass']*np.maximum(measured_speed,.5))-w0
            next_beta=measured_beta+beta_dot*dt;base_vx=base_speed*np.cos(next_beta);base_vy=base_speed*np.sin(next_beta)
            base_r=w0+(LF*fyf*np.cos(steer)-LR*fyr)/dp['Iz']*dt
        else: base_r=base_vx*np.tan(steer)/(LF+LR)
        base=np.c_[target_body,command,effective_steer,command_delta,base_vx,base_vy,base_r]
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
    nbase=6 if is_noslip else 10
    net=MLP(nbase+10).to(device)
    if a.resume: net.load_state_dict(torch.load(a.resume,map_location=device,weights_only=True))
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-5)
    mean_t=torch.tensor(mean,device=device);std_t=torch.tensor(std,device=device)
    cm_t=torch.tensor(cmd_mean,device=device);cs_t=torch.tensor(cmd_std,device=device)
    target_body_t=torch.tensor(target_body,device=device)
    # Compatibility alias for preprocessing/statistics code below.
    body_t=target_body_t
    observed_body_t=torch.tensor(observed_body,device=device)
    cmd_t=torch.tensor(command,device=device);imu_t=torch.tensor(imu,device=device)
    pose_t=torch.tensor(pose,dtype=torch.float32,device=device);scale=ACCEL_SCALE.to(device)
    state_component_weights=torch.tensor(
        [a.vx_loss_weight,a.vy_loss_weight,a.state_yaw_rate_loss_weight],
        device=device)

    history_offset=a.history-1
    def rollout(ids,training=False,collect=False,steps=None):
        ids=np.asarray(ids);j0=ids+history_offset;b=len(ids);steps=horizon if steps is None else steps
        # MPPI starts every solve from its observable state: odom vx, KF vy,
        # and signed/causally filtered IMU yaw-rate for IMU models.
        s=observed_body_t[j0].clone();q=pose_t[j0].clone()
        # Dynamic IMU is an initial condition. Future ax/ay are predicted below.
        axay=imu_t[j0,1:3].clone()
        hist=torch.stack([cmd_t[j0-d] for d in range(5,0,-1)],1)
        delta=torch.tensor(effective_steer[j0],device=device)
        losses=[];traj=[torch.cat((q,s,axay),1)] if collect else None
        for k in range(steps):
            ix=j0+k;u=cmd_t[ix];steer_target=torch.clamp(a.steer_scale*u[:,0]+a.steer_bias,-.55,.55)
            delta=delta+torch.clamp((steer_target-delta)/a.steer_time_constant,
                                    -a.max_steer_rate,a.max_steer_rate)*dt
            steer=delta
            base_ax=torch.clamp(a.kp_speed*(u[:,1]-s[:,0]),-8.,8.5)
            base_r=s[:,0]*torch.tan(steer)/(LF+LR)
            if is_noslip:
                unclamped=s[:,0]+base_ax*dt
                base_v=torch.minimum(torch.maximum(unclamped,
                    torch.minimum(torch.full_like(s[:,0],a.runtime_min_speed),s[:,0])),
                    torch.maximum(torch.full_like(s[:,0],a.runtime_max_speed),s[:,0]))
                classic=torch.stack((base_v,torch.zeros(b,device=device),base_r),1)
                fbase=torch.stack((s[:,0],s[:,2],u[:,0],u[:,1],
                                   classic[:,0],classic[:,2]),1)
                pred=torch.tanh(net(torch.cat(((fbase-mean_t)/std_t,
                    ((hist-cm_t)/cs_t).reshape(b,-1)),1)))*scale
                # Strict no-slip/no-IMU deployment: vy is neither measured nor
                # estimated, so do not feed a learned pseudo-vy back.
                next_v=(classic[:,0] if a.disable_velocity_residual
                        else classic[:,0]+pred[:,0]*dt)
                ns=torch.stack((next_v,
                                torch.zeros(b,device=device),
                                classic[:,2]+pred[:,2]*dt),1)
                next_axay=torch.stack(((ns[:,0]-s[:,0])/dt,
                    (ns[:,1]-s[:,1])/dt+ns[:,0]*ns[:,2]),1)
            elif is_slip or is_dynamic_residual:
                current_speed=torch.hypot(s[:,0],s[:,1])
                beta=torch.atan2(s[:,1],s[:,0])
                slip_ax=torch.clamp(a.kp_speed*(u[:,1]-current_speed),-8.,8.5)
                unclamped=current_speed+slip_ax*dt
                base_speed=torch.minimum(torch.maximum(unclamped,
                    torch.minimum(torch.full_like(current_speed,a.runtime_min_speed),current_speed)),
                    torch.maximum(torch.full_like(current_speed,a.runtime_max_speed),current_speed))
                base_vx=base_speed*torch.cos(beta);base_vy=base_speed*torch.sin(beta)
                if is_dynamic_residual:
                    dp=DYNAMIC_PARAMS;vx_safe=torch.clamp(torch.abs(s[:,0]),min=.5)
                    af=steer-torch.atan2(s[:,1]+LF*s[:,2],vx_safe);ar=-torch.atan2(s[:,1]-LR*s[:,2],vx_safe)
                    fzf=dp['mass']*9.81*LR/(LF+LR);fzr=dp['mass']*9.81*LF/(LF+LR)
                    fyf=fzf*dp['Df']*torch.sin(dp['Cf']*torch.atan(dp['Bf']*af-dp['Ef']*(dp['Bf']*af-torch.atan(dp['Bf']*af))))
                    fyr=fzr*dp['Dr']*torch.sin(dp['Cr']*torch.atan(dp['Br']*ar-dp['Er']*(dp['Br']*ar-torch.atan(dp['Br']*ar))))
                    beta_dot=(fyf*torch.cos(steer)+fyr)/(dp['mass']*torch.clamp(current_speed,min=.5))-s[:,2]
                    nbeta=beta+beta_dot*dt;base_vx=base_speed*torch.cos(nbeta);base_vy=base_speed*torch.sin(nbeta)
                    base_r=s[:,2]+(LF*fyf*torch.cos(steer)-LR*fyr)/dp['Iz']*dt
                else: base_r=base_vx*torch.tan(steer)/(LF+LR)
                classic=torch.stack((base_vx,base_vy,base_r),1)
                du=u[:,0]-hist[:,-1,0]
                fbase=torch.cat((s,u,delta[:,None],du[:,None],classic),1)
                pred=torch.tanh(net(torch.cat(((fbase-mean_t)/std_t,
                    ((hist-cm_t)/cs_t).reshape(b,-1)),1)))*scale
                next_vx=(classic[:,0] if a.disable_velocity_residual
                         else classic[:,0]+pred[:,0]*dt)
                ns=torch.stack((next_vx,classic[:,1]+pred[:,1]*dt,
                                classic[:,2]+pred[:,2]*dt),1) # ns : vx,vy,r
                next_axay=torch.stack(((ns[:,0]-s[:,0])/dt-s[:,1]*s[:,2],
                    (ns[:,1]-s[:,1])/dt+s[:,0]*s[:,2]),1)
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
            next_speed=torch.hypot(ns[:,0],ns[:,1]);next_beta=torch.atan2(ns[:,1],ns[:,0])
            nq=torch.stack((q[:,0]+a.position_speed_scale*next_speed*torch.cos(q[:,2]+next_beta)*dt,
                            q[:,1]+a.position_speed_scale*next_speed*torch.sin(q[:,2]+next_beta)*dt,
                            q[:,2]+ns[:,2]*dt),1)
            gt_s=target_body_t[ix+1];gt_q=pose_t[ix+1];gt_imu=imu_t[ix+1,1:3] # gt_s : vx,vy,r; gt_q : x,y,yaw; gt_imu : ax,ay
            state_component_loss=torch.nn.functional.smooth_l1_loss(
                ns,gt_s,reduction='none')
            state_loss=a.state_loss_weight*(
                state_component_loss*state_component_weights).sum(1)/state_component_weights.sum()
            yaw_rate_loss=a.yaw_rate_loss_weight*torch.nn.functional.smooth_l1_loss(
                ns[:,2],gt_s[:,2],reduction='none')
            yaw_loss=a.yaw_loss_weight*torch.nn.functional.smooth_l1_loss(nq[:,2],gt_q[:,2],reduction='none')
            pos_loss=a.position_loss_weight*torch.linalg.vector_norm(nq[:,:2]-gt_q[:,:2],dim=1)
            imu_loss=(.04*torch.nn.functional.smooth_l1_loss(next_axay,gt_imu,reduction='none').mean(1)
                      if is_e2e else 0.)
            losses.append(state_loss+yaw_rate_loss+yaw_loss+pos_loss+imu_loss)
            s,q,axay=ns,nq,next_axay;hist=torch.cat((hist[:,1:],u[:,None]),1)
            if collect:traj.append(torch.cat((q,s,axay),1))
        loss=torch.stack(losses,1).mean()
        return loss,(torch.stack(traj,1) if collect else None)

    best=float('inf');bad=0;history=[]
    fit_j=fit_starts+history_offset
    if a.balanced_sampling:
        speed_bin=np.clip((np.hypot(target_body[fit_j,0],target_body[fit_j,1])//1.).astype(int),0,4)
        steer_bin=np.digitize(np.abs(command[fit_j,0]),[.08,.20,.35])
        yaw_bin=np.digitize(np.abs(target_body[fit_j,2]),[.5,1.5,3.0])
        key=speed_bin*16+steer_bin*4+yaw_bin
        _,inverse,count=np.unique(key,return_inverse=True,return_counts=True)
        sample_probability=(1./count[inverse]);sample_probability/=sample_probability.sum()
    else: sample_probability=None
    for epoch in range(1,a.epochs+1):
        if a.curriculum:
            fraction=epoch/max(1,a.epochs)
            target_s=.1 if fraction<=.15 else (.3 if fraction<=.35 else (.6 if fraction<=.6 else (1.0 if fraction<=.8 else a.rollout_horizon)))
            train_horizon=min(horizon,max(1,round(target_s/dt)))
        else: train_horizon=horizon
        net.train();order=(rng.choice(fit_starts,len(fit_starts),replace=True,p=sample_probability)
                           if sample_probability is not None else rng.permutation(fit_starts));total=0.;count=0;t0=time.time()
        for begin in range(0,len(order),a.batch_size):
            ids=order[begin:begin+a.batch_size];opt.zero_grad(set_to_none=True)
            loss,_=rollout(ids,True,steps=train_horizon);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(ids);count+=len(ids)
        net.eval()
        vloss=0.;vpos=0;vend=[]
        with torch.no_grad():
            for i in range(0,len(val_starts),a.batch_size):
                vi=val_starts[i:i+a.batch_size];vl,vt=rollout(vi,collect=True)
                jj=vi+history_offset+horizon
                vloss+=float(vl)*len(vi)
                endpoint=torch.linalg.vector_norm(vt[:,-1,:2]-pose_t[jj,:2],dim=1)
                vpos+=float(endpoint.sum());vend.append(endpoint.cpu().numpy())
        vl=vloss/len(val_starts);vp=vpos/len(val_starts)
        tr=total/count;history.append((epoch,tr,vl,vp,time.time()-t0));print(epoch,train_horizon,tr,vl,vp,flush=True)
        # Select without touching held-out test bags.  The deployment objective
        # requested by this project is final 1 s trajectory distance.
        vp95=float(np.quantile(np.concatenate(vend),.95))
        score=(vp if a.checkpoint_objective=='position' else
               (vp95 if a.checkpoint_objective=='position_p95' else vl))
        if score<best-1e-5:best=score;bad=0;torch.save(net.state_dict(),out/'model.pt')
        else:bad+=1
        if bad>=a.patience:break
    net.load_state_dict(torch.load(out/'model.pt',map_location=device,weights_only=True));net.eval()
    alltraj=[]
    with torch.no_grad():
        for begin in range(0,len(test_starts),a.batch_size):alltraj.append(rollout(test_starts[begin:begin+a.batch_size],collect=True)[1].cpu().numpy())
    traj=np.concatenate(alltraj);j0=test_starts+history_offset
    gtpose=np.stack([pose[j0+k] for k in range(horizon+1)],1);gtstate=np.stack([target_body[j0+k] for k in range(horizon+1)],1)
    pe=np.linalg.norm(traj[:,:,:2]-gtpose[:,:,:2],axis=2)
    vxe=np.abs(traj[:,:,3]-gtstate[:,:,0])
    se=np.abs(np.hypot(traj[:,:,3],traj[:,:,4])-np.hypot(gtstate[:,:,0],gtstate[:,:,1]))
    we=np.abs(traj[:,:,5]-gtstate[:,:,2])
    metrics={'model':a.model,'action':'[steer_cmd, /drive.speed setpoint]','test_windows':len(test_starts),
      'split_policy':split_policy,'test_is_unseen_data':not split_policy.startswith('single-bag'),
      'yaw_rate_target':(a.slip_yaw_source if is_slip else
                         (a.yaw_target if is_noslip else 'recursive_imu')),
      'slip_yaw_source':(a.slip_yaw_source if is_slip else None),
      'kf_cornering_stiffness_N_per_rad':(
          {'front':a.kf_cf,'rear':a.kf_cr} if is_slip else None),
      'imu_axis_signs':{'wz':a.imu_wz_sign,'ax':a.imu_ax_sign,'ay':a.imu_ay_sign},
      'kinematic_noslip_input_vy_max_abs':(float(np.max(np.abs(observed_body[:,1]))) if a.model=='kinematic_noslip_noimu' else None),
      'slip_kinematic_with_imu_input_vy_max_abs':(float(np.max(np.abs(body[:,1]))) if is_slip else None),
      'input_normalization':a.normalization,
      'mlp_input_dim':nbase+10,
      'actuator_model':{'servo_time_constant_s':a.steer_time_constant,'max_steering_rate_rad_s':a.max_steer_rate},
      'steering_command_mapping':{'scale':a.steer_scale,'bias_rad':a.steer_bias},
      'training_sampling':('balanced speed/steer/yaw-rate bins' if a.balanced_sampling else 'uniform'),
      'training_curriculum_s':([.1,.3,.6,1.0,a.rollout_horizon] if a.curriculum else None),
      'velocity_residual_enabled':not a.disable_velocity_residual,
      'final_trajectory_mean_m':float(pe[:,-1].mean()),'final_trajectory_median_m':float(np.median(pe[:,-1])),'final_trajectory_p95_m':float(np.quantile(pe[:,-1],.95)),
      'final_trajectory_worst_m':float(pe[:,-1].max()),
      'final_speed_mae_mps':float(se[:,-1].mean()),'final_vx_mae_mps':float(vxe[:,-1].mean()),
      'final_yaw_rate_mae_radps':float(we[:,-1].mean()),
      'kp_speed':a.kp_speed,'dt':dt,'epochs':len(history),
      'rollout_horizon_s':a.rollout_horizon,
      'runtime_speed_limits_mps':[a.runtime_min_speed,a.runtime_max_speed],
      'position_speed_scale':a.position_speed_scale,
      'loss_weights':{'state':a.state_loss_weight,'position':a.position_loss_weight,
                      'yaw':a.yaw_loss_weight,'yaw_rate':a.yaw_rate_loss_weight,
                      'state_components':{'vx':a.vx_loss_weight,
                                          'vy':a.vy_loss_weight,
                                          'yaw_rate':a.state_yaw_rate_loss_weight}},
      'best_validation_loss':(best if np.isfinite(best) else None),
      'dynamic_imu_policy':('measured EMA ax/ay only at horizon start; predicted ax/ay recursively thereafter' if is_e2e else None),
      'dynamic_classic_params':(DYNAMIC_PARAMS if is_dynamic_residual else None)}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez_compressed(out/'test_predictions.npz',starts=test_starts,prediction=traj,gt_pose=gtpose,gt_state=gtstate,position_error=pe,speed_error=se,yaw_rate_error=we)
    np.savez(out/'normalization.npz',base_mean=mean,base_std=std,command_mean=cmd_mean,command_std=cmd_std)
    with (out/'history.csv').open('w',newline='') as f: w=csv.writer(f);w.writerow(('epoch','train','validation','validation_final_position_m','seconds'));w.writerows(history)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
