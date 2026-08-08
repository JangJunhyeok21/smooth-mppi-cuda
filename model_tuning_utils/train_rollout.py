#!/usr/bin/env python3
"""Train/evaluate hybrid dynamics on contiguous 1-second position rollouts."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

try:
    from model_tuning_utils.train_hybrid import classic_derivative, NAMES
    from model_tuning_utils.lateral_velocity_kf import estimate_dataset
except ModuleNotFoundError:  # direct execution: python model_tuning/train_rollout.py
    from train_hybrid import classic_derivative, NAMES
    from lateral_velocity_kf import estimate_dataset


def prepare(path, args):
    z = np.load(path); a = z["samples"].astype(np.float64); dt = float(z["dt"])
    n = len(a); win = min(args.pose_window, n // 2 * 2 - 1)
    bag_id=a[:,11].astype(int) if a.shape[1]>11 else np.zeros(n,dtype=int)
    x=np.full(n,np.nan); y=x.copy(); yaw=x.copy()
    for bid in np.unique(bag_id):
        ii=np.flatnonzero(bag_id==bid); w=min(win,len(ii)//2*2-1)
        if w<5: continue
        x[ii]=savgol_filter(a[ii,1],w,3); y[ii]=savgol_filter(a[ii,2],w,3)
        yaw[ii]=savgol_filter(np.unwrap(a[ii,3]),w,3)
    vx = a[:, 4]
    if getattr(args, "strict_no_imu", False):
        # Kinematic no-slip no-IMU path: do not invoke the lateral KF even indirectly
        # while selecting windows.  vy is structurally zero and yaw rate comes
        # from the odometry/state column in the extracted bag dataset.
        vy = np.zeros(n, dtype=np.float64)
        omega = a[:, 6]
    else:
        # Runtime-observable state: longitudinal odometry plus the same causal
        # 2-state KF used by SMPPI. No centered/future pose differentiation is used.
        vy, omega = estimate_dataset(a, z["columns"], dt)
    state = np.c_[np.hypot(vx, vy), np.arctan2(vy, np.maximum(vx, .1)), omega]
    # Keep speed setpoint: on this vehicle acceleration is frequently saturated and the
    # low-level longitudinal loop primarily follows AckermannDrive.speed.
    control = a[:, 7:10]
    pose = np.c_[x, y, yaw]
    deriv=np.full_like(state,np.nan)
    for bid in np.unique(bag_id):
        ii=np.flatnonzero(bag_id==bid); w=min(win,len(ii)//2*2-1)
        if w>=5:
            for j in range(3): deriv[ii,j]=savgol_filter(state[ii,j],w,3,deriv=1,delta=dt)
    impact = ((deriv[:,0] < getattr(args,"impact_decel",-10.)) &
              (state[:,0] > 1.) & (control[:,2] > 1.))
    impact_neighborhood=np.zeros(n,dtype=bool)
    radius=round(getattr(args,"impact_margin",.5)/dt)
    for index in np.flatnonzero(impact):
        lo,hi=max(0,index-radius),min(n,index+radius+1); local=np.arange(lo,hi)
        impact_neighborhood[local[bag_id[local]==bag_id[index]]]=True
    # A filtered dataset may drop stale-IMU samples inside a retained segment.
    # Treat the resulting time discontinuity as a hard boundary so a rollout
    # can never jump across unavailable controller observations.
    pose_jump = np.hypot(np.diff(a[:, 1]), np.diff(a[:, 2])) > args.max_pose_step
    time_jump = (bag_id[1:] != bag_id[:-1]) | (np.abs(np.diff(a[:,0])-dt) > .5*dt)
    jump = np.r_[False, pose_jump | time_jump]
    valid = ((state[:, 0] >= args.min_speed) & (state[:, 0] <= args.max_speed) &
             (np.abs(state[:, 1]) < args.max_beta) & (np.abs(state[:, 2]) < args.max_omega) &
             (np.abs(control[:, 0]) <= .55) & (np.abs(control[:, 1]) <= 10.) &
             (control[:, 2] >= 0.) & (control[:, 2] <= 12.) & ~jump &
             ~impact_neighborhood &
             np.all(np.isfinite(np.c_[pose, state, control, deriv]), axis=1))
    # A window is usable only if every sample is valid; never concatenate gaps.
    horizon = round(args.horizon / dt); total = args.history + horizon + 1
    good = np.convolve(valid.astype(np.int16), np.ones(total, dtype=np.int16), mode="valid") == total
    good &= bag_id[:len(good)] == bag_id[total-1:]
    starts = np.flatnonzero(good)
    if len(starts) < 100:
        raise SystemExit(f"only {len(starts)} contiguous windows; relax physical limits after inspecting data")
    return pose, state, control, deriv, starts, dt, horizon


def fit_classic(state, control, deriv, train_ids, args):
    ids = np.unique(np.concatenate([np.arange(i, i+args.history+1) for i in train_ids]))
    if len(ids) > args.max_fit_samples:
        ids = np.random.default_rng(args.seed).choice(ids, args.max_fit_samples, replace=False)
    fixed = (args.mass, args.iz, args.lf, args.lr, 9.81)
    x0 = np.array([.04, 6., 1.4, .8, .1, 6., 1.4, .8, .1])
    lo = np.array([0., .1, .5, .1, -1., .1, .5, .1, -1.])
    hi = np.array([.5, 30., 3., 2., 1., 30., 3., 2., 1.])
    scale = np.array([2., 4., 20.])
    fun = lambda q: ((classic_derivative(state[ids], control[ids, :2], q, fixed)-deriv[ids])/scale).ravel()
    return least_squares(fun, x0, bounds=(lo, hi), loss="soft_l1", max_nfev=args.max_nfev).x, fixed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset"); p.add_argument("-o", "--output", default="model_tuning/rollout_output")
    p.add_argument("--horizon", type=float, default=1.0); p.add_argument("--history", type=int, default=50)
    p.add_argument("--pose-window", type=int, default=21); p.add_argument("--max-pose-step", type=float, default=.25)
    p.add_argument("--min-speed", type=float, default=.7); p.add_argument("--max-speed", type=float, default=10.)
    p.add_argument("--max-beta", type=float, default=.7); p.add_argument("--max-omega", type=float, default=8.)
    p.add_argument("--impact-decel",type=float,default=-10.)
    p.add_argument("--impact-margin",type=float,default=.5)
    p.add_argument("--mass", type=float, default=3.74); p.add_argument("--iz", type=float, default=.04712)
    p.add_argument("--lf", type=float, default=.163); p.add_argument("--lr", type=float, default=.161)
    p.add_argument("--max-fit-samples", type=int, default=20000); p.add_argument("--max-nfev", type=int, default=200)
    p.add_argument("--classic-params", help="JSON from tune_classic_rollout.py; skips derivative fitting")
    p.add_argument("--longitudinal-model",choices=("accel","speed"),default="accel")
    p.add_argument("--resume", help="model state_dict checkpoint to continue with a lower learning rate")
    p.add_argument("--hidden", type=int, default=96); p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--rnn", choices=("lstm",), default="lstm",
                   help="recurrent residual architecture (GRU code was removed)")
    p.add_argument("--model", choices=("direct", "residual"), default="residual",
                   help="direct learns the full transition; residual corrects the classic model")
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--val-fraction", type=float, default=.2); p.add_argument("--position-target", type=float, default=.05)
    p.add_argument("--patience", type=int, default=25); p.add_argument("--seed", type=int, default=7)
    p.add_argument("--state-loss-weight", type=float, default=.15)
    p.add_argument("--residual-output-scale",type=float,nargs=3,default=(15.,5.,100.),
                   metavar=("V_DOT","BETA_DOT","OMEGA_DOT"),
                   help="tanh output limits for residual derivative correction")
    p.add_argument("--min-train-seconds",type=float,default=0.,
                   help="do not early-stop before this much wall-clock training time")
    p.add_argument("--device", default="cuda"); args = p.parse_args()
    try:
        import torch
        from torch import nn
    except ImportError as e: raise SystemExit(f"PyTorch required: {e}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    pose, state, control, deriv, starts, dt, horizon = prepare(args.dataset, args)
    # Chronological split with a full-window embargo prevents overlapping leakage.
    raw=np.load(args.dataset)["samples"]
    if raw.shape[1]>10:
        split_flag=raw[:,10].astype(int); train_starts=starts[split_flag[starts]==0]; val_starts=starts[split_flag[starts]==1]
        split_t=len(state)
    else:
        split_t = int(np.quantile(starts, 1-args.val_fraction)); embargo = args.history+horizon
        train_starts = starts[starts+embargo < split_t]; val_starts = starts[starts >= split_t]
    if not len(train_starts) or not len(val_starts): raise SystemExit("no train/validation windows after split")
    fixed=(args.mass,args.iz,args.lf,args.lr,9.81)
    actuator_meta={"type":"instantaneous","servo_time_constant_s":0.,
                   "max_steering_rate_rad_s":float("inf")}
    if args.classic_params:
        loaded=json.loads(Path(args.classic_params).read_text()); values=loaded.get("params",loaded)
        param_names=(("K_speed","K_drag","B_f","C_f","D_f","E_f","B_r","C_r","D_r","E_r")
                     if args.longitudinal_model=="speed" else NAMES)
        theta=np.array([values[n] for n in param_names],dtype=np.float64)
        fixed_meta=loaded.get("fixed_params",{})
        fixed=(args.mass,float(fixed_meta.get("I_z",args.iz)),args.lf,args.lr,9.81)
        actuator_meta=loaded.get("actuator_model",actuator_meta)
    else:
        if args.longitudinal_model=="speed": raise SystemExit("speed model requires --classic-params from tune_classic_rollout.py")
        param_names=NAMES
        theta, fixed = fit_classic(state, control, deriv, train_starts, args)
    tau=float(actuator_meta.get("servo_time_constant_s",0.))
    max_steer_rate=float(actuator_meta.get("max_steering_rate_rad_s",float("inf")))
    effective_steer=control[:,0].copy()
    if tau>0:
        bag_id=raw[:,11].astype(int) if raw.shape[1]>11 else np.zeros(len(raw),int)
        for bid in np.unique(bag_id):
            ii=np.flatnonzero(bag_id==bid); effective_steer[ii[0]]=control[ii[0],0]
            for k in range(1,len(ii)):
                rate=np.clip((control[ii[k],0]-effective_steer[ii[k-1]])/tau,
                             -max_steer_rate,max_steer_rate)
                effective_steer[ii[k]]=effective_steer[ii[k-1]]+rate*dt
    effective_control=control.copy();effective_control[:,0]=effective_steer
    if args.longitudinal_model=="speed":
        ks,kd,*tire=theta; lateral_theta=np.r_[0.,tire]
        classic=classic_derivative(state,effective_control[:,:2],lateral_theta,fixed)
        classic[:,0]=ks*(control[:,2]-state[:,0])-kd*state[:,0]
    else: classic = classic_derivative(state, control[:, :2], theta, fixed)
    feat = np.c_[state, control] if args.model == "direct" else np.c_[state, control, classic]
    train_points=np.flatnonzero(raw[:,10]==0) if raw.shape[1]>10 else np.arange(0,split_t)
    mean, std = feat[train_points].mean(0), feat[train_points].std(0).clip(1e-5)
    out_scale = np.array(
        args.residual_output_scale if args.model == "residual" else (5., 5., 30.),
        dtype=np.float32,
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.LSTM(feat.shape[1],args.hidden,batch_first=True)
            self.head=nn.Linear(args.hidden,3)
        def forward(self, z, h=None):
            o,h=self.rnn(z,h); return torch.tanh(self.head(o[:,-1]))*torch.as_tensor(out_scale,device=z.device),h
    dev=torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model=Net().to(dev)
    if args.resume: model.load_state_dict(torch.load(args.resume,map_location=dev,weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.5,patience=8,min_lr=1e-6)
    mean_t=torch.tensor(mean,dtype=torch.float32,device=dev); std_t=torch.tensor(std,dtype=torch.float32,device=dev)
    th=torch.tensor(theta,dtype=torch.float32,device=dev); mass,iz,lf,lr,_=fixed

    def dyn(s,u):
        v,beta,om=s.unbind(1); steer=u[:,0]
        if args.longitudinal_model=="speed":
            ks,kd,bf,cf,df,ef,br,cr,dr,er=th;vdot=ks*(u[:,2]-v)-kd*v
        else:
            accel=u[:,1];cm,bf,cf,df,ef,br,cr,dr,er=th;vdot=accel*(1-cm*v)
        vx=torch.clamp(v*torch.cos(beta),min=.2); vy=v*torch.sin(beta)
        af=steer-torch.atan2(vy+lf*om,vx); ar=-torch.atan2(vy-lr*om,vx)
        fzf=mass*9.81*lr/(lf+lr); fzr=mass*9.81*lf/(lf+lr)
        ff=fzf*df*torch.sin(cf*torch.atan(bf*af-ef*(bf*af-torch.atan(bf*af))))
        fr=fzr*dr*torch.sin(cr*torch.atan(br*ar-er*(br*ar-torch.atan(br*ar))))
        return torch.stack((vdot,(ff*torch.cos(steer)+fr)/(mass*torch.clamp(v,min=.5))-om,
                            (lf*ff*torch.cos(steer)-lr*fr)/iz),1)

    def batch(ids, grad):
        ids=np.asarray(ids); b=len(ids); h=None
        s=torch.tensor(state[ids+args.history-1],dtype=torch.float32,device=dev)
        ps=torch.tensor(pose[ids+args.history-1],dtype=torch.float32,device=dev)
        actual_steer=torch.tensor(effective_steer[ids+args.history-1],
                                  dtype=torch.float32,device=dev)
        # Warm recurrent memory with contiguous measured history.
        hist=np.stack([feat[i:i+args.history] for i in ids])
        _,h=model.rnn((torch.tensor(hist,dtype=torch.float32,device=dev)-mean_t)/std_t)
        losses=[]
        for k in range(horizon):
            j=ids+args.history-1+k; u=torch.tensor(control[j],dtype=torch.float32,device=dev)
            if tau>0:
                actual_steer=actual_steer+torch.clamp(
                    (u[:,0]-actual_steer)/tau,-max_steer_rate,max_steer_rate)*dt
            effective_u=u.clone();effective_u[:,0]=actual_steer
            base=dyn(s,effective_u)
            f=torch.cat((s,u),1) if args.model == "direct" else torch.cat((s,u,base),1)
            learned,h=model(((f-mean_t)/std_t).unsqueeze(1),h)
            derivative = learned if args.model == "direct" else base+learned
            s=s+derivative*dt
            yaw=ps[:,2]+s[:,2]*dt
            ps=torch.stack((ps[:,0]+s[:,0]*torch.cos(yaw+s[:,1])*dt,
                            ps[:,1]+s[:,0]*torch.sin(yaw+s[:,1])*dt,yaw),1)
            gt=torch.tensor(pose[j+1],dtype=torch.float32,device=dev)
            gt_s=torch.tensor(state[j+1],dtype=torch.float32,device=dev)
            state_scale=torch.tensor([2.,.2,2.],device=dev)
            state_error=torch.sum(((s-gt_s)/state_scale)**2,1)
            losses.append(torch.sum((ps[:,:2]-gt[:,:2])**2,1)+.02*(ps[:,2]-gt[:,2])**2+
                          args.state_loss_weight*state_error)
        final=torch.sqrt(torch.sum((ps[:,:2]-gt[:,:2])**2,1)+1e-12)
        return torch.stack(losses,1).mean(),final

    best=float("inf"); stale=0; out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    train_started=time.monotonic()
    for epoch in range(args.epochs):
        order=rng.permutation(train_starts)
        model.train()
        for q in range(0,len(order),args.batch_size):
            opt.zero_grad(); loss,_=batch(order[q:q+args.batch_size],True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
        model.eval(); errs=[]
        with torch.no_grad():
            for q in range(0,len(val_starts),args.batch_size): errs.append(batch(val_starts[q:q+args.batch_size],False)[1].cpu().numpy())
        e=np.concatenate(errs); score=float(e.mean())
        scheduler.step(score)
        elapsed=time.monotonic()-train_started
        print(f"epoch={epoch+1:03d} val_1s_mean={score:.4f}m p95={np.quantile(e,.95):.4f}m elapsed={elapsed:.1f}s")
        if score < best:
            best=score; stale=0; torch.save(model.state_dict(),out/'rollout_state.pt')
            metrics={"val_1s_mean_m":score,"val_1s_median_m":float(np.median(e)),"val_1s_p95_m":float(np.quantile(e,.95)),
                     "target_m":args.position_target,"target_met":score<=args.position_target,"train_windows":len(train_starts),
                     "val_windows":len(val_starts),"classic_params":dict(zip(param_names,map(float,theta)))}
            metrics["longitudinal_model"]=args.longitudinal_model
            metrics["fixed_params"]={"mass":fixed[0],"I_z":fixed[1],
                                     "l_f":fixed[2],"l_r":fixed[3]}
            metrics["actuator_model"]=actuator_meta
            metrics["model"] = args.model; metrics["rnn"] = args.rnn; metrics["history_s"] = args.history*dt
            metrics["output_scale"] = list(map(float,out_scale))
            (out/'rollout_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        else: stale+=1
        if elapsed>=args.min_train_seconds and (best <= args.position_target or stale >= args.patience): break
    print((out/'rollout_metrics.json').read_text())


if __name__ == "__main__": main()
