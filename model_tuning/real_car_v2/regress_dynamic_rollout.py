#!/usr/bin/env python3
"""Fit Pacejka parameters with recursive 1 s state/position rollouts.

Edit constants below and run without arguments.  The estimator state (KF vy,
signed IMU yaw rate) and actuator lag are treated as the observable contract;
only tire B,C,D,E are optimized. I_z, geometry and mass remain fixed.
"""
from pathlib import Path
import json
import numpy as np
import yaml
from scipy.optimize import differential_evolution, least_squares

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"model_tuning/data/real_car_v2_dynamic_residual.npz"
OUTPUT=ROOT/"model_tuning/results/dynamic_rollout_regression"
H=50;STRIDE=5;MAX_WINDOWS_PER_BAG=350;SEED=31;UPDATE_CONFIG=True
NAMES=("B_f","D_f","B_r","D_r")
FIXED_C=1.3;FIXED_E=0.0
BOUNDS=np.array([[.2,15.],[.2,2.5],[.2,15.],[.2,2.5]])

def expand(p):
    bf,df,br,dr=p
    return np.array([bf,FIXED_C,df,FIXED_E,br,FIXED_C,dr,FIXED_E])

def select_windows(x,b,split,valid,target_split):
    starts=[]
    for bid in np.unique(b[split==target_split]):
        candidates=np.array([i for i in range(len(x)-H) if b[i]==bid and split[i]==target_split
            and valid[i:i+H+1].all() and np.all(b[i:i+H+1]==bid)
            and np.mean(np.abs(x[i:i+H,0]))>=.5],int)
        if len(candidates)>MAX_WINDOWS_PER_BAG:
            # Deterministic coverage over the complete session, not a dense
            # cluster from its longest steady-state portion.
            candidates=candidates[np.linspace(0,len(candidates)-1,MAX_WINDOWS_PER_BAG).astype(int)]
        starts.extend(candidates[::STRIDE] if len(candidates) else [])
    return np.asarray(starts,int)

def rollout(params,x,starts,cfg,return_trace=False):
    n=len(starts);idx=starts[:,None]+np.arange(H+1);gt=x[idx,:3].astype(float)
    cmd=x[idx,:][:,:,3:5].astype(float);state=gt[:,0].copy();applied=x[starts,5].astype(float).copy()
    speed_ref=state[:,0].copy();pred=np.empty((n,H,3));dt=.02
    Bf,Cf,Df,Ef,Br,Cr,Dr,Er=expand(params);lf,lr,m,iz=map(float,(cfg['l_f'],cfg['l_r'],cfg['mass'],cfg['dynamic_mlp_I_z']));wb=lf+lr
    fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb
    for k in range(H):
        steer_target=np.clip(float(cfg['kinematic_steer_scale'])*cmd[:,k,0]+float(cfg['kinematic_steer_bias']),-.55,.55)
        steer_rate=np.clip((steer_target-applied)/float(cfg['steer_servo_time_constant']),-float(cfg['actuator_max_steer_rate']),float(cfg['actuator_max_steer_rate']))
        applied=np.clip(applied+steer_rate*dt,-.55,.55)
        speed_cmd=np.clip(cmd[:,k,1],float(cfg['min_speed']),4.0)
        tau=np.where(speed_cmd>=speed_ref,float(cfg['speed_reference_accel_time_constant']),float(cfg['speed_reference_brake_time_constant']))
        ref_rate=np.clip((speed_cmd-speed_ref)/np.maximum(tau,1e-3),-float(cfg['actuator_max_speed_reference_rate']),float(cfg['actuator_max_speed_reference_rate']))
        speed_ref+=ref_rate*dt
        vx,vy,r=state.T;speed=np.hypot(vx,vy);ax=np.clip(float(cfg['speed_servo_kp'])*(speed_ref-speed),float(cfg['min_accel']),float(cfg['max_accel']))
        safe=np.maximum(np.abs(vx),.5);af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe)
        baf=Bf*af;bar=Br*ar
        fyf=fzf*Df*np.sin(Cf*np.arctan(baf-Ef*(baf-np.arctan(baf))))
        fyr=fzr*Dr*np.sin(Cr*np.arctan(bar-Er*(bar-np.arctan(bar))))
        ay=(fyf*np.cos(applied)+fyr)/m;rdot=(lf*fyf*np.cos(applied)-lr*fyr)/iz
        state=np.c_[vx+(ax+vy*r)*dt,vy+(ay-vx*r)*dt,r+rdot*dt]
        # Match deployed low-speed rollout prior.
        state[np.abs(state[:,0])<float(cfg['kf_low_speed_threshold']),1]=0.0
        pred[:,k]=state
    return (pred,gt[:,1:]) if return_trace else pred

def relative_xy(states,scale):
    p=np.zeros((len(states),states.shape[1],3));dt=.02
    for k in range(states.shape[1]):
        prev=p[:,k-1] if k else np.zeros((len(states),3));vx,vy,r=states[:,k].T
        p[:,k,0]=prev[:,0]+scale*(vx*np.cos(prev[:,2])-vy*np.sin(prev[:,2]))*dt
        p[:,k,1]=prev[:,1]+scale*(vx*np.sin(prev[:,2])+vy*np.cos(prev[:,2]))*dt;p[:,k,2]=prev[:,2]+r*dt
    return p

def residual(params,x,starts,cfg,regularize=True):
    pred,gt=rollout(params,x,starts,cfg,True);time=np.linspace(.25,1.,H)[None,:,None]
    # vx is governed mostly by the separately fitted longitudinal actuator;
    # lateral fit emphasizes vy/r and accumulated map displacement.
    state=((pred-gt)*np.array([.35,2.0,1.5])[None,None,:]*time).ravel()
    pp=relative_xy(pred,float(cfg['kinematic_position_speed_scale']));gp=relative_xy(gt,float(cfg['kinematic_position_speed_scale']))
    pos=(2.0*(pp[:,-1,:2]-gp[:,-1,:2])).ravel()
    if not regularize:return np.r_[state,pos]
    # Weak identifiability prior, centered at the old rough-surface force peak
    # (about 35 N / static axle load ~= D 1.9), not at the bad 1-step solution.
    prior=np.array([6.,1.9,6.,1.9]);scale=np.array([8.,1.,8.,1.])
    # Prevent an extreme oversteer effective slope (Fy' ~= Fz*B*C*D).
    front_slope=params[0]*FIXED_C*params[1];rear_slope=params[2]*FIXED_C*params[3]
    oversteer_penalty=max(0.,front_slope-rear_slope)
    # Penalize numerical yaw chatter not present in measured trajectories.
    yaw_second=np.diff(pred[:,:,2],n=2,axis=1)-np.diff(gt[:,:,2],n=2,axis=1)
    return np.r_[state,pos,.04*(params-prior)/scale,.35*yaw_second.ravel(),.5*oversteer_penalty]

def metrics(params,x,starts,cfg):
    pred,gt=rollout(params,x,starts,cfg,True);e=np.abs(pred[:,-1]-gt[:,-1]);pp=relative_xy(pred,float(cfg['kinematic_position_speed_scale']));gp=relative_xy(gt,float(cfg['kinematic_position_speed_scale']));pe=np.linalg.norm(pp[:,-1,:2]-gp[:,-1,:2],axis=1)
    return {'windows':len(starts),'final_state_mae':e.mean(0).tolist(),'final_state_p95':np.quantile(e,.95,axis=0).tolist(),'trajectory_mean_m':float(pe.mean()),'trajectory_p95_m':float(np.quantile(pe,.95)),'trajectory_max_m':float(pe.max())}

def update_config(params):
    full_names=("B_f","C_f","D_f","E_f","B_r","C_r","D_r","E_r")
    lines=(ROOT/'config/params.yaml').read_text().splitlines();values=dict(zip(('dynamic_mlp_'+n for n in full_names),params));found=set()
    for i,line in enumerate(lines):
        for key,value in values.items():
            if line.strip().startswith(key+':'):
                indent=line[:len(line)-len(line.lstrip())];lines[i]=f'{indent}{key}: {value:.10g}  # 1 s recursive rollout regression';found.add(key)
    if found!=set(values):raise RuntimeError(f'missing config keys {set(values)-found}')
    (ROOT/'config/params.yaml').write_text('\n'.join(lines)+'\n')

def main():
    OUTPUT.mkdir(parents=True,exist_ok=True);d=np.load(DATA);x=d['features'].astype(float);b=d['bag_id'];s=d['split'];v=d['valid'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters']
    train=select_windows(x,b,s,v,0);val=select_windows(x,b,s,v,1);test=select_windows(x,b,s,v,2)
    old_full=np.array([3.879070152566808,1.6471076687680233,.0710062229162444,-1.,2.321287285513187,1.9234527357451916,.05906540313616536,-1.])
    current=np.array([old_full[0],old_full[2],old_full[4],old_full[6]])
    def cost(p):
        q=residual(p,x,train,cfg);delta=.3;a=np.abs(q);return np.mean(np.where(a<delta,.5*q*q,delta*(a-.5*delta)))
    de=differential_evolution(cost,BOUNDS,seed=SEED,popsize=9,maxiter=55,tol=5e-4,polish=False,workers=1)
    ls=least_squares(lambda p:residual(p,x,train,cfg),de.x,bounds=(BOUNDS[:,0],BOUNDS[:,1]),loss='soft_l1',f_scale=.3,max_nfev=300,verbose=1)
    fitted=ls.x
    report={'parameter_names':NAMES,'fixed_shape_factor_C':FIXED_C,'fixed_curvature_E':FIXED_E,'current_restricted':current.tolist(),'fitted':fitted.tolist(),'expanded_fitted':dict(zip(("B_f","C_f","D_f","E_f","B_r","C_r","D_r","E_r"),expand(fitted).tolist())),'force_peak_newtons':{'front':float(fitted[1]*float(cfg['mass'])*9.81*float(cfg['l_r'])/(float(cfg['l_f'])+float(cfg['l_r']))),'rear':float(fitted[3]*float(cfg['mass'])*9.81*float(cfg['l_f'])/(float(cfg['l_f'])+float(cfg['l_r'])))},'objective':'stable recursive 1.0 s vx/vy/yaw-rate plus final relative position; I_z fixed','current_metrics':{z:metrics(current,x,q,cfg) for z,q in [('train',train),('validation',val),('test',test)]},'fitted_metrics':{z:metrics(fitted,x,q,cfg) for z,q in [('train',train),('validation',val),('test',test)]}}
    (OUTPUT/'dynamic_rollout_params.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if UPDATE_CONFIG:update_config(expand(fitted))
if __name__=='__main__':main()
