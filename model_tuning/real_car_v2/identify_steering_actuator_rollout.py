#!/usr/bin/env python3
"""Joint steering-actuator/tire identification from recursive real-car rollouts.

This is intentionally diagnostic: without measured rack/servo angle actuator
lag and tire response are only jointly identifiable. It never edits params.yaml.
"""
import argparse, json
from pathlib import Path
import numpy as np
import yaml
from scipy.optimize import differential_evolution, least_squares

H=60; STRIDE=5
NAMES=("steer_scale","steer_bias","steer_tau","max_steer_rate","B_f","D_f","B_r","D_r")
BOUNDS=((.2,1.8),(-.12,.12),(.005,.35),(.3,8.),(.2,15.),(.01,2.5),(.2,15.),(.01,2.5))

def windows(a,n,split):
    cut=int(.6*len(a)); lo,hi=(0,cut) if split=="fit" else (cut,len(a))
    q=np.arange(lo,max(lo,hi-H-1),STRIDE)
    # Require steering excitation; constant-radius windows cannot identify lag.
    return q[np.array([np.ptp(a[i:i+H,n["steer"]])>.035 for i in q])]

def rollout(p,a,n,starts,cfg):
    idx=starts[:,None]+np.arange(H+1); gt=a[idx]
    x=gt[:,0,[n["vx"],n["vy"],n["omega"]]].copy()
    cmd=gt[:,:,n["steer"]]; applied=np.clip(p[0]*cmd[:,0]+p[1],-.55,.55)
    pred=np.empty((len(starts),H,3)); dt=.02
    scale,bias,tau,maxrate,Bf,Df,Br,Dr=p
    Cf,Ef,Cr,Er=map(float,(cfg["dynamic_mlp_C_f"],cfg["dynamic_mlp_E_f"],cfg["dynamic_mlp_C_r"],cfg["dynamic_mlp_E_r"]))
    lf,lr,m,iz=map(float,(cfg["l_f"],cfg["l_r"],cfg["mass"],cfg["dynamic_mlp_I_z"])); wb=lf+lr
    fzf=m*9.81*lr/wb; fzr=m*9.81*lf/wb
    for k in range(H):
        target=np.clip(scale*cmd[:,k]+bias,-.55,.55)
        applied=np.clip(applied+np.clip((target-applied)/tau,-maxrate,maxrate)*dt,-.55,.55)
        vx,vy,r=x.T; safe=np.maximum(np.abs(vx),.5)
        af=applied-np.arctan2(vy+lf*r,safe); ar=-np.arctan2(vy-lr*r,safe)
        z=Bf*af; fyf=fzf*Df*np.sin(Cf*np.arctan(z-Ef*(z-np.arctan(z))))
        z=Br*ar; fyr=fzr*Dr*np.sin(Cr*np.arctan(z-Er*(z-np.arctan(z))))
        ay=(fyf*np.cos(applied)+fyr)/m; rdot=(lf*fyf*np.cos(applied)-lr*fyr)/iz
        # Measured vx is an exogenous input here: this fit isolates lateral response.
        vx1=gt[:,k+1,n["vx"]]; vy1=vy+(ay-vx*r)*dt; r1=r+rdot*dt
        x=np.c_[vx1,vy1,r1]; pred[:,k]=x
    truth=gt[:,1:,[n["vx"],n["vy"],n["omega"]]]
    return pred,truth

def relpose(s,scale):
    out=np.zeros((len(s),s.shape[1],3)); dt=.02
    for k in range(s.shape[1]):
        prev=out[:,k-1] if k else 0.; vx,vy,r=s[:,k].T
        yaw=prev[:,2] if k else np.zeros(len(s)); out[:,k,0]=(prev[:,0] if k else 0)+scale*(vx*np.cos(yaw)-vy*np.sin(yaw))*dt
        out[:,k,1]=(prev[:,1] if k else 0)+scale*(vx*np.sin(yaw)+vy*np.cos(yaw))*dt; out[:,k,2]=yaw+r*dt
    return out

def residual(p,a,n,start,cfg,regularize=True):
    pr,gt=rollout(p,a,n,start,cfg); w=np.linspace(.2,1,H)[None,:,None]
    e=((pr-gt)*np.array([0.,2.,1.5])[None,None,:]*w).ravel()
    pp=relpose(pr,float(cfg["kinematic_position_speed_scale"])); gp=relpose(gt,float(cfg["kinematic_position_speed_scale"]))
    pose=(np.c_[2*(pp[:,-1,:2]-gp[:,-1,:2]),1.5*(pp[:,-1,2]-gp[:,-1,2])]).ravel()
    if not regularize:return np.r_[e,pose]
    old=np.array([cfg["kinematic_steer_scale"],cfg["kinematic_steer_bias"],cfg["steer_servo_time_constant"],cfg["actuator_max_steer_rate"],cfg["dynamic_mlp_B_f"],cfg["dynamic_mlp_D_f"],cfg["dynamic_mlp_B_r"],cfg["dynamic_mlp_D_r"]],float)
    return np.r_[e,pose,.015*(p-old)/np.array([.6,.05,.15,3.,5.,.5,5.,.5])]

def metrics(p,a,n,start,cfg):
    pr,gt=rollout(p,a,n,start,cfg); pp=relpose(pr,float(cfg["kinematic_position_speed_scale"])); gp=relpose(gt,float(cfg["kinematic_position_speed_scale"])); out={"windows":int(len(start))}
    for steps in (10,25,50,60):
        e=np.abs(pr[:,steps-1,1:]-gt[:,steps-1,1:]); pe=np.linalg.norm(pp[:,steps-1,:2]-gp[:,steps-1,:2],axis=1)
        out[f"{steps*.02:.1f}s"]={"vy_mae":float(e[:,0].mean()),"yaw_rate_mae":float(e[:,1].mean()),"position_mean_m":float(pe.mean()),"position_p95_m":float(np.quantile(pe,.95))}
    return out

def main():
    q=argparse.ArgumentParser();q.add_argument("dataset");q.add_argument("--params",default=str(Path(__file__).resolve().parents[2]/"config/params.yaml"));q.add_argument("--out",default="/tmp/steering_actuator_rollout.json");args=q.parse_args()
    z=np.load(args.dataset,allow_pickle=True);a=np.asarray(z["samples"],float);n={str(x):i for i,x in enumerate(z["columns"])};cfg=yaml.safe_load(Path(args.params).read_text())["/**"]["ros__parameters"]
    fit,val=windows(a,n,"fit"),windows(a,n,"validation");old=np.array([cfg["kinematic_steer_scale"],cfg["kinematic_steer_bias"],cfg["steer_servo_time_constant"],cfg["actuator_max_steer_rate"],cfg["dynamic_mlp_B_f"],cfg["dynamic_mlp_D_f"],cfg["dynamic_mlp_B_r"],cfg["dynamic_mlp_D_r"]],float)
    def objective(p):
        e=residual(p,a,n,fit,cfg); d=.25; u=np.abs(e); return np.mean(np.where(u<d,.5*e*e,d*(u-.5*d)))
    de=differential_evolution(objective,BOUNDS,seed=31,popsize=10,maxiter=45,tol=1e-4,polish=False,workers=1)
    ls=least_squares(lambda p:residual(p,a,n,fit,cfg),de.x,bounds=np.array(BOUNDS).T,loss="soft_l1",f_scale=.25,max_nfev=250)
    fitted=ls.x; report={"warning":"No steering feedback: actuator and tire parameters are jointly identifiable only; independent-bag validation required before deployment.","parameters":NAMES,"previous":old.tolist(),"fitted":fitted.tolist(),"fit_previous":metrics(old,a,n,fit,cfg),"fit_candidate":metrics(fitted,a,n,fit,cfg),"temporal_validation_previous":metrics(old,a,n,val,cfg),"temporal_validation_candidate":metrics(fitted,a,n,val,cfg)}
    Path(args.out).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
