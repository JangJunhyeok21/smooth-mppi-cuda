#!/usr/bin/env python3
"""Diagnostic alternative: identify a constrained four-parameter Pacejka model."""
from pathlib import Path
import json,numpy as np,yaml
from scipy.optimize import differential_evolution,least_squares
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz';OUT=ROOT/'model_tuning/results/dynamic_40ms_regression';SEED=31;H=25;MAX_PER_BAG=180
# B is a shape/stiffness factor, so allow a wider search instead of accepting
# an artificial B=15 boundary solution. D remains bounded because it scales
# peak lateral force and must not grow merely to compensate for observability.
BOUNDS=np.array(((.2,30.),(.15,2.8),(.2,30.),(.15,2.8)));C=1.3;E=0.
def expand(p):return np.array((p[0],C,p[1],E,p[2],C,p[3],E))
def starts(d,split):
 x,b,s,v=d['features'],d['bag_id'],d['split'],d['valid'];q=[]
 for bid in np.unique(b[s==split]):
  z=np.array([i for i in range(len(x)-2*H) if b[i]==bid and s[i]==split and v[i:i+2*H+1].all() and np.all(b[i:i+2*H+1]==bid) and np.mean(abs(x[i:i+2*H,0]))>.5])
  if len(z)>MAX_PER_BAG:z=z[np.linspace(0,len(z)-1,MAX_PER_BAG).astype(int)]
  q.extend(z[::3])
 return np.asarray(q,int)
def rollout(p,d,st,cfg):
 x=d['features'];n=len(st);state=x[st,:3].astype(float).copy();ap=x[st,5].astype(float).copy();sr=state[:,0].copy();pred=[];gt=[];Bf,Cf,Df,Ef,Br,Cr,Dr,Er=expand(p);lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;dt=.04
 for k in range(H):
  row=st+2*k;cmd=x[row,3:5];target=np.clip(float(cfg['kinematic_steer_scale'])*cmd[:,0]+float(cfg['kinematic_steer_bias']),-.55,.55);rate=np.clip((target-ap)/float(cfg['steer_servo_time_constant']),-float(cfg['actuator_max_steer_rate']),float(cfg['actuator_max_steer_rate']));ap=np.clip(ap+rate*dt,-.55,.55);speed=np.clip(cmd[:,1],float(cfg['min_speed']),4.);tau=np.where(speed>=sr,float(cfg['speed_reference_accel_time_constant']),float(cfg['speed_reference_brake_time_constant']));sr+=np.clip((speed-sr)/tau,-float(cfg['actuator_max_speed_reference_rate']),float(cfg['actuator_max_speed_reference_rate']))*dt;vx,vy,r=state.T;ax=np.clip(float(cfg['speed_servo_kp'])*(sr-vx),float(cfg['min_accel']),float(cfg['max_accel']));safe=np.maximum(abs(vx),.5);af=ap-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=Bf*af;br=Br*ar;fyf=fzf*Df*np.sin(Cf*np.arctan(bf-Ef*(bf-np.arctan(bf))));fyr=fzr*Dr*np.sin(Cr*np.arctan(br-Er*(br-np.arctan(br))));ay=(fyf*np.cos(ap)+fyr)/m;rd=(lf*fyf*np.cos(ap)-lr*fyr)/iz;state=np.c_[vx+(ax+vy*r)*dt,vy+(ay-vx*r)*dt,r+rd*dt]
  pred.append(state.copy());truth=x[st+2*(k+1),:3].copy()
  if 'teacher_vy' in d.files:truth[:,1]=d['teacher_vy'][st+2*(k+1)]
  gt.append(truth)
 return np.stack(pred,1),np.stack(gt,1)
def rel(q,scale):
 p=np.zeros((len(q),len(q[0]),3));dt=.04
 for k in range(q.shape[1]):
  old=p[:,k-1] if k else np.zeros((len(q),3));vx,vy,r=q[:,k].T;p[:,k,0]=old[:,0]+scale*(vx*np.cos(old[:,2])-vy*np.sin(old[:,2]))*dt;p[:,k,1]=old[:,1]+scale*(vx*np.sin(old[:,2])+vy*np.cos(old[:,2]))*dt;p[:,k,2]=old[:,2]+r*dt
 return p
def residual(p,d,st,cfg,reg=True):
 pr,gt=rollout(p,d,st,cfg);w=np.linspace(.25,1,H)[None,:,None];z=((pr-gt)*np.array((.4,2.,1.5))[None,None,:]*w).ravel();pp=rel(pr,1.0);gp=rel(gt,1.0);z=np.r_[z,2*(pp[:,-1,:2]-gp[:,-1,:2]).ravel()]
 if reg:z=np.r_[z,.04*(p-np.array((6,1.9,6,1.9)))/np.array((8,1,8,1)),.3*(np.diff(pr[:,:,2],2)-np.diff(gt[:,:,2],2)).ravel(),.5*max(0,p[0]*C*p[1]-p[2]*C*p[3])]
 return z
def metrics(p,d,st,cfg):
 pr,gt=rollout(p,d,st,cfg);e=abs(pr[:,-1]-gt[:,-1]);pe=np.linalg.norm(rel(pr,1.0)[:,-1,:2]-rel(gt,1.0)[:,-1,:2],axis=1);return {'windows':len(st),'state_mae':e.mean(0).tolist(),'state_p95':np.quantile(e,.95,axis=0).tolist(),'trajectory_mean_m':float(pe.mean()),'trajectory_p95_m':float(np.quantile(pe,.95))}
def main():
 OUT.mkdir(parents=True,exist_ok=True);d=np.load(DATA);cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];tr,va,te=(starts(d,i) for i in range(3));old=np.array((3.879070152566808,.0710062229162444,2.321287285513187,.05906540313616536));cost=lambda p:np.mean(np.minimum(abs(residual(p,d,tr,cfg)),.3)**2);de=differential_evolution(cost,BOUNDS,seed=SEED,popsize=8,maxiter=40,tol=8e-4,polish=False);ls=least_squares(lambda p:residual(p,d,tr,cfg),de.x,bounds=(BOUNDS[:,0],BOUNDS[:,1]),loss='soft_l1',f_scale=.3,max_nfev=220,verbose=1);fit=ls.x;tol=.01*(BOUNDS[:,1]-BOUNDS[:,0]);boundary={n:bool(abs(v-lo)<=t or abs(hi-v)<=t) for n,v,(lo,hi),t in zip(('B_f','D_f','B_r','D_r'),fit,BOUNDS,tol)};report={'model_dt':.04,'integration':'single Euler step at 0.04 s','parameter_names':['B_f','D_f','B_r','D_r'],'expanded_fitted':dict(zip(('B_f','C_f','D_f','E_f','B_r','C_r','D_r','E_r'),expand(fit).tolist())),'boundary_solution':boundary,'deployment_gate_passed':not any(boundary.values()),'old':{n:metrics(old,d,q,cfg) for n,q in zip(('train','validation','test'),(tr,va,te))},'fitted':{n:metrics(fit,d,q,cfg) for n,q in zip(('train','validation','test'),(tr,va,te))}};(OUT/'params.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
