#!/usr/bin/env python3
"""Build 40 ms residual-derivative transitions from audited 20 ms /drive data."""
from pathlib import Path
import json,sys,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import actuator_step,longitudinal_actuator_step,Contract,FEATURES,OUTPUTS
SOURCE=ROOT/'model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz';PARAMS=ROOT/'model_tuning/results/dynamic_40ms_regression/params.json';OUT=ROOT/'model_tuning/data/dynamic_40ms_residual.npz'
def main():
 d=np.load(SOURCE);x=d['features'].astype(float);b=d['bag_id'];sp=d['split'];valid=d['valid'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];c=Contract(dt=.04,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=float(cfg['min_accel']),max_accel=float(cfg['max_accel']),low_speed_center=float(cfg['dynamic_mlp_min_speed']));lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;dt=.04;features=[];targets=[];bags=[];splits=[];oks=[];rows=[]
 for i in range(len(x)-2):
  ok=valid[i:i+3].all() and b[i]==b[i+1]==b[i+2] and sp[i]==sp[i+2]
  state=x[i,:3].copy();ap=float(x[i,5]);sr=float(state[0]);hist=x[i,10:20].copy();previous=hist[-4];cmd0=x[i,3:5]
  ap,_=actuator_step(ap,cmd0[0],cmd0[1],state[0],c);sr,bax=longitudinal_actuator_step(sr,cmd0[1],np.hypot(state[0],state[1]),c);vx,vy,r=state;safe=max(abs(vx),.5);af=ap-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;fyf=fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(bf));fyr=fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(br));ay=(fyf*np.cos(ap)+fyr)/m;rd=(lf*fyf*np.cos(ap)-lr*fyr)/iz;state=np.array((vx+(bax+vy*r)*dt,vy+(ay-vx*r)*dt,r+rd*dt))
  feature=np.r_[x[i,:3],cmd0,ap,cmd0[0]-previous,state,hist];target=(x[i+2,:3]-state)/.04
  features.append(feature);targets.append(target);bags.append(b[i]);splits.append(sp[i]);oks.append(ok and np.isfinite(feature).all() and np.isfinite(target).all() and np.all(abs(target)<np.array((15,15,50))));rows.append(i)
 np.savez_compressed(OUT,features=np.asarray(features,np.float32),targets=np.asarray(targets,np.float32),bag_id=np.asarray(bags),split=np.asarray(splits),valid=np.asarray(oks),source_row=np.asarray(rows),source_features=x.astype(np.float32),source_bag_id=b,source_split=sp,source_valid=valid,feature_names=np.array(FEATURES),target_names=np.array(OUTPUTS),control_dt=.02,model_dt=.04);print(json.dumps({'output':str(OUT),'samples':len(features),'valid':int(np.sum(oks)),'split_valid':{str(k):int(np.sum(np.asarray(oks)&(np.asarray(splits)==k))) for k in range(3)}},indent=2))
if __name__=='__main__':main()
