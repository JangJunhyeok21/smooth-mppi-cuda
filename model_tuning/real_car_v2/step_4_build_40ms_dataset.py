#!/usr/bin/env python3
"""Step 4: build latency-aware 40 ms transitions from audited 20 ms data."""
from pathlib import Path
import json,os,sys,numpy as np,yaml
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import actuator_step,longitudinal_actuator_step,Contract,IMU_RESIDUAL_FEATURES,OUTPUTS
SOURCE=Path(os.environ.get('DYNAMIC_SOURCE_DATA',ROOT/'model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz'));PARAMS=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'));OUT=Path(os.environ.get('DYNAMIC_RESIDUAL_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual.npz'))
MPPI_COMPUTE_LATENCY_S=float(os.environ.get('MPPI_COMPUTE_LATENCY_S','.025'))
MPPI_COMPUTE_LATENCY_RANGE_S=(.018,.031)
def main():
 d=np.load(SOURCE);x=d['features'].astype(float);observations=d['observations'].astype(float);derivative_targets=d['targets'].astype(float);teacher_vy=d['teacher_vy'].astype(float) if 'teacher_vy' in d.files else x[:,1];b=d['bag_id'];sp=d['split'];valid=d['valid'];source_dt=float(d['dt']);latency_steps=max(0,int(round(MPPI_COMPUTE_LATENCY_S/source_dt)));latency_quantized=latency_steps*source_dt
 if latency_quantized>=.04:raise ValueError('quantized MPPI compute latency must be shorter than model_dt=0.04 s')
 cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];c=Contract(dt=.04,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=float(cfg['min_accel']),max_accel=float(cfg['max_accel']));lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;dt=.04;features=[];targets=[];bags=[];splits=[];oks=[];rows=[];command_rows=[]
 # Runtime keeps this actuator state across MPPI solves. Reconstruct it over
 # each complete bag; resetting it to vx at every training window creates an
 # artificial command lag that the longitudinal residual is forced to hide.
 c_source=Contract(**{**c.__dict__,'dt':source_dt})
 source_speed_reference=np.empty(len(x),dtype=np.float32)
 for bag in np.unique(b):
  idx=np.flatnonzero(b==bag);sr=float(x[idx[0],0])
  for j in idx:
   source_speed_reference[j]=sr
   sr,_=longitudinal_actuator_step(sr,float(x[j,4]),float(x[j,0]),c_source)
 for i in range(len(x)-2):
  command_i=i+latency_steps
  ok=valid[i:i+3].all() and b[i]==b[i+1]==b[i+2] and sp[i]==sp[i+2]
  state=x[i,:3].copy();ap=float(x[i,5]);sr=float(source_speed_reference[i]);hist=x[command_i,10:20].copy();previous=x[max(i,command_i-1),3];cmd0=x[command_i,3:5]
  ap,_=actuator_step(ap,cmd0[0],cmd0[1],state[0],c);sr,bax=longitudinal_actuator_step(sr,cmd0[1],state[0],c);vx,vy,r=state;safe=max(abs(vx),.5);af=ap-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;front_inner=bf-fit['E_f']*(bf-np.arctan(bf));rear_inner=br-fit['E_r']*(br-np.arctan(br));fyf=fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(front_inner));fyr=fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(rear_inner));ay=(fyf*np.cos(ap)+fyr)/m;rd=(lf*fyf*np.cos(ap)-lr*fyr)/iz;state=np.array((vx+(bax+vy*r)*dt,vy+(ay-vx*r)*dt,r+rd*dt))
  # The state is what MPPI subscribed at t. The action is published after the
  # measured Orin NX solve latency, and only the remaining part of the 40 ms
  # knot can affect the endpoint. Average derivative supervision over that
  # command-effective interval instead of leaking pre-command acceleration.
  target=np.mean(derivative_targets[command_i:i+2],axis=0)
  # imu_ax/imu_ay are causal measurements at the subscribed state time. At
  # recursive runtime only the first knot is measured; later knots feed back
  # the model-predicted ax/ay stored in State.
  feature=np.r_[x[i,:3],cmd0,ap,cmd0[0]-previous,state,hist,observations[i,:2]]
  features.append(feature);targets.append(target);bags.append(b[i]);splits.append(sp[i]);oks.append(ok and np.isfinite(feature).all() and np.isfinite(target).all() and np.all(abs(target)<np.array((15,15,50))));rows.append(i);command_rows.append(command_i)
 input_contract=str(d['vy_input_contract']) if 'vy_input_contract' in d.files else 'unknown'
 payload=dict(features=np.asarray(features,np.float32),targets=np.asarray(targets,np.float32),teacher_confidence=(d['teacher_vy_confidence'][np.asarray(rows)+2].astype(np.float32) if 'teacher_vy_confidence' in d.files else np.ones(len(rows),np.float32)),bag_id=np.asarray(bags),split=np.asarray(splits),valid=np.asarray(oks),source_row=np.asarray(rows),command_source_row=np.asarray(command_rows),source_features=x.astype(np.float32),source_speed_reference=source_speed_reference,source_teacher_vy=teacher_vy.astype(np.float32),source_teacher_vy_confidence=(d['teacher_vy_confidence'].astype(np.float32) if 'teacher_vy_confidence' in d.files else np.ones(len(x),np.float32)),source_bag_id=b,source_split=sp,source_valid=valid,feature_names=np.array(IMU_RESIDUAL_FEATURES),target_names=np.array(OUTPUTS),control_dt=source_dt,model_dt=.04,mppi_compute_latency_requested_s=MPPI_COMPUTE_LATENCY_S,mppi_compute_latency_quantized_s=latency_quantized,mppi_compute_latency_range_s=np.asarray(MPPI_COMPUTE_LATENCY_RANGE_S),command_response_time_in_first_knot_s=.04-latency_quantized,alignment_contract="state_at_subscribe_time_command_at_publish_time",speed_reference_contract="persistent_runtime_state_reconstructed_per_bag",vy_input_contract=input_contract,vy_teacher_contract="offline_smoother_with_causal_fallback")
 if 'observations' in d.files:
  payload['source_observations']=d['observations'].astype(np.float32);payload['observation_names']=d['observation_names']
 np.savez_compressed(OUT,**payload);print(json.dumps({'output':str(OUT),'samples':len(features),'valid':int(np.sum(oks)),'split_valid':{str(k):int(np.sum(np.asarray(oks)&(np.asarray(splits)==k))) for k in range(3)},'mppi_compute_latency_requested_s':MPPI_COMPUTE_LATENCY_S,'mppi_compute_latency_quantized_s':latency_quantized,'command_response_time_in_first_knot_s':.04-latency_quantized,'alignment_contract':'state_at_subscribe_time_command_at_publish_time'},indent=2))
if __name__=='__main__':main()
