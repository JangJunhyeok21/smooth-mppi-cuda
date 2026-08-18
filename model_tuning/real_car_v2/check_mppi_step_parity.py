#!/usr/bin/env python3
"""Verify exact 1-step and 30-step Python/CUDA parity for the 40 ms lag model."""
from pathlib import Path
import importlib.util,json,subprocess,sys
import numpy as np,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from contract import Contract,actuator_step,longitudinal_actuator_step,residual_gates

# Current deployed adaptive-KF-compatible safety-recursive 20-D checkpoint.
RESULT=ROOT/'model_tuning/results/adaptive_kf_physical_20d_safety_recursive'
RUNTIME_BINARY=ROOT/'config/dynamic_40ms_residual_servo_lag.bin'
PARAMS=ROOT/'model_tuning/results/dynamic_40ms_adaptive_kf_regression/params.json'
CUDA_EXE=ROOT/'build/smppi_cuda_controller/mppi_step_parity'
SIMULATOR_MODEL=Path('/home/a/f1tenth_gym_ros/src/f1tenth_gym/f1tenth_gym/envs/dynamic_models/dynamic_mlp_residual.py')
STEPS=30;TOLERANCE=2e-5

def load_weights(path):
 z=np.fromfile(path,dtype='<f4');assert z.size==3563;o=0
 def take(n):
  nonlocal o;q=z[o:o+n];o+=n;return q
 return take(1280).reshape(64,20),take(64),take(2048).reshape(32,64),take(32),take(96).reshape(3,32),take(3),take(20),take(20)
def infer(feature,w):
 w1,b1,w2,b2,w3,b3,mean,std=w;h=np.maximum(((feature-mean)/std)@w1.T+b1,0);h=np.maximum(h@w2.T+b2,0);return np.clip(h@w3.T+b3,(-8,-8,-30),(8,8,30))
def python_rollout(cfg,fit,w):
 c=Contract(dt=.04,steer_scale=cfg['kinematic_steer_scale'],steer_bias=cfg['kinematic_steer_bias'],steer_tau=cfg['steer_servo_time_constant'],max_steer_rate=cfg['actuator_max_steer_rate'],speed_kp=cfg['speed_servo_kp'],speed_accel_tau=cfg['speed_reference_accel_time_constant'],speed_brake_tau=cfg['speed_reference_brake_time_constant'],max_speed_reference_rate=cfg['actuator_max_speed_reference_rate'],position_speed_scale=cfg['kinematic_position_speed_scale'],min_accel=cfg['min_accel'],max_accel=cfg['max_accel'],low_speed_center=cfg['dynamic_mlp_min_speed'])
 lf,lr,m,iz=[cfg[q] for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];fzf=m*9.81*lr/(lf+lr);fzr=m*9.81*lf/(lf+lr)
 state=np.array([1.,-.5,.3,2.2,.15,-.4,.2,-.1,np.arctan2(.15,2.2)]);hist=np.array([-.10,2.,-.05,2.2,0.,2.5,.08,2.8,.12,3.,.07,2.]);rows=[]
 for k in range(STEPS):
  steer=.25-.012*k;speed=np.clip(3.5-.025*k,cfg['min_speed'],cfg['max_speed']);previous=hist[8];previous_applied=hist[10];hist[:8]=hist[2:10];hist[8:10]=(steer,speed)
  applied,_=actuator_step(previous_applied,steer,speed,state[3],c);speed_ref,bax=longitudinal_actuator_step(hist[11],speed,np.hypot(state[3],state[4]),c);vx,vy,r=state[3:6];safe=max(abs(vx),.5);af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe)
  def force(fz,prefix,a):
   B,C,D,E=(fit[f'{q}_{prefix}'] for q in 'BCDE');ba=B*a;return fz*D*np.sin(C*np.arctan(ba-E*(ba-np.arctan(ba))))
  fyf=force(fzf,'f',af);fyr=force(fzr,'r',ar);ay=(fyf*np.cos(applied)+fyr)/m;rd=(lf*fyf*np.cos(applied)-lr*fyr)/iz;base=np.array([vx+(bax+vy*r)*.04,vy+(ay-vx*r)*.04,r+rd*.04]);feature=np.r_[state[3:6],steer,speed,applied,steer-previous,base,hist[:10]].astype(np.float32);res=infer(feature,w)*residual_gates(vx,c);body=base+res*.04
  beta=np.arctan2(body[1],body[0]);ns=np.empty(9);ns[3:6]=body;ns[6:8]=(bax+res[0],ay+res[1]);ns[8]=beta;ns[0]=state[0]+c.position_speed_scale*np.hypot(*body[:2])*np.cos(state[2]+beta)*.04;ns[1]=state[1]+c.position_speed_scale*np.hypot(*body[:2])*np.sin(state[2]+beta)*.04;ns[2]=(state[2]+body[2]*.04+np.pi)%(2*np.pi)-np.pi;hist[10:12]=(applied,speed_ref);state=ns;rows.append(np.r_[state,hist])
 return np.asarray(rows)
def simulator_rollout(cfg,fit,binary):
 spec=importlib.util.spec_from_file_location('simulator_dynamic_mlp_residual',SIMULATOR_MODEL);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 weights,mean,std=module.load_weights(binary)
 # Simulator state order: [x,y,steer_cmd,vx,yaw,yaw_rate,beta,vy].
 state=np.array([1.,-.5,0.,2.2,.3,-.4,np.arctan2(.15,2.2),.15],np.float32)
 history=np.array([-.10,2.,-.05,2.2,0.,2.5,.08,2.8,.12,3.],np.float32);applied=np.float32(.07);speed_reference=np.float32(2.);rows=[]
 kwargs=dict(dt=.04,lf=cfg['l_f'],lr=cfg['l_r'],mass=cfg['mass'],min_speed=cfg['min_speed'],max_speed=cfg['max_speed'],min_accel=cfg['min_accel'],max_accel=cfg['max_accel'],speed_servo_kp=cfg['speed_servo_kp'],speed_accel_tau=cfg['speed_reference_accel_time_constant'],speed_brake_tau=cfg['speed_reference_brake_time_constant'],max_speed_reference_rate=cfg['actuator_max_speed_reference_rate'],steer_scale=cfg['kinematic_steer_scale'],steer_bias=cfg['kinematic_steer_bias'],steer_time_constant=cfg['steer_servo_time_constant'],max_steer_rate=cfg['actuator_max_steer_rate'],position_speed_scale=cfg['kinematic_position_speed_scale'],Bf=fit['B_f'],Cf=fit['C_f'],Df=fit['D_f'],Ef=fit['E_f'],Br=fit['B_r'],Cr=fit['C_r'],Dr=fit['D_r'],Er=fit['E_r'],Iz=cfg['dynamic_mlp_I_z'],low_speed_center=cfg['dynamic_mlp_min_speed'])
 for k in range(STEPS):
  steer=.25-.012*k;speed=np.clip(3.5-.025*k,cfg['min_speed'],cfg['max_speed'])
  state,history,applied,speed_reference,imu,_=module.step(
      state,steer,speed,history,applied,speed_reference,
      weights,mean,std,**kwargs)
  canonical=np.array([state[0],state[1],state[4],state[3],state[7],state[5],imu[1],imu[2],state[6]])
  rows.append(np.r_[canonical,history,applied,speed_reference])
 return np.asarray(rows)
def main():
 cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];binary=RUNTIME_BINARY
 trained=RESULT/'command_history_20d.bin'
 if binary.read_bytes()!=trained.read_bytes():raise RuntimeError(f'runtime binary differs from current trained model: {binary} != {trained}')
 expected=python_rollout(cfg,fit,load_weights(binary));simulated=simulator_rollout(cfg,fit,binary);sim_error=np.abs(simulated-expected)
 simulator_report={'status':'PASS' if sim_error.max()<TOLERANCE else 'FAIL','one_step_max_abs_error':float(sim_error[0].max()),'thirty_step_max_abs_error':float(sim_error.max()),'mean_abs_error':float(sim_error.mean())}
 if simulator_report['status']!='PASS':print(json.dumps({'status':'FAIL','fixture_contract':'40ms_lag','simulator':simulator_report},indent=2));raise SystemExit(1)
 args=[str(CUDA_EXE),str(binary),str(STEPS),*[str(fit[f'{q}_{a}']) for a in ('f','r') for q in 'BCDE'],str(cfg['dynamic_mlp_I_z']),str(cfg['min_speed']),str(cfg['max_speed'])];run=subprocess.run(args,text=True,capture_output=True)
 if run.returncode:print(json.dumps({'status':'SIMULATOR_PASS_CUDA_NOT_RUN','returncode':run.returncode,'stderr':run.stderr.strip(),'fixture_contract':'40ms_lag','simulator':simulator_report},indent=2));raise SystemExit(run.returncode)
 actual=np.loadtxt(run.stdout.splitlines());error=np.abs(actual-expected);report={'status':'PASS' if error.max()<TOLERANCE else 'FAIL','fixture_contract':'40ms_lag','steps':STEPS,'one_step_max_abs_error':float(error[0].max()),'thirty_step_max_abs_error':float(error.max()),'mean_abs_error':float(error.mean()),'tolerance':TOLERANCE};print(json.dumps(report,indent=2));raise SystemExit(0 if report['status']=='PASS' else 1)
if __name__=='__main__':main()
