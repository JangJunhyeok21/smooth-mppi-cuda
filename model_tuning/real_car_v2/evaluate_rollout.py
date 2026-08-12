#!/usr/bin/env python3
"""Bag-disjoint 60-step state rollout: physics-only versus residual v2."""
import json,sys,argparse
from pathlib import Path
import numpy as np,torch,yaml
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from contract import Contract,actuator_step,longitudinal_actuator_step,low_speed_gate
from train import Net
DATA=ROOT/"model_tuning/data/real_car_v2_dynamic_residual.npz";DEFAULT_RESULT=ROOT/"model_tuning/results/real_car_v2_dynamic_residual";H=60
def main():
 parser=argparse.ArgumentParser();parser.add_argument("result",nargs="?",default=str(DEFAULT_RESULT));parser.add_argument("--kf-low-speed-threshold",type=float,default=None);args=parser.parse_args();result=Path(args.result)
 d=np.load(DATA);x=d["features"].astype(np.float32);bag=d["bag_id"];valid=d["valid"];test=d["split"]==2
 cfg=yaml.safe_load((ROOT/"config/params.yaml").read_text())["/**"]["ros__parameters"]
 model=Net();model.load_state_dict(torch.load(result/"model.pt",map_location="cpu",weights_only=True));model.eval()
 blob=np.fromfile(result/"dynamic_residual_v2.bin",dtype="<f4");mean=blob[-40:-20];std=blob[-20:]
 c=Contract(steer_scale=float(cfg["kinematic_steer_scale"]),steer_bias=float(cfg["kinematic_steer_bias"]),steer_tau=float(cfg["steer_servo_time_constant"]),max_steer_rate=float(cfg["actuator_max_steer_rate"]),speed_kp=float(cfg["speed_servo_kp"]),speed_accel_tau=float(cfg["speed_reference_accel_time_constant"]),speed_brake_tau=float(cfg["speed_reference_brake_time_constant"]),max_speed_reference_rate=float(cfg["actuator_max_speed_reference_rate"]),position_speed_scale=float(cfg["kinematic_position_speed_scale"]),min_accel=float(cfg["min_accel"]),max_accel=float(cfg["max_accel"]),low_speed_center=float(cfg["dynamic_mlp_min_speed"]))
 lf,lr,m,iz=map(float,(cfg["l_f"],cfg["l_r"],cfg["mass"],cfg["dynamic_mlp_I_z"]));wb=lf+lr
 Bf,Cf,Df,Ef=[float(cfg[f"dynamic_mlp_{q}"]) for q in ("B_f","C_f","D_f","E_f")];Br,Cr,Dr,Er=[float(cfg[f"dynamic_mlp_{q}"]) for q in ("B_r","C_r","D_r","E_r")]
 continuous=(bag[1:]==bag[:-1])&valid[1:]&valid[:-1]&(np.linalg.norm(x[1:,:3]-x[:-1,7:10],axis=1)<1.0)
 starts=[i for i in range(len(x)-H) if test[i] and test[i+H] and valid[i:i+H+1].all() and continuous[i:i+H].all()]
 def run(i,use_net):
  state=x[i,:3].astype(float).copy();applied=float(x[i,5]);speed_reference=float(state[0]);hist=x[i,10:20].reshape(5,2).copy();trace=[]
  for k in range(H):
   cmd=x[i+k,3:5].copy();cmd[1]=np.clip(cmd[1],float(cfg["min_speed"]),float(cfg["max_speed"]));previous=hist[-1,0];applied,_=actuator_step(applied,*cmd,state[0],c);speed_reference,base_ax=longitudinal_actuator_step(speed_reference,cmd[1],np.hypot(state[0],state[1]),c)
   vx,vy,r=state;safe=max(abs(vx),.5);af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb
   fyf=fzf*Df*np.sin(Cf*np.arctan(Bf*af-Ef*(Bf*af-np.arctan(Bf*af))));fyr=fzr*Dr*np.sin(Cr*np.arctan(Br*ar-Er*(Br*ar-np.arctan(Br*ar))));bay=(fyf*np.cos(applied)+fyr)/m;brd=(lf*fyf*np.cos(applied)-lr*fyr)/iz
   base=np.array([vx+(base_ax+vy*r)*c.dt,vy+(bay-vx*r)*c.dt,r+brd*c.dt]);feat=np.r_[state,cmd,applied,cmd[0]-previous,base,hist.ravel()].astype(np.float32)
   if use_net:
    with torch.no_grad():res=np.clip(model(torch.from_numpy((feat-mean)/std)).numpy(),[-8,-8,-30],[8,8,30])*low_speed_gate(vx,c)
   else:res=np.zeros(3)
   state=base+res*c.dt
   threshold=float(cfg["kf_low_speed_threshold"]) if args.kf_low_speed_threshold is None else args.kf_low_speed_threshold
   if threshold>0 and abs(state[0])<threshold: state[1]=0.0
   hist=np.vstack((hist[1:],cmd));trace.append(state.copy())
  return np.array(trace)
 # Cap evaluation count uniformly for runtime while retaining all test bags.
 starts=starts[::max(1,len(starts)//1500)];gt=np.stack([x[i+1:i+H+1,:3] for i in starts]);p=np.stack([run(i,False) for i in starts]);n=np.stack([run(i,True) for i in starts])
 def metrics(q):
  e=np.abs(q-gt);return {"windows":len(starts),"all_step_mae":e.mean((0,1)).tolist(),"final_1p2s_mae":e[:,-1].mean(0).tolist(),"final_1p2s_p95":np.quantile(e[:,-1],.95,axis=0).tolist()}
 threshold=float(cfg["kf_low_speed_threshold"]) if args.kf_low_speed_threshold is None else args.kf_low_speed_threshold
 report={"horizon_steps":H,"horizon_s":H*c.dt,"kf_low_speed_threshold":threshold,"states":["vx","vy","yaw_rate"],"physics_only":metrics(p),"residual_v2":metrics(n)}
 suffix=str(threshold).replace(".","p");(result/f"rollout_60step_metrics_kf_{suffix}.json").write_text(json.dumps(report,indent=2)+"\n");np.savez_compressed(result/f"rollout_60step_predictions_kf_{suffix}.npz",starts=starts,gt=gt,physics=p,residual=n);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
