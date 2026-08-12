#!/usr/bin/env python3
"""Compare the training step with the exact CUDA MPPI device step.

Run after colcon build on a CUDA-visible machine. The CUDA executable calls
update_dynamic_mlp_residual itself; this script independently evaluates the
training PyTorch model and compares the complete next state/history.
"""
from pathlib import Path
import json, subprocess, sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from train import Net
RESULT=ROOT/"model_tuning/results/real_car_v2_dynamic_residual_all6_seed31"
CUDA_EXE=ROOT/"build/smppi_cuda_controller/mppi_step_parity"

def training_step():
    dt=np.float64(.02);mass=3.74;lf=.163;lr=.161;wheelbase=lf+lr
    state=np.array([1.,-.5,.3,2.2,.15,-.4,.2,-.1,np.arctan2(.15,2.2)])
    steer_cmd,speed_cmd=.25,min(2.0,3.5)
    history=np.array([-.10,2.,-.05,2.2,0.,2.5,.08,2.8,.12,3.,.07,2.])
    target=np.clip(.50927964*steer_cmd+.01015773,-.55,.55)
    steer_rate=np.clip((target-history[10])/.15514851356820727,-.8344090950084138,.8344090950084138)
    steer=np.clip(history[10]+steer_rate*dt,-.55,.55)
    tau=.04 if speed_cmd>=history[11] else .02
    ref_rate=np.clip((speed_cmd-history[11])/tau,-8.,8.)
    speed_ref=history[11]+ref_rate*dt
    current_speed=np.hypot(state[3],state[4])
    base_ax=np.clip(.7616888694734905*(speed_ref-current_speed),-1.,1.)
    safe=max(abs(state[3]),.5)
    af=steer-np.arctan2(state[4]+lf*state[5],safe);ar=-np.arctan2(state[4]-lr*state[5],safe)
    fzf=mass*9.81*lr/wheelbase;fzr=mass*9.81*lf/wheelbase
    def force(fz,B,C,D,E,a):
        ba=B*a;return fz*D*np.sin(C*np.arctan(ba-E*(ba-np.arctan(ba))))
    fyf=force(fzf,3.879070152566808,1.6471076687680233,.0710062229162444,-1.,af)
    fyr=force(fzr,2.321287285513187,1.9234527357451916,.05906540313616536,-1.,ar)
    base_ay=(fyf*np.cos(steer)+fyr)/mass
    base_rdot=(lf*fyf*np.cos(steer)-lr*fyr)/.04712
    base=np.array([state[3]+(base_ax+state[4]*state[5])*dt,
                   state[4]+(base_ay-state[3]*state[5])*dt,
                   state[5]+base_rdot*dt])
    feature=np.r_[state[3:6],steer_cmd,speed_cmd,steer,steer_cmd-history[8],base,history[:10]].astype(np.float32)
    blob=np.fromfile(RESULT/"dynamic_residual_v2.bin",dtype="<f4");mean,std=blob[-40:-20],blob[-20:]
    model=Net();model.load_state_dict(torch.load(RESULT/"model.pt",map_location="cpu",weights_only=True));model.eval()
    with torch.no_grad(): residual=model(torch.from_numpy((feature-mean)/std)).numpy().astype(float)
    residual=np.clip(residual,[-8,-8,-30],[8,8,30])/(1+np.exp(-(abs(state[3])-.8)/.2))
    next_body=base+residual*dt
    next_speed=np.hypot(next_body[0],next_body[1]);beta=np.arctan2(next_body[1],next_body[0])
    out=np.array([state[0]+.8633491306389823*next_speed*np.cos(state[2]+beta)*dt,
      state[1]+.8633491306389823*next_speed*np.sin(state[2]+beta)*dt,
      (state[2]+next_body[2]*dt+np.pi)%(2*np.pi)-np.pi,*next_body,
      base_ax+residual[0],base_ay+residual[1],beta])
    next_history=np.r_[history[2:10],steer_cmd,speed_cmd,steer,speed_ref]
    return np.r_[out,next_history]

def main():
    expected=training_step()
    run=subprocess.run([str(CUDA_EXE),str(RESULT/"dynamic_residual_v2.bin"),"lag"],text=True,capture_output=True)
    if run.returncode:
        print(json.dumps({"status":"CUDA_NOT_RUN","returncode":run.returncode,"stderr":run.stderr.strip(),
                          "training_step":expected.tolist()},indent=2));raise SystemExit(run.returncode)
    actual=np.fromstring(run.stdout,sep=" ")
    error=np.abs(actual-expected)
    report={"status":"PASS" if error.max()<2e-5 else "FAIL","values":int(len(actual)),
            "max_abs_error":float(error.max()),"mean_abs_error":float(error.mean()),
            "training_step":expected.tolist(),"mppi_cuda_step":actual.tolist()}
    print(json.dumps(report,indent=2));raise SystemExit(0 if report["status"]=="PASS" else 1)
if __name__=="__main__":main()
