"""Heavy TorchScript GRU plant used only by the simulator."""
from pathlib import Path
import numpy as np
import torch

class SimulatorGRUPlant:
    def __init__(self,path,history_steps=50,dt=.02):
        # A single 1x50 recurrent inference is latency-bound. The default
        # machine-wide OpenMP thread pool made each simulator tick take up to
        # a second under MPPI load; one worker is substantially faster and has
        # deterministic latency for this small batch.
        torch.set_num_threads(1)
        self.model=torch.jit.load(str(Path(path)),map_location='cpu').eval();self.history_steps=int(history_steps);self.dt=np.float32(dt);self.reset()
    def reset(self):
        self.history=np.zeros((self.history_steps,9),np.float32);self.count=0;self.applied=np.float32(0);self.speed_reference=np.float32(0);self.accel=np.zeros(2,np.float32)
    def step(self,state,steer_cmd,speed_cmd,*,steer_scale,steer_bias,steer_tau,max_steer_rate,speed_accel_tau,speed_brake_tau,max_speed_reference_rate,position_speed_scale):
        s=np.asarray(state,np.float32);dt=self.dt;target=np.clip(steer_scale*steer_cmd+steer_bias,-.55,.55);rate=np.clip((target-self.applied)/max(steer_tau,1e-3),-max_steer_rate,max_steer_rate);self.applied=np.float32(np.clip(self.applied+rate*dt,-.55,.55));tau=speed_accel_tau if speed_cmd>=self.speed_reference else speed_brake_tau;self.speed_reference=np.float32(self.speed_reference+np.clip((speed_cmd-self.speed_reference)/max(tau,1e-3),-max_speed_reference_rate,max_speed_reference_rate)*dt);row=np.asarray((s[3],s[7],s[5],self.accel[0],self.accel[1],steer_cmd,speed_cmd,self.applied,self.speed_reference),np.float32)
        if self.count==0:self.history[:]=row
        else:self.history[:-1]=self.history[1:];self.history[-1]=row
        self.count+=1
        with torch.no_grad():ax,ay,next_r=self.model(torch.from_numpy(self.history[None])).numpy()[0]
        # No recorded plant can identify sub-centimetre standstill dynamics.
        # Prevent learned sensor bias from moving the simulated car before a
        # real speed command arrives.
        if abs(float(speed_cmd)) < .05 and abs(float(s[3])) < .10:
            out=s.copy();out[2]=steer_cmd;out[3]=out[5]=out[6]=out[7]=0.;self.accel.fill(0.);return out
        vx,vy=s[3],s[7];next_vx=np.float32(vx+(ax+vy*next_r)*dt);next_vy=np.float32(vy+(ay-vx*next_r)*dt);out=s.copy();out[0]=s[0]+position_speed_scale*(next_vx*np.cos(s[4])-next_vy*np.sin(s[4]))*dt;out[1]=s[1]+position_speed_scale*(next_vx*np.sin(s[4])+next_vy*np.cos(s[4]))*dt;out[2]=steer_cmd;out[3]=next_vx;out[4]=np.arctan2(np.sin(s[4]+next_r*dt),np.cos(s[4]+next_r*dt));out[5]=next_r;out[6]=np.arctan2(next_vy,abs(next_vx)+1e-5);out[7]=next_vy;self.accel[:]=(ax,ay);return out
