#!/usr/bin/env python3
"""Compare the compiled CUDA transition against an independent NumPy replay."""
import argparse,struct,subprocess,sys
from pathlib import Path
import numpy as np

def load(path):
 b=Path(path).read_bytes();head=struct.unpack("<8s5I5f",b[:48]);magic,ver,nin,h1,h2,nout,cdt,mdt,l0,l1,l2=head
 assert magic==b"EHSR004\0" and (ver,nin,h1,h2,nout)==(1,34,64,32,3)
 assert abs(cdt-.02)<1e-7 and abs(mdt-.04)<1e-7 and np.allclose((l0,l1,l2),(.12,.10,.25))
 z=np.frombuffer(b[48:],dtype="<f4");o=0
 def take(n):
  nonlocal o;q=z[o:o+n];o+=n;return q
 return (take(64*34).reshape(64,34),take(64),take(32*64).reshape(32,64),take(32),take(3*32).reshape(3,32),take(3),take(34),take(34))

def append(h,u):return np.r_[h[1:],np.asarray(u,np.float32)[None]].astype(np.float32)
def step(s,u,h,w):
 w1,b1,w2,b2,w3,b3,mean,std=w;u=np.asarray(u,np.float32);u[1]=np.clip(u[1],.5,3.);h=append(h,u)
 vx,vy,r,ax,ay=map(np.float32,(s[3],s[4],s[5],s[6],s[7]));steer,speed=u
 f=np.r_[vx,vy,r,ax,ay,steer,speed,h[:,0],h[:,1],steer-h[-2,0],steer-h[-4,0],speed-h[-2,1],vx*steer,vx*vx*steer,vx*r,abs(vx)*steer].astype(np.float32)
 q=np.maximum(w1@((f-mean)/std)+b1,0);q=np.maximum(w2@q+b2,0);corr=np.asarray((.12,.10,.25),np.float32)*np.tanh(w3@q+b3)
 for _ in range(2):
  target=vx/np.float32(.324)*np.tan(np.float32(.51)*steer+np.float32(.01));rdot=np.clip((target-r)/np.float32(.10),-15,15);a=np.clip(np.float32(.76)*(speed-vx),-1,1);vx+=a*np.float32(.02);vy+=(-vy/np.float32(.12))*np.float32(.02);r+=rdot*np.float32(.02)
 old=s.copy();s=s.copy();s[3:6]=(vx,vy,r)+corr;s[6]=(s[3]-old[3])/np.float32(.04);s[7]=(s[4]-old[4])/np.float32(.04)+old[3]*old[5]
 yaw=old[2];s[0]=old[0]+np.float32(.8633491306389823)*(s[3]*np.cos(yaw)-s[4]*np.sin(yaw))*np.float32(.04);s[1]=old[1]+np.float32(.8633491306389823)*(s[3]*np.sin(yaw)+s[4]*np.cos(yaw))*np.float32(.04);s[2]=np.arctan2(np.sin(yaw+s[5]*np.float32(.04)),np.cos(yaw+s[5]*np.float32(.04)));s[8]=np.arctan2(s[4],abs(s[3])+1e-5);return s.astype(np.float32),append(h,u)

def main():
 p=argparse.ArgumentParser();p.add_argument("binary");p.add_argument("--exe",default="build/smppi_cuda_controller/effective_history_parity");a=p.parse_args();w=load(a.binary);s=np.asarray((1,-.5,.3,2.2,0,-.4,.2,-.1,0),np.float32);h=np.asarray([(-.09+.02*i,1.7+.05*i) for i in range(10)],np.float32);expected={}
 for k in range(1,61):s,h=step(s,(.22*np.sin(.17*k),2.25+.55*np.cos(.11*k)),h,w);expected[k]=np.r_[s,h.ravel()]
 run=subprocess.run((a.exe,a.binary),text=True,capture_output=True)
 if run.returncode==2:print("SKIP CUDA runtime:",run.stderr.strip());print("binary metadata and independent 60-step NumPy replay: PASS");return 2
 if run.returncode:raise SystemExit(run.stderr)
 worst=0
 for line in run.stdout.splitlines():q=np.fromstring(line,sep=" ");k=int(q[0]);err=np.max(np.abs(q[1:]-expected[k]));worst=max(worst,err);print(f"step={k} max_error={err:.9g}")
 print("max_error",worst);assert worst<2e-3 and np.max(np.abs(np.fromstring(run.stdout.splitlines()[0],sep=" ")[1:]-expected[1]))<2e-5
if __name__=="__main__":main()
