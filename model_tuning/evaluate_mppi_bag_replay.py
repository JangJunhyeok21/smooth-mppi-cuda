#!/usr/bin/env python3
"""Replay recorded controls through the deployed slip MLP with MPPI runtime limits."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch,yaml
from torch import nn
from model_tuning_utils.train_rollout import prepare

ROOT=Path(__file__).resolve().parents[1]
DATASET_PATH=ROOT/'model_tuning/data/ifac0807_mppi_observation.npz'
RESULT_PATH=ROOT/'model_tuning/results/ifac0807_mppi_observation_slip_fixed_lossckpt'
OUTPUT_PATH=ROOT/'model_tuning/results/default_mppi_bag_replay'
PARAMS_PATH=ROOT/'config/params.yaml';HORIZON_STEPS=50;HISTORY_OFFSET=49
POSITION_SPEED_SCALE=None;KF_CF=None;KF_CR=None

def load_runtime_binary(path, net):
 data=np.fromfile(path,dtype=np.float32)
 n_in=net.net[0].in_features;expected=64*n_in+2243+2*n_in
 if data.size!=expected:raise ValueError(f'{path}: expected {expected} float32 values for {n_in}-D MLP, got {data.size}')
 offset=0
 state={}
 for key,shape in (("net.0.weight",(64,n_in)),("net.0.bias",(64,)),
                   ("net.2.weight",(32,64)),("net.2.bias",(32,)),
                   ("net.4.weight",(3,32)),("net.4.bias",(3,))):
  count=int(np.prod(shape));state[key]=torch.from_numpy(data[offset:offset+count].copy().reshape(shape));offset+=count
 net.load_state_dict(state)
 return data[offset:offset+n_in].copy(),data[offset+n_in:offset+2*n_in].copy()

class MLP(nn.Module):
 def __init__(self,n_in):
  super().__init__();self.net=nn.Sequential(nn.Linear(n_in,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))
 def forward(self,x):return self.net(x)

def main():
 p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',default=str(DATASET_PATH));p.add_argument('result',nargs='?',default=str(RESULT_PATH));p.add_argument('-o','--output',default=str(OUTPUT_PATH));p.add_argument('--params',default=str(PARAMS_PATH));p.add_argument('--horizon-steps',type=int,default=HORIZON_STEPS);p.add_argument('--history-offset',type=int,default=HISTORY_OFFSET,help='state offset before rollout; at least 5 for command history');p.add_argument('--position-speed-scale',type=float,default=POSITION_SPEED_SCALE);p.add_argument('--kf-cf',type=float,default=KF_CF);p.add_argument('--kf-cr',type=float,default=KF_CR)
 p.add_argument('--weights-bin',default=None,help='load weights and normalization from the deployed MPPI binary')
 p.add_argument('--disable-mlp',action='store_true',help='evaluate the selected classic base only');a=p.parse_args()
 if a.history_offset<5:raise SystemExit('--history-offset must be >=5')
 cfg=yaml.safe_load(Path(a.params).read_text())['/**']['ros__parameters'];z=np.load(a.dataset);raw=z['samples'];dt=float(z['dt']);result=Path(a.result);meta=json.loads((result/'metrics.json').read_text());mkf=meta.get('kf_cornering_stiffness_N_per_rad') or {};signs=meta.get('imu_axis_signs') or {}
 kf_cf=(a.kf_cf if a.kf_cf is not None else float(mkf.get('front',cfg['kf_cornering_stiffness_front'])));kf_cr=(a.kf_cr if a.kf_cr is not None else float(mkf.get('rear',cfg['kf_cornering_stiffness_rear'])))
 prep=argparse.Namespace(pose_window=21,horizon=a.horizon_steps*dt,history=a.history_offset+1,max_pose_step=.25,min_speed=.3,max_speed=10.,max_beta=.7,max_omega=8.,impact_decel=-10.,impact_margin=.5,min_windows=1,strict_no_imu=False,imu_wz_sign=float(signs.get('wz',cfg['imu_wz_sign'])),imu_ay_sign=float(signs.get('ay',cfg['imu_ay_sign'])),kf_cornering_stiffness_front=kf_cf,kf_cornering_stiffness_rear=kf_cr)
 pose,polar,_,_,starts,_,horizon=prepare(a.dataset,prep);body=np.c_[polar[:,0]*np.cos(polar[:,1]),polar[:,0]*np.sin(polar[:,1]),polar[:,2]].astype(np.float32)
 # A combined archive stores 0=train and 1=test in column 10. Evaluate only
 # fully held-out windows when a test split exists; never mix training replay
 # into reported deployment metrics.
 if raw.shape[1]>10 and np.any(raw[:,10]==1):
  last=starts+a.history_offset+horizon
  starts=starts[(raw[starts,10]==1)&(raw[last,10]==1)]
  if not len(starts):raise SystemExit('no held-out test windows survived preprocessing')
 if a.weights_bin:
  packed_size=np.fromfile(a.weights_bin,dtype=np.float32).size
  n_in=(packed_size-2243)//66
  if n_in<=0 or 66*n_in+2243!=packed_size:raise ValueError(f'{a.weights_bin}: cannot infer MLP input dimension from {packed_size} floats')
 else:
  norm=np.load(result/'normalization.npz');n_in=int(len(norm['base_mean'])+10)
 net=MLP(n_in)
 if a.weights_bin:
  packed_mean,packed_std=load_runtime_binary(a.weights_bin,net)
 else:
  packed_mean=np.r_[norm['base_mean'],np.tile(norm['command_mean'],5)].astype(np.float32);packed_std=np.r_[norm['base_std'],np.tile(norm['command_std'],5)].astype(np.float32);net.load_state_dict(torch.load(result/'model.pt',map_location='cpu',weights_only=True))
 net.eval()
 if meta.get('slip_yaw_source')=='imu':
  bag=raw[:,11].astype(int);w=float(signs.get('wz',cfg['imu_wz_sign']))*raw[:,12].astype(np.float32)
  for bid in np.unique(bag):
   ii=np.flatnonzero(bag==bid)
   for k in range(1,len(ii)):w[ii[k]]=.25*w[ii[k]]+.75*w[ii[k-1]]
  body[:,2]=w
 bag=raw[:,11].astype(int);imu=raw[:,12:15].astype(np.float32)*np.array([float(signs.get('wz',cfg['imu_wz_sign'])),float(signs.get('ax',cfg['imu_ax_sign'])),float(signs.get('ay',cfg['imu_ay_sign']))],np.float32)
 for bid in np.unique(bag):
  ii=np.flatnonzero(bag==bid)
  for kk in range(1,len(ii)):imu[ii[kk]]=.25*imu[ii[kk]]+.75*imu[ii[kk-1]]
 mean=torch.tensor(packed_mean);std=torch.tensor(packed_std);scale=torch.tensor([8.,8.,30.])
 command=raw[:,[7,9]].astype(np.float32);limits=meta.get('runtime_speed_limits_mps',[cfg['min_speed'],cfg['max_speed']]);minv,maxv=map(float,limits);steer_meta=meta.get('steering_command_mapping',{});sa=float(steer_meta.get('scale',cfg['kinematic_steer_scale']));sb=float(steer_meta.get('bias_rad',cfg['kinematic_steer_bias']));act=meta.get('actuator_model',{});direct_steer=bool(act.get('direct_steer',False));tau=float(act.get('servo_time_constant_s',0.));maxrate=float(act.get('max_steering_rate_rad_s') if act.get('max_steering_rate_rad_s') is not None else float('inf'));mina,maxa=float(cfg['min_accel']),float(cfg['max_accel']);kp=float(meta.get('kp_speed',cfg['speed_servo_kp']));wb=float(cfg['l_f'])+float(cfg['l_r']);pscale=(a.position_speed_scale if a.position_speed_scale is not None else float(meta.get('position_speed_scale',cfg.get('kinematic_position_speed_scale',1.0))))
 yawact=meta.get('yaw_rate_actuator_model',{});yawtau=float(yawact.get('time_constant_s',cfg.get('kinematic_yaw_rate_time_constant',.1)));maxyawacc=float(yawact.get('max_yaw_accel_radps2',cfg.get('kinematic_max_yaw_accel',15.)))
 effective=np.empty(len(raw),np.float32);bag=raw[:,11].astype(int)
 for bid in np.unique(bag):
  ii=np.flatnonzero(bag==bid);effective[ii[0]]=np.clip(command[ii[0],0] if direct_steer else sa*command[ii[0],0]+sb,-.55,.55)
  for kk in range(1,len(ii)):
   target=np.clip(sa*command[ii[kk],0]+sb,-.55,.55);rate=np.clip((target-effective[ii[kk-1]])/max(tau,1e-6),-maxrate,maxrate) if tau>0 and not direct_steer else 0.;effective[ii[kk]]=effective[ii[kk-1]]+rate*dt if tau>0 and not direct_steer else np.clip(command[ii[kk-1],0],-.55,.55)
 all_pred=[];batch=512
 with torch.no_grad():
  for q in range(0,len(starts),batch):
   ids=starts[q:q+batch];j0=ids+a.history_offset;b=len(ids);uall=torch.tensor(command);hist=torch.stack([uall[j0-d] for d in range(5,0,-1)],1);hist[:,:,1].clamp_(minv,maxv)
   for _ in (None,):
    s=torch.tensor(body[j0]);xyq=torch.tensor(pose[j0],dtype=torch.float32);tr=[torch.cat((xyq,s),1)]
    h=hist.clone();delta=torch.tensor(effective[j0]);axay=torch.tensor(imu[j0,1:3])
    for k in range(horizon):
     u=uall[j0+k].clone();u[:,0].clamp_(-float(cfg['max_steer']),float(cfg['max_steer']));u[:,1].clamp_(minv,maxv);target=torch.clamp(sa*u[:,0]+sb,-.55,.55);delta=(delta+torch.clamp((target-delta)/tau,-maxrate,maxrate)*dt) if tau>0 and not direct_steer else torch.clamp(h[:,-1,0],-.55,.55);steer=delta
     speed=torch.hypot(s[:,0],s[:,1]);beta=torch.atan2(s[:,1],s[:,0]);baseax=torch.clamp(kp*(u[:,1]-speed),mina,maxa)
     # Runtime min/max constrain the command above, not the predicted state.
     basespeed=speed+baseax*dt
     if meta.get('model')=='dynamic_residual':
      dp=meta['dynamic_classic_params'];vxs=torch.clamp(torch.abs(s[:,0]),min=.5);af=steer-torch.atan2(s[:,1]+float(cfg['l_f'])*s[:,2],vxs);ar=-torch.atan2(s[:,1]-float(cfg['l_r'])*s[:,2],vxs);fzf=dp['mass']*9.81*float(cfg['l_r'])/wb;fzr=dp['mass']*9.81*float(cfg['l_f'])/wb;fyf=fzf*dp['Df']*torch.sin(dp['Cf']*torch.atan(dp['Bf']*af-dp['Ef']*(dp['Bf']*af-torch.atan(dp['Bf']*af))));fyr=fzr*dp['Dr']*torch.sin(dp['Cr']*torch.atan(dp['Br']*ar-dp['Er']*(dp['Br']*ar-torch.atan(dp['Br']*ar))));cay=(fyf*torch.cos(steer)+fyr)/dp['mass'];classic=torch.stack((s[:,0]+(baseax+s[:,1]*s[:,2])*dt,s[:,1]+(cay-s[:,0]*s[:,2])*dt,s[:,2]+(float(cfg['l_f'])*fyf*torch.cos(steer)-float(cfg['l_r'])*fyr)/dp['Iz']*dt),1)
     else:
      basevx=basespeed*torch.cos(beta);basevy=basespeed*torch.sin(beta);targetw=basevx*torch.tan(steer)/wb;basew=s[:,2]+torch.clamp((targetw-s[:,2])/yawtau,-maxyawacc,maxyawacc)*dt;classic=torch.stack((basevx,basevy,basew),1)
     if meta.get('model') in ('dynamic_imu','e2e_mlp'):
      basew=s[:,0]*torch.tan(steer)/wb;baseay=s[:,0]*basew;fbase=torch.cat((s,axay,u,baseax[:,None],baseay[:,None],basew[:,None]),1);f=torch.cat((fbase,h.reshape(b,-1)),1);pred=torch.tanh(net((f-mean)/std))*scale;next_axay=pred[:,:2];nw=pred[:,2];s=torch.stack((s[:,0]+(next_axay[:,0]+s[:,1]*s[:,2])*dt,s[:,1]+(next_axay[:,1]-s[:,0]*s[:,2])*dt,nw),1);axay=next_axay
     else:
      if n_in==18:fbase=torch.cat((s,u,classic),1)
      elif n_in==20:fbase=torch.cat((s,u,delta[:,None],(u[:,0]-h[:,-1,0])[:,None],classic),1)
      else:raise RuntimeError(f'unsupported {n_in}-D residual feature layout in bag replay')
      f=torch.cat((fbase,h.reshape(b,-1)),1);corr=(torch.zeros((b,3)) if a.disable_mlp else torch.tanh(net((f-mean)/std))*scale);s=classic+corr*dt
     sn=torch.hypot(s[:,0],s[:,1]);bn=torch.atan2(s[:,1],s[:,0]);xyq=torch.stack((xyq[:,0]+pscale*sn*torch.cos(xyq[:,2]+bn)*dt,xyq[:,1]+pscale*sn*torch.sin(xyq[:,2]+bn)*dt,xyq[:,2]+s[:,2]*dt),1);tr.append(torch.cat((xyq,s),1));h=torch.cat((h[:,1:],u[:,None]),1)
    all_pred.append(torch.stack(tr,1).numpy())
 gt=np.stack([pose[starts+a.history_offset+k] for k in range(horizon+1)],1);metrics={};arrays={'starts':starts,'gt_pose':gt}
 for label in ('current_cuda_unclamped_state',):
  pred=np.concatenate(all_pred);arrays[label]=pred
  points={}
  for step in (min(50,horizon),horizon):
   e=np.linalg.norm(pred[:,step,:2]-gt[:,step,:2],axis=1);yaw=np.arctan2(np.sin(pred[:,step,2]-gt[:,step,2]),np.cos(pred[:,step,2]-gt[:,step,2]))
   gt_speed=polar[starts+a.history_offset+step,0];gt_w=polar[starts+a.history_offset+step,2];ps=np.hypot(pred[:,step,3],pred[:,step,4])
   points[f'{step*dt:.2f}s']={'trajectory_mean_m':float(e.mean()),'trajectory_median_m':float(np.median(e)),'trajectory_p95_m':float(np.quantile(e,.95)),'trajectory_worst_m':float(e.max()),'speed_mae_mps':float(np.mean(abs(ps-gt_speed))),'yaw_rate_mae_radps':float(np.mean(abs(pred[:,step,5]-gt_w))),'yaw_mae_deg':float(np.degrees(np.mean(abs(yaw))))}
  pred_dot_vx=np.diff(pred[:,:,3],axis=1)/dt;pred_dot_vy=np.diff(pred[:,:,4],axis=1)/dt
  rows=starts[:,None]+a.history_offset+np.arange(1,horizon+1)[None,:]
  gt_vx=raw[rows,4];gt_vy=body[rows,1];gt_w=body[rows,2]
  gt_dot_vx=imu[rows,1]+gt_vy*gt_w;gt_dot_vy=imu[rows,2]-gt_vx*gt_w
  def mae(x,y):return float(np.mean(np.abs(x-y)))
  points['open_loop_all_steps']={'dot_vx_mae_mps2':mae(pred_dot_vx,gt_dot_vx),'dot_vy_mae_mps2':mae(pred_dot_vy,gt_dot_vy),'vx_mae_mps':mae(pred[:,1:,3],gt_vx),'vy_mae_mps':mae(pred[:,1:,4],gt_vy),'yaw_rate_mae_radps':mae(pred[:,1:,5],gt_w)}
  metrics[label]=points
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');np.savez_compressed(out/'predictions.npz',**arrays)
 import matplotlib.pyplot as plt
 labels=list(metrics);keys=('trajectory_mean_m','trajectory_p95_m','speed_mae_mps','yaw_rate_mae_radps');fig,axes=plt.subplots(2,2,figsize=(11,8))
 for ax,key in zip(axes.flat,keys):
  x=np.arange(len(labels));w=.35
  for j,t in enumerate((f'{min(50,horizon)*dt:.2f}s',f'{horizon*dt:.2f}s')):ax.bar(x+(j-.5)*w,[metrics[n][t][key] for n in labels],w,label=t)
  ax.set_xticks(x,('Unclamped state',));ax.set_title(key);ax.legend();ax.grid(axis='y',alpha=.25)
 fig.tight_layout();fig.savefig(out/'runtime_replay_comparison.png',dpi=180);plt.close(fig);print(json.dumps(metrics,indent=2))
 pred=arrays['current_cuda_unclamped_state'];final_error=np.linalg.norm(pred[:,-1,:2]-gt[:,-1,:2],axis=1);order=np.argsort(final_error)
 chosen=(order[0],order[len(order)//2],order[-1]);titles=('Best','Median','Worst')
 fig,axes=plt.subplots(4,3,figsize=(15,14))
 for col,(index,title) in enumerate(zip(chosen,titles)):
  axes[0,col].plot(gt[index,:,0],gt[index,:,1],'k-',lw=2,label='GT');axes[0,col].plot(pred[index,:,0],pred[index,:,1],'--',color='tab:orange',lw=2,label='Config .bin prediction');axes[0,col].set_title(f'{title}: {final_error[index]:.3f} m');axes[0,col].axis('equal');axes[0,col].grid(alpha=.25);axes[0,col].legend()
  t=np.arange(horizon+1)*dt;axes[1,col].plot(t,np.hypot(pred[index,:,3],pred[index,:,4]),'b--',label='Predicted speed');axes[1,col].plot(t,polar[starts[index]+a.history_offset:starts[index]+a.history_offset+horizon+1,0],'b-',label='GT speed');axes[1,col].plot(t,pred[index,:,5],'g--',label='Predicted yaw rate');axes[1,col].plot(t,polar[starts[index]+a.history_offset:starts[index]+a.history_offset+horizon+1,2],'g-',label='GT yaw rate');axes[1,col].set_xlabel('Time [s]');axes[1,col].grid(alpha=.25);axes[1,col].legend(fontsize=8)
  row0=starts[index]+a.history_offset;gt_vy_case=body[row0:row0+horizon+1,1]
  vy_mae=float(np.mean(np.abs(pred[index,:,4]-gt_vy_case)))
  axes[2,col].plot(t,gt_vy_case,color='tab:blue',lw=2,label=r'GT $v_y$')
  axes[2,col].plot(t,pred[index,:,4],'--',color='tab:orange',lw=2,label=r'Predicted $v_y$')
  axes[2,col].set_title(rf'$v_y$ MAE: {vy_mae:.3f} m/s');axes[2,col].set_ylabel(r'$v_y$ [m/s]');axes[2,col].set_xlabel('Time [s]');axes[2,col].grid(alpha=.25);axes[2,col].legend(fontsize=8)
  # Body-frame identity used by both training and CUDA: ay=d(vy)/dt+vx*r.
  pred_ay=np.diff(pred[index,:,4])/dt+pred[index,:-1,3]*pred[index,:-1,5]
  gt_ay=imu[row0+1:row0+horizon+1,2];ay_mae=float(np.mean(np.abs(pred_ay-gt_ay)))
  axes[3,col].plot(t[1:],gt_ay,color='tab:blue',lw=2,label=r'GT IMU $a_y$')
  axes[3,col].plot(t[1:],pred_ay,'--',color='tab:orange',lw=2,label=r'Predicted $a_y$')
  axes[3,col].set_title(rf'$a_y$ MAE: {ay_mae:.3f} m/s$^2$');axes[3,col].set_ylabel(r'$a_y$ [m/s$^2$]');axes[3,col].set_xlabel('Time [s]');axes[3,col].grid(alpha=.25);axes[3,col].legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/'config_bin_best_median_worst.png',dpi=180);plt.close(fig)
 pred=arrays['current_cuda_unclamped_state'];rows=starts[:,None]+a.history_offset+np.arange(1,horizon+1)[None,:]
 gt_vx=raw[rows,4];gt_vy=body[rows,1];gt_w=body[rows,2];gt_dot_vx=imu[rows,1]+gt_vy*gt_w;gt_dot_vy=imu[rows,2]-gt_vx*gt_w
 signals=((np.diff(pred[:,:,3],axis=1)/dt,gt_dot_vx,r'$\dot v_x$ [m/s$^2$]'),(np.diff(pred[:,:,4],axis=1)/dt,gt_dot_vy,r'$\dot v_y$ [m/s$^2$]'),(pred[:,1:,3],gt_vx,r'$v_x$ [m/s]'),(pred[:,1:,4],gt_vy,r'$v_y$ [m/s]'),(pred[:,1:,5],gt_w,'yaw rate [rad/s]'))
 fig,axes=plt.subplots(3,2,figsize=(12,10));axes.flat[-1].axis('off');t=np.arange(1,horizon+1)*dt
 for axis,(prediction,target,title) in zip(axes.flat,signals):axis.plot(t,np.mean(np.abs(prediction-target),axis=0));axis.set_title(title+' open-loop MAE');axis.set_xlabel('Time [s]');axis.grid(alpha=.25)
 fig.tight_layout();fig.savefig(out/'open_loop_state_derivative_mae.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
