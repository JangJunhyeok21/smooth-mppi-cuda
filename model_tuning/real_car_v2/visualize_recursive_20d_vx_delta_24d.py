#!/usr/bin/env python3
"""Compare recursive baseline_20d and vx_delta_history_24d on identical windows."""
from pathlib import Path
import json,os,sys
import matplotlib.pyplot as plt
import numpy as np,torch,yaml

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contract import Contract,actuator_step,longitudinal_actuator_step,residual_gates
BASE_NAME='dynamic_classic_residual_mlp_20d';NEW_NAME='vx_delta_history_24d'
BASE=Path(os.environ.get('BASELINE_EVALUATION',ROOT/'model_tuning/results/dynamic_classic_residual_mlp_20d_stage2/rollout_30step_metrics.npz'));NEW=Path(os.environ.get('VX_DELTA_MODEL',ROOT/'model_tuning/results/vx_delta_history_24d_stage2/model.pt'));DATA=Path(os.environ.get('DYNAMIC_RESIDUAL_DATA',ROOT/'model_tuning/data/dynamic_40ms_residual.npz'));PARAMS=Path(os.environ.get('DYNAMIC_CLASSIC_PARAMS',ROOT/'model_tuning/results/dynamic_40ms_regression/params.json'));OUT=Path(os.environ.get('VX_DELTA_COMPARISON_OUT',ROOT/'model_tuning/results/vx_delta_history_24d_stage2'));DT=.04

def infer(state,mean,std,feature):
 def array(key):return state[key].detach().cpu().numpy()
 h=np.maximum(((feature-mean)/std)@array('net.0.weight').T+array('net.0.bias'),0);h=np.maximum(h@array('net.2.weight').T+array('net.2.bias'),0);return np.clip(h@array('net.4.weight').T+array('net.4.bias'),(-8,-8,-30),(8,8,30))

def replay_new(starts,data,cfg,fit,checkpoint):
 x=data['source_features'].astype(float);state_dict=checkpoint['state_dict'];mean=checkpoint['mean'];std=checkpoint['std'];lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;c=Contract(dt=.04,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=float(cfg['min_accel']),max_accel=float(cfg['max_accel']),low_speed_center=float(cfg['dynamic_mlp_min_speed']));traces=[];accelerations=[]
 for start in starts:
  state=x[start,:3].copy();applied=float(x[start,5]);speed_reference=float(state[0]);commands=x[start-4:start+1,3:5].copy();speeds=x[start-8:start+1:2,0].copy();pose=np.zeros(3);trace=[np.r_[pose,state]];acc=[]
  for k in range(30):
   row=start+2*k;command=x[row,3:5]
   if k:commands=np.vstack((commands[1:],command))
   previous=commands[-2,0];current=state.copy();applied,_=actuator_step(applied,command[0],command[1],state[0],c);speed_reference,base_ax=longitudinal_actuator_step(speed_reference,command[1],state[0],c);vx,vy,r=state;safe=max(abs(vx),.5);af=applied-np.arctan2(vy+lf*r,safe);ar=-np.arctan2(vy-lr*r,safe);bf=fit['B_f']*af;br=fit['B_r']*ar;fi=bf-fit['E_f']*(bf-np.arctan(bf));ri=br-fit['E_r']*(br-np.arctan(br));fyf=fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(fi));fyr=fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(ri));base_ay=(fyf*np.cos(applied)+fyr)/m;base_rd=(lf*fyf*np.cos(applied)-lr*fyr)/iz;base_next=np.array((vx+(base_ax+vy*r)*DT,vy+(base_ay-vx*r)*DT,r+base_rd*DT));base_feature=np.r_[current,command,applied,command[0]-previous,base_next,commands.ravel()];feature=np.r_[base_feature,np.diff(speeds)] if len(mean)==24 else base_feature;res=infer(state_dict,mean,std,feature)*residual_gates(current[0],c);state=base_next+res*DT;speeds=np.r_[speeds[1:],state[0]];total_ax=base_ax+res[0];total_ay=base_ay+res[1];acc.append((total_ax,total_ay));yaw=pose[2];pose=np.array((pose[0]+c.position_speed_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*DT,pose[1]+c.position_speed_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*DT,yaw+state[2]*DT));trace.append(np.r_[pose,state])
  traces.append(trace);accelerations.append(acc)
 return np.asarray(traces),np.asarray(accelerations)

def stats(values):return {'mean':float(np.mean(values)),'p95':float(np.quantile(values,.95)),'max':float(np.max(values))}
def metrics(pred,truth,acc,gt_acc):
 return {'trajectory_m':stats(np.linalg.norm(pred[:,-1,:2]-truth[:,-1,:2],axis=1)),'yaw_rad':stats(np.abs((pred[:,-1,2]-truth[:,-1,2]+np.pi)%(2*np.pi)-np.pi)),'vx_mps':stats(np.abs(pred[:,-1,3]-truth[:,-1,3])),'vy_mps':stats(np.abs(pred[:,-1,4]-truth[:,-1,4])),'yaw_rate_rps':stats(np.abs(pred[:,-1,5]-truth[:,-1,5])),'ax_mps2':stats(np.mean(np.abs(acc[:,:,0]-gt_acc[:,:,0]),axis=1)),'ay_mps2':stats(np.mean(np.abs(acc[:,:,1]-gt_acc[:,:,1]),axis=1))}

def main():
 base=np.load(BASE);data=np.load(DATA);starts=base['starts'];baseline=base['predicted'];truth=base['ground_truth'];source=data['source_features'];obs=data['source_observations'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];fit=json.loads(PARAMS.read_text())['expanded_fitted'];checkpoint=torch.load(NEW,map_location='cpu',weights_only=False);new,new_acc=replay_new(starts,data,cfg,fit,checkpoint);base_acc=np.stack((np.diff(baseline[:,:,3],axis=1)/DT-baseline[:,:-1,4]*baseline[:,:-1,5],np.diff(baseline[:,:,4],axis=1)/DT+baseline[:,:-1,3]*baseline[:,:-1,5]),2);gt_acc=np.asarray([obs[start+2*np.arange(30),:2] for start in starts]);report={'models':{BASE_NAME:metrics(baseline,truth,base_acc,gt_acc),NEW_NAME:metrics(new,truth,new_acc,gt_acc)}}
 for field in report['models'][BASE_NAME]:
  old=report['models'][BASE_NAME][field]['mean'];value=report['models'][NEW_NAME][field]['mean'];report['models'][NEW_NAME][field]['change_vs_20d_percent']=100*(value/old-1)
 error=np.linalg.norm(new[:,-1,:2]-truth[:,-1,:2],axis=1);order=np.argsort(error);selected=(order[0],order[len(order)//2],order[-1]);report['selection']={'ranked_by':f'{NEW_NAME} final trajectory error','indices':[int(v) for v in selected]};OUT.mkdir(parents=True,exist_ok=True);(OUT/'recursive_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 t=np.arange(31)*DT;ta=np.arange(30)*DT;fig,axes=plt.subplots(8,3,figsize=(17,27));labels=('Best','Median','Worst')
 for column,(label,index) in enumerate(zip(labels,selected)):
  axes[0,column].plot(truth[index,:,0],truth[index,:,1],'k-',lw=2,label='GT');axes[0,column].plot(baseline[index,:,0],baseline[index,:,1],':',color='tab:blue',lw=2,label=BASE_NAME);axes[0,column].plot(new[index,:,0],new[index,:,1],'--',color='tab:orange',lw=2,label=NEW_NAME);axes[0,column].set_aspect('equal',adjustable='datalim');axes[0,column].set_title(f'{label} · row {int(starts[index])}\n{BASE_NAME}: {np.linalg.norm(baseline[index,-1,:2]-truth[index,-1,:2]):.3f} m\n{NEW_NAME}: {error[index]:.3f} m',fontsize=10);axes[0,column].set_xlabel('relative x [m]');axes[0,column].set_ylabel('relative y [m]')
  for row,(signal,title,unit) in enumerate(((3,'$v_x$','m/s'),(4,'$v_y$','m/s'),(5,'yaw rate','rad/s'),(2,'yaw','rad')),1):
   axes[row,column].plot(t,truth[index,:,signal],'k-',lw=2,label='GT');axes[row,column].plot(t,baseline[index,:,signal],':',color='tab:blue',lw=2,label=BASE_NAME);axes[row,column].plot(t,new[index,:,signal],'--',color='tab:orange',lw=2,label=NEW_NAME);axes[row,column].set_title(title);axes[row,column].set_ylabel(unit)
  for row,(channel,title) in enumerate(((0,'$a_x$'),(1,'$a_y$')),5):
   axes[row,column].plot(ta,gt_acc[index,:,channel],'k-',lw=2,label='GT IMU');axes[row,column].plot(ta,base_acc[index,:,channel],':',color='tab:blue',lw=2,label=BASE_NAME);axes[row,column].plot(ta,new_acc[index,:,channel],'--',color='tab:orange',lw=2,label=NEW_NAME);axes[row,column].set_title(title);axes[row,column].set_ylabel('m/s²')
  rows=starts[index]+2*np.arange(30);axes[7,column].plot(ta,source[rows,3],color='tab:blue',label='steer command');right=axes[7,column].twinx();right.plot(ta,source[rows,4],'--',color='tab:green',label='speed command');axes[7,column].set_title('commands');axes[7,column].set_ylabel('steer [rad]');right.set_ylabel('speed [m/s]')
  for row,axis in enumerate(axes[:,column]):axis.grid(alpha=.25);axis.set_xlabel('time [s]' if row else 'relative x [m]');axis.legend(fontsize=7)
 fig.suptitle(f'{BASE_NAME} vs {NEW_NAME} · identical held-out windows',y=.998);fig.subplots_adjust(hspace=.43,wspace=.28,top=.97);fig.savefig(OUT/'recursive_comparison_best_median_worst.png',dpi=180,bbox_inches='tight');plt.close(fig);np.savez_compressed(OUT/'recursive_comparison_traces.npz',starts=starts,ground_truth=truth,baseline_20d=baseline,vx_delta_history_24d=new,gt_acceleration=gt_acc,baseline_acceleration=base_acc,vx_delta_acceleration=new_acc);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
