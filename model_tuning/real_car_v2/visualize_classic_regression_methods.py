#!/usr/bin/env python3
"""Compare every classic-parameter regression method on identical GT windows."""
from pathlib import Path
import json,sys
import matplotlib.pyplot as plt
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from contract import Contract,actuator_step,longitudinal_actuator_step

DATA=ROOT/'model_tuning/data/dynamic_40ms_residual.npz'
PARAMS=ROOT/'model_tuning/results/dynamic_40ms_regression/advanced_params.json'
WINDOWS=ROOT/'model_tuning/results/dynamic_classic_residual_mlp_20d_stage2/rollout_30step_metrics.npz'
OUT=ROOT/'model_tuning/results/classic_regression_method_comparison';DT=.04;HORIZON=30
METHODS={
 'classic_current_8d':'current',
 'classic_de_robust_ls_8d':'de_robust_ls',
 'classic_adam_recursive_8d':'adam_differentiable',
 'classic_mlp_surrogate_8d':'mlp_surrogate',
}

def rollout(starts,x,cfg,parameters):
 lf,lr,m,iz=[float(cfg[q]) for q in ('l_f','l_r','mass','dynamic_mlp_I_z')];wb=lf+lr;fzf=m*9.81*lr/wb;fzr=m*9.81*lf/wb;c=Contract(dt=DT,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']),speed_kp=float(cfg['speed_servo_kp']),speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),position_speed_scale=float(cfg['kinematic_position_speed_scale']),min_accel=float(cfg['min_accel']),max_accel=float(cfg['max_accel']),low_speed_center=float(cfg['dynamic_mlp_min_speed']));Bf,Cf,Df,Ef,Br,Cr,Dr,Er=(parameters[q] for q in ('B_f','C_f','D_f','E_f','B_r','C_r','D_r','E_r'));traces=[];accelerations=[]
 for start in starts:
  state=x[start,:3].copy();applied=float(x[start,5]);speed_reference=float(state[0]);pose=np.zeros(3);trace=[np.r_[pose,state]];acc=[]
  for step in range(HORIZON):
   command=x[start+2*step,3:5];applied,_=actuator_step(applied,command[0],command[1],state[0],c);speed_reference,ax=longitudinal_actuator_step(speed_reference,command[1],state[0],c);vx,vy,r=state;safe=max(abs(vx),.5);alpha_front=applied-np.arctan2(vy+lf*r,safe);alpha_rear=-np.arctan2(vy-lr*r,safe);front=Bf*alpha_front;rear=Br*alpha_rear;front_inner=front-Ef*(front-np.arctan(front));rear_inner=rear-Er*(rear-np.arctan(rear));fyf=fzf*Df*np.sin(Cf*np.arctan(front_inner));fyr=fzr*Dr*np.sin(Cr*np.arctan(rear_inner));ay=(fyf*np.cos(applied)+fyr)/m;yaw_accel=(lf*fyf*np.cos(applied)-lr*fyr)/iz;state=np.array((vx+(ax+vy*r)*DT,vy+(ay-vx*r)*DT,r+yaw_accel*DT));acc.append((ax,ay));yaw=pose[2];pose=np.array((pose[0]+c.position_speed_scale*(state[0]*np.cos(yaw)-state[1]*np.sin(yaw))*DT,pose[1]+c.position_speed_scale*(state[0]*np.sin(yaw)+state[1]*np.cos(yaw))*DT,yaw+state[2]*DT));trace.append(np.r_[pose,state])
  traces.append(trace);accelerations.append(acc)
 return np.asarray(traces),np.asarray(accelerations)

def stat(value):return {'mean':float(np.mean(value)),'median':float(np.median(value)),'p95':float(np.quantile(value,.95)),'max':float(np.max(value))}
def metrics(prediction,truth,acceleration,gt_acceleration):
 return {'trajectory_m':stat(np.linalg.norm(prediction[:,-1,:2]-truth[:,-1,:2],axis=1)),'yaw_rad':stat(abs((prediction[:,-1,2]-truth[:,-1,2]+np.pi)%(2*np.pi)-np.pi)),'vx_mps':stat(abs(prediction[:,-1,3]-truth[:,-1,3])),'vy_mps':stat(abs(prediction[:,-1,4]-truth[:,-1,4])),'yaw_rate_rps':stat(abs(prediction[:,-1,5]-truth[:,-1,5])),'ax_mps2':stat(np.mean(abs(acceleration[:,:,0]-gt_acceleration[:,:,0]),axis=1)),'ay_mps2':stat(np.mean(abs(acceleration[:,:,1]-gt_acceleration[:,:,1]),axis=1))}

def main():
 data=np.load(DATA);window=np.load(WINDOWS);starts=window['starts'];truth=window['ground_truth'];x=data['source_features'].astype(float);observations=data['source_observations'];cfg=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters'];report_source=json.loads(PARAMS.read_text());gt_acceleration=np.asarray([observations[start+2*np.arange(HORIZON),:2] for start in starts]);predictions={};accelerations={};report={'window_count':len(starts),'horizon_s':HORIZON*DT,'selection_source':'identical held-out test starts','models':{}}
 for model,source_name in METHODS.items():
  parameters=report_source['methods'][source_name]['parameters'];prediction,acceleration=rollout(starts,x,cfg,parameters);predictions[model]=prediction;accelerations[model]=acceleration;report['models'][model]={'regression_source':source_name,'parameters':parameters,'metrics':metrics(prediction,truth,acceleration,gt_acceleration)}
 selected_name='classic_adam_recursive_8d';endpoint=np.linalg.norm(predictions[selected_name][:,-1,:2]-truth[:,-1,:2],axis=1);order=np.argsort(endpoint);selected=(order[0],order[len(order)//2],order[-1]);report['case_selection']={'model':selected_name,'metric':'final trajectory error','indices':[int(q) for q in selected]};OUT.mkdir(parents=True,exist_ok=True);(OUT/'classic_regression_method_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 colors=('tab:gray','tab:blue','tab:orange','tab:green');styles=(':','-.','--',(0,(3,1,1,1)));time=np.arange(HORIZON+1)*DT;accel_time=np.arange(HORIZON)*DT;fig,axes=plt.subplots(8,3,figsize=(18,28));labels=('Best','Median','Worst')
 for column,(case,index) in enumerate(zip(labels,selected)):
  axes[0,column].plot(truth[index,:,0],truth[index,:,1],'k-',lw=2.4,label='GT')
  for (name,_),color,style in zip(METHODS.items(),colors,styles):axes[0,column].plot(predictions[name][index,:,0],predictions[name][index,:,1],color=color,ls=style,lw=1.8,label=name)
  errors={name:float(np.linalg.norm(predictions[name][index,-1,:2]-truth[index,-1,:2])) for name in METHODS};axes[0,column].set_title(f'{case} · source row {int(starts[index])}\n'+ '\n'.join(f'{name}: {value:.3f} m' for name,value in errors.items()),fontsize=9);axes[0,column].set_aspect('equal',adjustable='datalim');axes[0,column].set_xlabel('relative x [m]');axes[0,column].set_ylabel('relative y [m]')
  for row,(signal,title,unit) in enumerate(((3,'$v_x$','m/s'),(4,'$v_y$','m/s'),(5,'yaw rate','rad/s'),(2,'yaw','rad')),1):
   axes[row,column].plot(time,truth[index,:,signal],'k-',lw=2.2,label='GT')
   for (name,_),color,style in zip(METHODS.items(),colors,styles):axes[row,column].plot(time,predictions[name][index,:,signal],color=color,ls=style,lw=1.7,label=name)
   axes[row,column].set_title(title);axes[row,column].set_ylabel(unit)
  for row,(channel,title) in enumerate(((0,'$a_x$'),(1,'$a_y$')),5):
   axes[row,column].plot(accel_time,gt_acceleration[index,:,channel],'k-',lw=2.2,label='GT IMU')
   for (name,_),color,style in zip(METHODS.items(),colors,styles):axes[row,column].plot(accel_time,accelerations[name][index,:,channel],color=color,ls=style,lw=1.7,label=name)
   axes[row,column].set_title(title);axes[row,column].set_ylabel('m/s²')
  rows=starts[index]+2*np.arange(HORIZON);axes[7,column].plot(accel_time,x[rows,3],color='tab:blue',label='steer command');right=axes[7,column].twinx();right.plot(accel_time,x[rows,4],'--',color='tab:green',label='speed command');axes[7,column].set_title('commands');axes[7,column].set_ylabel('steer [rad]');right.set_ylabel('speed [m/s]')
  for row,axis in enumerate(axes[:,column]):axis.grid(alpha=.25);axis.legend(fontsize=6.3,ncol=1);axis.set_xlabel('time [s]' if row else 'relative x [m]')
 fig.suptitle('Classic parameter regression methods · identical GT windows · residual disabled',y=.998,fontsize=16);fig.subplots_adjust(hspace=.52,wspace=.28,top=.965);fig.savefig(OUT/'classic_regression_methods_best_median_worst.png',dpi=180,bbox_inches='tight');plt.close(fig);np.savez_compressed(OUT/'classic_regression_method_traces.npz',starts=starts,ground_truth=truth,gt_acceleration=gt_acceleration,**{name:predictions[name] for name in METHODS},**{f'{name}_acceleration':accelerations[name] for name in METHODS});print(json.dumps(report,indent=2))
if __name__=='__main__':main()
