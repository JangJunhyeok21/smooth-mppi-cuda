#!/usr/bin/env python3
"""Visualize causal adaptive-KF tire slip angles on one extracted real-car bag."""
from pathlib import Path
import json,sys
import matplotlib.pyplot as plt
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset
from contract import Contract,actuator_step

DATASET=ROOT/'model_tuning/data/ifac0817_0818_autonomous_physics_clean/rosbag2_2026_08_17-23_57_13.npz'
BAG_DB=Path('/mnt/nas_custom/F1tenth/2026 IFAC/0817 (1)/rosbag2_2026_08_17-23_57_13/rosbag2_2026_08_17-23_57_13_0.db3')
PARAMS=ROOT/'config/params.yaml'
OUTPUT=ROOT/'model_tuning/results/slip_analysis_0817_235713'
OVERSTEER_REAR_MIN_DEG=3.0
OVERSTEER_EXCESS_DEG=2.0
OVERSTEER_MIN_SPEED=1.5

def stats(v):return {'mean_abs_deg':float(np.mean(np.abs(v))),'p95_abs_deg':float(np.quantile(np.abs(v),.95)),'p99_abs_deg':float(np.quantile(np.abs(v),.99)),'max_abs_deg':float(np.max(np.abs(v)))}

def stitched_time(raw_t, segment_id, dt):
 out=np.zeros_like(raw_t);offset=0.0
 for segment in np.unique(segment_id):
  idx=np.flatnonzero(segment_id==segment);local=raw_t[idx]-raw_t[idx[0]]
  out[idx]=offset+local;offset=out[idx[-1]]+dt
 return out

def load_full_newmcl_pose():
 import rosbag2_py
 from rclpy.serialization import deserialize_message
 from rosidl_runtime_py.utilities import get_message
 reader=rosbag2_py.SequentialReader()
 reader.open(rosbag2_py.StorageOptions(uri=str(BAG_DB),storage_id='sqlite3'),
             rosbag2_py.ConverterOptions('cdr','cdr'))
 topic_types={item.name:item.type for item in reader.get_all_topics_and_types()}
 message_type=get_message(topic_types['/newmcl_pose']);pose=[]
 while reader.has_next():
  topic,data,record_ns=reader.read_next()
  if topic!='/newmcl_pose':continue
  msg=deserialize_message(data,message_type);stamp=msg.header.stamp
  header_ns=int(stamp.sec)*1_000_000_000+int(stamp.nanosec)
  pose.append((header_ns if header_ns>0 else record_ns,msg.pose.position.x,msg.pose.position.y))
 return np.asarray(pose,dtype=float)

def load_raw_continuous_before_collision():
 """Causally align the raw bag and trim only the final impact-like event."""
 from model_tuning.extract_training_data import read_streams,causal_hold
 from model_tuning_utils.filter_collision_recovery_episodes import physical_inconsistency_mask
 pose,velocity,command,imu,applied=read_streams(
     BAG_DB,'/newmcl_pose','/odom','/ackermann_cmd','/imu/data','/drive')
 streams=(pose,velocity,command,imu,applied)
 start=max(v[0,0] for v in streams);end=min(v[-1,0] for v in streams);dt=.02
 times=np.arange(start,end,dt);aligned=[];valid=[]
 for stream,max_age in zip(streams,(.10,.10,.10,.05,.10)):
  held,ok=causal_hold(stream,times,max_age);aligned.append(held);valid.append(ok)
 keep=np.logical_and.reduce(valid);times=times[keep]
 pose,velocity,command,imu,applied=(v[keep] for v in aligned)
 base=np.c_[times-times[0],pose[:,1:4],velocity[:,1:4],command[:,1:4]]
 _,events=physical_inconsistency_mask(base,dt,pre_margin_s=1.2,post_margin_s=.5,
     moving_vx=.7,frozen_pose_speed=.12,distance_window_s=.5,min_odom_distance=.35,
     min_pose_odom_ratio=.65,impact_decel=-8.,max_pose_step=.30,max_yaw_step=.45)
 impacts=[event for event in events if 'impact_like_vx_drop' in event['causes']]
 cutoff_index=int(impacts[-1]['start_index']) if impacts else len(base)
 base,imu=base[:cutoff_index],imu[:cutoff_index]
 samples=np.c_[base,np.ones(len(base)),np.zeros(len(base)),imu[:,1:4]]
 columns=np.array(['t','x','y','yaw','vx','vy','omega','steer','accel','speed_cmd',
                   'split','bag_id','imu_wz','imu_ax','imu_ay'])
 cutoff_s=float(base[-1,0]+dt) if len(base) else 0.
 return samples,columns,dt,cutoff_s,events

def numbered_oversteer_episodes(over,segment,t,x,y,af,ar):
 episodes=[];start=None
 for i in range(len(over)):
  begins=over[i] and (i==0 or not over[i-1] or segment[i]!=segment[i-1])
  if begins:start=i
  ends=start is not None and (i==len(over)-1 or not over[i+1] or segment[i+1]!=segment[i])
  if ends:
   idx=np.arange(start,i+1);peak=idx[np.argmax(np.abs(ar[idx])-np.abs(af[idx]))]
   episodes.append({'number':len(episodes)+1,'start':start,'end':i,'peak':int(peak),
                    'duration_s':float(len(idx)*.02),'time_start_s':float(t[start]),
                    'time_end_s':float(t[i]),'x_m':float(x[peak]),'y_m':float(y[peak]),
                    'alpha_f_deg':float(af[peak]),'alpha_r_deg':float(ar[peak])})
   start=None
 return episodes

def main():
 s,columns,dt,collision_cutoff_s,filter_events=load_raw_continuous_before_collision();s=s.astype(float);names={str(v):i for i,v in enumerate(columns)};cfg=yaml.safe_load(PARAMS.read_text())['/**']['ros__parameters'];sign=np.ones(3,dtype=float);alpha=.25
 kfp=LateralVelocityKFParams(cornering_stiffness_front=float(cfg['kf_cornering_stiffness_front']),cornering_stiffness_rear=float(cfg['kf_cornering_stiffness_rear']),mass=float(cfg['mass']),yaw_inertia=float(cfg['I_z']),l_f=float(cfg['l_f']),l_r=float(cfg['l_r']),dt=dt,min_longitudinal_speed=float(cfg['kf_min_vx']),low_speed_threshold=float(cfg['kf_low_speed_threshold']),max_abs_vy=float(cfg['kf_max_abs_vy']),process_var_vy=float(cfg['kf_q_vy']),process_var_yaw_rate=float(cfg['kf_q_yaw_rate']),measurement_var_lateral_accel=float(cfg['kf_r_lateral_accel']),measurement_var_yaw_rate=float(cfg['kf_r_yaw_rate']),initial_var_vy=float(cfg['kf_initial_p_vy']),initial_var_yaw_rate=float(cfg['kf_initial_p_yaw_rate']),imu_lateral_accel_sign=float(cfg['imu_lateral_accel_sign']),nonlinear_dvy_threshold=float(cfg['kf_nonlinear_dvy_threshold']),nonlinear_dvy_width=float(cfg['kf_nonlinear_dvy_width']),nonlinear_inertial_blend=float(cfg['kf_nonlinear_inertial_blend']),nonlinear_process_noise_scale=float(cfg['kf_nonlinear_process_noise_scale']),nonlinear_ay_noise_scale=float(cfg['kf_nonlinear_ay_noise_scale']))
 vy,r=estimate_dataset(s,columns,dt,kfp,steer_scale=float(cfg['kf_steer_scale']),steer_bias=float(cfg['kf_steer_bias']),max_steer=float(cfg['kf_max_steer']),imu_ema_alpha=alpha,imu_wz_sign=float(sign[0]),imu_ay_sign=float(sign[2]))
 contract=Contract(dt=dt,steer_scale=float(cfg['kinematic_steer_scale']),steer_bias=float(cfg['kinematic_steer_bias']),steer_tau=float(cfg['steer_servo_time_constant']),max_steer_rate=float(cfg['actuator_max_steer_rate']))
 steer=s[:,names['steer']];speed_cmd=s[:,names['speed_cmd']];vx=s[:,names['vx']];segment=s[:,names['bag_id']].astype(int);applied=np.empty(len(s))
 for k in range(len(s)):
  target=np.clip(contract.steer_scale*steer[k]+contract.steer_bias,-.55,.55)
  if k==0 or segment[k]!=segment[k-1]:applied[k]=target
  else:applied[k],_=actuator_step(applied[k-1],steer[k],speed_cmd[k],vx[k],contract)
 safe=np.maximum(np.abs(vx),float(cfg['kf_min_vx']));lf,lr=float(cfg['l_f']),float(cfg['l_r'])
 alpha_f=applied-np.arctan2(vy+lf*r,safe);alpha_r=-np.arctan2(vy-lr*r,safe);af=np.rad2deg(alpha_f);ar=np.rad2deg(alpha_r);excess=np.abs(ar)-np.abs(af);over=(np.abs(vx)>=OVERSTEER_MIN_SPEED)&(np.abs(ar)>=OVERSTEER_REAR_MIN_DEG)&(excess>=OVERSTEER_EXCESS_DEG)
 t=stitched_time(s[:,names['t']],segment,dt);x=s[:,names['x']];y=s[:,names['y']];valid=np.isfinite(af)&np.isfinite(ar)&np.isfinite(x)&np.isfinite(y);moving=valid&(np.abs(vx)>=OVERSTEER_MIN_SPEED);ranking=np.argsort(np.where(moving,np.abs(ar),-1))[::-1];top=[]
 for i in ranking[:10]:top.append({'time_s':float(t[i]),'x_m':float(x[i]),'y_m':float(y[i]),'vx_mps':float(vx[i]),'vy_kf_mps':float(vy[i]),'yaw_rate_rps':float(r[i]),'applied_steer_deg':float(np.rad2deg(applied[i])),'alpha_f_deg':float(af[i]),'alpha_r_deg':float(ar[i]),'rear_minus_front_abs_deg':float(excess[i]),'oversteer':bool(over[i])})
 episodes=numbered_oversteer_episodes(over,segment,t,x,y,af,ar)
 report={'source':'raw causally aligned bag, continuous until final impact','bag':str(BAG_DB),'samples':len(s),'segments':1,'duration_s':float(t[-1]+dt),'collision_cutoff_s':collision_cutoff_s,'diagnostic_filter_events':filter_events,'vy_source':'causal adaptive 2-state KF','steering_source':'servo-lag estimate from ackermann_cmd','statistics_scope':f'|vx| >= {OVERSTEER_MIN_SPEED} m/s','front_slip':stats(af[moving]),'rear_slip':stats(ar[moving]),'rear_dominant_oversteer':{'criterion':f'vx>={OVERSTEER_MIN_SPEED}, |alpha_r|>={OVERSTEER_REAR_MIN_DEG} deg, |alpha_r|-|alpha_f|>={OVERSTEER_EXCESS_DEG} deg','samples':int(over.sum()),'fraction_of_moving_percent':float(100*over.sum()/max(1,moving.sum())),'duration_s':float(over.sum()*dt),'episodes':episodes},'top_rear_slip_samples':top}
 OUTPUT.mkdir(parents=True,exist_ok=True);(OUTPUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
 fig,axes=plt.subplots(3,2,figsize=(17,15),constrained_layout=True)
 axes[0,0].plot(t,af,label=r'$\alpha_f$ front');axes[0,0].plot(t,ar,label=r'$\alpha_r$ rear');axes[0,0].axhline(0,color='k',lw=.7);ylim=axes[0,0].get_ylim();axes[0,0].fill_between(t,ylim[0],ylim[1],where=over,color='red',alpha=.12,label='rear-dominant oversteer');axes[0,0].set_ylim(ylim);axes[0,0].set(title='Tire slip angles versus stitched segment time',xlabel='stitched time [s]',ylabel='slip angle [deg]');axes[0,0].legend()
 axes[0,1].plot(t,np.abs(ar),label=r'$|\alpha_r|$');axes[0,1].plot(t,np.abs(af),label=r'$|\alpha_f|$');axes[0,1].plot(t,excess,label=r'$|\alpha_r|-|\alpha_f|$');axes[0,1].axhline(OVERSTEER_EXCESS_DEG,color='r',ls='--');axes[0,1].set(title='Rear-slip dominance',xlabel='time [s]',ylabel='angle [deg]');axes[0,1].legend()
 sc=axes[1,0].scatter(x[valid],y[valid],c=np.abs(ar[valid]),s=14,cmap='magma',vmin=0,vmax=max(5,np.quantile(np.abs(ar[valid]),.99)));axes[1,0].scatter(x[over],y[over],facecolors='none',edgecolors='cyan',s=45,label='oversteer criterion');axes[1,0].set(title='Map position colored by rear slip',xlabel='map x [m]',ylabel='map y [m]');axes[1,0].axis('equal');axes[1,0].legend();fig.colorbar(sc,ax=axes[1,0],label=r'$|\alpha_r|$ [deg]')
 sc=axes[1,1].scatter(x[valid],y[valid],c=excess[valid],s=14,cmap='coolwarm',vmin=-max(5,np.quantile(np.abs(excess[valid]),.99)),vmax=max(5,np.quantile(np.abs(excess[valid]),.99)));axes[1,1].set(title='Map position: rear minus front slip magnitude',xlabel='map x [m]',ylabel='map y [m]');axes[1,1].axis('equal');fig.colorbar(sc,ax=axes[1,1],label=r'$|\alpha_r|-|\alpha_f|$ [deg]')
 axes[2,0].plot(t,vx,label='$v_x$');axes[2,0].plot(t,vy,label='adaptive KF $v_y$');axes[2,0].plot(t,r,label='yaw rate');axes[2,0].set(title='Estimated vehicle states',xlabel='time [s]');axes[2,0].legend()
 axes[2,1].plot(t,np.rad2deg(applied),label='estimated applied steer');axes[2,1].plot(t,np.rad2deg(steer),'--',label='ackermann steer command');axes[2,1].set(title='Steering used for slip calculation',xlabel='time [s]',ylabel='angle [deg]');axes[2,1].legend()
 for a in axes.ravel():a.grid(alpha=.25)
 fig.suptitle('0817 23:57:13 adaptive-KF front/rear tire slip analysis');fig.savefig(OUTPUT/'tire_slip_time_and_map.png',dpi=190);plt.close(fig);np.savez_compressed(OUTPUT/'slip_traces.npz',t=t,x=x,y=y,vx=vx,vy=vy,yaw_rate=r,applied_steer=applied,alpha_f=alpha_f,alpha_r=alpha_r,oversteer=over);print(json.dumps(report,indent=2))

 # Focused numbered view: one continuous raw trajectory before the final
 # impact. There is deliberately no gray auxiliary/background trajectory.
 fig,axes=plt.subplots(2,1,figsize=(15,15),constrained_layout=True)
 axes[0].plot(t,af,label=r'$\alpha_f$ front',lw=1.2);axes[0].plot(t,ar,label=r'$\alpha_r$ rear',lw=1.2)
 axes[0].axhline(0,color='k',lw=.7)
 ylo,yhi=axes[0].get_ylim()
 for episode in episodes:
  i0,i1,ip=episode['start'],episode['end'],episode['peak'];number=episode['number']
  axes[0].axvspan(t[i0],t[i1]+dt,color='red',alpha=.16)
  axes[0].text(t[ip],yhi-.04*(yhi-ylo),str(number),ha='center',va='top',fontsize=8,
               bbox=dict(boxstyle='circle,pad=.18',fc='white',ec='red',lw=.8))
 axes[0].set(title='Front/rear slip angles — numbered red regions are rear-dominant oversteer',xlabel='raw bag time before final impact [s]',ylabel='slip angle [deg]',ylim=(ylo,yhi));axes[0].legend();axes[0].grid(alpha=.25)
 sc=axes[1].scatter(x[moving],y[moving],c=np.abs(ar[moving]),s=17,cmap='magma',vmin=0,vmax=max(5,np.quantile(np.abs(ar[moving]),.99)),label='continuous raw-bag trajectory before final impact',zorder=2)
 for episode in episodes:
  ip=episode['peak'];number=episode['number']
  axes[1].scatter(x[ip],y[ip],s=95,facecolors='white',edgecolors='red',linewidths=1.4,zorder=3)
  axes[1].text(x[ip],y[ip],str(number),ha='center',va='center',fontsize=8,fontweight='bold',zorder=4)
 axes[1].set(title='Map1 continuous trajectory: same episode numbers at peak rear-slip locations',xlabel='map x [m]',ylabel='map y [m]');axes[1].axis('equal');axes[1].grid(alpha=.25);axes[1].legend(loc='best');fig.colorbar(sc,ax=axes[1],label=r'$|\alpha_r|$ [deg]')
 fig.suptitle('0817 23:57:13 continuous pre-collision slip and numbered oversteer regions');fig.savefig(OUTPUT/'tire_slip_numbered_closed_loop_map.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
