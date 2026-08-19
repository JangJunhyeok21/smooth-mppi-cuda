#!/usr/bin/env python3
"""Plot best/median/worst 1.2 s free rollouts of the selected model."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULT=ROOT/'model_tuning/results/dynamic_classic_residual_mlp_20d_stage2'
REPLAY=RESULT/'rollout_30step_metrics.npz'
DATA=ROOT/'model_tuning/data/dynamic_40ms_residual.npz'
MODEL_NAME='dynamic_classic_residual_mlp_20d'
OUTPUT=RESULT/f'{MODEL_NAME}_best_median_worst.png'
DT=.04

def wrapped(value):return (value+np.pi)%(2*np.pi)-np.pi

def main():
 replay=np.load(REPLAY);data=np.load(DATA);starts=replay['starts'];pred=replay['predicted'];gt=replay['ground_truth'];source=data['source_features'];observations=data['source_observations'];bags=data['source_bag_id']
 endpoint=np.linalg.norm(pred[:,-1,:2]-gt[:,-1,:2],axis=1);order=np.argsort(endpoint);selected=(order[0],order[len(order)//2],order[-1]);labels=('Best','Median','Worst');state_time=np.arange(pred.shape[1])*DT;accel_time=np.arange(pred.shape[1]-1)*DT
 fig,axes=plt.subplots(8,3,figsize=(17,27),squeeze=False)
 report=[]
 for column,(label,index) in enumerate(zip(labels,selected)):
  start=int(starts[index]);predicted=pred[index];truth=gt[index];rows=start+2*np.arange(pred.shape[1]-1);pred_ax=np.diff(predicted[:,3])/DT-predicted[:-1,4]*predicted[:-1,5];pred_ay=np.diff(predicted[:,4])/DT+predicted[:-1,3]*predicted[:-1,5];gt_ax=observations[rows,0];gt_ay=observations[rows,1]
  axis=axes[0,column];axis.plot(truth[:,0],truth[:,1],'k-',lw=2.2,label='GT');axis.plot(predicted[:,0],predicted[:,1],'--',color='tab:orange',lw=2,label='Predicted');axis.scatter(truth[0,0],truth[0,1],s=35,color='tab:green',zorder=4,label='Start');axis.set_aspect('equal',adjustable='datalim');axis.set_xlabel('relative x [m]');axis.set_ylabel('relative y [m]');axis.set_title(f'{label} · bag {int(bags[start])} · row {start}\nfinal trajectory error {endpoint[index]:.3f} m')
  for row,(signal,title,unit) in enumerate(((3,'Longitudinal velocity $v_x$','m/s'),(4,'Lateral velocity $v_y$','m/s'),(5,'Yaw rate $r$','rad/s')),1):
   axes[row,column].plot(state_time,truth[:,signal],'k-',lw=2,label='GT');axes[row,column].plot(state_time,predicted[:,signal],'--',color='tab:orange',lw=2,label='Predicted');axes[row,column].set_title(title);axes[row,column].set_ylabel(unit)
  gt_yaw=np.unwrap(truth[:,2]);pred_yaw=gt_yaw[0]+np.unwrap(wrapped(predicted[:,2]-predicted[0,2]));axes[4,column].plot(state_time,gt_yaw,'k-',lw=2,label='GT');axes[4,column].plot(state_time,pred_yaw,'--',color='tab:orange',lw=2,label='Predicted');axes[4,column].set_title('Accumulated yaw');axes[4,column].set_ylabel('rad')
  axes[5,column].plot(accel_time,gt_ax,'k-',lw=2,label='GT IMU $a_x$');axes[5,column].plot(accel_time,pred_ax,'--',color='tab:orange',lw=2,label='Predicted $a_x$');axes[5,column].set_title('Longitudinal acceleration');axes[5,column].set_ylabel('m/s²')
  axes[6,column].plot(accel_time,gt_ay,'k-',lw=2,label='GT IMU $a_y$');axes[6,column].plot(accel_time,pred_ay,'--',color='tab:orange',lw=2,label='Predicted $a_y$');axes[6,column].set_title('Lateral acceleration');axes[6,column].set_ylabel('m/s²')
  command_axis=axes[7,column];command_axis.plot(accel_time,source[rows,3],color='tab:blue',label='steer command');speed_axis=command_axis.twinx();speed_axis.plot(accel_time,source[rows,4],'--',color='tab:green',label='speed command');command_axis.set_title('Input commands');command_axis.set_ylabel('steer [rad]',color='tab:blue');speed_axis.set_ylabel('speed [m/s]',color='tab:green');lines,names=command_axis.get_legend_handles_labels();other,other_names=speed_axis.get_legend_handles_labels();command_axis.legend(lines+other,names+other_names,fontsize=8)
  for row,axis in enumerate(axes[:,column]):
   axis.grid(alpha=.28)
   if row not in (0,7):axis.legend(fontsize=8)
   if row>0:axis.set_xlabel('time [s]')
  report.append({'case':label.lower(),'index':int(index),'bag_id':int(bags[start]),'source_row':start,'trajectory_final_m':float(endpoint[index]),'vx_mae_mps':float(np.mean(abs(predicted[:,3]-truth[:,3]))),'vy_mae_mps':float(np.mean(abs(predicted[:,4]-truth[:,4]))),'yaw_rate_mae_radps':float(np.mean(abs(predicted[:,5]-truth[:,5]))),'yaw_mae_rad':float(np.mean(abs(wrapped(predicted[:,2]-truth[:,2])))),'ax_mae_mps2':float(np.mean(abs(pred_ax-gt_ax))),'ay_mae_mps2':float(np.mean(abs(pred_ay-gt_ay)))})
 fig.suptitle(f'{MODEL_NAME} · held-out aggressive test',y=.998,fontsize=16);fig.subplots_adjust(hspace=.42,wspace=.28,top=.97);fig.savefig(OUTPUT,dpi=180,bbox_inches='tight');plt.close(fig);OUTPUT.with_suffix('.json').write_text(json.dumps({'model_name':MODEL_NAME,'cases':report},indent=2)+'\n');print(json.dumps({'model_name':MODEL_NAME,'plot':str(OUTPUT),'cases':report},indent=2))
if __name__=='__main__':main()
