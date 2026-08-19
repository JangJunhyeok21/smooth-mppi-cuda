#!/usr/bin/env python3
"""Replay active MPPI dynamic+residual MLP over every extracted bag segment."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams, estimate_dataset
DATA_DIR=ROOT/"model_tuning/data/extracted_bags"
OUTPUT_DIR=ROOT/"model_tuning/results/all_bags_dynamic_residual_full_open_loop"
PARAMS_PATH=ROOT/"config/params.yaml"
USE_PLOT=False
IMU_WZ_SIGN=-1.;IMU_AX_SIGN=1.;IMU_AY_SIGN=-1.;IMU_EMA_ALPHA=.25
KF_CF=12.7222491;KF_CR=75.0944752


class MLP(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(20,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,3))
    def forward(self,x):return self.net(x)


def load_bin(path):
    data=np.fromfile(path,np.float32)
    if len(data)!=3563:raise ValueError(f"{path}: expected 3563 float32, got {len(data)}")
    net=MLP();state={};offset=0
    for key,shape in (("net.0.weight",(64,20)),("net.0.bias",(64,)),("net.2.weight",(32,64)),
                      ("net.2.bias",(32,)),("net.4.weight",(3,32)),("net.4.bias",(3,))):
        n=int(np.prod(shape));state[key]=torch.from_numpy(data[offset:offset+n].copy().reshape(shape));offset+=n
    net.load_state_dict(state);net.eval();return net,data[offset:offset+20],data[offset+20:offset+40]


def ema_segments(values,segments,alpha=.25):
    out=values.astype(np.float32).copy()
    for bid in np.unique(segments):
        ii=np.flatnonzero(segments==bid)
        for k in range(1,len(ii)):out[ii[k]]=alpha*out[ii[k]]+(1-alpha)*out[ii[k-1]]
    return out


def main():
    import matplotlib.pyplot as plt
    cfg=yaml.safe_load(PARAMS_PATH.read_text())['/**']['ros__parameters']
    if cfg['dynamics_model']!='dynamic_mlp_residual':raise SystemExit('params.yaml is not dynamic_mlp_residual')
    net,mean,std=load_bin(Path(cfg['dynamic_mlp_weights_path']));mean=torch.tensor(mean);std=torch.tensor(std)
    scale=torch.tensor([8.,8.,30.]);dt_default=.02;lf=float(cfg['l_f']);lr=float(cfg['l_r']);wb=lf+lr
    mass=float(cfg['mass']);g=9.81;fzf=mass*g*lr/wb;fzr=mass*g*lf/wb
    bf,cf,df,ef=[float(cfg[f'dynamic_mlp_{x}_f']) for x in ('B','C','D','E')]
    br,cr,dr,er=[float(cfg[f'dynamic_mlp_{x}_r']) for x in ('B','C','D','E')];iz=float(cfg['dynamic_mlp_I_z'])
    kp=float(cfg['speed_servo_kp']);mina=float(cfg['min_accel']);maxa=float(cfg['max_accel'])
    minv=float(cfg['min_speed']);maxv=float(cfg['max_speed']);maxsteer=float(cfg['max_steer']);pscale=float(cfg['kinematic_position_speed_scale'])
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True);report={};all_segment_metrics=[]
    for path in sorted(DATA_DIR.glob('*.npz')):
        z=np.load(path);raw=z['samples'];columns=z['columns'];dt=float(z['dt']);seg=raw[:,11].astype(int)
        imu=ema_segments(raw[:,12:15]*np.array([IMU_WZ_SIGN,IMU_AX_SIGN,IMU_AY_SIGN]),seg,IMU_EMA_ALPHA)
        kfp=LateralVelocityKFParams(cornering_stiffness_front=KF_CF,cornering_stiffness_rear=KF_CR,dt=dt)
        gtvy,_=estimate_dataset(raw,columns,dt,kfp,steer_scale=float(cfg['kf_steer_scale']),steer_bias=float(cfg['kf_steer_bias']),
            max_steer=float(cfg['kf_max_steer']),imu_ema_alpha=IMU_EMA_ALPHA,imu_wz_sign=IMU_WZ_SIGN,imu_ay_sign=IMU_AY_SIGN)
        predictions=[];targets=[];times=[];controls=[];offset=0.;segment_metrics=[]
        for bid in np.unique(seg):
            ii=np.flatnonzero(seg==bid)
            if len(ii)<7:continue
            u=raw[np.ix_(ii,[7,9])].astype(np.float32);u[:,0]=np.clip(u[:,0],-maxsteer,maxsteer);u[:,1]=np.clip(u[:,1],minv,maxv)
            k0=5;state=np.array([raw[ii[k0],4],gtvy[ii[k0]],imu[ii[k0],0]],np.float32)
            pose=np.array(raw[ii[k0],1:4],np.float64);history=u[:k0].copy();pred=[[*pose,*state,imu[ii[k0],1],imu[ii[k0],2]]]
            with torch.no_grad():
                for k in range(k0,len(ii)-1):
                    # CUDA update_dynamic_mlp_residual(): history[8] is the
                    # previous Ackermann command, so delta[t] = steer_cmd[t-1].
                    steer=float(np.clip(history[-1,0],-.55,.55));speed=float(np.hypot(state[0],state[1]));ax=float(np.clip(kp*(u[k,1]-speed),mina,maxa));safe=max(abs(float(state[0])),.5)
                    af=steer-np.arctan2(state[1]+lf*state[2],safe);ar=-np.arctan2(state[1]-lr*state[2],safe)
                    bfa=bf*af;bra=br*ar;fyf=fzf*df*np.sin(cf*np.arctan(bfa-ef*(bfa-np.arctan(bfa))));fyr=fzr*dr*np.sin(cr*np.arctan(bra-er*(bra-np.arctan(bra))))
                    cay=(fyf*np.cos(steer)+fyr)/mass;bvx=state[0]+(ax+state[1]*state[2])*dt;bvy=state[1]+(cay-state[0]*state[2])*dt;bw=state[2]+(lf*fyf*np.cos(steer)-lr*fyr)/iz*dt
                    base=np.array([*state,*u[k],steer,u[k,0]-history[-1,0],bvx,bvy,bw],np.float32);feature=np.r_[base,history.reshape(-1)]
                    corr=(torch.tanh(net((torch.tensor(feature)-mean)/std))*scale).numpy();state=np.array([bvx+corr[0]*dt,bvy+corr[1]*dt,bw+corr[2]*dt])
                    pay=ax+corr[0];pcy=cay+corr[1];ns=np.hypot(state[0],state[1]);beta=np.arctan2(state[1],state[0]);pose[0]+=pscale*ns*np.cos(pose[2]+beta)*dt;pose[1]+=pscale*ns*np.sin(pose[2]+beta)*dt;pose[2]=np.arctan2(np.sin(pose[2]+state[2]*dt),np.cos(pose[2]+state[2]*dt))
                    pred.append([*pose,*state,pay,pcy]);history=np.r_[history[1:],u[k:k+1]].reshape(5,2)
            pred=np.asarray(pred);rows=ii[k0:k0+len(pred)];target=np.c_[raw[rows,1:4],raw[rows,4],gtvy[rows],imu[rows,0:3]]
            local=np.arange(len(pred))*dt+offset;offset=local[-1]+5*dt;predictions.append(pred);targets.append(target);times.append(local);controls.append(u[k0:k0+len(pred)])
            pe=np.linalg.norm(pred[:,:2]-target[:,:2],axis=1);yawerr=np.abs(np.arctan2(np.sin(pred[:,2]-target[:,2]),np.cos(pred[:,2]-target[:,2])))
            segment_metrics.append({'segment':int(bid),'duration_s':float(local[-1]-local[0]),'trajectory_final_m':float(pe[-1]),'trajectory_mean_m':float(pe.mean()),'vx_mae_mps':float(np.mean(abs(pred[:,3]-target[:,3]))),'vy_mae_mps':float(np.mean(abs(pred[:,4]-target[:,4]))),'ax_mae_mps2':float(np.mean(abs(pred[:,6]-target[:,6]))),'ay_mae_mps2':float(np.mean(abs(pred[:,7]-target[:,7]))),'yaw_rate_mae_radps':float(np.mean(abs(pred[:,5]-target[:,5]))),'yaw_mae_deg':float(np.degrees(yawerr.mean()))})
        if not predictions:continue
        fig,axes=plt.subplots(4,2,figsize=(15,17));labels=((3,'vx [m/s]'),(4,'vy [m/s]'),(6,'ax [m/s²]'),(7,'ay [m/s²]'),(5,'yaw-rate [rad/s]'),(2,'yaw [rad]'))
        for n,(p,t) in enumerate(zip(predictions,targets)):
            axes[0,0].plot(t[:,0],t[:,1],'k-',label='GT' if n==0 else None);axes[0,0].plot(p[:,0],p[:,1],'--',label='Open-loop prediction' if n==0 else None)
        axes[0,0].axis('equal');axes[0,0].set_title('Full segment trajectories');axes[0,0].legend()
        axes[0,1].axis('off');axes[0,1].text(.02,.98,'\n'.join(f"segment {m['segment']}: {m['duration_s']:.1f}s, final={m['trajectory_final_m']:.2f}m" for m in segment_metrics),va='top',family='monospace')
        for axis,(col,label) in zip(axes.flat[2:],labels):
            for n,(p,t,tm) in enumerate(zip(predictions,targets,times)):axis.plot(tm,t[:,col],'k-',label='GT' if n==0 else None);axis.plot(tm,p[:,col],'--',label='Prediction' if n==0 else None)
            axis.set_title(label);axis.set_xlabel('concatenated segment time [s]');axis.legend(fontsize=8)
        for axis in axes.flat:axis.grid(alpha=.25)
        fig.suptitle(path.stem+' — active MPPI dynamic + residual full open-loop',y=.995);fig.subplots_adjust(hspace=.38,wspace=.25,top=.95,bottom=.06)
        out=OUTPUT_DIR/f'{path.stem}_full_open_loop.png';fig.savefig(out,dpi=180);plt.close(fig)
        replay_arrays={'dt':np.asarray(dt,dtype=np.float64),
                       'segment_count':np.asarray(len(predictions),dtype=np.int32)}
        for index,(prediction,target,control) in enumerate(zip(predictions,targets,controls)):
            if not np.allclose(prediction[0,:3],target[0,:3],rtol=0.,atol=1e-12):
                raise RuntimeError(f'{path.stem} segment {index}: prediction and GT initial pose differ')
            replay_arrays[f'prediction_{index}']=prediction
            replay_arrays[f'target_{index}']=target
            replay_arrays[f'control_{index}']=control
        replay_path=OUTPUT_DIR/f'{path.stem}_interactive_replay.npz'
        np.savez_compressed(replay_path,**replay_arrays)
        keys=('trajectory_final_m','trajectory_mean_m','vx_mae_mps','vy_mae_mps',
              'ax_mae_mps2','ay_mae_mps2','yaw_rate_mae_radps','yaw_mae_deg')
        bag_summary={key:float(np.mean([m[key] for m in segment_metrics])) for key in keys}
        report[path.stem]={'summary':bag_summary,'segments':segment_metrics,
                           'plot':str(out),'interactive_replay':str(replay_path)}
        all_segment_metrics.extend(segment_metrics)
    keys=('trajectory_final_m','trajectory_mean_m','vx_mae_mps','vy_mae_mps',
          'ax_mae_mps2','ay_mae_mps2','yaw_rate_mae_radps','yaw_mae_deg')
    report['_overall']={key:float(np.mean([m[key] for m in all_segment_metrics])) for key in keys}
    report['_model']={'dynamics_model':'dynamic_mlp_residual',
                      'steering_policy':'steer[t] = /ackermann_cmd.steering_angle[t-1]',
                      'weights':str(Path(cfg['dynamic_mlp_weights_path']))}
    bag_names=[name for name in report if not name.startswith('_')]
    short_names=[name.replace('rosbag2_2026_','') for name in bag_names]
    titles={'trajectory_final_m':'Final trajectory error [m]',
            'trajectory_mean_m':'Mean trajectory error [m]',
            'vx_mae_mps':r'$v_x$ MAE [m/s]','vy_mae_mps':r'$v_y$ MAE [m/s]',
            'ax_mae_mps2':r'$a_x$ MAE [m/s$^2$]','ay_mae_mps2':r'$a_y$ MAE [m/s$^2$]',
            'yaw_rate_mae_radps':'Yaw-rate MAE [rad/s]',
            'yaw_mae_deg':'Yaw MAE [deg]'}
    fig,axes=plt.subplots(4,2,figsize=(16,15))
    for axis,key in zip(axes.flat,keys):
        values=[report[name]['summary'][key] for name in bag_names]
        axis.bar(np.arange(len(values)),values,color='tab:blue')
        axis.set_xticks(np.arange(len(values)),short_names,rotation=25,ha='right')
        axis.set_title(titles[key]);axis.grid(axis='y',alpha=.25)
    fig.suptitle('dynamic_mlp_residual: previous /ackermann_cmd steer — all-bag open-loop',y=.995)
    fig.tight_layout();fig.savefig(OUTPUT_DIR/'all_bags_metric_comparison.png',dpi=180);plt.close(fig)
    (OUTPUT_DIR/'metrics.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    if USE_PLOT:plt.show()


if __name__=='__main__':main()
