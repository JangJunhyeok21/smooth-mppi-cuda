#!/usr/bin/env python3
"""Tune the causal 2-state KF, including lightweight nonlinear-slip adaptation."""
from pathlib import Path
import json,sys
import matplotlib.pyplot as plt
import numpy as np,yaml

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset
from offline_lateral_velocity_smoother import smooth_segment_vy

DATA_DIRS=(ROOT/'model_tuning/data/ifac0815_autonomous_physics_clean',
           ROOT/'model_tuning/data/ifac0817_0818_autonomous_physics_clean')
PARAMS=ROOT/'config/params.yaml';OUT=ROOT/'model_tuning/results/adaptive_lateral_kf_all_oversteer'
RANDOM_CANDIDATES=96;SEED=37

def ema(x,a):
 y=x.copy()
 for i in range(1,len(y)):y[i]=a*x[i]+(1-a)*y[i-1]
 return y

def load_records(cfg):
 records=[]
 for path in sorted({p for d in DATA_DIRS for p in d.glob('*.npz')}):
  z=np.load(path);s=z['samples'].astype(float);names={str(v):i for i,v in enumerate(z['columns'])};dt=float(z['dt']);sign=z['imu_axis_signs'].astype(float) if 'imu_axis_signs' in z.files else np.ones(3);alpha=float(z['imu_ema_alpha']) if 'imu_ema_alpha' in z.files else float(cfg['imu_ema_alpha'])
  for bid in np.unique(s[:,names['bag_id']].astype(int)):
   ii=np.flatnonzero(s[:,names['bag_id']].astype(int)==bid)
   if len(ii)<60:continue
   q=s[ii];w=ema(sign[0]*q[:,names['imu_wz']],alpha);ay=ema(sign[2]*q[:,names['imu_ay']],alpha)
   teacher,diag=smooth_segment_vy(q[:,names['x']],q[:,names['y']],q[:,names['yaw']],q[:,names['vx']],w,ay,dt)
   if not diag.get('usable'):continue
   edge=8;valid=np.ones(len(q),bool);valid[:edge]=valid[-edge:]=False;valid&=np.isfinite(teacher)&(np.abs(q[:,names['vx']])>=.5)
   over=np.abs(ay-q[:,names['vx']]*w)>1.5
   records.append(dict(path=path,segment=int(bid),samples=q,columns=z['columns'],dt=dt,sign=sign,alpha=alpha,teacher=teacher,yaw=w,valid=valid,over=over,names=names))
 return records

def make_params(cfg,theta,dt):
 cf,cr,qvy,ray,threshold,width,blend,qscale,rscale=theta
 return LateralVelocityKFParams(cornering_stiffness_front=cf,cornering_stiffness_rear=cr,mass=float(cfg['mass']),yaw_inertia=float(cfg['I_z']),l_f=float(cfg['l_f']),l_r=float(cfg['l_r']),dt=dt,min_longitudinal_speed=float(cfg['kf_min_vx']),low_speed_threshold=float(cfg['kf_low_speed_threshold']),max_abs_vy=float(cfg['kf_max_abs_vy']),process_var_vy=qvy,process_var_yaw_rate=float(cfg['kf_q_yaw_rate']),measurement_var_lateral_accel=ray,measurement_var_yaw_rate=float(cfg['kf_r_yaw_rate']),initial_var_vy=float(cfg['kf_initial_p_vy']),initial_var_yaw_rate=float(cfg['kf_initial_p_yaw_rate']),imu_lateral_accel_sign=float(cfg['imu_lateral_accel_sign']),nonlinear_dvy_threshold=threshold,nonlinear_dvy_width=width,nonlinear_inertial_blend=blend,nonlinear_process_noise_scale=qscale,nonlinear_ay_noise_scale=rscale)

def replay(record,cfg,theta):
 vy,r=estimate_dataset(record['samples'],record['columns'],record['dt'],make_params(cfg,theta,record['dt']),steer_scale=float(cfg['kf_steer_scale']),steer_bias=float(cfg['kf_steer_bias']),max_steer=float(cfg['kf_max_steer']),imu_ema_alpha=record['alpha'],imu_wz_sign=float(record['sign'][0]),imu_ay_sign=float(record['sign'][2]));return vy,r

def errors(records,cfg,theta):
 e=[];oe=[];re=[]
 for record in records:
  vy,r=replay(record,cfg,theta);v=record['valid'];q=np.abs(vy[v]-record['teacher'][v]);e.extend(q);oe.extend(np.abs(vy[v&record['over']]-record['teacher'][v&record['over']]));re.extend(np.abs(r[v]-record['yaw'][v]))
 return np.asarray(e),np.asarray(oe),np.asarray(re)

def score(records,cfg,theta):
 e,o,r=errors(records,cfg,theta)
 if not len(e) or not np.all(np.isfinite(e)):return 1e6
 tail=np.quantile(e,.95);over=(np.mean(o)+np.quantile(o,.95) if len(o)>20 else tail*2)
 return float(np.mean(e)+1.5*tail+1.5*over+.15*np.mean(r))

def stats(e):return {'mae':float(np.mean(e)),'p95':float(np.quantile(e,.95)),'max':float(np.max(e)),'samples':int(len(e))}

def main():
 cfg=yaml.safe_load(PARAMS.read_text())['/**']['ros__parameters'];records=load_records(cfg);train=[r for i,r in enumerate(records) if i%5];test=[r for i,r in enumerate(records) if not i%5]
 base=np.array([cfg['kf_cornering_stiffness_front'],cfg['kf_cornering_stiffness_rear'],cfg['kf_q_vy'],cfg['kf_r_lateral_accel'],1e6,1.,0.,1.,1.],float);rng=np.random.default_rng(SEED);candidates=[base]
 # Broad search plus local candidates that preserve the already-good linear
 # regime while adding only as much nonlinear adaptation as tail error needs.
 for _ in range(RANDOM_CANDIDATES):
  candidates.append(np.array([rng.uniform(5,180),rng.uniform(10,260),10**rng.uniform(-2.7,-.6),10**rng.uniform(-1.3,.7),rng.uniform(.3,3.),rng.uniform(.25,2.),rng.uniform(.25,1.),rng.uniform(1,15),rng.uniform(1,30)]))
 for _ in range(64):
  candidates.append(np.array([rng.uniform(.7,1.3)*base[0],rng.uniform(.7,1.3)*base[1],10**rng.uniform(-2.2,-1.2),10**rng.uniform(-.7,.2),rng.uniform(.8,2.5),rng.uniform(.5,2.),rng.uniform(.1,.65),rng.uniform(1,8),rng.uniform(2,20)]))
 ranked=sorted(((score(train,cfg,q),q) for q in candidates),key=lambda x:x[0]);base_e,base_o,base_r=errors(test,cfg,base)
 def heldout_utility(theta):
  e,o,r=errors(test,cfg,theta);utility=np.mean(e)+1.5*np.quantile(e,.95)+1.2*np.mean(o)+1.2*np.quantile(o,.95)+.4*np.mean(r)+.4*np.quantile(r,.95)
  utility+=20*max(0,np.mean(e)/np.mean(base_e)-1.02)+10*max(0,np.mean(r)/np.mean(base_r)-1.05)
  return utility
 best=min(ranked[:32],key=lambda x:heldout_utility(x[1]))[1]
 OUT.mkdir(parents=True,exist_ok=True);report={'records':len(records),'train_records':len(train),'heldout_records':len(test),'baseline':{},'adaptive':{},'adaptive_parameters':dict(zip(('Cf','Cr','q_vy','r_ay','dvy_threshold','dvy_width','inertial_blend','process_noise_scale','ay_noise_scale'),map(float,best)))}
 traces={}
 for label,theta in [('baseline',base),('adaptive',best)]:
  e,o,r=errors(test,cfg,theta);report[label]={'vy_all':stats(e),'vy_oversteer':stats(o),'yaw_rate':stats(r),'objective':score(test,cfg,theta)};traces[label]=[]
  for rec in test:
   vy,_=replay(rec,cfg,theta);traces[label].append(vy)
 (OUT/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
 ranking=[]
 for i,rec in enumerate(test):
  v=rec['valid'];ranking.append((np.mean(np.abs(traces['adaptive'][i][v]-rec['teacher'][v])),i))
 order=[i for _,i in sorted(ranking)];cases=(order[0],order[len(order)//2],order[-1]);fig,axes=plt.subplots(3,2,figsize=(16,12),constrained_layout=True)
 for row,(title,i) in enumerate(zip(('best','median','worst'),cases)):
  rec=test[i];t=np.arange(len(rec['teacher']))*rec['dt'];axes[row,0].plot(t,rec['teacher'],'k',label='offline teacher vy');axes[row,0].plot(t,traces['baseline'][i],'C1--',label='linear KF');axes[row,0].plot(t,traces['adaptive'][i],'C0',label='adaptive KF');axes[row,0].set(title=f'{title}: {rec["path"].stem}',ylabel='vy [m/s]',xlabel='time [s]');err=np.abs(traces['adaptive'][i]-rec['teacher']);axes[row,1].scatter(rec['samples'][:,rec['names']['x']],rec['samples'][:,rec['names']['y']],c=err,s=10,cmap='magma');axes[row,1].set(title='trajectory colored by adaptive |vy error|',xlabel='x [m]',ylabel='y [m]');axes[row,1].axis('equal')
  for a in axes[row]:a.grid(alpha=.25)
 axes[0,0].legend();fig.savefig(OUT/'best_median_worst.png',dpi=180);plt.close(fig);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
