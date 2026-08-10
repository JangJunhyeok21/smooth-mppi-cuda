#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Direct-run settings. Edit these values and run
# `python3 model_tuning/evaluate_model.py`; CLI arguments are optional.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / 'model_tuning/results/ifac0807_0808_actuator_regressed_yaw_curriculum'
OUTPUT_PATH = RESULT_PATH / 'visualization'
COMPARE_RESULT_PATH = None
COMPARE_LABELS = None
DATASET_PATH = PROJECT_ROOT / 'model_tuning/data/ifac0807_0808_hardcase_train_test.npz'
HISTORY_OFFSET = 5

def main():
 p=argparse.ArgumentParser();p.add_argument('result',nargs='?',default=str(RESULT_PATH));p.add_argument('-o','--output',default=str(OUTPUT_PATH));p.add_argument('--compare',default=None if COMPARE_RESULT_PATH is None else str(COMPARE_RESULT_PATH),help='optional second result directory');p.add_argument('--labels',nargs=2,default=COMPARE_LABELS,metavar=('RESULT','COMPARE'),help='labels for a two-result comparison');p.add_argument('--dataset',default=str(DATASET_PATH),help='extracted NPZ used to overlay /drive.speed');p.add_argument('--history-offset',type=int,default=HISTORY_OFFSET);a=p.parse_args();src=Path(a.result);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 z=np.load(src/'test_predictions.npz');e=z['position_error'];pred=z['prediction'];gt=z['gt_pose'];state=z['gt_state'];f=e[:,-1];ids=[int(np.argmin(f)),int(np.argmin(abs(f-np.median(f)))),int(np.argmax(f))];raw=(np.load(a.dataset)['samples'] if a.dataset else None)
 fig,ax=plt.subplots(2,3,figsize=(14,8))
 for j,(name,i) in enumerate(zip(('best','median','worst'),ids)):
  ax[0,j].plot(gt[i,:,0],gt[i,:,1],'k-',lw=2,label='GT');ax[0,j].plot(pred[i,:,0],pred[i,:,1],'r--',lw=2,label='prediction');ax[0,j].axis('equal');ax[0,j].grid(alpha=.3);ax[0,j].set_title(f'{name}: {f[i]:.3f} m');ax[0,j].legend()
  t=np.arange(pred.shape[1])*.02;l1=ax[1,j].plot(t,state[i,:,0],color='tab:blue',linestyle='-',label='GT vx');l2=ax[1,j].plot(t,pred[i,:,3],color='tab:blue',linestyle='--',label='Predicted vx');lc=[]
  if raw is not None:
   start=int(z['starts'][i])+a.history_offset;cmd=raw[start:start+len(t),9];lc=ax[1,j].plot(t,cmd,color='tab:orange',linestyle='-.',label='/drive.speed command')
  ax2=ax[1,j].twinx();l3=ax2.plot(t,gt[i,:,2],color='tab:green',linestyle='-',label='GT yaw');l4=ax2.plot(t,pred[i,:,2],color='tab:green',linestyle='--',label='Predicted yaw');ax[1,j].grid(alpha=.3);ax[1,j].set_xlabel('time [s]');ax[1,j].set_ylabel('velocity [m/s]',color='tab:blue');ax2.set_ylabel('yaw [rad]',color='tab:green');lines=l1+l2+lc+l3+l4;ax[1,j].legend(lines,[line.get_label() for line in lines],loc='best',fontsize=8)
 fig.tight_layout();fig.savefig(out/'best_median_worst_state.png',dpi=180);plt.close(fig)
 speed_error=np.abs(pred[:,:,3]-state[:,:,0]);yaw_error=np.abs(np.arctan2(np.sin(pred[:,:,2]-gt[:,:,2]),np.cos(pred[:,:,2]-gt[:,:,2])));t=np.arange(pred.shape[1])*.02
 fig,ax=plt.subplots(1,3,figsize=(15,4.2))
 for axis,values,title,unit,color in ((ax[0],e,'Position error','m','tab:red'),(ax[1],speed_error,'Velocity error','m/s','tab:blue'),(ax[2],yaw_error,'Yaw angle error','rad','tab:green')):
  axis.plot(t,np.mean(values,axis=0),color=color,lw=2,label='mean');axis.plot(t,np.median(values,axis=0),color=color,ls='--',label='median');axis.plot(t,np.percentile(values,95,axis=0),color=color,ls=':',label='95th percentile');axis.set(title=title,xlabel='open-loop time [s]',ylabel=f'absolute error [{unit}]');axis.grid(alpha=.3);axis.legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/'error_over_horizon.png',dpi=180);plt.close(fig)
 m=json.loads((src/'metrics.json').read_text());yf=yaw_error[:,-1];m.update({'final_yaw_mae_rad':float(yf.mean()),'final_yaw_median_rad':float(np.median(yf)),'final_yaw_p95_rad':float(np.percentile(yf,95)),'final_yaw_worst_rad':float(yf.max())});(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
 if a.compare:
  sources=[src,Path(a.compare)];fig,axes=plt.subplots(1,3,figsize=(15,4.2));summary={}
  for source_index,source in enumerate(sources):
   zz=np.load(source/'test_predictions.npz');mm=json.loads((source/'metrics.json').read_text());label=(a.labels[source_index] if a.labels else mm.get('input_normalization',source.name));pp=zz['prediction'];ss=zz['gt_state'];gg=zz['gt_pose'];ye=np.abs(np.arctan2(np.sin(pp[:,:,2]-gg[:,:,2]),np.cos(pp[:,:,2]-gg[:,:,2])));values=(zz['position_error'],np.abs(pp[:,:,3]-ss[:,:,0]),ye);tt=np.arange(pp.shape[1])*float(mm.get('dt',.02));yy=ye[:,-1];mm.update({'final_yaw_mae_rad':float(yy.mean()),'final_yaw_median_rad':float(np.median(yy)),'final_yaw_p95_rad':float(np.percentile(yy,95)),'final_yaw_worst_rad':float(yy.max())})
   summary[label]=mm
   for axis,value in zip(axes,values):axis.plot(tt,np.mean(value,axis=0),lw=2,label=f'{label} mean');axis.plot(tt,np.percentile(value,95,axis=0),ls=':',label=f'{label} P95')
  for axis,title,unit in zip(axes,('Trajectory','Velocity','Yaw angle'),('m','m/s','rad')):axis.set(title=f'{title} error',xlabel='open-loop time [s]',ylabel=f'absolute error [{unit}]');axis.grid(alpha=.3);axis.legend(fontsize=8)
  fig.tight_layout();fig.savefig(out/'normalization_comparison.png',dpi=180);plt.close(fig);(out/'normalization_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
