#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('result');p.add_argument('-o','--output',required=True);p.add_argument('--compare',help='optional second result directory');a=p.parse_args();src=Path(a.result);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 z=np.load(src/'test_predictions.npz');e=z['position_error'];pred=z['prediction'];gt=z['gt_pose'];state=z['gt_state'];f=e[:,-1];ids=[int(np.argmin(f)),int(np.argmin(abs(f-np.median(f)))),int(np.argmax(f))]
 fig,ax=plt.subplots(2,3,figsize=(14,8))
 for j,(name,i) in enumerate(zip(('best','median','worst'),ids)):
  ax[0,j].plot(gt[i,:,0],gt[i,:,1],'k-',lw=2,label='GT');ax[0,j].plot(pred[i,:,0],pred[i,:,1],'r--',lw=2,label='prediction');ax[0,j].axis('equal');ax[0,j].grid(alpha=.3);ax[0,j].set_title(f'{name}: {f[i]:.3f} m');ax[0,j].legend()
  t=np.arange(pred.shape[1])*.02;l1=ax[1,j].plot(t,state[i,:,0],color='tab:blue',linestyle='-',label='GT vx');l2=ax[1,j].plot(t,pred[i,:,3],color='tab:blue',linestyle='--',label='Predicted vx');ax2=ax[1,j].twinx();l3=ax2.plot(t,state[i,:,2],color='tab:green',linestyle='-',label='GT yaw rate');l4=ax2.plot(t,pred[i,:,5],color='tab:green',linestyle='--',label='Predicted yaw rate');ax[1,j].grid(alpha=.3);ax[1,j].set_xlabel('time [s]');ax[1,j].set_ylabel('vx [m/s]',color='tab:blue');ax2.set_ylabel('yaw rate [rad/s]',color='tab:green');lines=l1+l2+l3+l4;ax[1,j].legend(lines,[line.get_label() for line in lines],loc='best',fontsize=8)
 fig.tight_layout();fig.savefig(out/'best_median_worst_state.png',dpi=180);plt.close(fig)
 speed_error=np.abs(pred[:,:,3]-state[:,:,0]);yaw_error=np.abs(pred[:,:,5]-state[:,:,2]);t=np.arange(pred.shape[1])*.02
 fig,ax=plt.subplots(1,3,figsize=(15,4.2))
 for axis,values,title,unit,color in ((ax[0],e,'Position error','m','tab:red'),(ax[1],speed_error,'Velocity error','m/s','tab:blue'),(ax[2],yaw_error,'Yaw-rate error','rad/s','tab:green')):
  axis.plot(t,np.mean(values,axis=0),color=color,lw=2,label='mean');axis.plot(t,np.median(values,axis=0),color=color,ls='--',label='median');axis.plot(t,np.percentile(values,95,axis=0),color=color,ls=':',label='95th percentile');axis.set(title=title,xlabel='open-loop time [s]',ylabel=f'absolute error [{unit}]');axis.grid(alpha=.3);axis.legend(fontsize=8)
 fig.tight_layout();fig.savefig(out/'error_over_horizon.png',dpi=180);plt.close(fig)
 m=json.loads((src/'metrics.json').read_text());(out/'metrics.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
 if a.compare:
  sources=[src,Path(a.compare)];fig,axes=plt.subplots(1,3,figsize=(15,4.2));summary={}
  for source in sources:
   zz=np.load(source/'test_predictions.npz');mm=json.loads((source/'metrics.json').read_text());label=mm.get('input_normalization','zscore');pp=zz['prediction'];ss=zz['gt_state'];values=(zz['position_error'],np.abs(pp[:,:,3]-ss[:,:,0]),np.abs(pp[:,:,5]-ss[:,:,2]));tt=np.arange(pp.shape[1])*float(mm.get('dt',.02))
   summary[label]=mm
   for axis,value in zip(axes,values):axis.plot(tt,np.mean(value,axis=0),lw=2,label=f'{label} mean');axis.plot(tt,np.percentile(value,95,axis=0),ls=':',label=f'{label} P95')
  for axis,title,unit in zip(axes,('Trajectory','Velocity','Yaw rate'),('m','m/s','rad/s')):axis.set(title=f'{title} error',xlabel='open-loop time [s]',ylabel=f'absolute error [{unit}]');axis.grid(alpha=.3);axis.legend(fontsize=8)
  fig.tight_layout();fig.savefig(out/'normalization_comparison.png',dpi=180);plt.close(fig);(out/'normalization_comparison.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
