#!/usr/bin/env python3
"""Resample an aligned training NPZ without crossing bag/segment boundaries."""
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATASET_PATH=ROOT/'model_tuning/data/ifac0807_0808_hardcase_train_test.npz'
OUTPUT_PATH=ROOT/'model_tuning/data/default_resampled_training_data.npz'
TARGET_DT=.035

def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',nargs='?',default=str(DATASET_PATH));p.add_argument('-o','--output',default=str(OUTPUT_PATH));p.add_argument('--dt',type=float,default=TARGET_DT);a=p.parse_args()
    z=np.load(a.dataset);src=z['samples'].astype(np.float64);names=[str(x) for x in z['columns']];col={n:i for i,n in enumerate(names)}
    bag=src[:,col['bag_id']].astype(int);parts=[]
    continuous=('x','y','yaw','vx','vy','omega')
    held=('steer','accel','speed_cmd','imu_wz','imu_ax','imu_ay')
    for bid in np.unique(bag):
        rows=src[bag==bid];t=rows[:,col['t']];new_t=np.arange(t[0],t[-1]+1e-9,a.dt)
        if len(new_t)<2:continue
        out=np.empty((len(new_t),src.shape[1]),np.float64);out[:]=np.nan;out[:,col['t']]=new_t
        for name in continuous:
            if name not in col:continue
            values=rows[:,col[name]]
            if name=='yaw':values=np.unwrap(values)
            out[:,col[name]]=np.interp(new_t,t,values)
        hold=np.searchsorted(t,new_t,side='right')-1;hold=np.clip(hold,0,len(t)-1)
        for name in held:
            if name in col:out[:,col[name]]=rows[hold,col[name]]
        if 'split' in col:out[:,col['split']]=rows[0,col['split']]
        out[:,col['bag_id']]=bid;parts.append(out)
    dst=np.concatenate(parts);path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,samples=dst,dt=np.array(a.dt),columns=z['columns'],
                        source_dataset=np.array(str(Path(a.dataset).resolve())),
                        resampling=np.array('continuous linear interpolation; command/IMU causal zero-order hold'))
    print(f'saved {len(dst)} samples, {len(parts)} segments, dt={a.dt:.6f} s to {path}')
if __name__=='__main__':main()
