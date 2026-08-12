#!/usr/bin/env python3
"""Compare /mppi_optimal_trajectory against future pose/odom measurements."""
import argparse,json
from pathlib import Path
import numpy as np

def stamp(msg,record_ns):
    s=msg.header.stamp
    return s.sec+s.nanosec*1e-9 if s.sec or s.nanosec else record_ns*1e-9
def yaw(q): return np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi
def main():
    p=argparse.ArgumentParser();p.add_argument('bag');p.add_argument('--centerline',required=True);p.add_argument('--out',default='/tmp/mppi_run_analysis.json');a=p.parse_args()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    bag=Path(a.bag); files=sorted([*bag.glob('*.db3'),*bag.glob('*.mcap')]);uri=str(files[0] if files else bag);sid='mcap' if uri.endswith('.mcap') else 'sqlite3'
    r=rosbag2_py.SequentialReader();r.open(rosbag2_py.StorageOptions(uri=uri,storage_id=sid),rosbag2_py.ConverterOptions('cdr','cdr'));types={x.name:x.type for x in r.get_all_topics_and_types()};topics=('/newmcl_pose','/odom','/drive','/mppi_optimal_trajectory');mt={x:get_message(types[x]) for x in topics};pose=[];odom=[];drive=[];traj=[]
    while r.has_next():
        topic,raw,ns=r.read_next()
        if topic not in mt:continue
        m=deserialize_message(raw,mt[topic]);t=stamp(m,ns)
        if topic=='/newmcl_pose':pose.append((t,m.pose.position.x,m.pose.position.y,yaw(m.pose.orientation)))
        elif topic=='/odom':odom.append((t,m.twist.twist.linear.x,m.twist.twist.linear.y,m.twist.twist.angular.z))
        elif topic=='/drive':drive.append((t,m.drive.steering_angle,m.drive.speed))
        else:traj.append((t,np.asarray(m.predicted_x),np.asarray(m.predicted_y),np.asarray(m.predicted_yaw),np.asarray(m.predicted_v),np.asarray(m.predicted_vy),np.asarray(m.predicted_yaw_rate)))
    pose=np.asarray(pose);odom=np.asarray(odom);drive=np.asarray(drive);ref=np.loadtxt(a.centerline,delimiter=',',skiprows=1);d2=(pose[:,None,1]-ref[None,:,0])**2+(pose[:,None,2]-ref[None,:,1])**2;idx=d2.argmin(1);un=idx.astype(float)
    for i in range(1,len(un)):
        d=un[i]-un[i-1]
        if d < -len(ref)/2:un[i:]+=len(ref)
        elif d > len(ref)/2:un[i:]-=len(ref)
    report={'duration_s':float(pose[-1,0]-pose[0,0]),'distance_m':float(np.hypot(np.diff(pose[:,1]),np.diff(pose[:,2])).sum()),'laps':float((un[-1]-un[0])/len(ref)),'centerline_error_mean_p95_max_m':[float(x) for x in (np.sqrt(d2.min(1)).mean(),np.quantile(np.sqrt(d2.min(1)),.95),np.sqrt(d2.min(1)).max())],'speed_min_mean_max_mps':[float(odom[:,1].min()),float(odom[:,1].mean()),float(odom[:,1].max())],'command_speed_mean_max_mps':[float(drive[:,2].mean()),float(drive[:,2].max())],'horizons':{}}
    for sec in (.2,.5,1.,1.2):
        k=round(sec/.02)-1;es=[]
        for t,px,py,pyaw,pv,pvy,pr in traj:
            if len(px)<=k:continue
            j=np.searchsorted(pose[:,0],t+sec);o=np.searchsorted(odom[:,0],t+sec)
            if j>=len(pose) or o>=len(odom):continue
            es.append((np.hypot(px[k]-pose[j,1],py[k]-pose[j,2]),abs(wrap(pyaw[k]-pose[j,3])),pv[k]-odom[o,1],pvy[k]-odom[o,2],pr[k]-odom[o,3],drive[min(np.searchsorted(drive[:,0],t),len(drive)-1),1]))
        e=np.asarray(es);report['horizons'][str(sec)]={'samples':len(e),'position_mean_p95_max_m':[float(e[:,0].mean()),float(np.quantile(e[:,0],.95)),float(e[:,0].max())],'yaw_mean_p95_max_rad':[float(e[:,1].mean()),float(np.quantile(e[:,1],.95)),float(e[:,1].max())],'vx_bias_mae_p95_mps':[float(e[:,2].mean()),float(np.abs(e[:,2]).mean()),float(np.quantile(np.abs(e[:,2]),.95))],'vy_mae_p95_mps':[float(np.abs(e[:,3]).mean()),float(np.quantile(np.abs(e[:,3]),.95))],'yaw_rate_mae_p95_rps':[float(np.abs(e[:,4]).mean()),float(np.quantile(np.abs(e[:,4]),.95))]}
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
