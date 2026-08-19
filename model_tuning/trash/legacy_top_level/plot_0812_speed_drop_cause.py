#!/usr/bin/env python3
"""Visualize why MPPI reduced speed before the 0812 collision."""
import os
import sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BAG=Path('/mnt/nas_custom/F1tenth/2026 IFAC/0812/rosbag2_2026_08_12-17_27_52/rosbag2_2026_08_12-17_27_52_0.db3')
DATA=ROOT/'model_tuning/data/extracted_bags/rosbag2_2026_08_12-17_27_52.npz'
OUTPUT=ROOT/'model_tuning/results/0812_collision_analysis/speed_drop_slip_cause.png'
os.environ.setdefault('MPLCONFIGDIR','/tmp/matplotlib-smppi')
import matplotlib.pyplot as plt
from model_tuning.compare_ackermann_drive_topics import read_commands
from model_tuning.extract_training_data import read_streams
from model_tuning_utils.lateral_velocity_kf import LateralVelocityKFParams,estimate_dataset


def read_boundaries_and_rollouts():
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from nav_msgs.msg import Path as PathMessage
    from visualization_msgs.msg import MarkerArray
    reader=rosbag2_py.SequentialReader();reader.open(
        rosbag2_py.StorageOptions(uri=str(BAG),storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr','cdr'))
    boundaries={};rollouts=[]
    while reader.has_next():
        topic,raw,record_ns=reader.read_next();timestamp=record_ns*1e-9
        if topic in ('/mppi_left_boundary','/mppi_right_boundary'):
            msg=deserialize_message(raw,PathMessage)
            boundaries[topic]=np.asarray([(p.pose.position.x,p.pose.position.y) for p in msg.poses])
        elif topic=='/mppi_viz':
            msg=deserialize_message(raw,MarkerArray)
            marker=next((m for m in msg.markers if m.ns=='weighted_control_trajectory'),None)
            if marker and len(marker.points)>=2:
                points=np.asarray([(p.x,p.y) for p in marker.points])
                rollouts.append((timestamp,np.vstack((points[0],points[1::2]))))
    return boundaries,rollouts


def main():
    config=yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters']
    commands=read_commands(BAG);drive=commands['/drive'];ackermann=commands['/ackermann_cmd']
    common_start=max(drive[0,0],ackermann[0,0]);drive_time=drive[:,0]-common_start;ack_time=ackermann[:,0]-common_start
    pose,velocity,command,imu=read_streams(BAG,'/newmcl_pose','/odom','/ackermann_cmd','/imu/data')
    aligned_start=max(stream[0,0] for stream in (pose,velocity,command,imu));alignment_offset=common_start-aligned_start
    z=np.load(DATA);samples=z['samples'];dt=float(z['dt']);time=samples[:,0]-alignment_offset
    signed_imu=samples[:,12:15]*np.array([-1.,1.,-1.])
    for k in range(1,len(signed_imu)):signed_imu[k]=.25*signed_imu[k]+.75*signed_imu[k-1]
    kf_params=LateralVelocityKFParams(cornering_stiffness_front=12.7222491,cornering_stiffness_rear=75.0944752,dt=dt)
    kf_vy,_=estimate_dataset(samples,z['columns'],dt,kf_params,
        steer_scale=config['kf_steer_scale'],steer_bias=config['kf_steer_bias'],max_steer=config['kf_max_steer'],
        imu_ema_alpha=.25,imu_wz_sign=-1,imu_ay_sign=-1)
    yaw=np.unwrap(samples[:,3]);world_vx=savgol_filter(samples[:,1],21,3,deriv=1,delta=dt)
    world_vy=savgol_filter(samples[:,2],21,3,deriv=1,delta=dt)
    pose_vx=world_vx*np.cos(yaw)+world_vy*np.sin(yaw);pose_vy=-world_vx*np.sin(yaw)+world_vy*np.cos(yaw)
    pose_beta=np.degrees(np.arctan2(pose_vy,pose_vx));kf_beta=np.degrees(np.arctan2(kf_vy,samples[:,4]))
    boundaries,rollouts=read_boundaries_and_rollouts();boundary_points=np.vstack(tuple(boundaries.values()))
    boundary_rows=[]
    for timestamp,trajectory in rollouts:
        relative_time=timestamp-common_start
        if relative_time<0 or relative_time>7.9:continue
        distances=np.sqrt(((trajectory[:,None,:]-boundary_points[None,:,:])**2).sum(axis=2))
        boundary_rows.append((relative_time,float(distances.min()),trajectory))
    boundary_time=np.asarray([row[0] for row in boundary_rows]);predicted_boundary=np.asarray([row[1] for row in boundary_rows])

    slowdown_start=5.96;slowdown_end=6.79
    fig,axes=plt.subplots(4,2,figsize=(16,17));ax=axes[0,0]
    ax.plot(samples[:,1],samples[:,2],'k-',lw=2,label='Actual MCL trajectory')
    for name,points in boundaries.items():ax.plot(points[:,0],points[:,1],lw=1.2,label=name)
    for selected_time,color in ((5.96,'tab:green'),(6.20,'tab:blue'),(6.43,'tab:red'),(6.79,'tab:purple')):
        row=min(boundary_rows,key=lambda item:abs(item[0]-selected_time));ax.plot(row[2][:,0],row[2][:,1],'--',color=color,lw=1.8,label=f'weighted rollout t={row[0]:.2f}s')
        index=np.argmin(abs(time-row[0]));ax.scatter(samples[index,1],samples[index,2],c=color,s=70,zorder=4)
    ax.axis('equal');ax.set_title('Actual trajectory, boundaries, and MPPI weighted rollouts');ax.legend(fontsize=7);ax.grid(alpha=.25)
    ax=axes[0,1];ax.step(drive_time,drive[:,3],where='post',label='/drive speed (MPPI)');ax.step(ack_time,ackermann[:,3],where='post',label='/ackermann_cmd speed',alpha=.8)
    ax2=ax.twinx();ax2.step(drive_time,drive[:,2],where='post',color='tab:red',alpha=.65,label='/drive steer')
    ax.set_ylabel('speed command [m/s]');ax2.set_ylabel('steer command [rad]');ax.set_title('MPPI speed reduction while steering saturates')
    lines=ax.lines+ax2.lines;ax.legend(lines,[line.get_label() for line in lines],fontsize=8);ax.grid(alpha=.25)
    axes[1,0].plot(time,samples[:,4],label='odom vx');axes[1,0].plot(time,pose_vx,label='MCL-derived vx');axes[1,0].set_title('Longitudinal velocity')
    axes[1,1].plot(time,signed_imu[:,1],label='IMU ax');axes[1,1].plot(time,signed_imu[:,2],label='IMU ay');axes[1,1].plot(time,np.hypot(signed_imu[:,1],signed_imu[:,2]),label='combined |a|');axes[1,1].set_title('Braking and lateral acceleration')
    axes[2,0].plot(time,signed_imu[:,0],label='IMU yaw-rate');axes[2,0].set_title('Yaw-rate during braking turn')
    axes[2,1].plot(time,pose_beta,label='MCL-derived body slip beta');axes[2,1].plot(time,kf_beta,label='KF beta');axes[2,1].axhline(0,color='0.6',lw=.8);axes[2,1].set_title('Slip: pose-derived beta vs low-speed KF')
    axes[2,1].set_ylim(-35,35)
    axes[3,0].plot(boundary_time,predicted_boundary,label='weighted rollout min boundary distance');axes[3,0].axhline(config['collision_radius'],color='tab:red',ls='--',label='collision_radius=0.3m');axes[3,0].axhline(config['collision_radius']+.35,color='tab:orange',ls=':',label='soft safe distance=0.65m');axes[3,0].set_title('The cost term that triggered slowdown')
    axes[3,1].plot(time,np.abs(pose_beta),label='|MCL beta| [deg]');axes[3,1].plot(time,np.abs(signed_imu[:,2]),label='|ay| [m/s²]');axes[3,1].axhline(9.81,color='tab:red',ls='--',label='MPPI ay cost threshold');axes[3,1].set_title('Slip grows although ay heuristic never activates')
    for axis in axes.flat:
        axis.grid(alpha=.25)
        if axis is not axes[0,0]:
            axis.axvspan(slowdown_start,slowdown_end,color='tab:red',alpha=.10,label='_nolegend_');axis.set_xlim(0,9.1);axis.set_xlabel('time from autonomous /drive start [s]')
        if axis not in (axes[0,0],axes[0,1]):axis.legend(fontsize=8)
    fig.suptitle('0812 collision: boundary-cost slowdown followed by combined braking/turning slip',y=.995)
    fig.subplots_adjust(left=.07,right=.94,bottom=.05,top=.96,hspace=.42,wspace=.28);fig.savefig(OUTPUT,dpi=180);plt.close(fig);print(OUTPUT)


if __name__=='__main__':main()
