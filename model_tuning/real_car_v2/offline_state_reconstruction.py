#!/usr/bin/env python3
"""Robust non-causal reconstruction of latent body state from MCL and IMU.

No command, actuator, tire, or vehicle-dynamics model is used here.  The
result is a training target; odometry vx is retained only as a diagnostic.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.sparse import lil_matrix


def wrap(value):
    return (value + np.pi) % (2*np.pi) - np.pi


@dataclass
class ReconstructionConfig:
    pose_window_s: float = .30
    sigma_xy: float = .025
    sigma_yaw: float = .012
    sigma_gyro: float = .035
    sigma_ax: float = .9
    sigma_ay: float = .9
    sigma_multi: float = .06
    sigma_pose_velocity_prior: float = .12
    sigma_smooth_vx: float = .09
    sigma_smooth_vy: float = .12
    sigma_smooth_r: float = .18
    odom_vx_weight: float = 0.0
    estimate_gyro_bias: bool = True
    multi_horizons: tuple = (3, 5, 8, 25, 50)
    huber_scale: float = 1.5
    max_nfev: int = 80


def _initial_state(x, y, yaw, gyro, dt, window_s):
    n = len(x); window = max(5, int(round(window_s/dt)) | 1)
    window = min(window, n//2*2-1)
    if window >= 5:
        order = min(3, window-2)
        wx = savgol_filter(x, window, order, deriv=1, delta=dt)
        wy = savgol_filter(y, window, order, deriv=1, delta=dt)
    else:
        wx, wy = np.gradient(x, dt), np.gradient(y, dt)
    heading = np.unwrap(yaw)
    return np.c_[wx*np.cos(heading)+wy*np.sin(heading),
                 -wx*np.sin(heading)+wy*np.cos(heading), gyro]


def derive_state_from_mcl(x, y, yaw, dt, imu_ax=None, imu_ay=None,
                          imu_yaw_rate=None, odom_vx=None, window_s=.18,
                          yaw_rate_crossover_hz=3.0):
    """Create offline state directly from a smooth interpolated MCL pose.

    IMU and odometry are diagnostic-only and never influence the returned GT.
    """
    x=np.asarray(x,float);y=np.asarray(y,float);yaw=np.asarray(yaw,float);n=len(x)
    if n<10:raise ValueError("offline MCL derivative needs >=10 samples")
    window=max(7,int(round(window_s/dt))|1);window=min(window,n//2*2-1)
    if window<5:raise ValueError("segment is too short for MCL smoothing")
    order=min(3,window-2);heading=np.unwrap(yaw)
    sx=savgol_filter(x,window,order);sy=savgol_filter(y,window,order)
    sheading=savgol_filter(heading,window,order)
    wx=savgol_filter(x,window,order,deriv=1,delta=dt)
    wy=savgol_filter(y,window,order,deriv=1,delta=dt)
    mcl_yaw_rate=savgol_filter(heading,window,order,deriv=1,delta=dt)
    vx=wx*np.cos(sheading)+wy*np.sin(sheading)
    vy=-wx*np.sin(sheading)+wy*np.cos(sheading)
    raw_wx=np.gradient(x,dt);raw_wy=np.gradient(y,dt)
    raw_state=np.c_[raw_wx*np.cos(heading)+raw_wy*np.sin(heading),
                    -raw_wx*np.sin(heading)+raw_wy*np.cos(heading),
                    np.gradient(heading,dt)]
    if imu_yaw_rate is not None:
        gyro=np.asarray(imu_yaw_rate,float)
        # Zero-phase complementary estimate: MCL supplies absolute/low-frequency
        # yaw motion; gyro supplies fast transients. Constant gyro bias is in
        # the rejected low-frequency component and cannot become yaw drift.
        nyquist=.5/dt;cutoff=min(float(yaw_rate_crossover_hz),.8*nyquist)
        sos=butter(3,cutoff/nyquist,btype="lowpass",output="sos")
        gyro_low=sosfiltfilt(sos,gyro)
        mcl_low=sosfiltfilt(sos,mcl_yaw_rate)
        yaw_rate=mcl_low+(gyro-gyro_low)
    else:
        yaw_rate=mcl_yaw_rate.copy()
    state=np.c_[vx,vy,yaw_rate];smooth_pose=np.c_[sx,sy,wrap(sheading)]
    integrated=free_integrate(sx[0],sy[0],smooth_pose[0,2],state,dt)
    mean=.5*(state[:-1]+state[1:])
    ax=np.r_[np.diff(vx)/dt-mean[:,2]*mean[:,1],np.nan]
    ay=np.r_[np.diff(vy)/dt+mean[:,2]*mean[:,0],np.nan]
    xy_error=np.linalg.norm(integrated[:,:2]-smooth_pose[:,:2],axis=1)
    yaw_error=wrap(integrated[:,2]-smooth_pose[:,2])
    metrics={"method":"smoothed_mcl_pose_derivative_with_complementary_gyro_yaw_rate",
             "smoothing_window_s":float(window*dt),
             "yaw_rate_crossover_hz":float(yaw_rate_crossover_hz),
             "xy_rmse_m":float(np.sqrt(np.mean(xy_error**2))),
             "final_position_drift_m":float(xy_error[-1]),
             "xy_error":_statistics(xy_error),"yaw_error_rad":_statistics(yaw_error)}
    if imu_yaw_rate is not None:
        gyro=np.asarray(imu_yaw_rate,float);metrics["imu_yaw_rate_error_rad_s"]=_statistics(yaw_rate-gyro)
        metrics["mcl_yaw_rate_vs_imu_rad_s"]=_statistics(mcl_yaw_rate-gyro)
        metrics["time_alignment_diagnostics"]={"imu_gyro_vs_mcl_yaw_rate":_best_offset(mcl_yaw_rate,gyro,dt)}
    if imu_ax is not None:metrics["imu_ax_error_m_s2"]=_statistics(ax[:-1]-np.asarray(imu_ax)[:-1])
    if imu_ay is not None:metrics["imu_ay_error_m_s2"]=_statistics(ay[:-1]-np.asarray(imu_ay)[:-1])
    ground_speed=np.hypot(vx,vy)
    slip=((np.asarray(odom_vx)-ground_speed)/np.maximum(ground_speed,.1)
          if odom_vx is not None else np.full(n,np.nan))
    valid=np.isfinite(state).all(1);edge=window//2
    # Savitzky-Golay uses one-sided polynomial extrapolation at segment edges;
    # keep those values for plots but never expose them as supervised GT.
    valid[:edge]=False;valid[-edge:]=False
    return {"state":state,"initial_state":raw_state,"mcl_smoothed_pose":smooth_pose,
            "pose":integrated,"ax":ax,"ay":ay,"longitudinal_slip_ratio":slip,
            "valid_mask":valid,"metrics":metrics,
            "config":{"method":"smoothed MCL pose + complementary MCL/IMU yaw-rate",
                      "window_s":window*dt,"yaw_rate_crossover_hz":yaw_rate_crossover_hz,
                      "imu_acceleration_role":"diagnostic_only",
                      "imu_yaw_rate_role":"high_frequency_GT_component",
                      "odom_vx_role":"diagnostic_only"}}


def free_integrate(x0, y0, yaw0, state, dt):
    """Trapezoidal state integration with no intermediate pose injection."""
    n=len(state); pose=np.empty((n,3)); pose[0]=(x0,y0,yaw0)
    for k in range(n-1):
        vx,vy,r=.5*(state[k]+state[k+1])
        middle=wrap(pose[k,2]+.5*r*dt)
        pose[k+1,0]=pose[k,0]+(vx*np.cos(middle)-vy*np.sin(middle))*dt
        pose[k+1,1]=pose[k,1]+(vx*np.sin(middle)+vy*np.cos(middle))*dt
        pose[k+1,2]=wrap(pose[k,2]+r*dt)
    return pose


def _statistics(error):
    error=np.asarray(error)
    return {"rmse":float(np.sqrt(np.mean(error**2))),
            "median_abs":float(np.median(np.abs(error))),
            "p95_abs":float(np.quantile(np.abs(error),.95))}


def _best_offset(reference, measurement, dt, max_offset_s=.12):
    """Diagnostic-only offset scan; it never shifts data inside the fit."""
    limit=int(round(max_offset_s/dt)); best=(float("inf"),0)
    for shift in range(-limit,limit+1):
        if shift<0:a,b=reference[-shift:],measurement[:shift]
        elif shift>0:a,b=reference[:-shift],measurement[shift:]
        else:a,b=reference,measurement
        if len(a)<5:continue
        score=float(np.sqrt(np.mean((a-b)**2)))
        if score<best[0]:best=(score,shift)
    return {"measurement_offset_s":float(best[1]*dt),"rmse":best[0]}


def reconstruct_segment(x, y, yaw, imu_ax, imu_ay, imu_yaw_rate, dt,
                        odom_vx=None, config=ReconstructionConfig()):
    """Estimate [vx, vy, r] jointly and return states, pose, and diagnostics."""
    arrays=[np.asarray(q,float) for q in (x,y,yaw,imu_ax,imu_ay,imu_yaw_rate)]
    x,y,yaw,imu_ax,imu_ay,gyro=arrays; n=len(x)
    if n < 10 or not all(len(q)==n for q in arrays):
        raise ValueError("offline reconstruction needs >=10 equally-sized samples")
    initial=_initial_state(x,y,yaw,gyro,dt,config.pose_window_s)
    bias0=float(np.median(gyro-np.gradient(np.unwrap(yaw),dt))) if config.estimate_gyro_bias else 0.
    z0=np.r_[initial.ravel(),bias0] if config.estimate_gyro_bias else initial.ravel()
    heading=np.unwrap(yaw); dxy=np.diff(np.c_[x,y],axis=0); dyaw=np.diff(heading)

    def residual(z):
        state=z[:3*n].reshape(n,3); bias=z[-1] if config.estimate_gyro_bias else 0.
        mean=.5*(state[:-1]+state[1:]); angle=.5*(heading[:-1]+heading[1:])
        predicted=np.c_[(mean[:,0]*np.cos(angle)-mean[:,1]*np.sin(angle))*dt,
                        (mean[:,0]*np.sin(angle)+mean[:,1]*np.cos(angle))*dt]
        result=[((predicted-dxy)/config.sigma_xy).ravel(),
                (mean[:,2]*dt-dyaw)/config.sigma_yaw,
                (state[:,2]+bias-gyro)/config.sigma_gyro,
                ((np.diff(state[:,0])/dt-mean[:,2]*mean[:,1]-imu_ax[:-1])/config.sigma_ax),
                ((np.diff(state[:,1])/dt+mean[:,2]*mean[:,0]-imu_ay[:-1])/config.sigma_ay),
                ((state[:,:2]-initial[:,:2])/config.sigma_pose_velocity_prior).ravel()]
        for horizon in config.multi_horizons:
            if n<=horizon: continue
            world=np.c_[mean[:,0]*np.cos(angle)-mean[:,1]*np.sin(angle),
                        mean[:,0]*np.sin(angle)+mean[:,1]*np.cos(angle)]*dt
            cumulative=np.vstack((np.zeros(2),np.cumsum(world,axis=0)))
            prediction=cumulative[horizon:]-cumulative[:-horizon]
            measured=np.c_[x[horizon:]-x[:-horizon],y[horizon:]-y[:-horizon]]
            result.append(((prediction-measured)/(config.sigma_multi*np.sqrt(horizon))).ravel())
        if n>=3:
            second=state[2:]-2*state[1:-1]+state[:-2]
            result.extend((second[:,0]/config.sigma_smooth_vx,
                           second[:,1]/config.sigma_smooth_vy,
                           second[:,2]/config.sigma_smooth_r))
        if config.odom_vx_weight>0 and odom_vx is not None:
            result.append(np.sqrt(config.odom_vx_weight)*(state[:,0]-odom_vx))
        return np.concatenate(result)

    # Exact structural sparsity keeps finite differences practical for long bags.
    probe=residual(z0); sparsity=lil_matrix((len(probe),len(z0)),dtype=np.int8); row=0
    for k in range(n-1):
        sparsity[row:row+2,[3*k,3*k+1,3*(k+1),3*(k+1)+1]]=1; row+=2
    for k in range(n-1): sparsity[row,[3*k+2,3*(k+1)+2]]=1; row+=1
    for k in range(n):
        sparsity[row,3*k+2]=1
        if config.estimate_gyro_bias:sparsity[row,-1]=1
        row+=1
    for component in (0,1):
        for k in range(n-1):
            sparsity[row,3*k:3*k+3]=1; sparsity[row,3*(k+1):3*(k+1)+3]=1; row+=1
    for k in range(n):
        sparsity[row:row+2,[3*k,3*k+1]]=1; row+=2
    for horizon in config.multi_horizons:
        if n<=horizon:continue
        for k in range(n-horizon):
            columns=[]
            for j in range(k,k+horizon+1): columns.extend((3*j,3*j+1))
            sparsity[row:row+2,columns]=1; row+=2
    if n>=3:
        for component in range(3):
            for k in range(n-2):
                sparsity[row,[3*k+component,3*(k+1)+component,3*(k+2)+component]]=1; row+=1
    if config.odom_vx_weight>0 and odom_vx is not None:
        for k in range(n): sparsity[row,3*k]=1; row+=1
    assert row == len(probe)
    result=least_squares(residual,z0,loss="huber",f_scale=config.huber_scale,
                         jac_sparsity=sparsity.tocsr(),max_nfev=config.max_nfev,verbose=0)
    state=result.x[:3*n].reshape(n,3); bias=float(result.x[-1]) if config.estimate_gyro_bias else 0.
    pose=free_integrate(x[0],y[0],yaw[0],state,dt)
    mean=.5*(state[:-1]+state[1:])
    ax=np.r_[np.diff(state[:,0])/dt-mean[:,2]*mean[:,1],np.nan]
    ay=np.r_[np.diff(state[:,1])/dt+mean[:,2]*mean[:,0],np.nan]
    xy_error=np.linalg.norm(pose[:,:2]-np.c_[x,y],axis=1); yaw_error=wrap(pose[:,2]-yaw)
    metrics={"xy_rmse_m":float(np.sqrt(np.mean(xy_error**2))),
             "final_position_drift_m":float(xy_error[-1]),
             "xy_error":_statistics(xy_error),"yaw_error_rad":_statistics(yaw_error),
             "gyro_error_rad_s":_statistics(state[:,2]+bias-gyro),
             "ax_error_m_s2":_statistics(ax[:-1]-imu_ax[:-1]),
             "ay_error_m_s2":_statistics(ay[:-1]-imu_ay[:-1]),
             "time_alignment_diagnostics":{
                 "imu_gyro_vs_mcl_yaw_derivative":_best_offset(np.gradient(heading,dt),gyro,dt),
                 "imu_ax_vs_offline_ax":_best_offset(ax[:-1],imu_ax[:-1],dt),
                 "imu_ay_vs_offline_ay":_best_offset(ay[:-1],imu_ay[:-1],dt)},
             "gyro_bias_rad_s":bias,"optimizer_success":bool(result.success),
             "optimizer_cost":float(result.cost),"optimizer_nfev":int(result.nfev)}
    slip=(np.asarray(odom_vx)-state[:,0])/np.maximum(np.abs(state[:,0]),.1) if odom_vx is not None else np.full(n,np.nan)
    return {"state":state,"initial_state":initial,"pose":pose,"ax":ax,"ay":ay,
            "longitudinal_slip_ratio":slip,"valid_mask":np.isfinite(state).all(1),
            "metrics":metrics,"config":asdict(config)}


def save_validation_plot(path, t, x, y, yaw, imu_ax, imu_ay, gyro, odom_vx,
                         reconstruction, odom_vy=None, speed_cmd=None,title="offline state reconstruction"):
    """Write the required pose/state/acceleration diagnostic summary."""
    import matplotlib.pyplot as plt
    state=reconstruction["state"]; initial=reconstruction["initial_state"]
    pose=reconstruction["pose"]; ax=reconstruction["ax"]; ay=reconstruction["ay"]
    raw_r=np.gradient(np.unwrap(yaw),np.asarray(t))
    fig,axes=plt.subplots(6,2,figsize=(15,23)); fig.suptitle(title)
    axes[0,0].plot(x,y,label="MCL"); axes[0,0].plot(pose[:,0],pose[:,1],label="offline free integration")
    axes[0,0].axis("equal"); axes[0,0].set_title("XY trajectory")
    axes[0,1].plot(t,x,label="MCL X"); axes[0,1].plot(t,pose[:,0],label="reconstructed X")
    axes[1,0].plot(t,y,label="MCL Y"); axes[1,0].plot(t,pose[:,1],label="reconstructed Y")
    axes[1,1].plot(t,np.unwrap(yaw),label="MCL yaw"); axes[1,1].plot(t,np.unwrap(pose[:,2]),label="reconstructed yaw")
    axes[2,0].plot(t,np.hypot(initial[:,0],initial[:,1]),label="raw MCL ground speed")
    axes[2,0].plot(t,np.hypot(state[:,0],state[:,1]),label="offline ground speed")
    axes[2,0].plot(t,odom_vx,label="odom wheel speed")
    axes[2,1].plot(t,initial[:,1],label="MCL derivative init vy"); axes[2,1].plot(t,state[:,1],label="offline vy")
    axes[3,0].plot(t,raw_r,label="MCL yaw derivative"); axes[3,0].plot(t,gyro,label="IMU gyro"); axes[3,0].plot(t,state[:,2],label="offline yaw-rate")
    axes[3,1].plot(t,imu_ax,label="IMU ax"); axes[3,1].plot(t,ax,label="offline ax"); axes[3,1].plot(t,imu_ay,label="IMU ay"); axes[3,1].plot(t,ay,label="offline ay")
    slip=reconstruction["longitudinal_slip_ratio"]
    axes[4,0].scatter(np.hypot(state[:,0],state[:,1]),slip,s=3,alpha=.35); axes[4,0].set(xlabel="offline ground speed [m/s]",ylabel="slip ratio",title="wheel slip vs offline ground speed")
    axes[4,1].scatter(imu_ax,slip,s=3,alpha=.35); axes[4,1].set(xlabel="IMU ax [m/s2]",ylabel="slip ratio",title="wheel slip vs IMU ax")
    if speed_cmd is not None: axes[5,0].scatter(speed_cmd,slip,s=3,alpha=.35)
    axes[5,0].set(xlabel="speed command [m/s]",ylabel="slip ratio",title="wheel slip vs speed command")
    axes[5,1].plot(t,initial[:,0],color="tab:red",alpha=.5,label="raw MCL-difference body vx")
    axes[5,1].plot(t,state[:,0],color="tab:blue",label="offline body vx")
    axes[5,1].set_title("Body vx only (no odom comparison)")
    for axis in axes.flat:
        axis.grid(alpha=.25); axis.set_xlabel("time [s]")
        if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=150); plt.close(fig)

    # A separate measurement-consistency report keeps unlike units/states out
    # of the same panel and makes bias/time-offset patterns visible.
    consistency,ca=plt.subplots(5,1,figsize=(15,17),sharex=True)
    yaw_residual=state[:,2]-gyro
    ca[0].plot(t,gyro,color="tab:orange",alpha=.75,label="signed raw IMU yaw-rate")
    ca[0].plot(t,state[:,2],color="tab:blue",lw=1.7,label="classic KF/RTS yaw-rate")
    ca[0].plot(t,yaw_residual,color="tab:red",alpha=.55,label="GT - IMU residual")
    ca[0].set_ylabel("rad/s");ca[0].set_title("Yaw-rate measurement consistency")
    ax_residual=ax-imu_ax
    ca[1].plot(t,imu_ax,color="tab:orange",alpha=.7,label="signed raw IMU ax")
    ca[1].plot(t,ax,color="tab:blue",lw=1.5,label="state-derived offline ax")
    ca[1].plot(t,ax_residual,color="tab:red",alpha=.5,label="offline - IMU residual")
    ca[1].set_ylabel("m/s²");ca[1].set_title("Longitudinal acceleration consistency")
    ay_residual=ay-imu_ay
    ca[2].plot(t,imu_ay,color="tab:orange",alpha=.7,label="signed raw IMU ay")
    ca[2].plot(t,ay,color="tab:blue",lw=1.5,label="state-derived offline ay")
    ca[2].plot(t,ay_residual,color="tab:red",alpha=.5,label="offline - IMU residual")
    ca[2].set_ylabel("m/s²");ca[2].set_title("Lateral acceleration consistency")
    if odom_vy is not None:
        odom_vy=np.asarray(odom_vy,float);odom_vx=np.asarray(odom_vx,float);time=np.asarray(t,float)
        odom_dvx=np.gradient(odom_vx,time);odom_dvy=np.gradient(odom_vy,time)
        odom_ax=odom_dvx-gyro*odom_vy;odom_ay=odom_dvy+gyro*odom_vx
        ca[3].plot(t,imu_ax,color="tab:orange",label="signed raw IMU ax")
        ca[3].plot(t,odom_dvx,color="tab:green",alpha=.55,label="d(odom vx)/dt")
        ca[3].plot(t,odom_ax,color="tab:blue",alpha=.8,label="d(odom vx)/dt - r·odom_vy")
        ca[4].plot(t,imu_ay,color="tab:orange",label="signed raw IMU ay")
        ca[4].plot(t,odom_dvy,color="tab:green",alpha=.55,label="d(odom vy)/dt")
        ca[4].plot(t,odom_ay,color="tab:blue",alpha=.8,label="d(odom vy)/dt + r·odom_vx")
    ca[3].set_ylabel("m/s²");ca[3].set_title("IMU ax vs odom-vx slope")
    ca[4].set_ylabel("m/s²");ca[4].set_title("IMU ay vs odom-vy slope")
    for axis in ca:
        axis.grid(alpha=.25);axis.legend(fontsize=8);axis.set_xlabel("time [s]")
    consistency.suptitle(title+" — IMU/state consistency");consistency.tight_layout()
    consistency_path=path.with_name(path.stem+"_consistency"+path.suffix)
    consistency.savefig(consistency_path,dpi=150);plt.close(consistency)
