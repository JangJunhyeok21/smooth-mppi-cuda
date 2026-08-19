#!/usr/bin/env python3
"""Compare commit 67b4d4ba dynamics with the deployed 40 ms residual model.

Both models are evaluated on exactly the same source-session-disjoint 1.2 s
windows.  The old model accepted acceleration, whereas the current model accepts
a speed setpoint.  For bag replay, the old acceleration is reconstructed as the
bounded one-knot acceleration required to reach the recorded speed setpoint.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contract import Contract, actuator_step, longitudinal_actuator_step, residual_gates

DATASET_PATH = ROOT / "model_tuning/data/dynamic_40ms_residual.npz"
CURRENT_WEIGHTS = ROOT / "config/dynamic_40ms_residual_servo_lag.bin"
CURRENT_REGRESSION = ROOT / "model_tuning/results/dynamic_40ms_regression/params.json"
OUTPUT_DIR = ROOT / "model_tuning/results/overtake_67b4d4ba_vs_current"
HORIZON_S = 1.2
WINDOW_STRIDE_SOURCE_ROWS = 5

# Exact dynamics constants from commit 67b4d4ba.
OLD = dict(dt=.035, mass=3.74, lf=.163, lr=.161, iz=.04712, cm0=.04,
           Bf=6.7722, Cf=1.4462, Df=.445, Ef=.561,
           Br=8.0358, Cr=2.0762, Dr=.4161, Er=.7044,
           min_accel=-8., max_accel=8.5, min_speed=.5, max_speed=10.)


def load_mlp(path):
    z = np.fromfile(path, dtype="<f4")
    if len(z) != 3563:
        raise RuntimeError(f"{path}: expected 3563 float32 values, got {len(z)}")
    offset = 0
    def take(count):
        nonlocal offset
        value = z[offset:offset + count]
        offset += count
        return value
    return (take(1280).reshape(64, 20), take(64),
            take(2048).reshape(32, 64), take(32),
            take(96).reshape(3, 32), take(3), take(20), take(20))


def mlp(feature, weights):
    w1, b1, w2, b2, w3, b3, mean, std = weights
    hidden = np.maximum((feature - mean) / std @ w1.T + b1, 0.)
    hidden = np.maximum(hidden @ w2.T + b2, 0.)
    return np.clip(hidden @ w3.T + b3, (-8., -8., -30.), (8., 8., 30.))


def integrate_pose(pose, vx, vy, yaw_rate, dt, scale=1.):
    yaw = pose[2]
    return np.array((pose[0] + scale * (vx*np.cos(yaw)-vy*np.sin(yaw))*dt,
                     pose[1] + scale * (vx*np.sin(yaw)+vy*np.cos(yaw))*dt,
                     yaw + yaw_rate*dt))


def current_rollout(start, source, cfg, fit, weights):
    dt = .04
    steps = round(HORIZON_S/dt)
    state = source[start, :3].astype(float).copy()
    applied = float(source[start, 5])
    speed_reference = float(state[0])
    history = source[start, 10:20].reshape(5, 2).astype(float).copy()
    pose = np.zeros(3)
    trace = [np.r_[pose, state]]
    c = Contract(dt=dt, steer_scale=float(cfg['kinematic_steer_scale']),
        steer_bias=float(cfg['kinematic_steer_bias']),
        steer_tau=float(cfg['steer_servo_time_constant']),
        max_steer_rate=float(cfg['actuator_max_steer_rate']),
        speed_kp=float(cfg['speed_servo_kp']),
        speed_accel_tau=float(cfg['speed_reference_accel_time_constant']),
        speed_brake_tau=float(cfg['speed_reference_brake_time_constant']),
        max_speed_reference_rate=float(cfg['actuator_max_speed_reference_rate']),
        position_speed_scale=float(cfg['kinematic_position_speed_scale']),
        min_accel=float(cfg['min_accel']), max_accel=float(cfg['max_accel']),
        low_speed_center=float(cfg['dynamic_mlp_min_speed']))
    lf, lr, mass, iz = (float(cfg[k]) for k in ('l_f','l_r','mass','dynamic_mlp_I_z'))
    fzf = mass*9.81*lr/(lf+lr); fzr = mass*9.81*lf/(lf+lr)
    for step in range(steps):
        row = start + 2*step
        command = source[row, 3:5]
        if step:
            history = np.vstack((history[1:], command))
        previous_steer_cmd = history[-2, 0]
        current = state.copy()
        applied, _ = actuator_step(applied, command[0], command[1], state[0], c)
        speed_reference, base_ax = longitudinal_actuator_step(
            speed_reference, command[1], np.hypot(state[0], state[1]), c)
        vx, vy, yaw_rate = state
        safe_vx = max(abs(vx), .5)
        alpha_f = applied-np.arctan2(vy+lf*yaw_rate, safe_vx)
        alpha_r = -np.arctan2(vy-lr*yaw_rate, safe_vx)
        bf, br = fit['B_f']*alpha_f, fit['B_r']*alpha_r
        fyf = fzf*fit['D_f']*np.sin(fit['C_f']*np.arctan(bf))
        fyr = fzr*fit['D_r']*np.sin(fit['C_r']*np.arctan(br))
        ay = (fyf*np.cos(applied)+fyr)/mass
        yaw_accel = (lf*fyf*np.cos(applied)-lr*fyr)/iz
        base_next = np.array((vx+(base_ax+vy*yaw_rate)*dt,
                              vy+(ay-vx*yaw_rate)*dt,
                              yaw_rate+yaw_accel*dt))
        feature = np.r_[current, command, applied,
                        command[0]-previous_steer_cmd, base_next, history.ravel()]
        correction = mlp(feature, weights)*residual_gates(current[0], c)
        state = base_next+correction*dt
        pose = integrate_pose(pose, *state, dt, c.position_speed_scale)
        trace.append(np.r_[pose, state])
    return np.asarray(trace)


def old_rollout(start, source):
    dt = OLD['dt']; steps = round(HORIZON_S/dt)
    vx0, vy0, yaw_rate = source[start, :3]
    speed = float(np.hypot(vx0, vy0))
    beta = float(np.arctan2(vy0, max(abs(vx0), 1e-6)))
    pose = np.zeros(3); trace = [np.r_[pose, vx0, vy0, yaw_rate]]
    fzf = OLD['mass']*9.81*OLD['lr']/(OLD['lf']+OLD['lr'])
    fzr = OLD['mass']*9.81*OLD['lf']/(OLD['lf']+OLD['lr'])
    for step in range(steps):
        elapsed = step*dt
        row = start + int(np.floor(elapsed/.02 + 1e-9))
        steer, speed_cmd = source[row, 3:5]
        # The old node optimized acceleration and published next_v.  Reconstruct
        # that input from the recorded speed setpoint without feeding future GT.
        accel = np.clip((np.clip(speed_cmd, OLD['min_speed'], OLD['max_speed'])-speed)/dt,
                        OLD['min_accel'], OLD['max_accel'])
        if abs(speed) < .5:
            yaw_rate = speed*np.tan(steer)/(OLD['lf']+OLD['lr'])
            speed = speed+accel*dt
            beta = 0.
            vx, vy = speed, 0.
        else:
            vx, vy = speed*np.cos(beta), speed*np.sin(beta)
            alpha_f = steer-np.arctan2(vy+OLD['lf']*yaw_rate, vx)
            alpha_r = -np.arctan2(vy-OLD['lr']*yaw_rate, vx)
            bf, br = OLD['Bf']*alpha_f, OLD['Br']*alpha_r
            fyf = fzf*OLD['Df']*np.sin(OLD['Cf']*np.arctan(
                bf-OLD['Ef']*(bf-np.arctan(bf))))
            fyr = fzr*OLD['Dr']*np.sin(OLD['Cr']*np.arctan(
                br-OLD['Er']*(br-np.arctan(br))))
            yaw_accel = (OLD['lf']*fyf*np.cos(steer)-OLD['lr']*fyr)/OLD['iz']
            beta_dot = (fyf+fyr)/(OLD['mass']*speed)-yaw_rate
            speed = speed+accel*(1.-OLD['cm0']*speed)*dt
            yaw_rate = yaw_rate+yaw_accel*dt
            beta = beta+beta_dot*dt
            vx, vy = speed*np.cos(beta), speed*np.sin(beta)
        pose = integrate_pose(pose, vx, vy, yaw_rate, dt)
        trace.append(np.r_[pose, vx, vy, yaw_rate])
    return np.asarray(trace)


def ground_truth(start, source, position_speed_scale):
    steps = round(HORIZON_S/.02)
    states = source[start:start+steps+1:1, :3]
    pose = np.zeros(3); trace = [np.r_[pose, states[0]]]
    for state in states[1:]:
        pose = integrate_pose(pose, *state, .02, position_speed_scale)
        trace.append(np.r_[pose, state])
    return np.asarray(trace)


def endpoint_error(prediction, truth):
    p, g = prediction[-1], truth[-1]
    return np.array((np.linalg.norm(p[:2]-g[:2]),
                     abs(np.arctan2(np.sin(p[2]-g[2]), np.cos(p[2]-g[2]))),
                     abs(p[3]-g[3]), abs(p[4]-g[4]), abs(p[5]-g[5])))


def summary(errors):
    names = ('trajectory_m','yaw_rad','vx_mps','vy_mps','yaw_rate_rps')
    return {name:{'mean':float(np.mean(errors[:, i])),
                  'median':float(np.median(errors[:, i])),
                  'p95':float(np.quantile(errors[:, i], .95)),
                  'max':float(np.max(errors[:, i]))} for i,name in enumerate(names)}


def resample(trace, samples=61):
    old_t = np.linspace(0., HORIZON_S, len(trace))
    new_t = np.linspace(0., HORIZON_S, samples)
    return np.column_stack([np.interp(new_t, old_t, trace[:, i]) for i in range(trace.shape[1])])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = np.load(DATASET_PATH)
    source = archive['source_features'].astype(float)
    bag = archive['source_bag_id']; split = archive['source_split']; valid = archive['source_valid']
    cfg = yaml.safe_load((ROOT/'config/params.yaml').read_text())['/**']['ros__parameters']
    fit = json.loads(CURRENT_REGRESSION.read_text())['expanded_fitted']
    weights = load_mlp(CURRENT_WEIGHTS)
    all_report = {'comparison_commit':'67b4d4baddf0ff295bfd8cc84d611cc98618106d',
                  'horizon_s':HORIZON_S,
                  'fairness_note':'old acceleration reconstructed causally from recorded speed setpoint and predicted speed'}
    plot_payload = None
    for split_id, split_name in ((1,'validation'),(2,'test_aggressive')):
        starts = np.array([i for i in range(10, len(source)-61) if
            split[i]==split_id and split[i+60]==split_id and
            valid[i:i+61].all() and np.all(bag[i:i+61]==bag[i])])[::WINDOW_STRIDE_SOURCE_ROWS]
        old_traces=[]; current_traces=[]; gt_traces=[]; old_errors=[]; current_errors=[]
        for start in starts:
            gt = ground_truth(start, source, float(cfg['kinematic_position_speed_scale']))
            old = resample(old_rollout(start, source))
            current = resample(current_rollout(start, source, cfg, fit, weights))
            old_traces.append(old); current_traces.append(current); gt_traces.append(gt)
            old_errors.append(endpoint_error(old, gt)); current_errors.append(endpoint_error(current, gt))
        old_errors=np.asarray(old_errors); current_errors=np.asarray(current_errors)
        all_report[split_name]={'windows':len(starts),'overtake_67b4d4ba':summary(old_errors),
            'current_dynamic_40ms_yaw_preserved_stage2':summary(current_errors),
            'current_minus_old_mean':[float(x) for x in np.mean(current_errors-old_errors,axis=0)]}
        if split_name=='test_aggressive':
            plot_payload=(starts,np.asarray(old_traces),np.asarray(current_traces),np.asarray(gt_traces),old_errors,current_errors)
    (OUTPUT_DIR/'metrics.json').write_text(json.dumps(all_report,indent=2)+'\n')
    if plot_payload is not None:
        starts,old,current,gt,oe,ce=plot_payload
        # Rank by the old model trajectory error to expose where the historical
        # implementation succeeds and fails on the same unseen test windows.
        order=np.argsort(oe[:,0]); selected=(order[0],order[len(order)//2],order[-1])
        fig,axes=plt.subplots(3,3,figsize=(16,14))
        for col,(idx,label) in enumerate(zip(selected,('best','median','worst'))):
            axes[0,col].plot(gt[idx,:,0],gt[idx,:,1],'k',lw=2,label='GT')
            axes[0,col].plot(old[idx,:,0],old[idx,:,1],'--',label='67b4d4ba')
            axes[0,col].plot(current[idx,:,0],current[idx,:,1],label='current')
            axes[0,col].axis('equal');axes[0,col].set_title(f'{label}: old rank, row={starts[idx]}')
            t=np.linspace(0,HORIZON_S,61)
            axes[1,col].plot(t,gt[idx,:,3],'k',label='GT vx');axes[1,col].plot(t,old[idx,:,3],'--',label='old vx');axes[1,col].plot(t,current[idx,:,3],label='current vx')
            axes[1,col].plot(t,gt[idx,:,5],'k:',label='GT yaw-rate');axes[1,col].plot(t,old[idx,:,5],':',label='old yaw-rate');axes[1,col].plot(t,current[idx,:,5],':',label='current yaw-rate')
            axes[2,col].plot(t,gt[idx,:,2],'k',label='GT yaw');axes[2,col].plot(t,old[idx,:,2],'--',label='old yaw');axes[2,col].plot(t,current[idx,:,2],label='current yaw')
            axes[2,col].set_title(f'traj endpoint: old={oe[idx,0]:.3f} m, current={ce[idx,0]:.3f} m')
        for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=7);ax.set_xlabel('relative x [m]' if ax in axes[0] else 'time [s]')
        axes[0,0].set_ylabel('relative y [m]');axes[1,0].set_ylabel('vx [m/s], r [rad/s]');axes[2,0].set_ylabel('yaw [rad]')
        fig.suptitle('Held-out aggressive test: overtake 67b4d4ba vs current 40 ms residual',y=.995)
        fig.tight_layout();fig.savefig(OUTPUT_DIR/'best_median_worst.png',dpi=180);plt.close(fig)
    print(json.dumps(all_report,indent=2))


if __name__=='__main__':
    main()
