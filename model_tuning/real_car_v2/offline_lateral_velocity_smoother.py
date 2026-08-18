#!/usr/bin/env python3
"""Robust offline vy target using MCL pose and IMU lateral dynamics.

This is deliberately non-causal and is only suitable for training targets.
Runtime MPPI continues to use the causal two-state lateral KF.
"""
import numpy as np
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr


def _huber_weight(residual, scale):
    magnitude = np.abs(residual)
    return np.where(magnitude <= scale, 1.0, scale/np.maximum(magnitude, 1e-9))


def smooth_segment_vy(x, y, yaw, vx, yaw_rate, lateral_accel, dt,
                      pose_window_s=.30, pose_sigma=.25,
                      dynamics_sigma=1.0, low_speed_sigma=.35,
                      second_difference_sigma=.12, iterations=5):
    """Return offline body-frame vy and diagnostics for one continuous segment."""
    n = len(x)
    window = max(7, int(round(pose_window_s/dt)) | 1)
    window = min(window, n//2*2-1)
    if window < 7:
        return np.zeros(n), {"usable": False, "reason": "segment_too_short"}
    order = min(3, window-2); heading = np.unwrap(yaw)
    world_vx = savgol_filter(x, window, order, deriv=1, delta=dt)
    world_vy = savgol_filter(y, window, order, deriv=1, delta=dt)
    pose_vy = -np.sin(heading)*world_vx + np.cos(heading)*world_vy
    target_dvy = lateral_accel-vx*yaw_rate

    # Sparse linear residual blocks. Robust IRLS downweights localization
    # corrections and IMU impulses without allowing either sensor to dominate.
    identity = coo_matrix((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    row = np.repeat(np.arange(n-1), 2); col = np.c_[np.arange(n-1), np.arange(1, n)].ravel()
    derivative = coo_matrix((np.tile((-1/dt, 1/dt), n-1), (row, col)), shape=(n-1, n)).tocsr()
    if n >= 3:
        row2 = np.repeat(np.arange(n-2), 3); col2 = np.c_[np.arange(n-2), np.arange(1, n-1), np.arange(2, n)].ravel()
        second = coo_matrix((np.tile((1., -2., 1.), n-2), (row2, col2)), shape=(n-2, n)).tocsr()
    low_weight = np.clip(1.-np.abs(vx)/.7, 0., 1.)
    estimate = pose_vy.copy(); pose_weight = np.ones(n); dynamics_weight = np.ones(n-1)
    edge = window//2; pose_weight[:edge] = .05; pose_weight[-edge:] = .05
    for _ in range(iterations):
        blocks = [identity.multiply(np.sqrt(pose_weight)[:, None]/pose_sigma),
                  derivative.multiply(np.sqrt(dynamics_weight)[:, None]/dynamics_sigma),
                  identity.multiply(np.sqrt(low_weight)[:, None]/low_speed_sigma)]
        rhs = [np.sqrt(pose_weight)*pose_vy/pose_sigma,
               np.sqrt(dynamics_weight)*target_dvy[:-1]/dynamics_sigma,
               np.zeros(n)]
        if n >= 3:
            blocks.append(second/second_difference_sigma)
            rhs.append(np.zeros(n-2))
        estimate = lsqr(vstack(blocks).tocsr(), np.concatenate(rhs),
                        atol=1e-7, btol=1e-7, iter_lim=300)[0]
        pose_weight *= _huber_weight(estimate-pose_vy, .35)
        dynamics_weight *= _huber_weight(np.diff(estimate)/dt-target_dvy[:-1], 1.5)
        pose_weight = np.maximum(pose_weight, .01); dynamics_weight = np.maximum(dynamics_weight, .03)
    diagnostics = {"usable": True, "pose_vy_mae": float(np.mean(abs(estimate-pose_vy))),
                   "imu_dynamics_mae": float(np.mean(abs(np.diff(estimate)/dt-target_dvy[:-1]))),
                   "max_abs_vy": float(np.max(abs(estimate)))}
    return estimate, diagnostics
