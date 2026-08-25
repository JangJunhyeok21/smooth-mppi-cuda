"""40 ms Pacejka + residual MLP plant matching CUDA MPPI."""
from pathlib import Path
import numpy as np


def load_weights(path):
    flat = np.fromfile(Path(path), dtype="<f4")
    if flat.size not in (3563, 3695, 3827):
        raise ValueError(
            f"expected 3563 (20-D), 3695 (22-D IMU), or 3827 (24-D) float32 values, "
            f"got {flat.size}: {path}")
    input_dim = {3563: 20, 3695: 22, 3827: 24}[flat.size]
    offset = 0
    def take(count, shape):
        nonlocal offset
        value = flat[offset:offset + count].reshape(shape).copy()
        offset += count
        return value
    weights = (take(64*input_dim, (64, input_dim)), take(64, (64,)),
               take(32*64, (32, 64)), take(32, (32,)),
               take(3*32, (3, 32)), take(3, (3,)))
    return weights, take(input_dim, (input_dim,)), take(input_dim, (input_dim,))


def step(state, steer_command, speed_command, command_history, actuator_steer,
         actuator_speed_reference, weights, mean, std, *, vx_history=None, dt=.04,
         current_accel=None,
         lf=.163, lr=.161, mass=3.74, min_speed=.5, max_speed=4.,
         min_accel=-1., max_accel=1., speed_servo_kp=.7616888694734905,
         speed_accel_tau=.04, speed_brake_tau=.02,
         max_speed_reference_rate=8., steer_scale=.50927964,
         steer_bias=.01015773, steer_time_constant=.15514851356820727,
         max_steer=.4788,
         max_steer_rate=.8344090950084138,
         position_speed_scale=.8633491306389823,
         Bf=2.815932661385218, Cf=1.3, Df=.37939550896899304, Ef=0.,
         Br=.3215940126165719, Cr=1.3, Dr=2.799999998765151, Er=0.,
         Iz=.04712, mlp_max_residual_ax=0., mlp_max_residual_ay=8.,
         mlp_max_residual_yaw_accel=12.):
    """Return state/history/actuator states and synthetic FLU [wz, ax, ay].

    State layout is [x,y,steer_cmd,vx,yaw,yaw_rate,beta,vy].  The command is
    held for one explicit 40 ms rollout knot, exactly as CUDA does.
    """
    current = np.asarray(state, dtype=np.float32)
    history = np.asarray(command_history, dtype=np.float32).reshape(10).copy()
    dt = np.float32(dt)
    steer_command = np.float32(steer_command)
    speed_command = np.float32(np.clip(speed_command, min_speed, max_speed))
    previous_steer_command = np.float32(history[8])

    # CUDA shifts first: the MLP feature history includes the current command.
    history[:-2] = history[2:]
    history[-2:] = (steer_command, speed_command)

    target_steer = np.float32(np.clip(
        steer_scale*steer_command+steer_bias, -max_steer, max_steer))
    steer_rate = np.float32(np.clip(
        (target_steer-actuator_steer)/max(steer_time_constant, 1e-3),
        -max_steer_rate, max_steer_rate))
    applied_steer = np.float32(np.clip(
        actuator_steer+steer_rate*dt, -max_steer, max_steer))

    speed_tau = speed_accel_tau if speed_command >= actuator_speed_reference else speed_brake_tau
    reference_rate = np.float32(np.clip(
        (speed_command-actuator_speed_reference)/max(speed_tau, 1e-3),
        -max_speed_reference_rate, max_speed_reference_rate))
    speed_reference = np.float32(actuator_speed_reference+reference_rate*dt)

    vx, vy, yaw_rate = map(np.float32, (current[3], current[7], current[5]))
    # CUDA and training contract use longitudinal body velocity here.  Using
    # hypot(vx, vy) makes a negative vx look like excessive positive speed,
    # commands braking, and can drive the simulated vehicle backwards without
    # bound after one adverse residual step.
    current_speed = vx
    base_ax = np.float32(np.clip(
        speed_servo_kp*(speed_reference-current_speed), min_accel, max_accel))
    safe_vx = np.float32(max(abs(float(vx)), .5))
    alpha_f = np.float32(applied_steer-np.arctan2(vy+lf*yaw_rate, safe_vx))
    alpha_r = np.float32(-np.arctan2(vy-lr*yaw_rate, safe_vx))
    fzf = np.float32(mass*9.81*lr/(lf+lr)); fzr = np.float32(mass*9.81*lf/(lf+lr))
    bfa, bra = np.float32(Bf*alpha_f), np.float32(Br*alpha_r)
    front_inner = np.float32(bfa-Ef*(bfa-np.arctan(bfa)))
    rear_inner = np.float32(bra-Er*(bra-np.arctan(bra)))
    front_force = np.float32(fzf*Df*np.sin(Cf*np.arctan(front_inner)))
    rear_force = np.float32(fzr*Dr*np.sin(Cr*np.arctan(rear_inner)))
    dynamic_ay = np.float32((front_force*np.cos(applied_steer)+rear_force)/mass)
    dynamic_yaw_accel = np.float32(
        (lf*front_force*np.cos(applied_steer)-lr*rear_force)/Iz)
    # Exact CUDA low-speed contract.  Pacejka and the learned residual are
    # faded in between 0.2 and 0.5 m/s; below that, a stable kinematic yaw
    # response is used instead of evaluating an ill-conditioned slip model.
    blend_input = np.float32(np.clip((abs(float(vx))-.2)/.3, 0., 1.))
    dynamic_blend = np.float32(
        blend_input*blend_input*(3.-2.*blend_input))
    low_speed_tau = np.float32(.1)
    kinematic_yaw_rate = np.float32(
        vx*np.tan(applied_steer)/max(lf+lr, 1.e-6))
    kinematic_ay = np.float32(vx*yaw_rate-vy/low_speed_tau)
    kinematic_yaw_accel = np.float32(
        (kinematic_yaw_rate-yaw_rate)/low_speed_tau)
    base_ay = np.float32(
        dynamic_blend*dynamic_ay+(1.-dynamic_blend)*kinematic_ay)
    base_yaw_accel = np.float32(
        dynamic_blend*dynamic_yaw_accel
        +(1.-dynamic_blend)*kinematic_yaw_accel)
    base_next_vx = np.float32(vx+(base_ax+vy*yaw_rate)*dt)
    base_next_vy = np.float32(vy+(base_ay-vx*yaw_rate)*dt)
    base_next_yaw_rate = np.float32(yaw_rate+base_yaw_accel*dt)

    feature = np.concatenate((np.asarray([
        vx, vy, yaw_rate, steer_command, speed_command, applied_steer,
        steer_command-previous_steer_command, base_next_vx, base_next_vy,
        base_next_yaw_rate], np.float32), history))
    next_vx_history = None
    if mean.size == 22:
        if current_accel is None:
            current_accel = np.zeros(2, dtype=np.float32)
        feature = np.concatenate(
            (feature, np.asarray(current_accel, dtype=np.float32).reshape(2)))
    if mean.size == 24:
        if vx_history is None:
            vx_history = np.full(5, vx, dtype=np.float32)
        else:
            vx_history = np.asarray(vx_history, dtype=np.float32).reshape(5)
        feature = np.concatenate((feature, np.diff(vx_history))).astype(np.float32)
    w1, b1, w2, b2, w3, b3 = weights
    normalized = ((feature-mean)/std).astype(np.float32)
    hidden1 = np.maximum(w1@normalized+b1, np.float32(0.))
    hidden2 = np.maximum(w2@hidden1+b2, np.float32(0.))
    residual_limit = np.asarray((mlp_max_residual_ax, mlp_max_residual_ay,
                                 mlp_max_residual_yaw_accel), np.float32)
    residual = np.clip(w3@hidden2+b3, -residual_limit,
                       residual_limit).astype(np.float32)
    residual *= dynamic_blend

    next_vx = np.float32(base_next_vx+residual[0]*dt)
    next_vy = np.float32(base_next_vy+residual[1]*dt)
    next_yaw_rate = np.float32(base_next_yaw_rate+residual[2]*dt)
    next_speed = np.float32(np.hypot(next_vx, next_vy))
    next_beta = np.float32(np.arctan2(next_vy, next_vx))
    output = current.copy()
    output[0] = current[0]+position_speed_scale*next_speed*np.cos(current[4]+next_beta)*dt
    output[1] = current[1]+position_speed_scale*next_speed*np.sin(current[4]+next_beta)*dt
    output[2] = steer_command
    output[3] = next_vx
    output[4] = np.arctan2(np.sin(current[4]+next_yaw_rate*dt),
                           np.cos(current[4]+next_yaw_rate*dt))
    output[5] = next_yaw_rate; output[6] = next_beta; output[7] = next_vy
    synthetic_imu = np.asarray((next_yaw_rate, base_ax+residual[0],
                                base_ay+residual[1]), np.float32)
    if mean.size == 24:
        next_vx_history = np.concatenate(
            (vx_history[1:], np.asarray([next_vx], np.float32)))
    return (output, history, applied_steer, speed_reference, synthetic_imu,
            next_vx_history)
