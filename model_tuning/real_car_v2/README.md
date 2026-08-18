# Real-car dynamics model v2

This pipeline fixes the deployed contract to 50 Hz, ISO body axes (`x` forward,
`y` left, yaw CCW), causal commands, an explicit steering/speed actuator model,
and residual derivatives `[delta_ax, delta_ay, delta_yaw_accel]`.

Dataset NPZ fields are `features (N,20)`, `targets (N,3)`, `bag_id (N,)`, and
`valid (N,)`. Timestamps must come from message headers; samples crossing a
reset, collision, manual intervention, stale topic, or large timestamp gap must
be false in `valid`. Keep complete bags/sessions in only one split. The exact
feature order is in `contract.py` and matches `update_dynamic_mlp_residual`.

The canonical rosbag sources are not mocap topics:

| Quantity | Canonical topic | Use |
|---|---|---|
| GT map pose `(x,y,yaw)` | `/newmcl_pose` | trajectory audit/visualization and runtime pose initialization |
| longitudinal velocity `vx` | `/odom` | dynamic state and supervised transition target |
| applied steering/speed command | `/drive` | causal model input; `/ackermann_cmd` is not used by the recommended dataset |
| yaw-rate and acceleration | `/imu/data` | signed/EMA yaw-rate and KF/diagnostic signals |

`/newmcl_pose` is the authoritative GT pose source. It does not provide body
velocity. The MLP itself is translation/yaw invariant and therefore does not
receive absolute `x,y,yaw`; these values are advanced outside the MLP by the
MPPI pose equations. The current recursive training pose loss constructs a
relative trajectory by integrating measured `/odom` `vx`, KF `vy`, and IMU
yaw-rate. Absolute `/newmcl_pose` remains the reference used to audit the
extracted real trajectory and localization continuity.

## Model identification before residual training

Two different tire-related identification problems are used. The **linear
2-state KF** estimates the currently unmeasured lateral velocity. The
**nonlinear Pacejka dynamic model** predicts future states inside MPPI. KF
cornering stiffnesses and Pacejka parameters are different quantities and are
not interchangeable.

### 2-state lateral-velocity Kalman filter

The KF state, input, measurement, and output are

```math
\mathbf x_k^{KF}=\begin{bmatrix}v_{y,k}&r_k\end{bmatrix}^{T}
```

```math
\mathbf u_k^{KF}=\begin{bmatrix}\delta_k&v_{x,k}\end{bmatrix}^{T}
```

```math
\mathbf z_k^{KF}=\begin{bmatrix}a_{y,k}^{IMU}&r_k^{IMU}\end{bmatrix}^{T}
```

```math
\mathrm{KF\ output}=\begin{bmatrix}\widehat v_{y,k}&\widehat r_k\end{bmatrix}^{T}
```

This observer uses a linear tire model, not Pacejka:

```math
F_{yf}=C_f^{KF}\left(\delta-\frac{v_y+l_f r}{v_x^{safe}}\right)
```

```math
F_{yr}=C_r^{KF}\left(-\frac{v_y-l_r r}{v_x^{safe}}\right)
```

```math
\dot v_y=\frac{F_{yf}+F_{yr}}{m}-v_xr
```

```math
\dot r=\frac{l_fF_{yf}-l_rF_{yr}}{I_z}
```

Here `v_x_safe=max(abs(vx),kf_min_vx)`. The equations form
`xdot=A(vx)x+B*delta`. Euler prediction and covariance propagation are

```math
\mathbf x_{k+1}^{-}=(I+A_k\Delta t)\mathbf x_k+B\Delta t\,\delta_k
```

```math
P_{k+1}^{-}=A_{d,k}P_kA_{d,k}^{T}+Q
```

The lateral-acceleration measurement equation is

```math
a_y=-\frac{C_f^{KF}+C_r^{KF}}{mv_x^{safe}}v_y
+\frac{-l_fC_f^{KF}+l_rC_r^{KF}}{mv_x^{safe}}r
+\frac{C_f^{KF}}{m}\delta
```

and the second measurement is `r_IMU=r+noise`. The implementation then uses
the ordinary Kalman innovation, gain, state correction, and covariance
correction. The deployed scalar C++/Python implementation performs no heap
allocation and directly inverts the 2-by-2 innovation covariance.

`model_tuning/regress_kf_cornering_stiffness.py` estimates only

```text
C_f_KF = kf_cornering_stiffness_front [N/rad]
C_r_KF = kf_cornering_stiffness_rear  [N/rad]
```

`mass`, `I_z`, `l_f`, `l_r`, steering scale/bias, `Q`, `R`, initial
covariance, `kf_min_vx`, low-speed threshold, and the `vy` clamp remain fixed
from `config/params.yaml`.

Offline supervision is constructed as follows:

- `/newmcl_pose` is differentiated per continuous segment and rotated into the
  body frame to obtain `vy_MCL`;
- `USE_MCL_YAW_RATE_TARGET=True` uses differentiated MCL yaw as yaw-rate
  supervision, while `False` uses signed causal-EMA IMU yaw-rate;
- signed causal-EMA `/imu/data.linear_acceleration.y` supplies `ay`.

Fitting consists of two bounded robust least-squares stages:

1. `equation_fitted` fits `Cf_KF,Cr_KF` to lateral-acceleration and
   yaw-acceleration residuals of the continuous bicycle equations. This is an
   initializer, not the deployed result.
2. `fitted` replays the exact recursive KF predict/correct algorithm and
   minimizes `vy_KF-vy_MCL` and `r_KF-r_target`. This is the deployable pair.

Both stages use SciPy `least_squares` with `loss="soft_l1"`. Optional
prediction-only open-loop replay is evaluation only; it does not fit a third
set of parameters.

Run the KF regression after editing the user settings at the top of the file:

```bash
python model_tuning/regress_kf_cornering_stiffness.py
```

The script writes `kf_cornering_stiffness.json`, a predictions NPZ, and a PNG
under its configured `OUTPUT_PATH`; it does not update YAML automatically. The
currently deployed values `12.7222491/75.0944752 N/rad` came from
`model_tuning/results/ifac0807_kf_stiffness_regression`. The current script
default points to a newer all-ackermann experiment, so its output is not the
source of those deployed numbers. Select the intended dataset explicitly and
copy only a validated `fitted` pair into YAML before rebuilding all downstream
dynamic/residual data.

### 40 ms nonlinear classic dynamic regression

The classic baseline consumes the current body state and command and predicts
one 40 ms base transition:

```math
\begin{bmatrix}v_{x,k}&v_{y,k}&r_k\end{bmatrix}^{T},
\begin{bmatrix}\delta_{cmd,k}&v_{cmd,k}\end{bmatrix}^{T}
\longrightarrow
\begin{bmatrix}v_{x,k+1}^{base}&v_{y,k+1}^{base}&r_{k+1}^{base}\end{bmatrix}^{T}
```

`model_tuning/real_car_v2/regress_dynamic_40ms.py` estimates

```text
B_f, D_f, B_r, D_r
```

It fixes `C_f=C_r=1.3` and `E_f=E_r=0`. It also fixes `mass`,
`dynamic_mlp_I_z`, `l_f`, `l_r`, position-speed scale, steering
scale/bias/time constant/rate limit, and longitudinal actuator parameters from
YAML. It therefore does not estimate mass, inertia, servo lag, speed-response
lag, KF stiffness, or MLP weights.

The objective performs a 1.0 s free rollout (`25 x 0.04 s`) over
collision-clean train windows and contains:

- time-weighted `vx`, `vy`, and yaw-rate residuals;
- terminal relative-trajectory displacement residual;
- weak parameter regularization;
- yaw-rate second-difference regularization;
- a penalty against an implausible front/rear stiffness ordering.

Optimization first uses bounded `differential_evolution` to find a global
candidate and then bounded robust `least_squares(..., loss="soft_l1")`.
Current parameter bounds are

```text
0.2 <= B_f,B_r <= 30
0.15 <= D_f,D_r <= 2.8
```

Run it with

```bash
python model_tuning/real_car_v2/regress_dynamic_40ms.py
```

The result is written to
`model_tuning/results/dynamic_40ms_regression/params.json`. A parameter within
1 percent of a bound sets `boundary_solution=true` and fails the automatic
deployment gate. The current rear solution reaches this gate; deliberate
deployment records an explicit override. Dynamic regression runs automatically
before residual target construction in
`run_yaw_preserved_40ms_pipeline.py`; KF regression currently does not.

## Recommended model: `dynamic_40ms_yaw_preserved_stage2`

This is the currently selected real-car model. It keeps the high-speed
yaw-rate solution learned with the low-speed residual gate, while runtime and
evaluation leave `delta_ax` ungated. Only the poorly observable lateral and
yaw residuals are suppressed near standstill:

```text
delta_ax        <- delta_ax
delta_ay        <- low_speed_gate(vx) * delta_ay
delta_yaw_accel <- low_speed_gate(vx) * delta_yaw_accel
```

The time contract is fixed and checked by the ROS node:

```text
ROS solve/publish period (control_dt): 0.02 s = 50 Hz
one MPPI rollout knot (model_dt):      0.04 s
horizon_steps=60:                      2.4 s rollout horizon
```

### Model input/output and equations

The training model and CUDA MPPI model use the same state transition. At one
40 ms rollout knot the dynamic state and control are

```math
\mathbf{s}_k=
\begin{bmatrix}
x_k&y_k&\psi_k&v_{x,k}&v_{y,k}&r_k
\end{bmatrix}^{\mathsf T},\qquad
\mathbf{u}_k=
\begin{bmatrix}
\delta_{cmd,k}&v_{cmd,k}
\end{bmatrix}^{\mathsf T}.
```

- `x,y,psi`: map-frame pose from localization
- `vx`: body longitudinal velocity from odometry
- `vy`: runtime KF estimate used to initialize the MPPI rollout
- `r`: signed IMU yaw rate
- `delta_cmd`: steering command
- `v_cmd`: direct speed command; `Control.accel` stores this value for legacy
  ABI compatibility, but it is not an acceleration command

The ROS controller runs every `control_dt=0.02 s`, but each model transition
below uses `Delta t=model_dt=0.04 s`. Receding-horizon control therefore solves
and publishes at 50 Hz while its rollout knots are 40 ms apart.

#### Steering and speed-reference actuator states

The model recursively carries applied steering `delta_k` and speed reference
`v_ref,k`. They are initialized from the node and maintained independently in
every CUDA rollout:

```math
\delta_k^{target}=\mathrm{clip}(S_\delta\delta_{cmd,k}+b_\delta,
-\delta_{max},\delta_{max})
```

```math
\dot\delta_k=\mathrm{clip}\left(
\frac{\delta_k^{target}-\delta_{k-1}}{\tau_\delta},
-\dot\delta_{max},\dot\delta_{max}\right)
```

```math
\delta_k=\mathrm{clip}(\delta_{k-1}+\dot\delta_k\Delta t,
-\delta_{max},\delta_{max})
```

```math
\tau_v=\tau_{accel}\quad(v_{cmd,k}\ge v_{ref,k-1})
```

```math
\tau_v=\tau_{brake}\quad(v_{cmd,k}<v_{ref,k-1})
```

```math
\dot v_{ref,k}=\mathrm{clip}\left(
\frac{v_{cmd,k}-v_{ref,k-1}}{\tau_v},
-\dot v_{ref,max},\dot v_{ref,max}\right)
```

```math
v_{ref,k}=v_{ref,k-1}+\dot v_{ref,k}\Delta t
```

```math
a_{x,k}^{base}=\mathrm{clip}\left(
K_v\left(v_{ref,k}-\sqrt{v_{x,k}^2+v_{y,k}^2}\right),
a_{min},a_{max}\right)
```

#### Classic Pacejka dynamic model

With `v_safe=max(abs(vx),0.5)`, the front/rear slip angles are

```math
\alpha_{f,k}=\delta_k-
\mathrm{atan2}(v_{y,k}+l_f r_k,v_{safe,k}),\qquad
\alpha_{r,k}=-\mathrm{atan2}(v_{y,k}-l_r r_k,v_{safe,k}).
```

Static normal loads and Pacejka lateral forces are

```math
\begin{aligned}
F_{zf}&=mg\frac{l_r}{l_f+l_r},&
F_{zr}&=mg\frac{l_f}{l_f+l_r},\\
q_f&=B_f\alpha_f-E_f(B_f\alpha_f-\tan^{-1}(B_f\alpha_f)),&
q_r&=B_r\alpha_r-E_r(B_r\alpha_r-\tan^{-1}(B_r\alpha_r)),\\
F_{yf}&=F_{zf}D_f\sin(C_f\tan^{-1}(q_f)),&
F_{yr}&=F_{zr}D_r\sin(C_r\tan^{-1}(q_r)).
\end{aligned}
```

The classic lateral acceleration and yaw acceleration are

```math
a_{y,k}^{base}=\frac{F_{yf,k}\cos\delta_k+F_{yr,k}}{m},\qquad
\dot r_k^{base}=\frac{l_fF_{yf,k}\cos\delta_k-l_rF_{yr,k}}{I_z}.
```

The classic next body state supplied to the MLP is

```math
\begin{aligned}
v_{x,k+1}^{base}
&=v_{x,k}+(a_{x,k}^{base}+v_{y,k}r_k)\Delta t,\\
v_{y,k+1}^{base}
&=v_{y,k}+(a_{y,k}^{base}-v_{x,k}r_k)\Delta t,\\
r_{k+1}^{base}
&=r_k+\dot r_k^{base}\Delta t.
\end{aligned}
```

#### Residual MLP

The MLP is `20 -> 64 -> 32 -> 3` with ReLU hidden activations. Its exact input
order is

```math
\mathbf{z}_k=\left[
\begin{array}{c}
v_{x,k},v_{y,k},r_k,\delta_{cmd,k},v_{cmd,k},
\delta_k,\Delta\delta_{cmd,k},\\
v_{x,k+1}^{base},v_{y,k+1}^{base},r_{k+1}^{base},\\
\delta_{cmd,k-4},v_{cmd,k-4},
\delta_{cmd,k-3},v_{cmd,k-3},
\delta_{cmd,k-2},v_{cmd,k-2},\\
\delta_{cmd,k-1},v_{cmd,k-1},
\delta_{cmd,k},v_{cmd,k}
\end{array}
\right]^{\mathsf T}\in\mathbb{R}^{20},
```

where

```math
\Delta\delta_{cmd,k}=\delta_{cmd,k}-\delta_{cmd,k-1}.
```

Feature normalization is stored in the binary:

```math
\bar{\mathbf z}_k=\frac{\mathbf z_k-\boldsymbol\mu}{\boldsymbol\sigma},
\qquad
\begin{aligned}
\mathbf h_1&=\mathrm{ReLU}(W_1\bar{\mathbf z}_k+b_1),\\
\mathbf h_2&=\mathrm{ReLU}(W_2\mathbf h_1+b_2),\\
\begin{bmatrix}\Delta a_x&\Delta a_y&\Delta\dot r\end{bmatrix}^{\mathsf T}
&=W_3\mathbf h_2+b_3.
\end{aligned}
```

The output is not a next state and is not the total acceleration. It is the
derivative residual added to the classic model. Output safety clamps are
`[-8,8] m/s^2` for `delta_ax,delta_ay` and `[-30,30] rad/s^2` for
`delta_yaw_accel`.

Only lateral/yaw residuals use the low-speed gate:

```math
g(v_x)=\frac{1}{1+\exp(-(\lvert v_x\rvert-v_g)/0.2)},\qquad
\widetilde{\Delta\mathbf d}_k=
\begin{bmatrix}
\Delta a_x&g(v_x)\Delta a_y&g(v_x)\Delta\dot r
\end{bmatrix}^{\mathsf T}.
```

Keeping `delta_ax` ungated is essential: otherwise the classic
`a_max=1 m/s^2` limit makes measured standstill launches structurally
unreachable. `vy` and yaw residuals remain gated because they are poorly
observable near zero speed.

The corrected body state is

```math
\begin{aligned}
v_{x,k+1}&=v_{x,k+1}^{base}+\Delta a_{x,k}\Delta t,\\
v_{y,k+1}&=v_{y,k+1}^{base}+g(v_{x,k})\Delta a_{y,k}\Delta t,\\
r_{k+1}&=r_{k+1}^{base}+g(v_{x,k})\Delta\dot r_k\Delta t.
\end{aligned}
```

Finally MPPI advances the map pose with the corrected next body state:

```math
\begin{aligned}
x_{k+1}&=x_k+S_p(v_{x,k+1}\cos\psi_k-v_{y,k+1}\sin\psi_k)\Delta t,\\
y_{k+1}&=y_k+S_p(v_{x,k+1}\sin\psi_k+v_{y,k+1}\cos\psi_k)\Delta t,\\
\psi_{k+1}&=\mathrm{wrap}(\psi_k+r_{k+1}\Delta t).
\end{aligned}
```

#### Training targets and losses

For a measured transition spanning 40 ms, the supervised target is

```math
\mathbf y_k=
\frac{1}{\Delta t}
\begin{bmatrix}
v_{x,k+1}^{GT}-v_{x,k+1}^{base}\\
v_{y,k+1}^{GT}-v_{y,k+1}^{base}\\
r_{k+1}^{GT}-r_{k+1}^{base}
\end{bmatrix}
=
\begin{bmatrix}\Delta a_x^{GT}&\Delta a_y^{GT}&\Delta\dot r^{GT}\end{bmatrix}^{\mathsf T}.
```

The one-step stage fits these residual derivatives. Two subsequent stages run
the model freely for 1.2 s and supervise body state, position, and dense
yaw-rate errors, with extra weight on the first 0.24 s. High-speed and
yaw-rate recovery/sign-change windows are oversampled. Aggressive run1 supplies
3--4 m/s excitation; aggressive run2 remains completely held out for the
reported final evaluation.

### Simplest complete command

The direct-bag NPZ files must already exist under
`model_tuning/data/real_car_v2_drive/` and
`model_tuning/results/effective_vs_dynamic_0813/data/`. Then run:

```bash
cd /home/a/smooth-mppi-cuda
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate RL
python model_tuning/real_car_v2/run_yaw_preserved_40ms_pipeline.py
```

No CLI arguments are required. The switches at the top of
`run_yaw_preserved_40ms_pipeline.py` allow completed stages to be skipped.
The following subsections describe what each enabled stage actually does. They
also distinguish raw-bag extraction from `build_dataset.py`: the runner starts
from previously extracted 20 ms NPZ files and does **not** read rosbag2 storage
itself.

#### 1. `build_dataset.py`: combine audited 20 ms bag extracts

This stage reads every `*.npz` in these two directories:

- `model_tuning/data/real_car_v2_drive/`
- `model_tuning/results/effective_vs_dynamic_0813/data/`

The input files must have been created directly from rosbag messages. The
reconstructed diagnostic file
`prediction_vs_actual_run12_reconstructed.csv` and anything derived from it are
explicitly forbidden as training input.

Raw rosbag extraction is performed beforehand by
`model_tuning/extract_training_data.py`. Its current default observation
contract is:

| Quantity | Default topic | Extracted value |
|---|---|---|
| global pose | `/newmcl_pose` | global `x`, `y`, quaternion-derived yaw |
| longitudinal observation | `/odom` | `twist.twist.linear.x` as `vx` |
| command | `/ackermann_cmd` | steering angle, acceleration field, speed command |
| inertial observation | `/imu/data` | yaw-rate, `ax`, `ay` |

The topic names are configurable at the top of the extractor. In particular,
`build_dataset.py` does not rename or silently convert `/drive` into
`/ackermann_cmd`; it preserves each NPZ's `command_topic` in its manifest.
Therefore all source NPZ files used in one identification run should be
audited to ensure they contain the intended actuator-facing command. The
current extractor default is `/ackermann_cmd`, even though some older comments
and directory names contain the word `drive`.

For each bag, the extractor prefers the message header timestamp and falls back
to the rosbag record timestamp only when the header stamp is absent or zero. It
creates a 50 Hz common grid (`dt=0.02 s`) over the interval shared by all four
topics. Alignment is causal zero-order hold: at time `t`, only the newest
sample whose timestamp is less than or equal to `t` may be used. No future
pose, command, odometry, or IMU sample is interpolated into the present. The
current stale limits are 1.0 s for pose, odometry and command, and 0.05 s for
IMU.

Collision/reverse recovery is handled before an NPZ reaches this stage. The
filter uses measured `/odom` body `vx`, not `drive.acceleration`:

1. A reverse-recovery seed is detected when `vx < -0.15 m/s` persists for at
   least 0.06 s.
2. The suspected collision interval starts just after the last healthy
   `vx > 0.7 m/s` sample in the preceding 3 s. If none exists, a conservative
   0.5 s look-back is used.
3. Samples remain excluded through the reverse maneuver and are accepted again
   only after `vx > 0.3 m/s` is sustained for 0.5 s.
4. Every remaining continuous interval is assigned a separate segment ID, so
   recursive windows cannot cross a collision cut, timestamp gap, or recovery
   boundary. The later recursive-window selector also rejects an abnormally
   large state jump, but the extractor does not currently implement a separate
   localization-reset detector.

`build_dataset.py` then performs the runtime-equivalent preprocessing. It runs
the same 2-state lateral KF to obtain `vy` and filtered yaw-rate, applies the
configured IMU signs and causal EMA, removes only the stationary median bias
from IMU `ax`, recursively generates applied steering and speed reference, and
constructs the exact 20 CUDA features. It rejects non-finite rows, the first
five history rows, and samples outside the implemented sanity limits
(`|vx|<=6`, `|vy|<=2`, `|r|<=5`, `|ax|,|ay|<=15`, and
`|yaw_acceleration|<=40` in SI units).

The split is source-session disjoint rather than a random row split:

- `aggressive_boundary_run2.npz`: final test only
- `effective_speed30_run1.npz` and `rosbag2_2026_08_08-16_54_33.npz`:
  validation
- all remaining collision-clean segments: training, including aggressive
  run1 for otherwise sparse 3--4 m/s excitation

Every continuous segment receives a globally unique `bag_id`. The output is
`model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz`, accompanied by a
JSON manifest containing source path, segment, split, valid count, removed IMU
bias and command topic. Its targets are diagnostic 20 ms derivative residuals;
the next stage rebuilds the actual 40 ms supervised targets.

#### 2. `regress_dynamic_40ms.py`: identify the classic Pacejka baseline

This stage identifies the nonlinear classic model that the residual MLP will
correct. Each optimization rollout starts from the observed state

```math
\boldsymbol{s}_k=
\begin{bmatrix}v_{x,k}&v_{y,k}&r_k\end{bmatrix}^{\mathsf T}
```

and receives

```math
\boldsymbol{u}_k=
\begin{bmatrix}\delta_{cmd,k}&v_{cmd,k}\end{bmatrix}^{\mathsf T}.
```

The output of one model knot is the predicted next body state
`[base_next_vx, base_next_vy, base_next_yaw_rate]`. Although the source data is
on a 20 ms grid, one MPPI knot is one explicit Euler actuator/physics update at
`dt=0.04 s`; commands and ground truth are therefore indexed every two source
rows. The classic regression does not train an MLP and does not use residual
targets.

The steering actuator first evolves the carried applied-steering state:

```math
\begin{aligned}
\delta_k^{target}&=\mathrm{clip}(S_\delta\delta_{cmd,k}+b_\delta,
                                 -0.55,0.55),\\
\dot\delta_k&=\mathrm{clip}((\delta_k^{target}-\delta_{k-1})/\tau_\delta,
                             -\dot\delta_{max},\dot\delta_{max}),\\
\delta_k&=\mathrm{clip}(\delta_{k-1}+0.04\dot\delta_k,-0.55,0.55).
\end{aligned}
```

The carried speed reference is updated using separate acceleration/braking
time constants and a reference-rate limit. Longitudinal acceleration is
`clip(K_p(v_ref-hypot(vx,vy)), min_accel, max_accel)`. The lateral forces use
front and rear slip angles and normalized axle loads:

```math
\begin{aligned}
\alpha_f&=\delta_k-\tan^{-1}((v_y+l_f r)/\max(|v_x|,0.5)),\\
\alpha_r&=-\tan^{-1}((v_y-l_r r)/\max(|v_x|,0.5)),\\
F_{y,i}&=F_{z,i}D_i\sin(C_i\tan^{-1}(B_i\alpha_i)),\qquad i\in\{f,r\}.
\end{aligned}
```

The regression estimates only

```math
\boldsymbol{\theta}=\begin{bmatrix}B_f&D_f&B_r&D_r\end{bmatrix}^{\mathsf T}.
```

`C_f=C_r=1.3` and `E_f=E_r=0` are fixed. Mass, yaw inertia, axle distances,
steering scale/bias/time constant/rate limit, speed-reference dynamics,
longitudinal gain and acceleration limits are read from `config/params.yaml`
and are not re-estimated here. The bounds are `0.2<=B_f,B_r<=30` and
`0.15<=D_f,D_r<=2.8`.

This means `regress_dynamic_40ms.py` alone cannot determine whether those fixed
actuator and position parameters are correct. They must either be measured,
identified by a preceding regression, or retained as a separately validated
configuration. The current YAML is a mixture of those sources:

| YAML parameter | Current value | Provenance |
|---|---:|---|
| `kinematic_position_speed_scale` | 0.8633491306 | Fitted by `model_tuning/regress_kinematic_model.py` on `ifac_0808_0810_train_test.npz` using 1.0 s recursive pose/yaw/speed rollouts. The result is in `model_tuning/results/ifac_0808_0810_slip_regression/kinematic_model_regression.json`. |
| `steer_servo_time_constant` | 0.1551485136 s | Fitted by the same joint kinematic rollout regression. |
| `actuator_max_steer_rate` | 0.8344090950 rad/s | Fitted by the same joint kinematic rollout regression. |
| `kinematic_steer_scale` | 0.50927964 | A historical no-servo-lag/effective-model configuration restored in commit `a85a9bc`; it is **not** the fitted scale from the above 0808/0810 regression, which was 1.1330467889. |
| `kinematic_steer_bias` | 0.01015773 rad | Restored with the same historical configuration; the above regression produced 0.0095019621 rad. |
| `speed_servo_kp` | 0.7616888695 1/s | Fitted in the older 20 ms dynamic regression stored at `model_tuning/results/ifac_all_ackermann_dynamic_regression_previous_steer/dynamic_params.json`; it was not jointly re-identified by the current 40 ms classic regression. |
| `speed_reference_accel_time_constant` | 0.04 s | Retained held-out validation setting, not the accepted output of the current longitudinal regression. |
| `speed_reference_brake_time_constant` | 0.02 s | Retained held-out validation setting. |
| `actuator_max_speed_reference_rate` | 8.0 m/s² | Retained held-out validation setting. |

There is a separate longitudinal identification script,
`model_tuning/real_car_v2/regress_longitudinal_actuator.py`. It holds
`speed_servo_kp` fixed and estimates acceleration time constant, braking time
constant and reference slew-rate limit from 1.0 s recursive `/odom`-`vx`
rollouts. Its latest candidate was approximately `[0.002 s, 0.01977 s,
40.845 m/s²]`, but it reached the acceleration-time lower bound and made test
MAE slightly worse. Consequently `deployment_gate_passed` was false and the
YAML retained `[0.04 s, 0.02 s, 8.0 m/s²]`. See
`model_tuning/results/longitudinal_actuator_regression/regression.json`.

The joint `regress_kinematic_model.py` result also must not be copied only in
parts without revalidation: steering scale, bias, lag, rate, speed gain and
position scale are coupled when no physical steering-angle feedback is
available. The current YAML mixes its fitted lag/rate/position scale with a
different historical steering scale/bias and dynamic-regression speed gain.
This combination is known by held-out rollout selection, not by one
simultaneous identifiable regression.

Therefore the one-command yaw-preserved pipeline currently has the following
dependency:

```text
preselected/fixed actuator and position parameters in params.yaml
    -> regress_dynamic_40ms.py fits only B_f, D_f, B_r, D_r
    -> residual target generation and MLP training
```

It does **not** run `regress_kinematic_model.py` or
`regress_longitudinal_actuator.py`. If the vehicle, steering linkage, VESC
configuration, surface, command topic or localization scale changes, these
fixed values must be identified and held-out validated again before running
the 40 ms Pacejka/residual pipeline. Without steering feedback, steering
actuator parameters and tire parameters are only jointly identifiable, so an
independent validation bag is required even when the optimizer converges.

##### How to obtain every fixed parameter when the vehicle changes

Do not estimate every quantity in one unconstrained optimizer. Several terms
produce nearly the same trajectory change and are not independently observable
from pose, odometry and IMU alone. Use the following order, freeze each accepted
group, and rebuild/evaluate every downstream artifact after an accepted change.

###### A. Directly measure geometry, mass and hard limits

| Parameter | Preferred determination method | Regression fallback |
|---|---|---|
| `mass` | Weigh the complete race-ready car including battery and sensors. | Do not freely fit it together with tire `D`: lateral force scale and mass strongly compensate. If necessary, search only a narrow measured interval and validate on a separate bag. |
| `l_f`, `l_r` | Measure each axle to the loaded center of mass and verify that their sum equals wheelbase. | A narrow bounded yaw-response fit is possible, but is weak without steering-angle feedback. Keep wheelbase fixed. |
| `dynamic_mlp_I_z` | Use CAD mass properties, a bifilar/trifilar pendulum, or a dedicated yaw-inertia experiment. | Fit only inside a narrow prior with tire parameters fixed; otherwise `I_z` and tire yaw moment compensate. |
| `max_steer`, `min_accel`, `max_accel` | Use hardware travel limits and robust percentiles of collision-free measured response. | Treat these as safety limits, not unconstrained least-squares variables. |

Changing one of these values changes the classic target, so rerun steps 2--8
of the canonical pipeline. Changing a KF geometry parameter additionally
requires rebuilding the combined dataset because estimated `vy` changes.

###### B. Position/odometry scale

`kinematic_position_speed_scale` maps integrated `/odom` body velocity to
displacement in the `/newmcl_pose` map frame. Estimate it on long, mostly
straight, collision-free intervals:

```math
S_p^*=\arg\min_{S_p}\sum_j
\rho\left(S_p\sum_{k\in j}R(\psi_k)
\begin{bmatrix}v_{x,k}&v_{y,k}\end{bmatrix}^{\mathsf T}\Delta t
-\left(\boldsymbol p_{j,end}^{MCL}-\boldsymbol p_{j,start}^{MCL}\right)\right).
```

Set `DATASET_PATH` and `OUTPUT_PATH` at the top, then run:

```bash
python model_tuning/regress_position_speed_scale.py
```

The current implementation uses train rows only, 1.0 s windows, robust
`soft_l1`, pose-jump rejection, and prefers `|steer_cmd|<=0.08 rad` and
`|yaw_rate|<=0.3 rad/s`. Accept a value only when independent validation/test
displacement mean and p95 improve. Pacejka or the MLP must not be used to hide
an incorrect localization/odometry scale.

###### C. Steering command map and actuator dynamics

The preferred experiment uses measured wheel/rack angle or calibrated servo
feedback. Apply safe step and chirp commands at several speeds, align feedback
causally, and fit

```math
\delta^{target}=\mathrm{clip}(S_\delta\delta_{cmd}+b_\delta),
\qquad
\dot\delta=\mathrm{clip}((\delta^{target}-\delta)/\tau_\delta,
                         -\dot\delta_{max},\dot\delta_{max}).
```

Settled samples identify `kinematic_steer_scale` and
`kinematic_steer_bias`; transients identify `steer_servo_time_constant` and
`actuator_max_steer_rate`. Use separate left/right sweeps to detect deadband,
asymmetry and saturation.

Without steering feedback, use the following only as a diagnostic effective
command-to-yaw fit:

```bash
python model_tuning/real_car_v2/identify_steering_actuator_rollout.py \
  <collision-clean-dataset.npz> \
  --out model_tuning/results/steering_actuator_rollout.json
```

It jointly fits steering scale, bias, time constant, rate limit and front/rear
Pacejka `B,D`. It intentionally does not edit YAML because actuator and tire
parameters can compensate each other. Reject bound solutions and require
improvement in both temporal validation and source-session-disjoint bags.
After selecting a candidate, rerun `regress_dynamic_40ms.py` with its actuator
values fixed and verify that the tire solution remains inside its bounds.

`model_tuning/regress_kinematic_model.py` estimates the same actuator group
together with speed gain and position scale, but uses the older slip-kinematic
equations. Its output is suitable for initialization/comparison, not direct
copying into the 40 ms dynamic model without held-out dynamic evaluation.

###### D. Longitudinal actuator parameters

Use collision-free sessions containing acceleration, steady-speed and braking
transients. The fitted response model is

```math
\begin{aligned}
\dot v_{ref}&=\mathrm{clip}((v_{cmd}-v_{ref})/\tau_{accel/brake},
                            -\dot v_{ref,max},\dot v_{ref,max}),\\
a_x&=\mathrm{clip}(K_p(v_{ref}-\sqrt{v_x^2+v_y^2}),a_{min},a_{max}).
\end{aligned}
```

Run the current-contract tool with no arguments:

```bash
python model_tuning/real_car_v2/regress_longitudinal_actuator.py
```

It fits `speed_reference_accel_time_constant`,
`speed_reference_brake_time_constant`, and
`actuator_max_speed_reference_rate` using 1.0 s recursive `/odom vx` Huber
loss, while `speed_servo_kp` and acceleration clamps remain fixed. It updates
YAML only if the candidate is interior to its bounds and improves validation
and test MAE. A boundary result means the excitation cannot separate time
constant from rate saturation.

To identify `speed_servo_kp`, first fit it from unsaturated
speed-error/acceleration pairs with the time constants fixed, or extend the
search to `[K_p,tau_accel,tau_brake,max_rate]`. The latter requires acceleration
and braking steps of several amplitudes; otherwise these variables are not
separately observable. Use `/odom vx` as the response ground truth, never
command speed.

###### E. KF parameters

First fix IMU axes/signs and steering mapping. Then use
`regress_kf_cornering_stiffness.py` to estimate the KF `C_f,C_r` as described
above. Determine the remaining settings as follows:

- `kf_min_vx` is a numerical denominator floor chosen from stability analysis,
  not tire-regression output.
- Select `kf_low_speed_threshold` by comparing KF `vy` with offline
  pose-derived `vy`; use the lowest speed above which the observer is stable.
  The extractor and node must use the same value.
- Increase `kf_q_vy`/`kf_q_yaw_rate` if response is too slow; decrease them if
  estimates follow noise excessively.
- Estimate `kf_r_lateral_accel`/`kf_r_yaw_rate` from stationary and
  constant-speed sensor variance after axis/sign and bias correction, then tune
  on held-out recursive KF error.
- Choose initial covariance and reset gap from convergence behavior after a
  bag/node reset.

Any accepted KF change requires rerunning `build_dataset.py`, because all
later `vy`, slip angle, classic prediction and MLP targets depend on it.

###### F. Pacejka choices and full acceptance order

Without direct tire force, steering-angle or lateral-velocity sensing, fitting
all `B,C,D,E,m,I_z` simultaneously is ill-conditioned. The recommended model
fixes measured `m,I_z,l_f,l_r`, sets `C=1.3,E=0`, and fits only
`B_f,D_f,B_r,D_r`. With richer excitation and steering feedback, release one
additional parameter group at a time with regularization and physical bounds;
lower training loss alone is not an acceptance criterion.

```text
sensor axes/time alignment
  -> measured geometry/mass/limits
  -> position-speed scale
  -> steering map and actuator response
  -> longitudinal actuator response
  -> KF Cf/Cr and noise settings
  -> rebuild 20 ms combined dataset
  -> 40 ms Pacejka regression
  -> rebuild residual targets
  -> one-step MLP
  -> recursive stage1/stage2
  -> held-out evaluation, CUDA parity, deployment
```

The current bags have no steering-angle feedback, so the available code cannot
justify automatically replacing the mixed actuator values in `params.yaml`.
For that reason this documentation update does not change runtime parameters
or retrain the MLP. Retraining an unchanged contract would reproduce the same
targets, while changing a weakly identified actuator candidate first could
degrade real-car tail behavior. Retrain only after a candidate passes the
independent validation rules above.

Identification uses collision-clean training segments only. A candidate is
rolled out freely for 25 knots, or 1.0 s, instead of receiving the measured
state again at every step. Per bag, at most 180 approximately distributed
starts are retained, and windows with mean `|vx|<=0.5 m/s` are excluded. The
objective contains:

- time-increasing open-loop errors in `vx`, `vy`, and yaw-rate, weighted
  respectively by 0.4, 2.0, and 1.5;
- terminal relative-position error reconstructed from the predicted body
  velocities and yaw-rate;
- weak regularization toward a physically plausible reference parameter set;
- a yaw-rate second-difference term to discourage oscillatory dynamics;
- a penalty when the fitted front small-angle lateral gain exceeds the rear
  gain.

Optimization is performed in two passes. SciPy differential evolution first
searches globally using a clipped robust cost (`seed=31`, population 8,
40 iterations). Its solution initializes bounded nonlinear least squares with
`soft_l1` loss and scale 0.3 for local refinement. The report evaluates the old
and fitted parameters independently on train, validation and aggressive-test
rollouts. If any parameter is within 1% of its lower or upper bound,
`deployment_gate_passed` becomes false because that solution is weakly
identified; deployment then requires the explicit override currently present
in `deploy_dynamic_40ms_to_mppi.py`.

The fitted eight-field runtime representation
`[B_f,C_f,D_f,E_f,B_r,C_r,D_r,E_r]`, boundary status and rollout metrics are
written to `model_tuning/results/dynamic_40ms_regression/params.json`.

#### 3. `build_dynamic_40ms_dataset.py`: form residual derivative targets

This stage loads the fitted classic parameters and converts consecutive 20 ms
rows into one 40 ms transition. It reconstructs the applied steering, speed
reference and classic prediction exactly as the runtime contract does. The
feature vector contains the current state and command, applied steering,
command delta, classic next state and five-command history. The target is not
the full next state or the full acceleration; it is the derivative correction

```math
\boldsymbol{y}_k=
(\boldsymbol{s}_{k+1}^{GT}-\boldsymbol{s}_{k+1}^{base})/0.04
=\begin{bmatrix}\Delta a_x&\Delta a_y&\Delta\dot r\end{bmatrix}^{\mathsf T}.
```

All three involved 20 ms rows must be valid and belong to the same segment and
split. The output is `model_tuning/data/dynamic_40ms_residual.npz`.

#### 4. `train_dynamic_40ms.py`: one-step residual MLP training

The network is `20 -> 64 -> 32 -> 3`, with ReLU after both hidden layers. Its
input is normalized using training-only mean and standard deviation; its
outputs are `[delta_ax, delta_ay, delta_yaw_accel]`. AdamW minimizes weighted
Smooth-L1 loss with output weights `[1,2,2]`. Samples above 3 m/s receive four
times the base probability, while inverse-square-root bag weighting prevents
long bags from dominating. Validation rows select the checkpoint, with
50-epoch early stopping. The final layer is converted back to physical output
units before export, and the binary contains weights, biases, feature mean and
feature standard deviation in the exact 14252-byte CUDA layout.

#### 5. `finetune_dynamic_40ms_recursive.py`: two free-rollout stages

Stage1 starts from the one-step checkpoint and stage2 starts from stage1. Each
stage rolls the complete actuator, classic Pacejka and MLP model recursively
for 30 knots (1.2 s); future predicted states, rather than measured feedback,
become the next model inputs. Loss is applied at 0.2, 0.4, 0.8 and 1.2 s and
contains weighted `[vx,vy,r]`, relative position and dense yaw-rate error.
Yaw-angle loss is intentionally zero because this memoryless residual predicts
derivatives, while early yaw-rate samples receive an exponentially decaying
extra weight over approximately the first 0.2 s.

The runner sets `GATE_AX_RESIDUAL=1` for both recursive passes. Consequently
all three residual heads use the low-speed sigmoid gate during this particular
fine-tuning reproduction; evaluation/runtime deliberately leave longitudinal
`delta_ax` ungated while continuing to gate lateral/yaw residuals. Training
windows are oversampled for high speed and yaw-rate decay/sign-recovery cases,
and checkpoint selection minimizes validation mean yaw-rate loss plus twice
its 95th percentile.

#### 6. `evaluate_dynamic_40ms.py`: held-out open-loop evaluation

The stage runs 30-step/1.2 s free-recursive rollouts with the deployed
normalization, residual clamps and runtime residual gates. It evaluates the
validation split and the completely unseen aggressive test split, taking
candidate windows every five source rows. It saves mean, p95 and maximum
trajectory, yaw, `vx`, `vy`, and yaw-rate errors to JSON, plus predicted and GT
traces to a companion NPZ. Passing `--disable-mlp` is available for diagnostic
classic-only evaluation but is not used by the canonical runner.

#### 7. `deploy_dynamic_40ms_to_mppi.py`: validate and activate runtime files

Deployment checks that the learned binary contains exactly 3563 little-endian
float32 values (14252 bytes), reads the fitted Pacejka JSON and checks its
boundary gate. It then copies the binary to
`config/dynamic_40ms_residual_servo_lag.bin`, writes all eight Pacejka values,
the weight path and `model_dt: 0.04` into `config/params.yaml`, and selects
`dynamics_model: dynamic_mlp_residual_servo_lag`. These are runtime-loaded
data, so rebuilding with `colcon build` is unnecessary; the ROS node must be
restarted.

#### 8. `plot_highspeed_tail_comparison.py`: compare tail behavior

The final plotting stage reads prerecorded held-out aggressive run2 traces for
the effective model, the older 40 ms lag model and the newly trained model. It
saves a 1.2 s best/median/worst comparison. This is a visualization of fixed
held-out traces, not another optimizer step and not a source of training data.

With all switches enabled the runner executes this same sequence:

1. `build_dataset.py`: combine all direct-bag 20 ms NPZ data. Every continuous
   segment receives a unique `bag_id`; aggressive run1 is high-speed training
   excitation and aggressive run2 remains an unseen test bag.
2. `regress_dynamic_40ms_advanced.py`: compare robust DE+LS, differentiable
   multi-start Adam rollout learning and MLP-surrogate global search for all
   eight front/rear Pacejka `B,C,D,E` parameters. Select only by held-out bag
   validation open-loop score; mass, inertia and geometry remain fixed.
3. `build_dynamic_40ms_dataset.py`: calculate classic predictions and the
   derivative residual targets `[delta_ax, delta_ay, delta_yaw_accel]`.
4. `train_dynamic_40ms.py`: train the 20-64-32-3 MLP using one-step targets.
5. `finetune_dynamic_40ms_recursive.py`: perform two 1.2 s free-recursive
   passes with `GATE_AX_RESIDUAL=1`, producing
   `dynamic_40ms_yaw_preserved_stage1` and `stage2`.
6. `evaluate_dynamic_40ms.py`: evaluate validation and aggressive test bags
   with runtime-identical ungated `delta_ax` and save JSON/NPZ traces.
7. `deploy_dynamic_40ms_to_mppi.py`: verify and copy the 14252-byte CUDA
   binary, insert the fitted Pacejka parameters and weight path into
   `config/params.yaml`, and select `dynamic_mlp_residual_servo_lag`.
8. `plot_highspeed_tail_comparison.py`: save the effective/old/new held-out
   best/median/worst comparison.

All frequently changed paths and hyperparameters are constants at the top of
the corresponding script. No CLI arguments are needed for the canonical run.
The runner also has `RUN_*` switches at its top, so a completed expensive stage
can be skipped without constructing a long command.

`RUN_DEPLOY_TO_MPPI=True` enables the final YAML/deployment step. The deployment
script settings are at its top: `RESULT_PATH`, `REGRESSION_PATH`, `YAML_PATH`,
`RUNTIME_BINARY_PATH`, and `ACTIVATE_MODEL`. It changes runtime data only, so a
`colcon build` is not needed; restart the ROS node so it reloads YAML and binary.

For debugging, the equivalent stage commands are shown below. The two recursive
commands intentionally set `GATE_AX_RESIDUAL=1`; omitting it produces a
different checkpoint.

```bash
python model_tuning/real_car_v2/build_dataset.py
python model_tuning/real_car_v2/regress_dynamic_40ms.py
python model_tuning/real_car_v2/build_dynamic_40ms_dataset.py
python model_tuning/real_car_v2/train_dynamic_40ms.py
GATE_AX_RESIDUAL=1 python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_residual_seed31 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage1
GATE_AX_RESIDUAL=1 python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_yaw_preserved_stage1 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage2
python model_tuning/real_car_v2/evaluate_dynamic_40ms.py \
  model_tuning/results/dynamic_40ms_yaw_preserved_stage2 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage2/rollout_ax_ungated_metrics.json
python model_tuning/real_car_v2/plot_highspeed_tail_comparison.py
python model_tuning/real_car_v2/deploy_dynamic_40ms_to_mppi.py
```

The final artifacts are:

```text
model_tuning/results/dynamic_40ms_yaw_preserved_stage2/dynamic_40ms_residual.bin
model_tuning/results/dynamic_40ms_yaw_preserved_stage2/rollout_ax_ungated_metrics.json
config/dynamic_40ms_residual_servo_lag.bin
```

Its binary contract remains little-endian float32, 3563 floats / 14252 bytes:
three linear-layer weights and biases followed by 20 feature means and 20
feature standard deviations.

## Commands used for the reported 0813 result

The reported result was generated with the same stages as the runner. Before
the no-argument defaults were added, the parameterized training stages were:

```bash
python model_tuning/real_car_v2/train_dynamic_40ms.py \
  model_tuning/data/dynamic_40ms_residual.npz \
  --out model_tuning/results/dynamic_40ms_residual_seed31 \
  --epochs 300 --seed 31 --device cuda

python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_residual_seed31 \
  --out model_tuning/results/dynamic_40ms_recursive_seed31 \
  --epochs 100 --seed 31

python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_recursive_seed31 \
  --out model_tuning/results/dynamic_40ms_recursive_stage2_seed31 \
  --epochs 100 --seed 31
```

These long commands are retained here only as the experiment record. For a new
canonical run, use `run_yaw_preserved_40ms_pipeline.py` without arguments.

`regress_longitudinal_actuator.py` identifies acceleration/braking speed-reference
time constants and its slew-rate limit from 1 s recursive odom-vx rollouts. It
uses source-session-disjoint validation/test bags and writes a candidate report.
Set `UPDATE_CONFIG=True` only after full residual-rollout validation passes.
Run it before rebuilding the residual targets: otherwise the MLP is trained to
compensate for the old actuator parameters.

`check_mppi_step_parity.py` feeds the same state, command, actuator states and
history to the PyTorch training step and to the exact CUDA
`update_dynamic_mlp_residual` step. Run it on the Orin/GPU host after building;
the test fails when the maximum full-state/history difference exceeds `2e-5`.

The runtime names are intentionally distinct. `dynamic_mlp_residual` is the
causal no-lag contract (`steer[t]=steer_cmd[t-1]`, direct speed command), while
`dynamic_mlp_residual_servo_lag` recursively maintains applied steering and
speed-reference actuator states. Each rollout knot is one 40 ms update, while
ROS still resolves and publishes a new first control at 50 Hz. Its checkpoint
path must not be swapped with the 20 ms no-lag model.

Do not select the new binary on the car until: hold-out 1-step and 60-step
replay beat physics-only, p95/max errors do not regress, Python/CUDA predictions
match at 1 and 60 steps, OOD cases remain finite, and Orin MPPI latency meets the
50 Hz budget. Preserve `contract.json`, `split_manifest.json`, and metrics with
every checkpoint. The exporter intentionally retains the deployed 3563-float
format; the current default model is not changed automatically.
