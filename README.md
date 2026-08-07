# smppi_cuda_controller

Simulator counterpart: [Ros2-F1tenth-Simulator — no-slip kinematic version](https://github.com/kms8527/Ros2-F1tenth-Simulator/tree/986fe88c7bb5e5c8506b0ad31d330875b2a3d66c)

F1TENTH 자율주행을 위한 CUDA 가속 MPPI(Model Predictive Path Integral) 컨트롤러.  
NVIDIA GPU에서 대규모 병렬 샘플링으로 실시간 궤적을 최적화하고 조향/가속 명령을 출력한다.

---

## 알고리즘 개요

MPPI는 K개의 랜덤 제어 시퀀스를 GPU에서 병렬 rollout하고, 각 궤적의 비용을 기반으로  
최적 제어 입력을 가중 평균으로 산출하는 샘플링 기반 MPC이다.

```
u* = Σ w_k · ε_k      where   w_k ∝ exp(−J_k / λ)
```

### 차량 동역학 모델 (실행 중 선택 가능)

| `dynamics_model` | 모델 |
|---|---|
| `legacy_hybrid` | `v<0.5 m/s` kinematic, 그 이상 Pacejka dynamic |
| `kinematic` | 전 속도 kinematic bicycle |
| `kinematic_residual` | kinematic + CUDA GRU residual |
| `kinematic_mlp_residual` | kinematic + CUDA MLP residual (현재 설정) |

**파세이카 타이어 모델 (Pacejka Magic Formula):**
```
F_y = D · sin(C · atan(B · α))
```
- `α_f = δ − atan((vy + l_f·ω) / vx)`  (전륜 슬립각)
- `α_r = −atan((vy − l_r·ω) / vx)`     (후륜 슬립각)

---

## 비용 함수

| 항목 | 파라미터 | 설명 |
|------|----------|------|
| 경로 이탈 | `q_dist` | 레퍼런스 경로까지의 거리² |
| 속도 추종 | `q_v` | 목표 속도와의 편차 |
| 제어 변화율 | `q_du` | 조향·가속 변화량 |
| 조향량 | `q_steer` | 조향각 크기 패널티 |
| 충돌 | `q_collision` | 경계선 침범 시 급격한 패널티 |
| 횡가속도 | `q_lat_g` | 9.5 m/s² 초과 시 지수 패널티 |
| 진행 보상 | `q_progress` | 경로 방향 진행 보상 (음의 비용) |
| 탈출 속도 | `q_escape_vel` | 저속 고착 상황 탈출 |

---

## 노드 구성

### `smppi_node` (MPPI 컨트롤러)

**구독**

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/ego_racecar/odom` (현재 simulator 설정) | `nav_msgs/Odometry` | pose + body twist |
| `/imu/data` | `sensor_msgs/Imu` | residual의 causal IMU 입력 |
| `/mppi_target_path` | `nav_msgs/Path` | 레퍼런스 경로 + 목표 속도 |
| `/mppi_left_boundary` | `nav_msgs/Path` | 좌측 주행 경계 |
| `/mppi_right_boundary` | `nav_msgs/Path` | 우측 주행 경계 |

**발행**

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/ackermann_cmd` (`drive_topic`) | `ackermann_msgs/AckermannDriveStamped` | 조향 + 가속 명령 |
| `/mppi_viz` | `visualization_msgs/MarkerArray` | 샘플 궤적 시각화 |
| `/mppi_optimal_trajectory` | `smppi_cuda_controller/MppiTrajectory` | 최적 궤적 + 비용 분해 |

**제어 주기:** 20 ms (50 Hz, `control_rate_hz`로 설정)

---

### `path_publisher` (경로 발행기)

CSV 파일에서 센터라인을 읽어 곡률 기반 속도 프로파일을 계산하고  
레퍼런스 경로와 좌·우 경계를 발행한다.

**CSV 포맷:** `x, y, [width_left, width_right]`

**발행 토픽**

| 토픽 | 설명 |
|------|------|
| `/mppi_target_path` | 센터라인 + 속도 프로파일 (z 성분에 목표 속도 삽입) |
| `/mppi_left_boundary` | 좌측 경계 (transient_local QoS) |
| `/mppi_right_boundary` | 우측 경계 (transient_local QoS) |

---

## 주요 파라미터 (`config/params.yaml`)

### 샘플링

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `num_samples` | `8000` | 병렬 샘플 수 (시뮬: 10000) |
| `lambda` | `15.0` | 온도 파라미터 (클수록 평균에 가까움) |
| `noise_steer_std` | `0.4` | 조향 노이즈 표준편차 [rad/s] |
| `noise_accel_std` | `2.0` | 가속 노이즈 표준편차 [m/s³] |

### 속도 제한

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `target_speed` | `5.5` | 목표 속도 [m/s] |
| `max_speed` | `3.0` | 최대 허용 속도 [m/s] |
| `min_speed` | `0.5` | 최소 속도 [m/s] |
| `max_steer` | `0.38` | 최대 조향각 [rad] |
| `max_accel` / `min_accel` | `±9.0` | 가속도 한계 [m/s²] |

### 차량 모델

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `mass` | `3.74` | 차량 질량 [kg] |
| `l_f` / `l_r` | `0.163` / `0.161` | 앞/뒤 축까지 거리 [m] |
| `I_z` | `0.14` | 요 관성 모멘트 [kg·m²] |
| `B_f, C_f, D_f` | `1.5, 1.5, 40.0` | 전륜 파세이카 계수 |
| `B_r, C_r, D_r` | `1.5, 1.5, 35.5` | 후륜 파세이카 계수 |

### 토픽

| 파라미터 | 기본값 |
|----------|--------|
| `odom_topic` | `/ego_racecar/odom` (현재 시뮬레이터 설정) |
| `drive_topic` | `/sim_drive` (현재 시뮬레이터 설정) |
| `path_topic` | `/mppi_target_path` |

---

## 빌드 및 실행

```bash
# 빌드 (CUDA 필요)
colcon build --packages-select smppi_cuda_controller

source install/setup.bash

# 단독 실행 (런치 파일)
ros2 launch smppi_cuda_controller cuda_mppi.launch.py

# EKF와 통합 실행 (ekf_pose 패키지 런치 사용 권장)
ros2 launch ekf_pose smppi_with_ekf.launch.py
```

---

## Kinematic + MLP residual 모델 학습 및 배포

현재 `config/params.yaml`의 `dynamics_model: kinematic_mlp_residual`이 선택하는
모델이다. CUDA 구현은 `src/mppi_core.cu`의 `update_kinematic()`과
`update_kinematic_mlp_residual()`에 있으며, simulator는
`f1tenth_gym/envs/dynamic_models/kinematic_mlp.py`에서 같은 계산을 수행한다.

### 1. Classic kinematic 모델 입출력

MPPI 한 step의 상태와 action은 다음과 같다.

```math
\mathbf{s}_t=[x_t,y_t,\psi_t,v_{x,t},v_{y,t},r_t],\qquad
\mathbf{u}_t=[\delta_{cmd,t},a_t].
```

- `x, y`: global/map-frame 위치 `[m]`
- `psi`: global yaw `[rad]`
- `vx, vy`: body-frame 종·횡속도 `[m/s]`
- `r`: yaw rate `[rad/s]`
- `delta_cmd`: `/drive`의 조향 명령 `[rad]`
- `a`: MPPI가 샘플링하는 가속도 `[m/s²]`

학습된 조향 scale/bias를 먼저 적용한다.

```math
\delta_t=\operatorname{clip}(S_a\delta_{cmd,t}+S_b,-0.55,0.55).
```

현재 residual checkpoint는 `kinematic_no_slip=true`, 즉 `beta=0` 조건으로
학습했다. Classic step은 CUDA와 동일하게 다음과 같다.

```math
\begin{aligned}
v^+ &= \operatorname{clip}(v_x+a\Delta t,v_{min},v_{max}),\\
r_c^+ &= \frac{v_x\tan\delta}{l_f+l_r},\\
x_c^+ &= x+v_x\cos\psi\,\Delta t,\\
y_c^+ &= y+v_x\sin\psi\,\Delta t,\\
\psi_c^+ &= \operatorname{wrap}(\psi+r_c^+\Delta t),\\
v_{x,c}^+ &= v^+,\qquad v_{y,c}^+=0.
\end{aligned}
```

출력은 다음 pose와 body state
`[x_c^+, y_c^+, psi_c^+, vx_c^+, vy_c^+, r_c^+]`이다. `l_f=0.163 m`,
`l_r=0.161 m`는 고정하고, classic 학습에서는 실질적으로 `S_a`, `S_b`를 식별한다.

### 2. MLP residual 입출력과 구조

MLP 입력은 21차원이다. 순서를 변경하면 학습 checkpoint와 CUDA 출력이 달라진다.

```math
\mathbf z_t=[
v_x,v_y,r,
r_{imu},a_{x,imu},a_{y,imu},
\delta_{cmd},v_{cmd},
v_{x,c}^+,v_{y,c}^+,r_c^+,
\mathbf u_{t-5},\mathbf u_{t-4},\mathbf u_{t-3},\mathbf u_{t-2},\mathbf u_{t-1}
]^T.
```

여기서 각 과거 입력은 `u=[delta_cmd, v_cmd]`이고,

```math
v_{cmd}=\operatorname{clip}(v_x+a\Delta t,v_{min},v_{max})
```

이다. 따라서 현재 command 2차원, 과거 5개 command 10차원을 사용한다. IMU는 pose
시각보다 미래가 아닌 가장 최신 sample을 causal zero-order hold로 선택하고,
`zbar_t=0.25 z_t+0.75 zbar_{t-1}` EMA를 적용한다.

네트워크는 다음과 같다.

```math
\begin{aligned}
\tilde{\mathbf z}_t&=(\mathbf z_t-\boldsymbol\mu)/\boldsymbol\sigma,\\
\mathbf h_1&=\operatorname{SiLU}(W_1\tilde{\mathbf z}_t+b_1), &&21\rightarrow64,\\
\mathbf h_2&=\operatorname{SiLU}(W_2\mathbf h_1+b_2), &&64\rightarrow32,\\
\Delta\mathbf a&=[8,8,30]\odot\tanh(W_3\mathbf h_2+b_3), &&32\rightarrow3.
\end{aligned}
```

MLP 출력은 다음 상태 자체가 아니라
`[Delta ax, Delta ay, Delta r_dot]` 잔차 가속도이다.

```math
[v_x^+,v_y^+,r^+]
=[v_{x,c}^+,v_{y,c}^+,r_c^+]+\Delta\mathbf a\Delta t.
```

보정된 `vx+, vy+, r+`로 `x+, y+, psi+`를 다시 적분한다. 학습 대상은 세 Linear
layer의 weight/bias 총 3,587개이며, residual 학습 중 classic 파라미터는 고정된다.

### 3. Bag에서 사용하는 토픽

학습에는 다음 토픽만 사용한다.

| 용도 | 우선 토픽/타입 | 사용하는 field |
|---|---|---|
| GT pose 및 body state | `/mocap_odom` 우선, 없으면 `/odom` (`nav_msgs/Odometry`) | position, quaternion yaw, `twist.linear.x/y`, `twist.angular.z` |
| 실제 차량 입력 | `/drive` (`ackermann_msgs/AckermannDriveStamped`) | `drive.steering_angle`, `drive.speed` |
| 관성 입력 | `/imu/data` 또는 bag에 맞춘 `--imu-topic` (`sensor_msgs/Imu`) | `angular_velocity.z`, `linear_acceleration.x/y` |

`/mpc_cmd`는 planner 출력이며 사용하지 않는다. 수동·자율 주행 모두 실제 차량으로 전달된
`/drive`만 사용한다. `/mcl_pose`나 `/mocap_pose2`처럼 pose만 있는 토픽은 body-frame
`vx, vy, r`가 없으므로 현재 수집기가 직접 사용하지 않는다. 필요하면 먼저 odometry 형태로
변환해야 한다.

토픽 간 정렬은 기본적으로 rosbag record timestamp를 사용한다. IMU는 pose 시각 이하의
가장 최신 값만 선택하며 최대 age는 `0.05 s`이다. 미래 IMU를 이용하는 linear interpolation은
실제 MPPI에서 재현할 수 있으므로 배포 학습에 사용하지 않는다.

dataset column은 다음과 같다.

```text
t,x,y,yaw,vx,vy,omega,steer,accel,speed_cmd,split,bag_id,imu_wz,imu_ax,imu_ay
```

`split=0/1`은 bag 단위 train/test 구분이다. 같은 bag의 인접 window가 train과 test에
동시에 들어가지 않는다. 후진 recovery가 발견되면 `정상 주행 → 충돌 → 후진 → 정상 복귀`
전체 구간을 제거하고 연속 segment를 새 `bag_id`로 분리한다.

### 4. 처음부터 학습하는 실행 순서

아래 명령은 `/mnt/nas_custom/F1tenth/paper/T_iv` 아래에서 `/drive`와 odometry가 모두
존재하는 bag을 검색하는 예다. `collect_all_bags.py`는 pose jump bag을 거부하고,
충돌/후진 recovery 구간을 제거하며, source bag을 8:2로 분리한다.

```bash
cd /home/a/smooth-mppi-cuda
source /opt/ros/humble/setup.bash
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate RL
pip install -r model_tuning/requirements.txt

python model_tuning/collect_all_bags.py \
  /mnt/nas_custom/F1tenth/paper/T_iv \
  -o model_tuning/data/tiv_mppi_80_20.npz \
  --drive-topic /drive --dt 0.02 --seed 42
```

GT trajectory를 먼저 확인한다.

```bash
python model_tuning/plot_extracted_bag_gt_trajectories.py \
  model_tuning/data/tiv_mppi_80_20.npz \
  -o model_tuning/tiv_mppi_gt_check
```

pose 시각에 causal IMU를 붙인다. 실제 bag의 IMU 토픽이 `/sensors/imu/raw`이면
`--imu-topic`만 바꾼다.

```bash
python model_tuning/augment_dataset_with_imu.py \
  model_tuning/data/tiv_mppi_80_20.npz \
  model_tuning/data/tiv_mppi_80_20.manifest.json \
  -o model_tuning/data/tiv_mppi_80_20_imu_causal.npz \
  --imu-topic /imu/data --alignment causal_hold --max-age 0.05
```

먼저 현재 MPPI kinematic 식으로 classic 조향 파라미터를 식별한다.

```bash
python model_tuning/train_attached_dbm.py \
  model_tuning/data/tiv_mppi_80_20_imu_causal.npz \
  -o model_tuning/deploy_mppi_classic \
  --rnn none --classic-mode kinematic --kinematic-no-slip \
  --mppi-rollout-model \
  --epochs 250 --batch-size 512 --lr 3e-3 \
  --focus-speed 3.0 4.5 --focus-weight 3 --hard-weight 3 \
  --terminal-weight 3.0 --speed-change-loss-weight 0.4 \
  --yaw-rate-loss-weight 0.2 --device cuda
```

`deploy_mppi_classic/metrics.json`의 `params.Sa`, `params.Sb`를 MPPI와 simulator YAML의
`kinematic_steer_scale`, `kinematic_steer_bias`에 동일하게 반영한다.

그다음 classic checkpoint를 고정하고 MLP residual을 최소 10분 학습한다.

```bash
python model_tuning/train_attached_dbm.py \
  model_tuning/data/tiv_mppi_80_20_imu_causal.npz \
  -o model_tuning/deploy_mppi_mlp \
  --classic-params model_tuning/deploy_mppi_classic/metrics.json \
  --rnn mlp --use-imu --improved-residual \
  --classic-mode kinematic --kinematic-no-slip --mppi-rollout-model \
  --history 50 --epochs 1000 --batch-size 512 --lr 3e-3 \
  --min-train-seconds 600 --patience 40 \
  --focus-speed 3.0 4.5 --focus-weight 3 --hard-weight 3 \
  --terminal-weight 3.0 --speed-change-loss-weight 0.4 \
  --yaw-rate-loss-weight 0.2 --device cuda
```

주요 출력은 `residual_state.pt`, `normalization.npz`, `metrics.json`,
`test_errors.npz`, `history.csv`이다. `metrics.json`의 `test_1s_*`는 학습에 사용하지 않은
held-out bag 결과이다.

### 5. CUDA MPPI와 simulator에 배포

PyTorch weight를 CUDA binary로 내보내고, 같은 normalization을 CUDA header와 simulator
Python 모델에 동시에 반영한다.

```bash
python model_tuning/export_mlp_cuda.py \
  model_tuning/deploy_mppi_mlp/residual_state.pt \
  config/kinematic_mlp_residual.bin \
  --normalization model_tuning/deploy_mppi_mlp/normalization.npz \
  --header include/cuda_mppi_controller/kinematic_mlp_weights.hpp \
  --simulator-model /home/a/f1tenth_gym_ros/src/f1tenth_gym/f1tenth_gym/envs/dynamic_models/kinematic_mlp.py
```

MPPI `config/params.yaml`은 다음을 선택해야 한다.

```yaml
dynamics_model: kinematic_mlp_residual
mlp_weights_path: /home/a/smooth-mppi-cuda/config/kinematic_mlp_residual.bin
kinematic_steer_scale: <metrics.json의 Sa>
kinematic_steer_bias: <metrics.json의 Sb>
kinematic_no_slip: true
```

Simulator `f1tenth_gym_ros/config/sim.yaml`은 다음처럼 같은 binary와 파라미터를 사용한다.

```yaml
dynamics_model: kinematic_mlp
mlp_weights_path: /home/a/smooth-mppi-cuda/config/kinematic_mlp_residual.bin
kinematic_steer_scale: <동일한 Sa>
kinematic_steer_bias: <동일한 Sb>
kinematic_no_slip: true
```

양쪽을 다시 빌드하고 동일 step 출력을 검증한다.

```bash
cd /home/a/smooth-mppi-cuda
source /opt/ros/humble/setup.bash
colcon build --packages-select smppi_cuda_controller --symlink-install

cd /home/a/f1tenth_gym_ros
colcon build --packages-select f1tenth_gym_ros --symlink-install

cd /home/a/smooth-mppi-cuda
python3 model_tuning/test_mppi_sim_step.py
```

`test_mppi_sim_step.py`가 `PASS`를 출력해야 한다. 이 검사는 같은 initial state, IMU,
command history와 action에 대해 CUDA MPPI와 simulator의
`x,y,yaw,vx,vy,yaw_rate` 단일-step 출력이 허용 오차 안에서 같은지 확인한다.

### Simulator IMU-free plant 선택

실차 IMU로 학습한 21-input MLP를 simulator plant로 재귀 실행하면, MLP 출력에서 만든
synthetic `ax/ay/yaw_rate`가 다음 MLP 입력으로 되먹임되어 진동할 수 있다. 현재 simulator는
이 feedback을 제거한 18-input 모델을 사용한다.

```yaml
dynamics_model: kinematic_mlp
mlp_weights_path: /home/a/smooth-mppi-cuda/config/kinematic_mlp_no_imu.bin
```

입력은 `[vx,vy,odom_yaw_rate,steer_cmd,speed_cmd,classic_next(3),과거 command(10)]`이다.
여기서 `odom_yaw_rate`는 차량 상태이며, 제거된 값은 별도 IMU 채널
`[imu_yaw_rate,imu_ax,imu_ay]`이다. MPPI controller rollout은 센서-aware 21-input 모델을
계속 사용할 수 있고, simulator plant와 controller 내부 prediction model이 반드시 같은
checkpoint일 필요는 없다.
