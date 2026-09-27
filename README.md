# smppi_cuda_controller

F1TENTH 자율주행 레이싱을 위한 **CUDA 가속 SMPPI(Smooth Model Predictive Path Integral) 컨트롤러** (ROS 2 패키지).

GPU에서 수천~수만 개의 제어 시퀀스를 병렬 rollout 하고, 4-파라미터 Pacejka 동역학 모델 위에서 비용을 평가해
정보이론적 가중 평균으로 최적 조향·가속 명령을 산출한다. 노이즈를 제어 **변화율** 공간에서 샘플링한 뒤
2차 Butterworth 저역통과 필터를 통과시켜, 추가 스무딩 없이도 매끄러운 제어 입력을 얻는 것이 핵심이다.

| 항목 | 값 |
|------|----|
| 제어 주기 / 적분 간격 `dt` | 35 ms (≈ 28.6 Hz) |
| 예측 구간 `T` | 50 step (≈ 1.75 s) |
| 샘플 수 `K` | 10,000 (런치 파일 기본값) |
| 동역학 | Kinematic bicycle (`|v| < 0.5 m/s`) ↔ Pacejka 동역학 (`|v| ≥ 0.5 m/s`) |
| 대상 GPU | Jetson Orin Nano (sm_87), 데스크톱 Ampere/Ada (sm_86, sm_89) |

---

## 목차

1. [알고리즘](#1-알고리즘)
2. [차량 동역학 모델](#2-차량-동역학-모델)
3. [비용 함수](#3-비용-함수)
4. [CUDA 구현 구조](#4-cuda-구현-구조)
5. [ROS 2 노드 및 토픽](#5-ros-2-노드-및-토픽)
6. [파라미터](#6-파라미터)
7. [빌드 및 실행](#7-빌드-및-실행)
8. [파라미터 자동 탐색 (`optimizer.py`)](#8-파라미터-자동-탐색-optimizerpy)
9. [디렉터리 구조](#9-디렉터리-구조)
10. [알려진 제한사항](#10-알려진-제한사항)

---

## 1. 알고리즘

### 1.1 MPPI 업데이트

이전 주기 최적해를 한 칸 시프트한 평균 시퀀스 $\bar{u}_{0:T-1}$ 주위에서 $K$개의 샘플 궤적을 생성하고,
각 궤적의 누적 비용 $S_k$로부터 가중치를 계산한다.

$$
w_k = \frac{\exp\!\left(-\frac{1}{\lambda}(S_k - S_{\min})\right)}{\sum_{j=1}^{K}\exp\!\left(-\frac{1}{\lambda}(S_j - S_{\min})\right)},
\qquad
u^*_t = \sum_{k=1}^{K} w_k\, u_{k,t}
$$

- $S_{\min}$을 빼서 `exp` 언더플로를 방지한다 (log-sum-exp 안정화).
- $\lambda$(`lambda`)가 작을수록 최저 비용 샘플에 집중하고, 클수록 평균에 가까워진다.
- 비용이 NaN인 샘플은 $10^8$으로 치환해 도태시키고, 최저 비용이 $10^8$ 이상이면 비상 정지 명령 `(δ=0, a=-5)`을 출력한다.
- 출력 후 $u^*_{1:T-1}$을 다음 주기의 평균 시퀀스로 사용한다 (warm start).

### 1.2 Smooth 샘플링 (SMPPI)

일반 MPPI는 제어 입력 $u$ 자체에 백색잡음을 더해 입력이 채터링하기 쉽다.
본 구현은 **제어 변화율** $\dot{u}$ 공간에서 잡음을 샘플링하고 이를 적분한다.

$$
\epsilon_t \sim \mathcal{N}(0, \Sigma), \quad
\tilde{\epsilon}_t = \mathrm{BW}_2(\epsilon_t), \quad
u_{k,t} = u_{k,t-1} + \mathrm{clip}\!\left(\Delta\bar{u}_t + \tilde{\epsilon}_t\,dt,\; \pm\dot{u}_{\max}\,dt\right)
$$

- $\Sigma = \mathrm{diag}(\sigma_\delta^2, \sigma_a^2)$ — `noise_steer_std` [rad/s], `noise_accel_std` [m/s³]
- $\mathrm{BW}_2$: 차단 주파수 $f_c = 3$ Hz의 2차 Butterworth IIR 필터 (쌍선형 변환, 샘플러 내부에서 샘플별로 상태 유지)

$$
y_n = b_0 x_n + b_1 x_{n-1} + b_2 x_{n-2} - a_1 y_{n-1} - a_2 y_{n-2}
$$

- 변화율 한계 `max_steer_rate`, `max_accel_rate`로 클리핑 후, 절대값 한계(`max_steer`, `min/max_accel`)와 속도 한계(`min/max_speed`)를 적용한다.

---

## 2. 차량 동역학 모델

상태 $x = [p_x, p_y, \psi, v, \omega, \beta]$ (위치, 요각, 속도, 요레이트, 차체 슬립각), 입력 $u = [\delta, a]$.
전진 오일러 적분을 사용한다.

### 2.1 저속 — Kinematic Bicycle (`|v| < 0.5 m/s`)

동역학 모델의 $1/v$ 특이점을 피하기 위해 사용한다.

$$
\dot{p}_x = v\cos\psi,\quad \dot{p}_y = v\sin\psi,\quad \dot{\psi} = \frac{v\tan\delta}{l_f + l_r},\quad \dot{v} = a
$$

### 2.2 고속 — Pacejka 단일 트랙 동역학 (`|v| ≥ 0.5 m/s`)

타이어 슬립각:

$$
\alpha_f = \delta - \arctan\frac{v_y + l_f\omega}{v_x},\qquad
\alpha_r = -\arctan\frac{v_y - l_r\omega}{v_x}
$$

**4-파라미터 Magic Formula** (ForzaETH On-Track-SysID 규약):

$$
F_{y,i} = F_{z,i}\, D_i \sin\!\Big(C_i \arctan\big(B_i\alpha_i - E_i(B_i\alpha_i - \arctan B_i\alpha_i)\big)\Big)
$$

정하중은 노드가 기하학적으로 계산한다: $F_{zf} = mg\,l_r / l_{wb}$, $F_{zr} = mg\,l_f / l_{wb}$ ($l_{wb} = l_f + l_r$).
따라서 $D$는 **무차원 마찰계수**이다. ($E = 0$, $D_{new} = D_{old}/F_z$로 두면 과거 3-파라미터 모델과 정확히 일치한다.)

운동 방정식:

$$
\begin{aligned}
\dot{p}_x &= v\cos(\psi+\beta), & \dot{p}_y &= v\sin(\psi+\beta), & \dot{\psi} &= \omega \\
\dot{v} &= a\,(1 - C_{m0} v), &
\dot{\omega} &= \frac{l_f F_{y,f}\cos\delta - l_r F_{y,r}}{I_z}, &
\dot{\beta} &= \frac{F_{y,f} + F_{y,r}}{m v} - \omega
\end{aligned}
$$

비용 계산용 횡가속도: $a_y = (F_{y,f}\cos\delta + F_{y,r}) / m$.
$C_{m0}$(`Cm0`)는 고속에서 모터 역기전력에 의한 가속 감쇠를 근사한다.

---

## 3. 비용 함수

단계 비용 $q(x_t, u_t)$와 종단 보상, 실패 페널티로 구성된다. (`src/mppi_core.cu` — `compute_cost_cuda`, `rollout_kernel`)

### 3.1 단계 비용

| 항목 | 수식 | 파라미터 |
|------|------|----------|
| 경로 추종 | $q_{dist}\, d_{ref}^2$ | `q_dist` (현재 0 — 레이싱 라인은 진행 보상이 결정) |
| 진행 방향 속도 보상 | $-0.2\,q_v\, v\cos(\psi - \psi_{ref})$ | `q_v` |
| 제어 변화율 | $q_{du}(\Delta\delta^2 + \Delta a^2)$ | `q_du` |
| 조향량 | $q_{steer}\,\delta^2$ | `q_steer` |
| 경계 소프트 | $70\,(d_{safe} - d_{bnd})^2$, $d_{safe} = r_c + 0.35$ | `collision_radius` |
| 경계 하드 (softplus 배리어) | $q_{col}\log\!\big(1 + e^{-30\max(d_{bnd}-r_c,\,10^{-5})}\big)$, $d_{bnd} < 1.5\,r_c$ | `q_collision` |
| 횡가속도(그립) | $q_{lat}\,\max(0, |a_y| - a_{y,th})^2$ | `q_lat_g`, `lat_g_threshold` |
| 장애물 | $\sum_i q_{obs} / (d_i - r_{car})$, $d_i < 1.5$ m | `q_obs`, `car_radius` (현재 `num_obstacles = 0`) |

- $d_{bnd}$: 가장 가까운 센터라인 점의 법선 방향 횡오차 $e_y$와 좌·우 트랙 폭으로 계산한 경계까지 최단 거리 — $\min(w_L - e_y,\; w_R + e_y)$. 이전 인덱스 기준 30개 윈도우만 탐색하므로 스레드당 $O(1)$이다.
- 횡가속도 비용은 이차 증가형이다. 현재 타이어 파라미터의 이론적 최대 횡가속도 $(F_{zf}D_f + F_{zr}D_r)/m \approx 7.5$ m/s² 보다 낮은 임계값에서 페널티가 시작되도록 설정한다.

### 3.2 종단 보상 (마지막 step)

$$
-\,q_{progress}\cdot \Delta i \;-\; q_{escape}\cdot v_T^2
$$

- $\Delta i$: 예측 구간 동안 전진한 센터라인 인덱스 수 (랩 순환 보정, $[0, T+10]$으로 클리핑)
- $v_T^2$ 보상은 코너 탈출 속도를 높이는 궤적(Out-In-Out)을 선호하게 만든다.
- 즉, **목표 속도 프로파일을 추종하지 않고** `max_speed` 한도 안에서 진행 보상과 그립 비용의 균형으로 속도가 결정된다.

### 3.3 실패(fault) 처리

$|a_y| >$ `lat_g_fault_threshold` 이거나 $d_{bnd} < r_c$ 이면 해당 샘플을 즉시 종료하고

$$
S_k \mathrel{+}= 10000 - 50t \;-\; 5\,q_v\,\Delta i
$$

를 부과한다. 더 오래 버티고 더 멀리 간 실패 궤적의 가중치를 상대적으로 높여, 충돌이 불가피한 상황에서도 기울기 정보가 남도록 설계했다.
남은 구간은 감속 제어 `(0.1·δ, -2.0)`로 채운다.

---

## 4. CUDA 구현 구조

```
MPPISolver::solve()
 ├─ H→D  이전 최적 시퀀스 (T × Control)
 ├─ CPU  현재 위치의 최근접 센터라인 인덱스 탐색
 ├─ rollout_kernel<<<ceil(K/128), 128>>>      ← 1 thread = 1 sample, T step 순차 적분
 │    ├─ curand_normal → Butterworth 필터 → 변화율 적분/클리핑
 │    ├─ update_dynamics (kinematic / Pacejka)
 │    ├─ compute_min_boundary_distance (윈도우 탐색)
 │    └─ compute_cost_cuda + 종단 보상 → costs[k]
 ├─ D→H  costs (K), controls (K×T), states (K×T, visualize_candidates 일 때만)
 └─ CPU  compute_optimal_control: 가중 평균, warm-start 시프트, 최적 궤적 재적분
```

- `State`/`Control` 구조체는 `alignas(16)`/`alignas(8)`로 정렬해 메모리 트랜잭션을 최소화한다.
- `--use_fast_math`, `__sinf/__cosf/__expf` 내장 함수를 사용한다.
- cuRAND 상태는 `K × T`개를 생성자에서 한 번 초기화한다 (seed = 1234).
- 경로/경계 버퍼는 최대 1000점까지 디바이스에 상주한다.
- `visualize_candidates: false`로 두면 `K × T × 32 B` 상태 복사(10,000 샘플 기준 약 16 MB/주기)를 생략해 Jetson에서 지연을 크게 줄일 수 있다.

---

## 5. ROS 2 노드 및 토픽

### 5.1 `smppi_node` (노드 이름 `smppi_controller`)

**입력 — 상태 추정 모드 (`use_mcl_pose`)**

| 모드 | 토픽 | 타입 | 사용 필드 |
|------|------|------|-----------|
| `false` (시뮬레이터) | `odom_topic` (기본 `/odom0`) | `nav_msgs/Odometry` | pose + twist |
| `true` (실차) | `pose_topic` (런치: `/newmcl_pose`) | `geometry_msgs/PoseStamped` | x, y, yaw |
| | `velocity_topic` (기본 `/odom`) | `nav_msgs/Odometry` | vx, vy, ω |

> 실차 휠 오도메트리는 횡미끄러짐을 측정하지 못하므로 `vy = 0`, 즉 초기 슬립각 $\beta_0 = 0$으로 시작한다.

**입력 — 경로**

| 토픽 | 타입 | QoS |
|------|------|-----|
| `path_topic` (기본 `/mppi_target_path`) | `nav_msgs/Path` | depth 1 |
| `/mppi_left_boundary` | `nav_msgs/Path` | reliable, transient_local |
| `/mppi_right_boundary` | `nav_msgs/Path` | reliable, transient_local |

경로와 경계는 **최초 1회만** 수신해 GPU에 업로드한다.

**출력**

| 토픽 | 타입 | 내용 |
|------|------|------|
| `drive_topic` (기본 `/drive`) | `ackermann_msgs/AckermannDriveStamped` | `steering_angle`, `speed = v + a·dt` (min/max_speed 클리핑), `acceleration` |
| `/mppi_viz` | `visualization_msgs/MarkerArray` | 비용 상위 50개 샘플 + 최적 궤적 (속도에 따라 파랑→노랑→빨강) |
| `/mppi_optimal_trajectory` | `smppi_cuda_controller/MppiTrajectory` | 최적 제어 시퀀스와 비용 항목 분해 |

`/mppi_viz`, `/mppi_optimal_trajectory`는 `visualize_candidates: true`일 때만 발행된다.

### 5.2 `path_publisher`

센터라인 CSV를 읽어 레퍼런스 경로와 좌·우 경계를 1 Hz(런치 기본값)로 발행한다.

- 헤더 이름으로 열을 찾는다 (대소문자 무시):
  - 필수: `x_m`/`x`/`x_map`, `y_m`/`y`/`y_map`
  - 선택: `psi_rad`/`psi`/`yaw`/`heading_rad` (없으면 인접점 차분으로 계산), `w_tr_left_m`/`w_left_m`/`left_width_m`, `w_tr_right_m`/`w_right_m`/`right_width_m`
- 경계점 = 센터라인 ± 법선벡터 × 트랙 폭
- 헤딩은 언랩(unwrap)해 연속 각도로 저장한다.

| 파라미터 | 기본값 |
|----------|--------|
| `csv_file_path` | 런치 파일이 `data/<map_name>/<map_name>_centerline.csv`로 지정 |
| `frame_id` | `map` |
| `publish_rate` | 10.0 Hz (런치: 1.0) |

### 5.3 `MppiTrajectory.msg`

| 필드 | 타입 | 설명 |
|------|------|------|
| `header` | `std_msgs/Header` | frame `map` |
| `steer[]`, `accel[]` | `float32[]` | 최적 제어 시퀀스 (길이 T) |
| `dist_cost` | `float32` | 경로 이탈 비용 |
| `vel_cost` | `float32` | (현재 미기록) |
| `steer_rate_cost`, `accel_rate_cost` | `float32` | 제어 변화율 비용 |
| `steer_cost` | `float32` | 조향량 비용 |
| `slip_cost` | `float32` | 횡가속도 비용 |
| `boundary_cost` | `float32` | 경계 비용 |
| `yaw`, `ref_yaw` | `float32` | 최적 궤적 t=1 시점 헤딩 / 최근접 센터라인 헤딩 |

---

## 6. 파라미터

`config/params.yaml` 기준 값이다. (괄호는 yaml에 없을 때 노드 내부 기본값)

### 6.1 한계값

| 파라미터 | 값 | 단위 |
|----------|----|------|
| `max_steer` | 0.4788 | rad |
| `min_accel` / `max_accel` | -8.0 / 8.5 | m/s² |
| `min_speed` / `max_speed` | 0.5 / 4.0 | m/s |
| `max_steer_rate` | 8.0 | rad/s |
| `max_accel_rate` | 100.0 | m/s³ |

### 6.2 비용 가중치

| 파라미터 | 값 | 파라미터 | 값 |
|----------|----|----------|----|
| `q_v` | 1.0 | `q_collision` | 300.0 |
| `q_dist` | 0.0 | `collision_radius` | 0.3 m |
| `q_du` | 0.2 | `q_lat_g` | 500.0 |
| `q_steer` | 0.5 | `lat_g_threshold` | 5.5 m/s² |
| `q_progress` | 36.0 | `lat_g_fault_threshold` | 9.0 m/s² |
| `q_escape_vel` | 32.0 | `q_obs` / `car_radius` | 150.0 / 0.25 m |

### 6.3 샘플링

| 파라미터 | 값 | 설명 |
|----------|----|------|
| `num_samples` | 10000 (런치에서 지정, 노드 기본 8000) | 샘플 수 K |
| `lambda` | 15.0 | 온도 파라미터 |
| `noise_steer_std` | 0.4 rad/s | 조향 변화율 잡음 |
| `noise_accel_std` | 2.0 m/s³ | 가속 변화율 잡음 |
| `visualize_candidates` | true | 후보 궤적 D→H 복사 및 시각화 |

### 6.4 차량 모델 (F1TENTH, SysID 결과)

| 파라미터 | 값 | 파라미터 | 전륜 (f) | 후륜 (r) |
|----------|----|----------|----------|----------|
| `mass` | 3.74 kg | `B` | 6.0926 | 6.6457 |
| `l_f` / `l_r` | 0.163 / 0.161 m | `C` | 1.2447 | 2.2129 |
| `I_z` | 0.04712 kg·m² | `D` | 0.7955 | 0.7317 |
| `Cm0` | 0.04 | `E` | 0.7815 | 0.0597 |

> ⚠️ `D`는 무차원 마찰계수이다. 과거 3-파라미터(뉴턴 단위 `D`) 설정값을 그대로 넣으면 횡력이 $F_z$배(약 18배) 커진다.
> 새로 식별한 값을 적용할 때는 저속에서 단계적으로 검증할 것.

### 6.5 토픽

| 파라미터 | yaml | 시뮬 오버라이드 | 실차 오버라이드 |
|----------|------|----------------|----------------|
| `use_mcl_pose` | false | false | true |
| `odom_topic` | `/odom0` | `/odom0` | – |
| `pose_topic` | `/mcl_pose` | – | `/newmcl_pose` |
| `velocity_topic` | `/odom` | – | `/odom` |
| `drive_topic` | `/drive` | `/drive` | `/drive` |
| `path_topic` | `/mppi_target_path` | – | – |

---

## 7. 빌드 및 실행

### 7.1 의존성

- ROS 2 (`ament_cmake`, `rclcpp`, `rosidl_default_generators`)
- `ackermann_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `visualization_msgs`, `std_msgs`
- `f1_msgs` (워크스페이스 내 별도 패키지 필요)
- CUDA Toolkit (`nvcc`, cuRAND)

### 7.2 빌드

```bash
cd ~/capstone_ws
colcon build --packages-select smppi_cuda_controller
source install/setup.bash
```

- `CMAKE_BUILD_TYPE`은 `Release`로 고정되어 있다.
- 호스트가 `aarch64`이면 `sm_87`(Jetson Orin Nano), 그 외에는 `sm_86;sm_89`로 컴파일한다. 다른 GPU라면 `CMakeLists.txt`의 `CMAKE_CUDA_ARCHITECTURES`를 수정한다.

### 7.3 실행

```bash
ros2 launch smppi_cuda_controller cuda_mppi.launch.py
# 다른 파라미터 파일 사용
ros2 launch smppi_cuda_controller cuda_mppi.launch.py param_file:=/path/to/params.yaml
```

`launch/cuda_mppi.launch.py` 상단 변수로 모드를 전환한다.

```python
map_name = "map1"       # data/<map_name>/ 사용 (map1, icra2025)
is_simulation = True    # False: 실차 모드 (use_mcl_pose=True, /newmcl_pose 구독)
```

실행 중 로그에 10 주기마다 `MPPI: <solve 시간>ms | V: <속도>`가 출력된다. 35 ms 제어 주기를 넘지 않는지 확인할 것.

---

## 8. 파라미터 자동 탐색 (`optimizer.py`)

시뮬레이터 위에서 비용 가중치를 **그리드 서치**로 평가하는 ROS 2 Python 노드이다.

1. 조합마다 `ros2 run smppi_cuda_controller smppi_node --ros-args --params-file params.yaml -p ...`로 컨트롤러를 단독 실행 (탐색 변수만 덮어씀)
2. `/initialpose`로 차량을 리셋하고 `/odom0`, `/collision0`을 모니터링
3. **10랩 무충돌 완주** 시 `Finished`, 충돌 시 `Crashed`, 300 s 초과 시 `Timeout`
4. 결과를 `result/mppi_optimization_results.csv`에 누적 기록 (중단 후 재실행 시 이어서 진행)

```bash
# 시뮬레이터와 path_publisher 가 실행 중인 상태에서
python3 scripts/optimizer.py
```

탐색 대상: `q_v`, `q_dist`, `q_du`, `q_steer`, `q_lat_g`, `q_collision`, `q_progress`, `q_escape_vel`, `lat_g_threshold`, `lat_g_fault_threshold`.
`scripts/result_stale_2026-05/`는 3-파라미터 타이어 모델 시절의 과거 결과로, 현재 모델과는 직접 비교할 수 없다.

---

## 9. 디렉터리 구조

```
smooth-mppi-cuda/
├── CMakeLists.txt
├── package.xml
├── config/params.yaml                      # 기본 파라미터
├── include/cuda_mppi_controller/
│   └── cuda_mppi_core.hpp                  # State/Control/Params, MPPISolver 선언
├── src/
│   ├── mppi_core.cu                        # 동역학, 비용, rollout 커널, 가중 평균
│   ├── smppi_node.cpp                      # ROS 2 컨트롤러 노드
│   └── path_publisher.cpp                  # 센터라인 CSV → Path/경계 발행
├── msg/MppiTrajectory.msg
├── launch/cuda_mppi.launch.py
├── scripts/
│   ├── optimizer.py                        # 그리드 서치 튜너
│   └── result_stale_2026-05/               # 과거 탐색 결과
├── data/
│   ├── map1/        {_centerline.csv, _map.pgm, _map.yaml}
│   └── icra2025/    {_centerline.csv, _map.pgm, _map.yaml}
└── claude_log/bugfix_2026-06-25.md         # 버그 수정 기록
```

센터라인 CSV 예시 (`data/map1/map1_centerline.csv`):

```
x_m,y_m,w_tr_left_m,w_tr_right_m,w_total_m,left_x_m,left_y_m,right_x_m,right_y_m
```

---

## 10. 알려진 제한사항

- **경로 길이 1000점 제한**: 디바이스 버퍼가 1000점으로 고정되어 있어 그 이상은 잘린다.
- **경로 1회 수신**: 경로·경계를 처음 한 번만 받으므로, 레이싱 라인을 바꾸려면 노드를 재시작해야 한다.
- **`num_samples` 자료형**: 노드 내부에서 `int16_t`로 저장하므로 32767을 넘기면 오버플로가 발생한다.
- **디버그 비용 불일치**: `MppiTrajectory`의 `boundary_cost` 등은 호스트에서 별도 상수(`+0.4`, `150`, `-40`)로 재계산하므로 커널 비용(`+0.35`, `70`, `-30`)과 정확히 같지 않다. 경향 확인용으로만 사용할 것.
- **`claude_log/bugfix_2026-06-25.md`와 현재 코드 차이**: 로그의 일부 수정(하드 배리어 `fminf`, `int16_t` 오버플로 방지 등)은 이후 리팩터링 과정에서 현재 코드에 반영되어 있지 않다.
- 런치 파일의 `publish_debug_info` 파라미터는 현재 노드에서 사용하지 않는다.

---

## License

MIT — JangJunhyeok
