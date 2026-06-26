# smppi_cuda_controller

F1TENTH 자율주행을 위한 CUDA 가속 MPPI(Model Predictive Path Integral) 컨트롤러.  
NVIDIA GPU에서 대규모 병렬 샘플링으로 실시간 궤적을 최적화하고 조향/가속 명령을 출력한다.

---

## 알고리즘 개요

MPPI는 K개의 랜덤 제어 시퀀스를 GPU에서 병렬 rollout하고, 각 궤적의 비용을 기반으로  
최적 제어 입력을 가중 평균으로 산출하는 샘플링 기반 MPC이다.

```
u* = Σ w_k · ε_k      where   w_k ∝ exp(−J_k / λ)
```

### 차량 동역학 모델 (이중 모델)

| 속도 구간 | 모델 |
|-----------|------|
| `v < 0.5 m/s` | 순운동학 모델 (Kinematic Bicycle) — 특이점 방지 |
| `v ≥ 0.5 m/s` | 파세이카 동역학 모델 (Pacejka Tire) — 슬립각, 횡력 반영 |

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
| `/ekf_odom` (기본값) | `nav_msgs/Odometry` | EKF 융합 상태 (pose + twist) |
| `/mppi_target_path` | `nav_msgs/Path` | 레퍼런스 경로 + 목표 속도 |
| `/mppi_left_boundary` | `nav_msgs/Path` | 좌측 주행 경계 |
| `/mppi_right_boundary` | `nav_msgs/Path` | 우측 주행 경계 |

**발행**

| 토픽 | 타입 | 설명 |
|------|------|------|
| `/ackermann_cmd` (`drive_topic`) | `ackermann_msgs/AckermannDriveStamped` | 조향 + 가속 명령 |
| `/mppi_viz` | `visualization_msgs/MarkerArray` | 샘플 궤적 시각화 |
| `/mppi_optimal_trajectory` | `smppi_cuda_controller/MppiTrajectory` | 최적 궤적 + 비용 분해 |

**제어 주기:** 35 ms (~28.5 Hz)

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
| `odom_topic` | `/ekf_odom` (실차) / `/odom0` (시뮬) |
| `drive_topic` | `/drive` (실차) / `/ackermann_cmd0` (시뮬) |
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

### `cuda_mppi.launch.py` 내부 플래그

```python
is_simulation = False   # True: 시뮬레이터 모드, False: 실차(Jetson) 모드
map_name = "map1"       # data/ 하위 맵 이름
```

---

## 맵 데이터

```
data/
├── icra2025/
│   ├── icra2025_centerline.csv
│   └── icra2025_map.{pgm,yaml}
└── map1/
    ├── map1_centerline.csv
    └── map1_map.{pgm,yaml}
```

CSV 포맷: `x [m], y [m], width_left [m], width_right [m]`

---

## 커스텀 메시지

### `MppiTrajectory.msg`

최적 궤적의 비용 항목을 시간 단계별로 분해하여 발행한다.  
(디버깅 및 파라미터 튜닝에 활용)

| 필드 | 설명 |
|------|------|
| `dist_cost` | 경로 이탈 비용 |
| `vel_cost` | 속도 초과 비용 |
| `steer_rate_cost` | 조향 변화율 비용 |
| `accel_rate_cost` | 가속 변화율 비용 |
| `steer_cost` | 조향각 크기 비용 |
| `slip_cost` | 횡가속도 비용 |
| `boundary_cost` | 경계 침범 비용 |
| `yaw` / `ref_yaw` | 현재/레퍼런스 헤딩 |

---

## 의존성

- `rclcpp`, `ackermann_msgs`, `geometry_msgs`, `nav_msgs`
- `sensor_msgs`, `visualization_msgs`
- CUDA Toolkit (nvcc)
- `rosidl_default_generators` (커스텀 메시지)
