# Real-car v2 model pipeline

이 디렉터리가 현재 real-car MPPI dynamics 모델의 단일 학습 경로다.
단계 번호는 실행 순서이며, 독립 시각화 코드는 항상 `visualize_`로 시작한다.

## 요약

Step 1: raw data 정렬 및 초기 KF 생성
Step 2: longitudinal actuator 식별
Step 3: classic model/Pacejka + I_z 식별
Step 4: steering actuator 식별
Step 5: 초기 파라미터와 튜닝 파라미터의 EKF 및 classic rollout A/B 평가
        └─ 파라미터가 채택되면 Step 1/KF 재생성
Step 7: 40 ms transition 생성
Step 8: residual MLP one-step 학습
Step 9: recursive fine-tuning
Step 10: rollout 평가
Step 11: 배포

별도 실험 경로: `train_e2e_kf_predictor.py`
  - classic/Pacejka residual이 아닌 causal neural transition model이다.
  - 실제 odom callback 시점을 anchor로 사용하고 Step-1의
    `kf_x, kf_y, kf_yaw, kf_vx, kf_vy, kf_yaw_rate`를 미래 GT로 사용한다.
  - 40 ms one-step state loss와 1.2 s recursive state/pose loss로 학습한다.
  - CUDA MPPI에는 자동 배포하지 않으며 독립 A/B 후보로 artifact를 저장한다.

## Data flow

```text
ROS 2 bag
  -> step_1_extract_data.py
  -> step_2_identify_longitudinal_actuator.py
  -> step_3_identify_classic_model.py
  -> step_4_identify_steering_actuator.py
  -> classic params.json
  -> step_5_evaluate_velocity_observer.py
     (Step-1 snapshot vs tuned EKF, classic closed/open loop)
  -> 채택 시 Step 1 재실행
  -> step_6_train_evaluate_deploy_residual.py
     (direct Step-1 data -> one-step -> recursive -> A/B evaluation -> deploy)
```

## Numbered steps

1. `step_1_extract_data.py`
   - `/newmcl_pose`, `/odom`, `/ackermann_cmd`, `/drive`, `/imu/data`를 bag에서 읽는다.
   - 모든 sensor stream은 20 ms grid에 causal-hold 정렬한다.
   - 2026-08-17 이전 IMU는 `(wz, ax, ay)=(-1,+1,-1)`, 이후는 `(+1,+1,+1)`이다.
   - collision, manual takeover, localization/physics 불연속을 제거하고 bag별 NPZ를 만든다.
   - 실제 `/odom` callback timestamp를 anchor로 보존한다. 각 anchor의 입력과
     `anchor + 20, 40, ..., 1200 ms` 상태 GT를 pose/velocity 연속 보간 함수에서 생성한다.
   - callback dataset의 yaw는 wrap하지 않은 연속값이다. 보간 horizon 안의 pose/yaw
     jump나 필터링 구간 경계 통과는 학습 anchor에서 제외한다.
   - NPZ callback 계약:
     `callback_inputs[N,13]`, `callback_future_states[N,60,6]`,
     `callback_future_commands[N,60,2]`, `callback_future_offsets_s[60]`.
   - `/drive` topic이 없으면 manual bag으로 판별하며 `/ackermann_cmd`를 학습 command로 사용한다.
     `/drive`가 있으면 autonomous bag으로 기록하지만 두 command topic의 차이는 충돌 판정에 쓰지 않는다.
   - 충돌은 큰 odom/command 속도에 비해 MCL 위치 변화가 지속적으로 없거나,
     impact/reverse recovery가 검출될 때 판정한다. 첫 이상 이후 bag suffix는 사용하지 않는다.

2. `step_2_identify_longitudinal_actuator.py`
   - `K_v`, `tau_acc`, `tau_brake`를 online MPPI-model EKF `vx` rollout으로 식별한다.

3. `step_3_identify_classic_model.py`
   - 동일한 split에서 8-parameter Pacejka 후보를 회귀하고 validation score로 선택한다.
   - `I_z`를 Pacejka와 교대로 회귀한 뒤 9개 파라미터를 joint polish한다.
   - held-out best/p95/worst에서 GT, 기존 classic, 회귀 classic의 trajectory,
     `vy`, yaw rate 1.0 s open-loop 성능을 시각화한다.
   - 출력: `model_tuning/results/dynamic_40ms_regression/params.json`.
   - 구현 helper는 `classic_model_regression.py`, 제한된 진단 대안은
     `diagnostic_regress_classic_model_4param.py`다.

4. `step_4_identify_steering_actuator.py`
   - current YAML의 classic 파라미터를 고정하고 steering scale/bias와 servo lag를 식별한다.
   - 다음 alternating round의 Step 3가 갱신된 steering 파라미터를 다시 사용한다.

5. `step_5_evaluate_velocity_observer.py`
   - NPZ에 저장된 Step-1 초기 파라미터 snapshot과 현재 YAML 튜닝값을 A/B 비교한다.
   - EKF measurement-corrected 상태, classic closed-loop one-step prediction,
     classic open-loop free rollout을 각각 평가하고 best/median/worst를 시각화한다.
   - 파라미터가 채택되면 Step 1을 다시 실행한다.

6. `step_6_train_evaluate_deploy_residual.py`
   - Step 1의 bag별 `callback_*`와 `kf_*`를 직접 읽는다.
   - 40 ms classic transition과 KF GT의 residual을 메모리에서 계산하며 별도
     residual dataset NPZ는 생성하지 않는다.
   - 20-64-32-3 residual derivative MLP를 one-step target으로 학습한다.
   - 5/10/20/30-step free-recursive rollout loss로 fine-tuning한다.
   - held-out validation/test에서 1.2 s trajectory, yaw, vx, vy, yaw-rate 오차를 JSON/NPZ로 저장한다.
   - binary/parameter contract를 검증한 후 선택 모델을 runtime config에 배포한다.
   - residual과 classic-only를 같은 split에서 평가한다.
   - 검증된 binary와 classic parameter를 MPPI runtime config에 배포한다.
   - `--no-deploy`를 주면 학습과 평가까지만 수행한다.

### Experimental E2E KF predictor

```bash
/home/a/anaconda3/envs/RL/bin/python \
  model_tuning/real_car_v2/train_e2e_kf_predictor.py
```

출력은 기본적으로 `model_tuning/results/e2e_kf_predictor/` 아래의
`model.pt`, `normalization.npz`, `contract.json`, `metrics.json`,
`rollouts.png`에 저장된다. 기존 `callback_future_states`는 raw 보간값이므로
이 학습기는 이를 GT로 사용하지 않고 NPZ의 online MPPI-model EKF 필드를 callback 시각에
다시 보간한다.

배포 중인 classic+residual binary와 동일 callback/KF GT 조건으로 비교하려면
다음을 실행한다.

```bash
/home/a/anaconda3/envs/RL/bin/python \
  model_tuning/real_car_v2/compare_e2e_kf_vs_residual.py
```

비교 결과는 `model_tuning/results/e2e_kf_vs_residual/metrics.json`과
`comparison.png`에 저장된다.

## Run

Step 1은 NAS bag을 읽으므로 필요할 때 명시적으로 실행한다.

```bash
/usr/bin/python3 model_tuning/real_car_v2/step_1_extract_data.py
/usr/bin/python3 model_tuning/real_car_v2/run_pipeline.py
```

`run_pipeline.py`는 식별 → Step 6 A/B 평가 → 채택 시 KF 재생성 → 학습/평가를 수행한다.

## Shared helpers

- `contract.py`: Python/CUDA 공통 dynamics 및 actuator 계약
- `classic_model_kalman_filter.py`: runtime-equivalent online MPPI-model EKF
- `helper_filter_collision_recovery.py`: collision/manual/physics 구간 필터
- `offline_lateral_velocity_smoother.py`: non-causal teacher 생성 전용

## Naming and archive policy

- `step_N_*.py`: 실제 학습/deploy pipeline
- `visualize_*.py`: figure를 생성하는 독립 분석
- `evaluate_*.py`, `regress_*.py`, `validate_*.py`: 정량 리포트를 생성해 보존한 실험
- `model_tuning/trash/`: 현재 pipeline에서 사용되지 않는 legacy 코드. 삭제하지 않고 보존한다.

새 코드를 추가할 때 현재 pipeline의 일부면 다음 step 번호를 사용하고, 그림 생성이 목적이면 반드시 `visualize_` prefix를 사용한다.
