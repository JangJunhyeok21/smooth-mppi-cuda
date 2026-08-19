# Real-car v2 model pipeline

이 디렉터리가 현재 real-car MPPI dynamics 모델의 단일 학습 경로다.
단계 번호는 실행 순서이며, 독립 시각화 코드는 항상 `visualize_`로 시작한다.

## Data flow

```text
ROS 2 bag
  -> step_1_extract_data.py
  -> per-bag 20 ms NPZ
  -> step_2_build_20ms_dataset.py
  -> dynamic_40ms_all_drive_source_20ms.npz
  -> step_3_regress_classic_model.py
  -> classic params.json
  -> step_4_build_40ms_dataset.py
  -> dynamic_40ms_residual.npz
  -> step_5_train_residual_mlp.py
  -> step_6_finetune_recursive.py
  -> step_7_evaluate_rollout.py
  -> step_8_deploy_to_mppi.py
```

## Numbered steps

1. `step_1_extract_data.py`
   - `/newmcl_pose`, `/odom`, `/ackermann_cmd`, `/drive`, `/imu/data`를 bag에서 읽는다.
   - 모든 sensor stream은 20 ms grid에 causal-hold 정렬한다.
   - 2026-08-17 이전 IMU는 `(wz, ax, ay)=(-1,+1,-1)`, 이후는 `(+1,+1,+1)`이다.
   - collision, manual takeover, localization/physics 불연속을 제거하고 bag별 NPZ를 만든다.

2. `step_2_build_20ms_dataset.py`
   - 통과한 bag NPZ를 session-disjoint train/validation/test split으로 합친다.
   - runtime과 같은 causal lateral-velocity KF를 실행한다.
   - offline smoother는 teacher target으로만 사용한다.
   - 출력: `model_tuning/data/dynamic_40ms_all_drive_source_20ms.npz`와 JSON manifest.

3. `step_3_regress_classic_model.py`
   - 동일한 split에서 8-parameter Pacejka 후보를 회귀하고 validation score로 선택한다.
   - 출력: `model_tuning/results/dynamic_40ms_regression/params.json`.
   - 제한된 진단 대안은 `step_3_regress_classic_model_4param.py`다.

4. `step_4_build_40ms_dataset.py`
   - 20 ms source 두 sample을 하나의 40 ms MPPI model knot로 만든다.
   - Orin NX solve latency 18--31 ms를 nominal 25 ms, 20 ms grid 한 칸으로 양자화한다.
   - 출력: `model_tuning/data/dynamic_40ms_residual.npz`.

5. `step_5_train_residual_mlp.py`
   - 20-64-32-3 residual derivative MLP를 one-step target으로 학습한다.

6. `step_6_finetune_recursive.py`
   - 5/10/20/30-step free-recursive rollout loss로 fine-tuning한다.

7. `step_7_evaluate_rollout.py`
   - held-out validation/test에서 1.2 s trajectory, yaw, vx, vy, yaw-rate 오차를 JSON/NPZ로 저장한다.

8. `step_8_deploy_to_mppi.py`
   - binary/parameter contract를 검증한 후 선택 모델을 runtime config에 배포한다.

## Run

Step 1은 NAS bag을 읽으므로 필요할 때 명시적으로 실행한다.

```bash
/usr/bin/python3 model_tuning/real_car_v2/step_1_extract_data.py
/usr/bin/python3 model_tuning/real_car_v2/run_pipeline.py
```

`run_pipeline.py`는 step 2--8을 순서대로 실행한다. 파일 상단 switch로 완료된 단계를 끌 수 있다.

## Shared helpers

- `contract.py`: Python/CUDA 공통 dynamics 및 actuator 계약
- `helper_lateral_velocity_kf.py`: runtime-equivalent causal KF
- `helper_filter_collision_recovery.py`: collision/manual/physics 구간 필터
- `offline_lateral_velocity_smoother.py`: non-causal teacher 생성 전용

## Naming and archive policy

- `step_N_*.py`: 실제 학습/deploy pipeline
- `visualize_*.py`: figure를 생성하는 독립 분석
- `evaluate_*.py`, `regress_*.py`, `validate_*.py`: 정량 리포트를 생성해 보존한 실험
- `model_tuning/trash/`: 현재 pipeline에서 사용되지 않는 legacy 코드. 삭제하지 않고 보존한다.

새 코드를 추가할 때 현재 pipeline의 일부면 다음 step 번호를 사용하고, 그림 생성이 목적이면 반드시 `visualize_` prefix를 사용한다.
