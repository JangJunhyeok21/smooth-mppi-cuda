# SMPPI 제어기 상태별 알고리즘 정리

SMPPI 제어기는 `OvertakeFsm`(7-상태 유한 상태 기계)이 매 제어 주기마다 MPPI 솔버의
**① 참조 경로, ② 속도 상한, ③ 샘플 분포(멀티모달 여부·편향 비율)** 를 바꿔주는 구조다.
궤적 후보 생성 자체(잡음 → 버터워스 필터 → 동역학 롤아웃 → 충돌/경계/횡가속 비용)는
모든 상태에서 동일하며, FSM은 위 세 가지만 조정한다. 따라서 어떤 상태에서도
장애물 회피와 트랙 경계 유지는 MPPI 비용 수준에서 항상 작동한다.

관련 소스:

| 파일 | 역할 |
|---|---|
| `include/cuda_mppi_controller/overtake_fsm.hpp` | 상태 정의, `FsmCommand`, 설정값 기본치 |
| `src/overtake_fsm.cpp` | 상태 전이 로직 + 상태별 명령 생성 |
| `src/smppi_node_fsm.cpp` | ROS2 노드: 장애물 예측, 여유폭 계산, FSM ↔ 솔버 연결 |
| `src/mppi_core.cu` | CUDA 롤아웃 커널: 멀티모달 샘플링, IS 보정, 속도 강제, 모드-인식 가중합 |

---

## 1. 전체 제어 파이프라인 (35 ms 주기)

`smppi_node_fsm.cpp`의 `timer_callback()`(`smppi_node_fsm.cpp:364`)이 매 틱 아래 순서로 실행된다.

1. **장애물 처리** (`compute_bezier_obstacles`, `smppi_node_fsm.cpp:220`)
   - 동적 장애물: 베지에 곡선(p0–p3)으로 호라이즌(T=50, dt=0.035 s ≈ 1.75 s) 동안의 미래 위치 예측
   - 정적 장애물: `StaticObstacleBuffer`에 누적해 블라인드 코너에서도 비용에 유지
   - 이 비용은 **FSM 상태와 무관하게 항상** MPPI 비용에 포함된다.
2. **상대방 여유폭 계산** (`compute_opp_clearances`, `smppi_node_fsm.cpp:175`)
   - `/opponent_odom`의 신선도(0.5 s) 확인 후, 상대의 진행 방향·속도로
     `fsm_side_pred_time_`(0.6 s) 뒤 횡방향 위치를 예측:
     `e_y_pred = e_y_opp + opp_v·sin(opp_yaw − ref_yaw)·0.6` (트랙 폭으로 클램프)
   - 통과에 필요한 폭 `clearance_needed = collision_radius + 0.2 + 0.1`을 빼서
     좌측 여유 `h_pl = (l_dist − e_y_pred) − clearance_needed`,
     우측 여유 `h_pr = (r_dist + e_y_pred) − clearance_needed` 산출
   - 상대가 이동 중인 쪽의 여유는 줄고 반대쪽은 늘어난다 → 미리 반대쪽 선호
3. **FSM tick** (`overtake_fsm.cpp:47`) → 상태 전이 + `FsmCommand` 생성
4. **명령 적용** (`apply_fsm_command`, `smppi_node_fsm.cpp:313`)
   - `max_vel = target_speed`, `multimodal_enabled`, `modal_steer_offset`, `modal_ratio` 갱신
   - 바이패스 경로가 있으면 솔버 참조 경로를 교체, 없으면 센터라인 복귀
5. **MPPI solve** — 8000 샘플 × 50스텝 롤아웃 (Pacejka 타이어 동역학)
6. 속도 클램프 후 `/drive`(AckermannDriveStamped) 발행, `/fsm/state` 문자열 발행

---

## 2. 상태 전이도

```mermaid
stateDiagram-v2
    [*] --> SOLO

    SOLO --> FOLLOW : 감지 & dist < 5.0 m & 접근중
    FOLLOW --> SOLO : 미감지 or dist > 7.0 m
    FOLLOW --> OVERTAKE_PREP : 접근중 & dist < 3.5 m

    OVERTAKE_PREP --> SOLO : 미감지 or dist > 7.0 m
    OVERTAKE_PREP --> FOLLOW : 타임아웃 2.5 s
    OVERTAKE_PREP --> OVERTAKE_LEFT : h_pl ≥ 0.8 m 3틱 연속 (좌 선호)
    OVERTAKE_PREP --> OVERTAKE_RIGHT : h_pr ≥ 0.8 m 3틱 연속 (우 선호)

    OVERTAKE_LEFT --> MERGE : 상대를 2.0 m 앞지름
    OVERTAKE_RIGHT --> MERGE : 상대를 2.0 m 앞지름
    OVERTAKE_LEFT --> FOLLOW : h_pl < 0.4 m (중단)
    OVERTAKE_RIGHT --> FOLLOW : h_pr < 0.4 m (중단)

    MERGE --> SOLO : 미감지 or dist > 7.0 m

    note right of EMERGENCY
        모든 상태에서 dist < 0.5 m 이면
        즉시 EMERGENCY 진입 (최우선)
    end note
    EMERGENCY --> SOLO : 미감지 or dist > 5.0 m
    EMERGENCY --> FOLLOW : dist ≤ 5.0 m
```

- **접근중(gaining)** = `opp_detected && (ego.v − opp_v) > 0.3 m/s` (`overtake_fsm.cpp:59`)
  → 추월당하는 상황(상대가 더 빠름)에서는 FOLLOW/PREP으로 내려가지 않는다.
- **EMERGENCY**는 전이 로직 최상단에서 검사되어 어떤 상태이든 덮어쓴다 (`overtake_fsm.cpp:65`).
- **추월 완료 판정** `passed_opponent`: 상대 위치를 ego 진행 방향에 투영해
  `dx·cos(yaw) + dy·sin(yaw) < −merge_dist(2.0 m)` 이면 완료 (`overtake_fsm.cpp:13`).

---

## 3. 상태별 궤적 생성 요약

| 상태 | 참조 경로 | 멀티모달 | 목표 속도 (기본값) | 비고 |
|---|---|---|---|---|
| SOLO | 센터라인 | OFF | 6.0 m/s (`solo_speed`) | 일반 주행 |
| FOLLOW | 센터라인 | OFF | min(4.5, 상대속도 + 0.5) | 거리 유지 |
| OVERTAKE_PREP | 센터라인 | **ON** (선호쪽 75 % 편향) | 4.5 m/s (`follow_speed`) | 좌/우 동시 탐색 |
| OVERTAKE_LEFT | **바이패스 +0.5 m** | OFF | 6.5 m/s (`overtake_speed`) | 좌측 추월 |
| OVERTAKE_RIGHT | **바이패스 −0.5 m** | OFF | 6.5 m/s | 우측 추월 |
| MERGE | 센터라인 (복귀) | OFF | 6.0 m/s | 별도 복귀 궤적 없음 |
| EMERGENCY | 센터라인 | OFF | 0.0 m/s | 전 샘플 강제 감속 |

목표 속도는 CUDA 커널에서 `max_vel` 상한으로 강제된다:
`if (x.v > p.max_vel) u_clamped.accel = fminf(u_clamped.accel, -0.5f);` (`mppi_core.cu:369`)

---

## 4. 상태별 상세

### 4.1 SOLO — 단독 주행

- 센터라인을 참조로 한 일반 MPPI 주행. 멀티모달 OFF, `solo_speed`(6.0 m/s).
- **전이**: 상대가 `follow_dist`(5.0 m) 이내이고 ego가 접근 중(gaining)일 때만 FOLLOW로.

### 4.2 FOLLOW — 후방 추종

- 궤적 생성은 SOLO와 동일. 속도 상한만 `min(follow_speed, opp_v + 0.5)`로 묶어
  상대와의 거리를 유지한다 (`overtake_fsm.cpp:172`).
- **전이**:
  - 미감지 또는 `dist > clear_dist`(7.0 m) → SOLO
  - 접근 중 & `dist < prep_dist`(3.5 m) → OVERTAKE_PREP
    (진입 시각 기록, 상대 위치 저장, 사이드 선택 상태 리셋)

### 4.3 OVERTAKE_PREP — 추월 준비 (유일한 멀티모달 상태)

**샘플링** (`mppi_core.cu:312`): 8000개 샘플 중 앞쪽 `K × modal_ratio`개는 조향
잡음에 `+δ`(좌편향, `modal_steer_offset` = 0.15), 나머지는 `−δ`(우편향)를 더해
좌/우 두 갈래 궤적 다발을 동시에 탐색한다.

```cpp
// mppi_core.cu (rollout kernel)
if (p.multimodal_enabled) {
    float delta = p.modal_steer_offset;
    int split = static_cast<int>(K * p.modal_ratio);
    raw_steer_noise += (k < split) ? delta : -delta;
}
```

**Importance Sampling 보정** (`mppi_core.cu:336`): 편향된 제안 분포
`q = r·N(+δ,σ) + (1−r)·N(−δ,σ)` 에서 샘플링했으므로, 비용에
`−λ(log p(ε) − log q(ε))` 를 더해 MPPI 가중치를 이론적으로 보정한다
(log-sum-exp 안정화, r은 [0.05, 0.95]로 클램프).

**모드-인식 가중합** (`mppi_core.cu:585`): 최종 제어를 만들 때 최저 비용 샘플
`best_k_`가 속한 모드(좌/우 절반)만 softmax 가중합에 포함한다. 좌/우 샘플을
그냥 평균하면 서로 상쇄되어 상대 정면으로 향하는 mode collapse가 생기기 때문.

```cpp
// mppi_core.cu (compute_optimal_control)
int k_start = 0, k_end = K_;
if (params_.multimodal_enabled) {
    int split = static_cast<int>(K_ * params_.modal_ratio);
    bool best_is_left = (best_k_ < split);
    k_start = best_is_left ? 0 : split;
    k_end   = best_is_left ? split : K_;
}
```

**방향 결정 로직** (`overtake_fsm.cpp:106`):

1. 예측 여유폭이 큰 쪽을 선호 (`h_pl` vs `h_pr`).
   이미 선택된 방향은 반대쪽이 `side_switch_margin`(0.2 m) 이상 좋아질 때만
   교체(히스테리시스). 방향이 바뀌면 확인 카운터 리셋.
2. 선호 방향의 샘플 비율을 `prep_modal_ratio`(0.75)로 편향:
   좌 선호 → `modal_ratio = 0.75`, 우 선호 → `0.25`.
3. 선호 방향 여유폭이 `clear_threshold`(0.8 m) 이상을 **`side_confirm_ticks`(3틱)
   연속** 유지하면 OVERTAKE_LEFT/RIGHT 확정.

**전이**:
- 미감지 또는 `dist > clear_dist` → SOLO
- 체류 시간 > `prep_timeout_s`(2.5 s) → FOLLOW (후퇴 후 재시도)
- 방향 확정 → OVERTAKE_LEFT 또는 OVERTAKE_RIGHT

### 4.4 OVERTAKE_LEFT / OVERTAKE_RIGHT — 추월 실행

- 멀티모달 OFF. 대신 **바이패스 경로**를 참조 경로로 교체한다
  (`generate_bypass_path`, `overtake_fsm.cpp:22`): 센터라인의 각 점을 법선
  `(nx, ny) = (−sin yaw, cos yaw)` 방향으로 `±lateral_offset`(0.5 m) 평행이동
  (좌측이 +). yaw와 속도 프로파일은 센터라인 값을 그대로 사용.
- MPPI는 이 바이패스 경로를 추종하며 `overtake_speed`(6.5 m/s)로 가속.
  상대 차량 회피 자체는 여전히 MPPI의 장애물 비용이 담당한다.
- **전이**:
  - `passed_opponent()` — 상대가 ego 진행 방향 기준 `merge_dist`(2.0 m) 뒤 → MERGE
  - 선택한 쪽 여유폭 < `clear_threshold × abort_clear_factor` = 0.8 × 0.5 = **0.4 m**
    → 추월 **중단**, FOLLOW 복귀 후 재시도 (`overtake_fsm.cpp:140`)

### 4.5 MERGE — 센터라인 복귀

- 별도의 복귀 궤적을 생성하지 않는다. `FsmCommand`의 바이패스가 비어 있으므로
  `apply_fsm_command`가 참조 경로를 센터라인으로 되돌리고, MPPI가 자연스럽게
  복귀 궤적을 최적화한다. 속도는 `solo_speed`.
- **전이**: 미감지 또는 `dist > clear_dist`(7.0 m) → SOLO

### 4.6 EMERGENCY — 긴급 감속

- 상대와 `emergency_dist`(0.5 m) 이내면 **어느 상태에서든 최우선 진입**.
- `target_speed = 0` → `max_vel = 0`이 되어 커널에서 전 샘플이 강제 감속된다.
  조향은 여전히 MPPI가 최적화하므로 감속하면서 회피 조향은 가능하다.
- **전이**: 미감지 또는 `dist > follow_dist`(5.0 m) → SOLO, 그 외 → FOLLOW

---

## 5. 안전 관련 부가 메커니즘

- **Fault 패널티 + 진행 보상** (`mppi_core.cu:391`): 충돌(경계까지 거리 <
  `collision_radius`)이나 횡가속 한계(|ay| > 12.74 m/s² = 1.3 g) 초과 샘플은
  `total_cost += 10000 − t·50`. 늦게 부딪힐수록, 더 전진할수록 패널티가 줄어
  최악 상황에서도 풀브레이킹+조향으로 최대한 버티는 궤적이 선택된다.
- **비상 정지 fallback**: 최저 비용이 inf/1e8 이상이면 `{steer 0, accel −5.0}` 출력.
- **SMPPI 스무딩**: 조향/가속 잡음을 2차 버터워스 IIR 필터로 걸러
  액션 미분(derivative) 수준에서 부드러운 제어 시퀀스를 보장.

---

## 6. 파라미터 요약

`OvertakeFsm::Config` 기본값 (`overtake_fsm.hpp:51`). ROS 파라미터
(`fsm_*` 접두사, `smppi_node_fsm.cpp:637` 부근)로 오버라이드 가능.

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `follow_dist` | 5.0 m | FOLLOW 진입 거리 |
| `prep_dist` | 3.5 m | OVERTAKE_PREP 진입 거리 |
| `clear_dist` | 7.0 m | SOLO 복귀 거리 |
| `merge_dist` | 2.0 m | 추월 완료 판정 (상대 후방 투영 거리) |
| `prep_timeout_s` | 2.5 s | PREP 최대 체류 시간 |
| `emergency_dist` | 0.5 m | EMERGENCY 트리거 거리 |
| `clear_threshold` | 0.8 m | 추월 가능 판정 여유폭 |
| `abort_clear_factor` | 0.5 | 추월 중단 임계 = clear_threshold × factor |
| `lateral_offset` | 0.5 m | 바이패스 경로 횡방향 오프셋 |
| `modal_offset` | 0.15 | 멀티모달 조향 잡음 편향 δ |
| `prep_modal_ratio` | 0.75 | PREP 중 선호 방향 샘플 비율 |
| `side_confirm_ticks` | 3 | 방향 확정 전 연속 확인 틱 수 |
| `side_switch_margin` | 0.2 m | 선호 방향 교체 히스테리시스 |
| `solo_speed` | 6.0 m/s | SOLO/MERGE 목표 속도 |
| `follow_speed` | 4.5 m/s | FOLLOW/PREP 목표 속도 |
| `overtake_speed` | 6.5 m/s | OVERTAKE 목표 속도 |
| `emergency_speed` | 0.0 m/s | EMERGENCY 목표 속도 |
| `fsm_side_pred_time` (노드) | 0.6 s | 여유폭 계산 시 상대 횡위치 예측 시간 |
