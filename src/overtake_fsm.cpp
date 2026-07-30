#include "cuda_mppi_controller/overtake_fsm.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace mppi {

OvertakeFsm::OvertakeFsm(const Config& cfg) : cfg_(cfg) {}

float OvertakeFsm::dist_to_opp(const State& ego, float opp_x, float opp_y) const {
    return std::hypot(ego.x - opp_x, ego.y - opp_y);
}

bool OvertakeFsm::passed_opponent(const State& ego, float opp_x, float opp_y) const {
    // 상대방 위치에서 ego 방향으로의 투영 거리로 추월 완료 판정
    float dx = opp_x - ego.x;
    float dy = opp_y - ego.y;
    // ego의 진행 방향으로 상대방이 뒤에 있으면 (음수 투영) 추월 완료
    float forward = dx * std::cos(ego.yaw) + dy * std::sin(ego.yaw);
    return forward < -cfg_.merge_dist;
}

void OvertakeFsm::generate_bypass_path(
    float lateral_offset_m,
    const std::vector<float>& ref_xs,
    const std::vector<float>& ref_ys,
    const std::vector<float>& ref_yaws,
    FsmCommand& cmd) const
{
    cmd.bypass_xs.clear();
    cmd.bypass_ys.clear();
    cmd.bypass_yaws.clear();

    for (size_t i = 0; i < ref_xs.size(); ++i) {
        float yaw = ref_yaws[i];
        // 법선 방향 (좌측이 양수)
        float nx = -std::sin(yaw);
        float ny =  std::cos(yaw);
        cmd.bypass_xs.push_back(ref_xs[i] + lateral_offset_m * nx);
        cmd.bypass_ys.push_back(ref_ys[i] + lateral_offset_m * ny);
        cmd.bypass_yaws.push_back(yaw);
    }
}

FsmCommand OvertakeFsm::tick(
    const State& ego,
    bool  opp_detected,
    float opp_x, float opp_y, float opp_v,
    float h_pl, float h_pr,
    const std::vector<float>& ref_xs,
    const std::vector<float>& ref_ys,
    const std::vector<float>& ref_yaws)
{
    float dist = opp_detected ? dist_to_opp(ego, opp_x, opp_y)
                               : std::numeric_limits<float>::max();
    bool gaining = opp_detected && (ego.v - opp_v) > 0.3f;
    auto now = std::chrono::steady_clock::now();

    // ── 전이 로직 ─────────────────────────────────────────────────────

    // EMERGENCY: 거리 임계치 이내 (어느 상태에서든 우선 적용)
    if (opp_detected && dist < cfg_.emergency_dist) {
        state_ = FsmState::EMERGENCY;
        reset_side_selection();
    }
    else {
        switch (state_) {
        case FsmState::EMERGENCY:
            // 충분히 멀어지면 FOLLOW 또는 SOLO로 복귀
            if (!opp_detected || dist > cfg_.follow_dist)
                state_ = FsmState::SOLO;
            else
                state_ = FsmState::FOLLOW;
            break;

        case FsmState::SOLO:
            if (opp_detected && dist < cfg_.follow_dist && gaining)
                state_ = FsmState::FOLLOW;
            break;

        case FsmState::FOLLOW:
            if (!opp_detected || dist > cfg_.clear_dist) {
                state_ = FsmState::SOLO;
            } else if (gaining && dist < cfg_.prep_dist) {
                state_ = FsmState::OVERTAKE_PREP;
                prep_entry_time_ = now;
                last_opp_x_ = opp_x;
                last_opp_y_ = opp_y;
                reset_side_selection();
            }
            break;

        case FsmState::OVERTAKE_PREP: {
            double elapsed = std::chrono::duration<double>(now - prep_entry_time_).count();
            if (!opp_detected || dist > cfg_.clear_dist) {
                state_ = FsmState::SOLO;
                reset_side_selection();
            } else if (elapsed > cfg_.prep_timeout_s) {
                // PREP 타임아웃 → FOLLOW로 후퇴
                state_ = FsmState::FOLLOW;
                reset_side_selection();
            } else {
                // 선호 방향 선택: 예측 여유폭(상대방 위치+방향 반영)이 큰 쪽.
                // 이미 선택된 방향은 반대쪽이 margin 이상 좋아질 때만 교체.
                bool wide_left = has_preferred_side_
                    ? (preferred_left_ ? h_pl + cfg_.side_switch_margin >= h_pr
                                       : h_pl - cfg_.side_switch_margin >= h_pr)
                    : (h_pl >= h_pr);
                if (has_preferred_side_ && wide_left != preferred_left_)
                    side_confirm_count_ = 0;
                preferred_left_     = wide_left;
                has_preferred_side_ = true;

                // 선택 방향의 여유가 연속 유지되어야 추월 확정
                float h_sel = preferred_left_ ? h_pl : h_pr;
                if (h_sel >= cfg_.clear_threshold) side_confirm_count_++;
                else                               side_confirm_count_ = 0;

                if (side_confirm_count_ >= cfg_.side_confirm_ticks) {
                    state_ = preferred_left_ ? FsmState::OVERTAKE_LEFT
                                             : FsmState::OVERTAKE_RIGHT;
                    overtook_left_ = preferred_left_;
                    last_opp_x_ = opp_x;
                    last_opp_y_ = opp_y;
                    side_confirm_count_ = 0;
                }
            }
            break;
        }

        case FsmState::OVERTAKE_LEFT:
        case FsmState::OVERTAKE_RIGHT:
            if (passed_opponent(ego, last_opp_x_, last_opp_y_)) {
                state_ = FsmState::MERGE;
                reset_side_selection();
            } else if (opp_detected) {
                // 선택한 쪽 여유가 무너지면 추월 중단 → FOLLOW 복귀 후 재시도
                float h_sel = overtook_left_ ? h_pl : h_pr;
                if (h_sel < cfg_.clear_threshold * cfg_.abort_clear_factor) {
                    state_ = FsmState::FOLLOW;
                    reset_side_selection();
                }
            }
            break;

        case FsmState::MERGE:
            // 바이패스 경로 없이 센터라인으로 이미 복귀하는 상태
            // 상대방이 충분히 뒤처졌으면 SOLO 복귀
            if (!opp_detected || dist > cfg_.clear_dist) {
                state_ = FsmState::SOLO;
            }
            break;
        }
    }

    // ── 명령 생성 ─────────────────────────────────────────────────────
    FsmCommand cmd;
    cmd.state = state_;

    switch (state_) {
    case FsmState::SOLO:
        cmd.speed_cap = cfg_.solo_speed;
        break;

    case FsmState::FOLLOW:
        cmd.speed_cap = std::min(cfg_.follow_speed, opp_v + 0.5f);
        break;

    case FsmState::OVERTAKE_PREP:
        cmd.speed_cap = cfg_.follow_speed;
        break;

    case FsmState::OVERTAKE_LEFT:
        cmd.speed_cap = cfg_.overtake_speed;
        generate_bypass_path(cfg_.lateral_offset, ref_xs, ref_ys, ref_yaws, cmd);
        break;

    case FsmState::OVERTAKE_RIGHT:
        cmd.speed_cap = cfg_.overtake_speed;
        generate_bypass_path(-cfg_.lateral_offset, ref_xs, ref_ys, ref_yaws, cmd);
        break;

    case FsmState::MERGE:
        cmd.speed_cap = cfg_.solo_speed;
        // 바이패스 없음 → 센터라인으로 자동 복귀
        break;

    case FsmState::EMERGENCY:
        cmd.speed_cap = cfg_.emergency_speed;
        break;
    }

    if (opp_detected) {
        last_opp_x_ = opp_x;
        last_opp_y_ = opp_y;
    }

    return cmd;
}

} // namespace mppi
