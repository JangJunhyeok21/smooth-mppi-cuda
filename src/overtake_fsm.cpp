#include "cuda_mppi_controller/overtake_fsm.hpp"
#include <cmath>
#include <algorithm>

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
    const std::vector<float>& ref_vs,
    FsmCommand& cmd) const
{
    cmd.bypass_xs.clear();
    cmd.bypass_ys.clear();
    cmd.bypass_yaws.clear();
    cmd.bypass_vs.clear();

    for (size_t i = 0; i < ref_xs.size(); ++i) {
        float yaw = ref_yaws[i];
        // 법선 방향 (좌측이 양수)
        float nx = -std::sin(yaw);
        float ny =  std::cos(yaw);
        cmd.bypass_xs.push_back(ref_xs[i] + lateral_offset_m * nx);
        cmd.bypass_ys.push_back(ref_ys[i] + lateral_offset_m * ny);
        cmd.bypass_yaws.push_back(yaw);
        cmd.bypass_vs.push_back(ref_vs.empty() ? cfg_.overtake_speed : ref_vs[i]);
    }
}

FsmCommand OvertakeFsm::tick(
    const State& ego,
    bool  opp_detected,
    float opp_x, float opp_y, float opp_v,
    float h_pl, float h_pr,
    const std::vector<float>& ref_xs,
    const std::vector<float>& ref_ys,
    const std::vector<float>& ref_yaws,
    const std::vector<float>& ref_vs)
{
    float dist = opp_detected ? dist_to_opp(ego, opp_x, opp_y)
                               : std::numeric_limits<float>::max();
    bool gaining = opp_detected && (ego.v - opp_v) > 0.3f;
    auto now = std::chrono::steady_clock::now();

    // ── 전이 로직 ─────────────────────────────────────────────────────

    // EMERGENCY: 거리 임계치 이내 (어느 상태에서든 우선 적용)
    if (opp_detected && dist < cfg_.emergency_dist) {
        state_ = FsmState::EMERGENCY;
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
            }
            break;

        case FsmState::OVERTAKE_PREP: {
            double elapsed = std::chrono::duration<double>(now - prep_entry_time_).count();
            if (!opp_detected || dist > cfg_.clear_dist) {
                state_ = FsmState::SOLO;
            } else if (elapsed > cfg_.prep_timeout_s) {
                // PREP 타임아웃 → FOLLOW로 후퇴
                state_ = FsmState::FOLLOW;
            } else if (h_pl >= cfg_.clear_threshold) {
                state_ = FsmState::OVERTAKE_LEFT;
                overtook_left_ = true;
                last_opp_x_ = opp_x;
                last_opp_y_ = opp_y;
            } else if (h_pr >= cfg_.clear_threshold) {
                state_ = FsmState::OVERTAKE_RIGHT;
                overtook_left_ = false;
                last_opp_x_ = opp_x;
                last_opp_y_ = opp_y;
            }
            break;
        }

        case FsmState::OVERTAKE_LEFT:
        case FsmState::OVERTAKE_RIGHT:
            if (passed_opponent(ego, last_opp_x_, last_opp_y_)) {
                state_ = FsmState::MERGE;
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
    cmd.state              = state_;
    cmd.modal_steer_offset = cfg_.modal_offset;
    cmd.modal_ratio        = cfg_.modal_ratio;

    switch (state_) {
    case FsmState::SOLO:
        cmd.target_speed        = cfg_.solo_speed;
        cmd.multimodal_enabled  = false;
        break;

    case FsmState::FOLLOW:
        cmd.target_speed        = std::min(cfg_.follow_speed, opp_v + 0.5f);
        cmd.multimodal_enabled  = false;
        break;

    case FsmState::OVERTAKE_PREP:
        cmd.target_speed        = cfg_.follow_speed;
        cmd.multimodal_enabled  = true;
        // PREP 구간에서 좌/우 탐색: 바이패스 경로 없이 멀티모달만 활성화
        break;

    case FsmState::OVERTAKE_LEFT:
        cmd.target_speed        = cfg_.overtake_speed;
        cmd.multimodal_enabled  = false;
        generate_bypass_path(cfg_.lateral_offset, ref_xs, ref_ys, ref_yaws, ref_vs, cmd);
        break;

    case FsmState::OVERTAKE_RIGHT:
        cmd.target_speed        = cfg_.overtake_speed;
        cmd.multimodal_enabled  = false;
        generate_bypass_path(-cfg_.lateral_offset, ref_xs, ref_ys, ref_yaws, ref_vs, cmd);
        break;

    case FsmState::MERGE:
        cmd.target_speed        = cfg_.solo_speed;
        cmd.multimodal_enabled  = false;
        // 바이패스 없음 → 센터라인으로 자동 복귀
        break;

    case FsmState::EMERGENCY:
        cmd.target_speed        = cfg_.emergency_speed;
        cmd.multimodal_enabled  = false;
        break;
    }

    if (opp_detected) {
        last_opp_x_ = opp_x;
        last_opp_y_ = opp_y;
    }

    return cmd;
}

} // namespace mppi
