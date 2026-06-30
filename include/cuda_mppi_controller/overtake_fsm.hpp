#pragma once

#include <vector>
#include <string>
#include <chrono>
#include "cuda_mppi_controller/cuda_mppi_core.hpp"

namespace mppi {

enum class FsmState {
    SOLO,           // 단독 주행 (상대방 없음)
    FOLLOW,         // 후방 추종 (상대방 감지, 추월 준비 전)
    OVERTAKE_PREP,  // 추월 준비 (좌/우 여유 공간 평가 중, 멀티모달 ON)
    OVERTAKE_LEFT,  // 좌측 추월 실행
    OVERTAKE_RIGHT, // 우측 추월 실행
    MERGE,          // 추월 후 센터라인 복귀
    EMERGENCY,      // 긴급 감속 (충돌 임박)
};

inline std::string fsm_state_to_str(FsmState s) {
    switch (s) {
        case FsmState::SOLO:            return "SOLO";
        case FsmState::FOLLOW:          return "FOLLOW";
        case FsmState::OVERTAKE_PREP:   return "OVERTAKE_PREP";
        case FsmState::OVERTAKE_LEFT:   return "OVERTAKE_LEFT";
        case FsmState::OVERTAKE_RIGHT:  return "OVERTAKE_RIGHT";
        case FsmState::MERGE:           return "MERGE";
        case FsmState::EMERGENCY:       return "EMERGENCY";
        default:                        return "UNKNOWN";
    }
}

struct FsmCommand {
    FsmState state           {FsmState::SOLO};
    float    target_speed    {5.0f};
    bool     multimodal_enabled  {false};
    float    modal_steer_offset  {0.15f};
    float    modal_ratio         {0.5f};

    // 바이패스 경로 (비어있으면 센터라인 유지)
    std::vector<float> bypass_xs;
    std::vector<float> bypass_ys;
    std::vector<float> bypass_yaws;
    std::vector<float> bypass_vs;

    bool has_bypass_path() const { return !bypass_xs.empty(); }
};

class OvertakeFsm {
public:
    struct Config {
        float follow_dist     {5.0f};   // 이 거리 이내면 FOLLOW 진입 (m)
        float prep_dist       {3.5f};   // 이 거리 이내면 OVERTAKE_PREP 진입 (m)
        float clear_dist      {7.0f};   // 이 거리 이상이면 SOLO 복귀 (m)
        float merge_dist      {2.0f};   // 추월 완료 판정 여유 거리 (m, 상대방 후방)
        float prep_timeout_s  {2.5f};   // PREP 최대 체류 시간 (초)
        float emergency_dist  {0.5f};   // 긴급 감속 트리거 거리 (m)
        float clear_threshold {0.8f};   // 추월 가능 판정 여유 폭 (m)
        float lateral_offset  {0.5f};   // 바이패스 경로 횡방향 오프셋 (m)
        float modal_offset    {0.15f};
        float modal_ratio     {0.5f};
        float solo_speed      {6.0f};
        float follow_speed    {4.5f};
        float overtake_speed  {6.5f};
        float emergency_speed {0.0f};
    };

    explicit OvertakeFsm(const Config& cfg);

    // 매 제어 주기마다 호출. 상대방이 없으면 opp_detected=false로 전달.
    FsmCommand tick(
        const State& ego,
        bool  opp_detected,
        float opp_x, float opp_y, float opp_v,
        float h_pl,   // 상대방 좌측 여유 공간 (m)
        float h_pr,   // 상대방 우측 여유 공간 (m)
        const std::vector<float>& ref_xs,
        const std::vector<float>& ref_ys,
        const std::vector<float>& ref_yaws,
        const std::vector<float>& ref_vs);

    FsmState state() const { return state_; }

private:
    Config cfg_;
    FsmState state_ {FsmState::SOLO};
    std::chrono::steady_clock::time_point prep_entry_time_;
    float last_opp_x_ {0.0f};
    float last_opp_y_ {0.0f};
    bool  overtook_left_ {false};  // 추월 방향 기억 (MERGE 후 복귀용)

    // lateral offset으로 평행이동한 바이패스 경로 생성
    void generate_bypass_path(
        float lateral_offset_m,
        const std::vector<float>& ref_xs,
        const std::vector<float>& ref_ys,
        const std::vector<float>& ref_yaws,
        const std::vector<float>& ref_vs,
        FsmCommand& cmd) const;

    float dist_to_opp(const State& ego, float opp_x, float opp_y) const;

    // 추월 완료 여부: ego가 상대방을 merge_dist만큼 앞질렀는지
    bool passed_opponent(const State& ego, float opp_x, float opp_y) const;
};

} // namespace mppi
