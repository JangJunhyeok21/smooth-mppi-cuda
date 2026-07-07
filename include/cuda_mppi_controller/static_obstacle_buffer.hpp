#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace mppi {

// 정적 장애물 버퍼
// 정적으로 판정된 장애물을 맵 좌표로 저장해 시야 밖(블라인드 코너)에서도
// 유지하고, 그 지점이 다시 관측 가능해졌을 때 미검출이 반복되면
// "치워짐"으로 판정해 제거한다. 랩 카운터 없이 다음 랩 재확인이 구현된다.
class StaticObstacleBuffer {
public:
    struct Config {
        float merge_radius   {0.4f};   // 검출-엔트리 병합 거리 (m)
        int   min_hits       {2};      // confirmed 승격에 필요한 검출 횟수
        int   miss_limit     {5};      // 관측 가능 상태 연속 미검출 제거 임계
        float verify_range   {4.0f};   // 재확인 가능 거리 (m, 센서 신뢰 범위)
        float verify_fov_deg {240.0f}; // 재확인 가능 FOV (deg, ego yaw 기준)
        float pos_alpha      {0.3f};   // 위치 EMA 계수 (신규 관측 반영 비율)
    };

    struct Entry {
        float x {0.f};
        float y {0.f};
        int   hits   {0};
        int   misses {0};
        bool confirmed(int min_hits) const { return hits >= min_hits; }
    };

    explicit StaticObstacleBuffer(const Config& cfg) : cfg_(cfg) {}

    // 매 제어 주기 호출. detections는 이번 주기의 정적 검출 위치(맵 좌표).
    void update(const std::vector<std::pair<float, float>>& detections,
                float ego_x, float ego_y, float ego_yaw)
    {
        std::vector<bool> det_used(detections.size(), false);

        // 1. 기존 엔트리 갱신: 가장 가까운 미사용 검출과 병합
        for (auto& e : entries_) {
            int   best_j  = -1;
            float best_d2 = cfg_.merge_radius * cfg_.merge_radius;
            for (size_t j = 0; j < detections.size(); ++j) {
                if (det_used[j]) continue;
                float dx = detections[j].first  - e.x;
                float dy = detections[j].second - e.y;
                float d2 = dx * dx + dy * dy;
                if (d2 < best_d2) { best_d2 = d2; best_j = (int)j; }
            }
            if (best_j >= 0) {
                det_used[best_j] = true;
                e.x += cfg_.pos_alpha * (detections[best_j].first  - e.x);
                e.y += cfg_.pos_alpha * (detections[best_j].second - e.y);
                e.hits++;
                e.misses = 0;
            } else if (observable(e, ego_x, ego_y, ego_yaw)) {
                // 시야 안인데 미검출 → 치워졌을 가능성 누적
                e.misses++;
            }
            // 시야 밖(블라인드)이면 그대로 유지
        }

        // 2. 미매칭 검출 → 신규 엔트리
        for (size_t j = 0; j < detections.size(); ++j) {
            if (det_used[j]) continue;
            Entry e;
            e.x = detections[j].first;
            e.y = detections[j].second;
            e.hits = 1;
            entries_.push_back(e);
        }

        // 3. 제거: 관측 가능 상태에서 연속 미검출 한도 초과 → 치워짐 확정
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                [this](const Entry& e) { return e.misses >= cfg_.miss_limit; }),
            entries_.end());
    }

    // 노이즈 필터를 통과한(min_hits 이상 검출) 엔트리만 반환
    std::vector<Entry> confirmed() const {
        std::vector<Entry> out;
        for (const auto& e : entries_)
            if (e.confirmed(cfg_.min_hits)) out.push_back(e);
        return out;
    }

    const std::vector<Entry>& all() const { return entries_; }
    const Config& config() const { return cfg_; }

private:
    bool observable(const Entry& e, float ego_x, float ego_y, float ego_yaw) const {
        float dx = e.x - ego_x;
        float dy = e.y - ego_y;
        if (std::hypot(dx, dy) > cfg_.verify_range) return false;
        float ang = std::atan2(dy, dx) - ego_yaw;
        while (ang >  M_PI) ang -= 2.0f * M_PI;
        while (ang < -M_PI) ang += 2.0f * M_PI;
        return std::fabs(ang) < 0.5f * cfg_.verify_fov_deg * (float)M_PI / 180.0f;
    }

    Config cfg_;
    std::vector<Entry> entries_;
};

} // namespace mppi
