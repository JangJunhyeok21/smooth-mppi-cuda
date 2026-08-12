#include <gtest/gtest.h>
#include "cuda_mppi_controller/cuda_mppi_core.hpp"

// num_circles<=1이면 base_link 원점(offset=0) 하나로 수렴해야 한다
// (다중 원 코드 경로가 단일 원 경로와 동일한 결과를 내는지 확인하는 회귀 테스트 기반).
TEST(ComputeCircleOffsets, SingleCircleConvergesToBaseLinkOrigin) {
    auto offsets = mppi::compute_circle_offsets(1, 0.1f, 0.2f, 0.3f);
    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_FLOAT_EQ(offsets[0], 0.0f);
}

TEST(ComputeCircleOffsets, ZeroOrNegativeNumCirclesAlsoConverges) {
    auto offsets = mppi::compute_circle_offsets(0, 0.1f, 0.2f, 0.3f);
    ASSERT_EQ(offsets.size(), 1u);
    EXPECT_FLOAT_EQ(offsets[0], 0.0f);
}

// 요청된 특수 케이스: rear_overhang=front_overhang=0, num_circles=3 이면
// [0, wheelbase/2, wheelbase] (후륜축, 차체 중간, 전륜축 부근)가 나와야 한다.
TEST(ComputeCircleOffsets, ThreeCirclesNoOverhangSpecialCase) {
    const float wheelbase = 0.324f;
    auto offsets = mppi::compute_circle_offsets(3, 0.0f, 0.0f, wheelbase);
    ASSERT_EQ(offsets.size(), 3u);
    EXPECT_FLOAT_EQ(offsets[0], 0.0f);
    EXPECT_FLOAT_EQ(offsets[1], wheelbase / 2.0f);
    EXPECT_FLOAT_EQ(offsets[2], wheelbase);
}

// 일반화 버전: [-rear_overhang, wheelbase+front_overhang] 구간을 등간격으로 커버해야 한다.
TEST(ComputeCircleOffsets, GeneralCaseCoversFullSpanEvenly) {
    const float rear_overhang = 0.09f;
    const float front_overhang = 0.11f;
    const float wheelbase = 0.324f;
    auto offsets = mppi::compute_circle_offsets(4, rear_overhang, front_overhang, wheelbase);
    ASSERT_EQ(offsets.size(), 4u);

    const float x_min = -rear_overhang;
    const float x_max = wheelbase + front_overhang;
    EXPECT_NEAR(offsets.front(), x_min, 1e-5f);
    EXPECT_NEAR(offsets.back(), x_max, 1e-5f);

    for (size_t i = 1; i < offsets.size(); ++i) {
        float spacing = offsets[i] - offsets[i - 1];
        float expected_spacing = (x_max - x_min) / static_cast<float>(offsets.size() - 1);
        EXPECT_NEAR(spacing, expected_spacing, 1e-5f);
    }
}

TEST(ValidateCircleCoverage, PassesWhenRadiusAndSpacingAreSufficient) {
    // vehicle_width=0.30 → circle_radius(0.1575)는 이 값의 절반(0.15)보다 커야 통과
    std::vector<float> offsets = {0.0f, 0.162f, 0.324f};  // spacing=0.162
    float circle_radius = 0.1575f;                        // 0.30/2 * 1.05
    EXPECT_TRUE(mppi::validate_circle_coverage(offsets, circle_radius, 0.30f, nullptr));
}

TEST(ValidateCircleCoverage, FailsWhenRadiusTooSmallForWidth) {
    std::vector<float> offsets = {0.0f};
    float circle_radius = 0.05f;  // vehicle_width/2 = 0.15보다 작음
    std::string reason;
    EXPECT_FALSE(mppi::validate_circle_coverage(offsets, circle_radius, 0.30f, &reason));
    EXPECT_FALSE(reason.empty());
}

TEST(ValidateCircleCoverage, FailsWhenSpacingLeavesGap) {
    // circle_radius=0.1575 → 2*circle_radius=0.315 인데 간격이 0.4로 이보다 큼 → 빈틈 발생
    std::vector<float> offsets = {0.0f, 0.4f};
    float circle_radius = 0.1575f;
    std::string reason;
    EXPECT_FALSE(mppi::validate_circle_coverage(offsets, circle_radius, 0.30f, &reason));
    EXPECT_FALSE(reason.empty());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
