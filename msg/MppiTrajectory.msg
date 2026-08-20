# Weighted optimal trajectory horizon의 CUDA 목적함수 누적 항목.
float32 tracking_cost
float32 friction_ellipse_cost
float32 front_slip_cost
float32 rear_slip_cost
float32 steer_cost
float32 rate_cost
float32 boundary_cost
float32 obs_cost
# Terminal progress reward: -q_progress * progress_m (MPCC에서 음수).
float32 progress_cost
