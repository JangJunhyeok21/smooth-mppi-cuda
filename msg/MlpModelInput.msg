# Exact input supplied to the first-knot 22-D residual MLP for the selected
# optimal control. Feature order is the real_car_v2 training contract:
# [vx, vy, yaw_rate, steer_cmd, speed_cmd, applied_steer, steer_cmd_delta,
#  base_next_vx, base_next_vy, base_next_yaw_rate,
#  steer_t-4, speed_t-4, ..., steer_t, speed_t, imu_ax, imu_ay]
std_msgs/Header header
uint32 contract_version
string dynamics_model
float32 model_dt
bool imu_valid
float32[22] features
# Command actually sent after node-side speed safety/rate limiting. This can
# differ from features[4], which is the command used by the optimal rollout.
float32 published_steer
float32 published_speed
