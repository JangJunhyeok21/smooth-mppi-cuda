std_msgs/Header header
# Prediction knot interval. It must match MPPI model_dt for direct indexing.
float32 dt
uint32 horizon
# Obstacle-major flattened layout: index = obstacle_index * horizon + step.
int64[] obstacle_ids
float32[] x
float32[] y
float32[] yaw
# Predictor collision ellipse semi-axes [m], expressed in the obstacle yaw
# frame. They already include car_radius; MPPI adds only obstacle_soft_margin.
# For static objects these values are ignored by MPPI.
float32[] semi_major
float32[] semi_minor
# One flag per obstacle (not per knot). Dynamic objects use the ellipse above;
# static objects use the legacy circular car_radius cost.
bool[] is_dynamic
