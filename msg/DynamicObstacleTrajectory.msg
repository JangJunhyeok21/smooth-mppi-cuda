std_msgs/Header header
# Prediction knot interval. It must match MPPI model_dt for direct indexing.
float32 dt
uint32 horizon
# Packed obstacle-major layout. Dynamic objects contribute `horizon` points;
# static objects contribute one fixed-pose point. `is_dynamic` determines each
# object's packed length. MPPI expands static objects internally for CUDA.
int64[] obstacle_ids
float32[] x
float32[] y
float32[] yaw
# Predictor collision ellipse semi-axes [m], expressed in the obstacle yaw
# frame. They already include car_radius; MPPI adds only obstacle_soft_margin.
# Static objects also provide valid axes (major == minor), producing a circle.
float32[] semi_major
float32[] semi_minor
# One flag per obstacle (not per knot). It describes whether the poses were
# predicted dynamically; MPPI uses the supplied ellipse for both flag values.
bool[] is_dynamic
