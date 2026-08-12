# Real-car dynamics model v2

This pipeline fixes the deployed contract to 50 Hz, ISO body axes (`x` forward,
`y` left, yaw CCW), causal commands, an explicit steering/speed actuator model,
and residual derivatives `[delta_ax, delta_ay, delta_yaw_accel]`.

Dataset NPZ fields are `features (N,20)`, `targets (N,3)`, `bag_id (N,)`, and
`valid (N,)`. Timestamps must come from message headers; samples crossing a
reset, collision, manual intervention, stale topic, or large timestamp gap must
be false in `valid`. Keep complete bags/sessions in only one split. The exact
feature order is in `contract.py` and matches `update_dynamic_mlp_residual`.

On the training PC:

```bash
python3 regress_longitudinal_actuator.py
python3 build_dataset.py
python3 validate_dataset.py DATASET.npz
python3 train.py DATASET.npz --out result
python3 check_mppi_step_parity.py
```

`regress_longitudinal_actuator.py` identifies acceleration/braking speed-reference
time constants and its slew-rate limit from 1 s recursive odom-vx rollouts. It
uses source-session-disjoint validation/test bags and writes a candidate report.
Set `UPDATE_CONFIG=True` only after full residual-rollout validation passes.
Run it before rebuilding the residual targets: otherwise the MLP is trained to
compensate for the old actuator parameters.

`check_mppi_step_parity.py` feeds the same state, command, actuator states and
history to the PyTorch training step and to the exact CUDA
`update_dynamic_mlp_residual` step. Run it on the Orin/GPU host after building;
the test fails when the maximum full-state/history difference exceeds `2e-5`.

The runtime names are intentionally distinct. `dynamic_mlp_residual` is the
causal no-lag contract (`steer[t]=steer_cmd[t-1]`, direct speed command), while
`dynamic_mlp_residual_servo_lag` recursively maintains applied steering and
speed-reference actuator states. Their checkpoint paths must not be swapped.

Do not select the new binary on the car until: hold-out 1-step and 60-step
replay beat physics-only, p95/max errors do not regress, Python/CUDA predictions
match at 1 and 60 steps, OOD cases remain finite, and Orin MPPI latency meets the
50 Hz budget. Preserve `contract.json`, `split_manifest.json`, and metrics with
every checkpoint. The exporter intentionally retains the deployed 3563-float
format; the current default model is not changed automatically.
