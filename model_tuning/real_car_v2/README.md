ms dk# Real-car dynamics model v2

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
python3 validate_dataset.py DATASET.npz
python3 train.py DATASET.npz --out result
```

Do not select the new binary on the car until: hold-out 1-step and 60-step
replay beat physics-only, p95/max errors do not regress, Python/CUDA predictions
match at 1 and 60 steps, OOD cases remain finite, and Orin MPPI latency meets the
50 Hz budget. Preserve `contract.json`, `split_manifest.json`, and metrics with
every checkpoint. The exporter intentionally retains the deployed 3563-float
format; the current default model is not changed automatically.
