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
cd /home/a/smooth-mppi-cuda
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate RL
python model_tuning/real_car_v2/run_dynamic_40ms_pipeline.py
```

The runner above is the recommended command. It executes the complete 40 ms
dynamic + servo-lag residual pipeline in this order:

1. `build_dataset.py`: combine the direct-bag 20 ms NPZ files and assign
   session-disjoint train/validation/test splits.
2. `regress_dynamic_40ms.py`: fit the Pacejka classic model for one 40 ms MPPI
   knot using one explicit 40 ms actuator/physics update.
3. `build_dynamic_40ms_dataset.py`: calculate classic predictions and the
   derivative residual targets `[delta_ax, delta_ay, delta_yaw_accel]`.
4. `train_dynamic_40ms.py`: train the 20-64-32-3 MLP using one-step targets.
5. `finetune_dynamic_40ms_recursive.py`: perform 1.2 s free-recursive training.
   The runner performs two recursive passes; stage 2 starts from stage 1.
6. `evaluate_dynamic_40ms.py`: evaluate validation and aggressive test bags
   over a 1.2 s open-loop horizon and save JSON and NPZ traces.
7. `deploy_dynamic_40ms_to_mppi.py`: verify and copy the 14252-byte CUDA
   binary, insert the fitted Pacejka parameters and weight path into
   `config/params.yaml`, and select `dynamic_mlp_residual_servo_lag`.

All frequently changed paths and hyperparameters are constants at the top of
the corresponding script. No CLI arguments are needed for the canonical run.
The runner also has `RUN_*` switches at its top, so a completed expensive stage
can be skipped without constructing a long command.

`RUN_EXPORT_TO_MPPI=True` enables the final YAML/deployment step. The deployment
script settings are at its top: `RESULT_PATH`, `REGRESSION_PATH`, `YAML_PATH`,
`RUNTIME_BINARY_PATH`, and `ACTIVATE_MODEL`. It changes runtime data only, so a
`colcon build` is not needed; restart the ROS node so it reloads YAML and binary.

The individual no-argument commands are:

```bash
python model_tuning/real_car_v2/build_dataset.py
python model_tuning/real_car_v2/regress_dynamic_40ms.py
python model_tuning/real_car_v2/build_dynamic_40ms_dataset.py
python model_tuning/real_car_v2/train_dynamic_40ms.py
python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py
python model_tuning/real_car_v2/evaluate_dynamic_40ms.py
python model_tuning/real_car_v2/deploy_dynamic_40ms_to_mppi.py
```

The best artifact produced by the two-stage canonical run is:

```text
model_tuning/results/dynamic_40ms_recursive_stage2_seed31/dynamic_40ms_residual.bin
```

Its binary contract remains little-endian float32, 3563 floats / 14252 bytes:
three linear-layer weights and biases followed by 20 feature means and 20
feature standard deviations.

## Commands used for the reported 0813 result

The reported result was generated with the same stages as the runner. Before
the no-argument defaults were added, the parameterized training stages were:

```bash
python model_tuning/real_car_v2/train_dynamic_40ms.py \
  model_tuning/data/dynamic_40ms_residual.npz \
  --out model_tuning/results/dynamic_40ms_residual_seed31 \
  --epochs 300 --seed 31 --device cuda

python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_residual_seed31 \
  --out model_tuning/results/dynamic_40ms_recursive_seed31 \
  --epochs 100 --seed 31

python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_recursive_seed31 \
  --out model_tuning/results/dynamic_40ms_recursive_stage2_seed31 \
  --epochs 100 --seed 31
```

These long commands are retained here only as the experiment record. For a new
canonical run, use `run_dynamic_40ms_pipeline.py` without arguments.

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
speed-reference actuator states. Each rollout knot is one 40 ms update, while
ROS still resolves and publishes a new first control at 50 Hz. Its checkpoint
path must not be swapped with the 20 ms no-lag model.

Do not select the new binary on the car until: hold-out 1-step and 60-step
replay beat physics-only, p95/max errors do not regress, Python/CUDA predictions
match at 1 and 60 steps, OOD cases remain finite, and Orin MPPI latency meets the
50 Hz budget. Preserve `contract.json`, `split_manifest.json`, and metrics with
every checkpoint. The exporter intentionally retains the deployed 3563-float
format; the current default model is not changed automatically.
