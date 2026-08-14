# Real-car dynamics model v2

This pipeline fixes the deployed contract to 50 Hz, ISO body axes (`x` forward,
`y` left, yaw CCW), causal commands, an explicit steering/speed actuator model,
and residual derivatives `[delta_ax, delta_ay, delta_yaw_accel]`.

Dataset NPZ fields are `features (N,20)`, `targets (N,3)`, `bag_id (N,)`, and
`valid (N,)`. Timestamps must come from message headers; samples crossing a
reset, collision, manual intervention, stale topic, or large timestamp gap must
be false in `valid`. Keep complete bags/sessions in only one split. The exact
feature order is in `contract.py` and matches `update_dynamic_mlp_residual`.

## Recommended model: `dynamic_40ms_yaw_preserved_stage2`

This is the currently selected real-car model. It keeps the high-speed
yaw-rate solution learned with the low-speed residual gate, while runtime and
evaluation leave `delta_ax` ungated. Only the poorly observable lateral and
yaw residuals are suppressed near standstill:

```text
delta_ax        <- delta_ax
delta_ay        <- low_speed_gate(vx) * delta_ay
delta_yaw_accel <- low_speed_gate(vx) * delta_yaw_accel
```

The time contract is fixed and checked by the ROS node:

```text
ROS solve/publish period (control_dt): 0.02 s = 50 Hz
one MPPI rollout knot (model_dt):      0.04 s
horizon_steps=60:                      2.4 s rollout horizon
```

### Simplest complete command

The direct-bag NPZ files must already exist under
`model_tuning/data/real_car_v2_drive/` and
`model_tuning/results/effective_vs_dynamic_0813/data/`. Then run:

```bash
cd /home/a/smooth-mppi-cuda
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate RL
python model_tuning/real_car_v2/run_yaw_preserved_40ms_pipeline.py
```

No CLI arguments are required. The switches at the top of
`run_yaw_preserved_40ms_pipeline.py` allow completed stages to be skipped.
With all switches enabled it performs:

1. `build_dataset.py`: combine all direct-bag 20 ms NPZ data. Every continuous
   segment receives a unique `bag_id`; aggressive run1 is high-speed training
   excitation and aggressive run2 remains an unseen test bag.
2. `regress_dynamic_40ms.py`: fit the Pacejka classic model for one 40 ms MPPI
   knot using one explicit 40 ms actuator/physics update.
3. `build_dynamic_40ms_dataset.py`: calculate classic predictions and the
   derivative residual targets `[delta_ax, delta_ay, delta_yaw_accel]`.
4. `train_dynamic_40ms.py`: train the 20-64-32-3 MLP using one-step targets.
5. `finetune_dynamic_40ms_recursive.py`: perform two 1.2 s free-recursive
   passes with `GATE_AX_RESIDUAL=1`, producing
   `dynamic_40ms_yaw_preserved_stage1` and `stage2`.
6. `evaluate_dynamic_40ms.py`: evaluate validation and aggressive test bags
   with runtime-identical ungated `delta_ax` and save JSON/NPZ traces.
7. `deploy_dynamic_40ms_to_mppi.py`: verify and copy the 14252-byte CUDA
   binary, insert the fitted Pacejka parameters and weight path into
   `config/params.yaml`, and select `dynamic_mlp_residual_servo_lag`.
8. `plot_highspeed_tail_comparison.py`: save the effective/old/new held-out
   best/median/worst comparison.

All frequently changed paths and hyperparameters are constants at the top of
the corresponding script. No CLI arguments are needed for the canonical run.
The runner also has `RUN_*` switches at its top, so a completed expensive stage
can be skipped without constructing a long command.

`RUN_DEPLOY_TO_MPPI=True` enables the final YAML/deployment step. The deployment
script settings are at its top: `RESULT_PATH`, `REGRESSION_PATH`, `YAML_PATH`,
`RUNTIME_BINARY_PATH`, and `ACTIVATE_MODEL`. It changes runtime data only, so a
`colcon build` is not needed; restart the ROS node so it reloads YAML and binary.

For debugging, the equivalent stage commands are shown below. The two recursive
commands intentionally set `GATE_AX_RESIDUAL=1`; omitting it produces a
different checkpoint.

```bash
python model_tuning/real_car_v2/build_dataset.py
python model_tuning/real_car_v2/regress_dynamic_40ms.py
python model_tuning/real_car_v2/build_dynamic_40ms_dataset.py
python model_tuning/real_car_v2/train_dynamic_40ms.py
GATE_AX_RESIDUAL=1 python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_residual_seed31 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage1
GATE_AX_RESIDUAL=1 python model_tuning/real_car_v2/finetune_dynamic_40ms_recursive.py \
  model_tuning/results/dynamic_40ms_yaw_preserved_stage1 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage2
python model_tuning/real_car_v2/evaluate_dynamic_40ms.py \
  model_tuning/results/dynamic_40ms_yaw_preserved_stage2 \
  --out model_tuning/results/dynamic_40ms_yaw_preserved_stage2/rollout_ax_ungated_metrics.json
python model_tuning/real_car_v2/plot_highspeed_tail_comparison.py
python model_tuning/real_car_v2/deploy_dynamic_40ms_to_mppi.py
```

The final artifacts are:

```text
model_tuning/results/dynamic_40ms_yaw_preserved_stage2/dynamic_40ms_residual.bin
model_tuning/results/dynamic_40ms_yaw_preserved_stage2/rollout_ax_ungated_metrics.json
config/dynamic_40ms_residual_servo_lag.bin
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
canonical run, use `run_yaw_preserved_40ms_pipeline.py` without arguments.

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
