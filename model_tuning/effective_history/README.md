# Effective command-history state-residual dynamics

This is an experimental `/drive command -> vehicle state` model. It does not
identify or observe a physical steering/rack angle. Steering linkage, actuator
delay, tire relaxation, yaw inertia, and observer delay are represented only as
an effective response.

## Time contract

- controller callback: 0.02 s
- command history sample: 0.02 s
- model transition: 0.04 s
- transition target: `state[t] -> state[t+2]`
- recorded replay substeps: `command[t]`, then `command[t+1]`
- future MPPI substeps: the same control knot twice
- horizon: 60 model transitions = 2.4 s

The existing controller does **not** yet satisfy this contract: it derives its
only `Params::dt` from the 50 Hz callback and shifts its warm start once per
callback. The new checkpoint therefore must not be activated until the CUDA
model and phase-aware warm start have an independent parity test.

## Reproduce the gated smoke experiment

```bash
source /home/a/anaconda3/etc/profile.d/conda.sh
conda activate RL
python model_tuning/effective_history/build_dataset.py \
  --out model_tuning/results/effective_history_smoke_v2_20260813
python model_tuning/effective_history/train.py \
  model_tuning/results/effective_history_smoke_v2_20260813/dataset_model_dt004.npz \
  --out model_tuning/results/effective_history_smoke_v2_20260813/one_step_seed31 \
  --epochs 100 --seed 31
python model_tuning/effective_history/evaluate_rollout.py \
  model_tuning/results/effective_history_smoke_v2_20260813/dataset_model_dt004.npz \
  model_tuning/results/effective_history_smoke_v2_20260813/one_step_seed31/checkpoint.pt \
  --out model_tuning/results/effective_history_smoke_v2_20260813/one_step_seed31/rollout_metrics.json
```

## Free-recursive fine-tuning

The recursive stage never injects measured state after initialization. It feeds
predicted pose/body state, derived acceleration, and command history through all
60 steps; only the recorded commands remain exogenous.

```bash
python model_tuning/effective_history/finetune_recursive.py \
  model_tuning/results/effective_history_smoke_v2_20260813/dataset_model_dt004.npz \
  model_tuning/results/effective_history_smoke_v2_20260813/one_step_seed31/checkpoint.pt \
  --out model_tuning/results/effective_history_recursive_5_10_20_30_50_60_seed31 \
  --epochs 100 --batches-per-epoch 32 --batch-size 32 \
  --validation-windows 192 --lr 0.0002 --patience 18 --seed 31
python model_tuning/effective_history/evaluate_rollout.py \
  model_tuning/results/effective_history_smoke_v2_20260813/dataset_model_dt004.npz \
  model_tuning/results/effective_history_recursive_5_10_20_30_50_60_seed31/checkpoint.pt \
  --out model_tuning/results/effective_history_recursive_5_10_20_30_50_60_seed31/rollout_metrics.json \
  --max-windows 512
```

An audit found and fixed a recursive-history off-by-one: the initial history
already contains command[t], so the rollout appends command[t+1] after the first
transition and command[t+2] before the next feature. The corrected result is
`effective_history_recursive_correct_history_seed31` (epoch 46). On untouched
speed30 at 1.2 s it achieves position mean/p95 0.249/0.575 m, vx MAE 0.313 m/s,
and yaw-rate bias/MAE/p95 0.042/0.145/0.370 rad/s. Do not deploy the older
`effective_history_recursive_5_10_20_30_50_60_seed31` artifact.

The CUDA model has a separate enum and enforces binary metadata
`control_dt=0.02`, `model_dt=0.04`. Each future control is held for two 20 ms
history/baseline substeps and one state/cost knot is emitted at 40 ms. At a
50 Hz solve rate, warm-start advances one 40 ms knot every second callback.
Run `check_cuda_parity.py` on the CUDA target before driving; a machine without
an available NVIDIA device can validate the binary/NumPy contract but cannot
complete kernel parity.

The 0813 aligned NPZs do not retain source timestamps/topic ages, teleop state,
or invalid-reason fields. The generated manifest records this limitation. Raw
bag re-extraction remains mandatory before a production/full-data experiment.

## Deployment gate

Do not activate a binary unless all of these hold on speed30 recorded-command
open-loop replay at 1.2 s: position mean/p95 below 0.30/0.60 m, vx MAE below
0.35 m/s, yaw-rate MAE/p95 below 0.40/1.00 rad/s, no divergence, and Python/CUDA
parity below the specified tolerance. The one-step smoke checkpoint is not a
deployable checkpoint.
