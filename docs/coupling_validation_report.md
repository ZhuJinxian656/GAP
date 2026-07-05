# Coupling Validation Report

## Repo Entrypoints

- Train: `train.sh`, which calls `scripts/train.py` with Hydra config `gap_policy/config/GAP.yaml`.
- Eval: `eval.sh`, which calls the GAP-local RoboTwin wrapper `scripts/eval_policy.py`; deployment uses `deploy_policy.py`.
- Dataset: `gap_policy/dataset/gap_dataset.py`, reading zarr arrays under `data/<task>-<setting>-<num>-pi3-20-5.zarr`.
- Policy/model: `gap_policy/policy/gap.py`, class `GAPPolicy`.
- Configs: `gap_policy/config/GAP.yaml` plus task shape metadata in `gap_policy/config/task/demo_task.yaml`.
- Preprocessing: `process_data.sh`, which calls `scripts/process_data.py` to convert RoboTwin HDF5 episodes into GAP zarr features.
- Checkpoints: `scripts/train.py` writes to `checkpoints/<task>_<checkpoint_tag-or-setting>_<expert_data_num>/<epoch>.ckpt`.
- Logging: Weights & Biases through `wandb`; local training outputs go under `data/outputs/...`.

## Selected Task/Config

Selected task: `place_dual_shoes`, `demo_clean`, 50 demonstrations.

Reason: it is the only clearly available bimanual task with existing zarr data, raw RoboTwin HDF5 episodes, baseline checkpoints, and rollout result files. The action/state shape is 14, matching two 6-DoF arms plus two gripper scalars.

## Dataset Signal Availability

| Signal | Available? | Key / source | Notes |
| --- | --- | --- | --- |
| left EEF | yes | HDF5 `/endpose/left_endpose` | Position and orientation, shape `(T, 7)` |
| right EEF | yes | HDF5 `/endpose/right_endpose` | Position and orientation, shape `(T, 7)` |
| left gripper | yes | HDF5 `/endpose/left_gripper`, `/joint_action/left_gripper` | Scalar |
| right gripper | yes | HDF5 `/endpose/right_gripper`, `/joint_action/right_gripper` | Scalar |
| current proprio | yes | zarr `data/state`; HDF5 `/joint_action/vector` | Shape `14` |
| action | yes | zarr `data/action` | Shape `14`; sampled with horizon `20` |
| object pose/keypoints | no | none found in inspected HDF5 | `pointcloud` exists but is empty, shape `(T, 0)` |
| contact labels | no | none found | no contact/touch keys |
| future observations/actions | yes | sequence sampler over zarr `action`; optional future Pi3 from same zarr | Offline loss only in this run |

## Implemented Variants

- Baseline: existing GAP path, `policy.coupling.enabled=false`, no relation token.
- Current coupling: `policy.coupling.enabled=true`, `policy.coupling.mode=current`, `feature_mode=proprio_only_fallback`.
- Shuffled control: same relation-token parameter count, but `triadic_state` is shuffled across batch items.
- Zero control: same relation-token parameter count, but `triadic_state` is replaced by zeros.

Implementation notes:

- Added `policy.coupling` config fields while preserving old `policy.use_triadic_token=false` default.
- Reused the existing `gap_policy/triadic.py` extractor and `GAPPolicy` triadic token path.
- Added `scripts/add_triadic_state_to_zarr.py` to write `data/triadic_state` into an existing zarr from raw HDF5 without recomputing DINO/Pi3 features.
- Added an opt-in `training.val_ratio`; default remains `0.0`, so existing training behavior is unchanged unless explicitly overridden.
- Added a minimal validation loop when `training.val_ratio > 0`.

## Commands

Environment used:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM <command>
```

Torch in this environment reports `torch.cuda.is_available() == False`, so the smoke/proxy runs below used CPU.

Inspect HDF5 signal availability:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/inspect_robotwin_hdf5_keys.py \
  --path /data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5 \
  --sample-items 6
```

Add deployable coupling state to the 50-demo zarr:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/add_triadic_state_to_zarr.py \
  --zarr data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr \
  --hdf5-root /data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean \
  --expert-data-num 50 \
  --triadic-mode proprio_only_fallback \
  --overwrite
```

Baseline command:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/train.py \
  task_name=place_dual_shoes setting=demo_clean expert_data_num=50 \
  training.device=cpu training.seed=0 training.num_epochs=1 \
  training.max_train_steps=3 training.max_val_steps=3 \
  training.val_ratio=0.1 training.val_every=1 training.checkpoint_every=1 \
  checkpoint.save_ckpt=false logging.mode=disabled \
  dataloader.batch_size=2 val_dataloader.batch_size=2 \
  dataloader.num_workers=0 val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false val_dataloader.persistent_workers=false \
  dataloader.pin_memory=false val_dataloader.pin_memory=false \
  use_future_loss=false future_target_mode=none \
  policy.use_future_loss=false policy.future_target_mode=none \
  policy.use_triadic_token=false policy.triadic_mode=disabled \
  policy.coupling.enabled=false
```

Current coupling command:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/train.py \
  task_name=place_dual_shoes setting=demo_clean expert_data_num=50 \
  training.device=cpu training.seed=0 training.num_epochs=1 \
  training.max_train_steps=3 training.max_val_steps=3 \
  training.val_ratio=0.1 training.val_every=1 training.checkpoint_every=1 \
  checkpoint.save_ckpt=false logging.mode=disabled \
  dataloader.batch_size=2 val_dataloader.batch_size=2 \
  dataloader.num_workers=0 val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false val_dataloader.persistent_workers=false \
  dataloader.pin_memory=false val_dataloader.pin_memory=false \
  use_future_loss=false future_target_mode=none \
  policy.use_future_loss=false policy.future_target_mode=none \
  policy.coupling.enabled=true policy.coupling.mode=current \
  policy.coupling.feature_mode=proprio_only_fallback \
  policy.triadic_mode=disabled
```

Shuffled sanity-control command:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/train.py \
  task_name=place_dual_shoes setting=demo_clean expert_data_num=50 \
  training.device=cpu training.seed=0 training.num_epochs=1 \
  training.max_train_steps=3 training.max_val_steps=3 \
  training.val_ratio=0.1 training.val_every=1 training.checkpoint_every=1 \
  checkpoint.save_ckpt=false logging.mode=disabled \
  dataloader.batch_size=2 val_dataloader.batch_size=2 \
  dataloader.num_workers=0 val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false val_dataloader.persistent_workers=false \
  dataloader.pin_memory=false val_dataloader.pin_memory=false \
  use_future_loss=false future_target_mode=none \
  policy.use_future_loss=false policy.future_target_mode=none \
  policy.coupling.enabled=true policy.coupling.mode=shuffled \
  policy.coupling.feature_mode=proprio_only_fallback \
  policy.triadic_mode=disabled
```

Zero sanity-control command:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/train.py \
  task_name=place_dual_shoes setting=demo_clean expert_data_num=50 \
  training.device=cpu training.seed=0 training.num_epochs=1 \
  training.max_train_steps=3 training.max_val_steps=3 \
  training.val_ratio=0.1 training.val_every=1 training.checkpoint_every=1 \
  checkpoint.save_ckpt=false logging.mode=disabled \
  dataloader.batch_size=2 val_dataloader.batch_size=2 \
  dataloader.num_workers=0 val_dataloader.num_workers=0 \
  dataloader.persistent_workers=false val_dataloader.persistent_workers=false \
  dataloader.pin_memory=false val_dataloader.pin_memory=false \
  use_future_loss=false future_target_mode=none \
  policy.use_future_loss=false policy.future_target_mode=none \
  policy.coupling.enabled=true policy.coupling.mode=zero \
  policy.coupling.feature_mode=proprio_only_fallback \
  policy.triadic_mode=disabled
```

Rollout eval entrypoint, for a saved comparable checkpoint:

```bash
ROBOTWIN_ROOT=/data1/home/zhu_jinxian/project/robotwin \
  bash eval.sh place_dual_shoes demo_clean <ckpt_setting> 50 <checkpoint_num> 0 "0" 100
```

This report did not run rollout eval because the comparable runs above were deliberately short CPU smoke/proxy runs with `checkpoint.save_ckpt=false`.

## Results

All rows use `place_dual_shoes/demo_clean`, seed `0`, horizon `20`, action dim `14`, batch size `2`, `training.val_ratio=0.1`, `max_train_steps=3`, `max_val_steps=3`, Pi3 observation tokens enabled, and future Pi3 auxiliary loss disabled. Validation loss is stochastic diffusion action loss over the first 3 validation batches, so treat it as a smoke/proxy metric.

| variant | seed | train steps | train avg loss | val action loss | rollout success | aux loss | notes |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| baseline | 0 | 3 | 0.9289 | 1.564935 | not run | 0.0 | no relation token |
| current coupling | 0 | 3 | 1.3847 | 1.091488 | not run | 0.0 | `triadic_state` current |
| shuffled control | 0 | 3 | 1.4597 | 1.085965 | not run | 0.0 | same token/params, batch-shuffled coupling |
| zero control | 0 | 3 | 1.3856 | 1.089891 | not run | 0.0 | same token/params, zero coupling input |

Sanity tests run:

```bash
conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python -m unittest tests/test_triadic_state.py

conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python scripts/smoke_test_triadic_forward.py

conda run -p /data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM \
  python -m py_compile gap_policy/policy/gap.py
```

## Interpretation

- Current coupling improved over the no-token baseline in this tiny proxy: `1.091488` vs `1.564935` validation action loss.
- The shuffled and zero controls did not remove the gain: `1.085965` and `1.089891`, both essentially matching current coupling.
- Therefore the observed gain is not evidence that the current left/right coupling values carry meaningful task information. The more likely explanation is the extra relation token/MLP/position token, optimization noise, or the tiny 3-step budget.
- This repo/data can test EEF/proprio coupling cheaply, but it cannot test the stronger object/contact hypothesis without additional object pose, keypoints, masks, or contact labels.
- Because no comparable rollout was run, this is not a rollout-success conclusion.

## Decision

Probably not the bottleneck for the available EEF/proprio-only coupling signal.

The full research hypothesis remains inconclusive for object/contact coupling because the inspected `place_dual_shoes` data does not expose object pose/keypoints/contact labels. Under the controlled smoke run, current coupling did not beat shuffled/zero controls, so there is not enough evidence to justify a larger learned coupling cache or VLA-action-expert bridge from this signal alone.

## Next Minimal Step

Run one longer, still controlled CPU/GPU experiment with saved checkpoints: baseline, current coupling, and zero control for the same number of updates, then rollout-evaluate all three. Do not add object/contact claims unless a task or preprocessing path exposes object/contact state.
