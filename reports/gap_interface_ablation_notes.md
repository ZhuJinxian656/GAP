# GAP Interface Ablation Notes

This report is the source of truth for the current GAP fork inspection and the
planned interface ablation scaffold. It is intentionally conservative: object,
contact, mask, and triadic conclusions are only allowed when the required data
fields are actually present.

## 1. Repository and Fork Status

Commands run for Phase 1 inspection:

```bash
git status --short --branch
git remote -v
git rev-parse HEAD
find . -maxdepth 3 -type f | sort
```

Current state:

- Local repo: `/data1/home/zhu_jinxian/project/GAP`
- Current branch: `codex/gap-robotwin-baseline-sync`
- Current commit: `e9314bb57aa7e7e0dd8e241718ff45a535a360e5`
- Worktree status at inspection: clean
- Upstream remote: `origin -> https://github.com/Chongyang-99/GAP.git`
- User fork remote: `zhu -> git@github.com-zhu-gap:ZhuJinxian656/GAP.git`
- Current branch is tracking `zhu/codex/gap-robotwin-baseline-sync`.

This local checkout is not overwritten with upstream. It is a fork working copy
with a separate `zhu` remote for the private repository.

Diff relative to `origin/main`:

```text
24 files changed, 1606 insertions(+), 22 deletions(-)
```

Custom files/modules already present relative to upstream:

- `gap_policy/triadic.py`
- `scripts/inspect_robotwin_hdf5_keys.py`
- `scripts/make_eval_report.py`
- `scripts/run_gap_vanilla.sh`
- `scripts/run_gap_pairwise.sh`
- `scripts/run_gap_triadic.sh`
- `scripts/run_gap_proprio_fallback.sh`
- `scripts/smoke_test_triadic_forward.py`
- `scripts/train_ddp.py`
- `scripts/wait_and_eval.sh`
- `tests/test_triadic_state.py`
- `train_ddp.sh`

Existing modified files relative to upstream:

- `.gitignore`
- `deploy_policy.py`
- `eval.sh`
- `gap_policy/config/GAP.yaml`
- `gap_policy/dataset/gap_dataset.py`
- `gap_policy/model/common/lr_scheduler.py`
- `gap_policy/policy/gap.py`
- `process_data.sh`
- `scripts/eval_policy.py`
- `scripts/process_data.py`
- `scripts/train.py`
- `train.sh`

Existing report:

- `reports/triadic_gap_notes.md` is an older process note. It is useful
  context, but this file supersedes it for the interface-ablation project.
- Generated JSON inspection outputs are written under `reports/` but remain
  ignored by git unless explicitly allowlisted.

## 2. Answers to Phase 1 Inspection Questions

1. Is this fork modified relative to upstream?

Yes. The branch contains one local/fork commit above `origin/main`:
`e9314bb Add RoboTwin GAP baseline tooling`.

2. Are there already latent-mode / triadic / interaction modules?

At the initial inspection point there was a triadic helper module,
`gap_policy/triadic.py`, and triadic-related config, preprocessing, dataset,
policy, deploy, and smoke-test support. There was no `latent_mode` scaffold and
no `future_target_mode` scaffold. This report now also documents the scaffold
added after that inspection.

3. Which files already contain previous custom changes?

The custom changes are the files listed in the repository status section above:
triadic utilities, DDP training, eval/report helpers, shell wrappers, and
triadic preprocessing/policy hooks.

4. Is `gap_policy/triadic.py` present?

Yes.

5. If `triadic.py` exists, what does it do?

It builds optional relation-state vectors from explicit left EEF, right EEF,
object, gripper, and proprioception candidate keys. It supports:

- `disabled`
- `pairwise`
- `triadic`
- `proprio_only_fallback`

It does not fabricate object pose. When `pairwise` or `triadic` is requested,
object pose/position/keypoint data must be present.

6. Does it support real object relation or only `proprio_only_fallback`?

The code supports real object relation in principle if object pose/keypoint
arrays are available in the observation/HDF5. In the current downloaded
`place_dual_shoes` data, those fields are not available, so only EEF/proprio
fallback-style information is practically available. Real object-centric
triadic conclusions are not supported by the current data.

7. Does current available data contain object pose/keypoints/masks?

No. Local inspection of
`/data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5`
found zero HDF5 dataset keys containing:

```text
object, obj, seg, mask, contact, keypoint, bbox, depth
```

It does contain camera calibration, RGB, EEF endpose, grippers, joint action,
and an empty `/pointcloud` array with shape `(224, 0)`.

8. Does current code already support DDP?

Yes. This fork adds `scripts/train_ddp.py` and `train_ddp.sh`. The DDP path is
recommended for long multi-GPU baseline training once data and settings are
stable.

9. Which training path is recommended?

- Recommended stable single-GPU path: `train.sh`, which wraps
  `scripts/train.py`.
- Recommended multi-GPU long-run path: `train_ddp.sh`, which wraps
  `scripts/train_ddp.py`.
- Direct `scripts/train.py` is useful for debugging and Hydra overrides, but
  shell wrappers keep environment variables and arguments more reproducible.

## 3. Actual GAP Data Flow

### `scripts/process_data.py`

HDF5 keys read by default:

- `/joint_action/vector`
- `/observation/<camera>/rgb`

Default camera:

- `head_camera`

Camera names are configurable through `--cameras` and the `CAMERAS`
environment variable in `process_data.sh`.

Optional triadic mode may also try configured candidate keys for:

- left EEF
- right EEF
- object pose/position/keypoints
- left/right grippers
- proprioception

However, this is optional and not part of vanilla GAP preprocessing.

The preprocessing code does not read depth, segmentation, object masks, object
IDs, object poses, contact, or keypoints by default. It also does not save
masks or metadata needed for object-hand token selection.

Image and feature extraction:

- HDF5 RGB bytes are decoded with `cv2.imdecode(..., cv2.IMREAD_COLOR)`,
  yielding BGR images.
- DINOv3 extraction converts BGR to RGB, normalizes by ImageNet statistics,
  and saves patch-token features.
- Pi3 extraction converts BGR to RGB, resizes to dimensions divisible by 14,
  normalizes by Pi3 model statistics, runs Pi3 encoder/decode/point decoder,
  and saves `point_hidden` token features after `model.patch_start_idx`.

Important clarification:

- The saved Pi3 data are latent token features from `point_decoder`, not
  decoded XYZ pointmap coordinates.

Zarr arrays written by current preprocessing:

- `data/head_camera`
- `data/pi3_features`
- `data/dinov3_features`
- `data/state`
- `data/action`
- optional `data/triadic_state`
- `meta/episode_ends`

No mask arrays and no interaction-state arrays are written.

### Current Local Zarr

Inspected zarr:

```text
/data1/home/zhu_jinxian/project/GAP/data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr
```

Data arrays:

```text
action:          shape=(11510, 14), dtype=float32, chunks=(100, 14)
dinov3_features: shape=(11510, 1, 300, 1024), dtype=float32, chunks=(100, 1, 300, 1024)
head_camera:    shape=(11510, 3, 240, 320), dtype=uint8, chunks=(100, 3, 240, 320)
pi3_features:   shape=(11510, 1, 391, 1024), dtype=float32, chunks=(100, 1, 391, 1024)
state:          shape=(11510, 14), dtype=float32, chunks=(100, 14)
```

Meta arrays:

```text
episode_ends: shape=(50,)
first 10: [223, 441, 667, 935, 1176, 1468, 1682, 1906, 2128, 2423]
```

Missing from current zarr:

- `future_pi3_features`
- `pi3_object_hand_mask`
- `pi3_object_mask`
- `pi3_left_hand_mask`
- `pi3_right_hand_mask`
- `pi3_background_mask`
- `interaction_state`
- `future_interaction_state`

### `gap_policy/dataset/gap_dataset.py`

Zarr keys loaded by vanilla dataset:

- `dinov3_features`
- `state`
- `action`
- `pi3_features` when `use_pi3_features=True`
- `triadic_state` only when `use_triadic_token=True`

Returned batch structure:

```text
obs/dinov3_features: [N_views, N_dino_tokens, D]
obs/pi3_features:    [N_views, N_pi3_tokens, D], if enabled
obs/agent_pos:       [14]
obs/triadic_state:   [D_rel], if enabled
action:              [horizon, 14]
future_pi3_features: [N_views, N_pi3_tokens, D], if Pi3 enabled
```

The dataset does not load a stored `future_pi3_features` zarr key. It derives
the future target from `sample["pi3_features"][-1]`, i.e. the last frame in the
action chunk.

Current observation and target timing:

- First sampled timestep is used as current DINO/Pi3/state observation.
- Full sampled sequence is used as action chunk.
- Last sampled timestep's Pi3 features are used as future Pi3 target.

### `gap_policy/policy/gap.py`

Observation token construction:

- DINO features are flattened from `[B, N_views, N_patches, D]` to
  `[B, N_views * N_patches, D]`.
- Pi3 features are flattened similarly when enabled.
- Agent state is encoded by `self.state_encoder`.
- Optional triadic state is encoded by `self.triadic_encoder`.
- Tokens are concatenated as:
  - vanilla: `[CLS, state, DINO tokens, Pi3 tokens]`
  - triadic enabled: `[CLS, state, triadic, DINO tokens, Pi3 tokens]`

Hard-coded grids currently present:

- DINO positional grid: `15 x 20`
- Pi3 positional grid: `17 x 23`
- Pi3 future query grid: `17 x 23 = 391`

Decoder query construction:

- Action queries come from embedded noisy actions plus learned action positions.
- When `use_pi3_features=True`, 391 learned Pi3 future queries are appended to
  the action queries.
- The diffusion timestep is appended as a memory token.

Future Pi3 loss:

- Computed in `GAPPolicy.compute_loss()`.
- Ground truth is `batch["future_pi3_features"]`, derived by the dataset from
  the last timestep's Pi3 latent tokens.
- The target is reshaped from `[B, N_views, N_patches, D]` to
  `[B, N_patches, N_views * D]`.
- The prediction has shape `[B, 391, N_views * D]`.
- Loss is `F.mse_loss(pi3_pred, future_pi3_flat)`.
- Current hard-coded auxiliary loss weight is `0.1`.

Loss dictionary today:

- `action_loss`
- `pi3_loss` when Pi3 future target exists
- `total_loss` when Pi3 future target exists

### `deploy_policy.py`

Online eval recomputes DINO/Pi3 features from RoboTwin observations:

- `encode_obs()` reads `observation["observation"]["head_camera"]["rgb"]`.
- It reads state from `observation["joint_action"]["vector"]`.
- `extract_dinov3_features()` normalizes the RGB image and runs DINOv3.
- `extract_pi3_features()` normalizes the RGB image and runs Pi3 encoder,
  decoder, and point decoder.
- `get_action()` builds the same `obs_dict` keys expected by `GAPPolicy`.

Online eval is broadly consistent with training in feature type and tensor
shape, but any future latent-mode change that alters Pi3 token selection or
compression must update both training and deploy paths.

## 4. Pointmap vs Latent Clarification

The current code is not supervising decoded pointmap coordinates.

It supervises Pi3 latent token features:

- Preprocessing saves `point_hidden[:, model.patch_start_idx:].float()`.
- Dataset names the last timestep version `future_pi3_features`.
- Policy predicts `pi3_pred` from learned future queries.
- `compute_loss()` applies MSE between predicted Pi3 latent features and
  future Pi3 latent features.

Supervised tensor shape in the current local run:

```text
future_pi3_features: [B, 1, 391, 1024]
future target flat:  [B, 391, 1024]
pi3_pred:            [B, 391, 1024]
```

This is a future Pi3 latent-feature objective, not an XYZ pointmap coordinate
loss.

## 5. Data Supervision Availability

Raw HDF5 inspected:

```text
/data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5
```

Found fields:

- RGB cameras:
  - `/observation/head_camera/rgb`
  - `/observation/front_camera/rgb`
  - `/observation/left_camera/rgb`
  - `/observation/right_camera/rgb`
- Camera intrinsics/extrinsics:
  - `intrinsic_cv`, `extrinsic_cv`, `cam2world_gl` for the four cameras
- EEF/endpose:
  - `/endpose/left_endpose`, shape `(224, 7)`
  - `/endpose/right_endpose`, shape `(224, 7)`
- Grippers:
  - `/endpose/left_gripper`
  - `/endpose/right_gripper`
  - `/joint_action/left_gripper`
  - `/joint_action/right_gripper`
- Action/state:
  - `/joint_action/vector`, shape `(224, 14)`
  - `/joint_action/left_arm`
  - `/joint_action/right_arm`
- Pointcloud:
  - `/pointcloud`, shape `(224, 0)`, empty

Not found:

- object pose
- object keypoints
- object masks
- robot/hand masks
- segmentation labels
- object IDs/names
- contact
- depth maps
- non-empty point cloud

Availability answer:

- Are object masks available? No.
- Are robot hand/arm masks available? No.
- Are segmentation labels available? No.
- Are object IDs/names available? No.
- Are object poses available? No.
- Are object keypoints available? No.
- Are camera intrinsics/extrinsics available? Yes.
- Are depth maps available? No.
- Are left/right EEF poses available? Yes.
- Are gripper states available? Yes.
- Is it possible to build object-hand Pi3 token masks from current data? No.
- Is it possible to build future interaction-state targets from current data?
  Not the true left-object-right form, because object pose/keypoints are
  missing. EEF-only/proprio fallback is possible but is not object-centric.

Important issue found in the earlier `scripts/inspect_robotwin_hdf5_keys.py`:

- Its heuristic can produce false positives for object pose because it matches
  generic tokens such as `pose/pos` inside `/endpose/*`.
- Stronger availability checking must require explicit object-like tokens and
  should report missing object supervision clearly.
- This has now been corrected in the updated script: `object pose` and
  `object keypoints` report `<not found>` on the current HDF5.

## 6. Implemented Modes and Current Status

The ablation scaffold is now config-controlled through `latent_mode`,
`use_future_loss`, and `future_target_mode`. Defaults preserve vanilla GAP:

```yaml
latent_mode: pi3_full
use_future_loss: true
future_target_mode: pi3_full
future_loss_weight: 0.1
```

The `future_loss_weight` default is intentionally `0.1` because that was the
previous hard-coded auxiliary Pi3 loss weight in the fork. Keeping it at `0.1`
preserves vanilla behavior for old runs and checkpoints.

Current `latent_mode` support:

- `pi3_full`: implemented; this is the vanilla full-scene Pi3 token path.
- `dino_only`: implemented; drops Pi3 observation tokens while keeping DINO and
  proprioception.
- `pi3_pooled`: implemented; adaptive-average-pools the Pi3 token grid.
- `pi3_compressed`: implemented; adds a learned attention compressor for Pi3
  observation tokens.
- `pi3_random_tokens`: implemented; uses a fixed seeded subset of Pi3 tokens.
- `pi3_token_dropout`: implemented; applies training-time Pi3 token dropout
  while preserving dense token shape.
- `pi3_object_hand`: scaffolded with strict mask checks. It requires
  `data/pi3_object_hand_mask` in zarr and is not runnable on the current data.
- `pi3_background`: scaffolded with strict mask checks. It requires
  `data/pi3_background_mask` in zarr and is not runnable on the current data.

Current `future_target_mode` support:

- `pi3_full`: implemented; this is the vanilla future Pi3 latent objective.
- `none`: implemented through `use_future_loss=false` and/or
  `future_target_mode=none`; action loss remains active.
- `pi3_pooled`: implemented; future target is pooled to the configured grid.
- `pi3_compressed`: implemented as a fixed compressed projection target; the
  projection is held out of the loss gradient so the target branch does not
  collapse toward the prediction.
- `pi3_random_tokens`: implemented; predicts a fixed seeded subset of future
  Pi3 tokens.
- `pi3_token_dropout`: implemented as a dense future Pi3 objective; dropout is
  used on observation tokens, not the target.
- `pi3_object_hand`: scaffolded with strict `future_pi3_object_hand_mask`
  checks derived from the last sampled timestep's zarr mask.
- `pi3_background`: scaffolded with strict `future_pi3_background_mask` checks
  derived from the last sampled timestep's zarr mask.
- `interaction_state`: intentionally raises `NotImplementedError` in the policy
  unless a true `future_interaction_state` target exists. The current data lacks
  object pose/keypoints, so this mode cannot support a real object-centric
  conclusion.

Existing triadic modes:

- `pairwise` and `triadic` are implemented in code but require object
  pose/keypoints. Current `place_dual_shoes` data does not provide them.
- `proprio_only_fallback` is implemented and can run, but it is explicitly not
  evidence for object-centric or triadic object coupling.

## 7. Files Inspected

Phase 1/2 files inspected:

- `scripts/process_data.py`
- `gap_policy/dataset/gap_dataset.py`
- `gap_policy/policy/gap.py`
- `gap_policy/config/GAP.yaml`
- `gap_policy/config/task/demo_task.yaml`
- `scripts/train.py`
- `scripts/train_ddp.py`
- `deploy_policy.py`
- `process_data.sh`
- `train.sh`
- `train_ddp.sh`
- `eval.sh`
- `gap_policy/triadic.py`
- `scripts/inspect_robotwin_hdf5_keys.py`
- `reports/triadic_gap_notes.md`

## 8. Phase 3 Inspection Utilities

Implemented/updated:

- `scripts/inspect_gap_zarr.py`
- `scripts/inspect_robotwin_hdf5_keys.py`
- `scripts/check_object_hand_supervision_availability.py`

### `scripts/inspect_gap_zarr.py`

Command run:

```bash
python scripts/inspect_gap_zarr.py \
  --zarr data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr \
  --sample-rows 4 \
  --sample-values 2048
```

Result:

- Printed array names, shapes, dtypes, chunks, estimated sizes, episode ends,
  and sampled stats.
- Confirmed expected arrays:
  - `data/dinov3_features`
  - `data/pi3_features`
  - `data/state`
  - `data/action`
- Confirmed missing arrays:
  - `future_pi3_features` as a stored zarr key
  - all Pi3 token mask arrays
  - interaction-related arrays

### `scripts/inspect_robotwin_hdf5_keys.py`

Command run:

```bash
python scripts/inspect_robotwin_hdf5_keys.py \
  --path /data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5 \
  --max-files 1 \
  --sample-items 3 \
  --save-json reports/hdf5_key_summary.json
```

Result:

- Printed recursive HDF5 dataset paths, shapes, dtypes, and small safe samples.
- Saved JSON summary to `reports/hdf5_key_summary.json`.
- Confirmed:
  - RGB/camera calibration exists.
  - EEF/endpose and gripper fields exist.
  - object/id/name, object pose, object keypoints, segmentation, mask, bbox,
    contact, and depth are not found.
  - `/pointcloud` exists but is empty: shape `(224, 0)`.

### `scripts/check_object_hand_supervision_availability.py`

Command run:

```bash
python scripts/check_object_hand_supervision_availability.py \
  --path /data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5 \
  --zarr data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr \
  --max-files 1 \
  --output reports/object_hand_supervision_availability.json
```

Result:

```text
object_masks_available: False
robot_hand_or_arm_masks_available: False
segmentation_labels_available: False
object_ids_or_names_available: False
object_poses_available: False
object_keypoints_available: False
camera_intrinsics_available: True
camera_extrinsics_available: True
depth_maps_available: False
left_right_eef_poses_available: True
gripper_states_available: True
contact_available: False
precomputed_zarr_masks_available: False
precomputed_interaction_state_available: False
possible_to_build_object_hand_pi3_token_masks: False
possible_to_build_future_interaction_state_targets: False
missing_for_object_hand_pi3_token_masks:
  - hand/robot masks or segmentation labels
  - object masks or segmentation labels
  - precomputed Pi3 token masks
missing_for_future_interaction_state_targets:
  - object pose or object keypoints
```

JSON summary saved to:

```text
reports/object_hand_supervision_availability.json
```

## 9. Commands Run During Inspection

Repository:

```bash
git status --short --branch
git remote -v
git rev-parse HEAD
find . -maxdepth 3 -type f | sort
git diff --stat origin/main...HEAD
git diff --name-status origin/main...HEAD
git ls-files | sort
```

Code inspection:

```bash
rg -n "root\\[|create_dataset|zarr_data|features_3d|dinov3_features|triadic|episode_ends|future_pi3|point_decoder|object|segmentation|depth|mask|endpose|joint_action|observation" scripts/process_data.py
rg -n "keys =|future_pi3|triadic_state|pi3_features|dinov3_features|state|action|normalizer|mask|interaction" gap_policy/dataset/gap_dataset.py
rg -n "encode_observations|dinov3|pi3|state_encoder|cls_token|triadic|height=15|width=20|height=17|width=23|pi3_query|future_pi3|pi3_loss|loss_dict|compute_loss|predict_action|forward_diffusion" gap_policy/policy/gap.py
rg -n "extract_dinov3|extract_pi3|point_decoder|triadic|predict_action|Pi3|DINOV3|get_action|rgb|joint_action|raw_observation" deploy_policy.py
```

Data inspection:

```bash
python scripts/inspect_robotwin_hdf5_keys.py \
  /data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5 \
  --max_files 1 --sample_items 3

python - <<'PY'
import h5py
p = "/data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/demo_clean/data/episode0.hdf5"
keywords = ["object", "obj", "seg", "mask", "contact", "keypoint", "bbox", "depth",
            "pointcloud", "intrinsic", "extrinsic", "cam2world", "endpose",
            "gripper", "joint_action"]
with h5py.File(p, "r") as f:
    keys = []
    f.visititems(lambda name, obj: keys.append("/" + name) if hasattr(obj, "shape") else None)
print("dataset_count", len(keys))
for kw in keywords:
    hits = [k for k in keys if kw.lower() in k.lower()]
    print(kw, len(hits), hits[:20])
PY

python - <<'PY'
import os
import zarr
import numpy as np
p = "data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr"
r = zarr.open(p, mode="r")
print("zarr", os.path.abspath(p))
print("data_keys", sorted(r["data"].array_keys()))
print("meta_keys", sorted(r["meta"].array_keys()))
for k in sorted(r["data"].array_keys()):
    a = r["data"][k]
    print(k, "shape=", a.shape, "dtype=", a.dtype, "chunks=", a.chunks)
print("episode_ends_first10", r["meta"]["episode_ends"][:10].tolist())
PY
```

Tool validation:

```bash
chmod 755 scripts/inspect_gap_zarr.py \
  scripts/inspect_robotwin_hdf5_keys.py \
  scripts/check_object_hand_supervision_availability.py

python -m py_compile \
  scripts/inspect_gap_zarr.py \
  scripts/inspect_robotwin_hdf5_keys.py \
  scripts/check_object_hand_supervision_availability.py
```

Status:

- `py_compile` succeeded.
- `inspect_gap_zarr.py` succeeded on the current 50-demo zarr.
- `inspect_robotwin_hdf5_keys.py` succeeded on `episode0.hdf5` and saved JSON.
- `check_object_hand_supervision_availability.py` succeeded and saved JSON.

## 10. Interpretation Guide for Future Ablations

- If `dino_only` is much worse than `pi3_full`, current Pi3 geometry tokens are
  useful.
- If `no_future` is worse than `pi3_full`, future latent supervision contributes
  beyond current observation tokens.
- If `pi3_pooled` or `pi3_compressed` preserves most performance, full dense Pi3
  scene latent may be overcomplete.
- If random Pi3 tokens perform poorly, spatial/geometric structure matters.
- If `object_hand` performs close to `pi3_full`, useful Pi3 signal is
  concentrated around manipulation-relevant object/hand regions.
- If `background` performs poorly, background scene geometry is not the main
  source of gain.
- If `object_hand` beats full on strongly coupled tasks, object-hand interface
  may be a better coupling entrance than full scene latent.
- If `object_hand` cannot be run because masks are unavailable, no
  object-centric conclusion can be made from current data.
- If `interaction_state` cannot be built because object/EEF fields are
  unavailable, triadic relation experiments require simulator-side data export.

Current local conclusion:

- The present public `place_dual_shoes` data can support `pi3_full`,
  `dino_only`, `no_future`, pooled/compressed/random/dropout Pi3 latent
  ablations.
- It cannot support scientifically valid object-hand/background mask ablations
  without additional mask/segmentation export.
- It cannot support true object-centric triadic/interaction-state conclusions
  without object pose/keypoint export.

## 11. Engineering Scaffold Added

New or updated implementation files:

- `gap_policy/latent_modes.py`: shared helpers for mode validation, pooling,
  random token selection, dense token masks, token dropout, and the learned
  token compressor.
- `gap_policy/config/GAP.yaml`: exposes `latent_mode`, `use_future_loss`,
  `future_target_mode`, `future_loss_weight`, `latent`, and `future` config.
- `gap_policy/dataset/gap_dataset.py`: loads optional mask and interaction
  fields only when the chosen modes require them; missing optional fields now
  raise explicit errors naming the requested mode and available zarr arrays.
- `gap_policy/policy/gap.py`: routes observation Pi3 tokens and future Pi3
  targets through the selected modes while keeping vanilla parameter names and
  tensor shapes for `pi3_full`.
- `scripts/train.py` and `scripts/train_ddp.py`: pass the new config switches
  into the dataset and policy paths.
- `scripts/generate_gap_ablation_commands.py`: writes, but does not execute,
  commands for the minimal ablation grid.
- `scripts/generated_ablation_commands.sh`: generated command set for the
  current local `place_dual_shoes` 50-demo scaffold.
- `scripts/smoke_test_latent_modes.py`: synthetic policy test for observation
  latent modes.
- `scripts/smoke_test_future_modes.py`: synthetic policy test for future target
  modes.
- `scripts/smoke_test_dataset_optional_fields.py`: synthetic zarr test for
  optional dataset fields and strict missing-mask errors.
- `scripts/run_minimal_interface_smoke.sh`: one-command local smoke runner.
- `configs/task_subsets/coupling_probe.yaml`: task-subset placeholder for later
  coupling-category sweeps. Categories are intentionally marked TODO unless
  manually verified.

Generated large artifacts, data, checkpoints, zarr stores, pretrained weights,
wandb logs, and result videos remain ignored by git. This report is allowlisted
under `.gitignore` so the reasoning and inspection record can be committed.

## 12. Commands for Ablation Runs

Generate a command script without launching the sweep:

```bash
python scripts/generate_gap_ablation_commands.py \
  --tasks place_dual_shoes \
  --settings demo_clean \
  --expert-data-num 50 \
  --seeds 0 \
  --gpus 0 1 \
  --batch-size 32 \
  --num-epochs 200 \
  --checkpoint-every 100 \
  --variants vanilla dino_only no_future pi3_pooled pi3_compressed pi3_random pi3_dropout \
  --output scripts/generated_ablation_commands.sh
```

The generated script covers the local modes that can be run without additional
mask/object supervision:

- `vanilla`: `latent_mode=pi3_full`, `future_target_mode=pi3_full`
- `dino_only`: `latent_mode=dino_only`, `use_future_loss=false`
- `no_future`: `latent_mode=pi3_full`, `use_future_loss=false`
- `pi3_pooled`: pooled current and future Pi3 tokens
- `pi3_compressed`: compressed current and future Pi3 latent targets
- `pi3_random`: fixed random Pi3 token subset
- `pi3_dropout`: training-time Pi3 token dropout

Do not run `pi3_object_hand`, `pi3_background`, or `interaction_state` as
scientific experiments on the current local data. Their failure is the expected
signal that the required mask/object-pose supervision is absent.

## 13. Validation Status

Commands run after adding the scaffold:

```bash
python -m py_compile \
  gap_policy/latent_modes.py \
  gap_policy/policy/gap.py \
  gap_policy/dataset/gap_dataset.py \
  scripts/train.py \
  scripts/train_ddp.py \
  scripts/inspect_gap_zarr.py \
  scripts/inspect_robotwin_hdf5_keys.py \
  scripts/check_object_hand_supervision_availability.py \
  scripts/generate_gap_ablation_commands.py \
  scripts/smoke_test_latent_modes.py \
  scripts/smoke_test_future_modes.py \
  scripts/smoke_test_dataset_optional_fields.py

python scripts/smoke_test_latent_modes.py
python scripts/smoke_test_future_modes.py
python scripts/smoke_test_dataset_optional_fields.py

PYTHON_BIN=/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python \
  bash scripts/run_minimal_interface_smoke.sh
```

Results:

- Python compilation passed for the modified policy, dataset, latent helper,
  training, inspection, command-generation, and smoke-test scripts.
- Synthetic latent-mode smoke tests passed.
- Synthetic future-target smoke tests passed.
- Synthetic dataset optional-field smoke tests passed.
- The minimal interface smoke runner passed. In the current non-escalated tool
  environment `torch.cuda.is_available()` was false, so the optional one-step
  CUDA training smoke was skipped there.
- A checkpoint-load smoke confirmed that the existing vanilla checkpoint
  `checkpoints/place_dual_shoes_demo_clean_50/200.ckpt` loads with no missing
  or unexpected keys under the default `pi3_full` / `pi3_full` config.

Remaining validation before making scientific claims:

- Run at least the minimal train/eval pair for each non-object ablation mode.
- Prefer full 100-demo data if matching the paper setup is the goal.
- Export or obtain masks/object pose/keypoints before any object-hand,
  background, or true triadic/interaction-state claim.

## 14. From Scene Future Latents to Interaction Future Latents

GAP should be treated as the scene-level future Pi3 latent baseline:

```text
RGB / Pi3 scene latent -> diffusion action decoder
                    plus future Pi3 latent supervision
```

The central research question is whether the useful signal comes from the full
scene-level Pi3/future latent or from dense latent tokens that implicitly focus
on hand-object interaction regions.

The correct sequence is:

1. First test whether full-scene Pi3 tokens and future Pi3 supervision matter.
2. Then test whether dense Pi3 can be pooled, compressed, randomized, or
   dropped without losing performance.
3. Only after object/hand masks or object pose/keypoints are actually
   available should this fork claim object-hand, object-interface, or
   triadic future-interaction modeling.

Related motivation:

- PPI-style object pointflow / gripper keypose interfaces suggest that compact
  object-motion/contact-centric futures may replace full scene future latents.
- RoTri-Diff-style left-object-right modeling motivates explicit triadic
  relation interfaces.
- This fork should first establish whether GAP's gains are scene-level or
  interaction-region-level before adding true object-centric future latents.

## 15. Runtime Ablation Launch Log

On 2026-06-27, three single-GPU 200-epoch ablations were launched in tmux on
the 50-demo `place_dual_shoes/demo_clean` zarr. Each run writes to a separate
checkpoint directory through `checkpoint_tag`, so it will not overwrite the
vanilla baseline checkpoint.

| Session | GPU | Variant | Log | Checkpoint directory |
| --- | ---: | --- | --- | --- |
| `gap_ablate_dino_only` | 1 | `latent_mode=dino_only`, no Pi3, no future loss | `logs/gap_ablate_dino_only_gpu1.log` | `checkpoints/place_dual_shoes_demo_clean_dino_only_seed0_50/` |
| `gap_ablate_no_future` | 2 | full Pi3 observation, future loss disabled | `logs/gap_ablate_no_future_gpu2.log` | `checkpoints/place_dual_shoes_demo_clean_no_future_seed0_50/` |
| `gap_ablate_pi3_pooled` | 3 | pooled Pi3 observation plus pooled future target | `logs/gap_ablate_pi3_pooled_gpu3.log` | `checkpoints/place_dual_shoes_demo_clean_pi3_pooled_seed0_50/` |

Follow-up eval commands after epoch 200 checkpoints exist:

```bash
bash eval.sh place_dual_shoes demo_clean demo_clean_dino_only_seed0 50 200 1 0 10
bash eval.sh place_dual_shoes demo_clean demo_clean_no_future_seed0 50 200 2 0 10
bash eval.sh place_dual_shoes demo_clean demo_clean_pi3_pooled_seed0 50 200 3 0 10
```

To keep the machine at three concurrent training jobs, the second token-level
batch is queued through `scripts/launch_gap_ablation_queue.sh` in tmux session
`gap_ablate_queue`. It waits for `gap_ablate_dino_only`,
`gap_ablate_no_future`, and `gap_ablate_pi3_pooled` to exit, then launches:

| Session | GPU | Variant | Log | Checkpoint directory |
| --- | ---: | --- | --- | --- |
| `gap_ablate_pi3_compressed` | 1 | compressed Pi3 observation plus compressed future target | `logs/gap_ablate_pi3_compressed_gpu1.log` | `checkpoints/place_dual_shoes_demo_clean_pi3_compressed_seed0_50/` |
| `gap_ablate_pi3_random` | 2 | random Pi3 token subset plus random-token future target | `logs/gap_ablate_pi3_random_gpu2.log` | `checkpoints/place_dual_shoes_demo_clean_pi3_random_seed0_50/` |
| `gap_ablate_pi3_dropout` | 3 | full Pi3 future target with token dropout in observation | `logs/gap_ablate_pi3_dropout_gpu3.log` | `checkpoints/place_dual_shoes_demo_clean_pi3_dropout_seed0_50/` |

Queue status log:

```bash
tail -f logs/gap_ablation_queue.log
```

Evaluation environment smoke:

```bash
env PATH=/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin:$PATH \
  ROBOTWIN_ROOT=/data1/home/zhu_jinxian/project/robotwin \
  CKPT_PATH=/data1/home/zhu_jinxian/project/GAP/checkpoints/place_dual_shoes_demo_clean_50/200.ckpt \
  RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_smoke_eval \
  bash eval.sh place_dual_shoes demo_clean demo_clean_smoke_env 50 200 4 0 1
```

This completed a real RoboTwin rollout with DINOv3/Pi3/checkpoint loading and
saved `_result.txt` plus `episode0.mp4` under `results_smoke_eval/`. The
episode failed (`0/1`), but the purpose was to verify the evaluation stack and
avoid the earlier `sapien` import failure; success-rate claims still require
the full eval runs below.

The six ablation evals are queued through
`scripts/launch_gap_ablation_eval_queue.sh` in tmux session
`gap_eval_ablation_queue`. It waits for all six `200.ckpt` files and then runs
10 rollouts per ablation sequentially on GPU 4:

```bash
tail -f logs/gap_eval_ablation_queue.log
```

The eval queue refreshes `reports/gap_ablation_result_summary.md` after all
queued evals finish. It can also be updated manually:

```bash
python scripts/summarize_gap_ablation_results.py --root .
```
