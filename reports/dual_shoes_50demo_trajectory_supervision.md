# Released Public-50 Place Dual Shoes Trajectory Supervision Study

Generated: 2026-06-28

## Final Eval100 Update

The trajectory-supervision round has now been trained and evaluated. The earlier sections in this report preserve the implementation-time state and generated-command protocol; the authoritative final result analysis is:

```text
reports/dual_shoes_trajectory_eval100_analysis.md
reports/dual_shoes_trajectory_results_table.md
reports/dual_shoes_trajectory_results.json
```

Final eval100 readout:

| variant | eval100 |
| --- | ---: |
| `dino_only` | 24/100 |
| `pi3_eef_region` | 18/100 |
| `pi3_pooled` | 16/100 |
| `changed_token_future` | 9/100 |
| `vanilla` | 8/100 |
| `no_future` | 8/100 |
| `delta_future` | 8/100 |
| `pi3_non_eef_region` | 8/100 |
| `pi3_dropout` | 8/100 |
| `pi3_compressed` | 9/100 |
| `pi3_random` | 6/100 |

Conclusion:

> The current eval100 result does not support the narrow hypothesis that the implemented trajectory-coupled targets (`pi3_delta`, `pi3_changed_tokens`) improve over full-scene future Pi3 supervision. The more promising signal is spatial/bottlenecked supervision (`pi3_eef_region`, `pi3_pooled`) and the strongest control remains `dino_only`.

## Current State

- Repository: `/data1/home/zhu_jinxian/project/GAP`
- Git commit: `c0e41e45353f328fee9f4994ee3b3d8b9571270e`
- Branch: `codex/gap-robotwin-baseline-sync`
- Git status at report time:
  - Modified in this task: `gap_policy/config/GAP.yaml`, `gap_policy/latent_modes.py`, `gap_policy/policy/gap.py`
  - New in this task: trajectory supervision scripts, generated command scripts, diagnostic outputs, and this report
  - Pre-existing untracked helper files remain present and were not reverted: `scripts/audit_robotwin_data_sources.py`, `scripts/compare_clean50_randomized500_schema.py`, `scripts/generated_randomized100_candidate_commands.sh`, `scripts/validate_robotwin_demo_dir.py`

Environment variables:

| Variable | Observed | Reported local default |
| --- | --- | --- |
| `ROBOTWIN_ROOT` | unset | `/data1/home/zhu_jinxian/project/robotwin` |
| `RAW_DATA_ROOT` | unset | `/data1/home/zhu_jinxian/project/robotwin/data` |
| `OUTPUT_ROOT` | unset | `/data1/home/zhu_jinxian/project/GAP/data` |
| `RESULTS_ROOT` | unset | task-specific roots below |
| `GAP_PRETRAINED_ROOT` | unset | `/data1/home/zhu_jinxian/project/GAP/pretrained` |

Current known data status:

- Task: `place_dual_shoes`
- Task config: `demo_clean`
- Expert demos: `50`
- Raw local clean data resolves to `robotwin/dataset/place_dual_shoes/aloha-agilex_clean_50`
- Current zarr: `data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr`
- Zarr episode count: `50`
- Zarr total timesteps: `11510`
- Pi3 feature shape: `[11510, 1, 391, 1024]`
- DINOv3 feature shape: `[11510, 1, 300, 1024]`
- State/action shape: `[11510, 14]`
- EEF proxy masks exist: `pi3_eef_region_mask`, `pi3_non_eef_region_mask`
- True object masks, robot/hand masks, object poses, object keypoints, and contact labels are not available in the current zarr/HDF5.

Current known eval50 table:

| variant | success |
| --- | ---: |
| `vanilla` | `4/50 = 8%` |
| `dino_only` | `8/50 = 16%` |
| `no_future` | `1/50 = 2%` |
| `pi3_pooled` | `6/50 = 12%` |
| `pi3_compressed` | `2/50 = 4%` |
| `pi3_random` | `0/50 = 0%` |
| `pi3_dropout` | `4/50 = 8%` |
| `pi3_eef_region` | `6/50 = 12%` |
| `pi3_non_eef_region` | `5/50 = 10%` |

External reproduction clue:

- A public GitHub issue reports that another user obtained official released `place_dual_shoes-demo_clean-50` vanilla performance around `11/100`.
- The local vanilla result, `4/50 = 8%`, is close enough to treat the public 50-demo setting as a plausible low-success regime.

Updated conclusion:

> The public 50-demo Place Dual Shoes setting appears to be a valid low-success regime for mechanism study.

This report therefore treats the task as a released public 50-demo Place Dual Shoes mechanism study, not as paper-main 100-demo reproduction.

## Core Question

The mechanism question is now:

> Should future Pi3 supervision predict the complete future scene, or should it supervise the action-induced change?

The working hypothesis is that a bimanual action expert is not only learning `obs -> action`. It is also learning task phase inference and phase-conditioned two-arm motion. A full-scene future Pi3 target may contain useful geometry, but it also contains many static or weakly action-coupled scene tokens. More trajectory-coupled supervision targets are:

- `pi3_delta`: `z_future - z_current`
- `pi3_changed_tokens`: full future target, but loss only on tokens with large `||z_future - z_current||`
- `pi3_delta_changed_tokens`: optional combined mode, loss on delta only at changed tokens

These are proxy trajectory targets. They are not true object-hand masks and should not be described as object-centric or triadic evidence.

## Frozen Protocol

Dataset:

- `task = place_dual_shoes`
- `task_config = demo_clean`
- `expert_data_num = 50`
- `seed = 0` for the first mechanism round
- zarr: `data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr`

Training:

- `batch_size = 32`
- `num_epochs = 200`
- `checkpoint_every = 100`
- evaluated checkpoint: `200`
- single-GPU training for new variants
- vanilla GAP defaults preserved:
  - `latent_mode=pi3_full`
  - `use_future_loss=true`
  - `future_target_mode=pi3_full`

Evaluation:

- `test_num = 100` for new results
- existing-baseline eval root: `results_dual_shoes_50demo_eval100`
- new trajectory eval root: `results_dual_shoes_50demo_trajectory_eval100`
- no writes to `results_eval50_full`

Core variants:

| variant | purpose |
| --- | --- |
| `vanilla` | original GAP in the public-50 setting |
| `dino_only` | checks whether Pi3 current tokens are necessary |
| `no_future` | checks whether future supervision matters |
| `pi3_pooled` | checks dense Pi3 redundancy |
| `delta_future` | tests derivative-like trajectory-coupled supervision |
| `changed_token_future` | tests whether action-induced changed regions are better supervision targets than full scene |
| `delta_changed_token_future` | optional combined proxy, implemented because it was a small extension |

## Existing Baseline Eval100 Consolidation

Found existing checkpoint `200.ckpt` for all nine previous eval50 variants:

- `checkpoints/place_dual_shoes_demo_clean_50/200.ckpt`
- `checkpoints/place_dual_shoes_demo_clean_dino_only_seed0_50/200.ckpt`
- `checkpoints/place_dual_shoes_demo_clean_no_future_seed0_50/200.ckpt`
- `checkpoints/place_dual_shoes_demo_clean_pi3_pooled_seed0_50/200.ckpt`
- optional previous variants also exist: `pi3_compressed`, `pi3_random`, `pi3_dropout`, `pi3_eef_region`, `pi3_non_eef_region`

Generated:

- `scripts/generate_dual_shoes_eval100_commands.py`
- `scripts/generated_dual_shoes_eval100_commands.sh`
- `reports/dual_shoes_eval100_existing_baselines.json`

The generated eval100 script contains only the four key baselines:

- `vanilla`
- `dino_only`
- `no_future`
- `pi3_pooled`

Eval100 was not launched in this task. The reason is practical caution: four 100-rollout RoboTwin evaluations can be long and produce large rollout/video outputs. The commands are ready and write to `results_dual_shoes_50demo_eval100`, not `results_eval50_full`.

## Future Pi3 Copyability Diagnostic

Added:

- `scripts/analyze_future_pi3_copyability.py`
- output: `reports/future_pi3_copyability_dual_shoes50.json`
- per-episode CSV: `reports/future_pi3_copyability_dual_shoes50_per_episode.csv`

Command run:

```bash
/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python scripts/analyze_future_pi3_copyability.py --zarr data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr --output reports/future_pi3_copyability_dual_shoes50.json --csv reports/future_pi3_copyability_dual_shoes50_per_episode.csv --changed-token-percentile 90 --max-samples 512
```

Result summary on 512 uniformly sampled sequence starts:

| metric | value |
| --- | ---: |
| samples | `512` |
| Pi3 shape | `[11510, 1, 391, 1024]` |
| copy MSE, `MSE(z_current, z_future)` | `0.016599` |
| mean token delta norm | `2.5761` |
| p50 token delta norm | `1.1549` |
| p90 token delta norm | `7.5515` |
| p99 token delta norm | `14.4529` |
| static token ratio at threshold `1e-3` | `0.0020` |
| changed-token ratio at percentile 90 | `0.1000` |

EEF proxy mask diagnostic:

| region | selected token ratio | copy MSE | mean delta norm | note |
| --- | ---: | ---: | ---: | --- |
| `pi3_eef_region_mask` | `0.0483` | `0.02637` | `3.9341` | EEF proxy, not true object-hand |
| `pi3_non_eef_region_mask` | `0.9517` | `0.01610` | `2.5073` | EEF proxy complement |

Interpretation:

- The future target is not purely static, but high-change tokens are sparse.
- A percentile changed-token objective is a reasonable proxy for action-induced regions.
- The EEF proxy region has larger average delta, but it is not a true object-hand mask and cannot support object-centric claims.

## Action-Coupling Diagnostic

Added:

- `scripts/analyze_action_coupling_gain.py`
- output: `reports/action_coupling_gain_dual_shoes50.json`

Command run:

```bash
/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python scripts/analyze_action_coupling_gain.py --zarr data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr --output reports/action_coupling_gain_dual_shoes50.json --target pooled_delta --max-samples 512 --ridge-alpha 1.0 --train-ratio 0.8
```

Result summary:

| metric | value |
| --- | ---: |
| samples | `512` |
| train/test samples | `410 / 102` |
| target | `pooled_delta` |
| target dim | `1024` |
| current-only feature dim | `1038` |
| action chunk dim | `280` |
| MSE current-only | `0.00040323` |
| MSE current+action | `0.00033855` |
| coupling gain | `0.00006468` |
| relative gain | `0.1604` |

Interpretation:

- Adding expert action chunk improves simple future-delta prediction by about `16%` relative MSE on this sampled diagnostic.
- This supports the idea that `z_future - z_current` contains action-coupled signal.
- It does not by itself prove that a trained policy will improve; it motivates the controlled delta/changed-token training variants.

## Implemented Future Target Modes

Modified:

- `gap_policy/latent_modes.py`
- `gap_policy/policy/gap.py`
- `gap_policy/config/GAP.yaml`

New modes:

- `future_target_mode=pi3_delta`
- `future_target_mode=pi3_changed_tokens`
- `future_target_mode=pi3_delta_changed_tokens`

Implementation choice:

- `pi3_delta` uses option A from the task prompt: predict delta directly. The future prediction head output is interpreted as `delta_hat`, and the target is `z_future - z_current`.
- `pi3_changed_tokens` keeps the model predicting full future Pi3 tokens, but masks the loss to high-delta tokens.
- `pi3_delta_changed_tokens` predicts the delta target and masks the loss to high-delta tokens.

Config additions:

```yaml
future:
  changed_token_percentile: 90
  changed_token_topk: null
  changed_token_stopgrad_mask: true
```

Vanilla remains unchanged by default:

```yaml
use_future_loss: true
future_target_mode: pi3_full
```

`no_future` remains:

```yaml
use_future_loss: false
future_target_mode: none
```

Loss dictionary still includes:

- `action_loss`
- `future_loss`
- `pi3_loss`
- `future_loss_mode`
- `total_loss`

## Smoke Tests

Added:

- `scripts/smoke_test_trajectory_future_modes.py`

Commands run:

```bash
/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python scripts/smoke_test_trajectory_future_modes.py
/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python scripts/smoke_test_future_modes.py
/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin/python -m py_compile gap_policy/latent_modes.py gap_policy/policy/gap.py scripts/analyze_future_pi3_copyability.py scripts/analyze_action_coupling_gain.py scripts/generate_dual_shoes_eval100_commands.py scripts/generate_dual_shoes_trajectory_commands.py scripts/parse_dual_shoes_results.py scripts/smoke_test_trajectory_future_modes.py
```

Status:

- `scripts/smoke_test_trajectory_future_modes.py`: passed
- `scripts/smoke_test_future_modes.py`: passed
- `py_compile`: passed

The trajectory smoke test verifies:

- `pi3_full` target unchanged on synthetic tensors
- `pi3_delta` target shape and value
- `pi3_changed_tokens` mask shape and selected token count
- `pi3_delta_changed_tokens` finite loss
- `no_future` disables future loss
- all-zero delta does not crash
- missing/malformed Pi3 inputs raise clear errors

## New Training/Eval Commands

Added:

- `scripts/generate_dual_shoes_trajectory_commands.py`
- `scripts/generated_dual_shoes_trajectory_commands.sh`

Generated variants:

| variant | checkpoint tag | mode |
| --- | --- | --- |
| `delta_future` | `delta_future_seed0` | `future_target_mode=pi3_delta` |
| `changed_token_future` | `changed_token_future_seed0` | `future_target_mode=pi3_changed_tokens` |
| `delta_changed_token_future` | `delta_changed_token_future_seed0` | `future_target_mode=pi3_delta_changed_tokens` |

The generated script:

- preprocesses only if `data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr` is missing
- trains only if the corresponding checkpoint is missing
- evaluates only after checkpoint `200.ckpt` exists
- writes eval to `results_dual_shoes_50demo_trajectory_eval100`
- avoids old result root `results_eval50_full`

Training was not launched in this task. This keeps the current turn scoped to implementation, diagnostics, and reproducible command generation.

## Results Table

Added:

- `scripts/parse_dual_shoes_results.py`
- `reports/dual_shoes_trajectory_results.json`
- `reports/dual_shoes_trajectory_results_table.md`

Current status:

- Old eval50 rows are parsed with Wilson confidence intervals.
- Baseline eval100 rows are pending.
- Trajectory eval100 rows are pending.

See `reports/dual_shoes_trajectory_results_table.md` for the current combined table.

## Guardrails

This task did not:

- chase paper-main 100-demo reproduction
- run Place Empty Cup
- run segmentation replay
- modify zarr/HDF5/checkpoints/results in place
- implement true object-hand masks
- claim object-hand, triadic, or novelty results
- compare public-50 results to the paper main table

Current narrow claim:

> In the public 50-demo Place Dual Shoes setting, we can now test whether trajectory-coupled future Pi3 supervision is a better action-expert signal than full-scene future Pi3 supervision.

Recommended next command:

```bash
bash scripts/generated_dual_shoes_eval100_commands.sh
```

After the four baseline eval100 rows are stable, run:

```bash
bash scripts/generated_dual_shoes_trajectory_commands.sh
```
