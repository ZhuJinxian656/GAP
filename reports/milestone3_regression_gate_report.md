# Milestone 3 GAP Regression Gate Report

Date: 2026-06-30

## Objective

Answer three gating questions before interpreting the flow/action-UV ablation:

1. Can the current deploy/eval code re-evaluate the old DINO-only checkpoint near the old 24/100 eval100 baseline?
2. Can the current training code reproduce a reasonable DINO-only result with the old batch32 recipe?
3. After fixing flow hold-source normalization, does flow avoid collapse under a fair batch32 recipe?

## Executive Answer

| Question | Answer | Evidence |
| --- | --- | --- |
| Q1 old DINO-only re-eval | Yes. Current code can re-eval the old checkpoint at 24/100 in the final artifact. | `re_eval_old_dino_auto=24/100`, `re_eval_old_dino_model=24/100` |
| Q2 batch32 DINO-only reproduction | Yes. Fresh batch32 DINO-only reproduction reaches 19/100 in the final artifact. | `train_repro_dino_auto=19/100`; original job log also reached 21/100 |
| Q3 fair flow after hold-source fix | Yes, flow does not collapse. It is still below the best fair diffusion run in this seed. | `fair_flow_action_uv_auto=15/100`, `fair_flow_dino_only_auto=8/100` |

Conclusion: the earlier six-way batch256 ablation is confounded by recipe/update count and should not be used as method-level evidence. Flow is non-collapsed after the normalization fix, but this single-seed fair recipe still favors diffusion over flow.

## Code Changes

- `gap_policy/policy/gap.py`
  - Flow `source_mode=hold` now builds the source from raw `obs["agent_pos"]` and normalizes it with `self.normalizer["action"]`.
  - `source_mode=previous_action` now also normalizes raw previous-action sources with the action normalizer.
  - Training passes raw `batch["obs"]` into `_flow_source_actions`; inference stores raw observations in the context before normalization.
  - Added shape checks for flow source and reference action tensors.

- `deploy_policy.py`
  - Added `model_weight=auto|ema|model`.
  - `auto` preserves previous behavior: prefer EMA weights when present, otherwise use raw model weights.
  - Explicit `ema` or `model` now fails loudly if the checkpoint does not contain the requested key.

- `eval.sh`
  - Added fallback RobotWin root discovery for `../robotwin`.
  - Added `MODEL_WEIGHT` passthrough to deployment.

- Experiment scripts
  - `scripts/launch_gap_flow_interaction_ablation_tmux.sh` now defaults to `BATCH_SIZE=32` and warns that old batch256 six-way results are not comparable.
  - `scripts/launch_gap_flow_interaction_regression_gate.sh` provides a sequential regression gate.
  - `scripts/run_gap_flow_interaction_regression_job.sh` provides one tmux-friendly per-GPU gate job.
  - `scripts/watch_gap_regression_gate_eval100.sh` waits for parallel jobs and backfills missing eval100 artifacts.
  - `scripts/summarize_flow_interaction_regression_gate.py` writes a JSON and Markdown summary.
  - `scripts/smoke_test_flow_source_normalization.py` verifies that hold-source uses the action normalizer, not the agent_pos normalizer.

The regression job and watchdog now canonicalize `RESULTS_ROOT` to an absolute path after `cd ROOT_DIR`, so future evals do not accidentally write relative result paths under the RobotWin directory.

## Experimental Setup

Task: `place_dual_shoes`

Setting: `demo_clean`

Demos: `50`

Seed: `0`

Checkpoint epoch: `200`

Eval rollouts: `100`

Fair recipe batch size: `32`

For the old/new optimizer update comparison:

| Recipe | Updates per epoch | Total updates over 200 epochs |
| --- | ---: | ---: |
| batch32 old/fair recipe | 360 | 72000 |
| batch256 six-way recipe | 45 | 9000 |

The batch32 recipe therefore applies 8.0x more optimizer updates than the batch256 six-way ablation.

## Final Artifact Results

Primary result artifacts are in:

`/data1/home/zhu_jinxian/project/robotwin/results_flow_interaction_regression_gate`

They were summarized into:

- `reports/flow_interaction_regression_gate_report.md`
- `reports/flow_interaction_regression_gate_results.json`

| Stage | Eval100 success | Interpretation |
| --- | ---: | --- |
| `re_eval_old_dino_auto` | 24/100 | Current deploy/eval can recover the old DINO-only baseline level. |
| `re_eval_old_dino_model` | 24/100 | Explicit raw-model loading is also near the old 24/100 baseline. |
| `train_repro_dino_auto` | 19/100 | Batch32 DINO-only reproduction is reasonable. |
| `fair_diffusion_current_eef_auto` | 16/100 | Fair diffusion comparison point. |
| `fair_diffusion_action_uv_auto` | 19/100 | Best fair diffusion result in this gate. |
| `fair_flow_dino_only_auto` | 8/100 | Flow is low but not collapsed. |
| `fair_flow_action_uv_auto` | 15/100 | Best fair flow result; non-collapsed after the source normalization fix. |

Note: the original tmux logs and the final result artifacts differ slightly for some stochastic evals. The final `_result.txt` artifacts above are the canonical current values. The original logs still corroborate the same conclusions: old raw-model re-eval reached 25/100, batch32 DINO-only reached 21/100, fair flow action-UV reached 15/100, and fair flow DINO-only reached 8/100.

## Validation

Completed checks:

- `python -m py_compile deploy_policy.py gap_policy/policy/gap.py scripts/summarize_flow_interaction_regression_gate.py scripts/smoke_test_flow_source_normalization.py`
- `python scripts/smoke_test_flow_source_normalization.py`
- `bash -n eval.sh`
- `bash -n scripts/launch_gap_flow_interaction_regression_gate.sh`
- `bash -n scripts/run_gap_flow_interaction_regression_job.sh`
- `bash -n scripts/watch_gap_regression_gate_eval100.sh`

Runtime checks:

- All seven parallel gate jobs exited.
- No remaining GPU compute processes after stopping a redundant watchdog backfill.
- Error scan over gate logs found no `Traceback`, OOM, `Killed`, or NaN markers.

## Recommended Next Analysis

- Do not treat the batch256 six-way negative result as method evidence.
- Compare methods using the fair batch32 gate or rerun the full grid with a matched update budget.
- If time allows, rerun the fair gate over multiple seeds because the eval100 values show normal stochastic variance.
- Investigate why fair diffusion action-UV remains above fair flow action-UV on this seed, now that the flow source normalization bug is fixed.
