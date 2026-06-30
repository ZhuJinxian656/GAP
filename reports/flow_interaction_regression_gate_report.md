# Flow Interaction Regression Gate Report

## Existing Eval100 Baselines

| variant | success | result |
| --- | ---: | --- |
| vanilla | 8/100 (0.08) | `results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean/seed_0/200/_result.txt` |
| dino_only | 24/100 (0.24) | `results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt` |
| pi3_pooled | 16/100 (0.16) | `results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_pooled_seed0/seed_0/200/_result.txt` |
| pi3_eef_region | 18/100 (0.18) | `results_dual_shoes_50demo_eval100/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_eef_region_seed0/seed_0/200/_result.txt` |

## Latest Six-Way Eval100

| variant | success |
| --- | ---: |
| diffusion + DINO-only | 2/100 (0.02) |
| diffusion + current_eef | 4/100 (0.04) |
| diffusion + action_uv | 3/100 (0.03) |
| flow + DINO-only | 1/100 (0.01) |
| flow + current_eef | 1/100 (0.01) |
| flow + action_uv | 1/100 (0.01) |

## Decision Gate Status

| question | status |
| --- | --- |
| Q1 old DINO-only current eval near 24/100 | answered yes: old DINO-only re-eval is close enough at 24/100. |
| Q2 fresh batch32 DINO-only reproduction | answered yes: batch32 DINO-only recovered to 19/100; batch256 ablation is confounded. |
| Q3 fixed flow source under fair recipe | answered with current fair flow evidence: best flow gate is fair_flow_action_uv_clean_only_auto at 17/100; current-EEF control is 14/100. |

## Milestone 3D Candidate-UV Context

- `expert_final` supervises every intermediate/noised action candidate toward the final expert action-aligned UV, so it is not fully candidate-consistent for flow matching.
- `flow_interp` uses the flow lambda to interpolate from current EEF UV to expert final UV, which is a better approximation of the candidate interaction field when true FK-derived candidate UV is unavailable.
- `fair_flow_current_eef_auto` is required to separate gains from dynamic action-conditioned UV from gains that any EEF-local DINO token gives to flow.

## Regression Gate Results

| stage | success | result |
| --- | ---: | --- |
| fair_diffusion_action_uv_auto | 19/100 (0.19) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_diffusion_action_uv_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_diffusion_action_uv_seed0/seed_0/200/_result.txt` |
| fair_diffusion_current_eef_auto | 16/100 (0.16) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_diffusion_current_eef_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_diffusion_current_eef_seed0/seed_0/200/_result.txt` |
| fair_flow_action_uv_auto | 15/100 (0.15) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_action_uv_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_action_uv_seed0/seed_0/200/_result.txt` |
| fair_flow_action_uv_clean_only_auto | 17/100 (0.17) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_action_uv_clean_only_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_action_uv_clean_only_seed0/seed_0/200/_result.txt` |
| fair_flow_action_uv_flow_interp_auto | 10/100 (0.10) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_action_uv_flow_interp_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_action_uv_flow_interp_seed0/seed_0/200/_result.txt` |
| fair_flow_action_uv_none_auto | 10/100 (0.10) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_action_uv_none_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_action_uv_none_seed0/seed_0/200/_result.txt` |
| fair_flow_current_eef_auto | 14/100 (0.14) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_current_eef_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_current_eef_seed0/seed_0/200/_result.txt` |
| fair_flow_dino_only_auto | 8/100 (0.08) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/fair_flow_dino_only_auto/place_dual_shoes/GAP/demo_clean/demo_clean_fair_flow_dino_only_seed0/seed_0/200/_result.txt` |
| re_eval_old_dino_auto | 24/100 (0.24) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/re_eval_old_dino_auto/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt` |
| re_eval_old_dino_model | 24/100 (0.24) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/re_eval_old_dino_model/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt` |
| train_repro_dino_auto | 19/100 (0.19) | `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate/train_repro_dino_auto/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_repro_seed0/seed_0/200/_result.txt` |

## Optimizer Update Estimate

- zarr total steps: 11510 (zarr)
- old/fair batch32: 360 updates/epoch, 72000 updates over 200 epochs
- latest batch256: 45 updates/epoch, 9000 updates over 200 epochs
- update ratio: batch32 is 8.0x batch256

## Regression Diagnosis

- Old DINO-only re-eval is close to the 24/100 baseline (24/100), so deploy/eval code is probably okay.
- Batch32 DINO-only reproduction recovered (19/100), so the batch256 ablation is not a fair method result.
- Best fair flow gate result so far is fair_flow_action_uv_clean_only_auto at 17/100.

## Recommendation

Treat the batch256 six-way result as a recipe confound, not a method-level conclusion. Use the fair batch32 gate as the comparison point for further flow/action-UV analysis; after hold-source normalization, flow is non-collapsed but still below the best diffusion fair-recipe result in this seed.
