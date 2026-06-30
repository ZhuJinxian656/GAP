# Milestone 3D UV Mode Sweep Report

Task: `place_dual_shoes`
Setting: `demo_clean`
Demos: 50
Seed: 0
Epochs: 200
Eval rollouts: 100
Recipe: fair batch32
Results root: `/data1/home/zhu_jinxian/project/GAP/results_flow_interaction_regression_gate`

## Results

| variant | eval100 | status |
| --- | ---: | --- |
| fair_flow_dino_only_auto | 8/100 | existing |
| fair_flow_current_eef_auto | 14/100 | new |
| fair_flow_action_uv_auto / expert_final | 15/100 | existing equivalent |
| fair_flow_action_uv_flow_interp_auto | 10/100 | new |
| fair_flow_action_uv_clean_only_auto | 17/100 | new |
| fair_flow_action_uv_none_auto | 10/100 | new |
| fair_diffusion_action_uv_auto | 19/100 | existing |

## Interpretation Rules

1. If flow_current_eef ~= flow_action_uv_*: dynamic action-conditioned field is not yet adding beyond static EEF-local tokens.
2. If flow_interp > expert_final: candidate-consistent UV supervision helps and supports the A_lambda-induced interaction-field story.
3. If clean_only >= flow_interp: mid-action UV loss may be unnecessary or harmful; ActionToUVHead mainly needs clean FK-like calibration.
4. If none >= clean_only: UV auxiliary losses may be hurting action optimization; interaction readout should be trained by action loss only.
5. If flow_action_uv_* > flow_current_eef and > flow_dino_only: there is positive evidence for action-conditioned interaction field in flow.

## Readout

In this seed, clean_only is the best flow/action_uv variant at 17/100, beating expert_final at 15/100, current_eef at 14/100, and fair_flow_dino_only at 8/100. flow_interp and none both land at 10/100.

This supports rule 3 for the current run: the clean UV calibration appears more helpful than mid-action interpolated UV supervision. The action-conditioned interaction field still has positive evidence through clean_only because it beats both flow_current_eef and flow_dino_only, but the candidate-consistent interpolation did not improve this seed.
