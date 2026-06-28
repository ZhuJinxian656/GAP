# Dual Shoes Public-50 Eval100 Trajectory Analysis

Generated: 2026-06-28 23:12 CST

Scope:

- Task: `place_dual_shoes`
- Config: `demo_clean`
- Training demos: `50`
- Train seed: `0`
- Checkpoint: `200`
- Eval rollouts: `100`
- This is public-50 mechanism evidence, not paper-main 100-demo reproduction evidence.

## Question

The current mechanism question was:

> Should future Pi3 supervision predict the complete future scene, or should it supervise the action-induced change?

The working idea was that an action expert may need a trajectory-coupled future target. If so, `pi3_delta` or changed-token future supervision should outperform the original full-scene future Pi3 target.

## Eval100 Results

| variant | success | rate | readout |
| --- | ---: | ---: | --- |
| `dino_only` | 24/100 | 0.24 | Best result; Pi3/future branch is not necessary in this public-50 run. |
| `pi3_eef_region` | 18/100 | 0.18 | Best Pi3-based variant; EEF proxy is a useful but imperfect spatial bias. |
| `pi3_pooled` | 16/100 | 0.16 | Dense full-scene Pi3 tokens are not required. |
| `changed_token_future` | 9/100 | 0.09 | Only +1 over vanilla; not meaningful at n=100. |
| `pi3_compressed` | 9/100 | 0.09 | Roughly vanilla-level. |
| `vanilla` | 8/100 | 0.08 | Full Pi3 scene tokens plus full future Pi3 target. |
| `no_future` | 8/100 | 0.08 | Same as vanilla in eval100. |
| `delta_future` | 8/100 | 0.08 | Same as vanilla; delta target did not help. |
| `pi3_dropout` | 8/100 | 0.08 | Same as vanilla. |
| `pi3_non_eef_region` | 8/100 | 0.08 | Same as vanilla; below EEF proxy. |
| `pi3_random` | 6/100 | 0.06 | Below vanilla. |

Structured outputs:

- `reports/dual_shoes_trajectory_results.json`
- `reports/dual_shoes_trajectory_results_table.md`
- `reports/dual_shoes_eval100_existing_baselines.json`

Result roots:

- `results_dual_shoes_50demo_eval100`
- `results_dual_shoes_50demo_trajectory_eval100`

## Direct Answer

This eval100 pass does **not** support the narrow hypothesis that the current trajectory-coupled future targets are better supervision than the original full-scene future target.

- Full-scene future target, `vanilla`: `8/100`.
- Delta target, `delta_future`: `8/100`.
- Changed-token full-future target, `changed_token_future`: `9/100`.
- No future loss, `no_future`: `8/100`.

At 100 rollouts, `8/100` vs `9/100` is noise-level evidence. The Wilson intervals in `reports/dual_shoes_trajectory_results_table.md` overlap strongly. The result is best read as: these two action-change proxies did not produce a policy-level gain in this run.

## Mechanism Readout

The diagnostics still say the target has action-coupled signal:

- Future copyability diagnostic: high-change Pi3 tokens are sparse, with changed-token ratio `0.10` at the 90th percentile.
- Action-coupling diagnostic: adding the expert action chunk improved pooled-delta prediction by about `16%` relative MSE.

But that signal did not translate into eval improvement for the current auxiliary-loss implementation. The likely reason is that "feature delta is action-coupled" is weaker than "feature delta is the right policy supervision target." A raw delta target may remove absolute object/scene information that the policy still needs, while changed-token selection may identify feature motion or viewpoint sensitivity rather than true task-causal object-hand change.

The stronger positive signal is spatial, not temporal-delta:

- `pi3_eef_region`: `18/100`
- `pi3_non_eef_region`: `8/100`
- `pi3_pooled`: `16/100`
- `vanilla`: `8/100`

This suggests that reducing or biasing the Pi3 future target can help, but the useful reduction is more likely an interaction-region or bottleneck effect than the raw delta/changed-token objective tested here.

The strongest negative control remains:

- `dino_only`: `24/100`

That means the full Pi3/future branch is not necessary for the best observed policy in this public-50 single-seed regime. It may even be adding optimization noise or over-regularization relative to DINO-only features.

## Updated Hypothesis

The initial direction should be refined:

> The issue is probably not simply "full future scene vs action-induced feature change." The more promising target is "future/action supervision localized to true task-causal object-hand regions, while preserving enough absolute geometry for phase and placement."

In other words, trajectory coupling is still conceptually reasonable, but the current proxies are too crude:

- `pi3_delta` may discard useful absolute geometry.
- `pi3_changed_tokens` may select high feature-delta tokens that are not truly object-hand causal.
- EEF masking is spatially closer to the manipulation process, but it is still only an end-effector projection proxy.

## Training Notes

The two new variants were trained as simultaneous 4-GPU DDP jobs:

| variant | GPUs | per-GPU batch | global batch | final train signal |
| --- | --- | ---: | ---: | --- |
| `delta_future_seed0` | 0-3 | 32 | 128 | action loss `0.00018`, future loss `0.00908` |
| `changed_token_future_seed0` | 4-7 | 32 | 128 | action loss `0.00018`, future loss `0.01549` |

Both reached low training action loss, so the eval result is not an obvious training-crash artifact.

## Recommended Next Steps

1. Do not present `pi3_delta` or `pi3_changed_tokens` as supporting evidence for trajectory-coupled future supervision; the eval100 result does not support that.
2. Treat `delta_changed_token_future` as optional. It is implemented, but current evidence makes it a small follow-up rather than the main path.
3. Prioritize true object/robot/hand masks from RoboTwin/SAPIEN replay. The `pi3_eef_region` result makes real interaction-region supervision the most promising next mechanism test.
4. Test hybrid targets instead of pure delta replacement: absolute future target on an interaction mask plus a normalized delta auxiliary loss.
5. Repeat the key variants across more seeds/tasks before turning this into a claim: `dino_only`, `vanilla`, `no_future`, `pi3_pooled`, `pi3_eef_region`, true object-hand region, and true non-object-hand region.

## Current Claim

For public-50 `place_dual_shoes/demo_clean`, seed 0:

> Full-scene future Pi3 supervision is not the best observed mechanism, but the current action-change proxy targets also do not improve over it. The evidence points more toward spatially restricted or bottlenecked supervision, ideally with true object-hand masks, than toward raw future-delta supervision.
