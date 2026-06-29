# Flow Matching Interaction Eval100 Summary

Generated: 2026-06-29

## Scope

This pass tested action-conditioned DINO interaction readout inside the current GAP action generator, plus an optional flow-matching objective that reuses the existing transformer decoder as a velocity field.

The key comparison requested was:

```text
flow + action_uv
vs
diffusion + action_uv
vs
flow + DINO-only
```

The intended claim would only be supported if `flow + action_uv` clearly beat both controls.

## Validation Gates

Synthetic smoke passed:

```text
python scripts/smoke_test_flow_matching_forward.py
flow-matching action-uv smoke test passed: loss=3.479350, flow_loss=1.187051, uv_loss=24.081011
```

Real zarr debug training passed on `place_dual_shoes`, `demo_clean`, 50 demos:

| debug run | final debug loss |
| --- | ---: |
| `flow + DINO-only` | 0.0004 |
| `flow + action_uv` | 0.0064 |

The originally requested short-train command omitted `setting=demo_clean`; the available local zarr is:

```text
data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr
```

so the real-zarr checks were run against `setting=demo_clean`.

## Formal Six-Way Training

All six variants trained for 200 epochs with batch size 256. Checkpoints exist at epoch 100 and 200.

| variant | epoch 200 train loss | last-20-epoch range |
| --- | ---: | ---: |
| diffusion + DINO-only | 0.0009 | 0.0009-0.0010 |
| diffusion + current_eef | 0.0009 | 0.0008-0.0009 |
| diffusion + action_uv | 0.0604 | 0.0589-0.0613 |
| flow + DINO-only | 0.0005 | 0.0005-0.0005 |
| flow + current_eef | 0.0005 | 0.0005-0.0005 |
| flow + action_uv | 0.0252 | 0.0244-0.0255 |

Important caveat: `action_uv` total loss includes UV auxiliary terms, so its absolute value is not directly comparable with DINO-only/current_eef. It is comparable across diffusion and flow action_uv runs because both include the same auxiliary supervision.

## Eval100 Results

Result root:

```text
results_dual_shoes_flow_interaction_eval100
```

Each variant produced 100 rollout videos under:

```text
results_dual_shoes_flow_interaction_eval100/place_dual_shoes/GAP/demo_clean/<ckpt_setting>/seed_0/200/
```

| variant | eval100 success | count |
| --- | ---: | ---: |
| diffusion + DINO-only | 0.02 | 2/100 |
| diffusion + current_eef | 0.04 | 4/100 |
| diffusion + action_uv | 0.03 | 3/100 |
| flow + DINO-only | 0.01 | 1/100 |
| flow + current_eef | 0.01 | 1/100 |
| flow + action_uv | 0.01 | 1/100 |

The eval100 logs contain no traceback, OOM, NaN, or runtime error matches.

## Interpretation

This result does **not** support the requested positive claim. `flow + action_uv` does not beat either key control:

- It is worse than `diffusion + action_uv`: `1/100` vs `3/100`.
- It ties `flow + DINO-only`: `1/100` vs `1/100`.

Therefore, on the current `place_dual_shoes`, 50-demo, seed-0, 200-epoch setup, the evidence does not show that flow intermediate actions induce a useful interaction field. The implementation is usable for further experiments, but the current result should be treated as a negative ablation, not a mechanism win.

The best model in this six-way pass is `diffusion + current_eef` at `4/100`, followed by `diffusion + action_uv` at `3/100`.

