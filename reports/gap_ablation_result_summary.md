# GAP Interface Ablation Result Summary

Generated: 2026-06-28 14:30:20

Task: `place_dual_shoes`  Config: `demo_clean`  Demos: `50`  Seed: `0`  Checkpoint: `200`

Rollouts per variant: `10`

Results dir: `results`

| variant            | checkpoint | latest epoch/loss | successes | eval success | delta vs vanilla | result file                                                                                        | question                                                            |
| ------------------ | ---------- | ----------------- | --------- | ------------ | ---------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| vanilla            | 200 ready  | not found         | 3/10      | 0.300        | +0.000           | results/place_dual_shoes/GAP/demo_clean/demo_clean/seed_0/200/_result.txt                          | Full Pi3 scene tokens plus full future Pi3 latent target.           |
| dino_only          | 200 ready  | 200 / 0.0000      | 2/10      | 0.200        | -0.100           | results/place_dual_shoes/GAP/demo_clean/demo_clean_dino_only_seed0/seed_0/200/_result.txt          | No Pi3 observation and no future latent supervision.                |
| no_future          | 200 ready  | 200 / 0.0000      | 0/10      | 0.000        | -0.300           | results/place_dual_shoes/GAP/demo_clean/demo_clean_no_future_seed0/seed_0/200/_result.txt          | Full Pi3 observation, but future latent loss disabled.              |
| pi3_pooled         | 200 ready  | 200 / 0.0000      | 2/10      | 0.200        | -0.100           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_pooled_seed0/seed_0/200/_result.txt         | Pooled Pi3 tokens and pooled future target.                         |
| pi3_compressed     | 200 ready  | 200 / 0.0001      | 0/10      | 0.000        | -0.300           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_compressed_seed0/seed_0/200/_result.txt     | Compressed Pi3 bottleneck and compressed future target.             |
| pi3_random         | 200 ready  | 200 / 0.0001      | 1/10      | 0.100        | -0.200           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_random_seed0/seed_0/200/_result.txt         | Random Pi3 token subset and random-token future target.             |
| pi3_dropout        | 200 ready  | 200 / 0.0002      | 0/10      | 0.000        | -0.300           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_dropout_seed0/seed_0/200/_result.txt        | Full future Pi3 target with observation token dropout.              |
| pi3_eef_region     | 200 ready  | 200 / 0.0001      | 2/10      | 0.200        | -0.100           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_eef_region_seed0/seed_0/200/_result.txt     | EEF-projected hand-near Pi3 token proxy and matching future target. |
| pi3_non_eef_region | 200 ready  | 200 / 0.0002      | 1/10      | 0.100        | -0.200           | results/place_dual_shoes/GAP/demo_clean/demo_clean_pi3_non_eef_region_seed0/seed_0/200/_result.txt | Complement of the EEF-projected hand-near Pi3 proxy region.         |

## Interpretation Guardrails

- `dino_only` tests whether any Pi3/future-latent signal matters beyond DINOv3 plus proprioception.
- `no_future` tests whether future Pi3 supervision matters when full Pi3 observation tokens remain.
- `pi3_pooled`, `pi3_compressed`, `pi3_random`, and `pi3_dropout` test whether dense full-scene Pi3 token structure is necessary.
- `pi3_eef_region` and `pi3_non_eef_region` use projected end-effector positions as a hand-near proxy/control. They are useful directional probes, not true object-hand masks.
- Current zarr/HDF5 data does not include object masks, hand masks, object pose, contact labels, or non-empty point clouds. Therefore these runs cannot directly prove an object-hand-interaction-region mechanism; they can only show whether full-scene dense geometry appears necessary and whether an EEF-near proxy is competitive.

## Current Readout

- Vanilla baseline success rate is `0.300` (3/10).
- All configured ablation eval results are present.
- Best observed variant is `vanilla` at `0.300` (3/10).
- `dino_only` below vanilla is consistent with Pi3/future-latent helping, but by itself does not identify whether the useful signal is full-scene geometry.
- `no_future` below vanilla suggests future-latent supervision may matter within the Pi3 setup, but this is not enough to prove a full-scene geometry mechanism.
- `pi3_eef_region` and `pi3_non_eef_region` are close, so the EEF-near proxy is only a weak directional signal in this single-task run.
- `pi3_pooled` remaining competitive weakens a strict dense full-scene token requirement.
- This is still one task, one seed, and 50 demos; it is useful evidence for experiment direction, not a paper-scale conclusion.
