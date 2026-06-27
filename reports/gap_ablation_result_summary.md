# GAP Interface Ablation Result Summary

Generated: 2026-06-27 18:42:28

Task: `place_dual_shoes`  Config: `demo_clean`  Demos: `50`  Seed: `0`  Checkpoint: `200`

| variant        | checkpoint | latest epoch/loss | eval success | result file                                                               | question                                                  |
| -------------- | ---------- | ----------------- | ------------ | ------------------------------------------------------------------------- | --------------------------------------------------------- |
| vanilla        | ready      | not found         | 0.300        | results/place_dual_shoes/GAP/demo_clean/demo_clean/seed_0/200/_result.txt | Full Pi3 scene tokens plus full future Pi3 latent target. |
| dino_only      | pending    | 87 / 0.0004       | pending      | pending                                                                   | No Pi3 observation and no future latent supervision.      |
| no_future      | pending    | 42 / 0.0013       | pending      | pending                                                                   | Full Pi3 observation, but future latent loss disabled.    |
| pi3_pooled     | pending    | 49 / 0.0014       | pending      | pending                                                                   | Pooled Pi3 tokens and pooled future target.               |
| pi3_compressed | pending    | 1 / 0.1915        | pending      | pending                                                                   | Compressed Pi3 bottleneck and compressed future target.   |
| pi3_random     | pending    | 1 / 0.1569        | pending      | pending                                                                   | Random Pi3 token subset and random-token future target.   |
| pi3_dropout    | pending    | 1 / 0.1594        | pending      | pending                                                                   | Full future Pi3 target with observation token dropout.    |

## Interpretation Guardrails

- `dino_only` tests whether any Pi3/future-latent signal matters beyond DINOv3 plus proprioception.
- `no_future` tests whether future Pi3 supervision matters when full Pi3 observation tokens remain.
- `pi3_pooled`, `pi3_compressed`, `pi3_random`, and `pi3_dropout` test whether dense full-scene Pi3 token structure is necessary.
- Current zarr/HDF5 data does not include object masks, hand masks, object pose, contact labels, or non-empty point clouds. Therefore these runs cannot directly prove an object-hand-interaction-region mechanism; they can only show whether full-scene dense geometry appears necessary.

## Current Readout

- Vanilla baseline success rate is `0.300`.
- Pending ablation evals: `dino_only, no_future, pi3_pooled, pi3_compressed, pi3_random, pi3_dropout`.
