# GAP Mechanism Readout

Generated: 2026-06-28

Question:

> Is GAP's Pi3/future latent mainly using full-scene geometry, or is the useful signal closer to a hand-object interaction region?

Current answer:

The current single-task experiment does **not** support a strong full-scene geometry explanation. It also does **not** prove a true hand-object interaction-region explanation, because the available data does not contain object masks, robot/hand masks, object pose, object keypoints, contact labels, depth, or usable point clouds.

The strongest defensible conclusion is:

> On `place_dual_shoes` with 50 demos and seed 0, full-scene dense Pi3/future latent is not necessary for the best observed eval score. Future-latent supervision may still help inside Pi3-based policies, but the current EEF-region proxy is too weak to identify a true interaction-region mechanism.

## Evidence

Primary eval:

- Task: `place_dual_shoes`
- Config: `demo_clean`
- Training demos: `50`
- Train seed: `0`
- Checkpoint: `200`
- Eval rollouts per variant: `50`
- Result root: `results_eval50_full`
- Summary: `reports/gap_ablation_eval50_summary.md`

Results:

| variant            | success | count | delta vs vanilla | mechanism probe |
| ------------------ | ------- | ----- | ---------------- | --------------- |
| vanilla            | 0.08    | 4/50  | +0.00            | Full Pi3 scene tokens plus full future Pi3 target |
| dino_only          | 0.16    | 8/50  | +0.08            | No Pi3 observation and no future latent |
| no_future          | 0.02    | 1/50  | -0.06            | Full Pi3 observation, no future latent loss |
| pi3_pooled         | 0.12    | 6/50  | +0.04            | Pooled Pi3 tokens and pooled future target |
| pi3_compressed     | 0.04    | 2/50  | -0.04            | Compressed Pi3 bottleneck |
| pi3_random         | 0.00    | 0/50  | -0.08            | Random Pi3 token subset |
| pi3_dropout        | 0.08    | 4/50  | +0.00            | Full future Pi3 target with observation dropout |
| pi3_eef_region     | 0.12    | 6/50  | +0.04            | EEF-projected hand-near token proxy |
| pi3_non_eef_region | 0.10    | 5/50  | +0.02            | Complement of EEF-projected proxy |

## Full-Scene Geometry Hypothesis

Prediction if full-scene dense Pi3/future latent is the main mechanism:

- `vanilla` should outperform `dino_only`.
- Dense full Pi3 should outperform pooled/compressed/randomized token variants.
- `pi3_non_eef_region` should not be competitive if useful tokens are mainly localized around the hands.

Observed:

- `dino_only` is higher than vanilla: `8/50` vs `4/50`.
- `pi3_pooled` is higher than vanilla: `6/50` vs `4/50`.
- `pi3_dropout` matches vanilla: `4/50` vs `4/50`.
- `pi3_non_eef_region` is close to `pi3_eef_region`: `5/50` vs `6/50`.

Readout:

This does not support the claim that dense full-scene Pi3/future latent is the dominant useful signal in the current run. In fact, the best observed variant is `dino_only`, which removes Pi3 and future-latent supervision entirely.

## Future-Latent Hypothesis

Prediction if future-latent supervision matters inside the Pi3 setup:

- `no_future` should be worse than vanilla.

Observed:

- `no_future` is lower than vanilla: `1/50` vs `4/50`.

Readout:

This is consistent with future-latent supervision helping when Pi3 is used. However, because `dino_only` beats vanilla, this cannot be interpreted as proof that future Pi3 geometry is the central mechanism. It may be a weak regularizer, an optimization effect, or noise under this single-task setting.

## Interaction-Region Hypothesis

Prediction if hand-object interaction-region tokens are the useful part:

- A true object-hand region mask should beat background/non-interaction masks.
- The EEF-near proxy should beat its complement by a clear margin.

Observed:

- `pi3_eef_region` is only one success above `pi3_non_eef_region`: `6/50` vs `5/50`.
- The available supervision is only an EEF projection, not an object-hand mask.

Readout:

The current EEF proxy gives only a weak directional signal. It does not prove that GAP is using the hand-object interaction region. The data needed for a true object-centric test is missing.

## Missing Data For A True Proof

Current local HDF5/zarr inspection shows:

- object masks: missing
- robot/hand masks: missing
- segmentation labels: missing
- object IDs/names: missing
- object pose/keypoints: missing
- contact labels: missing
- depth maps: missing
- usable point clouds: missing
- EEF poses and camera calibration: present

Therefore, the current experiments can test whether dense full-scene Pi3 is necessary, but cannot directly test object-hand interaction masks.

## Next Experiments

Recommended next steps, in priority order:

1. Add a true object/robot mask source from RoboTwin/SAPIEN rendering, simulator actor poses, or segmentation buffers.
2. Build token masks for object, robot hand, object-hand union/intersection, non-interaction background, and EEF proxy.
3. Re-run the same ablation on at least 3 tasks and 3 seeds, preferably with 100 demos and 300 epochs for closer paper comparability.
4. Evaluate with at least 50 rollouts per variant; use 100 or 200 if runtime allows.
5. Compare `object_hand_region` vs `non_object_hand_region`, not just EEF vs non-EEF.
6. Keep `dino_only`, `no_future`, and `pi3_pooled` as controls, because they were the most informative negative controls in this run.

Practical implication:

For the next paper-reproduction phase, it is probably more valuable to obtain real object/robot masks or simulator object poses than to train more full-scene GAP variants on the same single task. The current result already weakens the full-scene story; the missing piece is a true interaction-region intervention.
