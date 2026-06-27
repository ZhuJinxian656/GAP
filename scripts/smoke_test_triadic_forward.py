#!/usr/bin/env python
"""
Tiny forward/loss smoke test for vanilla and triadic GAP policy variants.

This script skips cleanly if torch/diffusers are not installed.
"""
from __future__ import annotations

import os
import sys

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)


def main() -> int:
    try:
        import torch
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from gap_policy.policy.gap import GAPPolicy
        from gap_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
    except ImportError as exc:
        print(f"SKIP: missing dependency for GAP forward smoke test: {exc}")
        return 0

    device = torch.device("cpu")
    shape_meta = {"action": {"shape": [14]}}
    scheduler = DDIMScheduler(
        num_train_timesteps=4,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="sample",
    )

    common_kwargs = dict(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=2,
        n_action_steps=2,
        n_obs_steps=1,
        num_inference_steps=1,
        dinov3_feature_dim=16,
        use_pi3_features=False,
        state_dim=14,
        state_embed_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        encoder_dim_feedforward=32,
        decoder_depth=1,
        decoder_heads=4,
        decoder_dim_feedforward=32,
    )

    def make_batch(include_triadic: bool):
        obs = {
            "dinov3_features": torch.randn(2, 1, 300, 16),
            "agent_pos": torch.randn(2, 14),
        }
        normalizer_data = {
            "action": torch.randn(8, 2, 14),
            "agent_pos": torch.randn(8, 14),
        }
        if include_triadic:
            obs["triadic_state"] = torch.randn(2, 11)
            normalizer_data["triadic_state"] = torch.randn(8, 11)
        batch = {
            "obs": obs,
            "action": torch.randn(2, 2, 14),
        }
        return batch, normalizer_data

    for use_triadic in (False, True):
        kwargs = dict(common_kwargs)
        kwargs["use_triadic_token"] = use_triadic
        kwargs["triadic_mode"] = "triadic" if use_triadic else "disabled"
        policy = GAPPolicy(**kwargs).to(device)
        batch, normalizer_data = make_batch(use_triadic)
        normalizer = LinearNormalizer()
        normalizer.fit(normalizer_data, last_n_dims=1)
        normalizer["dinov3_features"] = SingleFieldLinearNormalizer.create_identity()
        policy.set_normalizer(normalizer)
        loss, loss_dict = policy.compute_loss(batch)
        assert torch.isfinite(loss), loss_dict
        print(f"ok use_triadic={use_triadic}: loss={loss.item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
