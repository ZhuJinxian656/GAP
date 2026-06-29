#!/usr/bin/env python
"""Tiny forward/loss smoke test for current-EEF DINO interaction tokens."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402
from gap_policy.policy.gap import GAPPolicy  # noqa: E402


def identity_normalizer():
    scale = np.array([1.0], dtype=np.float32)
    offset = np.array([0.0], dtype=np.float32)
    stat = {
        "min": np.array([0.0], dtype=np.float32),
        "max": np.array([1.0], dtype=np.float32),
        "mean": np.array([0.0], dtype=np.float32),
        "std": np.array([1.0], dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stat)


def main() -> int:
    device = torch.device("cpu")
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
    policy = GAPPolicy(
        shape_meta={"action": {"shape": [14]}},
        noise_scheduler=scheduler,
        horizon=2,
        n_action_steps=2,
        n_obs_steps=1,
        num_inference_steps=1,
        dinov3_feature_dim=16,
        dinov3_num_views=1,
        use_pi3_features=False,
        use_future_loss=False,
        future_target_mode="none",
        state_dim=14,
        state_embed_dim=16,
        encoder_depth=1,
        encoder_heads=4,
        encoder_dim_feedforward=32,
        decoder_depth=1,
        decoder_heads=4,
        decoder_dim_feedforward=32,
        dinov3_grid_height=15,
        dinov3_grid_width=20,
        use_interaction_field=True,
        interaction_field={
            "mode": "current_eef",
            "readout": "masked_pool",
            "grid_height": 15,
            "grid_width": 20,
            "use_pair_token": True,
            "append_tokens": True,
        },
    ).to(device)

    batch_size = 2
    num_tokens = 15 * 20
    obs = {
        "dinov3_features": torch.randn(batch_size, 1, num_tokens, 16, device=device),
        "agent_pos": torch.randn(batch_size, 14, device=device),
        "dino_eef_uv": torch.zeros(batch_size, 1, 2, 2, device=device),
        "dino_eef_valid": torch.ones(batch_size, 1, 2, device=device),
        "dino_eef_region_mask": torch.zeros(batch_size, 1, 2, num_tokens, device=device),
        "dino_pair_region_mask": torch.zeros(batch_size, 1, num_tokens, device=device),
    }
    obs["dino_eef_uv"][:, :, 0, :] = torch.tensor([3.0, 4.0], device=device)
    obs["dino_eef_uv"][:, :, 1, :] = torch.tensor([12.0, 9.0], device=device)
    obs["dino_eef_region_mask"][:, :, 0, :12] = 1.0
    obs["dino_eef_region_mask"][:, :, 1, 120:136] = 1.0
    obs["dino_pair_region_mask"][:, :, 60:90] = 1.0

    batch = {
        "obs": obs,
        "action": torch.randn(batch_size, 2, 14, device=device),
    }

    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.randn(8, 2, 14, device=device),
            "agent_pos": torch.randn(8, 14, device=device),
        },
        last_n_dims=1,
    )
    for key in (
        "dinov3_features",
        "dino_eef_uv",
        "dino_eef_valid",
        "dino_eef_region_mask",
        "dino_pair_region_mask",
    ):
        normalizer[key] = identity_normalizer()
    policy.set_normalizer(normalizer)

    loss, loss_dict = policy.compute_loss(batch)
    assert torch.isfinite(loss), loss_dict
    print(f"interaction-field forward smoke test passed: loss={loss.item():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
