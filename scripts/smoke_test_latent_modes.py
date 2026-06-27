#!/usr/bin/env python
"""Synthetic smoke tests for GAP latent modes."""
from __future__ import annotations

import os
import sys

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.policy.gap import GAPPolicy  # noqa: E402


def make_policy(**overrides):
    cfg = dict(
        shape_meta={"action": {"shape": [14]}},
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=8,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="sample",
        ),
        horizon=4,
        n_action_steps=4,
        n_obs_steps=1,
        num_inference_steps=2,
        dinov3_feature_dim=32,
        dinov3_num_views=1,
        use_pi3_features=True,
        pi3_feature_dim=32,
        pi3_num_views=1,
        pi3_embed_dim=32,
        state_dim=14,
        state_embed_dim=32,
        encoder_depth=1,
        encoder_heads=4,
        encoder_dim_feedforward=64,
        decoder_depth=1,
        decoder_heads=4,
        decoder_dim_feedforward=64,
        dinov3_grid_height=2,
        dinov3_grid_width=3,
        pi3_grid_height=4,
        pi3_grid_width=5,
        use_future_loss=False,
        future_target_mode="none",
        latent={"pi3_pool_grid": [2, 2], "pi3_num_compressed_tokens": 3, "pi3_random_num_tokens": 5},
    )
    cfg.update(overrides)
    return GAPPolicy(**cfg)


def make_obs(batch=2):
    return {
        "dinov3_features": torch.randn(batch, 1, 6, 32),
        "pi3_features": torch.randn(batch, 1, 20, 32),
        "agent_pos": torch.randn(batch, 14),
    }


def token_count(policy, obs):
    memory, _ = policy.encode_observations(obs)
    return memory.shape[1]


def expect_key_error(fn, text):
    try:
        fn()
    except KeyError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("Expected KeyError")


def main():
    obs = make_obs()

    full = make_policy(latent_mode="pi3_full")
    assert token_count(full, obs) == 1 + 1 + 6 + 20

    dino = make_policy(latent_mode="dino_only")
    assert token_count(dino, obs) == 1 + 1 + 6

    pooled = make_policy(latent_mode="pi3_pooled")
    assert token_count(pooled, obs) == 1 + 1 + 6 + 4

    compressed = make_policy(latent_mode="pi3_compressed")
    assert token_count(compressed, obs) == 1 + 1 + 6 + 3

    random_tokens = make_policy(latent_mode="pi3_random_tokens")
    assert token_count(random_tokens, obs) == 1 + 1 + 6 + 5

    dropout = make_policy(latent_mode="pi3_token_dropout", latent={"pi3_token_dropout_rate": 0.5})
    dropout.train()
    assert token_count(dropout, obs) == 1 + 1 + 6 + 20

    object_hand = make_policy(latent_mode="pi3_object_hand")
    expect_key_error(lambda: object_hand.encode_observations(obs), "pi3_object_hand_mask")

    background = make_policy(latent_mode="pi3_background")
    expect_key_error(lambda: background.encode_observations(obs), "pi3_background_mask")

    eef_region = make_policy(latent_mode="pi3_eef_region")
    expect_key_error(lambda: eef_region.encode_observations(obs), "pi3_eef_region_mask")
    obs_with_eef = dict(obs)
    obs_with_eef["pi3_eef_region_mask"] = torch.ones(2, 1, 20)
    assert token_count(eef_region, obs_with_eef) == 1 + 1 + 6 + 20

    non_eef_region = make_policy(latent_mode="pi3_non_eef_region")
    expect_key_error(lambda: non_eef_region.encode_observations(obs), "pi3_non_eef_region_mask")
    obs_with_non_eef = dict(obs)
    obs_with_non_eef["pi3_non_eef_region_mask"] = torch.ones(2, 1, 20)
    assert token_count(non_eef_region, obs_with_non_eef) == 1 + 1 + 6 + 20

    print("latent mode smoke tests passed")


if __name__ == "__main__":
    main()
