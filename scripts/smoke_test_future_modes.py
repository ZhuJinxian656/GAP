#!/usr/bin/env python
"""Synthetic smoke tests for future latent target modes."""
from __future__ import annotations

import os
import sys

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import numpy as np

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.policy.gap import GAPPolicy  # noqa: E402
from gap_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer  # noqa: E402


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
        latent_mode="pi3_full",
        latent={"pi3_pool_grid": [2, 2], "pi3_num_compressed_tokens": 3, "pi3_random_num_tokens": 5},
        future={"pi3_pool_grid": [2, 2], "pi3_num_compressed_tokens": 3},
        future_loss_weight=0.1,
    )
    cfg.update(overrides)
    return GAPPolicy(**cfg)


def make_batch(batch=2):
    return {
        "obs": {
            "dinov3_features": torch.randn(batch, 1, 6, 32),
            "pi3_features": torch.randn(batch, 1, 20, 32),
            "agent_pos": torch.randn(batch, 14),
        },
        "action": torch.randn(batch, 4, 14),
        "future_pi3_features": torch.randn(batch, 1, 20, 32),
    }


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


def set_test_normalizer(policy, batch):
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": batch["action"].detach().cpu().numpy(),
            "agent_pos": batch["obs"]["agent_pos"].detach().cpu().numpy(),
        },
        last_n_dims=1,
    )
    normalizer["dinov3_features"] = identity_normalizer()
    normalizer["pi3_features"] = identity_normalizer()
    policy.set_normalizer(normalizer)


def assert_loss(policy, batch, future_enabled):
    set_test_normalizer(policy, batch)
    loss, loss_dict = policy.compute_loss(batch)
    assert torch.isfinite(loss)
    assert "action_loss" in loss_dict
    assert "total_loss" in loss_dict
    if future_enabled:
        assert loss_dict["future_loss"] > 0
    else:
        assert loss_dict["future_loss"] == 0.0


def expect_error(fn, exc_type, text):
    try:
        fn()
    except exc_type as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def main():
    batch = make_batch()

    no_future = make_policy(use_future_loss=False, future_target_mode="none")
    no_future_batch = {"obs": batch["obs"], "action": batch["action"]}
    assert_loss(no_future, no_future_batch, future_enabled=False)

    full = make_policy(use_future_loss=True, future_target_mode="pi3_full")
    assert_loss(full, batch, future_enabled=True)

    pooled = make_policy(use_future_loss=True, future_target_mode="pi3_pooled")
    assert_loss(pooled, batch, future_enabled=True)

    compressed = make_policy(use_future_loss=True, future_target_mode="pi3_compressed")
    assert_loss(compressed, batch, future_enabled=True)

    random_tokens = make_policy(use_future_loss=True, future_target_mode="pi3_random_tokens")
    assert_loss(random_tokens, batch, future_enabled=True)

    object_hand = make_policy(use_future_loss=True, future_target_mode="pi3_object_hand")
    set_test_normalizer(object_hand, batch)
    expect_error(lambda: object_hand.compute_loss(batch), KeyError, "future_pi3_object_hand_mask")

    background = make_policy(use_future_loss=True, future_target_mode="pi3_background")
    set_test_normalizer(background, batch)
    expect_error(lambda: background.compute_loss(batch), KeyError, "future_pi3_background_mask")

    eef_region = make_policy(use_future_loss=True, future_target_mode="pi3_eef_region")
    set_test_normalizer(eef_region, batch)
    expect_error(lambda: eef_region.compute_loss(batch), KeyError, "future_pi3_eef_region_mask")
    eef_batch = {
        "obs": batch["obs"],
        "action": batch["action"],
        "future_pi3_features": batch["future_pi3_features"],
        "future_pi3_eef_region_mask": torch.ones(2, 1, 20),
    }
    assert_loss(eef_region, eef_batch, future_enabled=True)

    non_eef_region = make_policy(use_future_loss=True, future_target_mode="pi3_non_eef_region")
    set_test_normalizer(non_eef_region, batch)
    expect_error(lambda: non_eef_region.compute_loss(batch), KeyError, "future_pi3_non_eef_region_mask")
    non_eef_batch = {
        "obs": batch["obs"],
        "action": batch["action"],
        "future_pi3_features": batch["future_pi3_features"],
        "future_pi3_non_eef_region_mask": torch.ones(2, 1, 20),
    }
    assert_loss(non_eef_region, non_eef_batch, future_enabled=True)

    interaction = make_policy(use_future_loss=True, future_target_mode="interaction_state")
    set_test_normalizer(interaction, batch)
    expect_error(lambda: interaction.compute_loss(batch), NotImplementedError, "interaction_state")

    print("future mode smoke tests passed")


if __name__ == "__main__":
    main()
