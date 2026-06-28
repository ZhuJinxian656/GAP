#!/usr/bin/env python
"""Synthetic smoke tests for trajectory-coupled future Pi3 target modes."""
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
        future={
            "pi3_pool_grid": [2, 2],
            "pi3_num_compressed_tokens": 3,
            "changed_token_percentile": 90,
            "changed_token_topk": None,
            "changed_token_stopgrad_mask": True,
        },
        future_loss_weight=0.1,
    )
    cfg.update(overrides)
    return GAPPolicy(**cfg)


def make_batch(batch=2):
    current_pi3 = torch.randn(batch, 1, 20, 32)
    future_pi3 = current_pi3 + 0.1 * torch.randn(batch, 1, 20, 32)
    return {
        "obs": {
            "dinov3_features": torch.randn(batch, 1, 6, 32),
            "pi3_features": current_pi3,
            "agent_pos": torch.randn(batch, 14),
        },
        "action": torch.randn(batch, 4, 14),
        "future_pi3_features": future_pi3,
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


def assert_finite_loss(policy, batch, future_enabled=True):
    set_test_normalizer(policy, batch)
    loss, loss_dict = policy.compute_loss(batch)
    assert torch.isfinite(loss), loss
    assert "action_loss" in loss_dict
    assert "future_loss" in loss_dict
    assert "total_loss" in loss_dict
    if future_enabled:
        assert loss_dict["future_loss"] >= 0.0
    else:
        assert loss_dict["future_loss"] == 0.0


def expect_error(fn, exc_type, text):
    try:
        fn()
    except exc_type as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {exc_type.__name__}")


def flatten_future(features):
    batch, views, tokens, dim = features.shape
    return features.permute(0, 2, 1, 3).reshape(batch, tokens, views * dim)


def main():
    torch.manual_seed(7)
    batch = make_batch()

    full = make_policy(use_future_loss=True, future_target_mode="pi3_full")
    full_target, full_mask = full._prepare_future_pi3_target(batch["future_pi3_features"], batch)
    assert full_mask is None
    assert torch.allclose(full_target, flatten_future(batch["future_pi3_features"]))
    assert_finite_loss(full, batch, future_enabled=True)

    delta = make_policy(use_future_loss=True, future_target_mode="pi3_delta")
    delta_target, delta_mask = delta._prepare_future_pi3_target(batch["future_pi3_features"], batch)
    assert delta_mask is None
    expected_delta = batch["future_pi3_features"] - batch["obs"]["pi3_features"]
    assert delta_target.shape == flatten_future(expected_delta).shape
    assert torch.allclose(delta_target, flatten_future(expected_delta))
    assert_finite_loss(delta, batch, future_enabled=True)

    changed = make_policy(
        use_future_loss=True,
        future_target_mode="pi3_changed_tokens",
        future={"changed_token_topk": 3, "changed_token_percentile": 90, "changed_token_stopgrad_mask": True},
    )
    changed_target, changed_mask = changed._prepare_future_pi3_target(batch["future_pi3_features"], batch)
    assert changed_target.shape == flatten_future(batch["future_pi3_features"]).shape
    assert changed_mask.shape == batch["future_pi3_features"].shape[:-1]
    assert torch.all(changed_mask.sum(dim=(1, 2)) == 3)
    assert_finite_loss(changed, batch, future_enabled=True)

    delta_changed = make_policy(
        use_future_loss=True,
        future_target_mode="pi3_delta_changed_tokens",
        future={"changed_token_topk": 2, "changed_token_percentile": 90, "changed_token_stopgrad_mask": True},
    )
    delta_changed_target, delta_changed_mask = delta_changed._prepare_future_pi3_target(
        batch["future_pi3_features"],
        batch,
    )
    assert delta_changed_target.shape == flatten_future(expected_delta).shape
    assert torch.all(delta_changed_mask.sum(dim=(1, 2)) == 2)
    assert_finite_loss(delta_changed, batch, future_enabled=True)

    no_future = make_policy(use_future_loss=False, future_target_mode="none")
    assert_finite_loss(no_future, {"obs": batch["obs"], "action": batch["action"]}, future_enabled=False)

    zero_batch = make_batch(batch=1)
    zero_batch["future_pi3_features"] = zero_batch["obs"]["pi3_features"].clone()
    zero_changed = make_policy(use_future_loss=True, future_target_mode="pi3_changed_tokens")
    _, zero_mask = zero_changed._prepare_future_pi3_target(zero_batch["future_pi3_features"], zero_batch)
    assert zero_mask.shape == zero_batch["future_pi3_features"].shape[:-1]
    assert zero_mask.sum() > 0
    assert_finite_loss(zero_changed, zero_batch, future_enabled=True)

    missing = make_policy(use_future_loss=True, future_target_mode="pi3_delta")
    missing_batch = {
        "obs": {
            "dinov3_features": batch["obs"]["dinov3_features"],
            "agent_pos": batch["obs"]["agent_pos"],
        },
        "action": batch["action"],
        "future_pi3_features": batch["future_pi3_features"],
    }
    set_test_normalizer(missing, batch)
    expect_error(lambda: missing.compute_loss(missing_batch), KeyError, "pi3_features")

    malformed = make_policy(use_future_loss=True, future_target_mode="pi3_changed_tokens")
    malformed_batch = make_batch()
    malformed_batch["future_pi3_features"] = malformed_batch["future_pi3_features"][:, :, :-1]
    set_test_normalizer(malformed, batch)
    expect_error(lambda: malformed.compute_loss(malformed_batch), ValueError, "shapes must match")

    print("trajectory future mode smoke tests passed")


if __name__ == "__main__":
    main()
