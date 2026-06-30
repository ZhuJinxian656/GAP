#!/usr/bin/env python
"""Smoke tests for action-UV supervision mode behavior."""
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


def build_policy(device: torch.device, generative_mode: str, uv_mode: str) -> GAPPolicy:
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
    return GAPPolicy(
        shape_meta={"action": {"shape": [14]}},
        noise_scheduler=scheduler,
        horizon=3,
        n_action_steps=3,
        n_obs_steps=1,
        num_inference_steps=1,
        generative_mode=generative_mode,
        flow_matching={
            "prediction_type": "velocity",
            "source_mode": "hold",
            "lambda_min": 0.0,
            "lambda_max": 1.0,
            "num_inference_steps": 2,
            "action_source_noise_std": 0.0,
        },
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
            "mode": "action_uv",
            "readout": "masked_pool",
            "grid_height": 15,
            "grid_width": 20,
            "use_pair_token": True,
            "append_tokens": True,
            "sigma_start": 4.0,
            "sigma_end": 1.0,
            "pair_sigma_start": 5.0,
            "pair_sigma_end": 1.5,
            "action_uv_hidden_dim": 32,
            "uv_loss_weight": 0.05,
            "uv_supervision_mode": uv_mode,
            "flow_interp_requires_current_valid": True,
            "flow_interp_use_current_pair": True,
            "use_current_eef_fallback": True,
            "current_eef_fallback_weight": 0.25,
            "uv_loss_progress_gamma": 2.0,
            "uv_loss_min_progress": 0.05,
            "uv_loss_clamp_target": True,
            "clean_uv_loss_weight": 0.05,
            "clean_uv_progress": 1.0,
            "log_interaction_stats": True,
        },
    ).to(device)


def build_batch(device: torch.device):
    batch_size = 2
    horizon = 3
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
        "action": torch.randn(batch_size, horizon, 14, device=device),
        "target_dino_eef_uv_seq": torch.rand(batch_size, horizon, 1, 2, 2, device=device),
        "target_dino_eef_valid_seq": torch.ones(batch_size, horizon, 1, 2, device=device),
    }
    batch["target_dino_eef_uv_seq"][..., 0] *= 19.0
    batch["target_dino_eef_uv_seq"][..., 1] *= 14.0
    batch["target_dino_eef_valid_seq"][0, 1, 0, 1] = 0.0
    return batch


def install_normalizer(policy: GAPPolicy, device: torch.device) -> None:
    normalizer = LinearNormalizer()
    normalizer.fit(
        {
            "action": torch.randn(8, policy.horizon, policy.action_dim, device=device),
            "agent_pos": torch.randn(8, policy.state_dim, device=device),
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


def assert_mode(
    device: torch.device,
    generative_mode: str,
    uv_mode: str,
    *,
    expect_uv: bool,
    expect_clean: bool,
    expect_interp: bool = False,
) -> None:
    torch.manual_seed(123)
    policy = build_policy(device, generative_mode, uv_mode)
    batch = build_batch(device)
    install_normalizer(policy, device)
    loss, loss_dict = policy.compute_loss(batch)
    assert torch.isfinite(loss), (generative_mode, uv_mode, loss_dict)
    assert ("uv_loss" in loss_dict) == expect_uv, (generative_mode, uv_mode, loss_dict)
    assert ("clean_uv_loss" in loss_dict) == expect_clean, (generative_mode, uv_mode, loss_dict)
    assert "uv_supervision_mode_id" in loss_dict, (generative_mode, uv_mode, loss_dict)
    assert ("interp_lambda_mean" in loss_dict) == expect_interp, (generative_mode, uv_mode, loss_dict)


def assert_diffusion_flow_interp_error(device: torch.device) -> None:
    torch.manual_seed(123)
    policy = build_policy(device, "diffusion", "flow_interp")
    batch = build_batch(device)
    install_normalizer(policy, device)
    try:
        policy.compute_loss(batch)
    except ValueError as exc:
        assert "uv_supervision_mode='flow_interp' is currently only defined for flow_matching." in str(exc)
        return
    raise AssertionError("diffusion + flow_interp should raise a clear ValueError")


def main() -> int:
    device = torch.device("cpu")
    cases = [
        ("diffusion", "expert_final", True, True, False),
        ("diffusion", "clean_only", False, True, False),
        ("flow_matching", "expert_final", True, True, False),
        ("flow_matching", "flow_interp", True, True, True),
        ("flow_matching", "clean_only", False, True, False),
        ("flow_matching", "none", False, False, False),
    ]
    for generative_mode, uv_mode, expect_uv, expect_clean, expect_interp in cases:
        assert_mode(
            device,
            generative_mode,
            uv_mode,
            expect_uv=expect_uv,
            expect_clean=expect_clean,
            expect_interp=expect_interp,
        )
    assert_diffusion_flow_interp_error(device)
    print("action-UV supervision mode smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
