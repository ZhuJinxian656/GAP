#!/usr/bin/env python
"""Smoke test that flow hold source is normalized in action space."""
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


def manual_normalizer(dim: int, scale_value: float, offset_value: float) -> SingleFieldLinearNormalizer:
    scale = np.full((dim,), scale_value, dtype=np.float32)
    offset = np.full((dim,), offset_value, dtype=np.float32)
    stats = {
        "min": np.zeros((dim,), dtype=np.float32),
        "max": np.ones((dim,), dtype=np.float32),
        "mean": np.zeros((dim,), dtype=np.float32),
        "std": np.ones((dim,), dtype=np.float32),
    }
    return SingleFieldLinearNormalizer.create_manual(scale=scale, offset=offset, input_stats_dict=stats)


def identity_normalizer() -> SingleFieldLinearNormalizer:
    return manual_normalizer(dim=1, scale_value=1.0, offset_value=0.0)


def build_policy(device: torch.device) -> GAPPolicy:
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
        shape_meta={"action": {"shape": [4]}},
        noise_scheduler=scheduler,
        horizon=3,
        n_action_steps=3,
        n_obs_steps=1,
        num_inference_steps=1,
        generative_mode="flow_matching",
        flow_matching={
            "prediction_type": "velocity",
            "source_mode": "hold",
            "lambda_min": 0.0,
            "lambda_max": 1.0,
            "num_inference_steps": 1,
            "action_source_noise_std": 0.0,
        },
        dinov3_feature_dim=8,
        dinov3_num_views=1,
        use_pi3_features=False,
        use_future_loss=False,
        future_target_mode="none",
        state_dim=4,
        state_embed_dim=8,
        encoder_depth=1,
        encoder_heads=2,
        encoder_dim_feedforward=16,
        decoder_depth=1,
        decoder_heads=2,
        decoder_dim_feedforward=16,
        dinov3_grid_height=2,
        dinov3_grid_width=3,
        use_interaction_field=False,
        interaction_field={"mode": "disabled"},
    ).to(device)


def install_normalizer(policy: GAPPolicy) -> None:
    normalizer = LinearNormalizer()
    normalizer["action"] = manual_normalizer(policy.action_dim, scale_value=0.5, offset_value=-1.0)
    normalizer["agent_pos"] = manual_normalizer(policy.state_dim, scale_value=10.0, offset_value=7.0)
    normalizer["dinov3_features"] = identity_normalizer()
    policy.set_normalizer(normalizer)


def main() -> int:
    device = torch.device("cpu")
    torch.manual_seed(11)
    policy = build_policy(device)
    install_normalizer(policy)

    raw_agent_pos = torch.tensor(
        [[0.0, 1.0, -2.0, 3.0], [4.0, -5.0, 6.0, -7.0]],
        dtype=torch.float32,
        device=device,
    )
    raw_obs = {
        "agent_pos": raw_agent_pos,
        "dinov3_features": torch.randn(2, 1, 6, 8, device=device),
    }

    source = policy._flow_source_actions(
        raw_obs_dict=raw_obs,
        ref_actions=torch.zeros(2, policy.horizon, policy.action_dim, device=device),
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )
    expected = policy.normalizer["action"].normalize(
        raw_agent_pos.unsqueeze(1).expand(2, policy.horizon, policy.action_dim)
    )
    wrong_agent_pos_space = policy.normalizer["agent_pos"].normalize(raw_agent_pos).unsqueeze(1).expand_as(expected)

    assert torch.allclose(source, expected), "hold source must use action normalizer on raw agent_pos"
    assert not torch.allclose(source, wrong_agent_pos_space), "test setup failed: normalizers are not distinct"

    def zero_velocity(noised_actions, timestep, memory=None, memory_pos=None, context=None):
        return torch.zeros_like(noised_actions)

    policy.forward_diffusion = zero_velocity
    policy.eval()
    with torch.no_grad():
        result = policy.predict_action(raw_obs)
    expected_action = raw_agent_pos.unsqueeze(1).expand(2, policy.n_action_steps, policy.action_dim)
    assert torch.isfinite(result["action"]).all(), "predict_action produced non-finite actions"
    assert torch.allclose(result["action"], expected_action, atol=1.0e-5), (
        "flow inference hold source should round-trip to raw agent_pos when velocity is zero"
    )

    print("flow source normalization smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
