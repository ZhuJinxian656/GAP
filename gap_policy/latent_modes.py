"""Token-level latent mode helpers for GAP interface ablations."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PI3_LATENT_MODES = (
    "pi3_full",
    "dino_only",
    "pi3_none",
    "pi3_pooled",
    "pi3_compressed",
    "pi3_random_tokens",
    "pi3_token_dropout",
    "pi3_object_hand",
    "pi3_background",
    "pi3_eef_region",
    "pi3_non_eef_region",
)

FUTURE_TARGET_MODES = (
    "pi3_full",
    "none",
    "pi3_pooled",
    "pi3_compressed",
    "pi3_random_tokens",
    "pi3_token_dropout",
    "pi3_object_hand",
    "pi3_background",
    "pi3_eef_region",
    "pi3_non_eef_region",
    "interaction_state",
)

MASK_MODE_TO_KEY = {
    "pi3_object_hand": "pi3_object_hand_mask",
    "pi3_background": "pi3_background_mask",
    "pi3_eef_region": "pi3_eef_region_mask",
    "pi3_non_eef_region": "pi3_non_eef_region_mask",
}


def normalize_mode(mode: str, valid_modes: Iterable[str], *, field_name: str) -> str:
    mode = str(mode)
    if mode == "pi3_none":
        mode = "dino_only" if "dino_only" in valid_modes else mode
    valid_modes = tuple(valid_modes)
    if mode not in valid_modes:
        raise ValueError(f"Unsupported {field_name}={mode!r}. Expected one of {valid_modes}.")
    return mode


def mode_requires_pi3(mode: str) -> bool:
    return mode not in ("none", "dino_only", "pi3_none", "interaction_state")


def mode_requires_mask(mode: str) -> bool:
    return mode in MASK_MODE_TO_KEY


def mask_key_for_mode(mode: str) -> Optional[str]:
    return MASK_MODE_TO_KEY.get(mode)


def parse_pair(value: Sequence[int] | str | None, default: Tuple[int, int]) -> Tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.replace("x", ",").split(",") if piece.strip()]
        if len(pieces) != 2:
            raise ValueError(f"Expected pair value like '4,4' or '4x4', got {value!r}.")
        return int(pieces[0]), int(pieces[1])
    if len(value) != 2:
        raise ValueError(f"Expected pair with length 2, got {value!r}.")
    return int(value[0]), int(value[1])


def flatten_pi3_tokens(features: torch.Tensor) -> torch.Tensor:
    """Flatten [B, V, N, D] Pi3 tokens into [B, V*N, D]."""
    if features.ndim != 4:
        raise ValueError(f"Expected Pi3 features [B, V, N, D], got shape {tuple(features.shape)}.")
    batch, views, tokens, dim = features.shape
    return features.reshape(batch, views * tokens, dim)


def pool_pi3_grid(features: torch.Tensor, grid_size: Tuple[int, int], pool_grid: Tuple[int, int]) -> torch.Tensor:
    """Adaptive-average-pool Pi3 tokens from [B, V, H*W, D] to [B, V, Ph*Pw, D]."""
    if features.ndim != 4:
        raise ValueError(f"Expected Pi3 features [B, V, N, D], got shape {tuple(features.shape)}.")
    batch, views, tokens, dim = features.shape
    height, width = grid_size
    if tokens != height * width:
        raise ValueError(
            f"Pi3 token count {tokens} does not match configured grid {height}x{width}={height * width}."
        )
    pool_h, pool_w = pool_grid
    if pool_h <= 0 or pool_w <= 0:
        raise ValueError(f"Invalid pool grid {pool_grid}; both dimensions must be positive.")
    x = features.reshape(batch * views, height, width, dim).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, output_size=(pool_h, pool_w))
    return x.permute(0, 2, 3, 1).reshape(batch, views, pool_h * pool_w, dim)


def pool_grid_tokens(tokens: torch.Tensor, grid_size: Tuple[int, int], pool_grid: Tuple[int, int]) -> torch.Tensor:
    """Pool a single-grid token table [H*W, D] to [Ph*Pw, D]."""
    if tokens.ndim != 2:
        raise ValueError(f"Expected token table [N, D], got shape {tuple(tokens.shape)}.")
    height, width = grid_size
    if tokens.shape[0] != height * width:
        raise ValueError(
            f"Token count {tokens.shape[0]} does not match configured grid {height}x{width}={height * width}."
        )
    pooled = pool_pi3_grid(tokens.reshape(1, 1, tokens.shape[0], tokens.shape[1]), grid_size, pool_grid)
    return pooled.reshape(-1, tokens.shape[1])


def select_token_indices(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select token indices from [B, V, N, D] and preserve views."""
    if features.ndim != 4:
        raise ValueError(f"Expected Pi3 features [B, V, N, D], got shape {tuple(features.shape)}.")
    if indices.ndim != 1:
        raise ValueError("indices must be a 1D tensor.")
    if indices.numel() > features.shape[2]:
        raise ValueError(f"Requested {indices.numel()} tokens, but Pi3 feature only has {features.shape[2]}.")
    return features.index_select(dim=2, index=indices.to(features.device))


def normalize_token_mask(mask: torch.Tensor, features: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Return a float mask broadcastable to [B, V, N, D]."""
    if features.ndim != 4:
        raise ValueError(f"Expected features [B, V, N, D], got {tuple(features.shape)}.")
    batch, views, tokens, _ = features.shape
    if mask.ndim == 2:
        if mask.shape != (batch, tokens):
            raise ValueError(
                f"{mode} mask shape {tuple(mask.shape)} must be [B, N]={batch, tokens} "
                "or [B, V, N]."
            )
        mask = mask.unsqueeze(1).expand(-1, views, -1)
    elif mask.ndim == 3:
        if mask.shape[0] != batch or mask.shape[2] != tokens:
            raise ValueError(
                f"{mode} mask shape {tuple(mask.shape)} must match [B, V, N] with B={batch}, N={tokens}."
            )
        if mask.shape[1] == 1 and views != 1:
            mask = mask.expand(-1, views, -1)
        elif mask.shape[1] != views:
            raise ValueError(f"{mode} mask view dimension {mask.shape[1]} does not match features views {views}.")
    else:
        raise ValueError(f"{mode} mask must have shape [B, N] or [B, V, N], got {tuple(mask.shape)}.")
    return mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)


def apply_dense_token_mask(features: torch.Tensor, mask: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Zero non-selected Pi3 tokens while preserving dense token shape."""
    token_mask = normalize_token_mask(mask, features, mode=mode)
    return features * token_mask


def apply_token_dropout(features: torch.Tensor, dropout_rate: float, training: bool) -> torch.Tensor:
    """Drop Pi3 token features during training while preserving token shape."""
    if not training or dropout_rate <= 0:
        return features
    if dropout_rate >= 1:
        return torch.zeros_like(features)
    keep = torch.rand(features.shape[:-1], device=features.device, dtype=features.dtype) >= dropout_rate
    keep = keep.unsqueeze(-1)
    return features * keep / (1.0 - dropout_rate)


def make_random_indices(num_tokens: int, keep_tokens: int, seed: int) -> torch.Tensor:
    if keep_tokens <= 0:
        raise ValueError("pi3_random_num_tokens must be positive.")
    if keep_tokens > num_tokens:
        raise ValueError(f"pi3_random_num_tokens={keep_tokens} exceeds available tokens={num_tokens}.")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return torch.randperm(num_tokens, generator=generator)[:keep_tokens].sort().values


class LearnedTokenCompressor(nn.Module):
    """Small attention-pooling compressor from dense Pi3 tokens to K tokens."""

    def __init__(self, feature_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if feature_dim % num_heads != 0:
            num_heads = 1
        self.queries = nn.Parameter(torch.randn(num_tokens, feature_dim) * 0.02)
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected Pi3 features [B, V, N, D], got shape {tuple(features.shape)}.")
        batch = features.shape[0]
        flat = flatten_pi3_tokens(features)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        compressed, _ = self.attn(queries, flat, flat, need_weights=False)
        return self.norm(compressed)
