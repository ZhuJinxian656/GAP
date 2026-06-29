"""
GAP Policy
Uses pre-extracted DinoV3 and Pi3 features for vision encoding and transformer decoder for action denoising.

Simplified architecture:
1. Load pre-extracted DinoV3 and Pi3 features (no runtime extraction)
2. State encoder processes robot joint states
3. Transformer decoder attends to all patch tokens + state token for action prediction

Updated to align with ACT-DP-TP:
- Custom decoder layer supporting separate query_pos and memory_pos
- Learnable position embedding for action queries
- Diffusion timestep added to memory (not query)

Feature extraction mode:
- Pre-extracted features from process_data.py are used directly
- No runtime feature extraction (much faster training and inference)
"""
from typing import Any, Dict, Mapping, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
from torch import Tensor

from gap_policy.model.common.normalizer import LinearNormalizer
from gap_policy.policy.base_policy import BasePolicy
from gap_policy.common.pytorch_util import dict_apply
from gap_policy.common.model_util import print_params
from gap_policy.triadic import infer_triadic_dim
from gap_policy.latent_modes import (
    FUTURE_TARGET_MODES,
    PI3_LATENT_MODES,
    LearnedTokenCompressor,
    apply_dense_token_mask,
    apply_token_dropout,
    flatten_pi3_tokens,
    make_random_indices,
    mask_key_for_mode,
    mode_requires_mask,
    mode_requires_pi3,
    normalize_mode,
    parse_pair,
    pool_grid_tokens,
    pool_pi3_grid,
    select_token_indices,
)


# ============================================================================
# Custom Transformer Decoder supporting separate positional encodings
# (Aligned with ACT-DP-TP implementation)
# ============================================================================

class TransformerDecoderLayerWithPE(nn.Module):
    """
    Custom Transformer Decoder Layer that supports separate positional encodings
    for query and memory, following ACT-DP-TP's design.

    Position encodings are only added to Q and K in attention, not to V.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        batch_first: bool = True,
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings (separate from features)
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D] - Memory positional encoding
            query_pos: [B, N_query, D] - Query positional encoding
        """
        if self.norm_first:
            # Pre-norm (used in GAP)
            # Self-attention
            tgt2 = self.norm1(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt2,  # value不加位置编码
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout1(tgt2)

            # Cross-attention
            tgt2 = self.norm2(tgt)
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),  # query加位置
                key=self.with_pos_embed(memory, memory_pos),  # key加位置
                value=memory,  # value不加位置！
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)

            # FFN
            tgt2 = self.norm3(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout3(tgt2)
        else:
            # Post-norm (ACT-DP-TP uses this)
            # Self-attention
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(
                q, k, tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            # Cross-attention
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, memory_pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            # FFN
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt


class TransformerDecoderWithPE(nn.Module):
    """Custom Transformer Decoder supporting separate positional encodings"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        # Position encodings
        memory_pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt: [B, N_query, D]
            memory: [B, N_memory, D]
            memory_pos: [B, N_memory, D]
            query_pos: [B, N_query, D]
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_pos=memory_pos,
                query_pos=query_pos,
            )

        return output


# ============================================================================
# Transformer Encoder (from ACT-DP-TP) for context encoding
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer with positional encoding support.
    Adapted from ACT-DP-TP.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        norm_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.norm_first = norm_first

    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]):
        """Add position embedding to tensor"""
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D] - positional encoding
        """
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(
                q, k, src2,  # value不加位置编码
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            src = src + self.dropout1(src2)

            # FFN
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Post-norm
            q = k = self.with_pos_embed(src, pos)
            src2 = self.self_attn(
                q, k, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

            # FFN
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    """Standard Transformer Encoder stack"""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(encoder_layer) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            src: [B, N, D]
            pos: [B, N, D]
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        return output


class ActionToUVHead(nn.Module):
    """Predict action-conditioned left/right EEF UV trajectories on a DINO token grid."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        horizon: int,
        hidden_dim: int,
        num_views: int,
        grid_height: int,
        grid_width: int,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.num_views = int(num_views)
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.progress_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.horizon_pos_embed = nn.Parameter(torch.randn(self.horizon, hidden_dim) * 0.02)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_views * 2 * 2),
        )

    def forward(
        self,
        mid_actions: torch.Tensor,
        agent_pos: torch.Tensor,
        progress: torch.Tensor,
    ) -> torch.Tensor:
        if mid_actions.ndim != 3:
            raise ValueError(f"mid_actions must be shaped [B, H, action_dim], got {tuple(mid_actions.shape)}.")
        B, H, _ = mid_actions.shape
        if H > self.horizon:
            raise ValueError(f"mid_actions horizon {H} exceeds configured horizon {self.horizon}.")

        progress = progress.to(device=mid_actions.device, dtype=mid_actions.dtype).reshape(B, 1).clamp(0.0, 1.0)
        action_features = self.action_mlp(mid_actions)
        state_features = self.state_mlp(agent_pos.to(device=mid_actions.device, dtype=mid_actions.dtype)).unsqueeze(1)
        progress_features = self.progress_mlp(progress).unsqueeze(1)
        horizon_features = self.horizon_pos_embed[:H].to(device=mid_actions.device, dtype=mid_actions.dtype).unsqueeze(0)

        logits = self.output(action_features + state_features + progress_features + horizon_features)
        uv = torch.sigmoid(logits).reshape(B, H, self.num_views, 2, 2)
        scale = torch.tensor(
            [max(self.grid_width - 1, 1), max(self.grid_height - 1, 1)],
            device=mid_actions.device,
            dtype=mid_actions.dtype,
        )
        return uv * scale.view(1, 1, 1, 1, 2)


class GAPPolicy(BasePolicy):

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        generative_mode="diffusion",
        flow_matching=None,
        # Feature dimensions (from pre-extracted features)
        dinov3_feature_dim=1024,  # DinoV3 ViT-L/16 feature dimension
        dinov3_num_views=1,       # Number of camera views for DinoV3
        # Pi3 feature config
        use_pi3_features=True,    # Whether to use pre-extracted pi3 features
        pi3_feature_dim=1024,     # Pi3 point_hidden feature dimension
        pi3_num_views=1,          # Number of camera views for Pi3 (head, left_wrist, right_wrist)
        pi3_embed_dim=1024,       # Embedding dimension for pi3 features (match with vision_feat_dim)
        latent_mode="pi3_full",
        latent=None,
        use_future_loss=True,
        future_target_mode="pi3_full",
        future_loss_weight=0.1,
        future=None,
        dinov3_grid_height=15,
        dinov3_grid_width=20,
        pi3_grid_height=17,
        pi3_grid_width=23,
        # State encoder config
        state_dim=14,
        state_embed_dim=1024,     # Must match feature dimensions
        # Optional triadic relation token
        use_triadic_token=False,
        triadic_mode="disabled",
        triadic_dim=None,
        triadic_hidden_dim=1024,
        triadic_dropout=0.0,
        triadic_include_grippers=True,
        # Optional DINO EEF interaction-field tokens
        use_interaction_field=False,
        interaction_field=None,
        # Transformer encoder config (for context encoding - ACT-DP-TP style)
        encoder_depth=2,
        encoder_heads=8,
        encoder_dim_feedforward=2048,
        encoder_dropout=0.1,
        # Transformer decoder config (for denoising)
        decoder_depth=4,
        decoder_heads=8,
        decoder_dim_feedforward=2048,
        decoder_dropout=0.1,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # Parse action shape
        action_shape = shape_meta["action"]["shape"]
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        # Store feature dimensions (no vision encoder, use pre-extracted features)
        self.dinov3_feature_dim = dinov3_feature_dim
        self.dinov3_num_views = dinov3_num_views
        vision_feat_dim = dinov3_feature_dim  # 1024 for vitl16
        self.dinov3_grid_height = int(dinov3_grid_height)
        self.dinov3_grid_width = int(dinov3_grid_width)
        self.pi3_grid_height = int(pi3_grid_height)
        self.pi3_grid_width = int(pi3_grid_width)

        # State encoder (agent_pos)
        self.state_encoder = nn.Linear(state_dim, state_embed_dim)

        self.use_triadic_token = bool(use_triadic_token)
        self.triadic_mode = triadic_mode
        self.triadic_include_grippers = triadic_include_grippers
        if self.use_triadic_token:
            if triadic_mode == "disabled":
                raise ValueError("use_triadic_token=True requires triadic_mode to be pairwise, triadic, or proprio_only_fallback.")
            if triadic_dim is None:
                triadic_dim = infer_triadic_dim(triadic_mode, include_grippers=triadic_include_grippers)
            triadic_hidden_dim = triadic_hidden_dim or state_embed_dim
            self.triadic_dim = triadic_dim
            self.triadic_encoder = nn.Sequential(
                nn.Linear(triadic_dim, triadic_hidden_dim),
                nn.SiLU(),
                nn.Dropout(triadic_dropout),
                nn.Linear(triadic_hidden_dim, state_embed_dim),
            )
        else:
            self.triadic_dim = 0

        # Pi3 feature encoder (if using pre-extracted features)
        self.use_pi3_features = use_pi3_features
        self.latent_mode = normalize_mode(latent_mode, PI3_LATENT_MODES, field_name="latent_mode")
        self.use_pi3_obs_tokens = bool(use_pi3_features) and mode_requires_pi3(self.latent_mode)
        self.use_future_loss = bool(use_future_loss)
        self.future_target_mode = normalize_mode(
            future_target_mode,
            FUTURE_TARGET_MODES,
            field_name="future_target_mode",
        )
        self.future_loss_weight = float(future_loss_weight)
        self.predict_future_pi3 = (
            bool(use_pi3_features)
            and self.use_future_loss
            and mode_requires_pi3(self.future_target_mode)
        )
        self.predict_interaction_state = (
            self.use_future_loss and self.future_target_mode == "interaction_state"
        )
        self.latent_cfg = self._to_plain_dict(latent)
        self.future_cfg = self._to_plain_dict(future)
        self.latent_pool_grid = parse_pair(self.latent_cfg.get("pi3_pool_grid"), (4, 4))
        self.future_pool_grid = parse_pair(self.future_cfg.get("pi3_pool_grid"), self.latent_pool_grid)
        self.latent_num_compressed_tokens = int(self.latent_cfg.get("pi3_num_compressed_tokens", 16))
        self.future_num_compressed_tokens = int(self.future_cfg.get("pi3_num_compressed_tokens", 16))
        self.changed_token_percentile = float(self.future_cfg.get("changed_token_percentile", 90.0))
        self.changed_token_topk = self.future_cfg.get("changed_token_topk", None)
        self.changed_token_stopgrad_mask = bool(self.future_cfg.get("changed_token_stopgrad_mask", True))
        self.pi3_random_num_tokens = int(self.latent_cfg.get("pi3_random_num_tokens", 64))
        self.pi3_random_seed = int(self.latent_cfg.get("pi3_random_seed", 0))
        self.pi3_token_dropout_rate = float(self.latent_cfg.get("pi3_token_dropout_rate", 0.0))
        self.require_token_mask = bool(self.latent_cfg.get("require_token_mask", True))
        self.pi3_total_tokens = self.pi3_grid_height * self.pi3_grid_width
        if self.latent_mode == "pi3_random_tokens" or self.future_target_mode == "pi3_random_tokens":
            self.register_buffer(
                "pi3_random_indices",
                make_random_indices(self.pi3_total_tokens, self.pi3_random_num_tokens, self.pi3_random_seed),
                persistent=False,
            )
        else:
            self.register_buffer("pi3_random_indices", torch.empty(0, dtype=torch.long), persistent=False)
        if self.latent_mode == "pi3_compressed":
            self.pi3_latent_compressor = LearnedTokenCompressor(vision_feat_dim, self.latent_num_compressed_tokens)
            self.pi3_latent_compressed_pos_embed = nn.Parameter(
                torch.randn(self.latent_num_compressed_tokens, vision_feat_dim) * 0.02
            )
        if self.future_target_mode == "pi3_compressed":
            self.pi3_future_target_compressor = LearnedTokenCompressor(
                vision_feat_dim,
                self.future_num_compressed_tokens,
            )
            self.pi3_future_query_embed = nn.Parameter(
                torch.randn(self.future_num_compressed_tokens, vision_feat_dim) * 0.02
            )
            self.pi3_future_query_pos_embed = nn.Parameter(
                torch.randn(self.future_num_compressed_tokens, vision_feat_dim) * 0.02
            )
        if use_pi3_features:
            self.pi3_num_views = pi3_num_views
            self.pi3_feature_dim = pi3_feature_dim
            # Learned position embeddings for pi3 features (one per view)
            self.pi3_view_pos_embed = nn.Embedding(pi3_num_views, pi3_embed_dim)
            assert pi3_embed_dim == vision_feat_dim, \
                f"Pi3 embed dim ({pi3_embed_dim}) must match vision dim ({vision_feat_dim})"

        # Check dimensions match for token concatenation
        assert vision_feat_dim == state_embed_dim, \
            f"Vision dim ({vision_feat_dim}) must match state dim ({state_embed_dim})"

        # Feature dimension for decoder (same as vision/state dim)
        self.feature_dim = vision_feat_dim  # 1024

        self.use_interaction_field = bool(use_interaction_field)
        self.interaction_cfg = self._to_plain_dict(interaction_field)
        self.interaction_mode = self.interaction_cfg.get("mode", "disabled")
        self.interaction_readout = self.interaction_cfg.get("readout", "masked_pool")
        self.interaction_use_pair_token = bool(self.interaction_cfg.get("use_pair_token", True))
        self.interaction_append_tokens = bool(self.interaction_cfg.get("append_tokens", True))
        self.interaction_eps = float(self.interaction_cfg.get("eps", 1.0e-6))
        self.interaction_grid_height = int(self.interaction_cfg.get("grid_height", self.dinov3_grid_height))
        self.interaction_grid_width = int(self.interaction_cfg.get("grid_width", self.dinov3_grid_width))
        self.interaction_sigma_start = float(self.interaction_cfg.get("sigma_start", 4.0))
        self.interaction_sigma_end = float(self.interaction_cfg.get("sigma_end", 1.0))
        self.interaction_pair_sigma_start = float(self.interaction_cfg.get("pair_sigma_start", 5.0))
        self.interaction_pair_sigma_end = float(self.interaction_cfg.get("pair_sigma_end", 1.5))
        self.interaction_uv_loss_weight = float(self.interaction_cfg.get("uv_loss_weight", 0.05))
        self.interaction_use_current_eef_fallback = bool(self.interaction_cfg.get("use_current_eef_fallback", True))
        self.interaction_current_eef_fallback_weight = float(
            self.interaction_cfg.get("current_eef_fallback_weight", 0.25)
        )
        self.interaction_uv_loss_progress_gamma = float(self.interaction_cfg.get("uv_loss_progress_gamma", 2.0))
        self.interaction_uv_loss_min_progress = float(self.interaction_cfg.get("uv_loss_min_progress", 0.05))
        self.interaction_uv_loss_clamp_target = bool(self.interaction_cfg.get("uv_loss_clamp_target", True))
        self.interaction_clean_uv_loss_weight = float(self.interaction_cfg.get("clean_uv_loss_weight", 0.05))
        self.interaction_clean_uv_progress = float(self.interaction_cfg.get("clean_uv_progress", 1.0))
        self.interaction_log_stats = bool(self.interaction_cfg.get("log_interaction_stats", True))
        self.action_to_uv_head = None
        if self.use_interaction_field:
            if self.interaction_mode == "disabled":
                raise ValueError(
                    "use_interaction_field=True requires interaction_field.mode to be "
                    "current_eef or action_uv."
                )
            if self.interaction_mode not in ("current_eef", "action_uv"):
                raise ValueError(
                    f"Unsupported interaction_field.mode={self.interaction_mode!r}. "
                    "Expected disabled, current_eef, or action_uv."
                )
            if self.interaction_readout != "masked_pool":
                raise ValueError(
                    f"Unsupported interaction_field.readout={self.interaction_readout!r}. "
                    "Only 'masked_pool' is implemented."
                )
            expected_tokens = self.dinov3_grid_height * self.dinov3_grid_width
            configured_tokens = self.interaction_grid_height * self.interaction_grid_width
            if configured_tokens != expected_tokens:
                raise ValueError(
                    "interaction_field grid size must match the DINO grid used by the policy, "
                    f"got {self.interaction_grid_height}x{self.interaction_grid_width} "
                    f"vs {self.dinov3_grid_height}x{self.dinov3_grid_width}."
                )
            self.interaction_type_embed = nn.Embedding(3, self.feature_dim)
            self.interaction_pos_embed = nn.Embedding(3, self.feature_dim)
            if self.interaction_mode == "action_uv":
                self.action_to_uv_head = ActionToUVHead(
                    action_dim=action_dim,
                    state_dim=state_dim,
                    horizon=horizon,
                    hidden_dim=int(self.interaction_cfg.get("action_uv_hidden_dim", 512)),
                    num_views=dinov3_num_views,
                    grid_height=self.interaction_grid_height,
                    grid_width=self.interaction_grid_width,
                )

        # CLS token (learnable global context token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))

        # Learned position embeddings
        self.cls_pos_embed = nn.Embedding(1, self.feature_dim)  # CLS token position
        self.state_pos_embed = nn.Embedding(1, self.feature_dim)  # State token position
        if self.use_triadic_token:
            self.triadic_pos_embed = nn.Embedding(1, self.feature_dim)  # Triadic relation token position

        # ========== Context Encoder (ACT-DP-TP style) ==========
        # Context encoder processes [vision_patches, state] -> enriched features
        # No CLS token - following ACT-DP-TP diffusion encoder design
        encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
            activation='gelu',
            norm_first=True,
        )
        self.context_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=encoder_depth,
        )

        # Action query position embedding (learnable, aligned with ACT-DP-TP)
        self.action_pos_embed = nn.Embedding(horizon, self.feature_dim)

        # Action embedding (for noised actions) - multi-layer for better encoding
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep embedding (for diffusion timestep, will be concat as token to memory)
        self.timestep_embed = nn.Sequential(
            nn.Linear(256, self.feature_dim),
            nn.SiLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        # Timestep position embedding (learned, for timestep token - ACT-DP-TP style)
        self.timestep_pos_embed = nn.Embedding(1, self.feature_dim)

        # Custom Transformer decoder (supports separate query_pos and memory_pos)
        decoder_layer = TransformerDecoderLayerWithPE(
            d_model=self.feature_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = TransformerDecoderWithPE(
            decoder_layer,
            num_layers=decoder_depth,
        )

        # Output projection for actions
        self.action_head = nn.Linear(self.feature_dim, action_dim)

        # ========== Future Pi3 Feature Prediction ==========
        self.pi3_query_height = self.pi3_grid_height
        self.pi3_query_width = self.pi3_grid_width
        self.num_pi3_queries = self._mode_token_count(self.future_target_mode, for_future=True)

        if self.predict_future_pi3:
            # Keep these parameter names/shapes for vanilla checkpoint compatibility.
            self.pi3_query_embed = nn.Parameter(
                torch.randn(self.pi3_query_height, self.pi3_query_width, self.feature_dim)
            )
            self.pi3_query_pos_embed = nn.Parameter(
                torch.randn(self.pi3_query_height, self.pi3_query_width, self.feature_dim)
            )
            self.pi3_feature_head = nn.Linear(self.feature_dim, pi3_num_views * pi3_feature_dim)

            cprint(f"  [Future Pi3] prediction enabled ({self.future_target_mode})", "yellow")
            cprint(f"  [Future Pi3] queries: {self.num_pi3_queries}", "yellow")
            cprint(f"  [Future Pi3] output per query: {pi3_num_views} views * {pi3_feature_dim} dim", "yellow")
        else:
            cprint(f"  [Future Pi3] prediction disabled ({self.future_target_mode})", "yellow")

        # Noise scheduler
        self.noise_scheduler = noise_scheduler

        # Normalizer
        self.normalizer = LinearNormalizer()

        # Store config
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.generative_mode = str(generative_mode).lower()
        if self.generative_mode not in ("diffusion", "flow_matching"):
            raise ValueError(
                f"Unsupported generative_mode={generative_mode!r}. "
                "Expected 'diffusion' or 'flow_matching'."
            )
        self.flow_cfg = self._to_plain_dict(flow_matching)
        self.flow_source_mode = str(self.flow_cfg.get("source_mode", "hold")).lower()
        if self.flow_source_mode not in ("hold", "previous_action", "zeros", "noise"):
            raise ValueError(
                f"Unsupported flow_matching.source_mode={self.flow_source_mode!r}. "
                "Expected hold, previous_action, zeros, or noise."
            )
        self.flow_lambda_min = float(self.flow_cfg.get("lambda_min", 0.0))
        self.flow_lambda_max = float(self.flow_cfg.get("lambda_max", 1.0))
        if not self.flow_lambda_min < self.flow_lambda_max:
            raise ValueError(
                "flow_matching.lambda_min must be smaller than flow_matching.lambda_max, "
                f"got {self.flow_lambda_min} and {self.flow_lambda_max}."
            )
        flow_num_inference_steps = self.flow_cfg.get("num_inference_steps", None)
        if flow_num_inference_steps is None:
            flow_num_inference_steps = self.num_inference_steps
        self.flow_num_inference_steps = int(flow_num_inference_steps)
        if self.flow_num_inference_steps <= 0:
            raise ValueError("flow_matching.num_inference_steps must be positive.")
        self.flow_action_source_noise_std = float(self.flow_cfg.get("action_source_noise_std", 0.0))
        self.flow_prediction_type = str(self.flow_cfg.get("prediction_type", "velocity")).lower()
        if self.flow_prediction_type != "velocity":
            raise ValueError(
                f"Unsupported flow_matching.prediction_type={self.flow_prediction_type!r}. "
                "Only velocity prediction is implemented."
            )

        cprint("[GAP] Configuration:", "cyan")
        cprint(f"  Generative mode: {self.generative_mode}", "cyan")
        if self.generative_mode == "flow_matching":
            cprint(
                "  Flow matching: "
                f"source={self.flow_source_mode}, "
                f"lambda=[{self.flow_lambda_min}, {self.flow_lambda_max}], "
                f"steps={self.flow_num_inference_steps}",
                "cyan",
            )
        cprint(f"  Vision feature dim: {vision_feat_dim}", "cyan")
        cprint(f"  State embed dim: {state_embed_dim}", "cyan")
        cprint(f"  Triadic token: {self.use_triadic_token} ({triadic_mode}, dim={self.triadic_dim})", "cyan")
        cprint(f"  Interaction field: {self.use_interaction_field} ({self.interaction_mode})", "cyan")
        cprint(f"  Encoder depth: {encoder_depth} (Feature encoding)", "cyan")
        cprint(f"  Decoder depth: {decoder_depth} (Action denoising)", "cyan")
        cprint(f"  Action dim: {action_dim}", "cyan")
        cprint(f"  Horizon: {horizon}", "cyan")
        cprint(f"  N obs steps: {n_obs_steps}", "cyan")
        cprint(f"  N action steps: {n_action_steps}", "cyan")
        cprint(f"  [Architecture] DinoV3 + Transformer Encoder + Diffusion Decoder (ACT-DP-TP style)", "yellow")
        cprint(f"  [Feature Encoding] Encoder processes [vision, state] -> enriched features", "yellow")
        cprint(f"  [Positional Encoding] Vision: 2D sinusoidal (separate from features)", "green")
        cprint(f"  [Positional Encoding] State: Learned (separate from features)", "green")
        cprint(f"  [Positional Encoding] Action query: Learnable embedding", "green")
        cprint(f"  [Positional Encoding] Timestep: Learned (for timestep token)", "green")
        cprint(f"  [Diffusion Timestep] Concat as token to memory (ACT-DP-TP 'cat' mode)", "green")

        print_params(self)

    @staticmethod
    def _to_plain_dict(config: Optional[Any]) -> Dict[str, Any]:
        if config is None:
            return {}
        if isinstance(config, Mapping):
            return dict(config)
        if hasattr(config, "items"):
            return dict(config.items())
        data = {}
        for key in dir(config):
            if key.startswith("_"):
                continue
            try:
                value = getattr(config, key)
            except Exception:
                continue
            if not callable(value):
                data[key] = value
        return data

    def _mode_token_count(self, mode: str, *, for_future: bool) -> int:
        if mode in ("none", "dino_only", "pi3_none", "interaction_state"):
            return 0
        if mode == "pi3_pooled":
            grid = self.future_pool_grid if for_future else self.latent_pool_grid
            return int(grid[0] * grid[1])
        if mode == "pi3_compressed":
            return self.future_num_compressed_tokens if for_future else self.latent_num_compressed_tokens
        if mode == "pi3_random_tokens":
            return self.pi3_random_num_tokens
        return self.pi3_total_tokens

    def _full_pi3_pos(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.get_2d_sinusoidal_positional_encoding(
            height=self.pi3_grid_height,
            width=self.pi3_grid_width,
            embedding_dim=self.feature_dim,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1, -1)

    def _dino_pos(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.get_2d_sinusoidal_positional_encoding(
            height=self.dinov3_grid_height,
            width=self.dinov3_grid_width,
            embedding_dim=self.feature_dim,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1, -1)

    def _mask_for_mode(self, obs_dict: Dict[str, torch.Tensor], mode: str, *, future: bool = False) -> torch.Tensor:
        key = mask_key_for_mode(mode)
        if key is None:
            raise ValueError(f"Mode {mode!r} does not use a token mask.")
        lookup_key = f"future_{key}" if future else key
        if lookup_key not in obs_dict:
            raise KeyError(
                f"{mode} requires token mask {lookup_key!r}, but it is missing. "
                "Run preprocessing with real object/hand masks or use a non-mask latent mode."
            )
        return obs_dict[lookup_key]

    def _masked_pool_tokens(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"DINO tokens must be shaped [B, N, D], got {tuple(tokens.shape)}.")
        if mask.ndim == 2:
            mask_flat = mask
        elif mask.ndim == 3:
            mask_flat = mask.reshape(mask.shape[0], -1)
        else:
            raise ValueError(f"Interaction mask must be shaped [B, N] or [B, V, N], got {tuple(mask.shape)}.")

        if mask_flat.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"Interaction mask batch size {mask_flat.shape[0]} does not match token batch size {tokens.shape[0]}."
            )
        if mask_flat.shape[1] != tokens.shape[1]:
            if tokens.shape[1] % mask_flat.shape[1] != 0:
                raise ValueError(
                    f"Interaction mask token count {mask_flat.shape[1]} does not match DINO token count "
                    f"{tokens.shape[1]} and cannot be repeated across views."
                )
            repeat = tokens.shape[1] // mask_flat.shape[1]
            mask_flat = mask_flat.unsqueeze(1).expand(-1, repeat, -1).reshape(mask_flat.shape[0], -1)

        mask_flat = mask_flat.to(device=tokens.device, dtype=tokens.dtype)
        denom = mask_flat.sum(dim=1, keepdim=True).clamp_min(self.interaction_eps)
        return (tokens * mask_flat.unsqueeze(-1)).sum(dim=1) / denom

    def _current_eef_masks_from_obs(
        self,
        dinov3_tokens: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        required = (
            "dino_eef_uv",
            "dino_eef_valid",
            "dino_eef_region_mask",
            "dino_pair_region_mask",
        )
        missing = [key for key in required if key not in obs_dict]
        if missing:
            raise KeyError(
                "current_eef interaction field requires obs keys "
                f"{', '.join(required)}, missing: {', '.join(missing)}. "
                "Run scripts/generate_dino_eef_region_masks.py and enable dataset interaction fields."
            )

        dino_eef_uv = obs_dict["dino_eef_uv"]
        eef_valid = obs_dict["dino_eef_valid"].to(device=dinov3_tokens.device, dtype=dinov3_tokens.dtype)
        eef_mask = obs_dict["dino_eef_region_mask"].to(device=dinov3_tokens.device, dtype=dinov3_tokens.dtype)
        pair_mask = obs_dict["dino_pair_region_mask"].to(device=dinov3_tokens.device, dtype=dinov3_tokens.dtype)

        if dino_eef_uv.ndim != 4 or dino_eef_uv.shape[2:] != (2, 2):
            raise ValueError(f"dino_eef_uv must be shaped [B, V, 2, 2], got {tuple(dino_eef_uv.shape)}.")
        if eef_mask.ndim != 4 or eef_mask.shape[2] != 2:
            raise ValueError(
                "dino_eef_region_mask must be shaped [B, V, 2, N], "
                f"got {tuple(eef_mask.shape)}."
            )
        if eef_valid.ndim != 3 or eef_valid.shape[2] != 2:
            raise ValueError(f"dino_eef_valid must be shaped [B, V, 2], got {tuple(eef_valid.shape)}.")
        if pair_mask.ndim != 3:
            raise ValueError(f"dino_pair_region_mask must be shaped [B, V, N], got {tuple(pair_mask.shape)}.")

        left_mask = eef_mask[:, :, 0, :] * eef_valid[:, :, 0:1]
        right_mask = eef_mask[:, :, 1, :] * eef_valid[:, :, 1:2]
        both_valid = eef_valid[:, :, 0] * eef_valid[:, :, 1]
        pair_mask = pair_mask * both_valid.unsqueeze(-1)
        return left_mask, right_mask, pair_mask

    def _interaction_tokens_from_masks(
        self,
        dinov3_tokens: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
        pair_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled_tokens = [
            self._masked_pool_tokens(dinov3_tokens, left_mask),
            self._masked_pool_tokens(dinov3_tokens, right_mask),
        ]
        type_ids = [0, 1]

        if self.interaction_use_pair_token:
            if pair_mask is None:
                raise ValueError("Pair interaction token is enabled but pair_mask is None.")
            pooled_tokens.append(self._masked_pool_tokens(dinov3_tokens, pair_mask))
            type_ids.append(2)

        token_tensor = torch.stack(pooled_tokens, dim=1)
        type_ids_tensor = torch.tensor(type_ids, device=dinov3_tokens.device, dtype=torch.long)
        token_tensor = token_tensor + self.interaction_type_embed(type_ids_tensor).unsqueeze(0)
        pos_tensor = self.interaction_pos_embed(type_ids_tensor).unsqueeze(0).expand(dinov3_tokens.shape[0], -1, -1)
        return token_tensor, pos_tensor

    def _token_grid_xy(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(self.interaction_grid_height, device=device, dtype=dtype),
            torch.arange(self.interaction_grid_width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        return grid.view(1, 1, 1, -1, 2).expand(batch_size, -1, -1, -1, -1)

    def _interaction_progress(
        self,
        timestep: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        timestep = timestep.to(device=device, dtype=dtype).reshape(batch_size)
        max_train_timesteps = float(getattr(self.noise_scheduler.config, "num_train_timesteps", 100))
        return (1.0 - timestep / max(max_train_timesteps, 1.0)).clamp(0.0, 1.0)

    def _flow_lambda_to_timestep(
        self,
        flow_lambda: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if flow_lambda.ndim == 0:
            flow_lambda = flow_lambda.expand(batch_size)
        progress = flow_lambda.to(device=device, dtype=dtype).reshape(batch_size).clamp(0.0, 1.0)
        max_train_timesteps = float(getattr(self.noise_scheduler.config, "num_train_timesteps", 100))
        return (1.0 - progress) * max_train_timesteps

    def _sample_flow_lambdas(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        span = self.flow_lambda_max - self.flow_lambda_min
        return torch.rand(batch_size, device=device, dtype=dtype) * span + self.flow_lambda_min

    def _agent_pos_action_source(
        self,
        agent_pos: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        agent_pos = agent_pos.to(device=device, dtype=dtype)
        if agent_pos.ndim != 2 or agent_pos.shape[0] != batch_size:
            raise ValueError(
                "agent_pos source must be shaped [B, state_dim], "
                f"got {tuple(agent_pos.shape)} for batch_size={batch_size}."
            )
        if agent_pos.shape[-1] != self.action_dim:
            raise ValueError(
                "flow_matching.source_mode='hold' requires agent_pos dim to match action_dim, "
                f"got agent_pos={agent_pos.shape[-1]} and action_dim={self.action_dim}."
            )
        return agent_pos.unsqueeze(1).expand(batch_size, self.horizon, self.action_dim).clone()

    def _flow_source_actions(
        self,
        obs_dict: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator=None,
    ) -> torch.Tensor:
        mode = self.flow_source_mode
        if mode == "zeros":
            source = torch.zeros(batch_size, self.horizon, self.action_dim, device=device, dtype=dtype)
        elif mode == "noise":
            source = torch.randn(
                batch_size,
                self.horizon,
                self.action_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        elif mode == "previous_action":
            source = None
            if obs_dict is not None:
                for key in ("previous_action", "prev_action", "last_action"):
                    prev_action = obs_dict.get(key)
                    if prev_action is None:
                        continue
                    prev_action = prev_action.to(device=device, dtype=dtype)
                    if prev_action.ndim == 2:
                        if prev_action.shape != (batch_size, self.action_dim):
                            raise ValueError(
                                f"{key} must be shaped [B, action_dim], got {tuple(prev_action.shape)}."
                            )
                        source = prev_action.unsqueeze(1).expand(batch_size, self.horizon, self.action_dim).clone()
                    elif prev_action.ndim == 3:
                        if prev_action.shape[0] != batch_size or prev_action.shape[-1] != self.action_dim:
                            raise ValueError(
                                f"{key} must be shaped [B, H, action_dim], got {tuple(prev_action.shape)}."
                            )
                        if prev_action.shape[1] == self.horizon:
                            source = prev_action.clone()
                        elif prev_action.shape[1] == 1:
                            source = prev_action.expand(batch_size, self.horizon, self.action_dim).clone()
                        else:
                            raise ValueError(
                                f"{key} horizon must be 1 or {self.horizon}, got {prev_action.shape[1]}."
                            )
                    else:
                        raise ValueError(f"{key} must be rank 2 or 3, got {tuple(prev_action.shape)}.")
                    break
            if source is None:
                if obs_dict is None or "agent_pos" not in obs_dict:
                    raise KeyError(
                        "flow_matching.source_mode='previous_action' needs previous_action/prev_action/last_action "
                        "or agent_pos for hold fallback."
                    )
                source = self._agent_pos_action_source(obs_dict["agent_pos"], batch_size, device, dtype)
        elif mode == "hold":
            if obs_dict is None or "agent_pos" not in obs_dict:
                raise KeyError("flow_matching.source_mode='hold' requires obs['agent_pos'].")
            source = self._agent_pos_action_source(obs_dict["agent_pos"], batch_size, device, dtype)
        else:
            raise ValueError(f"Unsupported flow source mode {mode!r}.")

        if self.flow_action_source_noise_std > 0.0 and mode != "noise":
            source = source + self.flow_action_source_noise_std * torch.randn(
                source.shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        return source

    def _sigma_from_progress(self, progress: torch.Tensor, start: float, end: float) -> torch.Tensor:
        return end + (1.0 - progress) * (start - end)

    def _gaussian_masks_from_uv(
        self,
        pred_uv: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pred_uv.ndim != 5 or pred_uv.shape[-2:] != (2, 2):
            raise ValueError(f"pred_uv must be shaped [B, H, V, 2, 2], got {tuple(pred_uv.shape)}.")
        B = pred_uv.shape[0]
        grid = self._token_grid_xy(B, pred_uv.device, pred_uv.dtype)
        if sigma is None:
            sigma = torch.ones(B, device=pred_uv.device, dtype=pred_uv.dtype)
        sigma = sigma.to(device=pred_uv.device, dtype=pred_uv.dtype).reshape(B, 1, 1, 1).clamp_min(self.interaction_eps)
        valid_t = None
        if valid is not None:
            valid_t = valid.to(device=pred_uv.device, dtype=pred_uv.dtype).clamp(0.0, 1.0)

        masks = []
        for arm_idx in (0, 1):
            arm_uv = pred_uv[:, :, :, arm_idx, :].unsqueeze(-2)
            dist2 = (grid - arm_uv).pow(2).sum(dim=-1)
            mask = torch.exp(-0.5 * dist2 / sigma.pow(2))
            if valid_t is not None:
                mask = mask * valid_t[:, :, :, arm_idx].unsqueeze(-1)
            masks.append(mask.amax(dim=1))
        return masks[0], masks[1]

    def _pair_masks_from_uv(
        self,
        pred_uv: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pred_uv.ndim != 5 or pred_uv.shape[-2:] != (2, 2):
            raise ValueError(f"pred_uv must be shaped [B, H, V, 2, 2], got {tuple(pred_uv.shape)}.")
        B = pred_uv.shape[0]
        grid = self._token_grid_xy(B, pred_uv.device, pred_uv.dtype)
        if sigma is None:
            sigma = torch.ones(B, device=pred_uv.device, dtype=pred_uv.dtype)
        sigma = sigma.to(device=pred_uv.device, dtype=pred_uv.dtype).reshape(B, 1, 1, 1).clamp_min(self.interaction_eps)

        left = pred_uv[:, :, :, 0, :]
        right = pred_uv[:, :, :, 1, :]
        segment = right - left
        length2 = segment.pow(2).sum(dim=-1, keepdim=True).clamp_min(self.interaction_eps)
        point_delta = grid - left.unsqueeze(-2)
        t = (point_delta * segment.unsqueeze(-2)).sum(dim=-1) / length2
        t = t.clamp(0.0, 1.0)
        closest = left.unsqueeze(-2) + t.unsqueeze(-1) * segment.unsqueeze(-2)
        dist2 = (grid - closest).pow(2).sum(dim=-1)
        mask = torch.exp(-0.5 * dist2 / sigma.pow(2))
        if valid is not None:
            valid_t = valid.to(device=pred_uv.device, dtype=pred_uv.dtype).clamp(0.0, 1.0)
            both_valid = valid_t[:, :, :, 0] * valid_t[:, :, :, 1]
            mask = mask * both_valid.unsqueeze(-1)
        return mask.amax(dim=1)

    def _compute_current_eef_interaction_tokens(
        self,
        dinov3_tokens: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.interaction_mode != "current_eef":
            raise ValueError(f"Unsupported interaction_field.mode={self.interaction_mode!r}.")
        left_mask, right_mask, pair_mask = self._current_eef_masks_from_obs(dinov3_tokens, obs_dict)
        return self._interaction_tokens_from_masks(dinov3_tokens, left_mask, right_mask, pair_mask)

    def _compute_interaction_tokens(
        self,
        dinov3_tokens: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
        mid_actions: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        agent_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.interaction_mode == "current_eef":
            left_mask, right_mask, pair_mask = self._current_eef_masks_from_obs(dinov3_tokens, obs_dict)
            tokens, pos = self._interaction_tokens_from_masks(dinov3_tokens, left_mask, right_mask, pair_mask)
            return tokens, pos, {}

        if self.interaction_mode != "action_uv":
            raise ValueError(f"Unsupported interaction_field.mode={self.interaction_mode!r}.")
        if self.action_to_uv_head is None:
            raise RuntimeError("interaction_field.mode='action_uv' requires ActionToUVHead to be initialized.")
        if mid_actions is None or timestep is None or agent_pos is None:
            raise ValueError("action_uv interaction field requires mid_actions, timestep, and agent_pos.")

        B = dinov3_tokens.shape[0]
        progress = self._interaction_progress(timestep, B, mid_actions.device, mid_actions.dtype)
        pred_uv = self.action_to_uv_head(mid_actions, agent_pos, progress)
        sigma = self._sigma_from_progress(
            progress,
            self.interaction_sigma_start,
            self.interaction_sigma_end,
        )
        pair_sigma = self._sigma_from_progress(
            progress,
            self.interaction_pair_sigma_start,
            self.interaction_pair_sigma_end,
        )
        left_mask, right_mask = self._gaussian_masks_from_uv(pred_uv, sigma=sigma)
        pair_mask = self._pair_masks_from_uv(pred_uv, sigma=pair_sigma) if self.interaction_use_pair_token else None

        if self.interaction_use_current_eef_fallback:
            current_left, current_right, current_pair = self._current_eef_masks_from_obs(dinov3_tokens, obs_dict)
            fallback_weight = self.interaction_current_eef_fallback_weight
            left_mask = torch.maximum(left_mask, fallback_weight * current_left)
            right_mask = torch.maximum(right_mask, fallback_weight * current_right)
            if pair_mask is not None:
                pair_mask = torch.maximum(pair_mask, fallback_weight * current_pair)

        tokens, pos = self._interaction_tokens_from_masks(dinov3_tokens, left_mask, right_mask, pair_mask)
        aux = {
            "pred_uv": pred_uv,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "progress": progress,
        }
        if pair_mask is not None:
            aux["pair_mask"] = pair_mask
        return tokens, pos, aux

    def _append_interaction_tokens(
        self,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        context: Dict[str, Any],
        mid_actions: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        agent_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_interaction_field or not self.interaction_append_tokens:
            context["interaction_aux"] = {}
            return memory, memory_pos
        interaction_tokens, interaction_pos, aux = self._compute_interaction_tokens(
            context["dinov3_tokens"],
            context["obs_dict"],
            mid_actions=mid_actions,
            timestep=timestep,
            agent_pos=agent_pos,
        )
        context["interaction_aux"] = aux
        memory = torch.cat([memory, interaction_tokens], dim=1)
        memory_pos = torch.cat([memory_pos, interaction_pos], dim=1)
        return memory, memory_pos

    def _apply_pi3_latent_mode(
        self,
        pi3_features: torch.Tensor,
        obs_dict: Dict[str, torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mode = self.latent_mode
        B = pi3_features.shape[0]
        device = pi3_features.device
        if mode in ("dino_only", "pi3_none"):
            return None, None
        if mode == "pi3_full":
            return flatten_pi3_tokens(pi3_features), self._full_pi3_pos(B, device)
        if mode == "pi3_token_dropout":
            dropped = apply_token_dropout(pi3_features, self.pi3_token_dropout_rate, self.training)
            return flatten_pi3_tokens(dropped), self._full_pi3_pos(B, device)
        if mode == "pi3_pooled":
            pooled = pool_pi3_grid(
                pi3_features,
                (self.pi3_grid_height, self.pi3_grid_width),
                self.latent_pool_grid,
            )
            pos = self.get_2d_sinusoidal_positional_encoding(
                height=self.latent_pool_grid[0],
                width=self.latent_pool_grid[1],
                embedding_dim=self.feature_dim,
                device=device,
            ).unsqueeze(0).expand(B, -1, -1)
            return flatten_pi3_tokens(pooled), pos
        if mode == "pi3_random_tokens":
            selected = select_token_indices(pi3_features, self.pi3_random_indices)
            full_pos = self._full_pi3_pos(B, device)
            pos = full_pos.index_select(dim=1, index=self.pi3_random_indices.to(device))
            return flatten_pi3_tokens(selected), pos
        if mode == "pi3_compressed":
            compressed = self.pi3_latent_compressor(pi3_features)
            pos = self.pi3_latent_compressed_pos_embed.unsqueeze(0).expand(B, -1, -1)
            return compressed, pos
        if mode_requires_mask(mode):
            mask = self._mask_for_mode(obs_dict, mode)
            masked = apply_dense_token_mask(pi3_features, mask, mode=mode)
            return flatten_pi3_tokens(masked), self._full_pi3_pos(B, device)
        raise ValueError(f"Unsupported latent_mode={mode!r}")

    def _future_queries(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.future_target_mode
        full_query = self.pi3_query_embed.reshape(-1, self.feature_dim)
        full_pos = self.pi3_query_pos_embed.reshape(-1, self.feature_dim)
        if mode in (
            "pi3_full",
            "pi3_delta",
            "pi3_changed_tokens",
            "pi3_delta_changed_tokens",
            "pi3_token_dropout",
        ) or mode_requires_mask(mode):
            query = full_query
            pos = full_pos
        elif mode == "pi3_pooled":
            query = pool_grid_tokens(full_query, (self.pi3_grid_height, self.pi3_grid_width), self.future_pool_grid)
            pos = pool_grid_tokens(full_pos, (self.pi3_grid_height, self.pi3_grid_width), self.future_pool_grid)
        elif mode == "pi3_random_tokens":
            indices = self.pi3_random_indices.to(device)
            query = full_query.index_select(dim=0, index=indices)
            pos = full_pos.index_select(dim=0, index=indices)
        elif mode == "pi3_compressed":
            query = self.pi3_future_query_embed
            pos = self.pi3_future_query_pos_embed
        else:
            raise ValueError(f"Unsupported future_target_mode={mode!r} for Pi3 prediction.")
        return (
            query.unsqueeze(0).expand(batch_size, -1, -1),
            pos.unsqueeze(0).expand(batch_size, -1, -1),
        )

    def _prepare_future_pi3_target(
        self,
        future_pi3: torch.Tensor,
        batch: Dict[str, Any],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        mode = self.future_target_mode
        if mode == "pi3_full":
            target = future_pi3
            mask = None
        elif mode == "pi3_delta":
            target = future_pi3 - self._current_pi3_from_batch(batch, future_pi3)
            mask = None
        elif mode == "pi3_changed_tokens":
            current_pi3 = self._current_pi3_from_batch(batch, future_pi3)
            target = future_pi3
            mask = self._changed_token_mask(current_pi3, future_pi3)
        elif mode == "pi3_delta_changed_tokens":
            current_pi3 = self._current_pi3_from_batch(batch, future_pi3)
            target = future_pi3 - current_pi3
            mask = self._changed_token_mask(current_pi3, future_pi3)
        elif mode == "pi3_token_dropout":
            target = future_pi3
            mask = None
        elif mode == "pi3_pooled":
            target = pool_pi3_grid(
                future_pi3,
                (self.pi3_grid_height, self.pi3_grid_width),
                self.future_pool_grid,
            )
            mask = None
        elif mode == "pi3_random_tokens":
            target = select_token_indices(future_pi3, self.pi3_random_indices)
            mask = None
        elif mode == "pi3_compressed":
            # Keep the target projection fixed within the loss; otherwise the
            # target branch can move toward the prediction instead of defining
            # a stable compressed future-latent objective.
            with torch.no_grad():
                target = self.pi3_future_target_compressor(future_pi3)
            return target, None
        elif mode_requires_mask(mode):
            mask = self._mask_for_mode(batch, mode, future=True)
            target = apply_dense_token_mask(future_pi3, mask, mode=mode)
            mask = mask
        else:
            raise ValueError(f"Unsupported future_target_mode={mode!r} for future Pi3 target.")

        if target.ndim == 4:
            B, N_views, num_patches, pi3_dim = target.shape
            target = target.permute(0, 2, 1, 3).reshape(B, num_patches, N_views * pi3_dim)
        return target, mask

    def _current_pi3_from_batch(self, batch: Dict[str, Any], future_pi3: torch.Tensor) -> torch.Tensor:
        obs = batch.get("obs")
        if not isinstance(obs, dict) or "pi3_features" not in obs:
            raise KeyError(
                f"future_target_mode={self.future_target_mode!r} requires current "
                "batch['obs']['pi3_features'] to compute future-current deltas."
            )
        current_pi3 = obs["pi3_features"]
        if current_pi3.shape != future_pi3.shape:
            raise ValueError(
                "Current and future Pi3 feature shapes must match for trajectory-coupled "
                f"future targets, got current={tuple(current_pi3.shape)} and "
                f"future={tuple(future_pi3.shape)}."
            )
        return current_pi3

    def _changed_token_mask(self, current_pi3: torch.Tensor, future_pi3: torch.Tensor) -> torch.Tensor:
        if current_pi3.ndim != 4 or future_pi3.ndim != 4:
            raise ValueError(
                "Changed-token future modes require Pi3 features shaped [B, V, N, D], "
                f"got current={tuple(current_pi3.shape)} future={tuple(future_pi3.shape)}."
            )
        delta_norm = torch.linalg.vector_norm(future_pi3 - current_pi3, dim=-1)
        if self.changed_token_stopgrad_mask:
            delta_norm = delta_norm.detach()
        batch, views, tokens = delta_norm.shape
        flat = delta_norm.reshape(batch, views * tokens)

        if self.changed_token_topk is not None:
            topk = int(self.changed_token_topk)
            if topk <= 0:
                raise ValueError("future.changed_token_topk must be positive when set.")
            topk = min(topk, flat.shape[1])
            selected = torch.topk(flat, k=topk, dim=1).indices
            mask_flat = torch.zeros_like(flat, dtype=future_pi3.dtype)
            mask_flat.scatter_(1, selected, 1.0)
        else:
            percentile = min(max(float(self.changed_token_percentile), 0.0), 100.0)
            threshold = torch.quantile(flat.float(), percentile / 100.0, dim=1, keepdim=True)
            mask_flat = (flat >= threshold.to(flat.dtype)).to(dtype=future_pi3.dtype)

        empty = mask_flat.sum(dim=1, keepdim=True) == 0
        if empty.any():
            selected = torch.argmax(flat, dim=1, keepdim=True)
            fallback = torch.zeros_like(mask_flat)
            fallback.scatter_(1, selected, 1.0)
            mask_flat = torch.where(empty, fallback, mask_flat)
        return mask_flat.reshape(batch, views, tokens)

    def _future_loss(
        self,
        pi3_pred: torch.Tensor,
        future_target: torch.Tensor,
        future_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if future_mask is None:
            return F.mse_loss(pi3_pred, future_target, reduction="mean")
        if future_mask.ndim == 3:
            # [B, V, N] -> [B, N, V]
            mask = future_mask.to(device=pi3_pred.device, dtype=pi3_pred.dtype).permute(0, 2, 1)
            loss_feature_dim = pi3_pred.shape[-1]
            if loss_feature_dim % mask.shape[-1] != 0:
                raise ValueError(
                    f"Future prediction feature dim {loss_feature_dim} is not divisible by "
                    f"mask view count {mask.shape[-1]}."
                )
            per_view_dim = loss_feature_dim // mask.shape[-1]
            mask = mask.unsqueeze(-1).expand(-1, -1, -1, per_view_dim)
            mask = mask.reshape(mask.shape[0], mask.shape[1], loss_feature_dim)
        else:
            mask = future_mask.to(device=pi3_pred.device, dtype=pi3_pred.dtype).unsqueeze(-1)
        if mask.shape[1] != pi3_pred.shape[1]:
            raise ValueError(
                f"Future mask token count {mask.shape[1]} does not match prediction token count {pi3_pred.shape[1]}."
            )
        loss = (pi3_pred - future_target).pow(2)
        if mask.shape[-1] == 1 and loss.shape[-1] != 1:
            mask = mask.expand(-1, -1, loss.shape[-1])
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom

    def get_sinusoidal_timestep_embedding(self, timesteps, embedding_dim=256):
        """
        Generate sinusoidal timestep embeddings (for diffusion timestep)
        Args:
            timesteps: [B] tensor of timesteps
            embedding_dim: dimension of embedding
        Returns:
            embeddings: [B, embedding_dim]
        """
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1))
        return emb

    def get_sinusoidal_positional_encoding(self, seq_len, embedding_dim, device):
        """
        Generate sinusoidal positional encodings (for 1D sequence position)
        Args:
            seq_len: length of sequence (horizon)
            embedding_dim: dimension of embedding (feature_dim)
            device: torch device
        Returns:
            positional_encoding: [seq_len, embedding_dim]
        """
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float, device=device)
            * -(torch.log(torch.tensor(10000.0)) / embedding_dim)
        )

        pe = torch.zeros(seq_len, embedding_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def get_2d_sinusoidal_positional_encoding(self, height, width, embedding_dim, device, temperature=10000):
        """
        Generate 2D sinusoidal positional encodings for vision patches.
        Similar to DETR's PositionEmbeddingSine.

        Args:
            height: height of patch grid
            width: width of patch grid
            embedding_dim: dimension of embedding (must be even)
            device: torch device
            temperature: temperature for sinusoidal encoding

        Returns:
            pos: [height*width, embedding_dim] positional encodings
        """
        # Create coordinate grids
        y_embed = torch.arange(height, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, width)
        x_embed = torch.arange(width, dtype=torch.float32, device=device).unsqueeze(0).repeat(height, 1)

        # Normalize to [0, 1]
        y_embed = y_embed / height
        x_embed = x_embed / width

        # Scale to [0, 2π]
        y_embed = y_embed * 2 * torch.pi
        x_embed = x_embed * 2 * torch.pi

        # Generate sinusoidal embeddings
        # Split embedding_dim between x and y coordinates
        dim_t = torch.arange(embedding_dim // 4, dtype=torch.float32, device=device)
        dim_t = temperature ** (2 * dim_t / (embedding_dim // 2))

        pos_x = x_embed[:, :, None] / dim_t  # [H, W, D//4]
        pos_y = y_embed[:, :, None] / dim_t  # [H, W, D//4]

        # Interleave sin and cos: [H, W, D//4] -> [H, W, D//2]
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)

        # Concatenate x and y encodings: [H, W, D//2] + [H, W, D//2] -> [H, W, D]
        pos = torch.cat([pos_y, pos_x], dim=-1)  # [H, W, D]
        pos = pos.view(-1, embedding_dim)  # [H*W, D]

        return pos

    def encode_observations(self, obs_dict: Dict[str, torch.Tensor], return_context: bool = False):
        """
        Encode observations to memory context with transformer encoder.
        Uses pre-extracted DinoV3 and Pi3 features.

        Architecture:
        1. Load pre-extracted DinoV3 and Pi3 features
        2. Project features to common dimension
        3. Concatenate: [CLS, State, Triadic(optional), DinoV3 patches, Pi3 patches(optional)]
        4. Pass through encoder to get enriched memory
        5. Memory serves as context for decoder

        Args:
            obs_dict: dict with keys:
                - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                - 'agent_pos': [B, 14] robot state
                - 'triadic_state': [B, D_rel] relation state (if use_triadic_token=True)

        Returns:
            memory: [B, N_tokens, D] - encoder output features
            memory_pos: [B, N_tokens, D] - corresponding position encodings
        """
        agent_pos = obs_dict['agent_pos']  # [B, 14]
        dinov3_features = obs_dict['dinov3_features']  # [B, N_views, num_patches, D]
        B = agent_pos.shape[0]
        device = agent_pos.device

        # 1. Process DinoV3 features
        N_dinov3_views, N_dinov3_patches, _ = dinov3_features.shape[1:]
        # Flatten all views: [B, N_views, num_patches, D] -> [B, N_views*num_patches, D]
        dinov3_flat = dinov3_features.reshape(B, N_dinov3_views * N_dinov3_patches, -1)
        # Project features
        dinov3_encoded = dinov3_flat  # [B, N_dinov3_views*num_patches, D]

        # 2. Process Pi3 features (if available)
        pi3_encoded = None
        if self.use_pi3_features and 'pi3_features' in obs_dict:
            pi3_features = obs_dict['pi3_features']  # [B, N_views, num_patches, 1024]
            pi3_encoded, pi3_pos = self._apply_pi3_latent_mode(pi3_features, obs_dict)
        else:
            pi3_pos = None

        # 3. Encode state
        state_features = self.state_encoder(agent_pos)  # [B, D]
        state_features = state_features.unsqueeze(1)  # [B, 1, D]

        # 4. Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        triadic_features = None
        if self.use_triadic_token:
            if "triadic_state" not in obs_dict:
                raise KeyError(
                    "use_triadic_token=True, but obs_dict does not contain 'triadic_state'. "
                    "Preprocess data with --triadic_mode or use policy.use_triadic_token=false."
                )
            triadic_state = obs_dict["triadic_state"]
            triadic_features = self.triadic_encoder(triadic_state).unsqueeze(1)  # [B, 1, D]

        # 5. Concatenate all features: [CLS, State, Triadic(optional), DinoV3_patches, Pi3_patches]
        features_list = [cls_tokens, state_features, dinov3_encoded]
        if triadic_features is not None:
            features_list = [cls_tokens, state_features, triadic_features, dinov3_encoded]
        if pi3_encoded is not None:
            features_list.append(pi3_encoded)

        encoder_input = torch.cat(features_list, dim=1)  # [B, N_total_tokens, D]

        # 6. Generate position embeddings for all tokens
        # CLS position
        cls_pos = self.cls_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        # State position
        state_pos = self.state_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        dinov3_pos = self._dino_pos(B, device)  # [B, N_dinov3_patches*N_dinov3_views, D]
        pos_list = [cls_pos, state_pos]
        if self.use_triadic_token:
            triadic_pos = self.triadic_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
            pos_list.append(triadic_pos)
        pos_list.append(dinov3_pos)
        encoder_pos = torch.cat(pos_list, dim=1)  # [B, N_tokens, D]

        if pi3_encoded is not None:
            encoder_pos = torch.cat([encoder_pos, pi3_pos], dim=1)  # [B, N_tokens, D]
        
        # 7. Pass through transformer encoder
        encoder_output = self.context_encoder(encoder_input, pos=encoder_pos)  # [B, N_tokens, D]

        # 8. Use entire encoder output as memory
        memory = encoder_output  # [B, N_tokens, D]
        memory_pos = encoder_pos  # [B, N_tokens, D]

        if return_context:
            return {
                "memory": memory,
                "memory_pos": memory_pos,
                "dinov3_tokens": dinov3_encoded,
                "dinov3_pos": dinov3_pos,
                "agent_pos": agent_pos,
                "obs_dict": obs_dict,
            }

        return memory, memory_pos

    def forward_diffusion(
        self,
        noised_actions: torch.Tensor,
        timestep: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Forward pass of denoising model (ACT-DP-TP 'cat' mode)

        Timestep is concatenated as an additional token to memory (not added),
        with its own learned position embedding.

        Args:
            noised_actions: [B, horizon, action_dim] noised actions
            timestep: [B] diffusion timestep
            memory: [B, N_memory, D] memory features (without position)
            memory_pos: [B, N_memory, D] memory position encodings
            context: optional dict returned by encode_observations(..., return_context=True)

        Returns:
            model_output: [B, horizon, action_dim] predicted output
                (noise if prediction_type='epsilon', clean sample if 'sample')
        """
        if context is not None:
            memory = context["memory"]
            memory_pos = context["memory_pos"]
            memory, memory_pos = self._append_interaction_tokens(
                memory,
                memory_pos,
                context,
                mid_actions=noised_actions,
                timestep=timestep,
                agent_pos=context["agent_pos"],
            )
        if memory is None or memory_pos is None:
            raise ValueError("forward_diffusion requires memory/memory_pos or context.")

        B = noised_actions.shape[0]
        device = noised_actions.device

        # 1. Embed noised actions (no position added yet)
        action_tgt = self.action_embed(noised_actions)  # [B, horizon, D]

        # 1b. Add future Pi3 queries (if enabled)
        if self.predict_future_pi3:
            pi3_tgt, _ = self._future_queries(B, device)
            # Concatenate action queries and pi3 queries
            tgt = torch.cat([action_tgt, pi3_tgt], dim=1)  # [B, horizon+num_pi3_queries, D]
        else:
            tgt = action_tgt

        # 2. Embed diffusion timestep as a token (ACT-DP-TP 'cat' mode)
        t_emb = self.get_sinusoidal_timestep_embedding(timestep)  # [B, 256]
        t_emb = self.timestep_embed(t_emb)  # [B, D]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, D]

        # Get timestep position embedding
        t_pos = self.timestep_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]

        # Concat timestep token to memory (as an additional token)
        memory = torch.cat([memory, t_emb], dim=1)  # [B, N_memory+1, D]
        memory_pos = torch.cat([memory_pos, t_pos], dim=1)  # [B, N_memory+1, D]

        # 3. Get learnable query position embeddings
        action_query_pos = self.action_pos_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, horizon, D]

        if self.predict_future_pi3:
            _, pi3_query_pos = self._future_queries(B, device)
            query_pos = torch.cat([action_query_pos, pi3_query_pos], dim=1)  # [B, horizon+num_pi3_queries, D]
        else:
            query_pos = action_query_pos

        # 4. Pass through custom transformer decoder with separate positional encodings
        # Position encodings are only added in Q/K, not in V (see TransformerDecoderLayerWithPE)
        decoded = self.transformer_decoder(
            tgt=tgt,              # action + pi3 features without position
            memory=memory,        # memory with timestep token concatenated
            memory_pos=memory_pos,  # memory position (includes timestep position)
            query_pos=query_pos,    # query position (action + pi3, separate, learnable)
        )  # [B, horizon+num_pi3_queries, D]

        # 5. Split decoded features and project to respective spaces
        action_decoded = decoded[:, :self.horizon, :]  # [B, horizon, D]
        model_output = self.action_head(action_decoded)  # [B, horizon, action_dim]

        if self.predict_future_pi3:
            pi3_decoded = decoded[:, self.horizon:, :]  # [B, num_pi3_queries, D]
            # Each query predicts features for all views: [B, num_pi3_queries, num_views*pi3_dim]
            pi3_output = self.pi3_feature_head(pi3_decoded)  # [B, height*width, num_views*pi3_dim]
            return model_output, pi3_output
        else:
            return model_output

    # ========= inference  ============
    def conditional_sample(
        self,
        memory: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        generator=None,
        **kwargs,
    ):
        """
        Sample actions using DDPM

        Args:
            memory: [B, N_memory, D] memory features
            memory_pos: [B, N_memory, D] memory position encodings

        Returns:
            actions: [B, horizon, action_dim] denoised actions
        """
        if self.generative_mode == "flow_matching":
            return self.conditional_flow_sample(
                memory=memory,
                memory_pos=memory_pos,
                context=context,
                generator=generator,
                **kwargs,
            )

        if context is not None:
            memory = context["memory"]
            memory_pos = context["memory_pos"]
        if memory is None or memory_pos is None:
            raise ValueError("conditional_sample requires memory/memory_pos or context.")

        B = memory.shape[0]
        device = memory.device
        dtype = memory.dtype

        # Start from pure noise
        actions = torch.randn(
            (B, self.horizon, self.action_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        # Set timesteps
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising
        pi3_features_pred = None
        for t in self.noise_scheduler.timesteps:
            # Prepare timestep
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            # Predict noise (and pi3 features if enabled)
            forward_output = self.forward_diffusion(
                noised_actions=actions,
                timestep=timestep,
                memory=memory,
                memory_pos=memory_pos,
                context=context,
            )

            if self.predict_future_pi3:
                model_output, pi3_features_pred = forward_output
            else:
                model_output = forward_output

            # Denoise
            actions = self.noise_scheduler.step(
                model_output,
                t,
                actions,
            ).prev_sample

        if self.predict_future_pi3:
            return actions, pi3_features_pred
        else:
            return actions

    def conditional_flow_sample(
        self,
        memory: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        context: Optional[Dict[str, Any]] = None,
        generator=None,
        **kwargs,
    ):
        """
        Sample actions with deterministic Euler integration of a velocity field.

        The same transformer decoder is reused, but its output is interpreted as
        d(action) / d(lambda) instead of a DDIM sample/noise prediction.
        """
        if context is not None:
            memory = context["memory"]
            memory_pos = context["memory_pos"]
        if memory is None or memory_pos is None:
            raise ValueError("conditional_flow_sample requires memory/memory_pos or context.")

        B = memory.shape[0]
        device = memory.device
        dtype = memory.dtype
        source_obs = context["obs_dict"] if context is not None else None
        actions = self._flow_source_actions(
            source_obs,
            batch_size=B,
            device=device,
            dtype=dtype,
            generator=generator,
        )

        lambda_grid = torch.linspace(
            self.flow_lambda_min,
            self.flow_lambda_max,
            self.flow_num_inference_steps + 1,
            device=device,
            dtype=dtype,
        )
        pi3_features_pred = None
        for step_idx in range(self.flow_num_inference_steps):
            lambda_start = lambda_grid[step_idx]
            lambda_end = lambda_grid[step_idx + 1]
            lambda_mid = 0.5 * (lambda_start + lambda_end)
            dt = lambda_end - lambda_start
            flow_lambda = torch.full((B,), lambda_mid, device=device, dtype=dtype)
            timestep = self._flow_lambda_to_timestep(flow_lambda, B, device, dtype)

            forward_output = self.forward_diffusion(
                noised_actions=actions,
                timestep=timestep,
                memory=memory,
                memory_pos=memory_pos,
                context=context,
            )
            if self.predict_future_pi3:
                velocity, pi3_features_pred = forward_output
            else:
                velocity = forward_output
            actions = actions + dt * velocity

        if self.predict_future_pi3:
            return actions, pi3_features_pred
        return actions

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict actions from observations (single frame)

        Args:
            obs_dict: dict with keys:
                - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                - 'agent_pos': [B, 14] robot state

        Returns:
            result: dict with 'action' key
        """
        # Normalize input
        nobs = self.normalizer.normalize(obs_dict)

        # Encode observations to memory context (single frame)
        if self.use_interaction_field or self.generative_mode == "flow_matching":
            context = self.encode_observations(nobs, return_context=True)
            memory = None
            memory_pos = None
        else:
            context = None
            memory, memory_pos = self.encode_observations(nobs)

        # Sample actions (and pi3 features if enabled)
        sample_output = self.conditional_sample(
            memory=memory,
            memory_pos=memory_pos,
            context=context,
            **self.kwargs,
        )

        if self.predict_future_pi3:
            naction_pred, pi3_pred = sample_output
        else:
            naction_pred = sample_output

        # Unnormalize
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # Extract action steps to execute
        action = action_pred[:, :self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }

        if self.predict_future_pi3:
            result["pi3_features_pred"] = pi3_pred

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _target_uv_tensors(
        self,
        batch: Dict[str, Any],
        pred_uv: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        if "target_dino_eef_uv_seq" not in batch or "target_dino_eef_valid_seq" not in batch:
            return None, None, {}

        target_uv = batch["target_dino_eef_uv_seq"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        valid = batch["target_dino_eef_valid_seq"].to(device=pred_uv.device, dtype=pred_uv.dtype)
        if target_uv.ndim != 5 or target_uv.shape[-2:] != (2, 2):
            raise ValueError(
                "target_dino_eef_uv_seq must be shaped [B, H, V, 2, 2], "
                f"got {tuple(target_uv.shape)}."
            )
        if valid.ndim != 4 or valid.shape[-1] != 2:
            raise ValueError(
                "target_dino_eef_valid_seq must be shaped [B, H, V, 2], "
                f"got {tuple(valid.shape)}."
            )
        if target_uv.shape[:4] != pred_uv.shape[:4] or valid.shape != pred_uv.shape[:4]:
            raise ValueError(
                "Action UV target shapes must align with predicted UV, got "
                f"pred={tuple(pred_uv.shape)}, target={tuple(target_uv.shape)}, valid={tuple(valid.shape)}."
            )

        valid = valid.clamp(0.0, 1.0)
        valid_count = valid.sum().clamp_min(1.0)
        target_x = target_uv[..., 0]
        target_y = target_uv[..., 1]
        oob = (
            (target_x < 0.0)
            | (target_x > float(self.interaction_grid_width - 1))
            | (target_y < 0.0)
            | (target_y > float(self.interaction_grid_height - 1))
        ).to(dtype=target_uv.dtype)
        metrics = {
            "target_uv_oob_fraction": float(((oob * valid).sum() / valid_count).detach().item()),
            "target_uv_valid_fraction": float(valid.detach().mean().item()),
        }
        if self.interaction_uv_loss_clamp_target:
            target_uv = target_uv.clone()
            target_uv[..., 0] = target_uv[..., 0].clamp(0.0, float(self.interaction_grid_width - 1))
            target_uv[..., 1] = target_uv[..., 1].clamp(0.0, float(self.interaction_grid_height - 1))
        return target_uv, valid, metrics

    def _progress_uv_weights(
        self,
        progress: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if progress is None:
            return torch.ones(batch_size, device=device, dtype=dtype)
        progress = progress.to(device=device, dtype=dtype).reshape(batch_size)
        denom = max(1.0e-6, 1.0 - self.interaction_uv_loss_min_progress)
        raw = ((progress - self.interaction_uv_loss_min_progress) / denom).clamp(0.0, 1.0)
        return raw.pow(self.interaction_uv_loss_progress_gamma)

    def _action_uv_loss_from_prediction(
        self,
        pred_uv: torch.Tensor,
        target_uv: torch.Tensor,
        valid: torch.Tensor,
        progress: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        progress_weight = self._progress_uv_weights(
            progress,
            pred_uv.shape[0],
            pred_uv.device,
            pred_uv.dtype,
        )
        per_point_loss = (pred_uv - target_uv).pow(2).mean(dim=-1)
        weighted_valid = valid * progress_weight.view(-1, 1, 1, 1)
        denom = weighted_valid.sum()
        numerator = (per_point_loss * weighted_valid).sum()
        if float(denom.detach().item()) <= 0.0:
            uv_loss = numerator * 0.0
        else:
            uv_loss = numerator / denom
        metrics = {
            "uv_progress_weight_mean": float(progress_weight.detach().mean().item()),
            "uv_progress_weight_min": float(progress_weight.detach().min().item()),
            "uv_progress_weight_max": float(progress_weight.detach().max().item()),
            "pred_uv_mean": float(pred_uv.detach().mean().item()),
            "pred_uv_std": float(pred_uv.detach().std().item()),
        }
        return uv_loss, metrics

    def _action_uv_supervision_loss(
        self,
        batch: Dict[str, Any],
        aux: Dict[str, torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        if self.interaction_mode != "action_uv" or "pred_uv" not in aux:
            return None, {}

        pred_uv = aux["pred_uv"]
        target_uv, valid, target_metrics = self._target_uv_tensors(batch, pred_uv)
        if target_uv is None or valid is None:
            return None, {}

        progress = aux.get("progress")
        uv_loss, metrics = self._action_uv_loss_from_prediction(pred_uv, target_uv, valid, progress)
        metrics.update(target_metrics)
        metrics["uv_loss"] = float(uv_loss.detach().item())
        metrics["uv_loss_weight"] = self.interaction_uv_loss_weight
        if progress is not None:
            metrics["interaction_progress_mean"] = float(progress.detach().mean().item())
        return uv_loss, metrics

    def _clean_action_uv_supervision_loss(
        self,
        batch: Dict[str, Any],
        nobs: Dict[str, torch.Tensor],
        nactions: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        if (
            self.interaction_mode != "action_uv"
            or self.action_to_uv_head is None
            or self.interaction_clean_uv_loss_weight <= 0.0
        ):
            return None, {}
        clean_progress = torch.full(
            (nactions.shape[0],),
            self.interaction_clean_uv_progress,
            device=nactions.device,
            dtype=nactions.dtype,
        )
        clean_pred_uv = self.action_to_uv_head(nactions, nobs["agent_pos"], clean_progress)
        target_uv, valid, _ = self._target_uv_tensors(batch, clean_pred_uv)
        if target_uv is None or valid is None:
            return None, {}
        clean_uv_loss, _ = self._action_uv_loss_from_prediction(clean_pred_uv, target_uv, valid, clean_progress)
        metrics = {
            "clean_uv_loss": float(clean_uv_loss.detach().item()),
            "clean_uv_loss_weight": self.interaction_clean_uv_loss_weight,
            "clean_pred_uv_mean": float(clean_pred_uv.detach().mean().item()),
            "clean_pred_uv_std": float(clean_pred_uv.detach().std().item()),
        }
        return clean_uv_loss, metrics

    def _mask_debug_metrics(self, aux: Dict[str, torch.Tensor]) -> Dict[str, float]:
        metrics = {}
        for name, metric_prefix in (
            ("left_mask", "left_mask"),
            ("right_mask", "right_mask"),
            ("pair_mask", "pair_mask"),
        ):
            mask = aux.get(name)
            if mask is None:
                continue
            mask_detached = mask.detach()
            metrics[f"{metric_prefix}_mass"] = float(mask_detached.sum(dim=-1).mean().item())
            metrics[f"{metric_prefix}_max"] = float(mask_detached.amax(dim=-1).mean().item())
        pred_uv = aux.get("pred_uv")
        if pred_uv is not None:
            distance = torch.linalg.vector_norm(
                pred_uv.detach()[:, :, :, 0, :] - pred_uv.detach()[:, :, :, 1, :],
                dim=-1,
            )
            metrics["left_right_uv_distance_mean"] = float(distance.mean().item())
        return metrics

    def _compute_flow_matching_loss(self, batch):
        """Compute optional flow-matching loss in normalized action space."""
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B = nactions.shape[0]

        if self.use_interaction_field:
            context = self.encode_observations(nobs, return_context=True)
            memory = None
            memory_pos = None
        else:
            context = None
            memory, memory_pos = self.encode_observations(nobs)

        flow_lambda = self._sample_flow_lambdas(B, nactions.device, nactions.dtype)
        source_actions = self._flow_source_actions(
            nobs,
            batch_size=B,
            device=nactions.device,
            dtype=nactions.dtype,
        )
        lambda_view = flow_lambda.view(B, 1, 1)
        mid_actions = source_actions + lambda_view * (nactions - source_actions)
        target_velocity = nactions - source_actions
        timesteps = self._flow_lambda_to_timestep(flow_lambda, B, nactions.device, nactions.dtype)

        forward_output = self.forward_diffusion(
            noised_actions=mid_actions,
            timestep=timesteps,
            memory=memory,
            memory_pos=memory_pos,
            context=context,
        )

        if self.predict_future_pi3:
            pred_velocity, pi3_pred = forward_output
        else:
            pred_velocity = forward_output

        flow_loss = F.mse_loss(pred_velocity, target_velocity, reduction="none")
        flow_loss = reduce(flow_loss, "b ... -> b (...)", "mean")
        flow_loss = flow_loss.mean()

        loss = flow_loss
        loss_dict = {
            "action_loss": flow_loss.item(),
            "flow_loss": flow_loss.item(),
            "flow_prediction_type": self.flow_prediction_type,
            "flow_source_mode": self.flow_source_mode,
            "flow_lambda_mean": float(flow_lambda.detach().mean().item()),
            "flow_lambda_min": float(flow_lambda.detach().min().item()),
            "flow_lambda_max": float(flow_lambda.detach().max().item()),
            "flow_source_std": float(source_actions.detach().std().item()),
        }

        if self.predict_future_pi3:
            if "future_pi3_features" not in batch:
                raise KeyError(
                    "future Pi3 loss is enabled, but batch does not contain "
                    "'future_pi3_features'. Set use_future_loss=false or provide Pi3 features."
                )
            future_pi3 = batch["future_pi3_features"]
            future_target, future_mask = self._prepare_future_pi3_target(future_pi3, batch)
            future_loss = self._future_loss(pi3_pred, future_target, future_mask)
            loss = loss + self.future_loss_weight * future_loss
            loss_dict["future_loss"] = future_loss.item()
            loss_dict["pi3_loss"] = future_loss.item()
            loss_dict["future_loss_mode"] = self.future_target_mode
            loss_dict["total_loss"] = loss.item()
        elif self.predict_interaction_state:
            raise NotImplementedError(
                "future_target_mode='interaction_state' requires dataset-provided "
                "future_interaction_state and a prediction head; current public data "
                "does not expose object pose/keypoints needed for this mode."
            )
        else:
            loss_dict["future_loss"] = 0.0
            loss_dict["future_loss_mode"] = "none"
            loss_dict["total_loss"] = loss.item()

        if self.interaction_mode == "action_uv":
            aux = context.get("interaction_aux", {}) if context is not None else {}
            uv_loss, uv_metrics = self._action_uv_supervision_loss(batch, aux)
            if uv_loss is not None:
                loss = loss + self.interaction_uv_loss_weight * uv_loss
                loss_dict.update(uv_metrics)
                loss_dict["total_loss"] = loss.item()
            clean_uv_loss, clean_uv_metrics = self._clean_action_uv_supervision_loss(batch, nobs, nactions)
            if clean_uv_loss is not None:
                loss = loss + self.interaction_clean_uv_loss_weight * clean_uv_loss
                loss_dict.update(clean_uv_metrics)
                loss_dict["total_loss"] = loss.item()
            if self.interaction_log_stats:
                loss_dict.update(self._mask_debug_metrics(aux))

        return loss, loss_dict

    def compute_loss(self, batch):
        """
        Compute diffusion loss (single frame per batch)

        Args:
            batch: dict with 'obs' and 'action'
                obs: dict with keys:
                    - 'dinov3_features': [B, N_views, num_patches, D] pre-extracted DinoV3 features
                    - 'pi3_features': [B, N_views, num_patches, 1024] pre-extracted Pi3 features (optional)
                    - 'agent_pos': [B, 14] robot state
                action: [B, horizon, action_dim]

        Returns:
            loss: scalar tensor
            loss_dict: dict with loss components
        """
        if self.generative_mode == "flow_matching":
            return self._compute_flow_matching_loss(batch)

        # Normalize
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B = nactions.shape[0]

        # Encode observations to memory (single frame)
        if self.use_interaction_field:
            context = self.encode_observations(nobs, return_context=True)
            memory = None
            memory_pos = None
        else:
            context = None
            memory, memory_pos = self.encode_observations(nobs)  # [B, N_memory, D]

        # Sample timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=nactions.device,
        ).long()

        # Sample noise
        noise = torch.randn_like(nactions)

        # Add noise to actions
        noised_actions = self.noise_scheduler.add_noise(nactions, noise, timesteps)

        # Predict noise (and pi3 features if enabled)
        forward_output = self.forward_diffusion(
            noised_actions=noised_actions,
            timestep=timesteps,
            memory=memory,
            memory_pos=memory_pos,
            context=context,
        )

        if self.predict_future_pi3:
            pred, pi3_pred = forward_output
        else:
            pred = forward_output

        # Action loss
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = nactions
        elif pred_type == "v_prediction":
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = (
                self.noise_scheduler.alpha_t[timesteps],
                self.noise_scheduler.sigma_t[timesteps],
            )
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * nactions
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        action_loss = F.mse_loss(pred, target, reduction="none")
        action_loss = reduce(action_loss, "b ... -> b (...)", "mean")
        action_loss = action_loss.mean()

        loss_dict = {
            "action_loss": action_loss.item(),
        }

        # Future latent prediction loss (if enabled)
        if self.predict_future_pi3:
            if "future_pi3_features" not in batch:
                raise KeyError(
                    "future Pi3 loss is enabled, but batch does not contain "
                    "'future_pi3_features'. Set use_future_loss=false or provide Pi3 features."
                )
            future_pi3 = batch["future_pi3_features"]  # [B, N_views, num_patches, pi3_dim]
            future_target, future_mask = self._prepare_future_pi3_target(future_pi3, batch)
            future_loss = self._future_loss(pi3_pred, future_target, future_mask)

            loss_dict["future_loss"] = future_loss.item()
            loss_dict["pi3_loss"] = future_loss.item()  # backward-compatible metric name
            loss_dict["future_loss_mode"] = self.future_target_mode

            loss = action_loss + self.future_loss_weight * future_loss
            loss_dict["total_loss"] = loss.item()
        elif self.predict_interaction_state:
            raise NotImplementedError(
                "future_target_mode='interaction_state' requires dataset-provided "
                "future_interaction_state and a prediction head; current public data "
                "does not expose object pose/keypoints needed for this mode."
            )
        else:
            loss = action_loss
            loss_dict["future_loss"] = 0.0
            loss_dict["future_loss_mode"] = "none"
            loss_dict["total_loss"] = loss.item()

        if self.interaction_mode == "action_uv":
            aux = context.get("interaction_aux", {}) if context is not None else {}
            uv_loss, uv_metrics = self._action_uv_supervision_loss(batch, aux)
            if uv_loss is not None:
                loss = loss + self.interaction_uv_loss_weight * uv_loss
                loss_dict.update(uv_metrics)
                loss_dict["total_loss"] = loss.item()
            clean_uv_loss, clean_uv_metrics = self._clean_action_uv_supervision_loss(batch, nobs, nactions)
            if clean_uv_loss is not None:
                loss = loss + self.interaction_clean_uv_loss_weight * clean_uv_loss
                loss_dict.update(clean_uv_metrics)
                loss_dict["total_loss"] = loss.item()
            if self.interaction_log_stats:
                loss_dict.update(self._mask_debug_metrics(aux))

        return loss, loss_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
