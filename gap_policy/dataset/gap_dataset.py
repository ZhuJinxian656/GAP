"""
GAP Dataset
Loads pre-extracted DinoV3 and Pi3 features, state, and action from zarr format.
Features are already extracted during data processing to avoid runtime computation.
"""
import sys, os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(parent_directory, '..'))
sys.path.append(os.path.join(parent_directory, '../..'))

from typing import Dict
import torch
import numpy as np
import copy
import zarr
from gap_policy.common.pytorch_util import dict_apply
from gap_policy.common.replay_buffer import ReplayBuffer
from gap_policy.common.sampler import (
    SequenceSampler,
    get_val_mask,
    downsample_mask,
)
from gap_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from gap_policy.dataset.base_dataset import BaseDataset
from gap_policy.latent_modes import mask_key_for_mode


class GAPDataset(BaseDataset):
    """
    Dataset for GAP
    Loads pre-extracted DinoV3 features, Pi3 features, state (agent_pos), and action
    """

    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        task_name=None,
        use_pi3_features=True,  # Whether to load pi3 features
        model_3d="pi3",
        use_triadic_token=False,
        triadic_mode="disabled",
        latent_mode="pi3_full",
        use_future_loss=True,
        future_target_mode="pi3_full",
    ):
        super().__init__()
        self.task_name = task_name
        self.use_pi3_features = use_pi3_features
        self.use_triadic_token = bool(use_triadic_token) and triadic_mode != "disabled"
        self.latent_mode = latent_mode
        self.use_future_loss = bool(use_future_loss)
        self.future_target_mode = future_target_mode

        # Make path relative to this file
        current_file_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(current_file_path)
        zarr_path = os.path.join(parent_directory, zarr_path)

        # Load replay buffer with pre-extracted features, state, and action
        keys = ["dinov3_features", "state", "action"]
        if use_pi3_features:
            keys.append(f"{model_3d}_features")
        if self.use_triadic_token:
            keys.append("triadic_state")
        self.mask_keys = sorted(set(
            key
            for key in (
                mask_key_for_mode(latent_mode),
                mask_key_for_mode(future_target_mode),
            )
            if key is not None
        ))
        keys.extend(self.mask_keys)
        self.use_interaction_state = future_target_mode == "interaction_state"
        if self.use_interaction_state:
            keys.append("future_interaction_state")

        zarr_group = zarr.open(zarr_path, mode="r")
        available_keys = set(zarr_group["data"].array_keys())
        missing_keys = [key for key in keys if key not in available_keys]
        if missing_keys:
            reasons = []
            for key in missing_keys:
                if key in self.mask_keys:
                    reasons.append(
                        f"{key!r} is required by latent_mode={latent_mode!r} "
                        f"or future_target_mode={future_target_mode!r}."
                    )
                elif key == "future_interaction_state":
                    reasons.append(
                        "'future_interaction_state' is required by "
                        "future_target_mode='interaction_state'."
                    )
                else:
                    reasons.append(f"{key!r} is required by the GAP dataset configuration.")
            raise KeyError(
                "Requested zarr dataset is missing required arrays: "
                + ", ".join(missing_keys)
                + "\nAvailable data arrays: "
                + ", ".join(sorted(available_keys))
                + "\n"
                + "\n".join(reasons)
            )

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=keys
        )

        # Create train/val split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed
        )

        # Create sequence sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        print(f"GAP Dataset initialized:")
        print(f"  Task: {task_name}")
        print(f"  Total episodes: {self.replay_buffer.n_episodes}")
        print(f"  Train episodes: {train_mask.sum()}")
        print(f"  Val episodes: {val_mask.sum()}")
        print(f"  Horizon: {horizon}")
        print(f"  DinoV3 features shape: {self.replay_buffer['dinov3_features'].shape}")
        if use_pi3_features:
            print(f"  Pi3 features shape: {self.replay_buffer['pi3_features'].shape}")
        print(f"  State shape: {self.replay_buffer['state'].shape}")
        print(f"  Action shape: {self.replay_buffer['action'].shape}")
        if self.use_triadic_token:
            print(f"  Triadic state shape: {self.replay_buffer['triadic_state'].shape}")
        for key in self.mask_keys:
            print(f"  Mask {key} shape: {self.replay_buffer[key].shape}")
        if self.use_interaction_state:
            print(f"  Future interaction state shape: {self.replay_buffer['future_interaction_state'].shape}")

    def get_validation_dataset(self):
        """Create validation dataset with same parameters"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        """
        Get normalizer for action, agent_pos, dinov3_features, and pi3_features
        Features are already extracted and do not need normalization (identity transform)
        """
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],  # State is agent_pos
        }
        if self.use_triadic_token:
            data["triadic_state"] = self.replay_buffer["triadic_state"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # DinoV3 and Pi3 features use identity normalization (no transformation)
        # Features are already normalized during extraction
        normalizer["dinov3_features"] = self._get_identity_normalizer()
        if self.use_pi3_features:
            normalizer["pi3_features"] = self._get_identity_normalizer()
        for key in self.mask_keys:
            normalizer[key] = self._get_identity_normalizer()

        return normalizer

    def _get_identity_normalizer(self):
        """
        Identity normalization for pre-extracted features (no transformation)
        Input: x, Output: x (identity transform)
        """
        from gap_policy.model.common.normalizer import SingleFieldLinearNormalizer

        # Identity transform: scale=1, offset=0
        scale = np.array([1.0], dtype=np.float32)
        offset = np.array([0.0], dtype=np.float32)

        stat = {
            "min": np.array([0.0], dtype=np.float32),
            "max": np.array([1.0], dtype=np.float32),
            "mean": np.array([0.0], dtype=np.float32),
            "std": np.array([1.0], dtype=np.float32),
        }

        return SingleFieldLinearNormalizer.create_manual(
            scale=scale, offset=offset, input_stats_dict=stat
        )

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions for analysis"""
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        Convert raw sample to data dict (single frame + future pi3 features)
        Args:
            sample: dict with keys 'dinov3_features', 'pi3_features' (optional), 'state', 'action'
        Returns:
            data: dict with structure for training
        """
        # DinoV3 features [T, N_views, num_patches, embed_dim] - we only use first frame
        dinov3_features = sample["dinov3_features"].astype(np.float32)  # [T, N_views, num_patches, D]
        dinov3_features = dinov3_features[0]  # Take first frame: [N_views, num_patches, D]

        # Agent position (state) - only first frame
        agent_pos = sample["state"].astype(np.float32)  # [T, 14]
        agent_pos = agent_pos[0]  # Take first frame: [14]

        # Action sequence (full horizon for training)
        action = sample["action"].astype(np.float32)  # [T, 14]

        obs_dict = {
            "dinov3_features": dinov3_features,  # [N_views, num_patches, D] - single frame
            "agent_pos": agent_pos,  # [14] - single state
        }

        if self.use_triadic_token and "triadic_state" in sample:
            triadic_state = sample["triadic_state"].astype(np.float32)
            obs_dict["triadic_state"] = triadic_state[0]

        for key in self.mask_keys:
            mask = sample[key].astype(np.float32)
            obs_dict[key] = mask[0]
            data_key = f"future_{key}"
            # The same zarr mask array is indexed at the future timestep for
            # future-target masking. It is not a separate stored zarr key.
            sample_future = mask[-1]
            # data is created below; temporarily stash in obs_dict namespace.
            obs_dict[data_key] = sample_future

        # Add Pi3 features if available (current frame for encoding)
        if self.use_pi3_features and "pi3_features" in sample:
            # Pi3 features [T, N_views, num_patches, 1024] - we only use first frame for observation
            pi3_features = sample["pi3_features"].astype(np.float32)
            pi3_features_current = pi3_features[0]  # Take first frame: [N_views, num_patches, 1024]
            obs_dict["pi3_features"] = pi3_features_current

        data = {
            "obs": obs_dict,
            "action": action,  # [T, 14] - full action sequence
        }

        for key in self.mask_keys:
            data[f"future_{key}"] = obs_dict.pop(f"future_{key}")

        # Add future pi3 features (last frame of action chunk) for point map prediction
        if self.use_pi3_features and self.use_future_loss and "pi3_features" in sample:
            # Get the last frame's pi3 features as ground truth for prediction
            T = pi3_features.shape[0]
            future_pi3 = pi3_features[-1]  # Take last frame: [N_views, num_patches, 1024]
            data["future_pi3_features"] = future_pi3

        if self.use_interaction_state and "future_interaction_state" in sample:
            future_interaction = sample["future_interaction_state"].astype(np.float32)
            data["future_interaction_state"] = future_interaction[-1]

        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence sample
        Returns:
            torch_data: dict with obs, action, and future_pi3_features
                obs:
                    dinov3_features: [N_views, num_patches, D] float32
                    pi3_features: [N_views, num_patches, 1024] float32 (if use_pi3_features=True)
                    agent_pos: [14] float32
                    triadic_state: [D_rel] float32 (if use_triadic_token=True)
                action: [T, 14] float32
                future_pi3_features: [N_views, num_patches, 1024] float32 (if use_pi3_features=True)
                    - Ground truth pi3 features for the last frame of action chunk
                    - Used for point map prediction (PMP) auxiliary task
        """
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
