#!/usr/bin/env python
"""Synthetic zarr smoke tests for optional dataset fields."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import zarr

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.dataset.gap_dataset import GAPDataset  # noqa: E402


def write_zarr(path: str, include_mask: bool = False) -> None:
    root = zarr.group(path)
    data = root.create_group("data")
    meta = root.create_group("meta")
    steps = 8
    data.create_dataset("dinov3_features", data=np.random.randn(steps, 1, 6, 32).astype("float32"))
    data.create_dataset("pi3_features", data=np.random.randn(steps, 1, 20, 32).astype("float32"))
    data.create_dataset("state", data=np.random.randn(steps, 14).astype("float32"))
    data.create_dataset("action", data=np.random.randn(steps, 14).astype("float32"))
    if include_mask:
        mask = np.zeros((steps, 1, 20), dtype="float32")
        mask[..., :5] = 1
        data.create_dataset("pi3_object_hand_mask", data=mask)
    meta.create_dataset("episode_ends", data=np.array([steps], dtype=np.int64))


def main():
    tmp = tempfile.mkdtemp(prefix="gap_dataset_smoke_", dir="/tmp")
    try:
        zarr_path = os.path.join(tmp, "synthetic.zarr")
        write_zarr(zarr_path)

        vanilla = GAPDataset(
            zarr_path=zarr_path,
            horizon=4,
            pad_before=0,
            pad_after=0,
            max_train_episodes=1,
            use_pi3_features=True,
        )
        sample = vanilla[0]
        assert "future_pi3_features" in sample

        dino = GAPDataset(
            zarr_path=zarr_path,
            horizon=4,
            pad_before=0,
            pad_after=0,
            max_train_episodes=1,
            use_pi3_features=False,
            latent_mode="dino_only",
            use_future_loss=False,
            future_target_mode="none",
        )
        dino_sample = dino[0]
        assert "pi3_features" not in dino_sample["obs"]

        try:
            GAPDataset(
                zarr_path=zarr_path,
                horizon=4,
                pad_before=0,
                pad_after=0,
                max_train_episodes=1,
                use_pi3_features=True,
                latent_mode="pi3_object_hand",
            )
        except KeyError as exc:
            assert "pi3_object_hand_mask" in str(exc)
        else:
            raise AssertionError("Expected missing mask KeyError")

        zarr_with_mask = os.path.join(tmp, "synthetic_mask.zarr")
        write_zarr(zarr_with_mask, include_mask=True)
        masked = GAPDataset(
            zarr_path=zarr_with_mask,
            horizon=4,
            pad_before=0,
            pad_after=0,
            max_train_episodes=1,
            use_pi3_features=True,
            latent_mode="pi3_object_hand",
            future_target_mode="pi3_object_hand",
        )
        masked_sample = masked[0]
        assert "pi3_object_hand_mask" in masked_sample["obs"]
        assert "future_pi3_object_hand_mask" in masked_sample

        print("dataset optional-field smoke tests passed")
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
