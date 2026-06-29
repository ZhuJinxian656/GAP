#!/usr/bin/env python
"""Synthetic smoke test for scripts/check_dino_interaction_zarr.py."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np
import zarr

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from scripts.check_dino_interaction_zarr import check_zarr  # noqa: E402


def write_zarr(path: str) -> None:
    root = zarr.group(path)
    data = root.create_group("data")
    meta = root.create_group("meta")
    steps = 6
    num_tokens = 15 * 20
    data.create_dataset("dinov3_features", data=np.random.randn(steps, 1, num_tokens, 8).astype("float32"))
    data.create_dataset("state", data=np.random.randn(steps, 14).astype("float32"))
    data.create_dataset("action", data=np.random.randn(steps, 14).astype("float32"))

    dino_eef_uv = np.zeros((steps, 1, 2, 2), dtype="float32")
    dino_action_eef_uv = np.zeros((steps, 1, 2, 2), dtype="float32")
    dino_eef_valid = np.ones((steps, 1, 2), dtype="float32")
    dino_action_eef_valid = np.ones((steps, 1, 2), dtype="float32")
    dino_eef_region_mask = np.zeros((steps, 1, 2, num_tokens), dtype="float32")
    dino_pair_region_mask = np.zeros((steps, 1, num_tokens), dtype="float32")

    dino_eef_uv[:, :, 0, :] = np.array([2.0, 3.0], dtype="float32")
    dino_eef_uv[:, :, 1, :] = np.array([12.0, 8.0], dtype="float32")
    dino_action_eef_uv[:, :, 0, :] = np.array([3.0, 3.5], dtype="float32")
    dino_action_eef_uv[:, :, 1, :] = np.array([13.0, 8.5], dtype="float32")
    dino_eef_region_mask[:, :, 0, :10] = 1.0
    dino_eef_region_mask[:, :, 1, 120:140] = 1.0
    dino_pair_region_mask[:, :, 60:90] = 1.0

    data.create_dataset("dino_eef_uv", data=dino_eef_uv)
    data.create_dataset("dino_eef_valid", data=dino_eef_valid)
    data.create_dataset("dino_eef_region_mask", data=dino_eef_region_mask)
    data.create_dataset("dino_pair_region_mask", data=dino_pair_region_mask)
    data.create_dataset("dino_action_eef_uv", data=dino_action_eef_uv)
    data.create_dataset("dino_action_eef_valid", data=dino_action_eef_valid)
    meta.create_dataset("episode_ends", data=np.array([steps], dtype=np.int64))


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="gap_dino_interaction_check_", dir="/tmp")
    try:
        zarr_path = os.path.join(tmp, "synthetic.zarr")
        write_zarr(zarr_path)
        stats = check_zarr(zarr_path, token_grid=(15, 20), print_samples=2)
        assert stats["total_steps"] == 6.0
        assert stats["action_left_visible_fraction"] == 1.0
        print("DINO interaction zarr checker smoke test passed")
    finally:
        shutil.rmtree(tmp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
