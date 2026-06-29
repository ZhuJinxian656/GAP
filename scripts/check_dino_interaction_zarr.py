#!/usr/bin/env python
"""Sanity-check DINO EEF/action interaction arrays in a GAP zarr dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


REQUIRED_ARRAYS = (
    "dino_eef_uv",
    "dino_eef_valid",
    "dino_eef_region_mask",
    "dino_pair_region_mask",
    "dino_action_eef_uv",
    "dino_action_eef_valid",
)


def parse_token_grid(value: str) -> tuple[int, int]:
    pieces = tuple(int(piece) for piece in value.replace("x", ",").split(","))
    if len(pieces) != 2:
        raise ValueError("--token-grid must contain exactly two integers, e.g. 15,20")
    return pieces


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _fraction(numerator: np.ndarray, denominator: np.ndarray | float) -> float:
    denom = float(np.asarray(denominator).sum())
    if denom <= 0:
        return 0.0
    return float(np.asarray(numerator).sum() / denom)


def check_zarr(
    zarr_path: str | Path,
    token_grid: tuple[int, int] = (15, 20),
    print_samples: int = 5,
) -> dict[str, float]:
    zarr_path = Path(zarr_path).expanduser().resolve()
    root = zarr.open(str(zarr_path), mode="r")
    _require("data" in root, f"{zarr_path} is missing a 'data' group.")
    data = root["data"]
    available = set(data.array_keys())
    missing = [key for key in REQUIRED_ARRAYS if key not in available]
    _require(not missing, f"Missing required interaction arrays: {', '.join(missing)}")

    grid_h, grid_w = token_grid
    num_tokens = grid_h * grid_w
    arrays = {key: data[key] for key in REQUIRED_ARRAYS}
    total_steps = int(arrays["dino_eef_uv"].shape[0])

    expected_shapes = {
        "dino_eef_uv": (total_steps, None, 2, 2),
        "dino_eef_valid": (total_steps, None, 2),
        "dino_eef_region_mask": (total_steps, None, 2, num_tokens),
        "dino_pair_region_mask": (total_steps, None, num_tokens),
        "dino_action_eef_uv": (total_steps, None, 2, 2),
        "dino_action_eef_valid": (total_steps, None, 2),
    }
    num_views = int(arrays["dino_eef_uv"].shape[1])
    _require(num_views > 0, "dino_eef_uv must contain at least one view.")
    for key, expected in expected_shapes.items():
        arr = arrays[key]
        _require(
            arr.dtype == np.float32,
            f"{key} must have dtype float32, got {arr.dtype}.",
        )
        _require(
            len(arr.shape) == len(expected),
            f"{key} rank mismatch: expected {len(expected)} dims, got shape {arr.shape}.",
        )
        for dim_idx, dim in enumerate(expected):
            if dim is None:
                _require(
                    arr.shape[dim_idx] == num_views,
                    f"{key} view dim must match dino_eef_uv ({num_views}), got {arr.shape[dim_idx]}.",
                )
            else:
                _require(
                    arr.shape[dim_idx] == dim,
                    f"{key} shape mismatch at dim {dim_idx}: expected {dim}, got {arr.shape[dim_idx]} "
                    f"(full shape {arr.shape}).",
                )

    eef_uv = arrays["dino_eef_uv"][:]
    eef_valid = arrays["dino_eef_valid"][:].clip(0.0, 1.0)
    eef_mask = arrays["dino_eef_region_mask"][:].clip(0.0, 1.0)
    pair_mask = arrays["dino_pair_region_mask"][:].clip(0.0, 1.0)
    action_uv = arrays["dino_action_eef_uv"][:]
    action_valid = arrays["dino_action_eef_valid"][:].clip(0.0, 1.0)

    action_x = action_uv[..., 0]
    action_y = action_uv[..., 1]
    action_oob = (
        (action_x < 0.0)
        | (action_x > float(grid_w - 1))
        | (action_y < 0.0)
        | (action_y > float(grid_h - 1))
    ).astype(np.float32)
    current_action_valid = eef_valid * action_valid
    displacement = np.linalg.norm(action_uv - eef_uv, axis=-1)

    stats = {
        "total_steps": float(total_steps),
        "left_visible_fraction": float(eef_valid[..., 0].mean()),
        "right_visible_fraction": float(eef_valid[..., 1].mean()),
        "both_visible_fraction": float((eef_valid[..., 0] * eef_valid[..., 1]).mean()),
        "action_left_visible_fraction": float(action_valid[..., 0].mean()),
        "action_right_visible_fraction": float(action_valid[..., 1].mean()),
        "action_both_visible_fraction": float((action_valid[..., 0] * action_valid[..., 1]).mean()),
        "left_mask_token_fraction": float(eef_mask[:, :, 0, :].mean()),
        "right_mask_token_fraction": float(eef_mask[:, :, 1, :].mean()),
        "pair_mask_token_fraction": float(pair_mask.mean()),
        "target_uv_oob_fraction": _fraction(action_oob * action_valid, action_valid),
        "mean_current_to_action_uv_displacement": _fraction(displacement * current_action_valid, current_action_valid),
    }

    print(f"DINO interaction zarr: {zarr_path}")
    print(f"token_grid: {grid_h}x{grid_w} ({num_tokens} tokens)")
    print(f"total steps: {total_steps}")
    print(f"views: {num_views}")
    for key in REQUIRED_ARRAYS:
        print(f"{key}: shape={arrays[key].shape}, dtype={arrays[key].dtype}")
    print("stats:")
    for key, value in stats.items():
        if key == "total_steps":
            continue
        print(f"  {key}: {value:.6f}")

    sample_count = max(0, min(int(print_samples), total_steps))
    if sample_count:
        print("samples:")
        sample_indices = np.linspace(0, total_steps - 1, sample_count, dtype=np.int64)
        for idx in sample_indices:
            print(
                f"  step {int(idx):06d}: "
                f"current_uv={np.round(eef_uv[idx, 0], 3).tolist()} "
                f"current_valid={np.round(eef_valid[idx, 0], 3).tolist()} "
                f"action_uv={np.round(action_uv[idx, 0], 3).tolist()} "
                f"action_valid={np.round(action_valid[idx, 0], 3).tolist()}"
            )

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="GAP zarr dataset to inspect")
    parser.add_argument("--token-grid", default="15,20", help="DINO token grid as H,W")
    parser.add_argument("--print-samples", type=int, default=5)
    args = parser.parse_args()

    check_zarr(
        args.zarr,
        token_grid=parse_token_grid(args.token_grid),
        print_samples=args.print_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
