#!/usr/bin/env python3
"""Generate EEF-centered Pi3 token masks from RoboTwin HDF5 camera geometry.

This is a proxy for hand-near interaction regions, not a true object-hand mask:
the public HDF5 data exposes EEF poses and camera calibration, but not object
pose, object masks, robot masks, depth, or contact labels.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import numpy as np
import zarr


EPISODE_RE = re.compile(r"episode(\d+)\.hdf5$")


def episode_index(path: Path) -> int:
    match = EPISODE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Expected episode file named episode<N>.hdf5, got {path.name!r}.")
    return int(match.group(1))


def iter_episode_files(hdf5_root: Path, expert_data_num: int) -> Iterable[Path]:
    data_dir = hdf5_root / "data"
    for idx in range(expert_data_num):
        path = data_dir / f"episode{idx}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected HDF5 episode: {path}")
        yield path


def decode_image_size(encoded: np.bytes_) -> tuple[int, int]:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode HDF5 RGB frame.")
    height, width = image.shape[:2]
    return height, width


def project_points(
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = np.einsum("tij,tj->ti", extrinsics[:, :, :3], points_world) + extrinsics[:, :, 3]
    z = points_cam[:, 2]
    projected = np.einsum("tij,tj->ti", intrinsics, points_cam)
    uv = projected[:, :2] / projected[:, 2:3]
    valid = np.isfinite(uv).all(axis=1) & (z > 1e-6)
    return uv, valid


def disk_token_mask(
    uv: np.ndarray,
    valid: np.ndarray,
    image_size: tuple[int, int],
    token_grid: tuple[int, int],
    radius_tokens: float,
) -> np.ndarray:
    image_h, image_w = image_size
    grid_h, grid_w = token_grid
    yy, xx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
    token_x = (uv[:, 0] / image_w) * grid_w - 0.5
    token_y = (uv[:, 1] / image_h) * grid_h - 0.5

    masks = np.zeros((uv.shape[0], grid_h, grid_w), dtype=bool)
    for idx in range(uv.shape[0]):
        if not valid[idx]:
            continue
        if uv[idx, 0] < 0 or uv[idx, 0] >= image_w or uv[idx, 1] < 0 or uv[idx, 1] >= image_h:
            continue
        dist2 = (xx - token_x[idx]) ** 2 + (yy - token_y[idx]) ** 2
        masks[idx] = dist2 <= radius_tokens ** 2
    return masks.reshape(uv.shape[0], grid_h * grid_w)


def masks_for_episode(
    hdf5_path: Path,
    camera: str,
    token_grid: tuple[int, int],
    radius_tokens: float,
) -> tuple[np.ndarray, dict[str, float]]:
    with h5py.File(hdf5_path, "r") as f:
        cam_group = f["observation"][camera]
        intrinsics = cam_group["intrinsic_cv"][:]
        extrinsics = cam_group["extrinsic_cv"][:]
        image_size = decode_image_size(cam_group["rgb"][0])

        num_zarr_steps = f["joint_action"]["vector"].shape[0] - 1
        if num_zarr_steps <= 0:
            raise ValueError(f"{hdf5_path} has no usable zarr-aligned frames.")

        hand_masks = []
        visibility = []
        for side in ("left", "right"):
            points = f["endpose"][f"{side}_endpose"][:num_zarr_steps, :3]
            uv, valid = project_points(points, intrinsics[:num_zarr_steps], extrinsics[:num_zarr_steps])
            in_image = (
                valid
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < image_size[1])
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < image_size[0])
            )
            hand_masks.append(disk_token_mask(uv, valid, image_size, token_grid, radius_tokens))
            visibility.append(in_image)

    eef_mask = np.logical_or(hand_masks[0], hand_masks[1])
    visible_either = np.logical_or(visibility[0], visibility[1])

    # If both EEF projections are outside the camera, keep an empty EEF mask.
    # This preserves the meaning of "visible EEF region" instead of fabricating
    # a default center crop.
    stats = {
        "left_visible_fraction": float(visibility[0].mean()),
        "right_visible_fraction": float(visibility[1].mean()),
        "either_visible_fraction": float(visible_either.mean()),
        "eef_token_fraction": float(eef_mask.mean()),
    }
    return eef_mask.astype(np.float32), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hdf5-root",
        required=True,
        help="RoboTwin task/config root containing data/episode<N>.hdf5",
    )
    parser.add_argument("--zarr", required=True, help="GAP zarr dataset to update in-place")
    parser.add_argument("--expert-data-num", type=int, default=50)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--token-grid", default="17,23", help="Pi3 token grid as H,W")
    parser.add_argument("--radius-tokens", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    token_grid = tuple(int(piece) for piece in args.token_grid.replace("x", ",").split(","))
    if len(token_grid) != 2:
        raise ValueError("--token-grid must contain exactly two integers")

    hdf5_root = Path(args.hdf5_root).resolve()
    zarr_path = Path(args.zarr).resolve()
    zarr_root = zarr.open(str(zarr_path), mode="a")
    data = zarr_root["data"]
    episode_ends = np.asarray(zarr_root["meta"]["episode_ends"][:], dtype=np.int64)
    if episode_ends.shape[0] < args.expert_data_num:
        raise ValueError(
            f"Zarr has {episode_ends.shape[0]} episode ends, fewer than expert-data-num={args.expert_data_num}."
        )
    expected_steps = int(episode_ends[args.expert_data_num - 1])

    eef_masks = []
    per_episode_stats = []
    prev_end = 0
    for path in iter_episode_files(hdf5_root, args.expert_data_num):
        idx = episode_index(path)
        expected_len = int(episode_ends[idx] - prev_end)
        prev_end = int(episode_ends[idx])
        mask, stats = masks_for_episode(path, args.camera, token_grid, args.radius_tokens)
        if mask.shape[0] != expected_len:
            raise ValueError(
                f"{path.name} mask length {mask.shape[0]} does not match zarr episode length {expected_len}."
            )
        eef_masks.append(mask)
        per_episode_stats.append(stats)

    eef_mask = np.concatenate(eef_masks, axis=0).reshape(expected_steps, 1, token_grid[0] * token_grid[1])
    non_eef_mask = 1.0 - eef_mask

    for key, value in (
        ("pi3_eef_region_mask", eef_mask),
        ("pi3_non_eef_region_mask", non_eef_mask),
    ):
        if key in data:
            if not args.overwrite:
                raise FileExistsError(f"{key} already exists in {zarr_path}; pass --overwrite to replace it.")
            del data[key]
        data.create_dataset(key, data=value.astype(np.float32), chunks=(100, 1, token_grid[0] * token_grid[1]))

    mean_stats = {
        key: float(np.mean([stats[key] for stats in per_episode_stats]))
        for key in per_episode_stats[0]
    }
    print(f"Wrote pi3_eef_region_mask and pi3_non_eef_region_mask to {zarr_path}")
    print(f"shape: {eef_mask.shape}")
    print("mean stats:")
    for key, value in mean_stats.items():
        print(f"  {key}: {value:.4f}")
    print(f"  non_eef_token_fraction: {float(non_eef_mask.mean()):.4f}")


if __name__ == "__main__":
    main()
