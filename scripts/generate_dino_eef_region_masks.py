#!/usr/bin/env python3
"""Generate DINO-grid EEF and pair interaction masks from RoboTwin HDF5 geometry."""
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


def decode_rgb_frame(encoded: np.bytes_) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode HDF5 RGB frame.")
    return image


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


def pixel_uv_to_token_uv(
    uv: np.ndarray,
    image_size: tuple[int, int],
    token_grid: tuple[int, int],
) -> np.ndarray:
    image_h, image_w = image_size
    grid_h, grid_w = token_grid
    token_uv = np.zeros_like(uv, dtype=np.float32)
    token_uv[:, 0] = (uv[:, 0] / image_w) * grid_w - 0.5
    token_uv[:, 1] = (uv[:, 1] / image_h) * grid_h - 0.5
    return token_uv


def in_image_mask(uv: np.ndarray, valid: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_size
    return (
        valid
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < image_w)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < image_h)
    )


def token_mesh(token_grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w = token_grid
    return np.meshgrid(
        np.arange(grid_h, dtype=np.float32),
        np.arange(grid_w, dtype=np.float32),
        indexing="ij",
    )


def disk_token_mask(
    token_uv: np.ndarray,
    valid: np.ndarray,
    token_grid: tuple[int, int],
    radius_tokens: float,
) -> np.ndarray:
    grid_h, grid_w = token_grid
    yy, xx = token_mesh(token_grid)

    masks = np.zeros((token_uv.shape[0], grid_h, grid_w), dtype=bool)
    for idx in range(token_uv.shape[0]):
        if not valid[idx]:
            continue
        dist2 = (xx - token_uv[idx, 0]) ** 2 + (yy - token_uv[idx, 1]) ** 2
        masks[idx] = dist2 <= radius_tokens ** 2
    return masks.reshape(token_uv.shape[0], grid_h * grid_w)


def segment_token_mask(
    left_token_uv: np.ndarray,
    right_token_uv: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    token_grid: tuple[int, int],
    radius_tokens: float,
) -> np.ndarray:
    grid_h, grid_w = token_grid
    yy, xx = token_mesh(token_grid)
    points = np.stack([xx, yy], axis=-1)

    masks = np.zeros((left_token_uv.shape[0], grid_h, grid_w), dtype=bool)
    both_valid = left_valid & right_valid
    for idx in range(left_token_uv.shape[0]):
        if not both_valid[idx]:
            continue
        start = left_token_uv[idx]
        end = right_token_uv[idx]
        segment = end - start
        length2 = float(np.dot(segment, segment))
        if length2 <= 1e-12:
            closest = start
        else:
            t = np.clip(np.sum((points - start) * segment, axis=-1) / length2, 0.0, 1.0)
            closest = start + t[..., None] * segment
        dist2 = np.sum((points - closest) ** 2, axis=-1)
        masks[idx] = dist2 <= radius_tokens ** 2
    return masks.reshape(left_token_uv.shape[0], grid_h * grid_w)


def masks_for_episode(
    hdf5_path: Path,
    camera: str,
    token_grid: tuple[int, int],
    radius_tokens: float,
    pair_radius_tokens: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    with h5py.File(hdf5_path, "r") as f:
        cam_group = f["observation"][camera]
        intrinsics = cam_group["intrinsic_cv"][:]
        extrinsics = cam_group["extrinsic_cv"][:]
        image_size = decode_image_size(cam_group["rgb"][0])

        num_zarr_steps = f["joint_action"]["vector"].shape[0] - 1
        if num_zarr_steps <= 0:
            raise ValueError(f"{hdf5_path} has no usable zarr-aligned frames.")

        token_uvs = []
        visibilities = []
        action_token_uvs = []
        action_visibilities = []
        arm_masks = []
        for side in ("left", "right"):
            endpose = f["endpose"][f"{side}_endpose"][:, :3]

            points = endpose[:num_zarr_steps]
            uv, valid = project_points(points, intrinsics[:num_zarr_steps], extrinsics[:num_zarr_steps])
            visible = in_image_mask(uv, valid, image_size)
            token_uv = pixel_uv_to_token_uv(uv, image_size, token_grid)
            token_uv[~visible] = 0.0
            mask = disk_token_mask(token_uv, visible, token_grid, radius_tokens)

            action_points = endpose[1:num_zarr_steps + 1]
            action_uv, action_valid = project_points(
                action_points,
                intrinsics[1:num_zarr_steps + 1],
                extrinsics[1:num_zarr_steps + 1],
            )
            action_visible = in_image_mask(action_uv, action_valid, image_size)
            action_token_uv = pixel_uv_to_token_uv(action_uv, image_size, token_grid)
            action_token_uv[~action_visible] = 0.0

            token_uvs.append(token_uv)
            visibilities.append(visible)
            action_token_uvs.append(action_token_uv)
            action_visibilities.append(action_visible)
            arm_masks.append(mask)

    dino_eef_uv = np.stack(token_uvs, axis=1).astype(np.float32)
    dino_eef_valid = np.stack(visibilities, axis=1).astype(np.float32)
    dino_action_eef_uv = np.stack(action_token_uvs, axis=1).astype(np.float32)
    dino_action_eef_valid = np.stack(action_visibilities, axis=1).astype(np.float32)
    dino_eef_region_mask = np.stack(arm_masks, axis=1).astype(np.float32)
    dino_pair_region_mask = segment_token_mask(
        token_uvs[0],
        token_uvs[1],
        visibilities[0],
        visibilities[1],
        token_grid,
        pair_radius_tokens,
    ).astype(np.float32)

    stats = {
        "left_visible_fraction": float(visibilities[0].mean()),
        "right_visible_fraction": float(visibilities[1].mean()),
        "either_visible_fraction": float(np.logical_or(visibilities[0], visibilities[1]).mean()),
        "both_visible_fraction": float(np.logical_and(visibilities[0], visibilities[1]).mean()),
        "action_left_visible_fraction": float(action_visibilities[0].mean()),
        "action_right_visible_fraction": float(action_visibilities[1].mean()),
        "action_both_visible_fraction": float(np.logical_and(action_visibilities[0], action_visibilities[1]).mean()),
        "left_token_fraction": float(dino_eef_region_mask[:, 0].mean()),
        "right_token_fraction": float(dino_eef_region_mask[:, 1].mean()),
        "pair_token_fraction": float(dino_pair_region_mask.mean()),
    }
    return (
        dino_eef_uv,
        dino_eef_valid,
        dino_eef_region_mask,
        dino_pair_region_mask,
        dino_action_eef_uv,
        dino_action_eef_valid,
        stats,
    )


def parse_token_grid(value: str) -> tuple[int, int]:
    token_grid = tuple(int(piece) for piece in value.replace("x", ",").split(","))
    if len(token_grid) != 2:
        raise ValueError("--token-grid must contain exactly two integers")
    return token_grid


def write_dataset(data: zarr.Group, key: str, value: np.ndarray, chunks: tuple[int, ...], overwrite: bool) -> None:
    if key in data:
        if not overwrite:
            raise FileExistsError(f"{key} already exists; pass --overwrite to replace it.")
        del data[key]
    data.create_dataset(key, data=value.astype(np.float32), chunks=chunks)


def mask_to_image(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_size
    return cv2.resize(mask.astype(np.float32), (image_w, image_h), interpolation=cv2.INTER_NEAREST)


def write_episode_previews(
    hdf5_path: Path,
    camera: str,
    token_grid: tuple[int, int],
    dino_eef_uv: np.ndarray,
    dino_eef_valid: np.ndarray,
    dino_eef_region_mask: np.ndarray,
    dino_pair_region_mask: np.ndarray,
    vis_dir: Path,
    max_count: int,
) -> int:
    if max_count <= 0:
        return 0
    vis_dir.mkdir(parents=True, exist_ok=True)
    grid_h, grid_w = token_grid
    visible = dino_eef_valid.any(axis=1)
    frame_indices = np.flatnonzero(visible)
    if frame_indices.size == 0:
        frame_indices = np.arange(min(max_count, dino_eef_uv.shape[0]))
    else:
        frame_indices = frame_indices[:max_count]

    written = 0
    episode_id = episode_index(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        rgb_array = f["observation"][camera]["rgb"]
        for frame_idx in frame_indices:
            image = decode_rgb_frame(rgb_array[int(frame_idx)]).astype(np.float32)
            image_h, image_w = image.shape[:2]
            overlay = np.zeros_like(image)

            left = mask_to_image(dino_eef_region_mask[frame_idx, 0].reshape(grid_h, grid_w), (image_h, image_w))
            right = mask_to_image(dino_eef_region_mask[frame_idx, 1].reshape(grid_h, grid_w), (image_h, image_w))
            pair = mask_to_image(dino_pair_region_mask[frame_idx].reshape(grid_h, grid_w), (image_h, image_w))
            overlay[..., 2] = np.maximum(overlay[..., 2], left * 255.0)
            overlay[..., 0] = np.maximum(overlay[..., 0], right * 255.0)
            overlay[..., 1] = np.maximum(overlay[..., 1], pair * 200.0)

            blended = cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)
            for arm_idx, color in enumerate(((0, 0, 255), (255, 0, 0))):
                if not dino_eef_valid[frame_idx, arm_idx]:
                    continue
                token_x, token_y = dino_eef_uv[frame_idx, arm_idx]
                px = int(round((token_x + 0.5) / grid_w * image_w))
                py = int(round((token_y + 0.5) / grid_h * image_h))
                cv2.circle(blended, (px, py), 5, color, thickness=2)

            out_path = vis_dir / f"episode{episode_id:04d}_frame{int(frame_idx):05d}.png"
            cv2.imwrite(str(out_path), blended.astype(np.uint8))
            written += 1
            if written >= max_count:
                break
    return written


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
    parser.add_argument("--token-grid", default="15,20", help="DINO token grid as H,W")
    parser.add_argument("--radius-tokens", type=float, default=2.5)
    parser.add_argument("--pair-radius-tokens", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vis-dir", default=None, help="Optional directory for overlay PNG previews")
    parser.add_argument("--vis-count", type=int, default=0, help="Maximum number of overlay previews to write")
    args = parser.parse_args()

    token_grid = parse_token_grid(args.token_grid)
    num_tokens = token_grid[0] * token_grid[1]

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

    all_uv = []
    all_valid = []
    all_arm_masks = []
    all_pair_masks = []
    all_action_uv = []
    all_action_valid = []
    per_episode_stats = []
    prev_end = 0
    vis_written = 0
    vis_dir = Path(args.vis_dir).resolve() if args.vis_dir else None
    for path in iter_episode_files(hdf5_root, args.expert_data_num):
        idx = episode_index(path)
        expected_len = int(episode_ends[idx] - prev_end)
        prev_end = int(episode_ends[idx])
        uv, valid, arm_mask, pair_mask, action_uv, action_valid, stats = masks_for_episode(
            path,
            args.camera,
            token_grid,
            args.radius_tokens,
            args.pair_radius_tokens,
        )
        if uv.shape[0] != expected_len:
            raise ValueError(
                f"{path.name} mask length {uv.shape[0]} does not match zarr episode length {expected_len}."
            )
        all_uv.append(uv)
        all_valid.append(valid)
        all_arm_masks.append(arm_mask)
        all_pair_masks.append(pair_mask)
        all_action_uv.append(action_uv)
        all_action_valid.append(action_valid)
        per_episode_stats.append(stats)
        if vis_dir is not None and vis_written < args.vis_count:
            vis_written += write_episode_previews(
                path,
                args.camera,
                token_grid,
                uv,
                valid,
                arm_mask,
                pair_mask,
                vis_dir,
                args.vis_count - vis_written,
            )

    dino_eef_uv = np.concatenate(all_uv, axis=0).reshape(expected_steps, 1, 2, 2)
    dino_eef_valid = np.concatenate(all_valid, axis=0).reshape(expected_steps, 1, 2)
    dino_eef_region_mask = np.concatenate(all_arm_masks, axis=0).reshape(expected_steps, 1, 2, num_tokens)
    dino_pair_region_mask = np.concatenate(all_pair_masks, axis=0).reshape(expected_steps, 1, num_tokens)
    dino_action_eef_uv = np.concatenate(all_action_uv, axis=0).reshape(expected_steps, 1, 2, 2)
    dino_action_eef_valid = np.concatenate(all_action_valid, axis=0).reshape(expected_steps, 1, 2)

    write_dataset(data, "dino_eef_uv", dino_eef_uv, (100, 1, 2, 2), args.overwrite)
    write_dataset(data, "dino_eef_valid", dino_eef_valid, (100, 1, 2), args.overwrite)
    write_dataset(data, "dino_eef_region_mask", dino_eef_region_mask, (100, 1, 2, num_tokens), args.overwrite)
    write_dataset(data, "dino_pair_region_mask", dino_pair_region_mask, (100, 1, num_tokens), args.overwrite)
    write_dataset(data, "dino_action_eef_uv", dino_action_eef_uv, (100, 1, 2, 2), args.overwrite)
    write_dataset(data, "dino_action_eef_valid", dino_action_eef_valid, (100, 1, 2), args.overwrite)

    mean_stats = {
        key: float(np.mean([stats[key] for stats in per_episode_stats]))
        for key in per_episode_stats[0]
    }
    print(f"Wrote DINO EEF interaction masks to {zarr_path}")
    print(f"token_grid: {token_grid} ({num_tokens} tokens)")
    print(f"dino_eef_uv shape: {dino_eef_uv.shape}")
    print(f"dino_eef_valid shape: {dino_eef_valid.shape}")
    print(f"dino_eef_region_mask shape: {dino_eef_region_mask.shape}")
    print(f"dino_pair_region_mask shape: {dino_pair_region_mask.shape}")
    print(f"dino_action_eef_uv shape: {dino_action_eef_uv.shape}")
    print(f"dino_action_eef_valid shape: {dino_action_eef_valid.shape}")
    print("mean stats:")
    for key, value in mean_stats.items():
        print(f"  {key}: {value:.4f}")
    if vis_dir is not None:
        print(f"wrote {vis_written} preview image(s) to {vis_dir}")


if __name__ == "__main__":
    main()
