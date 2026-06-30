#!/usr/bin/env python
"""Visualize DINO features and DINO EEF/action interaction fields for GAP zarr data."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import zarr

GAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAP_ROOT))


def parse_token_grid(value: str) -> tuple[int, int]:
    pieces = tuple(int(piece) for piece in value.replace("x", ",").split(","))
    if len(pieces) != 2:
        raise ValueError("--token-grid must contain exactly two integers, e.g. 15,20")
    return pieces


def parse_timesteps(value: str) -> list[int]:
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(piece.strip()) for piece in value.split(",") if piece.strip()]


def decode_rgb_frame(encoded: Any) -> np.ndarray:
    if isinstance(encoded, np.ndarray) and encoded.ndim == 3:
        return encoded.astype(np.uint8)
    buffer = np.frombuffer(encoded.tobytes() if hasattr(encoded, "tobytes") else encoded, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode HDF5 RGB frame.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def episode_and_local_index(episode_ends: np.ndarray, global_idx: int) -> tuple[int, int]:
    episode_idx = int(np.searchsorted(episode_ends, global_idx, side="right"))
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    return episode_idx, int(global_idx - start)


def load_rgb(hdf5_root: Path, camera: str, episode_ends: np.ndarray, global_idx: int) -> np.ndarray:
    episode_idx, local_idx = episode_and_local_index(episode_ends, global_idx)
    hdf5_path = hdf5_root / "data" / f"episode{episode_idx}.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Missing HDF5 episode for zarr step {global_idx}: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        return decode_rgb_frame(f["observation"][camera]["rgb"][local_idx])


def normalize_image(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    low = values.reshape(-1, values.shape[-1]).min(axis=0)
    high = values.reshape(-1, values.shape[-1]).max(axis=0)
    denom = np.maximum(high - low, 1.0e-6)
    return np.clip((values - low) / denom, 0.0, 1.0)


def dino_pca_rgb(features: np.ndarray, token_grid: tuple[int, int]) -> np.ndarray:
    grid_h, grid_w = token_grid
    flat = features.astype(np.float32)
    centered = flat - flat.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    comps = centered @ vh[:3].T
    rgb = normalize_image(comps).reshape(grid_h, grid_w, 3)
    return (rgb * 255.0).astype(np.uint8)


def kmeans_labels(
    features: np.ndarray,
    k: int,
    max_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat = features.astype(np.float32)
    k = max(1, min(int(k), flat.shape[0]))
    init = rng.choice(flat.shape[0], size=k, replace=False)
    centers = flat[init].copy()
    labels = np.zeros(flat.shape[0], dtype=np.int64)
    for _ in range(max(1, int(max_iter))):
        dist2 = ((flat[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = dist2.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_idx in range(k):
            selected = flat[labels == cluster_idx]
            if selected.size:
                centers[cluster_idx] = selected.mean(axis=0)
    return labels


def kmeans_rgb(features: np.ndarray, token_grid: tuple[int, int], k: int, max_iter: int, rng: np.random.Generator) -> np.ndarray:
    palette = np.array(
        [
            [230, 57, 70],
            [29, 53, 87],
            [69, 123, 157],
            [42, 157, 143],
            [233, 196, 106],
            [244, 162, 97],
            [131, 56, 236],
            [255, 0, 110],
        ],
        dtype=np.uint8,
    )
    labels = kmeans_labels(features, k, max_iter, rng)
    grid_h, grid_w = token_grid
    return palette[labels % len(palette)].reshape(grid_h, grid_w, 3)


def resize_grid(grid_image: np.ndarray, image_size: tuple[int, int], interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    image_h, image_w = image_size
    return cv2.resize(grid_image, (image_w, image_h), interpolation=interpolation)


def token_to_pixel(token_uv: np.ndarray, image_size: tuple[int, int], token_grid: tuple[int, int]) -> tuple[int, int]:
    image_h, image_w = image_size
    grid_h, grid_w = token_grid
    px = int(round((float(token_uv[0]) + 0.5) / grid_w * image_w))
    py = int(round((float(token_uv[1]) + 0.5) / grid_h * image_h))
    return px, py


def overlay_mask(
    rgb: np.ndarray,
    mask: np.ndarray,
    token_grid: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    grid_h, grid_w = token_grid
    image = rgb.astype(np.float32).copy()
    mask_image = resize_grid(mask.reshape(grid_h, grid_w).astype(np.float32), rgb.shape[:2], cv2.INTER_LINEAR)
    overlay = np.zeros_like(image)
    overlay[..., 0] = color[0] * mask_image
    overlay[..., 1] = color[1] * mask_image
    overlay[..., 2] = color[2] * mask_image
    return np.clip(image * (1.0 - alpha * mask_image[..., None]) + overlay * alpha, 0, 255).astype(np.uint8)


def overlay_current_masks(rgb: np.ndarray, eef_mask: np.ndarray, pair_mask: np.ndarray, token_grid: tuple[int, int]) -> np.ndarray:
    image = overlay_mask(rgb, eef_mask[0], token_grid, (255, 60, 60), alpha=0.45)
    image = overlay_mask(image, eef_mask[1], token_grid, (60, 120, 255), alpha=0.45)
    image = overlay_mask(image, pair_mask, token_grid, (60, 220, 120), alpha=0.35)
    return image


def draw_uv_overlay(
    rgb: np.ndarray,
    uv: np.ndarray,
    valid: np.ndarray,
    token_grid: tuple[int, int],
    draw_pair: bool = True,
) -> np.ndarray:
    image = rgb.copy()
    colors = [(255, 50, 50), (50, 120, 255)]
    points = []
    for arm_idx, color in enumerate(colors):
        if valid[arm_idx] <= 0:
            points.append(None)
            continue
        px, py = token_to_pixel(uv[arm_idx], image.shape[:2], token_grid)
        points.append((px, py))
        cv2.circle(image, (px, py), 7, color, thickness=2)
        cv2.circle(image, (px, py), 2, color, thickness=-1)
    if draw_pair and points[0] is not None and points[1] is not None:
        cv2.line(image, points[0], points[1], (60, 220, 120), thickness=2)
    return image


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    panel = image.astype(np.uint8)
    banner = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([banner, panel], axis=0)


def compose_panels(images: list[np.ndarray], labels: list[str], cols: int = 3) -> np.ndarray:
    panels = [labeled_panel(image, label) for image, label in zip(images, labels)]
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: panel.shape[0], : panel.shape[1]] = panel
        padded.append(canvas)
    rows = []
    for start in range(0, len(padded), cols):
        row = padded[start:start + cols]
        while len(row) < cols:
            row.append(np.zeros((height, width, 3), dtype=np.uint8))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def sample_indices(total_steps: int, num_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(max(int(num_samples), 1), total_steps)
    return np.sort(rng.choice(total_steps, size=count, replace=False))


def no_checkpoint_visuals(args, root: zarr.Group, episode_ends: np.ndarray, token_grid: tuple[int, int]) -> list[dict[str, Any]]:
    data = root["data"]
    hdf5_root = Path(args.hdf5_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() / "no_checkpoint"
    rng = np.random.default_rng(args.seed)
    indices = sample_indices(data["dinov3_features"].shape[0], args.num_samples, args.seed)
    summary = []

    for sample_id, idx in enumerate(indices):
        idx = int(idx)
        rgb = load_rgb(hdf5_root, args.camera, episode_ends, idx)
        image_size = rgb.shape[:2]
        dino = data["dinov3_features"][idx, 0]
        pca = resize_grid(dino_pca_rgb(dino, token_grid), image_size, cv2.INTER_LINEAR)
        clusters = resize_grid(
            kmeans_rgb(dino, token_grid, args.kmeans_k, args.max_kmeans_iter, rng),
            image_size,
            cv2.INTER_NEAREST,
        )
        current_overlay = overlay_current_masks(
            rgb,
            data["dino_eef_region_mask"][idx, 0],
            data["dino_pair_region_mask"][idx, 0],
            token_grid,
        )
        current_pair = overlay_mask(rgb, data["dino_pair_region_mask"][idx, 0], token_grid, (60, 220, 120), alpha=0.45)
        target_uv = data["dino_action_eef_uv"][idx, 0]
        target_valid = data["dino_action_eef_valid"][idx, 0]
        target_overlay = draw_uv_overlay(rgb, target_uv, target_valid, token_grid, draw_pair=True)
        current_uv = data["dino_eef_uv"][idx, 0]
        current_valid = data["dino_eef_valid"][idx, 0]
        current_uv_overlay = draw_uv_overlay(rgb, current_uv, current_valid, token_grid)
        interp_overlays = []
        interp_summary = []
        for flow_lambda in parse_float_list(args.flow_lambdas):
            interp_uv = (1.0 - flow_lambda) * current_uv + flow_lambda * target_uv
            interp_valid = target_valid * current_valid
            interp_overlays.append(draw_uv_overlay(rgb, interp_uv, interp_valid, token_grid, draw_pair=True))
            interp_summary.append(
                {
                    "lambda": float(flow_lambda),
                    "uv": interp_uv.tolist(),
                    "valid": interp_valid.tolist(),
                }
            )

        panel = compose_panels(
            [
                rgb,
                pca,
                clusters,
                current_overlay,
                current_pair,
                current_uv_overlay,
                target_overlay,
                *interp_overlays,
            ],
            [
                "raw RGB",
                "DINO PCA",
                "DINO k-means",
                "current EEF masks",
                "current pair mask",
                "current EEF UV",
                "action target UV",
                *[f"interp UV l={flow_lambda:g}" for flow_lambda in parse_float_list(args.flow_lambdas)],
            ],
            cols=3,
        )
        out_path = out_dir / f"sample_{sample_id:03d}_step_{idx:06d}.png"
        save_rgb(out_path, panel)
        summary.append(
            {
                "step": idx,
                "output": str(out_path),
                "target_valid": target_valid.tolist(),
                "target_uv": target_uv.tolist(),
                "current_valid": current_valid.tolist(),
                "current_uv": current_uv.tolist(),
                "interp_targets": interp_summary,
            }
        )
    return summary


def _namespace_from_dict(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _namespace_from_dict(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_namespace_from_dict(value) for value in data]
    return data


def load_policy_from_checkpoint(checkpoint: str, device: str):
    import torch
    from gap_policy.policy.gap import GAPPolicy

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        raise ValueError("Checkpoint does not contain cfg.")
    if hasattr(cfg, "policy"):
        policy_cfg = cfg.policy
    else:
        policy_cfg = _namespace_from_dict(cfg)["policy"]
    if hasattr(policy_cfg, "items"):
        policy_cfg = dict(policy_cfg.items())
    else:
        policy_cfg = dict(policy_cfg)

    scheduler_cfg = policy_cfg["noise_scheduler"]
    if hasattr(scheduler_cfg, "items"):
        scheduler_cfg = dict(scheduler_cfg.items())
    target = scheduler_cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    scheduler_class = getattr(importlib.import_module(module_path), class_name)
    policy_cfg["noise_scheduler"] = scheduler_class(**scheduler_cfg)
    policy_cfg.pop("_target_", None)

    policy = GAPPolicy(**policy_cfg)
    state = ckpt["ema"] if ckpt.get("ema") is not None else ckpt.get("model")
    if state is None:
        raise ValueError("Checkpoint does not contain model or EMA weights.")
    policy.load_state_dict(state)
    if "normalizer" not in ckpt:
        raise ValueError("Checkpoint does not contain normalizer.")

    class FakedNormalizer:
        def __init__(self, state_dict):
            self._state_dict = state_dict

        def state_dict(self):
            return self._state_dict

    policy.set_normalizer(FakedNormalizer(ckpt["normalizer"]))
    policy.to(device).eval()
    return policy


def action_sequence(data: zarr.Group, idx: int, horizon: int, episode_ends: np.ndarray) -> np.ndarray:
    episode_idx, _ = episode_and_local_index(episode_ends, idx)
    episode_end = int(episode_ends[episode_idx])
    end = min(idx + horizon, episode_end)
    actions = data["action"][idx:end].astype(np.float32)
    if actions.shape[0] < horizon:
        pad = np.repeat(actions[-1:], horizon - actions.shape[0], axis=0)
        actions = np.concatenate([actions, pad], axis=0)
    return actions


def checkpoint_visuals(args, root: zarr.Group, episode_ends: np.ndarray, token_grid: tuple[int, int]) -> list[dict[str, Any]]:
    import torch

    data = root["data"]
    hdf5_root = Path(args.hdf5_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() / "checkpoint"
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = load_policy_from_checkpoint(args.checkpoint, device)
    if policy.interaction_mode != "action_uv":
        raise ValueError(f"Checkpoint interaction mode is {policy.interaction_mode!r}, expected 'action_uv'.")

    rng = np.random.default_rng(args.seed)
    indices = sample_indices(data["dinov3_features"].shape[0], args.num_samples, args.seed)
    timesteps = parse_timesteps(args.timesteps)
    flow_lambdas = parse_float_list(args.flow_lambdas)
    summary = []

    for sample_id, idx in enumerate(indices):
        idx = int(idx)
        rgb = load_rgb(hdf5_root, args.camera, episode_ends, idx)
        image_size = rgb.shape[:2]
        dino = data["dinov3_features"][idx, 0]
        pca = resize_grid(dino_pca_rgb(dino, token_grid), image_size, cv2.INTER_LINEAR)
        obs = {
            "dinov3_features": torch.from_numpy(data["dinov3_features"][idx:idx + 1].astype(np.float32)).to(device),
            "agent_pos": torch.from_numpy(data["state"][idx:idx + 1].astype(np.float32)).to(device),
            "dino_eef_uv": torch.from_numpy(data["dino_eef_uv"][idx:idx + 1].astype(np.float32)).to(device),
            "dino_eef_valid": torch.from_numpy(data["dino_eef_valid"][idx:idx + 1].astype(np.float32)).to(device),
            "dino_eef_region_mask": torch.from_numpy(data["dino_eef_region_mask"][idx:idx + 1].astype(np.float32)).to(device),
            "dino_pair_region_mask": torch.from_numpy(data["dino_pair_region_mask"][idx:idx + 1].astype(np.float32)).to(device),
        }
        if policy.use_pi3_features and "pi3_features" in data:
            obs["pi3_features"] = torch.from_numpy(data["pi3_features"][idx:idx + 1].astype(np.float32)).to(device)
        action_np = action_sequence(data, idx, policy.horizon, episode_ends)[None]
        action = torch.from_numpy(action_np.astype(np.float32)).to(device)
        nobs = policy.normalizer.normalize(obs)
        naction = policy.normalizer["action"].normalize(action)
        target_uv_seq = data["dino_action_eef_uv"][idx]
        target_valid_seq = data["dino_action_eef_valid"][idx]
        horizon_idx = min(max(int(args.horizon_index), 0), target_uv_seq.shape[0] - 1)
        target_uv = target_uv_seq[horizon_idx, 0]
        target_valid = target_valid_seq[horizon_idx, 0]
        current_uv = data["dino_eef_uv"][idx, 0]
        current_valid = data["dino_eef_valid"][idx, 0]
        current_overlay = draw_uv_overlay(rgb, current_uv, current_valid, token_grid, draw_pair=True)
        target_overlay = draw_uv_overlay(rgb, target_uv, target_valid, token_grid, draw_pair=True)

        if policy.generative_mode == "flow_matching":
            source_actions = policy._flow_source_actions(
                raw_obs_dict=obs,
                ref_actions=naction,
                batch_size=1,
                device=naction.device,
                dtype=naction.dtype,
            )
            iterator = [("lambda", float(flow_lambda)) for flow_lambda in flow_lambdas]
        else:
            source_actions = None
            iterator = [("timestep", int(timestep)) for timestep in timesteps]

        for sweep_kind, sweep_value in iterator:
            if sweep_kind == "lambda":
                lambda_tensor = torch.full((1,), float(sweep_value), device=device, dtype=naction.dtype)
                noised_actions = source_actions + lambda_tensor.view(1, 1, 1) * (naction - source_actions)
                timestep_tensor = policy._flow_lambda_to_timestep(lambda_tensor, 1, device, naction.dtype)
                interp_uv = (1.0 - float(sweep_value)) * current_uv + float(sweep_value) * target_uv
                interp_valid = target_valid * current_valid
                sweep_label = f"l={float(sweep_value):g}"
                out_suffix = f"lambda_{float(sweep_value):.2f}".replace(".", "p")
            else:
                timestep_tensor = torch.full((1,), int(sweep_value), device=device, dtype=torch.long)
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + idx + int(sweep_value))
                noise = torch.randn(naction.shape, device=device, dtype=naction.dtype, generator=generator)
                noised_actions = policy.noise_scheduler.add_noise(naction, noise, timestep_tensor)
                interp_uv = target_uv
                interp_valid = target_valid
                sweep_label = f"t={int(sweep_value)}"
                out_suffix = f"t_{int(sweep_value):03d}"

            with torch.no_grad():
                context = policy.encode_observations(nobs, return_context=True)
                _, _, aux = policy._compute_interaction_tokens(
                    context["dinov3_tokens"],
                    context["obs_dict"],
                    mid_actions=noised_actions,
                    timestep=timestep_tensor,
                    agent_pos=context["agent_pos"],
                )
            pred_uv = aux["pred_uv"][0].detach().cpu().numpy()
            left_mask = aux["left_mask"][0, 0].detach().cpu().numpy()
            right_mask = aux["right_mask"][0, 0].detach().cpu().numpy()
            pair_mask = aux.get("pair_mask")
            pair_np = pair_mask[0, 0].detach().cpu().numpy() if pair_mask is not None else np.zeros_like(left_mask)

            pred_overlay = draw_uv_overlay(
                rgb,
                pred_uv[horizon_idx, 0],
                np.ones(2, dtype=np.float32),
                token_grid,
                draw_pair=True,
            )
            pred_left = overlay_mask(rgb, left_mask, token_grid, (255, 60, 60), alpha=0.45)
            pred_right = overlay_mask(rgb, right_mask, token_grid, (60, 120, 255), alpha=0.45)
            pred_pair = overlay_mask(rgb, pair_np, token_grid, (60, 220, 120), alpha=0.45)
            interp_overlay = draw_uv_overlay(rgb, interp_uv, interp_valid, token_grid, draw_pair=True)
            panel = compose_panels(
                [rgb, current_overlay, target_overlay, interp_overlay, pred_overlay, pred_left, pred_right, pred_pair, pca],
                [
                    "raw RGB",
                    "current UV",
                    "expert target UV",
                    f"interp target {sweep_label}",
                    f"pred UV {sweep_label}",
                    "pred left mask",
                    "pred right mask",
                    "pred pair mask",
                    "DINO PCA",
                ],
                cols=3,
            )
            out_path = out_dir / f"sample_{sample_id:03d}_step_{idx:06d}_{out_suffix}.png"
            save_rgb(out_path, panel)
            row = {
                "step": idx,
                "horizon_index": int(horizon_idx),
                "output": str(out_path),
                "pred_uv_mean": float(pred_uv.mean()),
                "pred_uv_std": float(pred_uv.std()),
                "left_mask_mass": float(left_mask.sum()),
                "right_mask_mass": float(right_mask.sum()),
                "pair_mask_mass": float(pair_np.sum()),
                "current_uv": current_uv.tolist(),
                "current_valid": current_valid.tolist(),
                "target_uv": target_uv.tolist(),
                "target_valid": target_valid.tolist(),
                "interp_uv": interp_uv.tolist(),
                "interp_valid": interp_valid.tolist(),
            }
            if sweep_kind == "lambda":
                row["lambda"] = float(sweep_value)
            else:
                row["timestep"] = int(sweep_value)
            summary.append(row)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="GAP zarr dataset")
    parser.add_argument("--hdf5-root", required=True, help="RoboTwin task/config root containing data/episode<N>.hdf5")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--token-grid", default="15,20")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or a torch device like cuda:0")
    parser.add_argument("--timesteps", default="0,25,50,75,99")
    parser.add_argument("--flow-lambdas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--horizon-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans-k", type=int, default=6)
    parser.add_argument("--max-kmeans-iter", type=int, default=20)
    args = parser.parse_args()

    token_grid = parse_token_grid(args.token_grid)
    root = zarr.open(str(Path(args.zarr).expanduser().resolve()), mode="r")
    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError("Expected zarr with data group and meta/episode_ends.")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"no_checkpoint": no_checkpoint_visuals(args, root, episode_ends, token_grid)}
    if args.checkpoint:
        try:
            summary["checkpoint"] = checkpoint_visuals(args, root, episode_ends, token_grid)
        except Exception as exc:
            summary["checkpoint_error"] = str(exc)
            print(f"checkpoint visualization failed: {exc}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote visualization summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
