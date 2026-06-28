#!/usr/bin/env python3
"""Measure how copyable the future Pi3 target is from the current Pi3 tokens."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def build_pairs(episode_ends: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current: list[int] = []
    future: list[int] = []
    episodes: list[int] = []
    start = 0
    for episode, end in enumerate(episode_ends.astype(int).tolist()):
        for idx in range(start, end):
            current.append(idx)
            future.append(min(idx + horizon - 1, end - 1))
            episodes.append(episode)
        start = end
    return np.asarray(current), np.asarray(future), np.asarray(episodes)


def select_samples(
    current: np.ndarray,
    future: np.ndarray,
    episodes: np.ndarray,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples is None or max_samples >= len(current):
        return current, future, episodes
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    keep = np.linspace(0, len(current) - 1, num=max_samples, dtype=int)
    return current[keep], future[keep], episodes[keep]


def read_batch(array: zarr.Array, indices: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(array[int(index)]) for index in indices], axis=0)


def summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    return {
        "mean": float(values.mean()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="Path to GAP zarr.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument("--changed-token-percentile", type=float, default=90.0)
    parser.add_argument("--static-delta-threshold", type=float, default=1e-3)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--csv", default=None, help="Optional per-episode CSV output.")
    parser.add_argument("--plot", default=None, help="Optional histogram PNG output.")
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root or "pi3_features" not in root["data"]:
        raise KeyError(f"{zarr_path} does not contain data/pi3_features.")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError(f"{zarr_path} does not contain meta/episode_ends.")

    pi3 = root["data"]["pi3_features"]
    episode_ends = np.asarray(root["meta"]["episode_ends"])
    current_idx, future_idx, episode_idx = build_pairs(episode_ends, args.horizon)
    current_idx, future_idx, episode_idx = select_samples(
        current_idx,
        future_idx,
        episode_idx,
        args.max_samples,
    )

    mask_arrays = {
        key: root["data"][key]
        for key in ("pi3_eef_region_mask", "pi3_non_eef_region_mask")
        if key in root["data"]
    }
    mask_stats: dict[str, dict[str, Any]] = {
        key: {"sse": 0.0, "count": 0, "delta_norms": [], "selected_tokens": 0, "total_tokens": 0}
        for key in mask_arrays
    }

    total_sse = 0.0
    total_count = 0
    delta_norm_chunks: list[np.ndarray] = []
    token_episode_chunks: list[np.ndarray] = []
    per_episode: dict[int, dict[str, float]] = defaultdict(
        lambda: {"samples": 0, "sse": 0.0, "count": 0, "delta_norm_sum": 0.0, "tokens": 0}
    )

    for start in range(0, len(current_idx), args.batch_size):
        end = min(start + args.batch_size, len(current_idx))
        batch_current_idx = current_idx[start:end]
        batch_future_idx = future_idx[start:end]
        batch_episode_idx = episode_idx[start:end]

        current = read_batch(pi3, batch_current_idx).astype(np.float32)
        future = read_batch(pi3, batch_future_idx).astype(np.float32)
        delta = future - current
        delta_norm = np.linalg.norm(delta, axis=-1)
        total_sse += float(np.square(delta).sum())
        total_count += int(delta.size)

        flat_delta_norm = delta_norm.reshape(-1)
        delta_norm_chunks.append(flat_delta_norm.astype(np.float32, copy=False))
        token_episode_chunks.append(np.repeat(batch_episode_idx, delta_norm.shape[1] * delta_norm.shape[2]))

        per_sample_sse = np.square(delta).reshape(delta.shape[0], -1).sum(axis=1)
        per_sample_count = np.full(delta.shape[0], delta.shape[1] * delta.shape[2] * delta.shape[3])
        per_sample_delta_norm_sum = delta_norm.reshape(delta.shape[0], -1).sum(axis=1)
        per_sample_tokens = np.full(delta.shape[0], delta_norm.shape[1] * delta_norm.shape[2])
        for row, episode in enumerate(batch_episode_idx.tolist()):
            summary = per_episode[int(episode)]
            summary["samples"] += 1
            summary["sse"] += float(per_sample_sse[row])
            summary["count"] += int(per_sample_count[row])
            summary["delta_norm_sum"] += float(per_sample_delta_norm_sum[row])
            summary["tokens"] += int(per_sample_tokens[row])

        for key, array in mask_arrays.items():
            mask = read_batch(array, batch_current_idx).astype(np.float32) > 0.5
            selected = int(mask.sum())
            mask_stats[key]["selected_tokens"] += selected
            mask_stats[key]["total_tokens"] += int(mask.size)
            if selected == 0:
                continue
            mask_expanded = mask[..., None]
            mask_stats[key]["sse"] += float(np.square(delta * mask_expanded).sum())
            mask_stats[key]["count"] += int(selected * delta.shape[-1])
            mask_stats[key]["delta_norms"].append(delta_norm[mask].astype(np.float32, copy=False))

    delta_norms = np.concatenate(delta_norm_chunks) if delta_norm_chunks else np.asarray([], dtype=np.float32)
    token_episodes = np.concatenate(token_episode_chunks) if token_episode_chunks else np.asarray([], dtype=np.int64)
    changed_threshold = float(np.percentile(delta_norms, args.changed_token_percentile)) if delta_norms.size else 0.0
    static_mask = delta_norms <= args.static_delta_threshold
    changed_mask = delta_norms >= changed_threshold

    per_episode_rows = []
    for episode in sorted(per_episode):
        summary = per_episode[episode]
        episode_tokens = token_episodes == episode
        token_count = int(episode_tokens.sum())
        per_episode_rows.append(
            {
                "episode": episode,
                "samples": int(summary["samples"]),
                "copy_mse": float(summary["sse"] / max(summary["count"], 1)),
                "mean_delta_norm": float(summary["delta_norm_sum"] / max(summary["tokens"], 1)),
                "static_token_ratio": float(static_mask[episode_tokens].mean()) if token_count else 0.0,
                "changed_token_ratio": float(changed_mask[episode_tokens].mean()) if token_count else 0.0,
            }
        )

    mask_summary = {}
    for key, stats in mask_stats.items():
        values = np.concatenate(stats["delta_norms"]) if stats["delta_norms"] else np.asarray([], dtype=np.float32)
        mask_summary[key] = {
            "label": "EEF proxy mask, not a true object-hand mask",
            "copy_mse": float(stats["sse"] / max(stats["count"], 1)),
            "selected_token_ratio": float(stats["selected_tokens"] / max(stats["total_tokens"], 1)),
            "delta_norm": summarize_values(values),
        }

    summary = {
        "zarr": str(zarr_path),
        "pi3_shape": list(pi3.shape),
        "horizon": args.horizon,
        "samples": int(len(current_idx)),
        "episodes": int(len(episode_ends)),
        "copy_mse": float(total_sse / max(total_count, 1)),
        "delta_norm": summarize_values(delta_norms),
        "static_delta_threshold": args.static_delta_threshold,
        "static_token_ratio": float(static_mask.mean()) if delta_norms.size else 0.0,
        "changed_token_percentile": args.changed_token_percentile,
        "changed_token_threshold": changed_threshold,
        "changed_token_ratio": float(changed_mask.mean()) if delta_norms.size else 0.0,
        "per_episode": per_episode_rows,
        "mask_summaries": mask_summary,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["episode", "samples", "copy_mse", "mean_delta_norm", "static_token_ratio", "changed_token_ratio"],
            )
            writer.writeheader()
            writer.writerows(per_episode_rows)

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            plot_path = Path(args.plot)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(7, 4))
            plt.hist(delta_norms, bins=80)
            plt.axvline(changed_threshold, color="red", linestyle="--", label="changed threshold")
            plt.xlabel("||z_future - z_current|| per token")
            plt.ylabel("count")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except Exception as exc:  # pragma: no cover - optional plotting dependency
            summary["plot_error"] = str(exc)
            output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote copyability summary to {output}")
    print(f"copy_mse={summary['copy_mse']:.6g}")
    print(f"delta_norm_p50={summary['delta_norm']['p50']:.6g} p90={summary['delta_norm']['p90']:.6g}")
    print(f"changed_token_ratio={summary['changed_token_ratio']:.4f}")


if __name__ == "__main__":
    main()
