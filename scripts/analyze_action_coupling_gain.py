#!/usr/bin/env python3
"""Measure whether expert actions improve prediction of future Pi3 deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


def build_pairs(episode_ends: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current: list[int] = []
    future: list[int] = []
    chunk_start: list[int] = []
    chunk_end: list[int] = []
    start = 0
    for end in episode_ends.astype(int).tolist():
        for idx in range(start, end):
            current.append(idx)
            future.append(min(idx + horizon - 1, end - 1))
            chunk_start.append(idx)
            chunk_end.append(end)
        start = end
    return (
        np.asarray(current),
        np.asarray(future),
        np.asarray(chunk_start),
        np.asarray(chunk_end),
    )


def select_samples(arrays: tuple[np.ndarray, ...], max_samples: int | None) -> tuple[np.ndarray, ...]:
    if max_samples is None or max_samples >= len(arrays[0]):
        return arrays
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    keep = np.linspace(0, len(arrays[0]) - 1, num=max_samples, dtype=int)
    return tuple(array[keep] for array in arrays)


def read_batch(array: zarr.Array, indices: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(array[int(index)]) for index in indices], axis=0)


def read_action_chunks(action: zarr.Array, starts: np.ndarray, episode_ends: np.ndarray, horizon: int) -> np.ndarray:
    chunks = []
    for start, end in zip(starts.astype(int).tolist(), episode_ends.astype(int).tolist()):
        rows = []
        for offset in range(horizon):
            rows.append(np.asarray(action[min(start + offset, end - 1)]))
        chunks.append(np.stack(rows, axis=0))
    return np.stack(chunks, axis=0)


def changed_token_delta(delta: np.ndarray, percentile: float) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=-1)
    flat_norm = norm.reshape(norm.shape[0], -1)
    flat_delta = delta.reshape(delta.shape[0], -1, delta.shape[-1])
    thresholds = np.percentile(flat_norm, percentile, axis=1)
    targets = []
    for row in range(flat_norm.shape[0]):
        mask = flat_norm[row] >= thresholds[row]
        if not np.any(mask):
            mask[np.argmax(flat_norm[row])] = True
        targets.append(flat_delta[row, mask].mean(axis=0))
    return np.stack(targets, axis=0)


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_train.mean(axis=0, keepdims=True)

    x_train_std = (x_train - x_mean) / x_std
    x_test_std = (x_test - x_mean) / x_std
    y_centered = y_train - y_mean

    xtx = x_train_std.T @ x_train_std
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=xtx.dtype)
    xty = x_train_std.T @ y_centered
    try:
        weights = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(xtx + reg) @ xty
    return x_test_std @ weights + y_mean


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", choices=["pooled_delta", "changed_token_delta"], default="pooled_delta")
    parser.add_argument("--changed-token-percentile", type=float, default=90.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    zarr_path = Path(args.zarr)
    root = zarr.open(str(zarr_path), mode="r")
    data = root["data"]
    for key in ("pi3_features", "state", "action"):
        if key not in data:
            raise KeyError(f"{zarr_path} is missing data/{key}.")
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError(f"{zarr_path} is missing meta/episode_ends.")

    episode_ends = np.asarray(root["meta"]["episode_ends"])
    current_idx, future_idx, chunk_start, chunk_end = select_samples(
        build_pairs(episode_ends, args.horizon),
        args.max_samples,
    )

    x_current_parts = []
    x_action_parts = []
    y_parts = []
    for start in range(0, len(current_idx), args.batch_size):
        end = min(start + args.batch_size, len(current_idx))
        batch_current = current_idx[start:end]
        batch_future = future_idx[start:end]
        batch_chunk_start = chunk_start[start:end]
        batch_chunk_end = chunk_end[start:end]

        current_pi3 = read_batch(data["pi3_features"], batch_current).astype(np.float32)
        future_pi3 = read_batch(data["pi3_features"], batch_future).astype(np.float32)
        state = read_batch(data["state"], batch_current).astype(np.float32)
        action_chunk = read_action_chunks(data["action"], batch_chunk_start, batch_chunk_end, args.horizon).astype(np.float32)

        delta = future_pi3 - current_pi3
        pooled_current = current_pi3.mean(axis=(1, 2))
        if args.target == "pooled_delta":
            target = delta.mean(axis=(1, 2))
        else:
            target = changed_token_delta(delta, args.changed_token_percentile)

        x_current_parts.append(np.concatenate([pooled_current, state], axis=1))
        x_action_parts.append(action_chunk.reshape(action_chunk.shape[0], -1))
        y_parts.append(target)

    x_current = np.concatenate(x_current_parts, axis=0).astype(np.float64)
    x_action = np.concatenate(x_action_parts, axis=0).astype(np.float64)
    y = np.concatenate(y_parts, axis=0).astype(np.float64)
    x_current_action = np.concatenate([x_current, x_action], axis=1)

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in (0, 1).")
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(y))
    rng.shuffle(order)
    train_n = max(1, min(len(y) - 1, int(round(len(y) * args.train_ratio))))
    train_idx = order[:train_n]
    test_idx = order[train_n:]

    pred_current = ridge_fit_predict(
        x_current[train_idx],
        y[train_idx],
        x_current[test_idx],
        args.ridge_alpha,
    )
    pred_current_action = ridge_fit_predict(
        x_current_action[train_idx],
        y[train_idx],
        x_current_action[test_idx],
        args.ridge_alpha,
    )

    mse_current = mse(pred_current, y[test_idx])
    mse_current_action = mse(pred_current_action, y[test_idx])
    coupling_gain = mse_current - mse_current_action
    relative_gain = coupling_gain / max(mse_current, 1e-12)

    summary = {
        "zarr": str(zarr_path),
        "target": args.target,
        "samples": int(len(y)),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "target_dim": int(y.shape[1]),
        "current_feature_dim": int(x_current.shape[1]),
        "action_chunk_dim": int(x_action.shape[1]),
        "ridge_alpha": args.ridge_alpha,
        "train_ratio": args.train_ratio,
        "mse_current_only": mse_current,
        "mse_current_action": mse_current_action,
        "coupling_gain": coupling_gain,
        "relative_gain": relative_gain,
        "interpretation": (
            "Positive coupling_gain means the expert action chunk improved future-delta prediction; "
            "negative or near-zero gain means the target is mostly predictable from current state/features."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote action-coupling summary to {output}")
    print(f"mse_current_only={mse_current:.6g}")
    print(f"mse_current_action={mse_current_action:.6g}")
    print(f"relative_gain={relative_gain:.4f}")


if __name__ == "__main__":
    main()
