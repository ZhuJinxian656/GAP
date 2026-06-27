#!/usr/bin/env python
"""Inspect a GAP zarr dataset without dumping large arrays."""
from __future__ import annotations

import argparse
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import zarr


EXPECTED_ARRAYS = (
    "dinov3_features",
    "pi3_features",
    "state",
    "action",
    "future_pi3_features",
)

MASK_ARRAYS = (
    "pi3_object_hand_mask",
    "pi3_object_mask",
    "pi3_left_hand_mask",
    "pi3_right_hand_mask",
    "pi3_background_mask",
)

INTERACTION_KEYWORDS = (
    "eef",
    "ee",
    "endpose",
    "object_pose",
    "object_pos",
    "object_keypoint",
    "keypoint",
    "contact",
    "interaction",
    "gripper",
)


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def iter_arrays(group: zarr.Group, prefix: str = "") -> Iterable[Tuple[str, zarr.Array]]:
    for name in sorted(group.array_keys()):
        path = f"{prefix}/{name}" if prefix else name
        yield path, group[name]
    for name in sorted(group.group_keys()):
        child = group[name]
        path = f"{prefix}/{name}" if prefix else name
        yield from iter_arrays(child, path)


def array_nbytes(array: zarr.Array) -> int:
    itemsize = np.dtype(array.dtype).itemsize
    return int(np.prod(array.shape, dtype=np.int64)) * itemsize


def sample_array(array: zarr.Array, sample_rows: int, sample_values: int) -> np.ndarray:
    if len(array.shape) == 0:
        return np.asarray(array[()])
    rows = min(sample_rows, array.shape[0])
    if rows <= 0:
        return np.asarray([])
    sample = np.asarray(array[:rows])
    if sample.size > sample_values:
        flat = sample.reshape(-1)
        idx = np.linspace(0, flat.size - 1, sample_values, dtype=np.int64)
        sample = flat[idx]
    return sample


def stats_for_array(array: zarr.Array, sample_rows: int, sample_values: int) -> Dict[str, object]:
    sample = sample_array(array, sample_rows=sample_rows, sample_values=sample_values)
    stats: Dict[str, object] = {
        "sample_size": int(sample.size),
    }
    if sample.size == 0:
        return stats
    if sample.dtype.kind not in {"b", "i", "u", "f", "c"}:
        stats["note"] = "non-numeric sample omitted"
        return stats
    finite = np.isfinite(sample)
    finite_sample = sample[finite]
    stats["finite_ratio"] = float(finite.mean()) if finite.size else 1.0
    if finite_sample.size == 0:
        return stats
    stats.update(
        {
            "min": float(finite_sample.min()),
            "max": float(finite_sample.max()),
            "mean": float(finite_sample.mean()),
            "std": float(finite_sample.std()),
        }
    )
    return stats


def find_named_arrays(array_paths: List[str], names: Iterable[str]) -> Dict[str, List[str]]:
    result = {}
    for name in names:
        hits = [
            path
            for path in array_paths
            if path == name or path.endswith("/" + name)
        ]
        result[name] = hits
    return result


def find_keyword_arrays(array_paths: List[str], keywords: Iterable[str]) -> Dict[str, List[str]]:
    result = {}
    for keyword in keywords:
        lower = keyword.lower()
        hits = [path for path in array_paths if lower in path.lower()]
        result[keyword] = hits
    return result


def inspect_zarr(path: str, sample_rows: int, sample_values: int) -> None:
    root = zarr.open(path, mode="r")
    print(f"Zarr: {os.path.abspath(path)}")
    print(f"Groups: {sorted(root.group_keys()) if hasattr(root, 'group_keys') else []}")

    arrays = list(iter_arrays(root))
    if not arrays:
        print("No arrays found.")
        return

    print("\nArrays:")
    for name, array in arrays:
        print(
            f"  {name}: shape={array.shape}, dtype={array.dtype}, "
            f"chunks={array.chunks}, estimated_size={format_bytes(array_nbytes(array))}"
        )

    episode_ends = None
    for name, array in arrays:
        if name == "episode_ends" or name.endswith("/episode_ends"):
            episode_ends = array
            break
    if episode_ends is not None:
        first = np.asarray(episode_ends[: min(10, episode_ends.shape[0])]).tolist()
        print(f"\nEpisode ends: count={episode_ends.shape[0]}, first={first}")

    print("\nSample stats:")
    for name, array in arrays:
        stats = stats_for_array(array, sample_rows=sample_rows, sample_values=sample_values)
        print(f"  {name}: {stats}")

    array_paths = [name for name, _ in arrays]
    print("\nExpected GAP arrays:")
    for name, hits in find_named_arrays(array_paths, EXPECTED_ARRAYS).items():
        print(f"  {name}: {hits if hits else '<missing>'}")

    print("\nMask arrays:")
    for name, hits in find_named_arrays(array_paths, MASK_ARRAYS).items():
        print(f"  {name}: {hits if hits else '<missing>'}")

    print("\nInteraction-related arrays:")
    keyword_hits = find_keyword_arrays(array_paths, INTERACTION_KEYWORDS)
    any_hit = False
    for keyword, hits in keyword_hits.items():
        if hits:
            any_hit = True
            print(f"  {keyword}: {hits}")
    if not any_hit:
        print("  <none found>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect GAP zarr arrays, shapes, chunks, and sampled stats.")
    parser.add_argument("--zarr", required=True, help="Path to a GAP .zarr directory")
    parser.add_argument("--sample-rows", type=int, default=8, help="Leading rows to sample from each array")
    parser.add_argument("--sample-values", type=int, default=4096, help="Maximum scalar values used for stats")
    args = parser.parse_args()

    inspect_zarr(args.zarr, sample_rows=args.sample_rows, sample_values=args.sample_values)


if __name__ == "__main__":
    main()
