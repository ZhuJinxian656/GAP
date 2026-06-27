#!/usr/bin/env python
"""
Inspect RoboTwin HDF5 files and highlight candidate keys for triadic features.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from gap_policy.triadic import (  # noqa: E402
    DEFAULT_LEFT_EEF_KEYS,
    DEFAULT_LEFT_GRIPPER_KEYS,
    DEFAULT_OBJECT_KEYS,
    DEFAULT_PROPRIO_KEYS,
    DEFAULT_RIGHT_EEF_KEYS,
    DEFAULT_RIGHT_GRIPPER_KEYS,
)

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required to inspect RoboTwin HDF5 files.") from exc


CANDIDATE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "left end-effector pose/position": DEFAULT_LEFT_EEF_KEYS,
    "right end-effector pose/position": DEFAULT_RIGHT_EEF_KEYS,
    "left gripper state": DEFAULT_LEFT_GRIPPER_KEYS,
    "right gripper state": DEFAULT_RIGHT_GRIPPER_KEYS,
    "object pose/position/keypoints": DEFAULT_OBJECT_KEYS,
    "proprioception/state fallback": DEFAULT_PROPRIO_KEYS,
}

HEURISTICS: Dict[str, Tuple[str, ...]] = {
    "left end-effector pose/position": ("left", "eef", "ee", "tcp", "end_effector"),
    "right end-effector pose/position": ("right", "eef", "ee", "tcp", "end_effector"),
    "left gripper state": ("left", "gripper"),
    "right gripper state": ("right", "gripper"),
    "object pose/position/keypoints": ("object", "pose", "pos", "position", "keypoint", "point"),
    "object point cloud": ("object", "pointcloud", "point_cloud", "points"),
    "segmentation/object id": ("seg", "segment", "mask", "object_id", "id"),
}


def iter_hdf5_files(paths: Iterable[str], max_files: int) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith((".hdf5", ".h5")):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in names:
                    if name.endswith((".hdf5", ".h5")):
                        files.append(os.path.join(root, name))
                        if len(files) >= max_files:
                            return files
        else:
            print(f"[warn] Skipping non-HDF5 path: {path}", file=sys.stderr)
        if len(files) >= max_files:
            break
    return files


def describe_array(dataset, sample_items: int) -> str:
    shape = tuple(dataset.shape)
    dtype = dataset.dtype
    summary = f"shape={shape}, dtype={dtype}"
    if np.prod(shape, dtype=np.int64) == 0:
        return summary
    if dtype.kind in {"O", "S", "V"}:
        return summary + ", sample=<binary/object omitted>"
    try:
        sample = dataset[()]
        arr = np.asarray(sample)
        if arr.size > sample_items:
            arr = arr.reshape(-1)[:sample_items]
        summary += f", sample={np.array2string(arr, threshold=sample_items, edgeitems=sample_items)}"
    except Exception as exc:
        summary += f", sample=<unavailable: {exc}>"
    return summary


def collect_datasets(handle) -> Dict[str, object]:
    datasets: Dict[str, object] = {}

    def visitor(name, obj):
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            datasets[f"/{name}"] = obj

    handle.visititems(visitor)
    return datasets


def candidate_matches(datasets: Dict[str, object]) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {group: [] for group in CANDIDATE_GROUPS}
    lower_keys = {key.lower(): key for key in datasets}

    for group, candidates in CANDIDATE_GROUPS.items():
        for candidate in candidates:
            clean = candidate.strip("/").lower()
            for lower_key, original in lower_keys.items():
                if lower_key == f"/{clean}" or lower_key.endswith(f"/{clean}"):
                    matches[group].append(original)

    for group, tokens in HEURISTICS.items():
        matches.setdefault(group, [])
        for lower_key, original in lower_keys.items():
            if original in matches[group]:
                continue
            hits = [token for token in tokens if token in lower_key]
            if group.startswith("left") and "right" in lower_key:
                continue
            if group.startswith("right") and "left" in lower_key:
                continue
            if len(hits) >= 2:
                matches[group].append(original)

    return {key: sorted(set(value)) for key, value in matches.items()}


def inspect_file(path: str, sample_items: int) -> None:
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as handle:
        datasets = collect_datasets(handle)
        print("HDF5 datasets:")
        for key in sorted(datasets):
            print(f"  {key}: {describe_array(datasets[key], sample_items)}")

        print("\nCandidate triadic keys:")
        matches = candidate_matches(datasets)
        for group, keys in matches.items():
            if keys:
                print(f"  {group}:")
                for key in keys:
                    print(f"    - {key}: {describe_array(datasets[key], sample_items)}")
            else:
                print(f"  {group}: <not found>")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print RoboTwin HDF5 hierarchy and candidate triadic-relation keys."
    )
    parser.add_argument("paths", nargs="+", help="One or more .hdf5/.h5 files or directories")
    parser.add_argument("--max_files", type=int, default=5, help="Max files to inspect when a directory is provided")
    parser.add_argument("--sample_items", type=int, default=8, help="Max scalar values to print per dataset")
    args = parser.parse_args()

    files = iter_hdf5_files(args.paths, max_files=args.max_files)
    if not files:
        raise SystemExit("No HDF5 files found.")
    for path in files:
        inspect_file(path, sample_items=args.sample_items)


if __name__ == "__main__":
    main()
