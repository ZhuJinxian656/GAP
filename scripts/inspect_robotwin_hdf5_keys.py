#!/usr/bin/env python
"""Inspect RoboTwin HDF5 files and summarize candidate supervision fields."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("h5py is required to inspect RoboTwin HDF5 files.") from exc


KEYWORD_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "rgb/image": ("rgb", "image"),
    "depth": ("depth",),
    "camera": ("camera", "cam2world"),
    "intrinsics": ("intrinsic", "intrinsics"),
    "extrinsics": ("extrinsic", "extrinsics", "cam2world"),
    "segmentation": ("seg", "segmentation", "segment"),
    "mask": ("mask",),
    "object/id/name": ("object", "obj", "object_id", "object_name"),
    "object pose": ("object_pose", "object_pos", "object_position", "target_object"),
    "object keypoints": ("object_keypoint", "object_points", "keypoint"),
    "bbox": ("bbox", "bounding_box"),
    "contact": ("contact", "touch"),
    "eef/ee/tcp": ("eef", "ee", "tcp", "end_effector", "endpose"),
    "gripper": ("gripper",),
    "left": ("left",),
    "right": ("right",),
    "joint/action": ("joint", "action", "vector", "qpos", "qvel"),
    "pointcloud": ("pointcloud", "point_cloud"),
}


def iter_hdf5_files(paths: Iterable[str], max_files: int) -> List[str]:
    files: List[str] = []
    for path in paths:
        if os.path.isfile(path) and path.endswith((".hdf5", ".h5")):
            files.append(path)
        elif os.path.isdir(path):
            for root, _, names in os.walk(path):
                for name in sorted(names):
                    if name.endswith((".hdf5", ".h5")):
                        files.append(os.path.join(root, name))
                        if len(files) >= max_files:
                            return files
        else:
            print(f"[warn] Skipping non-HDF5 path: {path}", file=sys.stderr)
        if len(files) >= max_files:
            break
    return files


def collect_datasets(handle: h5py.File) -> Dict[str, h5py.Dataset]:
    datasets: Dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: object) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            datasets[f"/{name}"] = obj  # type: ignore[assignment]

    handle.visititems(visitor)
    return datasets


def safe_sample(dataset: h5py.Dataset, sample_items: int) -> object:
    shape = tuple(dataset.shape)
    dtype = dataset.dtype
    if np.prod(shape, dtype=np.int64) == 0:
        return []
    if dtype.kind in {"O", "S", "V"}:
        return "<binary/object omitted>"
    try:
        arr = np.asarray(dataset[()])
        if arr.size > sample_items:
            arr = arr.reshape(-1)[:sample_items]
        return arr.tolist()
    except Exception as exc:  # pragma: no cover - defensive for unusual HDF5 filters.
        return f"<unavailable: {exc}>"


def describe_dataset(dataset: h5py.Dataset, sample_items: int) -> Dict[str, object]:
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "sample": safe_sample(dataset, sample_items=sample_items),
    }


def keyword_hits(datasets: Mapping[str, h5py.Dataset]) -> Dict[str, List[str]]:
    lower_to_key = {key.lower(): key for key in datasets}
    result: Dict[str, List[str]] = {}
    for group, tokens in KEYWORD_GROUPS.items():
        hits = []
        for lower_key, original_key in lower_to_key.items():
            if any(token in lower_key for token in tokens):
                hits.append(original_key)
        result[group] = sorted(set(hits))
    return result


def inspect_file(path: str, sample_items: int, print_output: bool = True) -> Dict[str, object]:
    with h5py.File(path, "r") as handle:
        datasets = collect_datasets(handle)
        dataset_summary = {
            key: describe_dataset(dataset, sample_items=sample_items)
            for key, dataset in sorted(datasets.items())
        }
        hits = keyword_hits(datasets)

    summary = {
        "path": os.path.abspath(path),
        "dataset_count": len(dataset_summary),
        "datasets": dataset_summary,
        "keyword_hits": hits,
    }

    if print_output:
        print(f"\n=== {path} ===")
        print("HDF5 datasets:")
        for key, info in dataset_summary.items():
            print(f"  {key}: shape={tuple(info['shape'])}, dtype={info['dtype']}, sample={info['sample']}")

        print("\nKeyword hits:")
        for group, keys in hits.items():
            print(f"  {group}: {keys if keys else '<not found>'}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Print RoboTwin HDF5 hierarchy and candidate supervision keys.")
    parser.add_argument("paths", nargs="*", help="One or more .hdf5/.h5 files or directories")
    parser.add_argument("--path", action="append", default=[], help="Additional .hdf5/.h5 file or directory")
    parser.add_argument("--max-files", "--max_files", dest="max_files", type=int, default=5)
    parser.add_argument("--sample-items", "--sample_items", dest="sample_items", type=int, default=8)
    parser.add_argument("--save-json", default=None, help="Optional path to save machine-readable summary JSON")
    args = parser.parse_args()

    paths = list(args.paths) + list(args.path)
    if not paths:
        raise SystemExit("Provide at least one path via positional args or --path.")

    files = iter_hdf5_files(paths, max_files=args.max_files)
    if not files:
        raise SystemExit("No HDF5 files found.")

    summaries = [inspect_file(path, sample_items=args.sample_items) for path in files]

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump({"files": summaries}, f, indent=2)
        print(f"\nSaved JSON summary to: {args.save_json}")


if __name__ == "__main__":
    main()
