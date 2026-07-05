#!/usr/bin/env python
"""Add deployable triadic/coupling state to an existing GAP zarr dataset."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

GAP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GAP_ROOT))

from gap_policy.triadic import TriadicConfig, build_triadic_state_from_hdf5  # noqa: E402


def parse_key_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def build_config(args: argparse.Namespace) -> TriadicConfig:
    cfg = TriadicConfig(
        mode=args.triadic_mode,
        include_grippers=not args.no_triadic_grippers,
    )
    for attr, value in (
        ("left_eef_keys", parse_key_list(args.left_eef_keys)),
        ("right_eef_keys", parse_key_list(args.right_eef_keys)),
        ("object_keys", parse_key_list(args.object_keys)),
        ("left_gripper_keys", parse_key_list(args.left_gripper_keys)),
        ("right_gripper_keys", parse_key_list(args.right_gripper_keys)),
    ):
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def episode_path(hdf5_root: Path, episode_id: int) -> Path:
    path = hdf5_root / "data" / f"episode{episode_id}.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"Missing HDF5 episode: {path}")
    return path


def build_episode_state(path: Path, cfg: TriadicConfig, expected_len: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        triadic = build_triadic_state_from_hdf5(
            handle,
            cfg,
            warn_fn=lambda msg: print(f"Warning: {path.name}: {msg}"),
        )
        if triadic is None:
            raise RuntimeError(
                f"Could not build triadic_state for {path}. "
                "Inspect HDF5 keys or use --triadic-mode proprio_only_fallback."
            )
        vector_len = int(handle["/joint_action/vector"].shape[0])

    if triadic.shape[0] < expected_len:
        raise RuntimeError(
            f"{path} produced triadic_state length {triadic.shape[0]}, "
            f"but zarr episode needs {expected_len} rows."
        )
    if vector_len - 1 != expected_len:
        print(
            f"Warning: {path.name}: HDF5 vector_len - 1 is {vector_len - 1}, "
            f"zarr episode length is {expected_len}."
        )
    return np.asarray(triadic[:expected_len], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, help="Existing GAP zarr dataset to update in-place")
    parser.add_argument(
        "--hdf5-root",
        required=True,
        help="RoboTwin task/config root containing data/episode<N>.hdf5",
    )
    parser.add_argument("--expert-data-num", type=int, default=None)
    parser.add_argument(
        "--triadic-mode",
        choices=["pairwise", "triadic", "proprio_only_fallback"],
        default="proprio_only_fallback",
    )
    parser.add_argument("--no-triadic-grippers", action="store_true")
    parser.add_argument("--left-eef-keys", default=None)
    parser.add_argument("--right-eef-keys", default=None)
    parser.add_argument("--object-keys", default=None)
    parser.add_argument("--left-gripper-keys", default=None)
    parser.add_argument("--right-gripper-keys", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    zarr_path = Path(args.zarr).expanduser().resolve()
    hdf5_root = Path(args.hdf5_root).expanduser().resolve()
    root = zarr.open(str(zarr_path), mode="a")
    if "data" not in root or "meta" not in root or "episode_ends" not in root["meta"]:
        raise KeyError(f"{zarr_path} is not a GAP zarr with data/ and meta/episode_ends.")
    data = root["data"]
    if "triadic_state" in data and not args.overwrite:
        raise FileExistsError(f"{zarr_path}/data/triadic_state exists; pass --overwrite to replace it.")
    if "state" not in data:
        raise KeyError(f"{zarr_path} is missing data/state.")

    cfg = build_config(args)
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    if args.expert_data_num is not None:
        episode_ends = episode_ends[: args.expert_data_num]
    states = []
    start = 0
    for episode_id, end in enumerate(episode_ends):
        expected_len = int(end - start)
        states.append(build_episode_state(episode_path(hdf5_root, episode_id), cfg, expected_len))
        start = int(end)

    triadic_state = np.concatenate(states, axis=0).astype(np.float32)
    expected_rows = int(data["state"].shape[0])
    if triadic_state.shape[0] != expected_rows:
        raise RuntimeError(
            f"Built {triadic_state.shape[0]} triadic rows, but zarr data/state has {expected_rows} rows."
        )

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    data.create_dataset(
        "triadic_state",
        data=triadic_state,
        chunks=(100, triadic_state.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    print(f"Wrote data/triadic_state {triadic_state.shape} to {zarr_path}")
    print(f"triadic_mode={cfg.mode} include_grippers={cfg.include_grippers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
