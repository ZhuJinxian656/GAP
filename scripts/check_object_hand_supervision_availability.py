#!/usr/bin/env python
"""Check whether current data can support object-hand / interaction supervision."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import zarr

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


OBJECT_TOKENS = ("object", "obj", "target_object")
MASK_TOKENS = ("mask",)
SEGMENTATION_TOKENS = ("seg", "segment", "segmentation")
DEPTH_TOKENS = ("depth",)
CONTACT_TOKENS = ("contact", "touch")
KEYPOINT_TOKENS = ("keypoint", "keypoints", "object_points")
BBOX_TOKENS = ("bbox", "bounding_box")
CAM_INTRINSIC_TOKENS = ("intrinsic", "intrinsics")
CAM_EXTRINSIC_TOKENS = ("extrinsic", "extrinsics", "cam2world")
EEF_TOKENS = ("eef", "ee", "tcp", "end_effector", "endpose")
GRIPPER_TOKENS = ("gripper",)
HAND_MASK_TOKENS = ("left_hand", "right_hand", "hand_mask", "arm_mask", "robot_mask")

ZARR_MASK_ARRAYS = (
    "pi3_object_hand_mask",
    "pi3_object_mask",
    "pi3_left_hand_mask",
    "pi3_right_hand_mask",
    "pi3_background_mask",
)

ZARR_INTERACTION_ARRAYS = (
    "interaction_state",
    "future_interaction_state",
)


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
        if len(files) >= max_files:
            break
    return files


def hdf5_keys(path: str) -> List[str]:
    if h5py is None:
        raise SystemExit("h5py is required for HDF5 inspection.")
    keys: List[str] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, obj: object) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                keys.append("/" + name)
        handle.visititems(visitor)
    return sorted(keys)


def iter_zarr_arrays(group: zarr.Group, prefix: str = "") -> List[str]:
    arrays: List[str] = []
    for name in sorted(group.array_keys()):
        arrays.append(f"{prefix}/{name}" if prefix else name)
    for name in sorted(group.group_keys()):
        child = group[name]
        child_prefix = f"{prefix}/{name}" if prefix else name
        arrays.extend(iter_zarr_arrays(child, child_prefix))
    return arrays


def contains_any(text: str, tokens: Sequence[str]) -> bool:
    lower = text.lower()
    return any(token in lower for token in tokens)


def explicit_object_hits(keys: Sequence[str], tokens: Sequence[str] = OBJECT_TOKENS) -> List[str]:
    return [key for key in keys if contains_any(key, tokens)]


def hits(keys: Sequence[str], tokens: Sequence[str]) -> List[str]:
    return [key for key in keys if contains_any(key, tokens)]


def summarize_hdf5(paths: Sequence[str], max_files: int) -> Dict[str, object]:
    files = iter_hdf5_files(paths, max_files=max_files)
    all_keys: Set[str] = set()
    per_file: Dict[str, List[str]] = {}
    for path in files:
        keys = hdf5_keys(path)
        per_file[os.path.abspath(path)] = keys
        all_keys.update(keys)

    keys = sorted(all_keys)
    object_keys = explicit_object_hits(keys)
    object_pose_keys = [
        key for key in object_keys
        if contains_any(key, ("pose", "pos", "position", "transform"))
    ]
    object_keypoint_keys = [
        key for key in object_keys
        if contains_any(key, KEYPOINT_TOKENS)
    ]
    object_mask_keys = [
        key for key in object_keys
        if contains_any(key, MASK_TOKENS + SEGMENTATION_TOKENS)
    ]
    object_id_keys = [
        key for key in object_keys
        if contains_any(key, ("id", "name", "label"))
    ]
    hand_mask_keys = hits(keys, HAND_MASK_TOKENS)

    return {
        "files": list(per_file.keys()),
        "file_count": len(per_file),
        "dataset_count": len(keys),
        "all_keys": keys,
        "object_keys": object_keys,
        "object_pose_keys": object_pose_keys,
        "object_keypoint_keys": object_keypoint_keys,
        "object_mask_keys": object_mask_keys,
        "object_id_or_name_keys": object_id_keys,
        "hand_or_robot_mask_keys": hand_mask_keys,
        "segmentation_keys": hits(keys, SEGMENTATION_TOKENS),
        "mask_keys": hits(keys, MASK_TOKENS),
        "depth_keys": hits(keys, DEPTH_TOKENS),
        "contact_keys": hits(keys, CONTACT_TOKENS),
        "bbox_keys": hits(keys, BBOX_TOKENS),
        "camera_intrinsic_keys": hits(keys, CAM_INTRINSIC_TOKENS),
        "camera_extrinsic_keys": hits(keys, CAM_EXTRINSIC_TOKENS),
        "eef_pose_keys": hits(keys, EEF_TOKENS),
        "gripper_keys": hits(keys, GRIPPER_TOKENS),
    }


def summarize_zarr(path: str | None) -> Dict[str, object]:
    if path is None:
        return {"path": None, "array_keys": []}
    root = zarr.open(path, mode="r")
    array_keys = sorted(iter_zarr_arrays(root))
    return {
        "path": os.path.abspath(path),
        "array_keys": array_keys,
        "mask_arrays": [key for key in array_keys if any(key == name or key.endswith("/" + name) for name in ZARR_MASK_ARRAYS)],
        "interaction_arrays": [
            key for key in array_keys
            if any(key == name or key.endswith("/" + name) for name in ZARR_INTERACTION_ARRAYS)
        ],
        "object_like_arrays": explicit_object_hits(array_keys),
        "eef_like_arrays": hits(array_keys, EEF_TOKENS),
        "contact_like_arrays": hits(array_keys, CONTACT_TOKENS),
    }


def availability_from_summaries(hdf5_summary: Dict[str, object], zarr_summary: Dict[str, object]) -> Dict[str, object]:
    object_masks = bool(hdf5_summary.get("object_mask_keys")) or any(
        key.endswith("/pi3_object_mask") or key.endswith("/pi3_object_hand_mask")
        for key in zarr_summary.get("mask_arrays", [])
    )
    hand_masks = bool(hdf5_summary.get("hand_or_robot_mask_keys")) or any(
        key.endswith("/pi3_left_hand_mask") or key.endswith("/pi3_right_hand_mask") or key.endswith("/pi3_object_hand_mask")
        for key in zarr_summary.get("mask_arrays", [])
    )
    segmentation = bool(hdf5_summary.get("segmentation_keys"))
    object_pose = bool(hdf5_summary.get("object_pose_keys"))
    object_keypoints = bool(hdf5_summary.get("object_keypoint_keys"))
    camera_intrinsics = bool(hdf5_summary.get("camera_intrinsic_keys"))
    camera_extrinsics = bool(hdf5_summary.get("camera_extrinsic_keys"))
    depth = bool(hdf5_summary.get("depth_keys"))
    eef = bool(hdf5_summary.get("eef_pose_keys")) or bool(zarr_summary.get("eef_like_arrays"))
    grippers = bool(hdf5_summary.get("gripper_keys"))
    contacts = bool(hdf5_summary.get("contact_keys")) or bool(zarr_summary.get("contact_like_arrays"))
    zarr_interaction = bool(zarr_summary.get("interaction_arrays"))

    can_build_masks = bool(
        zarr_summary.get("mask_arrays")
        or (object_masks and (hand_masks or segmentation))
    )
    can_build_interaction = bool(zarr_interaction or (eef and (object_pose or object_keypoints)))

    missing_for_masks = []
    if not object_masks and not segmentation:
        missing_for_masks.append("object masks or segmentation labels")
    if not hand_masks and not segmentation:
        missing_for_masks.append("hand/robot masks or segmentation labels")
    if not zarr_summary.get("mask_arrays") and not can_build_masks:
        missing_for_masks.append("precomputed Pi3 token masks")

    missing_for_interaction = []
    if not eef:
        missing_for_interaction.append("left/right EEF pose or position")
    if not (object_pose or object_keypoints):
        missing_for_interaction.append("object pose or object keypoints")

    return {
        "object_masks_available": object_masks,
        "robot_hand_or_arm_masks_available": hand_masks,
        "segmentation_labels_available": segmentation,
        "object_ids_or_names_available": bool(hdf5_summary.get("object_id_or_name_keys")),
        "object_poses_available": object_pose,
        "object_keypoints_available": object_keypoints,
        "camera_intrinsics_available": camera_intrinsics,
        "camera_extrinsics_available": camera_extrinsics,
        "depth_maps_available": depth,
        "left_right_eef_poses_available": eef,
        "gripper_states_available": grippers,
        "contact_available": contacts,
        "precomputed_zarr_masks_available": bool(zarr_summary.get("mask_arrays")),
        "precomputed_interaction_state_available": zarr_interaction,
        "possible_to_build_object_hand_pi3_token_masks": can_build_masks,
        "possible_to_build_future_interaction_state_targets": can_build_interaction,
        "missing_for_object_hand_pi3_token_masks": sorted(set(missing_for_masks)),
        "missing_for_future_interaction_state_targets": sorted(set(missing_for_interaction)),
    }


def print_summary(summary: Mapping[str, object]) -> None:
    availability = summary["availability"]
    print("Object-hand supervision availability:")
    for key, value in availability.items():
        print(f"  {key}: {value}")

    hdf5_summary = summary["hdf5"]
    zarr_summary = summary["zarr"]
    print("\nKey evidence:")
    print(f"  HDF5 files inspected: {hdf5_summary.get('file_count', 0)}")
    print(f"  HDF5 object keys: {hdf5_summary.get('object_keys', [])}")
    print(f"  HDF5 EEF keys: {hdf5_summary.get('eef_pose_keys', [])}")
    print(f"  HDF5 segmentation keys: {hdf5_summary.get('segmentation_keys', [])}")
    print(f"  HDF5 mask keys: {hdf5_summary.get('mask_keys', [])}")
    print(f"  Zarr path: {zarr_summary.get('path')}")
    print(f"  Zarr mask arrays: {zarr_summary.get('mask_arrays', [])}")
    print(f"  Zarr interaction arrays: {zarr_summary.get('interaction_arrays', [])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether HDF5/zarr data supports object-hand supervision.")
    parser.add_argument("--path", action="append", default=[], help="Raw HDF5 file or directory")
    parser.add_argument("--zarr", default=None, help="Optional GAP zarr dataset")
    parser.add_argument("--max-files", type=int, default=5, help="Max HDF5 files to inspect")
    parser.add_argument(
        "--output",
        default="reports/object_hand_supervision_availability.json",
        help="JSON summary output path",
    )
    args = parser.parse_args()

    hdf5_summary = summarize_hdf5(args.path, max_files=args.max_files) if args.path else {
        "files": [],
        "file_count": 0,
        "dataset_count": 0,
        "all_keys": [],
    }
    zarr_summary = summarize_zarr(args.zarr)
    availability = availability_from_summaries(hdf5_summary, zarr_summary)
    summary = {
        "hdf5": hdf5_summary,
        "zarr": zarr_summary,
        "availability": availability,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nSaved JSON summary to: {args.output}")


if __name__ == "__main__":
    main()
