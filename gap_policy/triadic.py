"""
Utilities for optional object-centric triadic relation features.

The minimal relation interface is position based:
left end-effector <-> object, right end-effector <-> object, and optionally
left end-effector <-> right end-effector. Object pose is never fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import warnings

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for data inspection.
    torch = None


TRIADIC_MODES = (
    "disabled",
    "pairwise",
    "triadic",
    "proprio_only_fallback",
)


DEFAULT_LEFT_EEF_KEYS = (
    "/endpose/left_endpose",
    "/observation/left_eef_pose",
    "/observation/left_eef_pos",
    "/observation/left_ee_pose",
    "/observation/left_ee_pos",
    "/observation/left_end_effector_pose",
    "/observation/left_end_effector_position",
    "/observation/left_tcp_pose",
    "/observation/left_tcp_position",
    "/left_eef_pose",
    "/left_eef_pos",
)

DEFAULT_RIGHT_EEF_KEYS = (
    "/endpose/right_endpose",
    "/observation/right_eef_pose",
    "/observation/right_eef_pos",
    "/observation/right_ee_pose",
    "/observation/right_ee_pos",
    "/observation/right_end_effector_pose",
    "/observation/right_end_effector_position",
    "/observation/right_tcp_pose",
    "/observation/right_tcp_position",
    "/right_eef_pose",
    "/right_eef_pos",
)

DEFAULT_OBJECT_KEYS = (
    "/observation/object_pose",
    "/observation/object_pos",
    "/observation/object_position",
    "/observation/object_keypoints",
    "/observation/object_points",
    "/observation/target_object_pose",
    "/observation/target_object_pos",
    "/object_pose",
    "/object_pos",
    "/object_position",
    "/object_keypoints",
)

DEFAULT_LEFT_GRIPPER_KEYS = (
    "/endpose/left_gripper",
    "/joint_action/left_gripper",
    "/observation/left_gripper",
    "/observation/left_gripper_state",
    "/left_gripper",
    "/left_gripper_state",
)

DEFAULT_RIGHT_GRIPPER_KEYS = (
    "/endpose/right_gripper",
    "/joint_action/right_gripper",
    "/observation/right_gripper",
    "/observation/right_gripper_state",
    "/right_gripper",
    "/right_gripper_state",
)

DEFAULT_PROPRIO_KEYS = (
    "/joint_action/vector",
    "/observation/joint_action/vector",
    "/observation/agent_pos",
    "/agent_pos",
    "/state",
)


@dataclass
class TriadicConfig:
    mode: str = "disabled"
    include_grippers: bool = True
    left_eef_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_LEFT_EEF_KEYS)
    right_eef_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_RIGHT_EEF_KEYS)
    object_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_OBJECT_KEYS)
    left_gripper_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_LEFT_GRIPPER_KEYS)
    right_gripper_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_RIGHT_GRIPPER_KEYS)
    proprio_keys: Sequence[str] = field(default_factory=lambda: DEFAULT_PROPRIO_KEYS)
    proprio_left_slice: Tuple[int, int] = (0, 3)
    proprio_right_slice: Tuple[int, int] = (7, 10)
    proprio_left_gripper_index: int = 6
    proprio_right_gripper_index: int = 13


def make_triadic_config(config: Optional[Any] = None, **overrides: Any) -> TriadicConfig:
    """Build a TriadicConfig from a dataclass, dict/OmegaConf-like object, or kwargs."""
    if isinstance(config, TriadicConfig):
        data = config.__dict__.copy()
    elif config is None:
        data = {}
    elif isinstance(config, Mapping):
        data = dict(config)
    else:
        keys = TriadicConfig().__dict__.keys()
        data = {key: getattr(config, key) for key in keys if hasattr(config, key)}

    data.update({key: value for key, value in overrides.items() if value is not None})

    if "triadic_mode" in data and "mode" not in data:
        data["mode"] = data.pop("triadic_mode")
    if "triadic_include_grippers" in data and "include_grippers" not in data:
        data["include_grippers"] = data.pop("triadic_include_grippers")

    cfg = TriadicConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            if key.endswith("_slice") and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)

    if cfg.mode not in TRIADIC_MODES:
        raise ValueError(f"Unsupported triadic mode {cfg.mode!r}. Expected one of {TRIADIC_MODES}.")
    return cfg


def infer_triadic_dim(mode: str, include_grippers: bool = True) -> int:
    """Return the expected feature dimension for a triadic mode."""
    if mode == "disabled":
        return 0
    if mode == "pairwise":
        dim = 6
    elif mode == "triadic":
        dim = 9
    elif mode == "proprio_only_fallback":
        dim = 3
    else:
        raise ValueError(f"Unsupported triadic mode {mode!r}")
    if include_grippers:
        dim += 2
    return dim


def flatten_nested_mapping(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested observation dictionaries into slash-delimited keys."""
    flat: Dict[str, Any] = {}
    for key, value in data.items():
        str_key = str(key)
        path = f"{prefix}/{str_key}" if prefix else str_key
        if isinstance(value, Mapping):
            flat.update(flatten_nested_mapping(value, path))
        else:
            flat[path] = value
            flat[f"/{path}"] = value
    return flat


def build_triadic_state(
    obs_dict: Mapping[str, Any],
    config: Optional[Any] = None,
    warn_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """
    Build a minimal relation state from an observation mapping.

    Returns None when mode is disabled or required object-centric fields are
    unavailable. In proprio_only_fallback mode, the left/right relation is
    constructed from configured proprio slices if explicit EEF positions are
    missing.
    """
    cfg = make_triadic_config(config)
    if cfg.mode == "disabled":
        return None

    mapping = flatten_nested_mapping(obs_dict)
    left_pos = _extract_position(mapping, cfg.left_eef_keys)
    right_pos = _extract_position(mapping, cfg.right_eef_keys)
    object_pos = _extract_position(mapping, cfg.object_keys)

    if (left_pos is None or right_pos is None) and cfg.mode == "proprio_only_fallback":
        proprio = _extract_value(mapping, cfg.proprio_keys)
        if proprio is None and "agent_pos" in mapping:
            proprio = mapping["agent_pos"]
        if proprio is None:
            _warn("proprio_only_fallback requested, but no proprioception vector was found.", warn_fn)
            return None
        left_pos, right_pos = _positions_from_proprio(proprio, cfg)
        _warn(
            "Using proprio_only_fallback: left-right relation comes from configured proprio slices, "
            "not object-centric end-effector poses.",
            warn_fn,
        )

    if left_pos is None or right_pos is None:
        _warn("Triadic state unavailable: left or right end-effector position key is missing.", warn_fn)
        return None

    features = []
    if cfg.mode in ("pairwise", "triadic"):
        if object_pos is None:
            _warn(
                "Triadic state unavailable: object pose/position/keypoint key is missing. "
                "Object pose is not fabricated.",
                warn_fn,
            )
            return None
        features.extend([left_pos - object_pos, right_pos - object_pos])
        if cfg.mode == "triadic":
            features.append(left_pos - right_pos)
    elif cfg.mode == "proprio_only_fallback":
        features.append(left_pos - right_pos)

    if cfg.include_grippers:
        features.extend(_gripper_features(mapping, cfg, left_pos))

    return _concat(features)


def build_triadic_state_from_hdf5(
    root: Any,
    config: Optional[Any] = None,
    warn_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Any]:
    """Collect configured candidate arrays from an open h5py.File/Group."""
    cfg = make_triadic_config(config)
    if cfg.mode == "disabled":
        return None

    keys = []
    for candidates in (
        cfg.left_eef_keys,
        cfg.right_eef_keys,
        cfg.object_keys,
        cfg.left_gripper_keys,
        cfg.right_gripper_keys,
        cfg.proprio_keys,
    ):
        keys.extend(candidates)

    arrays: Dict[str, Any] = {}
    for key in keys:
        dataset = _find_hdf5_dataset(root, key)
        if dataset is not None:
            arrays[key] = dataset[()]
            arrays[_strip_slashes(key)] = arrays[key]
    return build_triadic_state(arrays, cfg, warn_fn=warn_fn)


def _positions_from_proprio(proprio: Any, cfg: TriadicConfig) -> Tuple[Any, Any]:
    proprio_arr = _as_array(proprio)
    left = proprio_arr[..., cfg.proprio_left_slice[0]:cfg.proprio_left_slice[1]]
    right = proprio_arr[..., cfg.proprio_right_slice[0]:cfg.proprio_right_slice[1]]
    if left.shape[-1] < 3 or right.shape[-1] < 3:
        raise ValueError(
            "Configured proprio slices must expose at least 3 values for each arm."
        )
    return left[..., :3], right[..., :3]


def _gripper_features(mapping: Mapping[str, Any], cfg: TriadicConfig, reference: Any) -> Sequence[Any]:
    left = _extract_scalar(mapping, cfg.left_gripper_keys)
    right = _extract_scalar(mapping, cfg.right_gripper_keys)

    proprio = _extract_value(mapping, cfg.proprio_keys)
    if proprio is None and "agent_pos" in mapping:
        proprio = mapping["agent_pos"]

    if left is None and proprio is not None:
        left = _scalar_from_proprio(proprio, cfg.proprio_left_gripper_index)
    if right is None and proprio is not None:
        right = _scalar_from_proprio(proprio, cfg.proprio_right_gripper_index)

    if left is None:
        left = _zeros_like_scalar(reference)
    if right is None:
        right = _zeros_like_scalar(reference)
    return left, right


def _extract_position(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    value = _extract_value(mapping, candidates)
    if value is None:
        return None
    arr = _as_array(value)
    if arr.shape[-1] < 3:
        return None
    if len(arr.shape) >= 3 and arr.shape[-1] >= 3:
        # Object keypoints or point sets: use centroid as the first minimal interface.
        token = _last_token(_matched_key(mapping, candidates) or "")
        if "keypoint" in token or "points" in token or "pointcloud" in token:
            arr = arr[..., :3].mean(axis=-2) if not _is_torch(arr) else arr[..., :3].mean(dim=-2)
            return arr
    return arr[..., :3]


def _extract_scalar(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    value = _extract_value(mapping, candidates)
    if value is None:
        return None
    arr = _as_array(value)
    if len(arr.shape) == 1:
        return arr[..., None]
    if arr.shape[-1:] == (1,):
        return arr
    return arr[..., :1]


def _extract_value(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    key = _matched_key(mapping, candidates)
    return None if key is None else mapping[key]


def _matched_key(mapping: Mapping[str, Any], candidates: Sequence[str]) -> Optional[str]:
    normalized_to_key = {_normalize_key(key): key for key in mapping.keys()}
    for candidate in candidates:
        norm = _normalize_key(candidate)
        if norm in normalized_to_key:
            return normalized_to_key[norm]
        suffix = "/" + norm
        for mapped_norm, original_key in normalized_to_key.items():
            if mapped_norm.endswith(suffix):
                return original_key
    return None


def _find_hdf5_dataset(root: Any, candidate: str) -> Optional[Any]:
    clean = _strip_slashes(candidate)
    for key in (candidate, clean, f"/{clean}"):
        try:
            if key in root:
                obj = root[key]
                if hasattr(obj, "shape"):
                    return obj
        except Exception:
            pass
    found = None

    def visitor(name: str, obj: Any) -> None:
        nonlocal found
        if found is not None or not hasattr(obj, "shape"):
            return
        if _normalize_key(name).endswith("/" + _normalize_key(clean)) or _normalize_key(name) == _normalize_key(clean):
            found = obj

    root.visititems(visitor)
    return found


def _scalar_from_proprio(proprio: Any, index: int) -> Optional[Any]:
    arr = _as_array(proprio)
    if arr.shape[-1] <= index:
        return None
    return arr[..., index:index + 1]


def _zeros_like_scalar(reference: Any) -> Any:
    shape = reference.shape[:-1] + (1,)
    if _is_torch(reference):
        return torch.zeros(shape, dtype=reference.dtype, device=reference.device)
    return np.zeros(shape, dtype=np.asarray(reference).dtype)


def _concat(features: Sequence[Any]) -> Any:
    if any(_is_torch(item) for item in features):
        tensors = [item if _is_torch(item) else torch.as_tensor(item) for item in features]
        return torch.cat(tensors, dim=-1)
    return np.concatenate([np.asarray(item) for item in features], axis=-1).astype(np.float32)


def _as_array(value: Any) -> Any:
    if _is_torch(value):
        return value
    return np.asarray(value, dtype=np.float32)


def _is_torch(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _normalize_key(key: str) -> str:
    return _strip_slashes(key).lower()


def _strip_slashes(key: str) -> str:
    return str(key).strip("/")


def _last_token(key: str) -> str:
    return _strip_slashes(key).split("/")[-1].lower()


def _warn(message: str, warn_fn: Optional[Callable[[str], None]]) -> None:
    if warn_fn is not None:
        warn_fn(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
