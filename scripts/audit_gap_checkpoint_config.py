#!/usr/bin/env python
"""Audit GAP checkpoints for recipe, model, and normalizer metadata."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


AUDIT_FIELDS = [
    ("task_name", ("task_name",)),
    ("setting", ("setting",)),
    ("expert_data_num", ("expert_data_num",)),
    ("checkpoint_tag", ("checkpoint_tag",)),
    ("dataloader.batch_size", ("dataloader", "batch_size")),
    ("training.num_epochs", ("training", "num_epochs")),
    ("training.checkpoint_every", ("training", "checkpoint_every")),
    ("training.lr_warmup_steps", ("training", "lr_warmup_steps")),
    ("optimizer.lr", ("optimizer", "lr")),
    ("training.use_ema", ("training", "use_ema")),
    ("policy.generative_mode", ("policy", "generative_mode")),
    ("policy.use_interaction_field", ("policy", "use_interaction_field")),
    ("policy.interaction_field.mode", ("policy", "interaction_field", "mode")),
    ("use_pi3_features", ("use_pi3_features",)),
    ("latent_mode", ("latent_mode",)),
    ("use_future_loss", ("use_future_loss",)),
    ("future_target_mode", ("future_target_mode",)),
    ("policy.use_pi3_features", ("policy", "use_pi3_features")),
    ("policy.latent_mode", ("policy", "latent_mode")),
    ("policy.use_future_loss", ("policy", "use_future_loss")),
    ("policy.future_target_mode", ("policy", "future_target_mode")),
    ("policy.num_inference_steps", ("policy", "num_inference_steps")),
    ("policy.noise_scheduler.prediction_type", ("policy", "noise_scheduler", "prediction_type")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print GAP checkpoint config/recipe metadata and optional zarr update estimates."
    )
    parser.add_argument("--ckpt", nargs="+", required=True, help="One or more GAP checkpoint paths.")
    parser.add_argument("--zarr", default=None, help="Optional zarr path used to estimate optimizer updates.")
    parser.add_argument("--out", default=None, help="Optional output path. .md writes markdown; otherwise JSON.")
    return parser.parse_args()


def to_plain(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def get_nested(mapping: Any, path: tuple[str, ...], default: Any = None) -> Any:
    cur = mapping
    for key in path:
        if isinstance(cur, Mapping):
            if key not in cur:
                return default
            cur = cur[key]
        else:
            if not hasattr(cur, key):
                return default
            cur = getattr(cur, key)
    return to_jsonable(cur)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def extract_normalizer_keys(normalizer_state: Any) -> list[str]:
    if not isinstance(normalizer_state, Mapping):
        return []
    keys = set()
    for key in normalizer_state.keys():
        parts = str(key).split(".")
        if len(parts) >= 2 and parts[0] == "params_dict":
            keys.add(parts[1])
        else:
            keys.add(parts[0])
    return sorted(keys)


def count_params(state_dict: Any) -> int | None:
    if not isinstance(state_dict, Mapping):
        return None
    total = 0
    saw_tensor = False
    for value in state_dict.values():
        if hasattr(value, "numel"):
            total += int(value.numel())
            saw_tensor = True
    return total if saw_tensor else None


def zarr_total_steps(zarr_path: str) -> int | None:
    import zarr

    group = zarr.open(zarr_path, mode="r")
    data = group["data"] if "data" in group else group
    for key in ("action", "state", "agent_pos", "dinov3_features"):
        if key in data:
            return int(data[key].shape[0])
    return None


def audit_checkpoint(path: str, total_steps: int | None) -> dict[str, Any]:
    import torch

    ckpt_path = Path(path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = to_plain(ckpt.get("cfg", {}))
    row: dict[str, Any] = {
        "checkpoint_path": str(ckpt_path),
        "epoch": to_jsonable(ckpt.get("epoch")),
        "global_step": to_jsonable(ckpt.get("global_step")),
    }
    for name, cfg_path in AUDIT_FIELDS:
        row[name] = get_nested(cfg, cfg_path)

    row["contains_ema"] = ckpt.get("ema") is not None
    row["contains_model"] = ckpt.get("model") is not None
    row["model_parameter_count"] = count_params(ckpt.get("model"))
    row["normalizer_keys"] = extract_normalizer_keys(ckpt.get("normalizer"))

    batch_size = row.get("dataloader.batch_size")
    num_epochs = row.get("training.num_epochs")
    if total_steps is not None and batch_size:
        updates_per_epoch = int(math.ceil(float(total_steps) / float(batch_size)))
        row["zarr_total_steps"] = total_steps
        row["estimated_updates_per_epoch"] = updates_per_epoch
        if num_epochs is not None:
            row["estimated_total_optimizer_updates"] = int(updates_per_epoch * int(num_epochs))
    return row


def write_markdown(rows: list[dict[str, Any]], path: str) -> None:
    lines = ["# GAP Checkpoint Config Audit", ""]
    for row in rows:
        lines.append(f"## {row['checkpoint_path']}")
        lines.append("")
        for key, value in row.items():
            lines.append(f"- `{key}`: `{json.dumps(to_jsonable(value), sort_keys=True)}`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def print_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(f"\n== {row['checkpoint_path']} ==")
        for key, value in row.items():
            if key == "checkpoint_path":
                continue
            print(f"{key}: {to_jsonable(value)}")


def main() -> int:
    args = parse_args()
    total_steps = None
    if args.zarr:
        total_steps = zarr_total_steps(args.zarr)
        print(f"zarr_path: {args.zarr}")
        print(f"zarr_total_steps: {total_steps}")

    rows = [audit_checkpoint(path, total_steps) for path in args.ckpt]
    print_rows(rows)

    if args.out:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        if out_path.lower().endswith(".md"):
            write_markdown(rows, out_path)
        else:
            Path(out_path).write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
