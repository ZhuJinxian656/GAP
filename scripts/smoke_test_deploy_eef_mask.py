#!/usr/bin/env python3
"""Smoke-test online EEF-region masks used by deploy_policy.py."""
from __future__ import annotations

import os
import sys

import cv2
import h5py
import numpy as np
import torch

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)

from deploy_policy import GAPPolicyWrapper  # noqa: E402


class FakePolicy:
    pi3_grid_height = 17
    pi3_grid_width = 23


def make_wrapper(mode: str) -> GAPPolicyWrapper:
    wrapper = object.__new__(GAPPolicyWrapper)
    wrapper.device = "cpu"
    wrapper.policy_model = FakePolicy()
    wrapper.latent_mode = mode
    wrapper.eef_mask_radius_tokens = 2.5
    wrapper.latent_mask_key = f"{mode}_mask"
    return wrapper


def raw_observation_from_hdf5(path: str, frame: int):
    with h5py.File(path, "r") as f:
        rgb_bytes = f["observation"]["head_camera"]["rgb"][frame]
        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        raw = {
            "observation": {
                "head_camera": {
                    "intrinsic_cv": f["observation"]["head_camera"]["intrinsic_cv"][frame],
                    "extrinsic_cv": f["observation"]["head_camera"]["extrinsic_cv"][frame],
                }
            },
            "endpose": {
                "left_endpose": f["endpose"]["left_endpose"][frame],
                "right_endpose": f["endpose"]["right_endpose"][frame],
            },
        }
    return rgb, raw


def main() -> None:
    hdf5_path = (
        "/data1/home/zhu_jinxian/project/robotwin/data/place_dual_shoes/"
        "demo_clean/data/episode0.hdf5"
    )
    eef = make_wrapper("pi3_eef_region")
    non_eef = make_wrapper("pi3_non_eef_region")

    rgb, raw = raw_observation_from_hdf5(hdf5_path, frame=120)
    eef_mask = eef._eef_region_mask(rgb, raw, complement=False)
    non_eef_mask = non_eef._eef_region_mask(rgb, raw, complement=True)

    assert tuple(eef_mask.shape) == (1, 1, 391)
    assert tuple(non_eef_mask.shape) == (1, 1, 391)
    assert eef_mask.dtype == torch.float32
    assert non_eef_mask.dtype == torch.float32
    assert torch.allclose(eef_mask + non_eef_mask, torch.ones_like(eef_mask))
    assert 0 < float(eef_mask.sum()) < 391

    obs_dict = {}
    eef.add_online_latent_masks(obs_dict, rgb, raw)
    assert "pi3_eef_region_mask" in obs_dict
    print("deploy EEF mask smoke test passed")


if __name__ == "__main__":
    main()
