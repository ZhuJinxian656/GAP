#!/usr/bin/env bash
set -euo pipefail

# Generated commands for public-50 Place Dual Shoes trajectory-supervision variants.
# The script preprocesses only if the zarr is missing, trains only if a checkpoint is missing,
# and evaluates only after the requested checkpoint exists.

ZARR_PATH=/data1/home/zhu_jinxian/project/GAP/data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr
if [ ! -d "${ZARR_PATH}" ]; then
  bash process_data.sh place_dual_shoes demo_clean 50 1
else
  printf "Zarr already exists: %s\n" "${ZARR_PATH}"
fi

# variant=delta_future
CKPT_PATH=/data1/home/zhu_jinxian/project/GAP/checkpoints/place_dual_shoes_delta_future_seed0_50/200.ckpt
if [ ! -f "${CKPT_PATH}" ]; then
  bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_full use_future_loss=true future_target_mode=pi3_delta use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=delta_future_seed0 exp_name=place_dual_shoes_demo_clean_50_delta_future_seed0 logging.name=place_dual_shoes_demo_clean_50_delta_future_seed0
else
  printf "Checkpoint already exists: %s\n" "${CKPT_PATH}"
fi
if [ -f "${CKPT_PATH}" ]; then
  RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_trajectory_eval100 bash eval.sh place_dual_shoes demo_clean delta_future_seed0 50 200 1 0 100
else
  printf "Skipping eval because checkpoint is still missing: %s\n" "${CKPT_PATH}"
fi

# variant=changed_token_future
CKPT_PATH=/data1/home/zhu_jinxian/project/GAP/checkpoints/place_dual_shoes_changed_token_future_seed0_50/200.ckpt
if [ ! -f "${CKPT_PATH}" ]; then
  bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_full use_future_loss=true future_target_mode=pi3_changed_tokens future.changed_token_percentile=90 future.changed_token_stopgrad_mask=true use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=changed_token_future_seed0 exp_name=place_dual_shoes_demo_clean_50_changed_token_future_seed0 logging.name=place_dual_shoes_demo_clean_50_changed_token_future_seed0
else
  printf "Checkpoint already exists: %s\n" "${CKPT_PATH}"
fi
if [ -f "${CKPT_PATH}" ]; then
  RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_trajectory_eval100 bash eval.sh place_dual_shoes demo_clean changed_token_future_seed0 50 200 1 0 100
else
  printf "Skipping eval because checkpoint is still missing: %s\n" "${CKPT_PATH}"
fi

# variant=delta_changed_token_future
CKPT_PATH=/data1/home/zhu_jinxian/project/GAP/checkpoints/place_dual_shoes_delta_changed_token_future_seed0_50/200.ckpt
if [ ! -f "${CKPT_PATH}" ]; then
  bash train.sh place_dual_shoes demo_clean 50 0 1 32 200 100 latent_mode=pi3_full use_future_loss=true future_target_mode=pi3_delta_changed_tokens future.changed_token_percentile=90 future.changed_token_stopgrad_mask=true use_pi3_features=true policy.use_pi3_features=true checkpoint_tag=delta_changed_token_future_seed0 exp_name=place_dual_shoes_demo_clean_50_delta_changed_token_future_seed0 logging.name=place_dual_shoes_demo_clean_50_delta_changed_token_future_seed0
else
  printf "Checkpoint already exists: %s\n" "${CKPT_PATH}"
fi
if [ -f "${CKPT_PATH}" ]; then
  RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_trajectory_eval100 bash eval.sh place_dual_shoes demo_clean delta_changed_token_future_seed0 50 200 1 0 100
else
  printf "Skipping eval because checkpoint is still missing: %s\n" "${CKPT_PATH}"
fi
