#!/usr/bin/env bash
set -euo pipefail

# Generated eval100 commands for public-50 Place Dual Shoes checkpoints.
# Review GPU availability before running; commands write outside results_eval50_full.

# variant=vanilla note=original GAP public-50 baseline
RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_eval100 bash eval.sh place_dual_shoes demo_clean demo_clean 50 200 1 0 100

# variant=dino_only note=DINO/state only, no Pi3 future loss
RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_eval100 bash eval.sh place_dual_shoes demo_clean demo_clean_dino_only_seed0 50 200 1 0 100

# variant=no_future note=Pi3 observations without future loss
RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_eval100 bash eval.sh place_dual_shoes demo_clean demo_clean_no_future_seed0 50 200 1 0 100

# variant=pi3_pooled note=pooled Pi3 observation and future target
RESULTS_ROOT=/data1/home/zhu_jinxian/project/GAP/results_dual_shoes_50demo_eval100 bash eval.sh place_dual_shoes demo_clean demo_clean_pi3_pooled_seed0 50 200 1 0 100
