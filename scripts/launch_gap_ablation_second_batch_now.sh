#!/usr/bin/env bash
set -euo pipefail

# Launch the second GAP interface-ablation batch immediately on idle GPUs.
# This is intentionally idempotent: existing 200-epoch checkpoints or tmux
# sessions are left alone.

ROOT_DIR="/data1/home/zhu_jinxian/project/GAP"
BWM_BIN="/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin"

cd "$ROOT_DIR"
mkdir -p logs

launch_train() {
  local session="$1"
  local gpu="$2"
  local variant="$3"
  shift 3

  local log_file="logs/${session}_gpu${gpu}.log"
  local checkpoint_dir="checkpoints/place_dual_shoes_demo_clean_${variant}_seed0_50"

  if [ -f "${checkpoint_dir}/200.ckpt" ]; then
    printf '[%s] Skip %s: %s/200.ckpt already exists\n' "$(date '+%F %T')" "$session" "$checkpoint_dir"
    return
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    printf '[%s] Skip %s: tmux session already exists\n' "$(date '+%F %T')" "$session"
    return
  fi

  printf '[%s] Launching %s on GPU %s -> %s\n' "$(date '+%F %T')" "$session" "$gpu" "$log_file"
  tmux new-session -d -s "$session" \
    "cd '$ROOT_DIR' && env PATH='$BWM_BIN':\$PATH WANDB_MODE=disabled bash train.sh place_dual_shoes demo_clean 50 0 '$gpu' 32 200 100 $* > '$log_file' 2>&1"
}

launch_train gap_ablate_pi3_compressed 5 pi3_compressed \
  latent_mode=pi3_compressed \
  use_future_loss=true \
  future_target_mode=pi3_compressed \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_compressed_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0

launch_train gap_ablate_pi3_random 6 pi3_random \
  latent_mode=pi3_random_tokens \
  use_future_loss=true \
  future_target_mode=pi3_random_tokens \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_random_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_random_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_random_seed0

launch_train gap_ablate_pi3_dropout 7 pi3_dropout \
  latent_mode=pi3_token_dropout \
  use_future_loss=true \
  future_target_mode=pi3_full \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_dropout_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0
