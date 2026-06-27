#!/usr/bin/env bash
set -euo pipefail

# Queue the second GAP interface-ablation batch after the current three-run
# batch exits. This keeps concurrency at three single-GPU training jobs.

ROOT_DIR="/data1/home/zhu_jinxian/project/GAP"
BWM_BIN="/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "$ROOT_DIR"
mkdir -p logs
exec >> logs/gap_ablation_queue.log 2>&1

printf '[%s] Queue started. Poll interval: %ss\n' "$(date '+%F %T')" "$POLL_SECONDS"

wait_for_sessions_to_exit() {
  local sessions=("$@")
  while true; do
    local active=()
    for session in "${sessions[@]}"; do
      if tmux has-session -t "$session" 2>/dev/null; then
        active+=("$session")
      fi
    done

    if [ "${#active[@]}" -eq 0 ]; then
      break
    fi

    printf '[%s] Waiting for sessions: %s\n' "$(date '+%F %T')" "${active[*]}"
    sleep "$POLL_SECONDS"
  done
}

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

wait_for_sessions_to_exit \
  gap_ablate_dino_only \
  gap_ablate_no_future \
  gap_ablate_pi3_pooled

launch_train gap_ablate_pi3_compressed 1 pi3_compressed \
  latent_mode=pi3_compressed \
  use_future_loss=true \
  future_target_mode=pi3_compressed \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_compressed_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_compressed_seed0

launch_train gap_ablate_pi3_random 2 pi3_random \
  latent_mode=pi3_random_tokens \
  use_future_loss=true \
  future_target_mode=pi3_random_tokens \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_random_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_random_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_random_seed0

launch_train gap_ablate_pi3_dropout 3 pi3_dropout \
  latent_mode=pi3_token_dropout \
  use_future_loss=true \
  future_target_mode=pi3_full \
  use_pi3_features=true \
  policy.use_pi3_features=true \
  checkpoint_tag=demo_clean_pi3_dropout_seed0 \
  exp_name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0 \
  logging.name=place_dual_shoes_demo_clean_50_pi3_dropout_seed0

printf '[%s] Second ablation batch launched.\n' "$(date '+%F %T')"
