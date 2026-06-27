#!/usr/bin/env bash
set -euo pipefail

# Wait for the six 200-epoch GAP ablation checkpoints, then evaluate each one
# sequentially. Evaluation is kept on a separate GPU to avoid perturbing the
# three-GPU training queue.

ROOT_DIR="/data1/home/zhu_jinxian/project/GAP"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data1/home/zhu_jinxian/project/robotwin}"
BWM_BIN="/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin"
POLL_SECONDS="${POLL_SECONDS:-300}"
EVAL_GPU="${EVAL_GPU:-4}"
TEST_NUM="${TEST_NUM:-10}"
SEED_LIST="${SEED_LIST:-0}"

cd "$ROOT_DIR"
mkdir -p logs
exec >> logs/gap_eval_ablation_queue.log 2>&1

printf '[%s] Eval queue started. GPU=%s test_num=%s poll=%ss\n' \
  "$(date '+%F %T')" "$EVAL_GPU" "$TEST_NUM" "$POLL_SECONDS"

VARIANTS=(
  dino_only
  no_future
  pi3_pooled
  pi3_compressed
  pi3_random
  pi3_dropout
)

checkpoint_path_for() {
  local variant="$1"
  printf 'checkpoints/place_dual_shoes_demo_clean_%s_seed0_50/200.ckpt' "$variant"
}

ckpt_setting_for() {
  local variant="$1"
  printf 'demo_clean_%s_seed0' "$variant"
}

result_path_for() {
  local variant="$1"
  local ckpt_setting
  ckpt_setting="$(ckpt_setting_for "$variant")"
  printf 'results/place_dual_shoes/GAP/demo_clean/%s/seed_0/200/_result.txt' "$ckpt_setting"
}

wait_for_all_checkpoints() {
  while true; do
    local missing=()
    for variant in "${VARIANTS[@]}"; do
      local ckpt
      ckpt="$(checkpoint_path_for "$variant")"
      if [ ! -f "$ckpt" ]; then
        missing+=("$ckpt")
      fi
    done

    if [ "${#missing[@]}" -eq 0 ]; then
      break
    fi

    printf '[%s] Waiting for checkpoints: %s\n' "$(date '+%F %T')" "${missing[*]}"
    sleep "$POLL_SECONDS"
  done
}

run_eval() {
  local variant="$1"
  local ckpt_setting
  local ckpt
  local result
  local log_file

  ckpt_setting="$(ckpt_setting_for "$variant")"
  ckpt="$(checkpoint_path_for "$variant")"
  result="$(result_path_for "$variant")"
  log_file="logs/gap_eval_${variant}_gpu${EVAL_GPU}.log"

  if [ "${FORCE_EVAL:-0}" != "1" ] && [ -f "$result" ]; then
    printf '[%s] Skip eval %s: result exists at %s\n' "$(date '+%F %T')" "$variant" "$result"
    return
  fi

  printf '[%s] Eval %s using %s -> %s\n' "$(date '+%F %T')" "$variant" "$ckpt" "$log_file"
  env PATH="$BWM_BIN:$PATH" \
    ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
    bash eval.sh place_dual_shoes demo_clean "$ckpt_setting" 50 200 "$EVAL_GPU" "$SEED_LIST" "$TEST_NUM" \
    > "$log_file" 2>&1
}

wait_for_all_checkpoints

for variant in "${VARIANTS[@]}"; do
  run_eval "$variant"
done

printf '[%s] Eval queue completed.\n' "$(date '+%F %T')"
