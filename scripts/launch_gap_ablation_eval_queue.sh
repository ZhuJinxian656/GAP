#!/usr/bin/env bash
set -euo pipefail

# Evaluate each GAP ablation as soon as its 200-epoch checkpoint appears.
# Evaluation is kept on a separate GPU to avoid perturbing the three-GPU
# training queue.

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

wait_for_checkpoint() {
  local variant="$1"
  local ckpt
  ckpt="$(checkpoint_path_for "$variant")"

  while [ ! -f "$ckpt" ]; do
    printf '[%s] Waiting for checkpoint %s: %s\n' "$(date '+%F %T')" "$variant" "$ckpt"
    sleep "$POLL_SECONDS"
  done
}

refresh_summary() {
  python scripts/summarize_gap_ablation_results.py \
    --root "$ROOT_DIR" \
    --output reports/gap_ablation_result_summary.md
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

for variant in "${VARIANTS[@]}"; do
  wait_for_checkpoint "$variant"
  run_eval "$variant"
  refresh_summary
done

printf '[%s] Eval queue completed.\n' "$(date '+%F %T')"
