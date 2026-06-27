#!/usr/bin/env bash
set -euo pipefail

# Evaluate GAP ablations on a single eval GPU. The queue polls all variants and
# runs whichever checkpoint is ready first, so later variants are not blocked by
# an earlier slow run.

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
  pi3_eef_region
  pi3_non_eef_region
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
    return 0
  fi

  printf '[%s] Eval %s using %s -> %s\n' "$(date '+%F %T')" "$variant" "$ckpt" "$log_file"
  env PATH="$BWM_BIN:$PATH" \
    ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
    bash eval.sh place_dual_shoes demo_clean "$ckpt_setting" 50 200 "$EVAL_GPU" "$SEED_LIST" "$TEST_NUM" \
    > "$log_file" 2>&1
}

declare -A DONE=()
completed=0
total=${#VARIANTS[@]}

while [ "$completed" -lt "$total" ]; do
  progressed=0
  waiting=()

  for variant in "${VARIANTS[@]}"; do
    if [ "${DONE[$variant]:-0}" = "1" ]; then
      continue
    fi

    ckpt="$(checkpoint_path_for "$variant")"
    result="$(result_path_for "$variant")"

    if [ "${FORCE_EVAL:-0}" != "1" ] && [ -f "$result" ]; then
      printf '[%s] Mark done %s: result exists at %s\n' "$(date '+%F %T')" "$variant" "$result"
      DONE[$variant]=1
      completed=$((completed + 1))
      progressed=1
      continue
    fi

    if [ -f "$ckpt" ]; then
      run_eval "$variant"
      refresh_summary
      DONE[$variant]=1
      completed=$((completed + 1))
      progressed=1
    else
      waiting+=("${variant}: ${ckpt}")
    fi
  done

  if [ "$completed" -lt "$total" ] && [ "$progressed" -eq 0 ]; then
    printf '[%s] Waiting for checkpoints (%s/%s complete): %s\n' \
      "$(date '+%F %T')" "$completed" "$total" "${waiting[*]}"
    sleep "$POLL_SECONDS"
  fi
done

printf '[%s] Eval queue completed.\n' "$(date '+%F %T')"
