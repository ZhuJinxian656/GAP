#!/usr/bin/env bash
set -euo pipefail

# Run a lightweight 100-epoch interim comparison while 200-epoch ablations are
# still training. This is only an early signal; the primary table remains the
# 200-epoch eval queue.

ROOT_DIR="/data1/home/zhu_jinxian/project/GAP"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data1/home/zhu_jinxian/project/robotwin}"
BWM_BIN="/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin"
EVAL_GPU="${EVAL_GPU:-4}"
TEST_NUM="${TEST_NUM:-10}"
SEED_LIST="${SEED_LIST:-0}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-100}"

cd "$ROOT_DIR"
mkdir -p logs
exec >> logs/gap_interim_eval_100.log 2>&1

printf '[%s] Interim 100 eval started. GPU=%s test_num=%s\n' \
  "$(date '+%F %T')" "$EVAL_GPU" "$TEST_NUM"

result_path_for() {
  local ckpt_setting="$1"
  printf 'results/place_dual_shoes/GAP/demo_clean/%s/seed_0/%s/_result.txt' \
    "$ckpt_setting" "$CHECKPOINT_NUM"
}

run_eval() {
  local label="$1"
  local ckpt_setting="$2"
  local result
  local log_file

  result="$(result_path_for "$ckpt_setting")"
  log_file="logs/gap_eval_${label}_${CHECKPOINT_NUM}_gpu${EVAL_GPU}.log"

  if [ "${FORCE_EVAL:-0}" != "1" ] && [ -f "$result" ]; then
    printf '[%s] Skip %s@%s: result exists at %s\n' \
      "$(date '+%F %T')" "$label" "$CHECKPOINT_NUM" "$result"
    return
  fi

  printf '[%s] Eval %s@%s -> %s\n' "$(date '+%F %T')" "$label" "$CHECKPOINT_NUM" "$log_file"
  env PATH="$BWM_BIN:$PATH" \
    ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
    bash eval.sh place_dual_shoes demo_clean "$ckpt_setting" 50 "$CHECKPOINT_NUM" \
      "$EVAL_GPU" "$SEED_LIST" "$TEST_NUM" \
    > "$log_file" 2>&1

  if [ -f "$result" ]; then
    printf '[%s] Result %s@%s: ' "$(date '+%F %T')" "$label" "$CHECKPOINT_NUM"
    tail -n 1 "$result"
  fi
}

run_eval vanilla demo_clean
run_eval dino_only demo_clean_dino_only_seed0

printf '[%s] Interim 100 eval completed.\n' "$(date '+%F %T')"
