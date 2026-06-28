#!/usr/bin/env bash
set -euo pipefail

# Non-overwriting higher-sample eval for the GAP geometry ablations.
# Results go to results_eval50/ by default, leaving the quick 10-rollout
# results/ table intact.

ROOT_DIR="${ROOT_DIR:-/data1/home/zhu_jinxian/project/GAP}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/data1/home/zhu_jinxian/project/robotwin}"
BWM_BIN="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_eval50}"
TEST_NUM="${TEST_NUM:-50}"
SEED_LIST="${SEED_LIST:-0}"
CHECKPOINT_NUM="${CHECKPOINT_NUM:-200}"
GPU_LIST="${GPU_LIST:-1 2 3 4 5 6 7}"
VARIANTS="${VARIANTS:-vanilla dino_only no_future pi3_pooled pi3_compressed pi3_random pi3_dropout pi3_eef_region pi3_non_eef_region}"

cd "$ROOT_DIR"
mkdir -p logs "$RESULTS_ROOT"

timestamp="$(date '+%F_%H%M%S')"
master_log="logs/gap_key_eval50_${timestamp}.log"
exec > >(tee -a "$master_log") 2>&1

printf '[%s] Key eval50 started. results_root=%s test_num=%s checkpoint=%s\n' \
  "$(date '+%F %T')" "$RESULTS_ROOT" "$TEST_NUM" "$CHECKPOINT_NUM"
printf '[%s] Variants: %s\n' "$(date '+%F %T')" "$VARIANTS"
printf '[%s] GPUs: %s\n' "$(date '+%F %T')" "$GPU_LIST"

ckpt_setting_for() {
  local variant="$1"
  if [ "$variant" = "vanilla" ]; then
    printf 'demo_clean'
  else
    printf 'demo_clean_%s_seed0' "$variant"
  fi
}

checkpoint_path_for() {
  local variant="$1"
  if [ "$variant" = "vanilla" ]; then
    printf 'checkpoints/place_dual_shoes_demo_clean_50/%s.ckpt' "$CHECKPOINT_NUM"
  else
    printf 'checkpoints/place_dual_shoes_demo_clean_%s_seed0_50/%s.ckpt' "$variant" "$CHECKPOINT_NUM"
  fi
}

result_path_for() {
  local variant="$1"
  local ckpt_setting
  ckpt_setting="$(ckpt_setting_for "$variant")"
  printf '%s/place_dual_shoes/GAP/demo_clean/%s/seed_0/%s/_result.txt' \
    "$RESULTS_ROOT" "$ckpt_setting" "$CHECKPOINT_NUM"
}

read -r -a gpu_array <<< "$GPU_LIST"
read -r -a variant_array <<< "$VARIANTS"

if [ "${#gpu_array[@]}" -eq 0 ]; then
  printf 'GPU_LIST must not be empty.\n' >&2
  exit 1
fi

status=0

idx=0
while [ "$idx" -lt "${#variant_array[@]}" ]; do
  pids=()

  for gpu in "${gpu_array[@]}"; do
    if [ "$idx" -ge "${#variant_array[@]}" ]; then
      break
    fi

    variant="${variant_array[$idx]}"
    idx=$((idx + 1))
    ckpt_setting="$(ckpt_setting_for "$variant")"
    ckpt_path="$(checkpoint_path_for "$variant")"
    result_path="$(result_path_for "$variant")"
    log_file="logs/gap_eval50_${variant}_gpu${gpu}.log"

    if [ ! -f "$ckpt_path" ]; then
      printf '[%s] Missing checkpoint for %s: %s\n' "$(date '+%F %T')" "$variant" "$ckpt_path" >&2
      status=1
      continue
    fi

    if [ "${FORCE_EVAL:-0}" != "1" ] && [ -f "$result_path" ]; then
      printf '[%s] Skip %s: result exists at %s\n' "$(date '+%F %T')" "$variant" "$result_path"
      continue
    fi

    printf '[%s] Launch %s on GPU %s -> %s\n' "$(date '+%F %T')" "$variant" "$gpu" "$log_file"
    (
      env PATH="$BWM_BIN:$PATH" \
        ROBOTWIN_ROOT="$ROBOTWIN_ROOT" \
        RESULTS_ROOT="$RESULTS_ROOT" \
        bash eval.sh place_dual_shoes demo_clean "$ckpt_setting" 50 "$CHECKPOINT_NUM" \
          "$gpu" "$SEED_LIST" "$TEST_NUM"
      if [ -f "$result_path" ]; then
        printf '[%s] Result %s: ' "$(date '+%F %T')" "$variant"
        tail -n 1 "$result_path"
      fi
    ) > "$log_file" 2>&1 &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
done

printf '[%s] Key eval50 finished with status=%s\n' "$(date '+%F %T')" "$status"
python scripts/summarize_gap_ablation_results.py \
  --root "$ROOT_DIR" \
  --results-dir "$RESULTS_ROOT" \
  --test-num "$TEST_NUM" \
  --output reports/gap_ablation_eval50_summary.md
exit "$status"
