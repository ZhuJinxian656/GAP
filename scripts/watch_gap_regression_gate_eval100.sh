#!/usr/bin/env bash
set -euo pipefail

# Watch the parallel regression-gate tmux jobs and backfill any missing eval100
# result after the corresponding job exits and checkpoint exists.

ROOT_DIR="${ROOT_DIR:-/data1/home/zhu_jinxian/project/GAP}"
BWM_BIN="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
TASK_NAME="${TASK_NAME:-place_dual_shoes}"
SETTING="${SETTING:-demo_clean}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
TEST_NUM="${TEST_NUM:-100}"
RESULTS_ROOT="${RESULTS_ROOT:-results_flow_interaction_regression_gate}"
POLL_SECONDS="${POLL_SECONDS:-300}"

cd "${ROOT_DIR}"
export PATH="${BWM_BIN}:${PATH}"

mkdir -p "${RESULTS_ROOT}" logs reports
RESULTS_ROOT="$(cd "${RESULTS_ROOT}" && pwd)"

printf 'Flow/action-UV eval100 watchdog\n'
printf '  root=%s\n' "${ROOT_DIR}"
printf '  task=%s setting=%s demos=%s seed=%s epochs=%s test_num=%s\n' \
  "${TASK_NAME}" "${SETTING}" "${EXPERT_DATA_NUM}" "${SEED}" "${EPOCHS}" "${TEST_NUM}"
printf '  poll_seconds=%s\n' "${POLL_SECONDS}"

old_dino_ckpt="checkpoints/${TASK_NAME}_${SETTING}_dino_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"

targets=(
  "re_eval_old_dino_auto|gap_gate_old_auto|${old_dino_ckpt}|${SETTING}_dino_only_seed${SEED}|auto|0"
  "re_eval_old_dino_model|gap_gate_old_model|${old_dino_ckpt}|${SETTING}_dino_only_seed${SEED}|model|1"
  "train_repro_dino_auto|gap_gate_repro_dino|checkpoints/${TASK_NAME}_${SETTING}_dino_only_repro_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt|${SETTING}_dino_only_repro_seed${SEED}|auto|2"
  "fair_diffusion_current_eef_auto|gap_gate_fair_diff_ceef|checkpoints/${TASK_NAME}_${SETTING}_fair_diffusion_current_eef_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt|${SETTING}_fair_diffusion_current_eef_seed${SEED}|auto|3"
  "fair_diffusion_action_uv_auto|gap_gate_fair_diff_auv|checkpoints/${TASK_NAME}_${SETTING}_fair_diffusion_action_uv_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt|${SETTING}_fair_diffusion_action_uv_seed${SEED}|auto|4"
  "fair_flow_dino_only_auto|gap_gate_fair_flow_dino|checkpoints/${TASK_NAME}_${SETTING}_fair_flow_dino_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt|${SETTING}_fair_flow_dino_only_seed${SEED}|auto|5"
  "fair_flow_action_uv_auto|gap_gate_fair_flow_auv|checkpoints/${TASK_NAME}_${SETTING}_fair_flow_action_uv_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt|${SETTING}_fair_flow_action_uv_seed${SEED}|auto|6"
)

session_alive() {
  tmux has-session -t "$1" 2>/dev/null
}

result_exists() {
  local eval_name="$1"
  local found
  found="$(find "${RESULTS_ROOT}/${eval_name}" -name _result.txt -print -quit 2>/dev/null || true)"
  [ -n "${found}" ]
}

run_eval() {
  local eval_name="$1"
  local ckpt_path="$2"
  local ckpt_setting="$3"
  local model_weight="$4"
  local gpu_id="$5"

  printf '\n[watchdog] Backfill eval %s on GPU%s model_weight=%s\n' \
    "${eval_name}" "${gpu_id}" "${model_weight}"
  env \
    RESULTS_ROOT="${RESULTS_ROOT}/${eval_name}" \
    CKPT_PATH="${ckpt_path}" \
    MODEL_WEIGHT="${model_weight}" \
    TEST_NUM="${TEST_NUM}" \
    bash eval.sh \
      "${TASK_NAME}" \
      "${SETTING}" \
      "${ckpt_setting}" \
      "${EXPERT_DATA_NUM}" \
      "${EPOCHS}" \
      "${gpu_id}" \
      "${SEED}" \
      "${TEST_NUM}"
}

while true; do
  alive=0
  for target in "${targets[@]}"; do
    IFS='|' read -r eval_name session_name ckpt_path ckpt_setting model_weight gpu_id <<< "${target}"
    if session_alive "${session_name}"; then
      alive=$((alive + 1))
    fi
  done

  timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
  printf '[watchdog] %s active target sessions: %s\n' "${timestamp}" "${alive}"

  if [ "${alive}" -eq 0 ]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

missing=0
for target in "${targets[@]}"; do
  IFS='|' read -r eval_name session_name ckpt_path ckpt_setting model_weight gpu_id <<< "${target}"

  if result_exists "${eval_name}"; then
    printf '[watchdog] Result exists: %s\n' "${eval_name}"
    continue
  fi

  if [ ! -f "${ckpt_path}" ]; then
    printf '[watchdog] Missing checkpoint for %s: %s\n' "${eval_name}" "${ckpt_path}" >&2
    missing=$((missing + 1))
    continue
  fi

  run_eval "${eval_name}" "${ckpt_path}" "${ckpt_setting}" "${model_weight}" "${gpu_id}"
done

printf '\n[watchdog] Refreshing regression-gate report\n'
"${BWM_BIN}/python" scripts/summarize_flow_interaction_regression_gate.py

if [ "${missing}" -ne 0 ]; then
  printf '[watchdog] Done with %s missing checkpoint(s). See log for details.\n' "${missing}" >&2
  exit 1
fi

printf '[watchdog] Done. All available eval100 results are present or backfilled.\n'
