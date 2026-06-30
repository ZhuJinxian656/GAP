#!/usr/bin/env bash
set -euo pipefail

# Watch the current fair_flow_current_eef gate job. If its main tmux session
# exits without an eval100 artifact, backfill eval100 from the epoch-200 ckpt.

ROOT_DIR="${ROOT_DIR:-/data1/home/zhu_jinxian/project/GAP}"
BWM_BIN="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
SESSION_NAME="${SESSION_NAME:-gap_gate_fair_flow_ceef}"
TARGET_LOG="${TARGET_LOG:-logs/gap_gate_fair_flow_ceef.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-logs/gap_gate_fair_flow_ceef_watchdog.log}"
POLL_SECONDS="${POLL_SECONDS:-300}"

TASK_NAME="${TASK_NAME:-place_dual_shoes}"
SETTING="${SETTING:-demo_clean}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-200}"
GPU_ID="${GPU_ID:-0}"
TEST_NUM="${TEST_NUM:-100}"
MODEL_WEIGHT="${MODEL_WEIGHT:-auto}"

CKPT_SETTING="${CKPT_SETTING:-${SETTING}_fair_flow_current_eef_seed${SEED}}"
EVAL_NAME="${EVAL_NAME:-fair_flow_current_eef_auto}"
CKPT_PATH="${CKPT_PATH:-checkpoints/${TASK_NAME}_${CKPT_SETTING}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results_flow_interaction_regression_gate/${EVAL_NAME}}"

cd "${ROOT_DIR}"
export PATH="${BWM_BIN}:${PATH}"
mkdir -p "$(dirname "${WATCHDOG_LOG}")" "${RESULTS_ROOT}"
exec >> "${WATCHDOG_LOG}" 2>&1

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

result_file() {
  find "${RESULTS_ROOT}" -name _result.txt -print -quit 2>/dev/null || true
}

session_alive() {
  tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -Fxq -- "$1"
}

echo "[watchdog] $(timestamp) watching session=${SESSION_NAME}"
echo "[watchdog] target_log=${TARGET_LOG}"
echo "[watchdog] ckpt=${CKPT_PATH}"
echo "[watchdog] results_root=${RESULTS_ROOT}"

while session_alive "${SESSION_NAME}"; do
  echo "[watchdog] $(timestamp) ${SESSION_NAME} still running"
  sleep "${POLL_SECONDS}"
done

echo "[watchdog] $(timestamp) ${SESSION_NAME} exited; checking eval artifact"
found="$(result_file)"
if [ -n "${found}" ]; then
  echo "[watchdog] result already exists: ${found}"
  exit 0
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[watchdog] missing checkpoint: ${CKPT_PATH}" >&2
  exit 1
fi

echo "[watchdog] backfill eval100 on GPU${GPU_ID}, model_weight=${MODEL_WEIGHT}"
env \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  CKPT_PATH="${CKPT_PATH}" \
  MODEL_WEIGHT="${MODEL_WEIGHT}" \
  TEST_NUM="${TEST_NUM}" \
  bash eval.sh \
    "${TASK_NAME}" \
    "${SETTING}" \
    "${CKPT_SETTING}" \
    "${EXPERT_DATA_NUM}" \
    "${EPOCHS}" \
    "${GPU_ID}" \
    "${SEED}" \
    "${TEST_NUM}"

found="$(result_file)"
if [ -n "${found}" ]; then
  echo "[watchdog] done: ${found}"
else
  echo "[watchdog] eval finished but no _result.txt found under ${RESULTS_ROOT}" >&2
  exit 1
fi
