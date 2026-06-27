#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GAP_ROOT}"

task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
ckpt_setting="${3:-${CKPT_SETTING:-${task_config}}}"
expert_data_num="${4:-${EXPERT_DATA_NUM:-50}}"
checkpoint_num="${5:-${CHECKPOINT_NUM:-300}}"
gpu_id="${6:-${GPU_ID:-5}}"
seeds="${7:-${SEEDS:-0}}"
test_num="${8:-${TEST_NUM:-10}}"

ckpt_file="${CKPT_FILE:-checkpoints/${task_name}_${ckpt_setting}_${expert_data_num}/${checkpoint_num}.ckpt}"
wait_interval="${WAIT_INTERVAL:-120}"
log_file="${LOG_FILE:-logs/gap_eval_after_${checkpoint_num}_${task_name}_${expert_data_num}_gpu${gpu_id}.log}"
bwm_bin="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
robotwin_root="${ROBOTWIN_ROOT:-/data1/home/zhu_jinxian/project/robotwin}"
pretrained_root="${GAP_PRETRAINED_ROOT:-${GAP_ROOT}/pretrained}"
report_dir="${RESULTS_ROOT:-${GAP_ROOT}/results}/${task_name}/GAP/${task_config}/${ckpt_setting}/seed_0/${checkpoint_num}"
stop_tmux_session="${STOP_TMUX_SESSION:-}"

mkdir -p "$(dirname "${log_file}")"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${log_file}"
}

log "waiting for checkpoint: ${ckpt_file}"
while [ ! -f "${ckpt_file}" ]; do
    sleep "${wait_interval}"
    log "still waiting for checkpoint: ${ckpt_file}"
done

log "checkpoint ready; starting eval task=${task_name} checkpoint=${checkpoint_num} gpu=${gpu_id} seeds=${seeds} test_num=${test_num}"
if [ -n "${stop_tmux_session}" ]; then
    log "requesting graceful stop for tmux session: ${stop_tmux_session}"
    tmux send-keys -t "${stop_tmux_session}" C-c || true
    sleep 10
fi

env \
    PATH="${bwm_bin}:${PATH}" \
    ROBOTWIN_ROOT="${robotwin_root}" \
    GAP_PRETRAINED_ROOT="${pretrained_root}" \
    WANDB_MODE=offline \
    bash eval.sh "${task_name}" "${task_config}" "${ckpt_setting}" "${expert_data_num}" "${checkpoint_num}" "${gpu_id}" "${seeds}" "${test_num}" \
    >> "${log_file}" 2>&1

log "eval finished; building report"
env PATH="${bwm_bin}:${PATH}" \
    python scripts/make_eval_report.py "${report_dir}" --output "${report_dir}/index.html" \
    >> "${log_file}" 2>&1
log "report ready: ${report_dir}/index.html"
