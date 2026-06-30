#!/usr/bin/env bash
set -euo pipefail

# Run one regression-gate job. This is intended for tmux-per-GPU launches so
# the long gate does not serialize unrelated experiments.

ROOT_DIR="${ROOT_DIR:-/data1/home/zhu_jinxian/project/GAP}"
BWM_BIN="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
TASK_NAME="${TASK_NAME:-place_dual_shoes}"
SETTING="${SETTING:-demo_clean}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
TEST_NUM="${TEST_NUM:-100}"
RESULTS_ROOT="${RESULTS_ROOT:-results_flow_interaction_regression_gate}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"
JOB="${JOB:?set JOB to one regression gate job name}"

cd "${ROOT_DIR}"
export PATH="${BWM_BIN}:${PATH}"

mkdir -p "${RESULTS_ROOT}" logs reports
RESULTS_ROOT="$(cd "${RESULTS_ROOT}" && pwd)"

printf 'Flow/action-UV regression single job\n'
printf '  job=%s\n' "${JOB}"
printf '  root=%s\n' "${ROOT_DIR}"
printf '  task=%s setting=%s demos=%s seed=%s gpu=%s\n' "${TASK_NAME}" "${SETTING}" "${EXPERT_DATA_NUM}" "${SEED}" "${GPU_ID}"
printf '  batch_size=%s epochs=%s checkpoint_every=%s test_num=%s\n' "${BATCH_SIZE}" "${EPOCHS}" "${CHECKPOINT_EVERY}" "${TEST_NUM}"

if [ "${BATCH_SIZE}" != "32" ]; then
  printf 'WARNING: BATCH_SIZE=%s. Fair regression-gate comparisons should use BATCH_SIZE=32.\n' "${BATCH_SIZE}"
fi

common_dino_overrides=(
  "latent_mode=dino_only"
  "policy.latent_mode=dino_only"
  "use_future_loss=false"
  "policy.use_future_loss=false"
  "future_target_mode=none"
  "policy.future_target_mode=none"
  "use_pi3_features=false"
  "policy.use_pi3_features=false"
)

run_eval() {
  local name="$1"
  local ckpt_path="$2"
  local ckpt_setting="$3"
  local model_weight="$4"
  local result_root="${RESULTS_ROOT}/${name}"

  printf '\n[gate-job] Eval %s model_weight=%s -> %s\n' "${name}" "${model_weight}" "${result_root}"
  env \
    RESULTS_ROOT="${result_root}" \
    CKPT_PATH="${ckpt_path}" \
    MODEL_WEIGHT="${model_weight}" \
    TEST_NUM="${TEST_NUM}" \
    bash eval.sh \
      "${TASK_NAME}" \
      "${SETTING}" \
      "${ckpt_setting}" \
      "${EXPERT_DATA_NUM}" \
      "${EPOCHS}" \
      "${GPU_ID}" \
      "${SEED}" \
      "${TEST_NUM}"
}

run_train() {
  local checkpoint_tag="$1"
  local exp_suffix="$2"
  shift 2
  local ckpt_path="checkpoints/${TASK_NAME}_${checkpoint_tag}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"
  local exp_name="${TASK_NAME}_${SETTING}_${EXPERT_DATA_NUM}_${exp_suffix}_seed${SEED}"

  if [ -f "${ckpt_path}" ] && [ "${FORCE_TRAIN}" != "true" ]; then
    printf '\n[gate-job] Skip train %s: checkpoint exists at %s\n' "${exp_suffix}" "${ckpt_path}"
    return
  fi

  printf '\n[gate-job] Train %s -> %s\n' "${exp_suffix}" "${ckpt_path}"
  WANDB_MODE="${WANDB_MODE:-offline}" bash train.sh \
    "${TASK_NAME}" \
    "${SETTING}" \
    "${EXPERT_DATA_NUM}" \
    "${SEED}" \
    "${GPU_ID}" \
    "${BATCH_SIZE}" \
    "${EPOCHS}" \
    "${CHECKPOINT_EVERY}" \
    "${common_dino_overrides[@]}" \
    "checkpoint_tag=${checkpoint_tag}" \
    "exp_name=${exp_name}" \
    "logging.name=${exp_name}" \
    "$@"
}

train_then_eval() {
  local checkpoint_tag="$1"
  local exp_suffix="$2"
  local eval_name="$3"
  shift 3
  local ckpt_path="checkpoints/${TASK_NAME}_${checkpoint_tag}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"

  run_train "${checkpoint_tag}" "${exp_suffix}" "$@"
  if [ "${EVAL_AFTER_TRAIN}" = "true" ]; then
    run_eval "${eval_name}" "${ckpt_path}" "${checkpoint_tag}" "auto"
  fi
}

old_dino_ckpt="checkpoints/${TASK_NAME}_${SETTING}_dino_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"

case "${JOB}" in
  re_eval_old_dino_auto)
    run_eval "re_eval_old_dino_auto" "${old_dino_ckpt}" "${SETTING}_dino_only_seed${SEED}" "auto"
    ;;
  re_eval_old_dino_model)
    run_eval "re_eval_old_dino_model" "${old_dino_ckpt}" "${SETTING}_dino_only_seed${SEED}" "model"
    ;;
  train_repro_dino_auto)
    train_then_eval "${SETTING}_dino_only_repro_seed${SEED}" "dino_only_repro" "train_repro_dino_auto"
    ;;
  fair_diffusion_current_eef_auto)
    train_then_eval "${SETTING}_fair_diffusion_current_eef_seed${SEED}" "fair_diffusion_current_eef" "fair_diffusion_current_eef_auto" \
      "policy.generative_mode=diffusion" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=current_eef"
    ;;
  fair_diffusion_action_uv_auto)
    train_then_eval "${SETTING}_fair_diffusion_action_uv_seed${SEED}" "fair_diffusion_action_uv" "fair_diffusion_action_uv_auto" \
      "policy.generative_mode=diffusion" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=action_uv"
    ;;
  fair_flow_dino_only_auto)
    train_then_eval "${SETTING}_fair_flow_dino_only_seed${SEED}" "fair_flow_dino_only" "fair_flow_dino_only_auto" \
      "policy.generative_mode=flow_matching" \
      "policy.use_interaction_field=false" \
      "policy.interaction_field.mode=disabled"
    ;;
  fair_flow_action_uv_auto)
    train_then_eval "${SETTING}_fair_flow_action_uv_seed${SEED}" "fair_flow_action_uv" "fair_flow_action_uv_auto" \
      "policy.generative_mode=flow_matching" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=action_uv"
    ;;
  *)
    printf 'Unknown JOB: %s\n' "${JOB}" >&2
    exit 2
    ;;
esac

printf '\n[gate-job] Done: %s\n' "${JOB}"
