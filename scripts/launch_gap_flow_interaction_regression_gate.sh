#!/usr/bin/env bash
set -euo pipefail

# Regression gate for flow/action-UV experiments.
# The previous six-way eval used BATCH_SIZE=256 and is not comparable with the
# older batch32 DINO-only baseline. Keep BATCH_SIZE=32 for fair comparisons.

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
EVAL_OLD_DINO="${EVAL_OLD_DINO:-true}"
TRAIN_REPRO_DINO="${TRAIN_REPRO_DINO:-true}"
TRAIN_INTERACTION_FAIR="${TRAIN_INTERACTION_FAIR:-false}"
TRAIN_FLOW_UV_MODE_SWEEP="${TRAIN_FLOW_UV_MODE_SWEEP:-false}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-true}"
FORCE_TRAIN="${FORCE_TRAIN:-false}"

cd "${ROOT_DIR}"
export PATH="${BWM_BIN}:${PATH}"

mkdir -p "${RESULTS_ROOT}" logs reports
RESULTS_ROOT="$(cd "${RESULTS_ROOT}" && pwd)"

printf 'Flow/action-UV regression gate\n'
printf '  root=%s\n' "${ROOT_DIR}"
printf '  task=%s setting=%s demos=%s seed=%s gpu=%s\n' "${TASK_NAME}" "${SETTING}" "${EXPERT_DATA_NUM}" "${SEED}" "${GPU_ID}"
printf '  batch_size=%s epochs=%s checkpoint_every=%s test_num=%s\n' "${BATCH_SIZE}" "${EPOCHS}" "${CHECKPOINT_EVERY}" "${TEST_NUM}"
printf '  train_interaction_fair=%s train_flow_uv_mode_sweep=%s\n' "${TRAIN_INTERACTION_FAIR}" "${TRAIN_FLOW_UV_MODE_SWEEP}"
printf 'WARNING: previous six-way eval with BATCH_SIZE=256 is not comparable with old batch32 baselines.\n'

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

  printf '\n[gate] Eval %s model_weight=%s -> %s\n' "${name}" "${model_weight}" "${result_root}"
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
    printf '\n[gate] Skip train %s: checkpoint exists at %s\n' "${exp_suffix}" "${ckpt_path}"
    return
  fi

  printf '\n[gate] Train %s -> %s\n' "${exp_suffix}" "${ckpt_path}"
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

old_dino_ckpt="checkpoints/${TASK_NAME}_${SETTING}_dino_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"

if [ "${EVAL_OLD_DINO}" = "true" ]; then
  if [ ! -f "${old_dino_ckpt}" ]; then
    printf 'Old DINO-only checkpoint not found: %s\n' "${old_dino_ckpt}" >&2
    exit 1
  fi
  run_eval "re_eval_old_dino_auto" "${old_dino_ckpt}" "${SETTING}_dino_only_seed${SEED}" "auto"
  run_eval "re_eval_old_dino_model" "${old_dino_ckpt}" "${SETTING}_dino_only_seed${SEED}" "model"
fi

if [ "${TRAIN_REPRO_DINO}" = "true" ]; then
  repro_tag="${SETTING}_dino_only_repro_seed${SEED}"
  repro_ckpt="checkpoints/${TASK_NAME}_${repro_tag}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt"
  run_train "${repro_tag}" "dino_only_repro"
  if [ "${EVAL_AFTER_TRAIN}" = "true" ]; then
    run_eval "train_repro_dino_auto" "${repro_ckpt}" "${repro_tag}" "auto"
  fi
fi

if [ "${TRAIN_INTERACTION_FAIR}" = "true" ]; then
  run_train "${SETTING}_fair_diffusion_current_eef_seed${SEED}" "fair_diffusion_current_eef" \
    "policy.generative_mode=diffusion" \
    "policy.use_interaction_field=true" \
    "policy.interaction_field.mode=current_eef"

  run_train "${SETTING}_fair_diffusion_action_uv_seed${SEED}" "fair_diffusion_action_uv" \
    "policy.generative_mode=diffusion" \
    "policy.use_interaction_field=true" \
    "policy.interaction_field.mode=action_uv"

  run_train "${SETTING}_fair_flow_dino_only_seed${SEED}" "fair_flow_dino_only" \
    "policy.generative_mode=flow_matching" \
    "policy.use_interaction_field=false" \
    "policy.interaction_field.mode=disabled"

  run_train "${SETTING}_fair_flow_current_eef_seed${SEED}" "fair_flow_current_eef" \
    "policy.generative_mode=flow_matching" \
    "policy.use_interaction_field=true" \
    "policy.interaction_field.mode=current_eef"

  run_train "${SETTING}_fair_flow_action_uv_seed${SEED}" "fair_flow_action_uv" \
    "policy.generative_mode=flow_matching" \
    "policy.use_interaction_field=true" \
    "policy.interaction_field.mode=action_uv"

  if [ "${TRAIN_FLOW_UV_MODE_SWEEP}" = "true" ]; then
    run_train "${SETTING}_fair_flow_action_uv_expert_final_seed${SEED}" "fair_flow_action_uv_expert_final" \
      "policy.generative_mode=flow_matching" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=action_uv" \
      "policy.interaction_field.uv_supervision_mode=expert_final"

    run_train "${SETTING}_fair_flow_action_uv_flow_interp_seed${SEED}" "fair_flow_action_uv_flow_interp" \
      "policy.generative_mode=flow_matching" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=action_uv" \
      "policy.interaction_field.uv_supervision_mode=flow_interp"

    run_train "${SETTING}_fair_flow_action_uv_clean_only_seed${SEED}" "fair_flow_action_uv_clean_only" \
      "policy.generative_mode=flow_matching" \
      "policy.use_interaction_field=true" \
      "policy.interaction_field.mode=action_uv" \
      "policy.interaction_field.uv_supervision_mode=clean_only"
  fi

  if [ "${EVAL_AFTER_TRAIN}" = "true" ]; then
    run_eval "fair_diffusion_current_eef_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_diffusion_current_eef_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_diffusion_current_eef_seed${SEED}" "auto"
    run_eval "fair_diffusion_action_uv_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_diffusion_action_uv_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_diffusion_action_uv_seed${SEED}" "auto"
    run_eval "fair_flow_dino_only_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_dino_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_dino_only_seed${SEED}" "auto"
    run_eval "fair_flow_current_eef_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_current_eef_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_current_eef_seed${SEED}" "auto"
    run_eval "fair_flow_action_uv_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_action_uv_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_action_uv_seed${SEED}" "auto"
    if [ "${TRAIN_FLOW_UV_MODE_SWEEP}" = "true" ]; then
      run_eval "fair_flow_action_uv_expert_final_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_action_uv_expert_final_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_action_uv_expert_final_seed${SEED}" "auto"
      run_eval "fair_flow_action_uv_flow_interp_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_action_uv_flow_interp_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_action_uv_flow_interp_seed${SEED}" "auto"
      run_eval "fair_flow_action_uv_clean_only_auto" "checkpoints/${TASK_NAME}_${SETTING}_fair_flow_action_uv_clean_only_seed${SEED}_${EXPERT_DATA_NUM}/${EPOCHS}.ckpt" "${SETTING}_fair_flow_action_uv_clean_only_seed${SEED}" "auto"
    fi
  fi
fi

printf '\n[gate] Done. Summarize with:\n'
printf '  %s scripts/summarize_flow_interaction_regression_gate.py\n' "${BWM_BIN}/python"
