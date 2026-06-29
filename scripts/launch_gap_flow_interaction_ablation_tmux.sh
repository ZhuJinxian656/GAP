#!/usr/bin/env bash
set -euo pipefail

# Flow/action-conditioned interaction ablation launcher.
#
# Stage 1: run the two real-zarr debug validation jobs in parallel.
# Stage 2: if both validations pass, launch the six formal ablations in tmux.

ROOT_DIR="${ROOT_DIR:-/data1/home/zhu_jinxian/project/GAP}"
BWM_BIN="${BWM_BIN:-/data1/home/zhu_jinxian/worldarena-dataengine-research/.conda/BWM/bin}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-200}"
FORMAL_CHECKPOINT_EVERY="${FORMAL_CHECKPOINT_EVERY:-100}"
SEED="${SEED:-0}"
EXPERT_DATA_NUM="${EXPERT_DATA_NUM:-50}"
TASK_NAME="${TASK_NAME:-place_dual_shoes}"
SETTING="${SETTING:-demo_clean}"
SKIP_DEBUG="${SKIP_DEBUG:-false}"

cd "$ROOT_DIR"
mkdir -p logs

MASTER_LOG="logs/gap_flow_interaction_orchestrator.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

printf '[%s] Flow interaction ablation orchestrator started.\n' "$(date '+%F %T')"
printf '[%s] Formal config: epochs=%s checkpoint_every=%s batch_size=%s seed=%s\n' \
  "$(date '+%F %T')" "$FORMAL_EPOCHS" "$FORMAL_CHECKPOINT_EVERY" "$BATCH_SIZE" "$SEED"

common_overrides=(
  "model_3d=pi3"
  "observation_chunk=20"
  "interval=5"
  "use_pi3_features=false"
  "policy.use_pi3_features=false"
  "use_future_loss=false"
  "policy.use_future_loss=false"
  "latent_mode=dino_only"
  "policy.latent_mode=dino_only"
  "future_target_mode=none"
  "policy.future_target_mode=none"
)

run_debug_validation() {
  local name="$1"
  local gpu="$2"
  shift 2

  local log_file="logs/${name}_gpu${gpu}.log"
  printf '[%s] Debug validation start: %s on GPU %s -> %s\n' \
    "$(date '+%F %T')" "$name" "$gpu" "$log_file"
  (
    cd "$ROOT_DIR"
    env PATH="$BWM_BIN:$PATH" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      WANDB_MODE=disabled \
      "$BWM_BIN/python" scripts/train.py \
        "task_name=$TASK_NAME" \
        "setting=$SETTING" \
        "expert_data_num=$EXPERT_DATA_NUM" \
        "training.seed=$SEED" \
        "training.device=cuda:0" \
        "training.debug=true" \
        "logging.mode=disabled" \
        "checkpoint.save_ckpt=false" \
        "${common_overrides[@]}" \
        "$@"
  ) > "$log_file" 2>&1
  printf '[%s] Debug validation done: %s\n' "$(date '+%F %T')" "$name"
}

launch_formal_train() {
  local session="$1"
  local gpu="$2"
  local checkpoint_tag="$3"
  shift 3

  local log_file="logs/${session}_gpu${gpu}.log"
  local checkpoint_dir="checkpoints/${TASK_NAME}_${checkpoint_tag}_${EXPERT_DATA_NUM}"

  if [ -f "${checkpoint_dir}/${FORMAL_EPOCHS}.ckpt" ]; then
    printf '[%s] Skip %s: checkpoint exists at %s/%s.ckpt\n' \
      "$(date '+%F %T')" "$session" "$checkpoint_dir" "$FORMAL_EPOCHS"
    return
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    printf '[%s] Skip %s: tmux session already exists\n' "$(date '+%F %T')" "$session"
    return
  fi

  printf '[%s] Launch formal train: %s on GPU %s -> %s\n' \
    "$(date '+%F %T')" "$session" "$gpu" "$log_file"
  tmux new-session -d -s "$session" \
    "cd '$ROOT_DIR' && env PATH='$BWM_BIN':\$PATH WANDB_MODE=disabled bash train.sh '$TASK_NAME' '$SETTING' '$EXPERT_DATA_NUM' '$SEED' '$gpu' '$BATCH_SIZE' '$FORMAL_EPOCHS' '$FORMAL_CHECKPOINT_EVERY' ${common_overrides[*]} checkpoint_tag='$checkpoint_tag' exp_name='${TASK_NAME}_${checkpoint_tag}_${EXPERT_DATA_NUM}' logging.name='${TASK_NAME}_${checkpoint_tag}_${EXPERT_DATA_NUM}' $* > '$log_file' 2>&1"
}

if [ "$SKIP_DEBUG" = "true" ]; then
  printf '[%s] Skip debug validation because SKIP_DEBUG=true.\n' "$(date '+%F %T')"
else
  set +e
  run_debug_validation gap_debug_flow_dino_only 0 \
    "policy.generative_mode=flow_matching" \
    "policy.use_interaction_field=false" \
    "policy.interaction_field.mode=disabled" &
  pid_debug_dino="$!"

  run_debug_validation gap_debug_flow_action_uv 1 \
    "policy.generative_mode=flow_matching" \
    "policy.use_interaction_field=true" \
    "policy.interaction_field.mode=action_uv" &
  pid_debug_action_uv="$!"

  status=0
  wait "$pid_debug_dino" || status=1
  wait "$pid_debug_action_uv" || status=1
  set -e

  if [ "$status" -ne 0 ]; then
    printf '[%s] Debug validation failed; formal ablations were not launched. See logs/gap_debug_flow_*.\n' \
      "$(date '+%F %T')"
    exit "$status"
  fi
fi

printf '[%s] Debug gate satisfied; launching formal six-way ablation.\n' "$(date '+%F %T')"

launch_formal_train gap_ablate_diffusion_dino_only 0 demo_clean_diffusion_dino_only_seed0 \
  "policy.generative_mode=diffusion" \
  "policy.use_interaction_field=false" \
  "policy.interaction_field.mode=disabled"

launch_formal_train gap_ablate_diffusion_current_eef 1 demo_clean_diffusion_current_eef_seed0 \
  "policy.generative_mode=diffusion" \
  "policy.use_interaction_field=true" \
  "policy.interaction_field.mode=current_eef"

launch_formal_train gap_ablate_diffusion_action_uv 2 demo_clean_diffusion_action_uv_seed0 \
  "policy.generative_mode=diffusion" \
  "policy.use_interaction_field=true" \
  "policy.interaction_field.mode=action_uv"

launch_formal_train gap_ablate_flow_dino_only 3 demo_clean_flow_dino_only_seed0 \
  "policy.generative_mode=flow_matching" \
  "policy.use_interaction_field=false" \
  "policy.interaction_field.mode=disabled"

launch_formal_train gap_ablate_flow_current_eef 4 demo_clean_flow_current_eef_seed0 \
  "policy.generative_mode=flow_matching" \
  "policy.use_interaction_field=true" \
  "policy.interaction_field.mode=current_eef"

launch_formal_train gap_ablate_flow_action_uv 5 demo_clean_flow_action_uv_seed0 \
  "policy.generative_mode=flow_matching" \
  "policy.use_interaction_field=true" \
  "policy.interaction_field.mode=action_uv"

printf '[%s] Formal ablation launch complete. Monitor with: tmux ls\n' "$(date '+%F %T')"
