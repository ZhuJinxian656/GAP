#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name="${1:-${TASK_NAME:-place_dual_shoes}}"
setting="${2:-${TASK_CONFIG:-demo_clean}}"
expert_data_num="${3:-${EXPERT_DATA_NUM:-100}}"
seed="${4:-${SEED:-0}}"
gpu_ids="${5:-${GPU_IDS:-0,1,2,3}}"
batch_size="${6:-${BATCH_SIZE:-32}}"
num_epochs="${7:-${NUM_EPOCHS:-300}}"
checkpoint_every="${8:-${CHECKPOINT_EVERY:-100}}"
extra_overrides=("${@:9}")
config_name="${CONFIG_NAME:-GAP}"
model_3d="${MODEL_3D:-pi3}"
observation_chunk="${OBSERVATION_CHUNK:-20}"
interval="${INTERVAL:-5}"
wandb_mode="${WANDB_MODE:-offline}"

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export WANDB_MODE="${wandb_mode}"

IFS=',' read -ra gpu_array <<< "${gpu_ids}"
nproc_per_node="${NPROC_PER_NODE:-${#gpu_array[@]}}"

printf 'Training GAP policy with DDP\n'
printf '  task=%s setting=%s expert_data_num=%s seed=%s gpus=%s nproc=%s\n' "${task_name}" "${setting}" "${expert_data_num}" "${seed}" "${gpu_ids}" "${nproc_per_node}"
printf '  per_gpu_batch_size=%s global_batch_size=%s num_epochs=%s checkpoint_every=%s\n' "${batch_size}" "$((batch_size * nproc_per_node))" "${num_epochs}" "${checkpoint_every}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${nproc_per_node}" \
    scripts/train_ddp.py \
    --config-name="${config_name}" \
    task_name="${task_name}" \
    setting="${setting}" \
    expert_data_num="${expert_data_num}" \
    training.seed="${seed}" \
    training.device="cuda" \
    dataloader.batch_size="${batch_size}" \
    val_dataloader.batch_size="${batch_size}" \
    training.num_epochs="${num_epochs}" \
    training.checkpoint_every="${checkpoint_every}" \
    logging.mode="${wandb_mode}" \
    model_3d="${model_3d}" \
    observation_chunk="${observation_chunk}" \
    interval="${interval}" \
    "${extra_overrides[@]}"
