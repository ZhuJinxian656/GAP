#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GAP_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ZARR_PATH="${ZARR_PATH:-data/place_dual_shoes-demo_clean-50-pi3-20-5.zarr}"

if [ -d "${ZARR_PATH}" ]; then
    "${PYTHON_BIN}" scripts/inspect_gap_zarr.py --zarr "${ZARR_PATH}" --sample-rows 2 --sample-values 1024
else
    printf 'Skipping zarr inspect; missing %s\n' "${ZARR_PATH}"
fi

"${PYTHON_BIN}" scripts/smoke_test_latent_modes.py
"${PYTHON_BIN}" scripts/smoke_test_future_modes.py
"${PYTHON_BIN}" scripts/smoke_test_dataset_optional_fields.py

if [ -d "${ZARR_PATH}" ] && "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    SMOKE_GPU_ID="${SMOKE_GPU_ID:-0}"
    printf 'CUDA and zarr detected. Running one-step training smoke on GPU %s.\n' "${SMOKE_GPU_ID}"
    WANDB_MODE=disabled bash train.sh \
        place_dual_shoes demo_clean 50 0 "${SMOKE_GPU_ID}" 1 1 999 \
        training.max_train_steps=1 \
        training.use_ema=false \
        dataloader.num_workers=0 \
        dataloader.persistent_workers=false \
        val_dataloader.num_workers=0 \
        val_dataloader.persistent_workers=false
else
    printf 'Skipping short training smoke; zarr missing or torch.cuda.is_available() is false.\n'
fi
