#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec bash "${GAP_ROOT}/train.sh" "$@" \
    policy.use_triadic_token=true \
    policy.triadic_mode=pairwise
