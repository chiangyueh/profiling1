#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export MATMUL_V3_CAMPAIGN=remaining_core_sweep
exec bash run_npu.sh
