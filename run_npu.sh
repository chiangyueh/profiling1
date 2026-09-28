#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "usage: bash run_npu.sh" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2

exec bash build.sh --run_example mat_mul_v3 eager --example_name=matmul
