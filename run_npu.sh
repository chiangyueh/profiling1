#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2

for shrink_mode in 0 1; do
    echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
    MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
        bash build.sh \
        --run_example mat_mul_v3 eager \
        --example_name=matmul
    echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
done
