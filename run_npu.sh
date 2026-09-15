#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || "$1" != "full" ]]; then
    echo "usage: bash run_npu.sh full" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2

# original begin: this A/B loop only linked the installed built-in operator
# for shrink_mode in 0 1; do
#     echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
#     MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
#         bash build.sh \
#         --run_example mat_mul_v3 eager \
#         --example_name=matmul
#
#     echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
# done
# original end: this A/B loop only linked the installed built-in operator

# new begin: build and install the modified custom operator, then run strict A/B processes
vendor_name="matmulab"

bash build.sh \
    --pkg \
    --soc=ascend910b \
    --ops=mat_mul_v3 \
    --vendor_name="${vendor_name}" \
    -j1 \
    -O3

shopt -s nullglob
run_packages=(build_out/cann-ops-nn-${vendor_name}-linux.*.run)
if [[ "${#run_packages[@]}" -ne 1 ]]; then
    echo "fatal: expected one custom operator package, found ${#run_packages[@]}" >&2
    exit 1
fi

"${run_packages[0]}" --quiet --install-path="${ASCEND_OPP_PATH}"
source "${ASCEND_OPP_PATH}/vendors/${vendor_name}_nn/bin/set_env.bash"

for shrink_mode in 0 1; do
    echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
    MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
        bash build.sh \
        --run_example mat_mul_v3 eager cust \
        --vendor_name="${vendor_name}" \
        --example_name=matmul
    echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
done
# new end: build and install the modified custom operator, then run strict A/B processes
