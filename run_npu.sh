#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
unset ASCEND_CUSTOM_OPP_PATH

cmake -S . -B build \
    -DBUILD_PATH="${PWD}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUSTOM=FALSE \
    -DENABLE_BINARY=FALSE \
    -DENABLE_PACKAGE=FALSE \
    -DENABLE_TEST=FALSE \
    -DASCEND_OP_NAME=mat_mul_v3 \
    -DASCEND_COMPILE_OPS=mat_mul_v3
cmake --build build --target ophost_nn -- -j1

example_source="matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp"
example_binary="${PWD}/build/test_aclnn_matmul"
runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi

g++ "${example_source}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -lopapi_nn -lopapi_math "${runtime_library}" -lnnopbase \
    -o "${example_binary}"

tiling_library="${PWD}/build/libophost_nn.so"
if [[ ! -f "${tiling_library}" ]]; then
    echo "fatal: host tiling library was not produced: ${tiling_library}" >&2
    exit 1
fi
legacy_library="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"
if [[ ! -f "${legacy_library}" ]]; then
    echo "fatal: installed MatMul host dependency was not found: ${legacy_library}" >&2
    exit 1
fi
ln -sfn "${legacy_library}" "${PWD}/build/libophost_comm_legacy.so"

for shrink_mode in 0 1; do
    echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
    MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
        LD_PRELOAD="${legacy_library}:${tiling_library}${LD_PRELOAD:+:${LD_PRELOAD}}" \
        "${example_binary}"
    echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
done
