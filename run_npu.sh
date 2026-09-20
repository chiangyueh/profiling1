#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2

build_log="$(mktemp)"
run_log="$(mktemp)"
trap 'rm -f "${build_log}" "${run_log}"' EXIT

if ! cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUSTOM=FALSE -DENABLE_BINARY=FALSE -DENABLE_PACKAGE=FALSE \
    -DENABLE_TEST=FALSE -DASCEND_OP_NAME=mat_mul_v3 \
    -DASCEND_COMPILE_OPS=mat_mul_v3 >"${build_log}" 2>&1; then
    cat "${build_log}"
    exit 1
fi
if ! cmake --build build --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}"
    exit 1
fi

legacy_so="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"
ln -sfn "${legacy_so}" build/libophost_comm_legacy.so
export MATMUL_V3_HOST_LIBRARY="${PWD}/build/libophost_nn.so"

if ! bash build.sh --run_example mat_mul_v3 eager --example_name=matmul >"${run_log}" 2>&1; then
    cat "${run_log}"
    exit 1
fi
sed -n 's/^\[[^]]*\] \({"tiling":.*\)$/\1/p' "${run_log}"
