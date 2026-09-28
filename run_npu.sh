#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2

build_dir="${PWD}/build"
build_log="$(mktemp)"
trap 'rm -f "${build_log}"' EXIT

if ! cmake -S . -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUSTOM=FALSE \
    -DENABLE_BINARY=FALSE \
    -DENABLE_PACKAGE=FALSE \
    -DENABLE_TEST=FALSE \
    -DASCEND_OP_NAME=mat_mul_v3 \
    -DASCEND_COMPILE_OPS='mat_mul_v3;matmul.common' >"${build_log}" 2>&1 || \
   ! cmake --build "${build_dir}" --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

legacy=""
for path in \
    "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so" \
    "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"; do
    if [[ -f "${path}" ]]; then
        legacy="${path}"
        break
    fi
done
if [[ -z "${legacy}" ]]; then
    echo "fatal: libophost_comm_legacy.so not found" >&2
    exit 1
fi
ln -sfn -- "${legacy}" "${build_dir}/libophost_comm_legacy.so"

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi

runner="${build_dir}/test_aclnn_matmul_balanced"
if ! g++ matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp \
    -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/include" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    -lopapi_nn -lopapi_math "${runtime_library}" -lnnopbase \
    -lregister -lopp_registry -lunified_dlog -lmetadef -ldl \
    -o "${runner}" >"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"
export MATMUL_HOST_LIBRARY="${build_dir}/libophost_nn.so"
exec "${runner}"
