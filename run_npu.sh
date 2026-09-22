#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH
unset MATMUL_BASE_MODE MATMUL_BASE_EXPERIMENT_SELECTED MATMUL_SPLITK_MODE
unset MATMUL_DETERMINISTIC_ADAPTIVE MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

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
    -DASCEND_COMPILE_OPS=mat_mul_v3 >"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi
if ! cmake --build "${build_dir}" --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

host_library="${build_dir}/libophost_nn.so"
opapi_nn=""
opapi_math=""
legacy_common=""
for path in \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_nn.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_nn.so" \
    "${ASCEND_OPP_PATH}/lib64/libopapi_nn.so"; do
    if [[ -f "${path}" ]]; then
        opapi_nn="${path}"
        break
    fi
done
for path in \
    "${ASCEND_OPP_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_math.so"; do
    if [[ -f "${path}" ]]; then
        opapi_math="${path}"
        break
    fi
done
for path in \
    "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so" \
    "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"; do
    if [[ -f "${path}" ]]; then
        legacy_common="${path}"
        break
    fi
done
if [[ ! -f "${host_library}" || -z "${opapi_nn}" || -z "${opapi_math}" || -z "${legacy_common}" ]]; then
    exit 1
fi
ln -sfn -- "${legacy_common}" "${build_dir}/libophost_comm_legacy.so"

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
runner="${build_dir}/test_splitk_routes"
if ! g++ matmul/mat_mul_v3/examples/test_splitk_routes.cpp \
    -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${opapi_nn}" "${opapi_math}" "${runtime_library}" \
    -lnnopbase -lregister -lopp_registry -ldl \
    -Wl,-rpath,"$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}")" \
    -o "${runner}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

dtypes=(fp16_fp16 fp16_fp32 bf16_bf16 bf16_fp32 fp32_fp32)
layouts=(NN NT TN TT)
mn_pairs=(
    "1 1" "1 64" "1 256" "8 8" "8 128" "16 16" "16 256"
    "32 32" "32 128" "32 512" "64 16" "64 64" "64 256"
    "96 96" "128 32" "128 128" "128 384" "192 64" "256 16"
    "256 64" "256 256" "384 32" "384 128" "512 16" "512 64"
    "768 32" "1024 16"
)
k_values=(1024 1536 2048 4096 4608 6144 7680 8192 12288 16384 24576 27392 32768 49152 65536)
workloads=()
pair_count="${#mn_pairs[@]}"
for dtype in "${dtypes[@]}"; do
    for layout in "${layouts[@]}"; do
        for k_index in "${!k_values[@]}"; do
            k="${k_values[${k_index}]}"
            for offset in 0 1 2 3; do
                pair_index=$(( (k_index * 4 + offset) % pair_count ))
                read -r m n <<<"${mn_pairs[${pair_index}]}"
                workloads+=("${dtype}" "${layout}" "${m}" "${n}" "${k}")
            done
        done
    done
done

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"
exec "${runner}" "${workloads[@]}"
