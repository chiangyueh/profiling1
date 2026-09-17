#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH

host_build="${PWD}/build"
build_log="$(mktemp)"
run_log="$(mktemp)"
installed_host=""
host_backup=""
host_replaced=0

cleanup() {
    if [[ "${host_replaced}" -eq 1 && -f "${host_backup}" && -n "${installed_host}" ]]; then
        cp "${host_backup}" "${installed_host}"
    fi
    rm -f "${host_backup}" "${build_log}" "${run_log}"
}
trap cleanup EXIT

if ! cmake -S . -B "${host_build}" \
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
if ! cmake --build "${host_build}" --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

installed_host_link="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_nn.so"
if [[ ! -e "${installed_host_link}" ]]; then
    echo "fatal: ${installed_host_link} does not exist" >&2
    exit 1
fi
installed_host="$(readlink -f "${installed_host_link}")"
host_backup="${host_build}/libophost_nn.so.official"
cp "${installed_host}" "${host_backup}"
host_replaced=1
cp "${host_build}/libophost_nn.so" "${installed_host}"

example_source="matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp"
example_binary="${host_build}/test_aclnn_matmul"
runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
if ! g++ "${example_source}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -lopapi_nn -lopapi_math "${runtime_library}" -lnnopbase \
    -o "${example_binary}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

shape_args=()
default_m=(1024 1280 1536 1792 2048 2304 2560 3072 3584 4096 5120 6144)
default_n=(1152 1408 1664 1920 2176 2432 2816 3200 3712 4352 4864 5632 6400)
default_k=(4096 4608 5120 5632 6144 6656 7168 7680 8192 9216 10240)
m_offset=(3 7 11 13 17 19 23)
n_offset=(5 9 15 21 27 33 39)
k_offset=(1 3 5 7 9 11 13)

#NEW
# 60 large aligned shapes followed by 60 large unaligned shapes.
for ((shape_index = 0; shape_index < 60; ++shape_index)); do
    shape_args+=(
        "${default_m[shape_index % ${#default_m[@]}]}"
        "${default_n[(shape_index * 5 + 3) % ${#default_n[@]}]}"
        "${default_k[(shape_index * 7 + 1) % ${#default_k[@]}]}"
    )
done
for ((shape_index = 0; shape_index < 60; ++shape_index)); do
    shape_args+=(
        "$((default_m[(shape_index * 7 + 2) % ${#default_m[@]}] + m_offset[shape_index % ${#m_offset[@]}]))"
        "$((default_n[(shape_index * 3 + 4) % ${#default_n[@]}] + n_offset[(shape_index * 2 + 1) % ${#n_offset[@]}]))"
        "$((default_k[(shape_index * 5 + 6) % ${#default_k[@]}] + k_offset[(shape_index * 3 + 2) % ${#k_offset[@]}]))"
    )
done
if [[ "$#" -gt 0 ]]; then
    if (( $# % 3 != 0 )); then
        echo "fatal: shapes must be supplied as M N K triples" >&2
        exit 2
    fi
    shape_args=("$@")
fi

shape_count=$((${#shape_args[@]} / 3))
chunk_size=10
for ((chunk_start = 0; chunk_start < shape_count; chunk_start += chunk_size)); do
    chunk_count=$((shape_count - chunk_start))
    if ((chunk_count > chunk_size)); then
        chunk_count=${chunk_size}
    fi
    chunk_args=("${shape_args[@]:chunk_start * 3:chunk_count * 3}")

    if ! MATMUL_V3_SHRINK_IDLE_CORES=1 "${example_binary}" "${chunk_args[@]}" 2>>"${run_log}"; then
        printf 'fatal: shrink measurement failed for shapes %d-%d\n' \
            "$((chunk_start + 1))" "$((chunk_start + chunk_count))" >&2
        cat "${run_log}" >&2
        exit 1
    fi

    if ! MATMUL_V3_SHRINK_IDLE_CORES=0 "${example_binary}" "${chunk_args[@]}" 2>>"${run_log}"; then
        printf 'fatal: original measurement failed for shapes %d-%d\n' \
            "$((chunk_start + 1))" "$((chunk_start + chunk_count))" >&2
        cat "${run_log}" >&2
        exit 1
    fi
done
