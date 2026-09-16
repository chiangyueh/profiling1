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

shape_args=(
    4 128 5376
    4 192 5376
    4 256 5376
    7 128 6400
    7 192 6400
    7 256 6400
    8 128 6144
    8 192 6144
    8 256 6144
    10 128 5120
    10 192 5120
    10 256 5120
    12 128 4096
    12 192 4096
    12 256 4096
    16 128 4096
    16 192 4096
    16 256 4096
)

if ! shrinked_raw="$(MATMUL_V3_SHRINK_IDLE_CORES=1 "${example_binary}" "${shape_args[@]}" 2>>"${run_log}")"; then
    printf '%s\n' "${shrinked_raw}" >&2
    cat "${run_log}" >&2
    exit 1
fi

shape_count=$((${#shape_args[@]} / 3))
mapfile -t shrinked_records < <(printf '%s\n' "${shrinked_raw}" | awk '/^SKIP$|^[0-9]+([.][0-9]+)?$/')
if [[ "${#shrinked_records[@]}" -ne "${shape_count}" ]]; then
    echo "fatal: shrink output count does not match shape count" >&2
    exit 1
fi

eligible_indices=()
original_args=()
for ((shape_index = 0; shape_index < shape_count; ++shape_index)); do
    if [[ "${shrinked_records[shape_index]}" == "SKIP" ]]; then
        continue
    fi
    arg_index=$((shape_index * 3))
    eligible_indices+=("${shape_index}")
    original_args+=("${shape_args[arg_index]}" "${shape_args[arg_index + 1]}" "${shape_args[arg_index + 2]}")
done
if [[ "${#eligible_indices[@]}" -eq 0 ]]; then
    echo "fatal: no candidate shape produced an actual core reduction" >&2
    exit 1
fi

if ! original_raw="$(MATMUL_V3_SHRINK_IDLE_CORES=0 "${example_binary}" "${original_args[@]}" 2>>"${run_log}")"; then
    printf '%s\n' "${original_raw}" >&2
    cat "${run_log}" >&2
    exit 1
fi
mapfile -t original_latencies < <(printf '%s\n' "${original_raw}" | awk '/^[0-9]+([.][0-9]+)?$/')
if [[ "${#original_latencies[@]}" -ne "${#eligible_indices[@]}" ]]; then
    echo "fatal: original output count does not match eligible shape count" >&2
    exit 1
fi

for ((result_index = 0; result_index < ${#eligible_indices[@]}; ++result_index)); do
    shape_index="${eligible_indices[result_index]}"
    arg_index=$((shape_index * 3))
    m="${shape_args[arg_index]}"
    n="${shape_args[arg_index + 1]}"
    k="${shape_args[arg_index + 2]}"
    printf '{"shape":"M%s_N%s_K%s_NT","shrinked_latency":"%s","original_latency":"%s"}\n' \
        "${m}" "${n}" "${k}" "${shrinked_records[shape_index]}" "${original_latencies[result_index]}"
done
