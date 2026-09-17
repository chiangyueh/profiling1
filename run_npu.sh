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

cleanup() {
    rm -f "${build_log}" "${run_log}"
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

#NEW
v3_host_library="${host_build}/libophost_nn.so"
if [[ ! -f "${v3_host_library}" ]]; then
    echo "fatal: independently built MatMulV3 host library is missing" >&2
    exit 1
fi

#NEW
v2_shrink_source="matmul/mat_mul_v2_shrink/matmul_v2_shrink_tiling.cpp"
v2_shrink_library="${host_build}/libmatmul_v2_shrink_tiling.so"
if [[ ! -f "${v2_shrink_library}" || "${v2_shrink_source}" -nt "${v2_shrink_library}" ]]; then
    if ! g++ -std=c++17 -O2 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=0 "${v2_shrink_source}" \
        -I "${ASCEND_HOME_PATH}/include" \
        -I "${ASCEND_HOME_PATH}/x86_64-linux/include" \
        -L "${ASCEND_HOME_PATH}/lib64" \
        -L "${ASCEND_HOME_PATH}/x86_64-linux/lib64" \
        -lopp_registry -lregister \
        -o "${v2_shrink_library}" >>"${build_log}" 2>&1; then
        cat "${build_log}" >&2
        exit 1
    fi
fi

#NEW
v2_official_library="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_legacy.so"
if [[ ! -f "${v2_official_library}" ]]; then
    echo "fatal: ${v2_official_library} does not exist" >&2
    exit 1
fi

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
    -lopapi_nn -lopapi_math "${runtime_library}" -lnnopbase -lregister -ldl \
    -o "${example_binary}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

shape_args=()
default_m=(1 3 5 7 9 11 13 15 16 17 24 32 48 64 96 128 192 256 384 512)
default_n=(65 80 96 112 128 160 192 256 320 384 512 640 768 1024 1280 1536 1792 2048)
default_k=(512 768 1024 1536 2048 3072 4096 5120 5632 6144 6656 7168 8192 10240 12288 16384)
m_offset=(0 2 4 6 8 10 12)
n_offset=(0 3 5 7 9 11 13)
k_offset=(0 1 3 5 7 9 11)

#NEW
# The first 100 shapes emphasize skinny matrices, where an idle-core reduction
# can exist. The remaining 20 widen M/N so the same V2 path is not tested only
# on one narrow shape class.
for ((shape_index = 0; shape_index < 100; ++shape_index)); do
    shape_args+=(
        "${default_m[shape_index % ${#default_m[@]}]}"
        "${default_n[(shape_index * 5 + 3) % ${#default_n[@]}]}"
        "${default_k[(shape_index * 7 + 1) % ${#default_k[@]}]}"
    )
done
for ((shape_index = 0; shape_index < 20; ++shape_index)); do
    shape_args+=(
        "$((default_m[(shape_index * 7 + 13) % ${#default_m[@]}] + m_offset[shape_index % ${#m_offset[@]}]))"
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

if ! shrinked_raw="$(MATMUL_SHRINK_MODE=1 \
    MATMUL_V3_SHRINK_IDLE_CORES=1 \
    MATMUL_V3_HOST_LIBRARY="${v3_host_library}" \
    MATMUL_V2_OFFICIAL_LIBRARY="${v2_official_library}" \
    MATMUL_V2_SHRINK_LIBRARY="${v2_shrink_library}" \
    "${example_binary}" "${shape_args[@]}" 2>>"${run_log}")"; then
    echo "fatal: shrink measurement failed" >&2
    if [[ -n "${shrinked_raw}" ]]; then
        printf '%s\n' "${shrinked_raw}" >&2
    fi
    cat "${run_log}" >&2
    exit 1
fi
mapfile -t shrinked_results < <(printf '%s\n' "${shrinked_raw}" | \
    awk -F'|' 'NF == 5 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+([.][0-9]+)?$/')
expected_result_count=$((${#shape_args[@]} / 3))
if [[ "${#shrinked_results[@]}" -ne "${expected_result_count}" ]]; then
    printf 'fatal: shrink output count mismatch: expected=%d actual=%d\n' \
        "${expected_result_count}" "${#shrinked_results[@]}" >&2
    exit 1
fi

original_args=()
for shrinked_result in "${shrinked_results[@]}"; do
    IFS='|' read -r candidate_m candidate_n candidate_k _ _ <<<"${shrinked_result}"
    original_args+=("${candidate_m}" "${candidate_n}" "${candidate_k}")
done

if ! original_raw="$(MATMUL_SHRINK_MODE=0 MATMUL_V3_SHRINK_IDLE_CORES=0 \
    MATMUL_V3_HOST_LIBRARY="${v3_host_library}" \
    MATMUL_V2_OFFICIAL_LIBRARY="${v2_official_library}" \
    "${example_binary}" "${original_args[@]}" 2>>"${run_log}")"; then
    echo "fatal: original measurement failed" >&2
    if [[ -n "${original_raw}" ]]; then
        printf '%s\n' "${original_raw}" >&2
    fi
    cat "${run_log}" >&2
    exit 1
fi
mapfile -t original_results < <(printf '%s\n' "${original_raw}" | \
    awk -F'|' 'NF == 5 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $4 ~ /^[0-9]+([.][0-9]+)?$/')
if [[ "${#original_results[@]}" -ne "${#shrinked_results[@]}" ]]; then
    printf 'fatal: original output count mismatch: expected=%d actual=%d\n' \
        "${#shrinked_results[@]}" "${#original_results[@]}" >&2
    exit 1
fi

for ((result_index = 0; result_index < ${#shrinked_results[@]}; ++result_index)); do
    IFS='|' read -r m n k shrinked_latency branch <<<"${shrinked_results[result_index]}"
    IFS='|' read -r original_m original_n original_k original_latency original_branch \
        <<<"${original_results[result_index]}"
    if [[ "${m}" != "${original_m}" || "${n}" != "${original_n}" || "${k}" != "${original_k}" ]]; then
        echo "fatal: result shape order changed between shrinked and original runs" >&2
        exit 1
    fi
    if [[ "${branch}" != "${original_branch}" ]]; then
        echo "fatal: natural branch changed between shrinked and original runs for M${m}_N${n}_K${k}" >&2
        exit 1
    fi
    printf '{"shape":"M%s_N%s_K%s_NN","branch":"%s","shrinked_latency":"%s","original_latency":"%s"}\n' \
        "${m}" "${n}" "${k}" "${branch}" "${shrinked_latency}" "${original_latency}"
done
