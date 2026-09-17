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
official_opapi_nn_library=""
for candidate in \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_nn.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_nn.so" \
    "${ASCEND_OPP_PATH}/lib64/libopapi_nn.so"; do
    if [[ -f "${candidate}" ]]; then
        official_opapi_nn_library="${candidate}"
        break
    fi
done
if [[ -z "${official_opapi_nn_library}" ]]; then
    echo "fatal: installed CANN libopapi_nn.so is missing" >&2
    exit 1
fi
official_opapi_nn_dir="$(dirname -- "${official_opapi_nn_library}")"

#NEW
official_opapi_math_library=""
for candidate in \
    "${ASCEND_OPP_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_math.so"; do
    if [[ -f "${candidate}" ]]; then
        official_opapi_math_library="${candidate}"
        break
    fi
done
if [[ -z "${official_opapi_math_library}" ]]; then
    echo "fatal: installed CANN libopapi_math.so is missing" >&2
    exit 1
fi
official_opapi_math_dir="$(dirname -- "${official_opapi_math_library}")"
runtime_path="${official_opapi_nn_dir}:${official_opapi_math_dir}:${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

#NEW
official_legacy_common_library=""
for candidate in \
    "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so" \
    "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"; do
    if [[ -f "${candidate}" ]]; then
        official_legacy_common_library="${candidate}"
        break
    fi
done
if [[ -z "${official_legacy_common_library}" ]]; then
    echo "fatal: installed CANN libophost_comm_legacy.so is missing" >&2
    exit 1
fi
#NEW
if ! nm -D "${official_opapi_nn_library}" | awk '$3 ~ /^aclnnMatmulGetWorkspaceSize(@.*)?$/ {found=1} END {exit !found}'; then
    echo "fatal: installed CANN libopapi_nn.so has no aclnnMatmulGetWorkspaceSize" >&2
    exit 1
fi
if ! nm -D "${official_legacy_common_library}" | awk '$3 ~ /^LegacyMmCheckHitV3Shape(@.*)?$/ {found=1} END {exit !found}'; then
    echo "fatal: installed CANN legacy library has no MatMul V3 selector" >&2
    exit 1
fi

example_source="matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp"
example_binary="${host_build}/test_aclnn_matmul"
runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
common_link_args=(
    -std=gnu++17 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${official_opapi_nn_library}" "${official_opapi_math_library}" \
    "${runtime_library}" -lnnopbase -lregister -lopp_registry -ldl
)
if ! g++ "${example_source}" "${common_link_args[@]}" \
    -Wl,-rpath,"${official_opapi_nn_dir}:${official_opapi_math_dir}" \
    -o "${example_binary}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

#NEW
# The runner must use CANN's complete official API. Only the MatMulV3 host
# tiler is repository-local.
loaded_opapi="$(LD_LIBRARY_PATH="${runtime_path}" ldd "${example_binary}" | awk '$1 == "libopapi_nn.so" {print $3}')"
loaded_math="$(LD_LIBRARY_PATH="${runtime_path}" ldd "${example_binary}" | awk '$1 == "libopapi_math.so" {print $3}')"
if [[ -z "${loaded_opapi}" || "$(readlink -f -- "${loaded_opapi}")" != "$(readlink -f -- "${official_opapi_nn_library}")" ]]; then
    echo "fatal: runner did not resolve installed CANN libopapi_nn.so" >&2
    exit 1
fi
if [[ -z "${loaded_math}" || "$(readlink -f -- "${loaded_math}")" != "$(readlink -f -- "${official_opapi_math_library}")" ]]; then
    echo "fatal: runner did not resolve installed CANN libopapi_math.so" >&2
    exit 1
fi

nt_shape_args=()
nn_shape_args=()
al1_m=(1 2 3 4 5 6 7)
al1_n=(64 80 96 112 128 144 160 176 192 208 224 240 256 272 288 304)
al1_k=(4096 4608 5120 5632 6144 6656 7168)

#NEW
# Sixty statically legal AL1 shapes plus twenty NT probes for BASE, BL1 and
# Split-K selection. The actual branch is always reported by the local tiler.
for ((shape_index = 0; shape_index < 60; ++shape_index)); do
    nt_shape_args+=(
        "${al1_m[shape_index % ${#al1_m[@]}]}"
        "${al1_n[(shape_index / ${#al1_m[@]}) % ${#al1_n[@]}]}"
        "${al1_k[(shape_index * 3 + shape_index / ${#al1_m[@]}) % ${#al1_k[@]}]}"
    )
done
nt_shape_args+=(
    5 1024 16384 384 112 16384 96 640 16384 192 112 10240
    24 80 16384 13 384 16384 16 80 10240 5 2048 16384
    384 256 16384 26 89 16389 1920 32 27392 2048 16 32768
    2048 64 96 4096 64 96 8192 64 96 12288 80 192
    16384 96 256 20480 112 128 512 512 32768 768 256 27392
)

#NEW
# Forty varied NN shapes plus targeted small-MN, large-K and NKM probes.
nn_m=(1 3 5 7 9 11 13 15 16 17 24 32 48 64 96 128 192 256 384 512)
nn_n=(65 80 96 112 128 160 192 256 320 384 512 640 768 1024 1280 1536 1792 2048)
nn_k=(512 768 1024 1536 2048 3072 4096 5120 5632 6144 6656 7168 8192 10240 12288 16384)
for ((shape_index = 0; shape_index < 40; ++shape_index)); do
    nn_shape_args+=(
        "${nn_m[shape_index % ${#nn_m[@]}]}"
        "${nn_n[(shape_index * 5 + 3) % ${#nn_n[@]}]}"
        "${nn_k[(shape_index * 7 + 1) % ${#nn_k[@]}]}"
    )
done
nn_shape_args+=(
    16 16 6144 32 32 16384 64 64 16384 96 32 32768
    128 64 32768 256 128 32768 384 256 16384 512 512 16384
    768 256 27392 1024 512 32768 1920 32 27392 2048 16 32768
    1024 1024 4096 768 768 8192 512 1024 16384 256 1536 16384
    128 2048 16384 64 1024 32768 96 640 27392 192 384 32768
)

if [[ "$#" -gt 0 ]]; then
    if (( $# % 3 != 0 )); then
        echo "fatal: shapes must be supplied as M N K triples" >&2
        exit 2
    fi
    nt_shape_args=("$@")
    nn_shape_args=("$@")
fi

#NEW
run_measurements() {
    local layout="$1"
    local mode="$2"
    local requested_core="$3"
    shift 3
    local transpose_b=0
    local raw=""
    local rc=0
    if [[ "${layout}" == "NT" ]]; then
        transpose_b=1
    fi
    if [[ "${requested_core}" == "0" ]]; then
        if raw="$(env -u MATMUL_V3_FORCE_CORE_NUM \
            MATMUL_B_TRANSPOSE="${transpose_b}" \
            MATMUL_V3_MEASUREMENT_MODE="${mode}" \
            MATMUL_V3_ONLY=1 MATMUL_V3_SHRINK_IDLE_CORES=0 \
            MATMUL_V3_HOST_LIBRARY="${v3_host_library}" \
            MATMUL_LEGACY_COMMON_LIBRARY="${official_legacy_common_library}" \
            LD_LIBRARY_PATH="${runtime_path}" \
            "${example_binary}" "$@" 2>>"${run_log}")"; then
            :
        else
            rc=$?
        fi
    else
        if raw="$(MATMUL_V3_FORCE_CORE_NUM="${requested_core}" \
            MATMUL_B_TRANSPOSE="${transpose_b}" \
            MATMUL_V3_MEASUREMENT_MODE="${mode}" \
            MATMUL_V3_ONLY=1 MATMUL_V3_SHRINK_IDLE_CORES=0 \
            MATMUL_V3_HOST_LIBRARY="${v3_host_library}" \
            MATMUL_LEGACY_COMMON_LIBRARY="${official_legacy_common_library}" \
            LD_LIBRARY_PATH="${runtime_path}" \
            "${example_binary}" "$@" 2>>"${run_log}")"; then
            :
        else
            rc=$?
        fi
    fi
    printf '%s\n' "${raw}" | awk '/^\{"shape":/'
    if [[ "${rc}" -ne 0 ]]; then
        printf '{"layout":"%s","mode":"%s","requested_core":%s,"status":"RUNNER_ERROR","rc":%d}\n' \
            "${layout}" "${mode}" "${requested_core}" "${rc}"
    fi
}

#NEW
# Interleaving high and low core counts reduces correlation with thermal drift.
core_order=(20 1 19 2 18 3 17 4 16 5 15 6 14 7 13 8 12 9 11 10)
for layout in NT NN; do
    if [[ "${layout}" == "NT" ]]; then
        active_shapes=("${nt_shape_args[@]}")
    else
        active_shapes=("${nn_shape_args[@]}")
    fi
    run_measurements "${layout}" official_pre 0 "${active_shapes[@]}"
    for requested_core in "${core_order[@]}"; do
        run_measurements "${layout}" core_sweep "${requested_core}" "${active_shapes[@]}"
    done
    run_measurements "${layout}" official_post 0 "${active_shapes[@]}"
done
