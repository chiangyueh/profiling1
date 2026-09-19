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
#NEW
selected_shapes="$(mktemp)"

cleanup() {
    rm -f "${build_log}" "${run_log}" "${selected_shapes}"
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

#NEW
export MATMUL_V3_HOST_LIBRARY="${v3_host_library}"
export MATMUL_LEGACY_COMMON_LIBRARY="${official_legacy_common_library}"
export LD_LIBRARY_PATH="${runtime_path}"

#NEW
if [[ "$#" -gt 0 ]]; then
    if (( $# % 3 != 0 )); then
        echo "fatal: shapes must be supplied as M N K triples" >&2
        exit 2
    fi
    while [[ "$#" -gt 0 ]]; do
        printf 'fp32\tNN\t%s\t%s\t%s\tUSER\n' "$1" "$2" "$3" >>"${selected_shapes}"
        printf 'fp32\tNT\t%s\t%s\t%s\tUSER\n' "$1" "$2" "$3" >>"${selected_shapes}"
        shift 3
    done
else
    python3 scripts/core_oracle_sampler.py select-shrink \
        --runner "${example_binary}" --selected "${selected_shapes}" \
        --run-log "${run_log}" --quota 20 --discovery-batch 64
fi

#NEW
if ! python3 scripts/core_oracle_sampler.py compare \
    --runner "${example_binary}" --selected "${selected_shapes}" \
    --run-log "${run_log}" --quota 20 --batch-size 8; then
    cat "${run_log}" >&2
    echo "fatal: no valid MatMulV3 shrink comparison was produced" >&2
    exit 1
fi
