#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH
unset MATMUL_BASE_MODE MATMUL_BASE_EXPERIMENT_SELECTED MATMUL_SPLITK_MODE
unset MATMUL_STRUCTURAL_CANDIDATES MATMUL_STRUCTURAL_REFERENCE_BASE_N
unset MATMUL_STRUCTURAL_REFERENCE_BASE_K MATMUL_STRUCTURAL_TOTAL_K_LOOPS
unset MATMUL_STRUCTURAL_L2_MAX_N_BLOCK
unset MATMUL_ANALYTIC_N_QUANTUM MATMUL_ANALYTIC_MIN_BASE_K
unset MATMUL_ANALYTIC_MAX_BASE_M MATMUL_ANALYTIC_MAX_BASE_N MATMUL_ANALYTIC_TARGET_TASKS
unset MATMUL_ANALYTIC_TAIL_SLOTS MATMUL_ANALYTIC_CRITICAL_CUBE
unset MATMUL_ANALYTIC_TOTAL_CUBE MATMUL_ANALYTIC_PANEL_TRAFFIC
unset MATMUL_ANALYTIC_SELECTED_TASKS MATMUL_ANALYTIC_SELECTED_WAVES
unset MATMUL_ANALYTIC_SELECTED_K_ITERATIONS MATMUL_ANALYTIC_SELECTED_N_TAIL_WASTE
unset MATMUL_ANALYTIC_SELECTED_ACTIVE_CORES
unset MATMUL_ANALYTIC_SELECTED_M_TASKS MATMUL_ANALYTIC_SELECTED_N_TASKS
unset MATMUL_ANALYTIC_L2_DIMENSIONS MATMUL_ANALYTIC_L2_M_BLOCK MATMUL_ANALYTIC_L2_N_BLOCK
unset MATMUL_ANALYTIC_L2_M_WINDOWS MATMUL_ANALYTIC_L2_N_WINDOWS
unset MATMUL_ANALYTIC_L2_WINDOW_BYTES MATMUL_ANALYTIC_L2_ESTIMATED_TRAFFIC
unset MATMUL_DETERMINISTIC_ADAPTIVE MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED
unset MATMUL_BASE_FULL_M_SWEEP
unset MATMUL_FORCE_BASE_ONLY
unset MATMUL_CAMPAIGN

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

build_dir="${PWD}/build"
build_log="$(mktemp)"
trap 'rm -f "${build_log}"' EXIT

printf '# stage=host_build status=begin\n'
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
printf '# stage=host_build status=passed\n'

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
    printf '{"fatal":"required_library_missing","host":%s,"opapi_nn":%s,"opapi_math":%s,"legacy":%s}\n' \
        "$([[ -f "${host_library}" ]] && printf true || printf false)" \
        "$([[ -n "${opapi_nn}" ]] && printf true || printf false)" \
        "$([[ -n "${opapi_math}" ]] && printf true || printf false)" \
        "$([[ -n "${legacy_common}" ]] && printf true || printf false)" >&2
    exit 1
fi
ln -sfn -- "${legacy_common}" "${build_dir}/libophost_comm_legacy.so"
export LD_LIBRARY_PATH="${build_dir}:$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
runner="${build_dir}/test_three_shape_base"
printf '# stage=runner_build status=begin\n'
if ! g++ matmul/mat_mul_v3/examples/test_splitk_routes.cpp \
    matmul/mat_mul_v3/op_host/op_api/matmul.cpp \
    -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -I "${PWD}/common/stub/op_api" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${opapi_nn}" "${opapi_math}" "${runtime_library}" \
    -lnnopbase -lregister -lopp_registry -lunified_dlog -lmetadef -ldl \
    -Wl,-rpath,"$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}")" \
    -o "${runner}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi
printf '# stage=runner_build status=passed\n'

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
panel_rc=0
printf '# campaign=THIRD_SHAPE_CORE_BALANCE shapes=1 measurement_order=OCCO\n'
set +e
MATMUL_CAMPAIGN=THIRD_SHAPE_CORE_BALANCE MATMUL_TARGET_PASSES=1 \
    "${runner}" \
    fp16_fp16 NN 4096 512 7168 | python3 tools/compact_three_shape_base.py
pipeline_status=("${PIPESTATUS[@]}")
panel_rc="${pipeline_status[0]}"
converter_rc="${pipeline_status[1]}"
set -e
if [[ "${converter_rc}" -ne 0 ]]; then
    printf 'fatal: CSV conversion failed rc=%d\n' "${converter_rc}" >&2
    exit "${converter_rc}"
fi
if [[ "${panel_rc}" -ne 0 ]]; then
    exit "${panel_rc}"
fi
