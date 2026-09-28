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
workload_manifest="${PWD}/data/independent_base_npu_verified.tsv"
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
runner="${build_dir}/test_wide_n_panel_reuse_base"
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

printf '# stage=workload_generation status=begin\n'
if [[ ! -s "${workload_manifest}" ]]; then
    printf '{"fatal":"preverified_manifest_missing"}\n' >&2
    exit 1
fi
manifest_rows="$(wc -l < "${workload_manifest}")"
if ! manifest_audit="$(awk -F '\t' '
    NF != 6 { invalid++ }
    NF == 6 {
        rows++
        dtype = $1
        layout = $2
        m = $3 + 0
        n = $4 + 0
        k = $5 + 0
        cell = $6 + 0
        dtype_band = dtype == "bf16_bf16" ? 1 : 0
        n_band = n <= 512 ? 0 : (n <= 8192 ? 1 : 2)
        k_band = k <= 512 ? 0 : (k <= 8192 ? 1 : 2)
        expected_cell = dtype_band * 9 + n_band * 3 + k_band
        if ((dtype != "fp16_fp16" && dtype != "bf16_bf16") || layout != "NN" ||
            m < 1 || m > 4096 || n < 1 || n > 65536 || k < 1 || k > 65536 ||
            cell < 0 || cell > 17 || cell != expected_cell) invalid++
        seen[m]++
        seen_cell[m, cell]++
        if (rows == 1 || n < min_n) min_n = n
        if (rows == 1 || k < min_k) min_k = k
        if (n > max_n) max_n = n
        if (k > max_k) max_k = k
        if (n % 16 == 0) aligned_n++; else unaligned_n++
        if (k % 16 == 0) aligned_k++; else unaligned_k++
    }
    END {
        missing = 0
        bad_cells = 0
        minimum = 999999
        maximum = 0
        for (m = 1; m <= 4096; ++m) {
            if (seen[m] == 0) missing++
            if (seen[m] < minimum) minimum = seen[m]
            if (seen[m] > maximum) maximum = seen[m]
            for (cell = 0; cell < 18; ++cell) {
                if (seen_cell[m, cell] != 2) bad_cells++
            }
        }
        printf "rows=%d distinct_m=%d missing_m=%d candidates_per_m_min=%d candidates_per_m_max=%d bad_cells=%d n_range=%d..%d k_range=%d..%d invalid=%d", \
            rows, length(seen), missing, minimum, maximum, bad_cells, min_n, max_n, min_k, max_k, invalid
        exit(invalid != 0 || missing != 0 || length(seen) != 4096 || bad_cells != 0 ||
             rows != 147456 || min_n != 1 || max_n != 65536 || min_k != 1 || max_k != 65536 ||
             aligned_n == 0 || unaligned_n == 0 || aligned_k == 0 || unaligned_k == 0)
    }
' "${workload_manifest}")"; then
    printf '{"fatal":"manifest_does_not_cover_every_m","audit":"%s"}\n' "${manifest_audit}" >&2
    exit 1
fi
printf '# stage=workload_generation status=passed source=host_preverified_all_m_base %s\n' "${manifest_audit}"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export MATMUL_BASE_FULL_M_SWEEP=1
export MATMUL_FORCE_BASE_ONLY=1

success_target="${MATMUL_SUCCESS_TARGET:-0}"
m_quota=0
m_cell_quota="${MATMUL_M_CELL_QUOTA:-1}"
cell_quota="${MATMUL_CELL_QUOTA:-0}"
if [[ "${cell_quota}" -eq 0 ]]; then
    cell_quota_json=null
    theoretical_maximum_pairs_json=null
else
    cell_quota_json="${cell_quota}"
    theoretical_maximum_pairs_json="$((200 * cell_quota))"
fi
panel_rc=0
printf '# campaign=INDEPENDENT_BASE_SELECTOR target_passes=%d maximum_per_joint_cell=%s theoretical_maximum_pairs=%s measurement_order=OCCO\n' \
    "${success_target}" "${cell_quota_json}" "${theoretical_maximum_pairs_json}"
set +e
MATMUL_CAMPAIGN=INDEPENDENT_BASE_SELECTOR MATMUL_TARGET_PASSES="${success_target}" \
    MATMUL_CELL_QUOTA="${cell_quota}" MATMUL_M_QUOTA="${m_quota}" \
    MATMUL_M_CELL_QUOTA="${m_cell_quota}" \
    "${runner}" --manifest "${workload_manifest}" | python3 tools/compact_matmul_log.py
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
