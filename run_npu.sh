#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH
unset MATMUL_BASE_MODE MATMUL_BASE_EXPERIMENT_SELECTED MATMUL_SPLITK_MODE
unset MATMUL_ANALYTIC_SCORE_N128 MATMUL_ANALYTIC_SCORE_N256 MATMUL_ANALYTIC_SCORE_N512
unset MATMUL_DETERMINISTIC_ADAPTIVE MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED
unset MATMUL_CAMPAIGN

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

build_dir="${PWD}/build"
build_log="$(mktemp)"
workload_manifest="$(mktemp)"
trap 'rm -f "${build_log}" "${workload_manifest}"' EXIT

printf '{"stage":"host_build","status":"begin"}\n'
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
printf '{"stage":"host_build","status":"passed"}\n'

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

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
runner="${build_dir}/test_wide_n_panel_reuse_base"
printf '{"stage":"runner_build","status":"begin"}\n'
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
printf '{"stage":"runner_build","status":"passed"}\n'

printf '{"stage":"workload_generation","status":"begin"}\n'
python3 - >"${workload_manifest}" <<'PY'
import random

rng = random.Random(8507)
m_values = (1, 3, 7, 8, 12, 16, 24, 32, 48, 64, 80, 96, 112, 128)
n_tiles = tuple(range(41, 131))
k_values = (512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288,
            15104, 16384, 18432, 20480, 24576, 32768, 49152, 65536)
byte_limit = 1024 * 1024 * 1024

rows = set()
for dtype in ("fp16_fp16", "bf16_bf16"):
    for m in m_values:
        for n_tile in n_tiles:
            for n_tail in (0, 17, 127):
                n = n_tile * 256 + n_tail
                for k in k_values:
                    record = (dtype, "NN", m, n, k)
                    total_bytes = 2 * (m * k + k * n + m * n)
                    if total_bytes <= byte_limit:
                        rows.add(record)
rows = list(rows)
rng.shuffle(rows)
for record in rows:
    print("\t".join(str(value) for value in record))
PY

adaptive_count="$(wc -l <"${workload_manifest}")"
printf '{"stage":"workload_generation","status":"passed","coverage":"analytic_m_n_k_dtype_strata","candidates":%d}\n' "${adaptive_count}"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

success_target="${MATMUL_SUCCESS_TARGET:-3000}"
cell_quota="${MATMUL_CELL_QUOTA:-0}"
if [[ "${cell_quota}" -eq 0 ]]; then
    cell_quota_json=null
    theoretical_maximum_pairs_json=null
else
    cell_quota_json="${cell_quota}"
    theoretical_maximum_pairs_json="$((200 * cell_quota))"
fi
panel_rc=0
printf '{"campaign_begin":"WIDE_N_ANALYTIC_SELECTOR","target_passes":%d,"maximum_per_joint_cell":%s,"theoretical_maximum_pairs":%s}\n' \
    "${success_target}" "${cell_quota_json}" "${theoretical_maximum_pairs_json}"
set +e
MATMUL_CAMPAIGN=WIDE_N_ANALYTIC_SELECTOR MATMUL_TARGET_PASSES="${success_target}" MATMUL_CELL_QUOTA="${cell_quota}" \
    "${runner}" --manifest "${workload_manifest}"
panel_rc=$?
set -e
printf '{"campaign_complete":"WIDE_N_ANALYTIC_SELECTOR","process_result_code":%d}\n' "${panel_rc}"
if [[ "${panel_rc}" -ne 0 ]]; then
    exit "${panel_rc}"
fi
printf '{"validation_complete":true,"campaigns":1,"candidate_branch":"WIDE_N_ANALYTIC_SELECTOR"}\n'
