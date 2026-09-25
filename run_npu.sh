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
unset MATMUL_ANALYTIC_SELECTED_TASKS MATMUL_ANALYTIC_SELECTED_WAVES
unset MATMUL_ANALYTIC_SELECTED_K_ITERATIONS MATMUL_ANALYTIC_SELECTED_N_TAIL_WASTE
unset MATMUL_ANALYTIC_SELECTED_ACTIVE_CORES
unset MATMUL_ANALYTIC_SELECTED_M_TASKS MATMUL_ANALYTIC_SELECTED_N_TASKS
unset MATMUL_ANALYTIC_L2_DIMENSIONS MATMUL_ANALYTIC_L2_M_BLOCK MATMUL_ANALYTIC_L2_N_BLOCK
unset MATMUL_ANALYTIC_L2_M_WINDOWS MATMUL_ANALYTIC_L2_N_WINDOWS
unset MATMUL_ANALYTIC_L2_WINDOW_BYTES MATMUL_ANALYTIC_L2_ESTIMATED_TRAFFIC
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
from math import gcd

def stepped(first, last, step, offsets=(0,)):
    return sorted({base + offset for base in range(first, last + 1, step)
                   for offset in offsets if first <= base + offset <= last})

m_groups = (
    list(range(129, 257)),
    list(range(257, 513)),
    stepped(513, 1024, 4, (0, 1, 3)),
    stepped(1025, 2048, 8, (0, 1, 7)),
    stepped(2049, 4096, 16, (0, 1, 15)),
)
n_groups = (
    stepped(16, 512, 16, (0, 1, 15)),
    stepped(513, 2048, 64, (0, 1, 63)),
    stepped(2049, 8192, 128, (0, 1, 127)),
    stepped(8193, 24576, 128, (0, 1, 127)),
    stepped(24577, 65536, 256, (0, 1, 255)),
)
k_groups = (
    sorted(set(stepped(512, 8192, 128, (0, 1))) | {8192}),
    sorted(set(stepped(8193, 16384, 128, (0, 1))) | {16384}),
    sorted(set(stepped(16385, 24576, 128, (0, 1))) | {24576}),
    sorted(set(stepped(24577, 65536, 256, (0, 1))) | {65536}),
)
byte_limit = 1024 * 1024 * 1024
per_cell_limit = 20000

def cell_records(dtype, dtype_index, m_index, n_index, k_index):
    ms = m_groups[m_index]
    ns = n_groups[n_index]
    ks = k_groups[k_index]
    total = len(ms) * len(ns) * len(ks)
    seed = 8510 + dtype_index * 1009 + m_index * 211 + n_index * 43 + k_index * 17
    offset = seed % total
    stride = (seed * 2 + 1) % total
    if stride == 0:
        stride = 1
    while gcd(stride, total) != 1:
        stride += 2
        if stride >= total:
            stride = 1
    emitted = 0
    attempt_limit = min(total, per_cell_limit * 8)
    for iteration in range(attempt_limit):
        flat = (offset + iteration * stride) % total
        k = ks[flat % len(ks)]
        flat //= len(ks)
        n = ns[flat % len(ns)]
        flat //= len(ns)
        m = ms[flat]
        if 2 * (m * k + k * n + m * n) > byte_limit:
            continue
        yield dtype, "NN", m, n, k
        emitted += 1
        if emitted >= per_cell_limit:
            break

active = []
for dtype_index, dtype in enumerate(("fp16_fp16", "bf16_bf16")):
    for m_index in range(len(m_groups)):
        for n_index in range(len(n_groups)):
            for k_index in range(len(k_groups)):
                active.append(cell_records(dtype, dtype_index, m_index, n_index, k_index))

while active:
    next_active = []
    for records in active:
        try:
            print("\t".join(str(value) for value in next(records)))
            next_active.append(records)
        except StopIteration:
            pass
    active = next_active
PY

adaptive_count="$(wc -l <"${workload_manifest}")"
printf '{"stage":"workload_generation","status":"passed","coverage":"200_interleaved_multi_m_tile_dtype_m_n_k_cells","candidates":%d}\n' "${adaptive_count}"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

success_target="${MATMUL_SUCCESS_TARGET:-10000}"
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
