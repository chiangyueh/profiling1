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
unset MATMUL_CAMPAIGN

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

build_dir="${PWD}/build"
build_log="$(mktemp)"
workload_manifest="$(mktemp)"
comparison_manifest="$(mktemp)"
panel_log="$(mktemp)"
trap 'rm -f "${build_log}" "${workload_manifest}" "${comparison_manifest}" "${panel_log}"' EXIT

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
python3 - "${PWD}/data/wide_n_result48_shapes.csv" >"${workload_manifest}" <<'PY'
import csv
import random
import sys

with open(sys.argv[1], newline="") as source:
    historical = {
        (f"{row['input_dtype']}_{row['output_dtype']}", row["layout"],
         int(row["m"]), int(row["n"]), int(row["k"]))
        for row in csv.DictReader(source)
    }
if len(historical) != 300:
    raise SystemExit(f"expected 300 result48 shapes, found {len(historical)}")

rng = random.Random(8506)
m_ranges = ((1, 9), (10, 31), (32, 63), (64, 95), (96, 128))
n_ranges = ((4865, 7168), (7169, 12288), (12289, 18432), (18433, 24576), (24577, 32768))
k_ranges = ((512, 8192), (8193, 15871), (15872, 23552), (23553, 32768), (32769, 65536))
n_offsets = (0, 1, 17, 63, 127, 191)
k_offsets = (0, 1, 17, 31, 63)
byte_limit = 1024 * 1024 * 1024

def quantized(lo, hi, quantum, offset):
    first = (lo - offset + quantum - 1) // quantum
    last = (hi - offset) // quantum
    if first > last:
        return rng.randint(lo, hi)
    return rng.randint(first, last) * quantum + offset

rows = set()
sample_index = 0
candidates_per_cell = 256
maximum_attempts_per_cell = 100000
for dtype in ("fp16_fp16", "bf16_bf16"):
    for m_lo, m_hi in m_ranges:
        for n_lo, n_hi in n_ranges:
            for k_lo, k_hi in k_ranges:
                minimum_bytes = 2 * (m_lo * k_lo + k_lo * n_lo + m_lo * n_lo)
                if minimum_bytes > byte_limit:
                    continue
                cell_rows = set()
                attempts = 0
                while len(cell_rows) < candidates_per_cell and attempts < maximum_attempts_per_cell:
                    attempts += 1
                    m = rng.randint(m_lo, m_hi)
                    n = quantized(n_lo, n_hi, 256, n_offsets[sample_index % len(n_offsets)])
                    k = quantized(k_lo, k_hi, 64, k_offsets[sample_index % len(k_offsets)])
                    sample_index += 1
                    record = (dtype, "NN", m, n, k)
                    total_bytes = 2 * (m * k + k * n + m * n)
                    if total_bytes <= byte_limit and record not in historical:
                        cell_rows.add(record)
                rows.update(cell_rows)
rows = list(rows)
rng.shuffle(rows)
for record in rows:
    print("\t".join(str(value) for value in record))
PY

adaptive_count="$(wc -l <"${workload_manifest}")"
printf '{"stage":"workload_generation","status":"passed","coverage":"new_stratified_shapes_excluding_result48","candidates":%d}\n' "${adaptive_count}"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

campaign_failures=0
success_target="${MATMUL_SUCCESS_TARGET:-5000}"
cell_quota="${MATMUL_CELL_QUOTA:-64}"
theoretical_maximum_pairs=$((250 * cell_quota))
run_campaign() {
    local campaign="$1"
    local target="$2"
    shift 2
    local rc=0
    printf '{"campaign_begin":"%s","target_passes":%d}\n' "${campaign}" "${target}"
    MATMUL_CAMPAIGN="${campaign}" MATMUL_TARGET_PASSES="${target}" "$@" || rc=$?
    printf '{"campaign_complete":"%s","process_result_code":%d}\n' "${campaign}" "${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        campaign_failures=$((campaign_failures + 1))
    fi
}

panel_rc=0
printf '{"campaign_begin":"WIDE_N_PANEL_ONLY","target_passes":%d,"maximum_per_joint_cell":%d,"theoretical_maximum_pairs":%d}\n' \
    "${success_target}" "${cell_quota}" "${theoretical_maximum_pairs}"
set +e
MATMUL_CAMPAIGN=WIDE_N_PANEL_ONLY MATMUL_TARGET_PASSES="${success_target}" MATMUL_CELL_QUOTA="${cell_quota}" \
    "${runner}" --manifest "${workload_manifest}" | tee "${panel_log}"
panel_status=("${PIPESTATUS[@]}")
set -e
panel_rc="${panel_status[0]}"
printf '{"campaign_complete":"WIDE_N_PANEL_ONLY","process_result_code":%d}\n' "${panel_rc}"
if [[ "${panel_rc}" -ne 0 ]]; then
    campaign_failures=$((campaign_failures + 1))
fi

python3 - "${panel_log}" >"${comparison_manifest}" <<'PY'
import json
import sys

seen = set()
for line in open(sys.argv[1]):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if (row.get("candidate_branch") != "WIDE_N_PANEL_ONLY" or
            row.get("correctness") != "PASS" or row.get("result_code") != 0):
        continue
    shape = row["shape"].split("_")
    record = (f"{row['input_dtype']}_{row['output_dtype']}", shape[3],
              int(shape[0][1:]), int(shape[1][1:]), int(shape[2][1:]))
    if record not in seen:
        seen.add(record)
        print("\t".join(str(value) for value in record))
PY

comparison_count="$(wc -l <"${comparison_manifest}")"
if [[ "${comparison_count}" -eq 0 ]]; then
    printf '{"fatal":"panel_only_produced_no_comparable_shape"}\n' >&2
    exit 1
fi
printf '{"ablation_comparison_shapes":%d,"source":"WIDE_N_PANEL_ONLY_PASS"}\n' "${comparison_count}"
run_campaign WIDE_N_WINDOW_ONLY "${comparison_count}" "${runner}" --manifest "${comparison_manifest}"
printf '{"overnight_complete":true,"campaigns":2,"campaign_process_failures":%d}\n' "${campaign_failures}"
