#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=3
REPEAT=10
SAMPLES=15
MEASURED_SHAPES=20
EXPECTED_VARIANTS=4
NUMERIC_PREFLIGHT_MAX_MIB=64

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Audits all 12 C220 suffix ownership equations and performs paired' \
        'direct-baseline/candidate measurements for every strict core deletion.'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        -d|--device) PHYSICAL_DEVICE="${2:?missing physical NPU ID}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "fatal: unsupported argument: $1" >&2; exit 2 ;;
    esac
done
[[ "${MODE}" == "full" ]] || { usage >&2; exit 2; }
[[ "${PHYSICAL_DEVICE}" =~ ^[0-9]+$ ]] || {
    echo "fatal: physical NPU ID must be a non-negative integer" >&2
    exit 2
}

cd "${ROOT}"
RUN_LOG="${ROOT}/run.log"
: >"${RUN_LOG}"
exec 3>&1
exec >>"${RUN_LOG}" 2>&1

announce() {
    printf '%s\n' "$*"
    printf '%s\n' "$*" >&3
}

export CANN_ROOT=/usr/local/Ascend/ascend-toolkit/8.1
export ASCEND_MATMUL_MANUAL_ENV=1
export ASCENDC_SOC_VERSION=Ascend910B3
export SOC_VERSION=Ascend910B3
export ASCEND_RT_VISIBLE_DEVICES="${PHYSICAL_DEVICE}"
export DEVICE_ID=0
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
unset ASCEND_CUSTOM_OPP_PATH || true
while IFS='=' read -r name _; do
    case "${name}" in
        *RUNTIME_KB*|*TUNING_BANK*) unset "${name}" ;;
    esac
done < <(env)
source "${ROOT}/scripts/env.sh" >/dev/null

CAMPAIGN_ID="$({
    sha256sum \
        run_npu.sh \
        tools/generate_matmul_core_ownership_matrix.py \
        tools/analyze_matmul_core_ownership.py \
        scripts/build_all.sh \
        scripts/env.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        matmul_rule_selector/core_ownership_rules.py \
        matmul_rule_selector/improved_selector.py \
        matmul_rule_selector/source_family_audit_contract.json
    find matmul_rule_selector/baseline_core -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_core_ownership_v1/${CAMPAIGN_ID}"
BASELINE_PACKET_DIR="${CAMPAIGN_DIR}/baseline_packets"
CANDIDATE_PACKET_DIR="${CAMPAIGN_DIR}/candidate_packets"
BASELINE_MANIFEST="${CAMPAIGN_DIR}/baseline_manifest.csv"
CANDIDATE_MANIFEST="${CAMPAIGN_DIR}/candidate_manifest.csv"
BASELINE_VARIANTS="${CAMPAIGN_DIR}/baseline_variants"
CANDIDATE_VARIANTS="${CAMPAIGN_DIR}/candidate_variants"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
AUDIT="${CAMPAIGN_DIR}/branch_audit.jsonl"
BASELINE_RESULTS="${CAMPAIGN_DIR}/baseline_results.jsonl"
CANDIDATE_RESULTS="${CAMPAIGN_DIR}/candidate_results.jsonl"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
mkdir -p "${CAMPAIGN_DIR}"

cleanup_generated_state() {
    if [[ -d "${CAMPAIGN_DIR}" ]]; then
        find "${CAMPAIGN_DIR}" -mindepth 1 -delete
        rmdir -- "${CAMPAIGN_DIR}" 2>/dev/null || true
    fi
    if [[ -d "${ROOT}/build" ]]; then
        find "${ROOT}/build" -mindepth 1 -delete
        rmdir -- "${ROOT}/build" 2>/dev/null || true
    fi
}

on_error() {
    local rc=$?
    local fatal_message diagnostic_tail
    fatal_message="RULE_VALIDATION_FATAL rc=${rc} line=${BASH_LINENO[0]} log=${RUN_LOG}"
    diagnostic_tail="$(tail -40 "${RUN_LOG}" 2>/dev/null || true)"
    printf '%s\n%s\n' "${fatal_message}" "${diagnostic_tail}" >"${RUN_LOG}"
    printf '%s\n' "${fatal_message}" >&3
    printf '%s\n' "${diagnostic_tail}" >&3
    cleanup_generated_state || true
    exit "${rc}"
}
trap on_error ERR

fail() {
    announce "RULE_VALIDATION_FATAL error=$* log=${RUN_LOG}"
    return 1
}

emit_final_results() {
    local final_lines
    final_lines="$(python3 - "${SUMMARY}" "${ANALYSIS}" "${MEASURED_SHAPES}" <<'PY'
import csv
import json
import sys
from collections import defaultdict

with open(sys.argv[1], newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
with open(sys.argv[2], encoding='utf-8') as stream:
    analysis = json.load(stream)
expected = int(sys.argv[3])
if len(rows) != expected:
    raise SystemExit(f'expected {expected} final rows, found {len(rows)}')
print('FINAL_RESULTS_BEGIN')
for row in analysis['branch_audit']:
    print(
        'CORE_BRANCH_AUDIT '
        f"suffix={row['suffix']} branch={row['branch']} "
        f"source_cores={row['source_cores']} required_cores={row['required_cores']} "
        f"action={row['action']} proof={row['proof_kind']}"
    )
by_suffix = defaultdict(list)
for row in rows:
    by_suffix[row['kernel_suffix']].append(row)
    print(
        'FINAL_RESULT '
        f"id={row['workload_id']} suffix={row['kernel_suffix']} branch={row['branch']} "
        f"m={row['m']} n={row['n']} k={row['k']} dtype={row['dtype']} "
        f"trans_a={row['trans_a']} trans_b={row['trans_b']} "
        f"baseline_ms={float(row['baseline_median_ms']):.9g} "
        f"candidate_ms={float(row['candidate_median_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} "
        f"cores_before={row['cores_before']} cores_after={row['cores_after']} "
        f"winner={row['median_winner']} separation={row['sample_separation']}"
    )
for suffix in sorted(by_suffix, key=int):
    group = by_suffix[suffix]
    print(
        'FINAL_BRANCH_RESULT '
        f"suffix={suffix} shapes={len(group)} "
        f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in group)} "
        f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in group)} "
        f"clear_baseline_wins={sum(row['sample_separation'] == 'CLEAR_BASELINE_WINNER' for row in group)} "
        f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in group)}"
    )
aggregate = analysis['aggregate']
print(
    'FINAL_RESULT_SUMMARY '
    f"audited_suffixes={aggregate['audited_suffixes']} "
    f"measured_changed_suffixes={aggregate['measured_changed_suffixes']} "
    f"shapes={aggregate['measured_shapes']} "
    f"candidate_wins={aggregate['candidate_wins']} "
    f"baseline_wins={aggregate['baseline_wins']} "
    f"clear_candidate_wins={aggregate['clear_candidate_wins']} "
    f"clear_baseline_wins={aggregate['clear_baseline_wins']} "
    f"overlap={aggregate['overlap']}"
)
print('FINAL_RESULTS_END')
PY
)"
    printf '%s\n' "${final_lines}" >"${RUN_LOG}"
    printf '%s\n' "${final_lines}" >&3
}

announce "RUN_LOG path=${RUN_LOG}"
announce "SOURCE_REVISION commit=$(git rev-parse HEAD 2>/dev/null || printf unknown)"
announce "CAMPAIGN_READY operator=matmul selector=twelve_suffix_core_ownership audited_suffixes=12 paired_shapes=${MEASURED_SHAPES} compiled_variants=${EXPECTED_VARIANTS} physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID}"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "comparison=same_direct_kernel_same_suffix_same_packet_except_usedCoreNum"
announce "forbidden=cost_model,candidate_enumeration,measured_latency_at_selection,history_lookup,repo_lookup,tiling_bank,installed_host_tiler"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_core_ownership_matrix.py \
    --baseline-dir "${BASELINE_PACKET_DIR}" \
    --candidate-dir "${CANDIDATE_PACKET_DIR}" \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --baseline-variants "${BASELINE_VARIANTS}" \
    --candidate-variants "${CANDIDATE_VARIANTS}" \
    --selection "${SELECTION}" \
    --audit "${AUDIT}"
actual_variants="$(find "${CANDIDATE_VARIANTS}" -maxdepth 1 -type f -name '*.csv' | wc -l)"
[[ "${actual_variants}" -eq "${EXPECTED_VARIANTS}" ]] || \
    fail "generated ${actual_variants} variants; expected ${EXPECTED_VARIANTS}"
announce "CORE_OWNERSHIP_PACKET_GENERATION passed audited_suffixes=12 paired_shapes=${MEASURED_SHAPES} variants=${EXPECTED_VARIANTS}"
announce "CAMPAIGN_STAGE_TIMING stage=packet_generation wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))"

python3 - "${CANDIDATE_MANIFEST}" "${NUMERIC_PREFLIGHT_MAX_MIB}" <<'PY'
import csv
import sys
width = {'fp16': 2, 'bf16': 2, 'fp32': 4}
with open(sys.argv[1], newline='', encoding='utf-8') as stream:
    rows = list(csv.DictReader(stream))
limit = int(sys.argv[2]) * 1024 * 1024
largest = max(((int(row['m']) * int(row['k']) + int(row['k']) * int(row['n'])) * width[row['dtype']], row['workload_id']) for row in rows)
if largest[0] > limit:
    raise SystemExit(f'input cap exceeded: {largest}')
print(f"INPUT_CAP_AUDIT passed largest={largest[1]} bytes={largest[0]} limit={limit}")
PY

python3 - "${DEVICE_ID}" <<'PY'
import ctypes
import sys
device = int(sys.argv[1])
acl = ctypes.CDLL('libascendcl.so', mode=ctypes.RTLD_GLOBAL)
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclInit.restype = ctypes.c_int
acl.aclFinalize.restype = ctypes.c_int
acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
acl.aclrtSetDevice.restype = ctypes.c_int
acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
acl.aclrtResetDevice.restype = ctypes.c_int
initialized = False
device_set = False
try:
    rc = acl.aclInit(None)
    if rc: raise RuntimeError(f'aclInit failed rc={rc}')
    initialized = True
    rc = acl.aclrtSetDevice(device)
    if rc: raise RuntimeError(f'aclrtSetDevice failed rc={rc} user_device={device}')
    device_set = True
    print(f'DEVICE_PREFLIGHT passed user_device={device}')
finally:
    if device_set: acl.aclrtResetDevice(device)
    if initialized: acl.aclFinalize()
PY

build_started_ns="$(date +%s%N)"
variant_index=0
for candidate_variant in "${CANDIDATE_VARIANTS}"/*.csv; do
    filename="$(basename "${candidate_variant}" .csv)"
    variant="${filename##*__}"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    baseline_variant="${BASELINE_VARIANTS}/${filename}.csv"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    variant_index=$((variant_index + 1))
    announce "DIRECT_VARIANT_BUILD ${variant_index}/${EXPECTED_VARIANTS} begin variant=${variant} jobs=1"
    BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" scripts/build_all.sh
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    [[ -x "${runner}" ]] || fail "direct runner missing: ${runner}"
    "${runner}" --manifest "${baseline_variant}" --validate-input >/dev/null
    "${runner}" --manifest "${candidate_variant}" --validate-input >/dev/null
    announce "DIRECT_VARIANT_BUILD ${variant_index}/${EXPECTED_VARIANTS} passed variant=${variant}"
done
[[ "${variant_index}" -eq "${EXPECTED_VARIANTS}" ]] || fail "variant build count mismatch"
announce "CAMPAIGN_STAGE_TIMING stage=variant_build wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))"

: >"${BASELINE_RESULTS}"
: >"${CANDIDATE_RESULTS}"
measurement_started_ns="$(date +%s%N)"
variant_index=0
for candidate_variant in "${CANDIDATE_VARIANTS}"/*.csv; do
    filename="$(basename "${candidate_variant}" .csv)"
    variant="${filename##*__}"
    baseline_variant="${BASELINE_VARIANTS}/${filename}.csv"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    variant_index=$((variant_index + 1))
    shapes=$(( $(wc -l <"${candidate_variant}") - 1 ))
    announce "PAIRED_CANARY ${variant_index}/${EXPECTED_VARIANTS} begin variant=${variant} shapes=${shapes}"
    "${runner}" --manifest "${baseline_variant}" --device "${DEVICE_ID}" --warmup 0 --repeat 1 --samples 1 >/dev/null
    "${runner}" --manifest "${candidate_variant}" --device "${DEVICE_ID}" --warmup 0 --repeat 1 --samples 1 >/dev/null
    announce "PAIRED_CANARY ${variant_index}/${EXPECTED_VARIANTS} passed variant=${variant}"
    "${runner}" --manifest "${baseline_variant}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" \
        >>"${BASELINE_RESULTS}" 2>&1
    "${runner}" --manifest "${candidate_variant}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" \
        >>"${CANDIDATE_RESULTS}" 2>&1
    announce "PAIRED_MEASUREMENT ${variant_index}/${EXPECTED_VARIANTS} passed variant=${variant} shapes=${shapes}"
done
announce "CAMPAIGN_STAGE_TIMING stage=paired_direct_measurement wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))"

python3 tools/analyze_matmul_core_ownership.py \
    --baseline-manifest "${BASELINE_MANIFEST}" \
    --candidate-manifest "${CANDIDATE_MANIFEST}" \
    --baseline-log "${BASELINE_RESULTS}" \
    --candidate-log "${CANDIDATE_RESULTS}" \
    --selection "${SELECTION}" \
    --audit "${AUDIT}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"

emit_final_results
cleanup_generated_state
printf '%s\n' "TWELVE_SUFFIX_CORE_VALIDATION_COMPLETE log=${RUN_LOG}" >&3
