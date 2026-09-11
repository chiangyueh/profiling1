#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=2
REPEAT=20
SAMPLES=7
EXPECTED_BRANCHES=12
EXPECTED_RULE_GROUPS=14
VALIDATION_SHAPES=5

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Checks all CANN 8.1 rule branches on the host, then measures only five' \
        'structurally discriminative improved packets. Their audited existing' \
        'public MatMulV3 measurements are reused; official baselines are not rerun.'
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
        tools/generate_matmul_rule_matrix.py \
        tools/analyze_matmul_rule_matrix.py \
        tools/check_matmul_rule_coverage.py \
        tools/direct_matmul_tiling.py \
        scripts/build_all.sh \
        scripts/env.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        matmul_rule_selector/branch_contract.json \
        matmul_rule_selector/validation_contract.json
    find matmul_rule_selector -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_rule_discriminative_v3/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/improved_manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
RUNNER_LOG="${RUN_LOG}"
REFERENCE_PROFILE="${CAMPAIGN_DIR}/historical_official_profile.csv"
REFERENCE_SAMPLES="${CAMPAIGN_DIR}/historical_official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
mkdir -p "${CAMPAIGN_DIR}"

emit_final_results() {
    local final_lines
    [[ -s "${SUMMARY}" ]] || {
        announce "FINAL_RESULTS_MISSING summary=${SUMMARY}"
        return 1
    }
    final_lines="$(python3 - "${SUMMARY}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 5:
    raise SystemExit(f"expected 5 final rows, found {len(rows)}")
print("FINAL_RESULTS_BEGIN")
for row in rows:
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} "
        f"axis={row['selection_axis']} "
        f"official_ms={float(row['historical_official_median_ms']):.9g} "
        f"improved_ms={float(row['current_improved_median_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} "
        f"winner={row['median_winner']} "
        f"separation={row['sample_separation']}"
    )
print(
    "FINAL_RESULT_SUMMARY "
    f"shapes={len(rows)} "
    f"improved_wins={sum(row['median_winner'] == 'improved' for row in rows)} "
    f"official_wins={sum(row['median_winner'] == 'official' for row in rows)} "
    f"clear_improvements={sum(row['sample_separation'] == 'CLEAR_IMPROVED' for row in rows)} "
    f"clear_regressions={sum(row['sample_separation'] == 'CLEAR_REGRESSION' for row in rows)} "
    f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in rows)}"
)
print("FINAL_RESULTS_END")
PY
    )"
    while IFS= read -r result_line; do
        announce "${result_line}"
    done <<<"${final_lines}"
}

on_error() {
    local rc=$?
    announce "RULE_VALIDATION_FATAL rc=${rc} line=${BASH_LINENO[0]} results=${CAMPAIGN_DIR} log=${RUN_LOG}"
    tail -40 "${RUN_LOG}" >&3 || true
    exit "${rc}"
}
trap on_error ERR

fail() {
    echo "fatal: $*"
    announce "RULE_VALIDATION_FATAL rc=1 results=${CAMPAIGN_DIR} log=${RUN_LOG}"
    tail -40 "${RUN_LOG}" >&3 || true
    exit 1
}

announce "RUN_LOG path=${RUN_LOG}"
announce "CAMPAIGN_READY operator=matmul host_branches=${EXPECTED_BRANCHES} rule_groups=${EXPECTED_RULE_GROUPS} npu_shapes=${VALIDATION_SHAPES} improved_measurements=${VALIDATION_SHAPES} reused_official_references=${VALIDATION_SHAPES} physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID}"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "selection=predeclared_structural_decision_boundaries_not_latency_ranked"
announce "selector=shape_and_hardware_formula_only"
announce "official_reference=audited_existing_cann81_measurements_not_rerun"
announce "forbidden=cost_model,history_lookup_at_runtime,repo_lookup,tiling_bank,candidate_search,official_tiling_seed"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20 visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
announce "results=${CAMPAIGN_DIR}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status": "complete"' "${ANALYSIS}"; then
    emit_final_results
    announce "DISCRIMINATIVE_COMPLETE cached=1 analysis=${ANALYSIS} summary=${SUMMARY} log=${RUN_LOG}"
    exit 0
fi

device_preflight_started_ns="$(date +%s%N)"
python3 - "${DEVICE_ID}" <<'PY'
import ctypes
import os
import sys

device = int(sys.argv[1])
acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclInit.restype = ctypes.c_int
acl.aclFinalize.argtypes = []
acl.aclFinalize.restype = ctypes.c_int
acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
acl.aclrtGetDeviceCount.restype = ctypes.c_int
acl.aclrtSetDevice.argtypes = [ctypes.c_int32]
acl.aclrtSetDevice.restype = ctypes.c_int
acl.aclrtResetDevice.argtypes = [ctypes.c_int32]
acl.aclrtResetDevice.restype = ctypes.c_int

initialized = False
device_set = False
try:
    rc = acl.aclInit(None)
    if rc != 0:
        raise RuntimeError(f"aclInit failed rc={rc}")
    initialized = True
    count = ctypes.c_uint32()
    rc = acl.aclrtGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        raise RuntimeError(f"aclrtGetDeviceCount failed rc={rc}")
    if device < 0 or device >= count.value:
        raise RuntimeError(
            f"runtime user device {device} is outside available range 0..{count.value - 1}"
        )
    rc = acl.aclrtSetDevice(device)
    if rc != 0:
        raise RuntimeError(
            f"aclrtSetDevice failed rc={rc} user_device={device} "
            f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')} "
            f"available_user_devices={count.value}"
        )
    device_set = True
    print(
        "DEVICE_PREFLIGHT_DETAIL status=passed "
        f"user_device={device} available_user_devices={count.value} "
        f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}"
    )
finally:
    if device_set:
        acl.aclrtResetDevice(device)
    if initialized:
        acl.aclFinalize()
PY
device_preflight_wall_ms=$(( ($(date +%s%N) - device_preflight_started_ns) / 1000000 ))
announce "DEVICE_PREFLIGHT passed physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID} wall_ms=${device_preflight_wall_ms}"

coverage_started_ns="$(date +%s%N)"
coverage_json="$(python3 tools/check_matmul_rule_coverage.py)"
python3 -c 'import json,sys; x=json.loads(sys.argv[1]); assert x["installed_cann81_dispatch_branches"] == 12 and x["modified_rule_groups"] == 14 and x["modified_branch_witnesses"] == 12 and x["status"] == "PASS"' "${coverage_json}"
printf '%s\n' "${coverage_json}"
coverage_wall_ms=$(( ($(date +%s%N) - coverage_started_ns) / 1000000 ))
announce "HOST_RULE_COVERAGE passed branches=${EXPECTED_BRANCHES} rule_groups=${EXPECTED_RULE_GROUPS}"
announce "CAMPAIGN_STAGE_TIMING stage=host_rule_coverage wall_ms=${coverage_wall_ms}"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_rule_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}" \
    --historical-profile "${REFERENCE_PROFILE}" \
    --historical-samples "${REFERENCE_SAMPLES}"
printf '%s\n' 'SELECTION_RECORDS_BEGIN'
sed 's/^/SELECTION_RECORD /' "${SELECTION}"
printf '%s\n' 'SELECTION_RECORDS_END'
printf '%s\n' 'IMPROVED_MANIFEST_CSV_BEGIN'
cat "${MANIFEST}"
printf '%s\n' 'IMPROVED_MANIFEST_CSV_END'
printf '%s\n' 'HISTORICAL_OFFICIAL_PROFILE_CSV_BEGIN'
cat "${REFERENCE_PROFILE}"
printf '%s\n' 'HISTORICAL_OFFICIAL_PROFILE_CSV_END'
printf '%s\n' 'HISTORICAL_OFFICIAL_SAMPLES_CSV_BEGIN'
cat "${REFERENCE_SAMPLES}"
printf '%s\n' 'HISTORICAL_OFFICIAL_SAMPLES_CSV_END'
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
announce "DISCRIMINATIVE_PACKET_GENERATION passed shapes=${VALIDATION_SHAPES}"
announce "CAMPAIGN_STAGE_TIMING stage=discriminative_packet_generation wall_ms=${generation_wall_ms}"

build_started_ns="$(date +%s%N)"
variant_count=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    variant_count=$((variant_count + 1))
    announce "DIRECT_VARIANT_BUILD ${variant_count}/${VALIDATION_SHAPES} begin variant=${variant} jobs=1"
    if ! BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" \
        scripts/build_all.sh; then
        fail "direct variant build failed: variant=${variant}"
    fi
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    [[ -x "${runner}" ]] || fail "direct runner missing after build: ${runner}"
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    announce "DIRECT_VARIANT_BUILD ${variant_count}/${VALIDATION_SHAPES} passed variant=${variant}"
done
[[ "${variant_count}" -eq "${VALIDATION_SHAPES}" ]] || \
    fail "generated ${variant_count} variants, expected ${VALIDATION_SHAPES}"
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=selected_variant_build wall_ms=${build_wall_ms}"

measurement_started_ns="$(date +%s%N)"
packet_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    packet_index=$((packet_index + 1))
    announce "IMPROVED_MEASUREMENT ${packet_index}/${VALIDATION_SHAPES} begin variant=${variant} manifest=$(basename "${packet_manifest}")"
    "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --samples "${SAMPLES}"
    announce "IMPROVED_MEASUREMENT ${packet_index}/${VALIDATION_SHAPES} passed variant=${variant}"
done
[[ "${packet_index}" -eq "${VALIDATION_SHAPES}" ]] || \
    fail "measured ${packet_index} improved packets, expected ${VALIDATION_SHAPES}"
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=improved_measurement wall_ms=${measurement_wall_ms}"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_rule_matrix.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUNNER_LOG}" \
    --reference-profile "${REFERENCE_PROFILE}" \
    --reference-samples "${REFERENCE_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"
printf '%s\n' 'FINAL_ANALYSIS_JSON_BEGIN'
cat "${ANALYSIS}"
printf '%s\n' 'FINAL_ANALYSIS_JSON_END'
printf '%s\n' 'FINAL_SUMMARY_CSV_BEGIN'
cat "${SUMMARY}"
printf '%s\n' 'FINAL_SUMMARY_CSV_END'
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
emit_final_results
announce "DISCRIMINATIVE_OUTPUT analysis=${ANALYSIS} summary=${SUMMARY} packets=${PACKET_DIR} log=${RUN_LOG}"
