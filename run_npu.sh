#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=10
REPEAT=20
SAMPLES=30
EXPECTED_BRANCHES=13
EXPECTED_AUDIT_ITEMS=33
VALIDATION_SHAPES=62
EXPECTED_VARIANTS=0

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Audits every installed CANN 8.1 candidate family, then measures 38' \
        'independent rule winners, 13 branch probes, 11 structural ablations, and one' \
        'same-campaign official MatMulV3 reference for every test shape.'
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
        tools/audit_matmul_candidate_engine.py \
        tools/direct_matmul_tiling.py \
        scripts/build_all.sh \
        scripts/env.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        matmul_rule_selector/candidate_audit_spec.json \
        matmul_rule_selector/family_candidate_contract.json \
        matmul_rule_selector/validation_contract.json
    find matmul_rule_selector -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
    find npu_cost_model -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_rule_selector_paired_v3/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/improved_manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
RUNNER_LOG="${RUN_LOG}"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
mkdir -p "${CAMPAIGN_DIR}"

emit_final_results() {
    local final_lines
    [[ -s "${SUMMARY}" ]] || {
        announce "FINAL_RESULTS_MISSING summary=${SUMMARY}"
        return 1
    }
    final_lines="$(python3 - "${SUMMARY}" "${VALIDATION_SHAPES}" <<'PY'
import csv
import sys
from collections import defaultdict

with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
expected = int(sys.argv[2])
if len(rows) != expected:
    raise SystemExit(f"expected {expected} final rows, found {len(rows)}")
print("FINAL_RESULTS_BEGIN")
by_axis = defaultdict(list)
by_role = defaultdict(list)
for row in rows:
    by_axis[row["selection_axis"]].append(row)
    by_role[row["case_role"]].append(row)
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} "
        f"axis={row['selection_axis']} "
        f"role={row['case_role']} "
        f"applicable={row['required_applicable_family']} "
        f"selected={row['selected_family']} "
        f"official_ms={float(row['official_median_ms']):.9g} "
        f"candidate_ms={float(row['candidate_median_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} "
        f"winner={row['median_winner']} "
        f"separation={row['sample_separation']}"
    )
for axis in sorted(by_axis):
    axis_rows = by_axis[axis]
    print(
        "FINAL_RULE_RESULT "
        f"axis={axis} "
        f"shapes={len(axis_rows)} "
        f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in axis_rows)} "
        f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in axis_rows)} "
        f"clear_official_wins={sum(row['sample_separation'] == 'CLEAR_OFFICIAL_WINNER' for row in axis_rows)} "
        f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in axis_rows)}"
    )
for role in sorted(by_role):
    role_rows = by_role[role]
    print(
        "FINAL_ROLE_SUMMARY "
        f"role={role} "
        f"shapes={len(role_rows)} "
        f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in role_rows)} "
        f"official_wins={sum(row['median_winner'] == 'official' for row in role_rows)} "
        f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in role_rows)} "
        f"clear_official_wins={sum(row['sample_separation'] == 'CLEAR_OFFICIAL_WINNER' for row in role_rows)} "
        f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in role_rows)}"
    )
selector_rows = by_role["selector_top1"]
print(
    "FINAL_SELECTOR_SUMMARY "
    f"shapes={len(selector_rows)} "
    f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in selector_rows)} "
    f"official_wins={sum(row['median_winner'] == 'official' for row in selector_rows)} "
    f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in selector_rows)} "
    f"clear_official_wins={sum(row['sample_separation'] == 'CLEAR_OFFICIAL_WINNER' for row in selector_rows)} "
    f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in selector_rows)}"
)
print(
    "FINAL_RESULT_SUMMARY "
    f"shapes={len(rows)} "
    f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in rows)} "
    f"official_wins={sum(row['median_winner'] == 'official' for row in rows)} "
    f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in rows)} "
    f"clear_official_wins={sum(row['sample_separation'] == 'CLEAR_OFFICIAL_WINNER' for row in rows)} "
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
source_revision="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
announce "SOURCE_REVISION commit=${source_revision}"
announce "CAMPAIGN_READY operator=matmul focus=all_installed_family_branches_and_boundaries host_branches=${EXPECTED_BRANCHES} audit_items=${EXPECTED_AUDIT_ITEMS} npu_shapes=${VALIDATION_SHAPES} selector_top1_measurements=38 branch_probe_measurements=13 structural_probe_measurements=11 official_measurements=${VALIDATION_SHAPES} measurement_batches=derived_from_complete_manifest compiled_variants=all_manifest_dtype_suffix_pairs physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID}"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "selection=all_applicable_installed_families_then_hard_legality_then_protocol_specific_hardware_rules_then_critical_path_tiebreak"
announce "selector=shape_and_frozen_hardware_only"
announce "official_reference=same_campaign_installed_aclnn_matmul_public_api"
announce "forbidden=measured_latency_at_selection,history_lookup_at_runtime,repo_lookup,tiling_bank,official_tiling_seed"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20 visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
announce "results=${CAMPAIGN_DIR}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status": "complete"' "${ANALYSIS}"; then
    emit_final_results
    announce "PAIRED_VALIDATION_COMPLETE cached=1 analysis=${ANALYSIS} summary=${SUMMARY} log=${RUN_LOG}"
    exit 0
fi

device_preflight_started_ns="$(date +%s%N)"
if python3 - "${DEVICE_ID}" <<'PY'
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

loaded_paths = []
try:
    with open("/proc/self/maps", encoding="utf-8") as stream:
        loaded_paths = sorted({
            line.split()[-1]
            for line in stream
            if "libascendcl.so" in line and line.split()[-1].startswith("/")
        })
except OSError:
    pass
print(
    "DEVICE_PREFLIGHT_LIBRARY "
    f"ascendcl={','.join(loaded_paths) if loaded_paths else 'unresolved'}"
)

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
then
    device_preflight_rc=0
else
    device_preflight_rc=$?
fi
device_preflight_wall_ms=$(( ($(date +%s%N) - device_preflight_started_ns) / 1000000 ))
if [[ "${device_preflight_rc}" -ne 0 ]]; then
    printf '%s\n' 'DEVICE_DIAGNOSTICS_BEGIN'
    printf 'uid='; id
    printf 'kernel='; uname -a
    printf 'cann_root=%s\n' "${CANN_ROOT}"
    printf 'visible_devices=%s runtime_user_device=%s physical_device=%s\n' \
        "${ASCEND_RT_VISIBLE_DEVICES}" "${DEVICE_ID}" "${PHYSICAL_DEVICE}"
    for version_path in \
        /usr/local/Ascend/driver/version.info \
        /usr/local/Ascend/driver/version.cfg \
        /etc/ascend_install.info; do
        if [[ -r "${version_path}" ]]; then
            printf 'VERSION_FILE path=%s\n' "${version_path}"
            sed -n '1,80p' "${version_path}"
        fi
    done
    ls -l \
        /dev/davinci_manager \
        "/dev/davinci${PHYSICAL_DEVICE}" \
        /dev/devmm_svm \
        /dev/hisi_hdc 2>&1 || true
    npu_smi="$(command -v npu-smi || true)"
    if [[ -z "${npu_smi}" && -x /usr/local/Ascend/driver/tools/npu-smi ]]; then
        npu_smi=/usr/local/Ascend/driver/tools/npu-smi
    fi
    if [[ -n "${npu_smi}" ]]; then
        printf 'NPU_SMI path=%s\n' "${npu_smi}"
        "${npu_smi}" info 2>&1 || true
    else
        printf '%s\n' 'NPU_SMI unavailable'
    fi
    printf '%s\n' 'DEVICE_DIAGNOSTICS_END'
    fail "device preflight failed rc=${device_preflight_rc}; no build or measurement was started"
fi
announce "DEVICE_PREFLIGHT passed physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID} wall_ms=${device_preflight_wall_ms}"

audit_started_ns="$(date +%s%N)"
audit_json="$(python3 tools/audit_matmul_candidate_engine.py)"
python3 -c 'import json,sys; x=json.loads(sys.argv[1]); assert x["status"] == "PASS" and x["mandatory_count"] == int(sys.argv[2]) and x["passed_count"] == int(sys.argv[2]) and not x["failed_ids"]' "${audit_json}" "${EXPECTED_AUDIT_ITEMS}"
printf '%s\n' "${audit_json}"
audit_wall_ms=$(( ($(date +%s%N) - audit_started_ns) / 1000000 ))
announce "HOST_CANDIDATE_AUDIT passed items=${EXPECTED_AUDIT_ITEMS} branches=${EXPECTED_BRANCHES}"
announce "CAMPAIGN_STAGE_TIMING stage=host_candidate_audit wall_ms=${audit_wall_ms}"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_rule_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}"
EXPECTED_VARIANTS="$(find "${VARIANT_DIR}" -maxdepth 1 -type f -name '*.csv' | wc -l)"
[[ "${EXPECTED_VARIANTS}" -ge 12 ]] || \
    fail "generated ${EXPECTED_VARIANTS} dtype/suffix variants; expected at least 12"
printf '%s\n' 'SELECTION_RECORDS_BEGIN'
sed 's/^/SELECTION_RECORD /' "${SELECTION}"
printf '%s\n' 'SELECTION_RECORDS_END'
printf '%s\n' 'CANDIDATE_MANIFEST_CSV_BEGIN'
cat "${MANIFEST}"
printf '%s\n' 'CANDIDATE_MANIFEST_CSV_END'
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
announce "RULE_WINNER_AND_PROBE_PACKET_GENERATION passed shapes=${VALIDATION_SHAPES} variants=${EXPECTED_VARIANTS}"
announce "CAMPAIGN_STAGE_TIMING stage=rule_winner_and_probe_packet_generation wall_ms=${generation_wall_ms}"

official_build_started_ns="$(date +%s%N)"
announce "OFFICIAL_RUNNER_BUILD begin jobs=1"
if ! BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh; then
    fail "official runner build failed"
fi
official_runner="${ROOT}/build/official_matmul_runner"
[[ -x "${official_runner}" ]] || fail "official runner missing after build: ${official_runner}"
"${official_runner}" --candidates "${MANIFEST}" --validate-input >/dev/null
official_build_wall_ms=$(( ($(date +%s%N) - official_build_started_ns) / 1000000 ))
announce "OFFICIAL_RUNNER_BUILD passed"
announce "CAMPAIGN_STAGE_TIMING stage=official_runner_build wall_ms=${official_build_wall_ms}"

build_started_ns="$(date +%s%N)"
variant_count=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    variant_count=$((variant_count + 1))
    announce "DIRECT_VARIANT_BUILD ${variant_count}/${EXPECTED_VARIANTS} begin variant=${variant} jobs=1"
    if ! BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" \
        scripts/build_all.sh; then
        fail "direct variant build failed: variant=${variant}"
    fi
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    [[ -x "${runner}" ]] || fail "direct runner missing after build: ${runner}"
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    announce "DIRECT_VARIANT_BUILD ${variant_count}/${EXPECTED_VARIANTS} passed variant=${variant}"
done
[[ "${variant_count}" -eq "${EXPECTED_VARIANTS}" ]] || \
    fail "generated ${variant_count} variants, expected ${EXPECTED_VARIANTS}"
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=selected_variant_build wall_ms=${build_wall_ms}"

measurement_started_ns="$(date +%s%N)"
announce "OFFICIAL_MEASUREMENT begin shapes=${VALIDATION_SHAPES}"
"${official_runner}" \
    --candidates "${MANIFEST}" \
    --output "${OFFICIAL_PROFILE}" \
    --samples-output "${OFFICIAL_SAMPLES}" \
    --device "${DEVICE_ID}" \
    --warmup "${WARMUP}" \
    --repeat "${REPEAT}" \
    --samples "${SAMPLES}" \
    --numeric-preflight-max-mib 256 \
    --structured-full-preflight \
    --validate-after-measurement
announce "OFFICIAL_MEASUREMENT passed shapes=${VALIDATION_SHAPES}"
printf '%s\n' 'OFFICIAL_PROFILE_CSV_BEGIN'
cat "${OFFICIAL_PROFILE}"
printf '%s\n' 'OFFICIAL_PROFILE_CSV_END'
printf '%s\n' 'OFFICIAL_SAMPLES_CSV_BEGIN'
cat "${OFFICIAL_SAMPLES}"
printf '%s\n' 'OFFICIAL_SAMPLES_CSV_END'

batch_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    batch_index=$((batch_index + 1))
    batch_shapes=$(( $(wc -l <"${packet_manifest}") - 1 ))
    announce "CANDIDATE_MEASUREMENT_BATCH ${batch_index}/${EXPECTED_VARIANTS} begin variant=${variant} shapes=${batch_shapes} manifest=$(basename "${packet_manifest}")"
    "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --samples "${SAMPLES}"
    announce "CANDIDATE_MEASUREMENT_BATCH ${batch_index}/${EXPECTED_VARIANTS} passed variant=${variant} shapes=${batch_shapes}"
done
[[ "${batch_index}" -eq "${EXPECTED_VARIANTS}" ]] || \
    fail "measured ${batch_index} batches, expected ${EXPECTED_VARIANTS}"
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=paired_npu_measurement wall_ms=${measurement_wall_ms}"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_rule_matrix.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUNNER_LOG}" \
    --official-profile "${OFFICIAL_PROFILE}" \
    --official-samples "${OFFICIAL_SAMPLES}" \
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
announce "PAIRED_VALIDATION_COMPLETE cached=0 analysis=${ANALYSIS} summary=${SUMMARY} packets=${PACKET_DIR} log=${RUN_LOG}"
