#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=3
REPEAT=10
SAMPLES=15
VALIDATION_SHAPES=100
EXPECTED_VARIANTS=1
NUMERIC_PREFLIGHT_MAX_MIB=64

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Audits the complete source family chain, then measures exactly one' \
        'closed-form improved tiling and one official reference per workload.'
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
        tools/generate_matmul_unique_formula_matrix.py \
        tools/analyze_matmul_rule_matrix.py \
        tools/direct_matmul_tiling.py \
        scripts/build_all.sh \
        scripts/env.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        matmul_rule_selector/formula_rules.py \
        matmul_rule_selector/improved_selector.py \
        matmul_rule_selector/source_family_audit_contract.json \
        matmul_rule_selector/unique_formula_validation_contract.json
    find matmul_rule_selector/baseline_core -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_al1_instruction_deletion_v1/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/improved_manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
THEORY_AUDIT="${CAMPAIGN_DIR}/theory_audit.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
RUNNER_LOG="${CAMPAIGN_DIR}/direct_results.jsonl"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
mkdir -p "${CAMPAIGN_DIR}"

cleanup_generated_state() {
    local generated_file generated_dir
    for generated_file in \
        "${MANIFEST}" "${SELECTION}" "${THEORY_AUDIT}" "${RUNNER_LOG}" \
        "${OFFICIAL_PROFILE}" "${OFFICIAL_SAMPLES}" "${ANALYSIS}"; do
        [[ ! -e "${generated_file}" ]] || rm -f -- "${generated_file}"
    done
    for generated_dir in "${PACKET_DIR}" "${VARIANT_DIR}" "${SEQUENCE_DIR}"; do
        if [[ -d "${generated_dir}" ]]; then
            find "${generated_dir}" -mindepth 1 -delete
            rmdir -- "${generated_dir}" 2>/dev/null || true
        fi
    done
}

cleanup_build_state() {
    if [[ -d "${ROOT}/build" ]]; then
        find "${ROOT}/build" -mindepth 1 -delete
        rmdir -- "${ROOT}/build" 2>/dev/null || true
    fi
}

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
for row in rows:
    by_axis[row["selection_axis"]].append(row)
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} "
        f"m={row['m']} n={row['n']} k={row['k']} "
        f"dtype={row['dtype']} trans_a={row['trans_a']} trans_b={row['trans_b']} "
        f"axis={row['selection_axis']} "
        f"official_ms={float(row['official_median_ms']):.9g} "
        f"candidate_ms={float(row['candidate_median_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} "
        f"cores_before={row['source_launched_aic']} "
        f"cores_after={row['candidate_launched_aic']} "
        f"eliminated_idle_aic={row['eliminated_idle_aic']} "
        f"eliminated_a_copy_bytes={row['eliminated_full_a_copy_bytes']} "
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
    printf '%s\n' "${final_lines}" >"${RUN_LOG}"
    printf '%s\n' "${final_lines}" >&3
}

on_error() {
    local rc=$?
    local fatal_message diagnostic_tail
    fatal_message="RULE_VALIDATION_FATAL rc=${rc} line=${BASH_LINENO[0]} results=${CAMPAIGN_DIR} log=${RUN_LOG}"
    diagnostic_tail="$(tail -40 "${RUN_LOG}" 2>/dev/null || true)"
    printf '%s\n%s\n' "${fatal_message}" "${diagnostic_tail}" >"${RUN_LOG}"
    printf '%s\n' "${fatal_message}" >&3
    printf '%s\n' "${diagnostic_tail}" >&3
    cleanup_generated_state || true
    rm -f -- "${SUMMARY}" || true
    cleanup_build_state || true
    exit "${rc}"
}
trap on_error ERR

fail() {
    local fatal_message diagnostic_tail
    fatal_message="RULE_VALIDATION_FATAL rc=1 error=$* results=${CAMPAIGN_DIR} log=${RUN_LOG}"
    diagnostic_tail="$(tail -40 "${RUN_LOG}" 2>/dev/null || true)"
    printf '%s\n%s\n' "${fatal_message}" "${diagnostic_tail}" >"${RUN_LOG}"
    printf '%s\n' "${fatal_message}" >&3
    printf '%s\n' "${diagnostic_tail}" >&3
    cleanup_generated_state || true
    rm -f -- "${SUMMARY}" || true
    cleanup_build_state || true
    exit 1
}

announce "RUN_LOG path=${RUN_LOG}"
source_revision="$(git rev-parse HEAD 2>/dev/null || printf unknown)"
announce "SOURCE_REVISION commit=${source_revision}"
announce "CAMPAIGN_READY operator=matmul selector=al1_idle_aic_copy_elimination random_seed=910813 npu_shapes=${VALIDATION_SHAPES} candidate_measurements=${VALIDATION_SHAPES} official_measurements=${VALIDATION_SHAPES} compiled_variants=${EXPECTED_VARIANTS} physical_device=${PHYSICAL_DEVICE} runtime_user_device=${DEVICE_ID}"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "numeric_preflight_limit_mib=${NUMERIC_PREFLIGHT_MAX_MIB}"
announce "selection=one_proof_carrying_al1_packet_per_shape_no_cross_family_ranking"
announce "selector=shape_and_frozen_910b3_hardware_instruction_deletion_only"
announce "official_reference=same_campaign_installed_aclnn_matmul_public_api"
announce "forbidden=cost_model,measured_latency_at_selection,history_lookup_at_runtime,repo_lookup,tiling_bank,candidate_enumeration,pareto,installed_host_tiler"
announce "unmodified_paths=reported_as_baseline_equivalent_and_excluded_from_improved_measurement"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20 visible_devices=${ASCEND_RT_VISIBLE_DEVICES}"
announce "results=${CAMPAIGN_DIR}"

if [[ -s "${SUMMARY}" ]] && [[ "$(( $(wc -l <"${SUMMARY}") - 1 ))" -eq "${VALIDATION_SHAPES}" ]]; then
    cleanup_generated_state
    cleanup_build_state
    emit_final_results
    printf '%s\n' "CERTIFIED_AL1_VALIDATION_COMPLETE cached=1 summary=${SUMMARY} log=${RUN_LOG}" >&3
    exit 0
fi
: >"${RUNNER_LOG}"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_unique_formula_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --theory-audit "${THEORY_AUDIT}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}"
actual_variants="$(find "${VARIANT_DIR}" -maxdepth 1 -type f -name '*.csv' | wc -l)"
[[ "${actual_variants}" -eq "${EXPECTED_VARIANTS}" ]] || \
    fail "generated ${actual_variants} dtype/suffix variants; expected ${EXPECTED_VARIANTS}"
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
announce "CERTIFIED_AL1_PACKET_GENERATION passed shapes=${VALIDATION_SHAPES} source_families=7 source_suffixes=12 variants=${EXPECTED_VARIANTS} complete_tilings_per_shape=1 packet_bytes=272 proof=AL1_IDLE_AIC_FULL_A_COPY_ELIMINATION"
announce "CAMPAIGN_STAGE_TIMING stage=unique_formula_packet_generation wall_ms=${generation_wall_ms}"

input_cap_audit="$(python3 - "${MANIFEST}" "${NUMERIC_PREFLIGHT_MAX_MIB}" <<'PY'
import csv
import sys

width = {"fp16": 2, "bf16": 2, "fp32": 4}
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
limit = int(sys.argv[2]) * 1024 * 1024
sizes = {
    row["workload_id"]: (
        int(row["m"]) * int(row["k"]) + int(row["k"]) * int(row["n"])
    ) * width[row["dtype"]]
    for row in rows
}
largest = max(sizes, key=sizes.get)
max_k = max(int(row["k"]) for row in rows)
if sizes[largest] > limit or max_k > 60000:
    raise SystemExit(
        f"input preflight contract invalid: largest={largest} "
        f"bytes={sizes[largest]} limit={limit} max_k={max_k}"
    )
print(
    f"largest={largest} bytes={sizes[largest]} "
    f"mib={sizes[largest] / 1048576:g} limit_mib={limit / 1048576:g} max_k={max_k}"
)
PY
)"
announce "INPUT_CAP_AUDIT passed ${input_cap_audit}"

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
    --numeric-preflight-max-mib "${NUMERIC_PREFLIGHT_MAX_MIB}" \
    --structured-full-preflight \
    --validate-after-measurement
announce "OFFICIAL_MEASUREMENT passed shapes=${VALIDATION_SHAPES}"

batch_index=0
candidate_batch_failures=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    batch_index=$((batch_index + 1))
    batch_shapes=$(( $(wc -l <"${packet_manifest}") - 1 ))
    announce "CANDIDATE_CANARY ${batch_index}/${EXPECTED_VARIANTS} begin variant=${variant} shapes=${batch_shapes}"
    if canary_output="$("${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup 0 \
        --repeat 1 \
        --samples 1 \
        --allow-partial 2>&1)"; then
        canary_rc=0
    else
        canary_rc=$?
    fi
    canary_invalid="$(grep -c 'DIRECT_MATMUL_RESULT .*"status":"failed"' <<<"${canary_output}" || true)"
    if [[ "${canary_rc}" -ne 0 || "${canary_invalid}" -ne 0 ]]; then
        printf '%s\n' "${canary_output}" | sed 's/^/CERTIFIED_AL1_CANARY_RECORD /'
        candidate_batch_failures=$((candidate_batch_failures + 1))
        announce "CANDIDATE_CANARY ${batch_index}/${EXPECTED_VARIANTS} failed variant=${variant} rc=${canary_rc} invalid_shapes=${canary_invalid}; formal_measurement=skipped"
        continue
    fi
    announce "CANDIDATE_CANARY ${batch_index}/${EXPECTED_VARIANTS} passed variant=${variant} shapes=${batch_shapes}"
    announce "CANDIDATE_MEASUREMENT_BATCH ${batch_index}/${EXPECTED_VARIANTS} begin variant=${variant} shapes=${batch_shapes} manifest=$(basename "${packet_manifest}")"
    if "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --samples "${SAMPLES}" >>"${RUNNER_LOG}" 2>&1; then
        announce "CANDIDATE_MEASUREMENT_BATCH ${batch_index}/${EXPECTED_VARIANTS} passed variant=${variant} shapes=${batch_shapes}"
    else
        candidate_batch_failures=$((candidate_batch_failures + 1))
        tail -40 "${RUNNER_LOG}" || true
        announce "CANDIDATE_MEASUREMENT_BATCH ${batch_index}/${EXPECTED_VARIANTS} failed variant=${variant}; continuing_remaining_variants=1"
    fi
done
[[ "${batch_index}" -eq "${EXPECTED_VARIANTS}" ]] || \
    fail "measured ${batch_index} batches, expected ${EXPECTED_VARIANTS}"
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=paired_npu_measurement wall_ms=${measurement_wall_ms}"
[[ "${candidate_batch_failures}" -eq 0 ]] || \
    fail "all ${EXPECTED_VARIANTS} variants were attempted; ${candidate_batch_failures} variant batches failed correctness or execution"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_rule_matrix.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUNNER_LOG}" \
    --official-profile "${OFFICIAL_PROFILE}" \
    --official-samples "${OFFICIAL_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
announce "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
cleanup_generated_state
cleanup_build_state
emit_final_results
printf '%s\n' "CERTIFIED_AL1_VALIDATION_COMPLETE cached=0 summary=${SUMMARY} log=${RUN_LOG}" >&3
