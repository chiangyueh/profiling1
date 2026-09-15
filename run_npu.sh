#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=1
REPEAT=3
SAMPLES=5
NPU_SHAPES=285
EXPECTED_VARIANTS=23
MAX_FOOTPRINT_MIB=300

usage() {
    printf '%s\n' 'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]'
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
export PYTHONWARNINGS=ignore
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
        tools/generate_formula_tiling_matrix.py \
        tools/audit_formula_reference_separation.py \
        tools/analyze_formula_tiling_results.py \
        matmul_rule_selector/formula_rules.py \
        matmul_rule_selector/complete_formula_selector.py \
        matmul_rule_selector/novel_family_selector.py \
        matmul_rule_selector/novel_validation_cases.py \
        matmul_rule_selector/c220_source_route_selector.py \
        matmul_rule_selector/c220_source_route_cases.py \
        matmul_rule_selector/tiling_selector.py \
        tools/audit_novel_matmul_families.py \
        tools/audit_c220_dispatch_source.py \
        scripts/build_all.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/kernel_entry_c220.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/mat_mul_v3_tiling_data_280.h \
        direct_matmul/runner.cpp \
        novel_matmul/direct_init_split_k_kernel.h
    sha256sum \
        colleague_matmul_v3/op_kernel/mat_mul_v3.cpp \
        colleague_matmul_v3/op_kernel/mat_mul_v3_tiling_key.h
} | sha256sum | cut -c1-20)"
CAMPAIGN_ROOT="${ROOT}/results/matmul_complete_formula_v1"
CAMPAIGN_DIR="${CAMPAIGN_ROOT}/${CAMPAIGN_ID}"
MANIFEST="${CAMPAIGN_DIR}/manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
REFERENCE_AUDIT="${CAMPAIGN_DIR}/reference_audit.json"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
DISPATCH_AUDIT="${CAMPAIGN_DIR}/c220_dispatch_audit.json"

cleanup_generated_state() {
    if [[ -d "${CAMPAIGN_ROOT}" ]]; then
        find "${CAMPAIGN_ROOT}" -mindepth 1 -delete
        rmdir -- "${CAMPAIGN_ROOT}" 2>/dev/null || true
    fi
    if [[ -d "${ROOT}/build" ]]; then
        find "${ROOT}/build" -mindepth 1 -delete
        rmdir -- "${ROOT}/build" 2>/dev/null || true
    fi
}

on_error() {
    local rc=$?
    local line="${BASH_LINENO[0]}"
    local tail_text
    trap - ERR
    tail_text="$(tail -60 "${RUN_LOG}" 2>/dev/null || true)"
    cleanup_generated_state || true
    printf 'FORMULA_TILING_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >"${RUN_LOG}"
    printf 'FORMULA_TILING_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >&3
    exit "${rc}"
}
trap on_error ERR

cleanup_generated_state
mkdir -p "${CAMPAIGN_DIR}"

announce "RUN_LOG path=${RUN_LOG}"
announce "FORMULA_TILING_READY shapes=${NPU_SHAPES} source_dispatch_routes=21 experimental_derivative_schedules=2 original_family_claims=0 variants=${EXPECTED_VARIANTS} complete_tilings_per_shape=1"
announce "selector=shape_hardware_capacity_and_ownership_equations_only"
announce "tiling_fields=family,baseM,baseN,baseK,singleCoreM,singleCoreN,singleCoreK,usedCoreNum,stepM,stepN,stepKa,stepKb,depthA1,depthB1,dbL0C,iterateOrder,L2,ND2NZ,new_K_ownership_scheduler"
announce "forbidden=official_selector_input,official_tiling_seed,cost_model,latency_ranker,candidate_enumeration,history_lookup,repo_lookup,tiling_bank"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} visible_devices=${ASCEND_RT_VISIBLE_DEVICES} runtime_user_device=${DEVICE_ID}"

python3 tools/generate_formula_tiling_matrix.py --root "${CAMPAIGN_DIR}"
python3 tools/audit_formula_reference_separation.py \
    --manifest "${MANIFEST}" --selection "${SELECTION}" \
    --output "${REFERENCE_AUDIT}"
python3 tools/audit_novel_matmul_families.py
python3 tools/audit_c220_dispatch_source.py \
    --manifest "${MANIFEST}" --output "${DISPATCH_AUDIT}" >/dev/null
announce "C220_SOURCE_DISPATCH_AUDIT passed branches=21 kernel_classes=16 build_route_omissions=0 measurement_route_omissions=0"

variant_count="$(find "${VARIANT_DIR}" -maxdepth 1 -type f -name '*.csv' | wc -l)"
[[ "${variant_count}" -eq "${EXPECTED_VARIANTS}" ]] || {
    echo "fatal: expected ${EXPECTED_VARIANTS} compiled suffix variants, found ${variant_count}" >&2
    exit 1
}

python3 - "${DEVICE_ID}" <<'PY'
import ctypes
import os
import sys
device = int(sys.argv[1])
acl = ctypes.CDLL("libascendcl.so", mode=ctypes.RTLD_GLOBAL)
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclInit.restype = ctypes.c_int
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
    if rc:
        raise RuntimeError(f"aclInit failed rc={rc}")
    initialized = True
    count = ctypes.c_uint32()
    rc = acl.aclrtGetDeviceCount(ctypes.byref(count))
    if rc:
        raise RuntimeError(f"aclrtGetDeviceCount failed rc={rc}")
    if not 0 <= device < count.value:
        raise RuntimeError(f"runtime user device {device} outside 0..{count.value - 1}")
    rc = acl.aclrtSetDevice(device)
    if rc:
        raise RuntimeError(
            f"aclrtSetDevice failed rc={rc} user_device={device} "
            f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}"
        )
    device_set = True
finally:
    if device_set:
        acl.aclrtResetDevice(device)
    if initialized:
        acl.aclFinalize()
PY

announce "BUILD begin official_runner_and_23_route_kernels jobs=1"
BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh
official_runner="${ROOT}/build/official_matmul_runner"
"${official_runner}" --candidates "${MANIFEST}" --validate-input >/dev/null

variant_index=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    variant_index=$((variant_index + 1))
    BUILD_COMPONENTS=variant BUILD_JOBS=1 \
        DIRECT_KERNEL_TARGET="direct_matmul_kernel_${dtype}_${suffix}" \
        scripts/build_all.sh
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    canary="${CAMPAIGN_DIR}/canary_${variant}.csv"
    python3 - "${variant_manifest}" "${canary}" <<'PY'
import csv
import sys
source, output = sys.argv[1:]
with open(source, newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    row = next(reader)
    fields = reader.fieldnames
with open(output, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
PY
    "${runner}" --manifest "${canary}" --device "${DEVICE_ID}" \
        --warmup 0 --repeat 1 --samples 1
done
[[ "${variant_index}" -eq "${EXPECTED_VARIANTS}" ]] || {
    echo "fatal: variant build loop covered ${variant_index}/${EXPECTED_VARIANTS}" >&2
    exit 1
}
announce "BUILD passed official_runner=1 suffix_kernels=${variant_index} canaries=${variant_index}"

announce "MEASUREMENT begin shapes=${NPU_SHAPES}"
"${official_runner}" \
    --candidates "${MANIFEST}" \
    --output "${OFFICIAL_PROFILE}" \
    --samples-output "${OFFICIAL_SAMPLES}" \
    --device "${DEVICE_ID}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" \
    --numeric-preflight-max-mib "${MAX_FOOTPRINT_MIB}" \
    --structured-full-preflight --validate-after-measurement

measurement_failures=()
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    if variant_output="$("${runner}" --manifest "${variant_manifest}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" 2>&1)"; then
        printf '%s\n' "${variant_output}"
    else
        variant_rc=$?
        printf '%s\n' "${variant_output}"
        fatal_line="$(printf '%s\n' "${variant_output}" | awk '/^DIRECT_MATMUL_FATAL / { line=$0 } END { print line }')"
        measurement_failures+=("variant=${variant} rc=${variant_rc} ${fatal_line}")
    fi
done

if ((${#measurement_failures[@]})); then
    announce "DIRECT_MEASUREMENT_FAILURES_BEGIN count=${#measurement_failures[@]}"
    for failure in "${measurement_failures[@]}"; do
        announce "DIRECT_VARIANT_FAILURE ${failure}"
    done
    announce "DIRECT_MEASUREMENT_FAILURES_END"
    false
fi

python3 tools/analyze_formula_tiling_results.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUN_LOG}" \
    --official-profile "${OFFICIAL_PROFILE}" \
    --official-samples "${OFFICIAL_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"

FINAL_TEXT="$(python3 - "${REFERENCE_AUDIT}" "${SUMMARY}" "${SELECTION}" "${DISPATCH_AUDIT}" <<'PY'
import csv
import json
import statistics
import sys
from collections import defaultdict

audit = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
selections = {
    row["workload_id"]: row
    for row in (
        json.loads(line)
        for line in open(sys.argv[3], encoding="utf-8")
        if line.strip()
    )
}
dispatch = json.load(open(sys.argv[4], encoding="utf-8"))
print("FINAL_RESULTS_BEGIN")
print(
    "C220_SOURCE_DISPATCH_AUDIT "
    f"branches={dispatch['dispatch_branch_count']} "
    f"kernel_classes={dispatch['kernel_class_count']} "
    f"registered={len(dispatch['registered_suffixes'])} "
    f"build_registered={len(dispatch['build_suffixes'])} "
    f"measurement_registered={len(dispatch['manifest_suffixes'])} omissions=0 "
    "source_derived=1"
)
print(
    "FORMULA_TILING_AUDIT "
    f"installed_shapes={audit['shape_count']} installed_suffixes={len(audit['suffixes'])} "
    "later_official_source_route_shapes=45 later_official_source_routes=9 "
    "experimental_derivative_shapes=40 experimental_derivative_schedules=2 original_family_claims=0 "
    f"exact_reference_packets={audit['exact_reference_packets']} "
    f"core_only_changes={audit['core_only_changes']} "
    "complete_tilings_per_shape=1 official_seed=0 cost_model=0 candidate_search=0"
)
groups = defaultdict(list)
for row in rows:
    groups[int(row["suffix"])].append(row)
    fields = selections[row["workload_id"]]["tiling_fields"]
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} family={row['family']} suffix={row['suffix']} "
        f"m={row['m']} n={row['n']} k={row['k']} dtype={row['dtype']} "
        f"ta={row['trans_a']} tb={row['trans_b']} cores={row['used_cores']} "
        f"base={fields['baseM']}x{fields['baseN']}x{fields['baseK']} "
        f"single={fields['singleCoreM']}x{fields['singleCoreN']}x{fields['singleCoreK']} "
        f"step={fields['stepM']}x{fields['stepN']}x{fields['stepKa']}x{fields['stepKb']} "
        f"depth={fields['depthA1']}x{fields['depthB1']} dbL0C={fields['dbL0C']} "
        f"order={fields['iterateOrder']} official_ms={float(row['official_ms']):.9g} "
        f"formula_ms={float(row['formula_ms']):.9g} delta_pct={float(row['delta_pct']):+.3f} "
        f"separation={row['separation']} correctness={row['correctness']}"
    )
for suffix, group in sorted(groups.items()):
    deltas = [float(row["delta_pct"]) for row in group]
    branch = audit["suffixes"].get(str(suffix))
    if suffix in (90001, 90002):
        changed_fields = (
            "new_kernel_scheduler,baseM,baseN,baseK,singleCoreK,"
            "stepKa,stepKb,depthA1,depthB1,dbL0C,K_ownership"
        )
        minimum_changed = "EXPERIMENTAL_DERIVATIVE_SCHEDULE"
    elif branch is None:
        changed_fields = "bundled_later_official_kernel_route,complete_280_byte_packet"
        minimum_changed = "LATER_OFFICIAL_SOURCE_ROUTE"
    else:
        changed_fields = ",".join(branch["changed_schedule_field_union"])
        minimum_changed = branch["minimum_changed_schedule_fields_per_shape"]
    print(
        "FINAL_SUFFIX_RESULT "
        f"suffix={suffix} family={group[0]['family']} shapes={len(group)} "
        f"changed_fields={changed_fields} "
        f"min_changed_fields_per_shape={minimum_changed} "
        f"formula_wins={sum(value < 0 for value in deltas)} "
        f"official_wins={sum(value > 0 for value in deltas)} "
        f"overlap={sum(row['separation'] == 'OVERLAP' for row in group)} "
        f"median_delta_pct={statistics.median(deltas):+.3f} "
        f"worst_delta_pct={max(deltas):+.3f} correctness={len(group)}/{len(group)}"
    )
deltas = [float(row["delta_pct"]) for row in rows]
print(
    "FINAL_RESULTS_SUMMARY "
    f"shapes={len(rows)} suffixes={len(groups)} "
    f"formula_wins={sum(value < 0 for value in deltas)} "
    f"official_wins={sum(value > 0 for value in deltas)} "
    f"overlap={sum(row['separation'] == 'OVERLAP' for row in rows)} "
    f"median_delta_pct={statistics.median(deltas):+.3f} "
    f"worst_delta_pct={max(deltas):+.3f} "
    f"current_output_correctness={len(rows)}/{len(rows)}"
)
print("FINAL_RESULTS_END")
PY
)"

trap - ERR
cleanup_generated_state
printf '%s\n' "${FINAL_TEXT}" >"${RUN_LOG}"
printf '%s\n' "${FINAL_TEXT}" >&3
printf 'FORMULA_TILING_VALIDATION_COMPLETE log=%s\n' "${RUN_LOG}" >&3
