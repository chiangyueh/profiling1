#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=3
REPEAT=10
SAMPLES=15
NPU_SHAPES=8
SOLVED_EXPANDED_SHAPES=12
EXPECTED_VARIANTS=4
MAX_FOOTPRINT_MIB=320

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Audits all twelve installed suffix rules, solves five expanded C220' \
        'families, and measures the three families buildable with CANN 8.1.'
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
        tools/audit_matmul_family_solver.py \
        tools/generate_matmul_c220_experimental_matrix.py \
        tools/analyze_matmul_rule_matrix.py \
        scripts/build_all.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry_c220.cpp \
        direct_matmul/mat_mul_v3_tiling_data_280.h \
        direct_matmul/runner.cpp \
        matmul_rule_selector/c220_experimental_selector.py \
        matmul_rule_selector/c220_validation_contract.json \
        matmul_rule_selector/expanded_family_rules.py \
        matmul_rule_selector/core_ownership_rules.py \
        matmul_rule_selector/improved_selector.py
    find colleague_matmul_v3/op_kernel -type f -print0 | sort -z | xargs -0 sha256sum
    find matmul_rule_selector/baseline_core -type f -name '*.py' -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_expanded_family_v1/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
AUDIT_LOG="${CAMPAIGN_DIR}/family_audit.txt"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
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
    local line="${BASH_LINENO[0]}"
    local tail_text
    trap - ERR
    tail_text="$(tail -50 "${RUN_LOG}" 2>/dev/null || true)"
    cleanup_generated_state || true
    printf 'RULE_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >"${RUN_LOG}"
    printf 'RULE_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >&3
    exit "${rc}"
}
trap on_error ERR

announce "RUN_LOG path=${RUN_LOG}"
announce "SOURCE_REVISION commit=$(git rev-parse HEAD 2>/dev/null || printf unknown) result13_tag=result13-core-ownership-v1"
announce "CAMPAIGN_READY operator=matmul installed_suffixes=12 installed_audit_shapes=60 expanded_families=5 solved_expanded_shapes=${SOLVED_EXPANDED_SHAPES} npu_shapes=${NPU_SHAPES} variants=${EXPECTED_VARIANTS}"
announce "selection=disjoint_shape_and_c220_hardware_equations_only"
announce "forbidden=cost_model,latency_ranker,candidate_bank,history_lookup,repo_lookup,official_tiling_seed,fallback_kernel"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} visible_devices=${ASCEND_RT_VISIBLE_DEVICES} runtime_user_device=${DEVICE_ID}"

python3 tools/audit_matmul_family_solver.py >"${AUDIT_LOG}"
grep -q 'installed_suffixes=12 installed_shapes=60 expanded_families=5 expanded_shapes=12 cann81_buildable_expanded_shapes=8' "${AUDIT_LOG}"

python3 tools/generate_matmul_c220_experimental_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}"
actual_variants="$(find "${VARIANT_DIR}" -maxdepth 1 -type f -name '*.csv' | wc -l)"
[[ "${actual_variants}" -eq "${EXPECTED_VARIANTS}" ]]

python3 - "${MANIFEST}" "${MAX_FOOTPRINT_MIB}" <<'PY'
import csv
import sys
width = {"fp16": 2, "bf16": 2, "fp32": 4}
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
limit = int(sys.argv[2]) * 1024 * 1024
largest = max((
    (int(row["m"]) * int(row["k"]) + int(row["k"]) * int(row["n"]) +
     2 * int(row["m"]) * int(row["n"])) * width[row["dtype"]] +
    int(row["workspace_bytes"]), row["workload_id"]
) for row in rows)
if largest[0] > limit:
    raise SystemExit(f"footprint cap exceeded: {largest}")
print(f"FOOTPRINT_AUDIT passed largest={largest[1]} bytes={largest[0]} limit={limit}")
PY

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
    if rc: raise RuntimeError(f"aclInit failed rc={rc}")
    initialized = True
    count = ctypes.c_uint32()
    rc = acl.aclrtGetDeviceCount(ctypes.byref(count))
    if rc: raise RuntimeError(f"aclrtGetDeviceCount failed rc={rc}")
    if not 0 <= device < count.value:
        raise RuntimeError(f"runtime user device {device} outside 0..{count.value - 1}")
    rc = acl.aclrtSetDevice(device)
    if rc:
        raise RuntimeError(
            f"aclrtSetDevice failed rc={rc} user_device={device} "
            f"visible_devices={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '')}"
        )
    device_set = True
    print(f"DEVICE_PREFLIGHT passed user_device={device} available={count.value}")
finally:
    if device_set: acl.aclrtResetDevice(device)
    if initialized: acl.aclFinalize()
PY

announce "OFFICIAL_RUNNER_BUILD begin jobs=1"
BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh
official_runner="${ROOT}/build/official_matmul_runner"
[[ -x "${official_runner}" ]]
"${official_runner}" --candidates "${MANIFEST}" --validate-input >/dev/null
announce "OFFICIAL_RUNNER_BUILD passed"

variant_index=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    variant_index=$((variant_index + 1))
    announce "EXPANDED_VARIANT_BUILD ${variant_index}/${EXPECTED_VARIANTS} begin variant=${variant} jobs=1"
    BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" scripts/build_all.sh
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    [[ -x "${runner}" ]]
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    "${runner}" --manifest "${variant_manifest}" --device "${DEVICE_ID}" \
        --warmup 0 --repeat 1 --samples 1 >/dev/null
    announce "EXPANDED_VARIANT_BUILD ${variant_index}/${EXPECTED_VARIANTS} canary_passed variant=${variant}"
done
[[ "${variant_index}" -eq "${EXPECTED_VARIANTS}" ]]

announce "OFFICIAL_MEASUREMENT begin shapes=${NPU_SHAPES}"
"${official_runner}" \
    --candidates "${MANIFEST}" \
    --output "${OFFICIAL_PROFILE}" \
    --samples-output "${OFFICIAL_SAMPLES}" \
    --device "${DEVICE_ID}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" \
    --numeric-preflight-max-mib "${MAX_FOOTPRINT_MIB}" \
    --structured-full-preflight --validate-after-measurement
announce "OFFICIAL_MEASUREMENT passed shapes=${NPU_SHAPES}"

batch_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    batch_index=$((batch_index + 1))
    announce "EXPANDED_MEASUREMENT ${batch_index}/${EXPECTED_VARIANTS} begin variant=${variant}"
    "${runner}" --manifest "${packet_manifest}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}"
    announce "EXPANDED_MEASUREMENT ${batch_index}/${EXPECTED_VARIANTS} passed variant=${variant}"
done
[[ "${batch_index}" -eq "${EXPECTED_VARIANTS}" ]]

python3 tools/analyze_matmul_rule_matrix.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUN_LOG}" \
    --official-profile "${OFFICIAL_PROFILE}" \
    --official-samples "${OFFICIAL_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"

FINAL_TEXT="$(python3 - "${AUDIT_LOG}" "${SUMMARY}" "${SELECTION}" <<'PY'
import csv
import json
import sys
from collections import defaultdict

audit = open(sys.argv[1], encoding="utf-8").read().strip()
with open(sys.argv[2], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
selections = [json.loads(line) for line in open(sys.argv[3], encoding="utf-8") if line.strip()]
print("FINAL_RESULTS_BEGIN")
print(audit)
for row in selections:
    if not row["npu_eligible"]:
        print(
            "EXPANDED_FAMILY_NOT_MEASURED "
            f"family={row['formula_family']} suffix={row['kernel_suffix']} "
            f"reason={row['toolchain_contract']['cann81_kernel_build']} "
            "fallback_kernel=0"
        )
for row in rows:
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} family={row['family']} suffix={row['kernel_suffix']} "
        f"m={row['m']} n={row['n']} k={row['k']} dtype={row['dtype']} "
        f"cores={row['used_cores']} official_ms={float(row['official_median_ms']):.9g} "
        f"candidate_ms={float(row['candidate_median_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} winner={row['median_winner']} "
        f"separation={row['sample_separation']} correctness={row['correctness']}"
    )
groups = defaultdict(list)
for row in rows:
    groups[row["family"]].append(row)
for family in sorted(groups):
    group = groups[family]
    print(
        "FINAL_FAMILY_RESULT "
        f"family={family} shapes={len(group)} "
        f"candidate_wins={sum(row['median_winner'] == 'candidate' for row in group)} "
        f"official_wins={sum(row['median_winner'] == 'official' for row in group)} "
        f"clear_candidate_wins={sum(row['sample_separation'] == 'CLEAR_CANDIDATE_WINNER' for row in group)} "
        f"clear_official_wins={sum(row['sample_separation'] == 'CLEAR_OFFICIAL_WINNER' for row in group)} "
        f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in group)}"
    )
print(
    "FINAL_RESULT_SUMMARY "
    f"installed_suffixes_audited=12 expanded_families_solved=5 "
    f"npu_shapes={len(rows)} candidate_wins={sum(row['median_winner'] == 'candidate' for row in rows)} "
    f"official_wins={sum(row['median_winner'] == 'official' for row in rows)} "
    f"overlap={sum(row['sample_separation'] == 'OVERLAPPING_SAMPLES' for row in rows)}"
)
print("FINAL_RESULTS_END")
PY
)"

trap - ERR
cleanup_generated_state
printf '%s\n' "${FINAL_TEXT}" >"${RUN_LOG}"
printf '%s\n' "${FINAL_TEXT}" >&3
printf 'EXPANDED_FAMILY_VALIDATION_COMPLETE log=%s\n' "${RUN_LOG}" >&3
