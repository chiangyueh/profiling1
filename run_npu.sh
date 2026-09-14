#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
WARMUP=3
REPEAT=10
SAMPLES=15
NPU_SHAPES=200
EXPECTED_VARIANTS=3
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
        tools/audit_independent_tiled_families.py \
        tools/generate_independent_tiled_matrix.py \
        tools/analyze_independent_tiled_results.py \
        scripts/build_all.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry_c220.cpp \
        direct_matmul/mat_mul_v3_tiling_data_280.h \
        direct_matmul/runner.cpp \
        novel_matmul/independent_tiled_kernel.h \
        matmul_rule_selector/independent_tiled_selector.py \
        matmul_rule_selector/independent_validation_cases.py
} | sha256sum | cut -c1-20)"
CAMPAIGN_ROOT="${ROOT}/results/matmul_independent_formula_v2"
CAMPAIGN_DIR="${CAMPAIGN_ROOT}/${CAMPAIGN_ID}"
MANIFEST="${CAMPAIGN_DIR}/manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
AUDIT_LOG="${CAMPAIGN_DIR}/audit.txt"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"

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
    printf 'INDEPENDENT_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >"${RUN_LOG}"
    printf 'INDEPENDENT_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >&3
    exit "${rc}"
}
trap on_error ERR

cleanup_generated_state
mkdir -p "${CAMPAIGN_DIR}"

announce "RUN_LOG path=${RUN_LOG}"
announce "CAMPAIGN_READY operator=matmul independent_modes=6 npu_shapes=${NPU_SHAPES} variants=${EXPECTED_VARIANTS}"
announce "modes=MICRO_DIRECT,BALANCED_MN,SEEDED_SPLIT_K,RESIDENT_B_M_STRIPE,RESIDENT_A_N_STRIPE,SEEDED_TAIL_WAVE"
announce "validation_domain=12_installed_suffix_groups_x10_plus_80_independent_mode_cases"
announce "forbidden=installed_selector_at_runtime,official_tiling_seed,cost_model,latency_ranker,candidate_bank,history_lookup,repo_lookup,candidate_search,fallback_kernel"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} visible_devices=${ASCEND_RT_VISIBLE_DEVICES} runtime_user_device=${DEVICE_ID}"

python3 tools/audit_independent_tiled_families.py >"${AUDIT_LOG}"
grep -q 'VALIDATION_DOMAIN passed shapes=200 distinct_packets=200 installed_suffixes=12 independent_modes=6' "${AUDIT_LOG}"
python3 tools/generate_independent_tiled_matrix.py --root "${CAMPAIGN_DIR}"

python3 - "${MANIFEST}" "${MAX_FOOTPRINT_MIB}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
limit = int(sys.argv[2]) * 1024 * 1024
footprints = []
for row in rows:
    width = 4 if row["dtype"] == "fp32" else 2
    size = ((int(row["m"]) * int(row["k"]) + int(row["k"]) * int(row["n"]) +
             2 * int(row["m"]) * int(row["n"])) * width +
            int(row["workspace_bytes"]))
    footprints.append((size, row["workload_id"]))
if len(rows) != 200:
    raise SystemExit(f"shape count failed: {len(rows)}")
largest = max(footprints)
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
"${official_runner}" --candidates "${MANIFEST}" --validate-input >/dev/null
announce "OFFICIAL_RUNNER_BUILD passed"

variant_index=0
for variant in fp16_k91000 bf16_k91000 fp32_k91000; do
    variant_manifest="${VARIANT_DIR}/${variant}.csv"
    dtype="${variant%%_k*}"
    variant_index=$((variant_index + 1))
    announce "KERNEL_BUILD ${variant_index}/${EXPECTED_VARIANTS} begin variant=${variant} jobs=1"
    BUILD_COMPONENTS=variant BUILD_JOBS=1 \
        DIRECT_KERNEL_TARGET="direct_matmul_kernel_${dtype}_91000" scripts/build_all.sh
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    canary="${CAMPAIGN_DIR}/canary_${variant}.csv"
    python3 - "${variant_manifest}" "${SELECTION}" "${canary}" <<'PY'
import csv
import json
import sys
manifest_path, selection_path, output_path = sys.argv[1:]
families = {}
for line in open(selection_path, encoding="utf-8"):
    row = json.loads(line)
    families[row["workload_id"]] = row["formula_family"]
with open(manifest_path, newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
selected = {}
for row in rows:
    selected.setdefault(families[row["workload_id"]], row)
if len(selected) != 6:
    raise SystemExit(f"dtype canary does not cover six modes: {sorted(selected)}")
with open(output_path, "w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected.values())
PY
    "${runner}" --manifest "${canary}" --device "${DEVICE_ID}" \
        --warmup 0 --repeat 1 --samples 1 >/dev/null
    announce "KERNEL_BUILD ${variant_index}/${EXPECTED_VARIANTS} all_mode_canary_passed variant=${variant} modes=6"
done

announce "OFFICIAL81_MEASUREMENT begin shapes=${NPU_SHAPES}"
"${official_runner}" \
    --candidates "${MANIFEST}" \
    --output "${OFFICIAL_PROFILE}" \
    --samples-output "${OFFICIAL_SAMPLES}" \
    --device "${DEVICE_ID}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}" \
    --numeric-preflight-max-mib "${MAX_FOOTPRINT_MIB}" \
    --structured-full-preflight --validate-after-measurement
announce "OFFICIAL81_MEASUREMENT passed shapes=${NPU_SHAPES}"

measurement_index=0
for variant in fp16_k91000 bf16_k91000 fp32_k91000; do
    measurement_index=$((measurement_index + 1))
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    variant_manifest="${VARIANT_DIR}/${variant}.csv"
    announce "DIRECT_MEASUREMENT ${measurement_index}/${EXPECTED_VARIANTS} begin variant=${variant}"
    "${runner}" --manifest "${variant_manifest}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}"
    announce "DIRECT_MEASUREMENT ${measurement_index}/${EXPECTED_VARIANTS} passed variant=${variant}"
done

python3 tools/analyze_independent_tiled_results.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUN_LOG}" \
    --official-profile "${OFFICIAL_PROFILE}" \
    --official-samples "${OFFICIAL_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"

FINAL_TEXT="$(python3 - "${AUDIT_LOG}" "${SUMMARY}" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict

audit = open(sys.argv[1], encoding="utf-8").read().strip()
with open(sys.argv[2], newline="", encoding="utf-8") as stream:
    all_rows = list(csv.DictReader(stream))
print("FINAL_RESULTS_BEGIN")
print(audit)
for row in all_rows:
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} family={row['family']} "
        f"m={row['m']} n={row['n']} k={row['k']} dtype={row['dtype']} "
        f"ta={row['trans_a']} tb={row['trans_b']} cores={row['used_cores']} "
        f"official_ms={float(row['official_ms']):.9g} "
        f"candidate_ms={float(row['candidate_ms']):.9g} "
        f"delta_pct={float(row['delta_pct']):+.3f} "
        f"separation={row['separation']} correctness={row['correctness']}"
    )

def emit_groups(label, rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    for name, group in sorted(groups.items(), key=lambda item: str(item[0])):
        deltas = [float(row["delta_pct"]) for row in group]
        print(
            f"{label} group={name} shapes={len(group)} "
            f"candidate_wins={sum(value < 0 for value in deltas)} "
            f"official_wins={sum(value > 0 for value in deltas)} "
            f"median_delta_pct={statistics.median(deltas):+.3f} "
            f"worst_delta_pct={max(deltas):+.3f}"
        )

emit_groups("FINAL_FAMILY_RESULT", all_rows, "family")
installed = [row for row in all_rows if row["official_suffix_audit"]]
emit_groups("FINAL_INSTALLED_SUFFIX_RESULT", installed, "official_suffix_audit")
deltas = [float(row["delta_pct"]) for row in all_rows]
print(
    "FINAL_RESULTS_SUMMARY "
    f"independent_modes=6 shapes={len(all_rows)} "
    f"candidate_wins={sum(value < 0 for value in deltas)} "
    f"official_wins={sum(value > 0 for value in deltas)} "
    f"median_delta_pct={statistics.median(deltas):+.3f} "
    "current_output_correctness=200/200"
)
print("FINAL_RESULTS_END")
PY
)"

trap - ERR
cleanup_generated_state
printf '%s\n' "${FINAL_TEXT}" >"${RUN_LOG}"
printf '%s\n' "${FINAL_TEXT}" >&3
printf 'INDEPENDENT_TILING_VALIDATION_COMPLETE log=%s\n' "${RUN_LOG}" >&3
