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
        tools/audit_novel_matmul_families.py \
        tools/generate_novel_matmul_matrix.py \
        tools/analyze_novel_matmul_results.py \
        scripts/build_all.sh \
        cmake_npu/CMakeLists.txt \
        direct_matmul/kernel_entry_c220.cpp \
        direct_matmul/mat_mul_v3_tiling_data_280.h \
        direct_matmul/runner.cpp \
        novel_matmul/seeded_split_k_kernel.h \
        matmul_rule_selector/novel_family_selector.py \
        matmul_rule_selector/novel_validation_cases.py \
        matmul_rule_selector/official_family_provenance.json
    find colleague_matmul_v3/op_kernel -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_novel_family_v1/${CAMPAIGN_ID}"
MANIFEST="${CAMPAIGN_DIR}/manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
AUDIT_LOG="${CAMPAIGN_DIR}/audit.txt"
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
    tail_text="$(tail -60 "${RUN_LOG}" 2>/dev/null || true)"
    cleanup_generated_state || true
    printf 'NOVEL_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >"${RUN_LOG}"
    printf 'NOVEL_VALIDATION_FATAL rc=%s line=%s log=%s\n%s\n' \
        "${rc}" "${line}" "${RUN_LOG}" "${tail_text}" >&3
    exit "${rc}"
}
trap on_error ERR

announce "RUN_LOG path=${RUN_LOG}"
announce "CAMPAIGN_READY operator=matmul repository_owned_families=2 npu_shapes=${NPU_SHAPES} closest_official_controls=100 variants=${EXPECTED_VARIANTS}"
announce "new_families=SEEDED_ATOMIC_SPLIT_K,SEEDED_TAIL_WAVE_SPLIT_K"
announce "provenance=custom_suffixes_90001_90002 later_official_suffix_41_is_control_only"
announce "forbidden=cost_model,latency_ranker,candidate_bank,history_lookup,repo_lookup,official_tiling_seed,candidate_search,fallback_kernel"
announce "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
announce "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} visible_devices=${ASCEND_RT_VISIBLE_DEVICES} runtime_user_device=${DEVICE_ID}"

python3 tools/audit_novel_matmul_families.py >"${AUDIT_LOG}"
grep -q 'repository_owned_families=2 shapes=200 distinct_packets=200' "${AUDIT_LOG}"
python3 tools/generate_novel_matmul_matrix.py --root "${CAMPAIGN_DIR}"

python3 - "${MANIFEST}" "${MAX_FOOTPRINT_MIB}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
limit = int(sys.argv[2]) * 1024 * 1024
footprints = [
    ((int(row["m"]) * int(row["k"]) + int(row["k"]) * int(row["n"]) +
      2 * int(row["m"]) * int(row["n"])) * 4 + int(row["workspace_bytes"]),
     row["workload_id"])
    for row in rows
]
if len(rows) != 200 or any(int(row["k"]) > 60000 for row in rows):
    raise SystemExit("shape count or official K limit failed")
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
for variant in fp32_k41 fp32_k90001 fp32_k90002; do
    variant_manifest="${VARIANT_DIR}/${variant}.csv"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_fp32_${suffix}"
    variant_index=$((variant_index + 1))
    announce "KERNEL_BUILD ${variant_index}/${EXPECTED_VARIANTS} begin variant=${variant} jobs=1"
    BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" scripts/build_all.sh
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    canary="${CAMPAIGN_DIR}/canary_${variant}.csv"
    { head -n 1 "${variant_manifest}"; sed -n '2p' "${variant_manifest}"; } >"${canary}"
    "${runner}" --manifest "${canary}" --device "${DEVICE_ID}" \
        --warmup 0 --repeat 1 --samples 1 >/dev/null
    announce "KERNEL_BUILD ${variant_index}/${EXPECTED_VARIANTS} canary_passed variant=${variant}"
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
for variant in fp32_k41 fp32_k90001 fp32_k90002; do
    measurement_index=$((measurement_index + 1))
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    variant_manifest="${VARIANT_DIR}/${variant}.csv"
    announce "DIRECT_MEASUREMENT ${measurement_index}/${EXPECTED_VARIANTS} begin variant=${variant}"
    "${runner}" --manifest "${variant_manifest}" --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" --repeat "${REPEAT}" --samples "${SAMPLES}"
    announce "DIRECT_MEASUREMENT ${measurement_index}/${EXPECTED_VARIANTS} passed variant=${variant}"
done

python3 tools/analyze_novel_matmul_results.py \
    --manifest "${MANIFEST}" \
    --control-manifest "${VARIANT_DIR}/fp32_k41.csv" \
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
    rows = list(csv.DictReader(stream))
print("FINAL_RESULTS_BEGIN")
print(audit)
for row in rows:
    closest = "none"
    if row["closest_official_median_ms"]:
        closest = (
            f"suffix41_ms={float(row['closest_official_median_ms']):.9g} "
            f"vs_suffix41_pct={float(row['vs_closest_official_delta_pct']):+.3f} "
            f"suffix41_separation={row['vs_closest_official_separation']}"
        )
    print(
        "FINAL_RESULT "
        f"id={row['workload_id']} family={row['family']} suffix={row['kernel_suffix']} "
        f"m={row['m']} n={row['n']} k={row['k']} scale={row['scale_band']} "
        f"cores={row['used_cores']} official81_ms={float(row['official81_median_ms']):.9g} "
        f"candidate_ms={float(row['candidate_median_ms']):.9g} "
        f"vs_official81_pct={float(row['vs_official81_delta_pct']):+.3f} "
        f"official81_separation={row['vs_official81_separation']} {closest} "
        f"correctness={row['correctness']}"
    )
groups = defaultdict(list)
for row in rows:
    groups[row["family"]].append(row)
for family, group in sorted(groups.items()):
    public_delta = [float(row["vs_official81_delta_pct"]) for row in group]
    control_delta = [float(row["vs_closest_official_delta_pct"]) for row in group
                     if row["vs_closest_official_delta_pct"]]
    text = (
        "FINAL_FAMILY_RESULT "
        f"family={family} shapes={len(group)} "
        f"official81_wins={sum(value < 0 for value in public_delta)} "
        f"official81_losses={sum(value > 0 for value in public_delta)} "
        f"official81_median_delta_pct={statistics.median(public_delta):+.3f}"
    )
    if control_delta:
        text += (
            f" suffix41_wins={sum(value < 0 for value in control_delta)} "
            f"suffix41_losses={sum(value > 0 for value in control_delta)} "
            f"suffix41_median_delta_pct={statistics.median(control_delta):+.3f}"
        )
    else:
        text += " closest_c220_official_control=NONE"
    print(text)
print("FINAL_RESULTS_SUMMARY repository_owned_families=2 shapes=200 current_output_correctness=200/200")
print("FINAL_RESULTS_END")
PY
)"

trap - ERR
cleanup_generated_state
printf '%s\n' "${FINAL_TEXT}" >"${RUN_LOG}"
printf '%s\n' "${FINAL_TEXT}" >&3
printf 'NOVEL_FAMILY_VALIDATION_COMPLETE log=%s\n' "${RUN_LOG}" >&3
