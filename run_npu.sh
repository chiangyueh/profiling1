#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
SHAPE_COUNT=3
DIRECT_RECORDS=12
INSTALLED_REFERENCE_RECORDS=3
TOTAL_RECORDS=15

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Narrow R5 L2-policy validation on three affected shapes. Each shape executes
an ABBA sequence: unchanged direct reference, modified R5 packet, modified R5
packet, unchanged direct reference. The installed aclnnMatmul path is a
separate, explicitly labelled context reference. Neither reference path is
eligible for credit as new code, and the original MatMulV3 selector is never
called to generate a direct packet.
USAGE
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
export CANN_ROOT="${CANN_ROOT:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCENDC_SOC_VERSION="${ASCENDC_SOC_VERSION:-Ascend910B3}"
export SOC_VERSION="${SOC_VERSION:-${ASCENDC_SOC_VERSION}}"
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

CANN_VERSION_FILE="${CANN_ROOT}/version.cfg"
[[ -f "${CANN_VERSION_FILE}" ]] || {
    echo "fatal: CANN version.cfg is missing: ${CANN_VERSION_FILE}" >&2
    exit 2
}
grep -Eq '^toolkit_running_version=.*:8\.1' "${CANN_VERSION_FILE}" || {
    echo "fatal: this direct campaign requires installed CANN 8.1" >&2
    exit 2
}

CAMPAIGN_FILES=(
    tools/generate_matmul_r5_l2_pair.py
    tools/analyze_matmul_r5_l2_pair.py
    tools/direct_matmul_tiling.py
    tools/profile_direct_matmul.py
    run_npu.sh
    scripts/build_all.sh
    scripts/env.sh
    runner/official_matmul_runner.cpp
    direct_matmul/kernel_entry.cpp
    direct_matmul/mat_mul_v3_tiling_data.h
    direct_matmul/runner.cpp
    "${CANN_VERSION_FILE}"
)
CAMPAIGN_ID="$({
    sha256sum "${CAMPAIGN_FILES[@]}"
    find cmake_npu host compat npu_cost_model -type f -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_r5_l2_pair_v1/${CAMPAIGN_ID}"
CATALOG="${CAMPAIGN_DIR}/workloads.csv"
CANDIDATES="${CAMPAIGN_DIR}/paired_packets.csv"
FREEZE_AUDIT="${CAMPAIGN_DIR}/packet_freeze.jsonl"
FREEZE_RECORD="${CAMPAIGN_DIR}/candidate_freeze.json"
OUT_STEM="${CAMPAIGN_DIR}/measurement"
DETAILS_DIR="${OUT_STEM}_details"
LOG_DIR="${CAMPAIGN_DIR}/logs"
ANALYSIS="${CAMPAIGN_DIR}/comparison.json"
mkdir -p "${CAMPAIGN_DIR}" "${DETAILS_DIR}" "${LOG_DIR}"

echo "CAMPAIGN_READY operator=matmul shapes=${SHAPE_COUNT} direct_records=${DIRECT_RECORDS} installed_references=${INSTALLED_REFERENCE_RECORDS} records=${TOTAL_RECORDS} device=${PHYSICAL_DEVICE}"
echo "design=ABBA_per_shape_unchanged_reference_modified_r5_modified_r5_unchanged_reference"
echo "provenance=original_selector_never_called_unchanged_reference_marked_and_credit_forbidden"
echo "UNCHANGED_PATH_POLICY direct_control=reference_only installed_matmulv3=reference_only performance_credit=forbidden"
echo "SHARED_KERNEL_PATH status=unchanged_execution_infrastructure performance_credit=forbidden"
echo "MODIFIED_PATH_POLICY rule=R5_L2_TWO_WAVE_THIN_N changed_fields=bank_l2_m_tile_count,bank_l2_m_tile_block"
echo "logs=${LOG_DIR}"

HOST_BUILD_HASH="$({
    find host compat -type f -print0
    printf '%s\0' scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
HOST_BUILD_STAMP="${ROOT}/build/.matmul_platform_host.sha256"
host_build_started_ns="$(date +%s%N)"
if [[ ! -x build/matmul_tiling_search || ! -f "${HOST_BUILD_STAMP}" ||
      "$(cat "${HOST_BUILD_STAMP}" 2>/dev/null || true)" != "${HOST_BUILD_HASH}" ]]; then
    echo "PLATFORM_HOST_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=host BUILD_JOBS=1 scripts/build_all.sh >"${CAMPAIGN_DIR}/platform_host_build.log" 2>&1; then
        echo "PLATFORM_HOST_BUILD failed log=${CAMPAIGN_DIR}/platform_host_build.log"
        tail -20 "${CAMPAIGN_DIR}/platform_host_build.log"
        exit 1
    fi
    printf '%s\n' "${HOST_BUILD_HASH}" >"${HOST_BUILD_STAMP}"
    echo "PLATFORM_HOST_BUILD passed"
    host_build_cached=0
else
    echo "PLATFORM_HOST_BUILD cached"
    host_build_cached=1
fi
host_build_wall_ms=$(( ($(date +%s%N) - host_build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=platform_host_build wall_ms=${host_build_wall_ms} cached=${host_build_cached}"

PLATFORM_TEMP="$(mktemp -d "${TMPDIR:-/tmp}/matmul-r5-platform.XXXXXX")"
cleanup() {
    if [[ -n "${PLATFORM_TEMP:-}" && "${PLATFORM_TEMP}" == */matmul-r5-platform.* ]]; then
        rm -rf -- "${PLATFORM_TEMP}"
    fi
}
trap cleanup EXIT
platform_started_ns="$(date +%s%N)"
build/matmul_tiling_search --platform-only --soc "${ASCENDC_SOC_VERSION}" --output "${PLATFORM_TEMP}/empty.csv" --all-output "${PLATFORM_TEMP}/empty_all.csv" --tiling-dir "${PLATFORM_TEMP}/tilings" |
    tee "${CAMPAIGN_DIR}/platform_query.log"
platform_wall_ms=$(( ($(date +%s%N) - platform_started_ns) / 1000000 ))
PLATFORM_LINE="$(sed -n '/^CANN platform=/{p;q;}' "${CAMPAIGN_DIR}/platform_query.log")"
platform_field() {
    printf '%s\n' "${PLATFORM_LINE}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
PLATFORM_AIC_CORES="$(platform_field cores)"
PLATFORM_L2_BYTES="$(platform_field L2)"
[[ "${PLATFORM_AIC_CORES}" =~ ^[0-9]+$ && "${PLATFORM_AIC_CORES}" -ge 20 ]] || {
    echo "fatal: platform query did not report at least 20 AIC cores" >&2
    exit 1
}
[[ "${PLATFORM_L2_BYTES}" =~ ^[0-9]+$ && "${PLATFORM_L2_BYTES}" -gt 0 ]] || {
    echo "fatal: platform query did not report a positive L2 capacity" >&2
    exit 1
}
export PLATFORM_AIC_CORES PLATFORM_L2_BYTES
echo "CAMPAIGN_STAGE_TIMING stage=platform_query wall_ms=${platform_wall_ms}"

freeze_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_r5_l2_pair.py --workloads "${CATALOG}" --candidates "${CANDIDATES}" --audit "${FREEZE_AUDIT}" --aic-cores "${PLATFORM_AIC_CORES}" --l2-bytes "${PLATFORM_L2_BYTES}"
CANDIDATE_SHA256="$(sha256sum "${CANDIDATES}" | cut -d' ' -f1)"
python3 - "${FREEZE_RECORD}" "${CANDIDATE_SHA256}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps({
    "schema": "matmul_r5_l2_candidate_freeze_v1",
    "candidate_sha256": sys.argv[2],
    "shape_count": 3,
    "direct_records": 12,
    "frozen_before_any_installed_reference_execution": True,
    "original_matmulv3_selector_executed": False,
    "unchanged_reference_performance_credit_eligible": False,
    "global_selector_claim_allowed": False,
}, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
freeze_wall_ms=$(( ($(date +%s%N) - freeze_started_ns) / 1000000 ))
echo "R5_CANDIDATES_FROZEN sha256=${CANDIDATE_SHA256} original_selector_executed=0 wall_ms=${freeze_wall_ms}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status":"complete"' "${ANALYSIS}"; then
    echo "R5_PAIR_CAMPAIGN_COMPLETE shapes=${SHAPE_COUNT} records=${TOTAL_RECORDS}"
    echo "comparison=${ANALYSIS} packets=${CANDIDATES} audit=${FREEZE_AUDIT} logs=${LOG_DIR}"
    exit 0
fi

RUNNER_BUILD_HASH="$({
    printf '%s\0' runner/official_matmul_runner.cpp cmake_npu/CMakeLists.txt scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
RUNNER_BUILD_STAMP="${ROOT}/build/.matmul_official_runner.sha256"
runner_build_started_ns="$(date +%s%N)"
if [[ ! -x build/official_matmul_runner || ! -f "${RUNNER_BUILD_STAMP}" ||
      "$(cat "${RUNNER_BUILD_STAMP}" 2>/dev/null || true)" != "${RUNNER_BUILD_HASH}" ]]; then
    echo "OFFICIAL_RUNNER_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh >"${CAMPAIGN_DIR}/official_runner_build.log" 2>&1; then
        echo "OFFICIAL_RUNNER_BUILD failed log=${CAMPAIGN_DIR}/official_runner_build.log"
        tail -20 "${CAMPAIGN_DIR}/official_runner_build.log"
        exit 1
    fi
    printf '%s\n' "${RUNNER_BUILD_HASH}" >"${RUNNER_BUILD_STAMP}"
    echo "OFFICIAL_RUNNER_BUILD passed"
    runner_build_cached=0
else
    echo "OFFICIAL_RUNNER_BUILD cached"
    runner_build_cached=1
fi
runner_build_wall_ms=$(( ($(date +%s%N) - runner_build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=official_runner_build wall_ms=${runner_build_wall_ms} cached=${runner_build_cached}"

direct_preflight_started_ns="$(date +%s%N)"
python3 tools/direct_matmul_tiling.py --candidates "${CANDIDATES}" --output-dir "${DETAILS_DIR}/direct_tilings" --manifest "${DETAILS_DIR}/direct_manifest.csv" --l2-bytes "${PLATFORM_L2_BYTES}" --aic-cores "${PLATFORM_AIC_CORES}" >/dev/null
direct_preflight_wall_ms=$(( ($(date +%s%N) - direct_preflight_started_ns) / 1000000 ))
echo "DIRECT_TILING_PREFLIGHT passed direct_records=${DIRECT_RECORDS} wall_ms=${direct_preflight_wall_ms}"

echo "NPU_MEASUREMENT_BEGIN direct_records=${DIRECT_RECORDS} installed_references=${INSTALLED_REFERENCE_RECORDS} records=${TOTAL_RECORDS}"
echo "UNCHANGED_PATH_EXECUTION rows=6 role=unchanged_direct_reference performance_credit_eligible=0"
echo "INSTALLED_PATH_EXECUTION rows=3 role=context_reference performance_credit_eligible=0"
echo "MODIFIED_PATH_EXECUTION rows=6 role=modified_r5_l2_candidate performance_credit_eligible=1"
export MEASUREMENT_JSONL_LOG_DIRECTORY="${LOG_DIR}"
export MEASUREMENT_JSONL_LOG_MAX_BYTES=52428800
profile_started_ns="$(date +%s%N)"
set +e
python3 tools/profile_direct_matmul.py --variant-builder scripts/build_all.sh --variant-runner-directory build/direct_runners --official-runner build/official_matmul_runner --candidates "${CANDIDATES}" --workloads "${CATALOG}" --manifest "${DETAILS_DIR}/direct_manifest.csv" --tiling-directory "${DETAILS_DIR}/direct_tilings" --profile-output "${DETAILS_DIR}/profile.csv" --samples-output "${DETAILS_DIR}/samples.csv" --official-output "${DETAILS_DIR}/official_profile.csv" --official-samples-output "${DETAILS_DIR}/official_samples.csv" --log-directory "${LOG_DIR}" --log-max-bytes "${MEASUREMENT_JSONL_LOG_MAX_BYTES}" --l2-bytes "${PLATFORM_L2_BYTES}" --aic-cores "${PLATFORM_AIC_CORES}" --device "${DEVICE_ID}" --warmup 2 --repeat 20 --samples 7 --progress-every 1 --require-fresh-direct-sequence 2>&1 | awk '
        /INSTALLED_PUBLIC_REFERENCE_|DIRECT_VARIANT_|DIRECT_MEASUREMENT_|fatal:|Traceback/ {
            print; fflush();
        }
    ' | tee "${CAMPAIGN_DIR}/measurement_progress.log"
profile_pipeline_status=("${PIPESTATUS[@]}")
profile_rc="${profile_pipeline_status[0]}"
if [[ "${profile_rc}" -eq 0 &&
      ( "${profile_pipeline_status[1]}" -ne 0 || "${profile_pipeline_status[2]}" -ne 0 ) ]]; then
    profile_rc=1
fi
set -e
profile_wall_ms=$(( ($(date +%s%N) - profile_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=npu_measurement wall_ms=${profile_wall_ms}" |
    tee -a "${CAMPAIGN_DIR}/measurement_progress.log"
if [[ "${profile_rc}" -ne 0 ]]; then
    echo "NPU_MEASUREMENT_INCOMPLETE log=${CAMPAIGN_DIR}/measurement_progress.log records=${LOG_DIR}"
    exit "${profile_rc}"
fi

POST_MEASUREMENT_SHA256="$(sha256sum "${CANDIDATES}" | cut -d' ' -f1)"
[[ "${POST_MEASUREMENT_SHA256}" == "${CANDIDATE_SHA256}" ]] || {
    echo "fatal: frozen R5 packets changed after reference execution" >&2
    exit 1
}
echo "R5_CANDIDATE_FREEZE_VERIFIED sha256=${POST_MEASUREMENT_SHA256}"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_r5_l2_pair.py --workloads "${CATALOG}" --candidates "${CANDIDATES}" --profile "${DETAILS_DIR}/profile.csv" --samples "${DETAILS_DIR}/samples.csv" --official-profile "${DETAILS_DIR}/official_profile.csv" --official-samples "${DETAILS_DIR}/official_samples.csv" --audit "${FREEZE_AUDIT}" --candidate-sha256 "${CANDIDATE_SHA256}" --output "${ANALYSIS}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "comparison=${ANALYSIS} packets=${CANDIDATES} audit=${FREEZE_AUDIT} logs=${LOG_DIR}"
