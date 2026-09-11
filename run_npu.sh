#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
SAMPLES=7
EXPECTED_BRANCHES=12

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Measures every kernel branch dispatched by the installed CANN 8.1' \
        'MatMulV3 implementation. Each branch has one independently generated' \
        'improved packet and one installed public-API reference measurement.' \
        'No candidate search, cost model, history, or official tiling seed is used.'
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
        matmul_rule_selector/branch_contract.json
    find matmul_rule_selector -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_rule_matrix_v2/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/improved_manifest.csv"
WORKLOADS="${CAMPAIGN_DIR}/official_workloads.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
RUNNER_LOG="${CAMPAIGN_DIR}/direct_runner.log"
OFFICIAL_LOG="${CAMPAIGN_DIR}/official_runner.log"
OFFICIAL_PROFILE="${CAMPAIGN_DIR}/official_profile.csv"
OFFICIAL_SAMPLES="${CAMPAIGN_DIR}/official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
BUILD_LOG_DIR="${CAMPAIGN_DIR}/build_logs"
mkdir -p "${CAMPAIGN_DIR}" "${BUILD_LOG_DIR}"

on_error() {
    local rc=$?
    echo "RULE_MATRIX_FATAL rc=${rc} line=${BASH_LINENO[0]} results=${CAMPAIGN_DIR}" >&2
    exit "${rc}"
}
trap on_error ERR

echo "CAMPAIGN_READY operator=matmul installed_branches=${EXPECTED_BRANCHES} official_shapes=${EXPECTED_BRANCHES} improved_packets=${EXPECTED_BRANCHES} device=${PHYSICAL_DEVICE}"
echo "measurement=1_warmup+${SAMPLES}_device_event_samples+validate_last_timed_output"
echo "design=one_independent_improved_packet_and_one_installed_public_reference_per_branch"
echo "selector=shape_and_hardware_formula_only"
echo "forbidden=cost_model,history,repo_lookup,tiling_bank,candidate_search,official_tiling_seed"
echo "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20"
echo "results=${CAMPAIGN_DIR}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status": "complete"' "${ANALYSIS}"; then
    echo "RULE_MATRIX_COMPLETE cached=1 analysis=${ANALYSIS} summary=${SUMMARY}"
    exit 0
fi

coverage_started_ns="$(date +%s%N)"
python3 tools/check_matmul_rule_coverage.py
coverage_wall_ms=$(( ($(date +%s%N) - coverage_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=branch_coverage wall_ms=${coverage_wall_ms}"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_rule_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}" \
    --workloads "${WORKLOADS}"
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=packet_generation wall_ms=${generation_wall_ms}"

official_build_started_ns="$(date +%s%N)"
official_build_log="${BUILD_LOG_DIR}/official.log"
echo "OFFICIAL_RUNNER_BUILD begin jobs=1"
if ! BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh >"${official_build_log}" 2>&1; then
    echo "OFFICIAL_RUNNER_BUILD failed log=${official_build_log}" >&2
    tail -30 "${official_build_log}" >&2
    exit 1
fi
echo "OFFICIAL_RUNNER_BUILD passed"
official_build_wall_ms=$(( ($(date +%s%N) - official_build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=official_runner_build wall_ms=${official_build_wall_ms}"

build_started_ns="$(date +%s%N)"
variant_count=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    log="${BUILD_LOG_DIR}/${variant}.log"
    variant_count=$((variant_count + 1))
    echo "DIRECT_VARIANT_BUILD ${variant_count}/${EXPECTED_BRANCHES} begin variant=${variant} jobs=1"
    if ! BUILD_COMPONENTS=variant BUILD_JOBS=1 DIRECT_KERNEL_TARGET="${target}" \
        scripts/build_all.sh >"${log}" 2>&1; then
        echo "DIRECT_VARIANT_BUILD failed variant=${variant} log=${log}" >&2
        tail -30 "${log}" >&2
        exit 1
    fi
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    [[ -x "${runner}" ]] || {
        echo "fatal: direct runner missing after build: ${runner}" >&2
        exit 1
    }
    "${runner}" --manifest "${variant_manifest}" --validate-input >/dev/null
    echo "DIRECT_VARIANT_BUILD ${variant_count}/${EXPECTED_BRANCHES} passed variant=${variant}"
done
[[ "${variant_count}" -eq "${EXPECTED_BRANCHES}" ]] || {
    echo "fatal: generated ${variant_count} variants, expected ${EXPECTED_BRANCHES}" >&2
    exit 1
}
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=variant_build wall_ms=${build_wall_ms}"

official_started_ns="$(date +%s%N)"
echo "INSTALLED_PUBLIC_REFERENCE_BEGIN shapes=${EXPECTED_BRANCHES}"
"${ROOT}/build/official_matmul_runner" \
    --candidates "${WORKLOADS}" \
    --output "${OFFICIAL_PROFILE}" \
    --samples-output "${OFFICIAL_SAMPLES}" \
    --device "${DEVICE_ID}" \
    --warmup 1 \
    --repeat 1 \
    --samples "${SAMPLES}" \
    --numeric-preflight-max-mib 256 \
    --structured-full-preflight \
    --validate-after-measurement 2>&1 | tee "${OFFICIAL_LOG}"
official_wall_ms=$(( ($(date +%s%N) - official_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=official_measurement wall_ms=${official_wall_ms}"

: >"${RUNNER_LOG}"
measurement_started_ns="$(date +%s%N)"
packet_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    packet_index=$((packet_index + 1))
    echo "IMPROVED_MEASUREMENT ${packet_index}/${EXPECTED_BRANCHES} variant=${variant} manifest=$(basename "${packet_manifest}")"
    "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup 1 \
        --repeat 1 \
        --samples "${SAMPLES}" 2>&1 | tee -a "${RUNNER_LOG}"
done
[[ "${packet_index}" -eq "${EXPECTED_BRANCHES}" ]] || {
    echo "fatal: measured ${packet_index} improved packets, expected ${EXPECTED_BRANCHES}" >&2
    exit 1
}
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=improved_measurement wall_ms=${measurement_wall_ms}"

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
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "RULE_MATRIX_OUTPUT analysis=${ANALYSIS} summary=${SUMMARY} packets=${PACKET_DIR}"
