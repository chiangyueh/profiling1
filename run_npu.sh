#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
SAMPLES=7

usage() {
    printf '%s\n' \
        'Usage: ./run_npu.sh --mode full [-d PHYSICAL_NPU_ID]' \
        '' \
        'Runs four frozen MatMulV3 rule A/B workloads. Each shape has exactly' \
        'two unique packets: the byte-exact CANN 8.1 baseline and one rule' \
        'candidate. No cost model, tiling bank, lookup table, or candidate' \
        'search is used.'
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
        tools/generate_matmul_rule_ab.py \
        tools/analyze_matmul_rule_ab.py \
        scripts/build_all.sh \
        scripts/env.sh \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp
    find matmul_rule_selector -type f -name '*.py' -print0 |
        sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_rule_ab_v1/${CAMPAIGN_ID}"
PACKET_DIR="${CAMPAIGN_DIR}/packets"
MANIFEST="${CAMPAIGN_DIR}/manifest.csv"
SELECTION="${CAMPAIGN_DIR}/selection.jsonl"
VARIANT_DIR="${CAMPAIGN_DIR}/variants"
SEQUENCE_DIR="${CAMPAIGN_DIR}/sequence"
RUNNER_LOG="${CAMPAIGN_DIR}/direct_runner.log"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
BUILD_LOG_DIR="${CAMPAIGN_DIR}/build_logs"
mkdir -p "${CAMPAIGN_DIR}" "${BUILD_LOG_DIR}"

echo "CAMPAIGN_READY operator=matmul shapes=4 unique_tilings_per_shape=2 packets=8 device=${PHYSICAL_DEVICE}"
echo "design=baseline_then_one_rule_candidate_per_shape samples_per_packet=${SAMPLES}"
echo "selector=pure_python_rules baseline=frozen_byte_exact_cann81"
echo "forbidden=cost_model,history,repo_lookup,tiling_bank,candidate_search"
echo "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20"
echo "results=${CAMPAIGN_DIR}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status": "complete"' "${ANALYSIS}"; then
    echo "RULE_AB_COMPLETE cached=1 analysis=${ANALYSIS} summary=${SUMMARY}"
    exit 0
fi

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_rule_ab.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}"
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=packet_generation wall_ms=${generation_wall_ms}"

build_started_ns="$(date +%s%N)"
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    log="${BUILD_LOG_DIR}/${variant}.log"
    echo "DIRECT_VARIANT_BUILD begin variant=${variant} jobs=1"
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
    echo "DIRECT_VARIANT_BUILD passed variant=${variant}"
done
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=variant_build wall_ms=${build_wall_ms}"

: >"${RUNNER_LOG}"
measurement_started_ns="$(date +%s%N)"
packet_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    packet_index=$((packet_index + 1))
    echo "RULE_AB_MEASUREMENT ${packet_index}/8 variant=${variant} manifest=$(basename "${packet_manifest}")"
    "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup 1 \
        --repeat 1 \
        --samples "${SAMPLES}" | tee -a "${RUNNER_LOG}"
done
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=npu_measurement wall_ms=${measurement_wall_ms}"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_rule_ab.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUNNER_LOG}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "RULE_AB_OUTPUT analysis=${ANALYSIS} summary=${SUMMARY} packets=${PACKET_DIR}"
