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
RUNNER_LOG="${CAMPAIGN_DIR}/direct_runner.log"
REFERENCE_PROFILE="${CAMPAIGN_DIR}/historical_official_profile.csv"
REFERENCE_SAMPLES="${CAMPAIGN_DIR}/historical_official_samples.csv"
ANALYSIS="${CAMPAIGN_DIR}/analysis.json"
SUMMARY="${CAMPAIGN_DIR}/summary.csv"
BUILD_LOG_DIR="${CAMPAIGN_DIR}/build_logs"
mkdir -p "${CAMPAIGN_DIR}" "${BUILD_LOG_DIR}"

on_error() {
    local rc=$?
    echo "RULE_VALIDATION_FATAL rc=${rc} line=${BASH_LINENO[0]} results=${CAMPAIGN_DIR}" >&2
    exit "${rc}"
}
trap on_error ERR

echo "CAMPAIGN_READY operator=matmul host_branches=${EXPECTED_BRANCHES} rule_groups=${EXPECTED_RULE_GROUPS} npu_shapes=${VALIDATION_SHAPES} improved_measurements=${VALIDATION_SHAPES} reused_official_references=${VALIDATION_SHAPES} device=${PHYSICAL_DEVICE}"
echo "measurement=${WARMUP}_warmup+${SAMPLES}_device_event_samples+repeat_${REPEAT}+validate_last_timed_output"
echo "selection=predeclared_structural_decision_boundaries_not_latency_ranked"
echo "selector=shape_and_hardware_formula_only"
echo "official_reference=audited_existing_cann81_measurements_not_rerun"
echo "forbidden=cost_model,history_lookup_at_runtime,repo_lookup,tiling_bank,candidate_search,official_tiling_seed"
echo "CANN_ENV root=${CANN_ROOT} soc=${SOC_VERSION} aic=20"
echo "results=${CAMPAIGN_DIR}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status": "complete"' "${ANALYSIS}"; then
    echo "DISCRIMINATIVE_COMPLETE cached=1 analysis=${ANALYSIS} summary=${SUMMARY}"
    exit 0
fi

coverage_started_ns="$(date +%s%N)"
coverage_json="$(python3 tools/check_matmul_rule_coverage.py)"
python3 -c 'import json,sys; x=json.loads(sys.argv[1]); assert x["installed_cann81_dispatch_branches"] == 12 and x["modified_rule_groups"] == 14 and x["modified_branch_witnesses"] == 12 and x["status"] == "PASS"' "${coverage_json}"
printf '%s\n' "${coverage_json}"
coverage_wall_ms=$(( ($(date +%s%N) - coverage_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=host_rule_coverage wall_ms=${coverage_wall_ms}"

generation_started_ns="$(date +%s%N)"
python3 tools/generate_matmul_rule_matrix.py \
    --output-dir "${PACKET_DIR}" \
    --manifest "${MANIFEST}" \
    --selection "${SELECTION}" \
    --variant-dir "${VARIANT_DIR}" \
    --sequence-dir "${SEQUENCE_DIR}" \
    --historical-profile "${REFERENCE_PROFILE}" \
    --historical-samples "${REFERENCE_SAMPLES}"
generation_wall_ms=$(( ($(date +%s%N) - generation_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=discriminative_packet_generation wall_ms=${generation_wall_ms}"

build_started_ns="$(date +%s%N)"
variant_count=0
for variant_manifest in "${VARIANT_DIR}"/*.csv; do
    variant="$(basename "${variant_manifest}" .csv)"
    dtype="${variant%%_k*}"
    suffix="${variant##*_k}"
    target="direct_matmul_kernel_${dtype}_${suffix}"
    log="${BUILD_LOG_DIR}/${variant}.log"
    variant_count=$((variant_count + 1))
    echo "DIRECT_VARIANT_BUILD ${variant_count}/${VALIDATION_SHAPES} begin variant=${variant} jobs=1"
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
    echo "DIRECT_VARIANT_BUILD ${variant_count}/${VALIDATION_SHAPES} passed variant=${variant}"
done
[[ "${variant_count}" -eq "${VALIDATION_SHAPES}" ]] || {
    echo "fatal: generated ${variant_count} variants, expected ${VALIDATION_SHAPES}" >&2
    exit 1
}
build_wall_ms=$(( ($(date +%s%N) - build_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=selected_variant_build wall_ms=${build_wall_ms}"

: >"${RUNNER_LOG}"
measurement_started_ns="$(date +%s%N)"
packet_index=0
for packet_manifest in "${SEQUENCE_DIR}"/*.csv; do
    filename="$(basename "${packet_manifest}" .csv)"
    variant="${filename##*__}"
    runner="${ROOT}/build/direct_runners/direct_matmul_${variant}"
    packet_index=$((packet_index + 1))
    echo "IMPROVED_MEASUREMENT ${packet_index}/${VALIDATION_SHAPES} variant=${variant} manifest=$(basename "${packet_manifest}")"
    "${runner}" \
        --manifest "${packet_manifest}" \
        --device "${DEVICE_ID}" \
        --warmup "${WARMUP}" \
        --repeat "${REPEAT}" \
        --samples "${SAMPLES}" 2>&1 | tee -a "${RUNNER_LOG}"
done
[[ "${packet_index}" -eq "${VALIDATION_SHAPES}" ]] || {
    echo "fatal: measured ${packet_index} improved packets, expected ${VALIDATION_SHAPES}" >&2
    exit 1
}
measurement_wall_ms=$(( ($(date +%s%N) - measurement_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=improved_measurement wall_ms=${measurement_wall_ms}"

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_rule_matrix.py \
    --manifest "${MANIFEST}" \
    --runner-log "${RUNNER_LOG}" \
    --reference-profile "${REFERENCE_PROFILE}" \
    --reference-samples "${REFERENCE_SAMPLES}" \
    --selection "${SELECTION}" \
    --output-json "${ANALYSIS}" \
    --output-csv "${SUMMARY}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "DISCRIMINATIVE_OUTPUT analysis=${ANALYSIS} summary=${SUMMARY} packets=${PACKET_DIR}"
