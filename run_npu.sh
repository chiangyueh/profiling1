#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=""
PHYSICAL_DEVICE="${PHYSICAL_NPU_ID:-2}"
SHAPE_COUNT=14
MODEL_RECORDS=14
TOTAL_RECORDS=28

usage() {
    cat <<'USAGE'
Usage: profiling/run_npu.sh --mode full [-d PHYSICAL_NPU_ID]

Deployment-style MatMul comparison.  For each of 14 FP16/NN shapes, a
bounded shape+hardware cost model freezes exactly one direct tiling before a
separate installed aclnnMatmul reference is run.  The NPU measures only those
two executions per shape; model selection never consumes reference tilings,
callbacks, RuntimeKb records, or measured latency.
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
MATMUL_V3_SOURCE="${CANN_ROOT}/opp/built-in/op_impl/ai_core/tbe/impl/ascendc/mat_mul_v3/mat_mul_v3.cpp"
[[ -f "${CANN_VERSION_FILE}" ]] || {
    echo "fatal: CANN version.cfg is missing: ${CANN_VERSION_FILE}" >&2
    exit 2
}
grep -Eq '^toolkit_running_version=.*:8\.1' "${CANN_VERSION_FILE}" || {
    echo "fatal: this direct campaign requires installed CANN 8.1" >&2
    exit 2
}
[[ -f "${MATMUL_V3_SOURCE}" ]] || {
    echo "fatal: installed CANN 8.1 MatMulV3 source is missing" >&2
    exit 2
}

catalog_started_ns="$(date +%s%N)"
CATALOG_TMP="$(mktemp "${TMPDIR:-/tmp}/matmul-deployment.XXXXXX.csv")"
cleanup() {
    [[ -f "${CATALOG_TMP:-}" ]] && rm -f -- "${CATALOG_TMP}"
}
trap cleanup EXIT
python3 tools/generate_matmul_deployment_workloads.py \
    --output "${CATALOG_TMP}" >/dev/null
catalog_wall_ms=$(( ($(date +%s%N) - catalog_started_ns) / 1000000 ))
CATALOG_SHA256="$(sha256sum "${CATALOG_TMP}" | cut -d' ' -f1)"

CAMPAIGN_ID="$({
    printf 'catalog_sha256=%s\n' "${CATALOG_SHA256}"
    sha256sum \
        tools/generate_matmul_deployment_workloads.py \
        tools/select_matmul_deployment_tilings.py \
        tools/analyze_matmul_deployment_comparison.py \
        tools/direct_matmul_tiling.py \
        tools/profile_direct_matmul.py \
        tools/rank_npu_results.py \
        run_npu.sh \
        scripts/run_search.sh \
        scripts/profile_npu.sh \
        runner/official_matmul_runner.cpp \
        direct_matmul/kernel_entry.cpp \
        direct_matmul/mat_mul_v3_tiling_data.h \
        direct_matmul/runner.cpp \
        npu_cost_model/*.py \
        "${CANN_VERSION_FILE}"
    find cmake_npu -type f -print0 | sort -z | xargs -0 sha256sum
} | sha256sum | cut -c1-20)"
CAMPAIGN_DIR="${ROOT}/results/matmul_deployment_compare_v1/${CAMPAIGN_ID}"
CATALOG="${CAMPAIGN_DIR}/workloads.csv"
CANDIDATES="${CAMPAIGN_DIR}/model_top1.csv"
SCORED_POOL="${CAMPAIGN_DIR}/model_scored_pool.csv"
SELECTION_AUDIT="${CAMPAIGN_DIR}/model_selection.jsonl"
TILING_DIR="${CAMPAIGN_DIR}/tilings"
OUT_STEM="${CAMPAIGN_DIR}/measurement"
DETAILS_DIR="${OUT_STEM}_details"
LOG_DIR="${CAMPAIGN_DIR}/logs"
ANALYSIS="${CAMPAIGN_DIR}/comparison.json"
FREEZE_RECORD="${CAMPAIGN_DIR}/candidate_freeze.json"
mkdir -p "${CAMPAIGN_DIR}" "${TILING_DIR}" "${LOG_DIR}"
cp "${CATALOG_TMP}" "${CATALOG}"

if [[ -s "${ANALYSIS}" ]] && grep -q '"status":"complete"' "${ANALYSIS}"; then
    echo "MATMUL_DEPLOYMENT_COMPARISON_COMPLETE shapes=${SHAPE_COUNT} records=${TOTAL_RECORDS}"
    echo "comparison=${ANALYSIS} selection=${SELECTION_AUDIT} logs=${LOG_DIR}"
    exit 0
fi

echo "CAMPAIGN_READY operator=matmul shapes=${SHAPE_COUNT} model_candidates=${MODEL_RECORDS} installed_references=${SHAPE_COUNT} records=${TOTAL_RECORDS} device=${PHYSICAL_DEVICE}"
echo "measurement=1_warmup+3_device_event_samples+validate_last_timed_output"
echo "design=one_independent_model_top1_plus_one_separate_installed_matmul_reference_per_shape"
echo "candidate_selection=bounded_shape_hardware_cost_model_max_32_internal_scores_per_shape"
echo "independence=no_reference_seed_no_callback_bytes_no_runtimekb_no_latency_history"
echo "logs=${LOG_DIR}"
echo "CAMPAIGN_STAGE_TIMING stage=workload_catalog wall_ms=${catalog_wall_ms}"

HOST_BUILD_HASH="$({
    find host compat -type f -print0
    printf '%s\0' scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
HOST_BUILD_STAMP="${ROOT}/build/.matmul_platform_host.sha256"
host_build_started_ns="$(date +%s%N)"
if [[ ! -x build/matmul_tiling_search || ! -f "${HOST_BUILD_STAMP}" || \
      "$(cat "${HOST_BUILD_STAMP}" 2>/dev/null || true)" != "${HOST_BUILD_HASH}" ]]; then
    echo "PLATFORM_HOST_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=host BUILD_JOBS=1 scripts/build_all.sh \
        >"${CAMPAIGN_DIR}/platform_host_build.log" 2>&1; then
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

RUNNER_BUILD_HASH="$({
    printf '%s\0' runner/official_matmul_runner.cpp \
        cmake_npu/CMakeLists.txt scripts/build_all.sh "${CANN_VERSION_FILE}"
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
RUNNER_BUILD_STAMP="${ROOT}/build/.matmul_official_runner.sha256"
runner_build_started_ns="$(date +%s%N)"
if [[ ! -x build/official_matmul_runner || ! -f "${RUNNER_BUILD_STAMP}" || \
      "$(cat "${RUNNER_BUILD_STAMP}" 2>/dev/null || true)" != "${RUNNER_BUILD_HASH}" ]]; then
    echo "OFFICIAL_RUNNER_BUILD begin jobs=1"
    if ! BUILD_COMPONENTS=official BUILD_JOBS=1 scripts/build_all.sh \
        >"${CAMPAIGN_DIR}/official_runner_build.log" 2>&1; then
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

export DISABLE_MEASUREMENT_HISTORY=1
export SEARCH_SCOPE=matmul_deployment_top1_v1
export SEARCH_OUTPUT="${CANDIDATES}"
export SEARCH_ALL_OUTPUT="${SCORED_POOL}"
export SEARCH_TILING_DIR="${TILING_DIR}"
export DEPLOYMENT_SELECTION_AUDIT="${SELECTION_AUDIT}"
export MEASUREMENT_JSONL_LOG_DIRECTORY="${LOG_DIR}"
export MEASUREMENT_JSONL_LOG_MAX_BYTES=52428800
if [[ -s "${CANDIDATES}" && -s "${SCORED_POOL}" && -s "${SELECTION_AUDIT}" ]]; then
    export REUSE_DEPLOYMENT_SELECTION=1
fi

SEARCH_LOG="${CAMPAIGN_DIR}/model_selection.log"
search_started_ns="$(date +%s%N)"
set +e
"${ROOT}/scripts/run_search.sh" "${CATALOG}" 2>&1 | \
    tee "${SEARCH_LOG}" | awk '
        /MODEL_TOP1 \[/ || /MATMUL_DEPLOYMENT_SELECTION/ || /fatal:/ || /^CANN platform=/ {
            print; fflush();
        }
    '
search_pipeline_status=("${PIPESTATUS[@]}")
search_rc="${search_pipeline_status[0]}"
if [[ "${search_rc}" -eq 0 && \
      ( "${search_pipeline_status[1]}" -ne 0 || "${search_pipeline_status[2]}" -ne 0 ) ]]; then
    search_rc=1
fi
set -e
search_wall_ms=$(( ($(date +%s%N) - search_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=model_tiling_selection wall_ms=${search_wall_ms}" | tee -a "${SEARCH_LOG}"
if [[ "${search_rc}" -ne 0 ]]; then
    echo "MODEL_SELECTION_FAILED log=${SEARCH_LOG}"
    exit "${search_rc}"
fi

PLATFORM_LINE="$(sed -n '/^CANN platform=/{p;q;}' "${SEARCH_LOG}")"
platform_field() {
    printf '%s\n' "${PLATFORM_LINE}" |
        sed -n "s/.*[[:space:]]$1=\\([0-9][0-9.]*\\).*/\\1/p"
}
PLATFORM_AIC_CORES="$(platform_field cores)"
PLATFORM_L0A_BYTES="$(platform_field L0A)"
PLATFORM_L0B_BYTES="$(platform_field L0B)"
PLATFORM_L0C_BYTES="$(platform_field L0C)"
PLATFORM_L1_BYTES="$(platform_field L1)"
PLATFORM_L2_BYTES="$(platform_field L2)"
PLATFORM_L2_BPC="$(platform_field L2_Bpc_per_core)"
PLATFORM_HBM_BPC="$(platform_field HBM_Bpc_per_core)"
for value in \
    "${PLATFORM_AIC_CORES}" "${PLATFORM_L0A_BYTES}" \
    "${PLATFORM_L0B_BYTES}" "${PLATFORM_L0C_BYTES}" \
    "${PLATFORM_L1_BYTES}" "${PLATFORM_L2_BYTES}" \
    "${PLATFORM_L2_BPC}" "${PLATFORM_HBM_BPC}"; do
    [[ -n "${value}" ]] || {
        echo "MODEL_SELECTION_FAILED invalid_platform_log=${SEARCH_LOG}"
        exit 1
    }
done
export PLATFORM_AIC_CORES PLATFORM_L0A_BYTES PLATFORM_L0B_BYTES
export PLATFORM_L0C_BYTES PLATFORM_L1_BYTES PLATFORM_L2_BYTES
export PLATFORM_L2_BPC PLATFORM_HBM_BPC

python3 - "${CATALOG}" "${CANDIDATES}" "${SCORED_POOL}" "${SELECTION_AUDIT}" \
    tools/select_matmul_deployment_tilings.py <<'PY'
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

catalog_path, candidate_path, pool_path, audit_path, selector_path = map(Path, sys.argv[1:])
catalog = list(csv.DictReader(catalog_path.open(newline="", encoding="utf-8")))
candidates = list(csv.DictReader(candidate_path.open(newline="", encoding="utf-8")))
pool = list(csv.DictReader(pool_path.open(newline="", encoding="utf-8")))
audits = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line]
ids = [row["workload_id"] for row in catalog]
candidate_ids = [row["workload_id"] for row in candidates]
pool_counts = Counter(row["workload_id"] for row in pool)
audit_ids = [row["workload_id"] for row in audits]
forbidden_columns = {
    column for column in (candidates[0] if candidates else {})
    if any(token in column.lower() for token in (
        "callback", "source_route", "runtime_kb", "measurement_history",
    ))
}
tree = ast.parse(selector_path.read_text(encoding="utf-8"), filename=str(selector_path))
banned_imports = {
    "tbe", "te", "refine_matmul_v3_candidates",
    "generate_matmul_source_frontier_candidates",
}
imports = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imports.append(node.module)
bad_imports = [name for name in imports if name.split(".")[0] in banned_imports]
source_text = selector_path.read_text(encoding="utf-8").lower()
bad_tokens = [token for token in (
    "op_tiling", "invoke_official", "parse_seed", "callback_raw_tiling_hex",
    "source_raw_tiling_hex", "runtime_kb", "measurement_history",
) if token in source_text]
candidate_by_id = {row["workload_id"]: row for row in candidates}
pool_top = {row["workload_id"]: row for row in pool if row.get("new_model_rank") == "1"}
audit_by_id = {row["workload_id"]: row for row in audits}
valid = (
    len(catalog) == 14
    and len(ids) == len(set(ids))
    and all(row.get("required_successful_tilings") == "1" for row in catalog)
    and len(candidates) == 14
    and candidate_ids == ids
    and len(set(candidate_ids)) == 14
    and audit_ids == ids
    and not forbidden_columns
    and not bad_imports
    and not bad_tokens
    and set(pool_counts) == set(ids)
    and set(pool_top) == set(ids)
    and all(0 < count <= 32 for count in pool_counts.values())
    and all(
        row.get("rank") == "1"
        and row.get("new_model_rank") == "1"
        and row.get("candidate_role") == "searched"
        and row.get("source") == "independent_hardware_cost_model"
        and row.get("model_input_source") == "shape_hardware_and_cost_model_only"
        and row.get("tiling_parameter_origin") == "independent_model_generation"
        and row.get("is_reserve") == "0"
        and row.get("required_successful_tilings") == "1"
        and int(row.get("candidate_budget") or 0) == 32
        and 0 < int(row.get("generated_candidate_count") or 0) <= 32
        and int(row.get("legal_candidate_count") or 0) == pool_counts[row["workload_id"]]
        and len(row.get("model_schedule_sha256", "")) == 64
        and pool_top[row["workload_id"]].get("model_schedule_sha256")
            == row.get("model_schedule_sha256")
        for row in candidates
    )
    and all(
        audit_by_id[workload_id].get("record_type") == "model_top1_frozen"
        and audit_by_id[workload_id].get("baseline_data_consumed") is False
        and audit_by_id[workload_id].get("measured_latency_consumed") is False
        and audit_by_id[workload_id].get("selected", {}).get("model_schedule_sha256")
            == candidate_by_id[workload_id].get("model_schedule_sha256")
        for workload_id in ids
    )
)
if not valid:
    raise SystemExit(
        "independent model candidate contract failed: "
        f"catalog={len(catalog)} candidates={len(candidates)} "
        f"pool={len(pool)} audits={len(audits)} "
        f"forbidden_columns={sorted(forbidden_columns)} "
        f"bad_imports={bad_imports} bad_tokens={bad_tokens}"
    )
PY
echo "MODEL_INDEPENDENCE_AUDIT passed shapes=${SHAPE_COUNT} selected=${MODEL_RECORDS} max_internal_per_shape=32"

CANDIDATE_SHA256="$(sha256sum "${CANDIDATES}" | cut -d' ' -f1)"
python3 - "${FREEZE_RECORD}" "${CANDIDATE_SHA256}" "${search_wall_ms}" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1])
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps({
    "schema": "matmul_deployment_candidate_freeze_v1",
    "candidate_sha256": sys.argv[2],
    "model_selection_wall_ms": int(sys.argv[3]),
    "shape_count": 14,
    "selected_candidates": 14,
    "frozen_before_baseline_execution": True,
}, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
echo "MODEL_CANDIDATES_FROZEN sha256=${CANDIDATE_SHA256} file=${CANDIDATES}"

direct_preflight_started_ns="$(date +%s%N)"
python3 tools/direct_matmul_tiling.py \
    --candidates "${CANDIDATES}" \
    --output-dir "${DETAILS_DIR}/direct_tilings" \
    --manifest "${DETAILS_DIR}/direct_manifest.csv" \
    --l2-bytes "${PLATFORM_L2_BYTES}" \
    --aic-cores "${PLATFORM_AIC_CORES}" >/dev/null
direct_preflight_wall_ms=$(( ($(date +%s%N) - direct_preflight_started_ns) / 1000000 ))
echo "DIRECT_TILING_PREFLIGHT passed candidates=${MODEL_RECORDS} wall_ms=${direct_preflight_wall_ms}"

echo "NPU_MEASUREMENT_BEGIN shapes=${SHAPE_COUNT} model_candidates=${MODEL_RECORDS} installed_references=${SHAPE_COUNT} records=${TOTAL_RECORDS}"
export KEEP_DETAILS=1
export WARMUP=1
export REPEAT=1
export SAMPLES=3
export PROFILE_PROGRESS_EVERY=1

PROFILE_LOG="${CAMPAIGN_DIR}/measurement_progress.log"
profile_started_ns="$(date +%s%N)"
set +e
"${ROOT}/scripts/profile_npu.sh" \
    "${CANDIDATES}" "${OUT_STEM}" "${CATALOG}" \
    2>&1 | awk '
        /INSTALLED_PUBLIC_REFERENCE_|DIRECT_VARIANT_|DIRECT_MEASUREMENT_|NPU_RESULTS_READY|fatal:|Traceback/ {
            print; fflush();
        }
    ' | tee "${PROFILE_LOG}"
profile_pipeline_status=("${PIPESTATUS[@]}")
profile_rc="${profile_pipeline_status[0]}"
if [[ "${profile_rc}" -eq 0 && \
      ( "${profile_pipeline_status[1]}" -ne 0 || "${profile_pipeline_status[2]}" -ne 0 ) ]]; then
    profile_rc=1
fi
set -e
profile_wall_ms=$(( ($(date +%s%N) - profile_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=npu_measurement wall_ms=${profile_wall_ms}" | tee -a "${PROFILE_LOG}"
if [[ "${profile_rc}" -ne 0 ]]; then
    echo "NPU_MEASUREMENT_INCOMPLETE log=${PROFILE_LOG} records=${LOG_DIR}"
    exit "${profile_rc}"
fi

POST_MEASUREMENT_SHA256="$(sha256sum "${CANDIDATES}" | cut -d' ' -f1)"
[[ "${POST_MEASUREMENT_SHA256}" == "${CANDIDATE_SHA256}" ]] || {
    echo "fatal: model candidates changed after baseline execution" >&2
    exit 1
}
echo "MODEL_CANDIDATE_FREEZE_VERIFIED sha256=${POST_MEASUREMENT_SHA256}"

for required in \
    "${DETAILS_DIR}/profile.csv" \
    "${DETAILS_DIR}/official_profile.csv"; do
    [[ -s "${required}" ]] || {
        echo "ANALYSIS_FAILED missing=${required}"
        exit 1
    }
done

analysis_started_ns="$(date +%s%N)"
python3 tools/analyze_matmul_deployment_comparison.py \
    --workloads "${CATALOG}" \
    --candidates "${CANDIDATES}" \
    --profile "${DETAILS_DIR}/profile.csv" \
    --official-profile "${DETAILS_DIR}/official_profile.csv" \
    --selection-audit "${SELECTION_AUDIT}" \
    --candidate-sha256 "${CANDIDATE_SHA256}" \
    --output "${ANALYSIS}"
analysis_wall_ms=$(( ($(date +%s%N) - analysis_started_ns) / 1000000 ))
echo "CAMPAIGN_STAGE_TIMING stage=analysis wall_ms=${analysis_wall_ms}"
echo "comparison=${ANALYSIS} model_tilings=${CANDIDATES} selection=${SELECTION_AUDIT} logs=${LOG_DIR} measurement_log=${PROFILE_LOG}"
