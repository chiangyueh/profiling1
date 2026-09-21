#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH
unset MATMUL_V3_VECTOR_CROSSOVER_SWEEP

host_build="${PWD}/build"
build_log="$(mktemp)"
run_log="$(mktemp)"
#NEW
selected_shapes="$(mktemp)"

cleanup() {
    rm -f "${build_log}" "${run_log}" "${selected_shapes}"
}
trap cleanup EXIT

if ! cmake -S . -B "${host_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUSTOM=FALSE \
    -DENABLE_BINARY=FALSE \
    -DENABLE_PACKAGE=FALSE \
    -DENABLE_TEST=FALSE \
    -DASCEND_OP_NAME=mat_mul_v3 \
    -DASCEND_COMPILE_OPS=mat_mul_v3 >"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi
if ! cmake --build "${host_build}" --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

#NEW
v3_host_library="${host_build}/libophost_nn.so"
if [[ ! -f "${v3_host_library}" ]]; then
    echo "fatal: independently built MatMulV3 host library is missing" >&2
    exit 1
fi

#NEW
official_opapi_nn_library=""
for candidate in \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_nn.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_nn.so" \
    "${ASCEND_OPP_PATH}/lib64/libopapi_nn.so"; do
    if [[ -f "${candidate}" ]]; then
        official_opapi_nn_library="${candidate}"
        break
    fi
done
if [[ -z "${official_opapi_nn_library}" ]]; then
    echo "fatal: installed CANN libopapi_nn.so is missing" >&2
    exit 1
fi
official_opapi_nn_dir="$(dirname -- "${official_opapi_nn_library}")"

#NEW
official_opapi_math_library=""
for candidate in \
    "${ASCEND_OPP_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_math.so"; do
    if [[ -f "${candidate}" ]]; then
        official_opapi_math_library="${candidate}"
        break
    fi
done
if [[ -z "${official_opapi_math_library}" ]]; then
    echo "fatal: installed CANN libopapi_math.so is missing" >&2
    exit 1
fi
official_opapi_math_dir="$(dirname -- "${official_opapi_math_library}")"
runtime_path="${official_opapi_nn_dir}:${official_opapi_math_dir}:${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

#NEW
official_legacy_common_library=""
for candidate in \
    "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so" \
    "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"; do
    if [[ -f "${candidate}" ]]; then
        official_legacy_common_library="${candidate}"
        break
    fi
done
if [[ -z "${official_legacy_common_library}" ]]; then
    echo "fatal: installed CANN libophost_comm_legacy.so is missing" >&2
    exit 1
fi
#NEW
if ! nm -D "${official_opapi_nn_library}" | awk '$3 ~ /^aclnnMatmulGetWorkspaceSize(@.*)?$/ {found=1} END {exit !found}'; then
    echo "fatal: installed CANN libopapi_nn.so has no aclnnMatmulGetWorkspaceSize" >&2
    exit 1
fi
if ! nm -D "${official_legacy_common_library}" | awk '$3 ~ /^LegacyMmCheckHitV3Shape(@.*)?$/ {found=1} END {exit !found}'; then
    echo "fatal: installed CANN legacy library has no MatMul V3 selector" >&2
    exit 1
fi

example_source="matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp"
example_binary="${host_build}/test_aclnn_matmul"
runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
common_link_args=(
    -std=gnu++17 \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${official_opapi_nn_library}" "${official_opapi_math_library}" \
    "${runtime_library}" -lnnopbase -lregister -lopp_registry -ldl
)
if ! g++ "${example_source}" "${common_link_args[@]}" \
    -Wl,-rpath,"${official_opapi_nn_dir}:${official_opapi_math_dir}" \
    -o "${example_binary}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

#NEW
# The runner must use CANN's complete official API. Only the MatMulV3 host
# tiler is repository-local.
loaded_opapi="$(LD_LIBRARY_PATH="${runtime_path}" ldd "${example_binary}" | awk '$1 == "libopapi_nn.so" {print $3}')"
loaded_math="$(LD_LIBRARY_PATH="${runtime_path}" ldd "${example_binary}" | awk '$1 == "libopapi_math.so" {print $3}')"
if [[ -z "${loaded_opapi}" || "$(readlink -f -- "${loaded_opapi}")" != "$(readlink -f -- "${official_opapi_nn_library}")" ]]; then
    echo "fatal: runner did not resolve installed CANN libopapi_nn.so" >&2
    exit 1
fi
if [[ -z "${loaded_math}" || "$(readlink -f -- "${loaded_math}")" != "$(readlink -f -- "${official_opapi_math_library}")" ]]; then
    echo "fatal: runner did not resolve installed CANN libopapi_math.so" >&2
    exit 1
fi

#NEW
export MATMUL_V3_HOST_LIBRARY="${v3_host_library}"
export MATMUL_LEGACY_COMMON_LIBRARY="${official_legacy_common_library}"
export LD_LIBRARY_PATH="${runtime_path}"

#NEW
build_vector_split_k_dot_kernel() {
    local kernel_name="MatMulV3_VectorSplitKDot"
    local kernel_bin_dir="${host_build}/vector_split_k_dot_v4_bin"
    local custom_object="${kernel_bin_dir}/${kernel_name}.o"
    local custom_json="${kernel_bin_dir}/${kernel_name}.json"

    if [[ -f "${custom_object}" && -f "${custom_json}" ]] &&
       ! find matmul/mat_mul_v3/op_kernel -type f -newer "${custom_object}" -print -quit | grep -q .; then
        printf '%s\n' "${custom_object}"
        return 0
    fi

    local tbe_ascendc="${host_build}/tbe/ascendc"
    local tbe_dynamic="${host_build}/tbe/dynamic"
    local param_dir="${host_build}/vector_split_k_dot_v4_params"
    rm -rf -- "${kernel_bin_dir}" "${param_dir}" \
        "${tbe_ascendc}/mat_mul_v3"
    mkdir -p -- "${kernel_bin_dir}" "${tbe_ascendc}/mat_mul_v3" \
        "${tbe_ascendc}/common/act" "${tbe_ascendc}/common/matmul_act" \
        "${tbe_dynamic}" "${param_dir}"
    cp -a matmul/mat_mul_v3/op_kernel/. "${tbe_ascendc}/mat_mul_v3/"
    cp -a common/act/. "${tbe_ascendc}/common/act/"
    cp -a matmul/common/matmul_act/. "${tbe_ascendc}/common/matmul_act/"

    local ops_info="${host_build}/autogen/exc/aic-ascend910b-ops-info.ini"
    local opc_options="${host_build}/autogen/custom_opc_options.ini"
    if [[ ! -f "${ops_info}" || ! -f "${opc_options}" ]]; then
        echo "fatal: MatMulV3 kernel metadata was not generated" >&2
        return 1
    fi
    python3 scripts/util/ascendc_impl_build.py "${ops_info}" "" "" \
        "${tbe_ascendc}" "${tbe_dynamic}" "${host_build}/autogen" >>"${build_log}" 2>&1
    python3 scripts/util/ascendc_bin_param_build.py "${ops_info}" "${param_dir}" ascend910b \
        --opc-config-file "${opc_options}" --ops MatMulV3 >>"${build_log}" 2>&1

    local fp32_param
    fp32_param="$(python3 - "${param_dir}" <<'PY'
import glob
import json
import os
import sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*_param.json"))):
    with open(path, encoding="utf-8") as stream:
        node = json.load(stream)["op_list"][0]
    required = [item for item in node["inputs"] if item.get("paramType") == "required"]
    outputs = [item for item in node["outputs"] if item.get("paramType") == "required"]
    if (len(required) == 2 and len(outputs) == 1 and
            all(item.get("dtype") == "float32" and item.get("format") == "ND" for item in required + outputs)):
        print(path)
        break
PY
)"
    if [[ -z "${fp32_param}" || ! -f "${fp32_param}" || ! -f "${tbe_dynamic}/mat_mul_v3.py" ]]; then
        echo "fatal: FP32/ND MatMulV3 kernel input was not generated" >&2
        return 1
    fi

    python3 - "${fp32_param}" "${kernel_name}" <<'PY'
import json
import re
import sys

path, kernel_name = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)
nodes = data.get("op_list", [])
if len(nodes) != 1:
    raise SystemExit("expected one MatMulV3 compile node")
nodes[0]["bin_filename"] = kernel_name
with open(path, "w", encoding="utf-8") as stream:
    json.dump(data, stream, separators=(",", ":"))
PY

    asc_opc "${tbe_dynamic}/mat_mul_v3.py" --main_func=mat_mul_v3 \
        --input_param="${fp32_param}" --soc_version=Ascend910B1 --output="${kernel_bin_dir}" \
        --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic \
        --deterministic=false --tiling_key=2162688 >>"${build_log}" 2>&1
    if [[ ! -f "${kernel_bin_dir}/${kernel_name}.o" || ! -f "${kernel_bin_dir}/${kernel_name}.json" ]]; then
        echo "fatal: VECTOR_SPLIT_K_DOT single-key kernel was not generated" >&2
        return 1
    fi
    if ! python3 - "${custom_json}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
keys = metadata.get("supportInfo", {}).get("tilingKey", [])
kernel_name = "MatMulV3_VectorSplitKDot"
kernel_list = metadata.get("kernelList", [])
if (metadata.get("core_type") != "AIV" or "2162688" not in keys or
        metadata.get("binFileName") != kernel_name or
        metadata.get("kernelName") != kernel_name or
        not any(item.get("kernelName") == kernel_name + "_2162688" for item in kernel_list)):
    raise SystemExit(1)
PY
    then
        echo "fatal: VECTOR_SPLIT_K_DOT kernel metadata has the wrong core type or tiling key" >&2
        return 1
    fi
    if ! readelf -Ws "${custom_object}" | awk \
        '$4 == "FUNC" && $5 == "GLOBAL" && $8 == "MatMulV3_VectorSplitKDot_2162688" {found=1} END {exit !found}'; then
        echo "fatal: VECTOR_SPLIT_K_DOT ELF entry is missing" >&2
        return 1
    fi
    printf '%s\n' "${custom_object}"
}

#NEW
if [[ "${MATMUL_V3_CAMPAIGN:-vector_split_k_dot}" == "vector_split_k_dot" ]]; then
    if [[ "$#" -ne 0 ]]; then
        echo "fatal: VECTOR_SPLIT_K_DOT validation uses its own theory-directed shapes" >&2
        exit 2
    fi
    vector_binary="$(build_vector_split_k_dot_kernel)" || {
        cat "${build_log}" >&2
        exit 1
    }
    vector_shapes=()
    vector_mn_pairs=(
        1 17  1 32  1 48  1 64  1 96  1 128  1 192  1 256  1 384  1 512  1 768  1 1024
        2 17  2 32  2 48  2 64  2 96  2 128  2 192  2 256  2 384  2 512
        3 17  3 20  3 24  3 32  3 48  3 64  3 96  3 128
        4 17  4 24  4 32  4 48  4 64  4 96
        8 17  8 24  8 32  8 48
    )
    for vector_k in 4096 16384; do
        for ((pair_index = 0; pair_index < ${#vector_mn_pairs[@]}; pair_index += 2)); do
            vector_shapes+=("${vector_mn_pairs[pair_index]}" "${vector_mn_pairs[pair_index + 1]}" "${vector_k}")
        done
    done
    vector_shapes+=(
        1 48 2048  2 32 2048  3 18 2048  4 32 2048
        1 48 8192  2 32 8192  3 18 8192  4 32 8192
        1 48 32768 2 32 32768 3 18 32768 4 32 32768
        12 17 8192 12 32 8192 16 17 8192 16 32 8192
        24 17 8192 24 32 8192 32 17 8192 32 32 8192
    )
    common_measurement_env=(
        MATMUL_DATA_TYPE=fp32 MATMUL_OUTPUT_DATA_TYPE=fp32
        MATMUL_A_TRANSPOSE=0 MATMUL_B_TRANSPOSE=1 MATMUL_V3_ONLY=1
        MATMUL_TRANSPOSED_COLUMN_PATTERN=1
        MATMUL_V3_SHRINK_IDLE_CORES=0 MATMUL_V3_WARMUP=3 MATMUL_V3_REPEATS=30
    )
    env -u ASCEND_CUSTOM_OPP_PATH -u MATMUL_V3_FORCE_CORE_NUM -u MATMUL_V3_MEASUREMENT_PLAN \
        "${common_measurement_env[@]}" MATMUL_V3_SKIP_ROUTE_PREFILTER=0 \
        MATMUL_V3_DISABLE_REPO_LOOKUP=0 \
        MATMUL_V3_DISABLE_VECTOR_SPLIT_K_DOT=1 \
        MATMUL_V3_MEASUREMENT_MODE=official "${example_binary}" "${vector_shapes[@]}" >"${run_log}"
    candidate_shapes=()
    while read -r candidate_m candidate_n candidate_k; do
        candidate_shapes+=("${candidate_m}" "${candidate_n}" "${candidate_k}")
    done < <(python3 - "${run_log}" <<'PY'
import json
import re
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("mode") != "official" or row.get("branch") != "BASE" or row.get("status") != "OK":
        continue
    match = re.fullmatch(r"M(\d+)_N(\d+)_K(\d+)_NT", row.get("shape", ""))
    if match:
        print(*match.groups())
PY
)
    if [[ "${#candidate_shapes[@]}" -eq 0 ]]; then
        echo "fatal: none of the theory-directed shapes reached the official BASE fallback" >&2
        exit 1
    fi
    env -u ASCEND_CUSTOM_OPP_PATH -u MATMUL_V3_FORCE_CORE_NUM -u MATMUL_V3_MEASUREMENT_PLAN \
        "${common_measurement_env[@]}" MATMUL_V3_VECTOR_BINARY="${vector_binary}" \
        MATMUL_V3_SKIP_ROUTE_PREFILTER=1 \
        MATMUL_V3_DISABLE_REPO_LOOKUP=1 \
        MATMUL_V3_VECTOR_CROSSOVER_SWEEP=1 \
        MATMUL_V3_DISABLE_VECTOR_SPLIT_K_DOT=0 MATMUL_V3_MEASUREMENT_MODE=vector_split_k_dot \
        "${example_binary}" "${candidate_shapes[@]}" >>"${run_log}"
    if ! python3 - "${run_log}" <<'PY'
import json
import re
import sys

official = {}
candidates = []
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue
    if row.get("mode") == "official" and row.get("branch") == "BASE" and row.get("status") == "OK":
        official[row["shape"]] = row
    elif row.get("mode") == "vector_split_k_dot":
        candidates.append(row)

passed = 0
for candidate in candidates:
    reference = official.get(candidate.get("shape"))
    candidate_tiling = candidate.get("tiling") if isinstance(candidate.get("tiling"), dict) else {}
    reference_tiling = reference.get("tiling") if reference and isinstance(reference.get("tiling"), dict) else {}
    valid = (
        reference is not None
        and candidate.get("branch") == "VECTOR_SPLIT_K_DOT"
        and candidate.get("status") == "OK"
        and candidate.get("correctness") == "PASS"
        and candidate_tiling.get("tiling_key") == 2162688
    )
    if valid:
        passed += 1
    official_latency = reference.get("latency_ms") if reference else None
    candidate_latency = candidate.get("latency_ms")
    delta = None
    if official_latency not in (None, 0) and candidate_latency is not None:
        delta = round((candidate_latency / official_latency - 1.0) * 100.0, 3)
    match = re.fullmatch(r"M(\d+)_N(\d+)_K(\d+)_NT", candidate.get("shape", ""))
    m, n, k = (map(int, match.groups()) if match else (0, 0, 0))
    output_dots = m * n
    padded_dots = ((m + 15) // 16 * 16) * ((n + 15) // 16 * 16) if output_dots else 0
    candidate_core = candidate.get("actual_core")
    print(json.dumps({
        "shape": candidate.get("shape"),
        "output_dots": output_dots,
        "cube_padding_ratio": round(padded_dots / output_dots, 3) if output_dots else None,
        "vector_partition": ("split_k" if candidate_core and candidate_core > output_dots
                             else "output_parallel"),
        "official_branch": reference.get("branch") if reference else None,
        "candidate_branch": candidate.get("branch"),
        "official_tiling_key": reference_tiling.get("tiling_key"),
        "candidate_tiling_key": candidate_tiling.get("tiling_key"),
        "official_core": reference.get("actual_core") if reference else None,
        "candidate_core": candidate_core,
        "official_latency_ms": official_latency,
        "candidate_latency_ms": candidate_latency,
        "delta_pct": delta,
        "correctness": candidate.get("correctness"),
        "candidate_result_code": candidate.get("result_code"),
        "candidate_failure_stage": candidate.get("failure_stage"),
        "candidate_failure_detail": candidate.get("failure_detail"),
        "status": "OK" if valid else "INVALID_CANDIDATE"
    }, separators=(",", ":")))

raise SystemExit(0 if passed > 0 else 1)
PY
    then
        echo "fatal: VECTOR_SPLIT_K_DOT produced no correct NPU measurement" >&2
        exit 1
    fi
    exit 0
fi

#NEW
if [[ -z "${MATMUL_V3_CAMPAIGN:-}" || "${MATMUL_V3_CAMPAIGN:-}" == "all_io_core_sweep" ]]; then
    if [[ "$#" -ne 0 ]]; then
        echo "fatal: all-route/I-O core sweep uses its own branch-directed shapes" >&2
        exit 2
    fi
    all_io_selected="${host_build}/all_io_core_sweep_v1_selected.tsv"
    priority_selected="${host_build}/all_io_core_sweep_priority_v1_selected.tsv"
    all_io_checkpoint="${host_build}/all_io_core_sweep_v1_checkpoint.jsonl"
    #NEW: The exhaustive 4..20 pass is a coarse response-curve sweep.  One
    # warmup removes the first-launch path and ten timed launches provide a
    # long enough aggregate event for the shortest observed kernels.  Only
    # the eventual formula winner and core-20 reference need deeper retests.
    export MATMUL_V3_WARMUP="${MATMUL_V3_WARMUP:-1}"
    export MATMUL_V3_REPEATS="${MATMUL_V3_REPEATS:-10}"
    export MATMUL_V3_RUNNER_IDLE_TIMEOUT_SECONDS="${MATMUL_V3_RUNNER_IDLE_TIMEOUT_SECONDS:-180}"
    python3 scripts/core_oracle_sampler.py select-all-io-core-sweep \
        --runner "${example_binary}" --selected "${priority_selected}" \
        --run-log "${run_log}" --quota 50 --discovery-batch 64 \
        --priority-only --resume-only
    python3 scripts/core_oracle_sampler.py measure-all-io-core-sweep \
        --runner "${example_binary}" --selected "${priority_selected}" \
        --run-log "${run_log}" --checkpoint "${all_io_checkpoint}" \
        --priority-only
    python3 scripts/core_oracle_sampler.py select-all-io-core-sweep \
        --runner "${example_binary}" --selected "${all_io_selected}" \
        --run-log "${run_log}" --quota 50 --discovery-batch 64
    python3 scripts/core_oracle_sampler.py measure-all-io-core-sweep \
        --runner "${example_binary}" --selected "${all_io_selected}" \
        --run-log "${run_log}" --checkpoint "${all_io_checkpoint}"
    exit 0
fi

#NEW
if [[ "${MATMUL_V3_CAMPAIGN:-}" == "remaining_core_sweep" ]]; then
    if [[ "$#" -ne 0 ]]; then
        echo "fatal: remaining core sweep uses its own branch-directed shapes" >&2
        exit 2
    fi
    remaining_selected="${host_build}/remaining_core_sweep_selected.tsv"
    remaining_checkpoint="${host_build}/remaining_core_sweep_checkpoint.jsonl"
    python3 scripts/core_oracle_sampler.py select-remaining \
        --runner "${example_binary}" --selected "${remaining_selected}" \
        --run-log "${run_log}" --quota 20 --discovery-batch 64
    python3 scripts/core_oracle_sampler.py measure-remaining \
        --runner "${example_binary}" --selected "${remaining_selected}" \
        --run-log "${run_log}" --checkpoint "${remaining_checkpoint}"
    exit 0
fi

#NEW
if [[ "${MATMUL_V3_CAMPAIGN:-}" == "shrink_core_validation" ]]; then
    if [[ "$#" -ne 0 ]]; then
        echo "fatal: shrink core validation uses its own fresh branch-directed shapes" >&2
        exit 2
    fi
    python3 scripts/core_oracle_sampler.py select-shrink-core-validation \
        --runner "${example_binary}" --selected "${selected_shapes}" \
        --run-log "${run_log}" --discovery-batch 64
    python3 scripts/core_oracle_sampler.py compare-core-validation \
        --runner "${example_binary}" --selected "${selected_shapes}" \
        --run-log "${run_log}" --quota 50 --batch-size 8
    exit 0
fi

#NEW
if [[ "$#" -gt 0 ]]; then
    if (( $# % 3 != 0 )); then
        echo "fatal: shapes must be supplied as M N K triples" >&2
        exit 2
    fi
    while [[ "$#" -gt 0 ]]; do
        printf 'fp32\tNN\t%s\t%s\t%s\tUSER\n' "$1" "$2" "$3" >>"${selected_shapes}"
        printf 'fp32\tNT\t%s\t%s\t%s\tUSER\n' "$1" "$2" "$3" >>"${selected_shapes}"
        shift 3
    done
else
    python3 scripts/core_oracle_sampler.py select-shrink \
        --runner "${example_binary}" --selected "${selected_shapes}" \
        --run-log "${run_log}" --quota 30 --discovery-batch 64
fi

#NEW
if ! python3 scripts/core_oracle_sampler.py compare \
    --runner "${example_binary}" --selected "${selected_shapes}" \
    --run-log "${run_log}" --quota 30 --batch-size 8; then
    cat "${run_log}" >&2
    echo "fatal: no valid MatMulV3 shrink comparison was produced" >&2
    exit 1
fi
