#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH
unset MATMUL_BASE_MODE MATMUL_BASE_EXPERIMENT_SELECTED MATMUL_SPLITK_MODE
unset MATMUL_DETERMINISTIC_ADAPTIVE MATMUL_DETERMINISTIC_ADAPTIVE_CHANGED
unset MATMUL_CAMPAIGN

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

build_dir="${PWD}/build"
build_log="$(mktemp)"
trap 'rm -f "${build_log}"' EXIT

printf '{"stage":"host_build","status":"begin"}\n'
if ! cmake -S . -B "${build_dir}" \
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
if ! cmake --build "${build_dir}" --target ophost_nn -- -j1 >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi
printf '{"stage":"host_build","status":"passed"}\n'

host_library="${build_dir}/libophost_nn.so"
opapi_nn=""
opapi_math=""
legacy_common=""
for path in \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_nn.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_nn.so" \
    "${ASCEND_OPP_PATH}/lib64/libopapi_nn.so"; do
    if [[ -f "${path}" ]]; then
        opapi_nn="${path}"
        break
    fi
done
for path in \
    "${ASCEND_OPP_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/lib64/libopapi_math.so" \
    "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64/libopapi_math.so"; do
    if [[ -f "${path}" ]]; then
        opapi_math="${path}"
        break
    fi
done
for path in \
    "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so" \
    "${ASCEND_HOME_PATH}/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/$(uname -m)/libophost_comm_legacy.so"; do
    if [[ -f "${path}" ]]; then
        legacy_common="${path}"
        break
    fi
done
if [[ ! -f "${host_library}" || -z "${opapi_nn}" || -z "${opapi_math}" || -z "${legacy_common}" ]]; then
    printf '{"fatal":"required_library_missing","host":%s,"opapi_nn":%s,"opapi_math":%s,"legacy":%s}\n' \
        "$([[ -f "${host_library}" ]] && printf true || printf false)" \
        "$([[ -n "${opapi_nn}" ]] && printf true || printf false)" \
        "$([[ -n "${opapi_math}" ]] && printf true || printf false)" \
        "$([[ -n "${legacy_common}" ]] && printf true || printf false)" >&2
    exit 1
fi
ln -sfn -- "${legacy_common}" "${build_dir}/libophost_comm_legacy.so"

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
runner="${build_dir}/test_four_route_overnight_v1"
printf '{"stage":"runner_build","status":"begin"}\n'
if ! g++ matmul/mat_mul_v3/examples/test_splitk_routes.cpp \
    matmul/mat_mul_v3/op_host/op_api/matmul.cpp \
    -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -I "${PWD}/common/stub/op_api" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${opapi_nn}" "${opapi_math}" "${runtime_library}" \
    -lnnopbase -lregister -lopp_registry -lunified_dlog -lmetadef -ldl \
    -Wl,-rpath,"$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}")" \
    -o "${runner}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi
printf '{"stage":"runner_build","status":"passed"}\n'

build_edge_kernel() {
    local kernel_name="MatMulV3_CubeVectorEdge"
    local kernel_dir="${build_dir}/cube_vector_edge_bin"
    local object="${kernel_dir}/${kernel_name}.o"
    local metadata="${kernel_dir}/${kernel_name}.json"
    if [[ -f "${object}" && -f "${metadata}" ]] &&
       ! find matmul/mat_mul_v3/op_kernel -type f -newer "${object}" -print -quit | grep -q .; then
        printf '%s\n' "${object}"
        return 0
    fi

    local ascendc_dir="${build_dir}/tbe/ascendc"
    local dynamic_dir="${build_dir}/tbe/dynamic"
    local param_dir="${build_dir}/cube_vector_edge_params"
    rm -rf -- "${kernel_dir}" "${param_dir}" "${ascendc_dir}/mat_mul_v3"
    mkdir -p -- "${kernel_dir}" "${ascendc_dir}/mat_mul_v3" \
        "${ascendc_dir}/common/act" "${ascendc_dir}/common/matmul_act" \
        "${dynamic_dir}" "${param_dir}"
    cp -a matmul/mat_mul_v3/op_kernel/. "${ascendc_dir}/mat_mul_v3/"
    cp -a common/act/. "${ascendc_dir}/common/act/"
    cp -a matmul/common/matmul_act/. "${ascendc_dir}/common/matmul_act/"

    local ops_info="${build_dir}/autogen/exc/aic-ascend910b-ops-info.ini"
    local opc_options="${build_dir}/autogen/custom_opc_options.ini"
    python3 scripts/util/ascendc_impl_build.py "${ops_info}" "" "" \
        "${ascendc_dir}" "${dynamic_dir}" "${build_dir}/autogen" >>"${build_log}" 2>&1
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
    values = [item for item in node["inputs"] + node["outputs"] if item.get("paramType") == "required"]
    if len(values) == 3 and all(item.get("dtype") == "float32" and item.get("format") == "ND" for item in values):
        print(path)
        break
PY
)"
    if [[ -z "${fp32_param}" ]]; then
        return 1
    fi
    python3 - "${fp32_param}" "${kernel_name}" <<'PY'
import json
import sys
path, name = sys.argv[1:]
with open(path, encoding="utf-8") as stream:
    data = json.load(stream)
data["op_list"][0]["bin_filename"] = name
with open(path, "w", encoding="utf-8") as stream:
    json.dump(data, stream, separators=(",", ":"))
PY
    asc_opc "${dynamic_dir}/mat_mul_v3.py" --main_func=mat_mul_v3 \
        --input_param="${fp32_param}" --soc_version=Ascend910B1 --output="${kernel_dir}" \
        --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic \
        --deterministic=false --tiling_key=3211264 >>"${build_log}" 2>&1
    if [[ ! -f "${object}" || ! -f "${metadata}" ]]; then
        return 1
    fi
    if ! readelf -Ws "${object}" | awk \
        '$4 == "FUNC" && $5 == "GLOBAL" && $8 == "MatMulV3_CubeVectorEdge_3211264_mix_aic" {aic=1}
         $4 == "FUNC" && $5 == "GLOBAL" && $8 == "MatMulV3_CubeVectorEdge_3211264_mix_aiv" {aiv=1}
         END {exit !(aic && aiv)}'; then
        return 1
    fi
    printf '%s\n' "${object}"
}

printf '{"stage":"workload_generation","status":"begin"}\n'
mapfile -t k_parallel_workloads < <(python3 - <<'PY'
import itertools
import random

dtypes = ("fp16_fp16", "fp16_fp32", "bf16_bf16", "bf16_fp32", "fp32_fp32")
layouts = ("NT", "TN")
m_values = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192)
n_values = (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 256)
k_values = (8192, 10240, 12288, 14336, 16384, 18432, 20480, 24576,
            28672, 32768, 36864, 40960, 49152, 57344, 65536)
rows = list(itertools.product(dtypes, layouts, m_values, n_values, k_values))
random.Random(8503).shuffle(rows)
for row in rows:
    for value in row:
        print(value)
PY
)

homogeneous_dtypes=(fp16_fp16 bf16_bf16 fp32_fp32)
layouts=(NN NT TN TT)

rectangular_workloads=()
serial=0
for k in 256 384 512 640 768 1024 1280 1536 2048 2560 3072 4096; do
    for long_dim in 2561 3073 3585 4097 4609 5121 5633 6145 6657 7169 7681 8193; do
        for short_dim in 17 23 31 39 47 55 63 71 79 87 95 103 111 119 127 143; do
            dtype="${homogeneous_dtypes[$((serial % ${#homogeneous_dtypes[@]}))]}"
            layout="${layouts[$(((serial / ${#homogeneous_dtypes[@]}) % ${#layouts[@]}))]}"
            rectangular_workloads+=("${dtype}" "${layout}" "${short_dim}" "${long_dim}" "${k}")
            serial=$((serial + 1))
            dtype="${homogeneous_dtypes[$((serial % ${#homogeneous_dtypes[@]}))]}"
            layout="${layouts[$(((serial / ${#homogeneous_dtypes[@]}) % ${#layouts[@]}))]}"
            rectangular_workloads+=("${dtype}" "${layout}" "${long_dim}" "${short_dim}" "${k}")
            serial=$((serial + 1))
        done
    done
done

reuse_workloads=()
serial=0
for k in 512 768 1024 1280 1536 2048 2560 3072 4096 5120 6144 8192; do
    for m in 257 385 513 641 769 897 1025 1153 1281 1409 1537 1665 1793 1921 2049 2305; do
        for n in 257 385 513 641 769 897 1025 1153 1281 1409 1537 1665 1793 1921 2049 2305; do
            dtype="${homogeneous_dtypes[$((serial % ${#homogeneous_dtypes[@]}))]}"
            layout="${layouts[$(((serial / ${#homogeneous_dtypes[@]}) % ${#layouts[@]}))]}"
            reuse_workloads+=("${dtype}" "${layout}" "${m}" "${n}" "${k}")
            serial=$((serial + 1))
        done
    done
done

edge_workloads=()
for k in 8192 9216 10240 12288 14336 16384 18432 20480 22528 24576 28672 32768; do
    for m in 129 130 131 257 258 259 385 386 389 513 514 519 641 643 769 773; do
        for n in 513 514 515 641 642 643 769 770 773 897 899 1025 1027 1153 1157 1281; do
            edge_workloads+=(fp32_fp32 NT "${m}" "${n}" "${k}")
        done
    done
done
printf '{"stage":"workload_generation","status":"passed","k_parallel":%d,"rectangular":%d,"reuse":%d,"edge":%d}\n' \
    "$(( ${#k_parallel_workloads[@]} / 5 ))" "$(( ${#rectangular_workloads[@]} / 5 ))" \
    "$(( ${#reuse_workloads[@]} / 5 ))" "$(( ${#edge_workloads[@]} / 5 ))"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

campaign_failures=0
run_campaign() {
    local campaign="$1"
    local target="$2"
    shift 2
    local rc=0
    printf '{"campaign_begin":"%s","target_passes":%d}\n' "${campaign}" "${target}"
    MATMUL_CAMPAIGN="${campaign}" MATMUL_TARGET_PASSES="${target}" "$@" || rc=$?
    printf '{"campaign_complete":"%s","process_result_code":%d}\n' "${campaign}" "${rc}"
    if [[ "${rc}" -ne 0 ]]; then
        campaign_failures=$((campaign_failures + 1))
    fi
}

run_campaign K_PARALLEL_SPLIT_K 300 "${runner}" "${k_parallel_workloads[@]}"
run_campaign RECTANGULAR_CUBE 200 "${runner}" "${rectangular_workloads[@]}"
run_campaign REUSE_DIRECTED 200 "${runner}" "${reuse_workloads[@]}"
printf '{"stage":"edge_kernel_build","status":"begin"}\n'
if edge_binary="$(build_edge_kernel)"; then
    printf '{"stage":"edge_kernel_build","status":"passed"}\n'
    export MATMUL_EDGE_BINARY="${edge_binary}"
    run_campaign CUBE_VECTOR_EDGE 200 "${runner}" "${edge_workloads[@]}"
else
    cat "${build_log}" >&2
    printf '{"campaign_complete":"CUBE_VECTOR_EDGE","process_result_code":4,"reason":"kernel_build_failed"}\n'
    campaign_failures=$((campaign_failures + 1))
fi
printf '{"overnight_complete":true,"campaigns":4,"campaign_process_failures":%d}\n' "${campaign_failures}"
