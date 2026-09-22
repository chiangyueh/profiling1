#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /usr/local/Ascend/cann-8.5.0/set_env.sh >/dev/null
export ASCEND_RT_VISIBLE_DEVICES=2
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
unset ASCEND_CUSTOM_OPP_PATH

if [[ "$#" -ne 0 ]]; then
    exit 2
fi

build_dir="${PWD}/build"
build_log="$(mktemp)"
trap 'rm -f "${build_log}"' EXIT

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
    exit 1
fi

runtime_library="-lacl_rt"
if [[ -f "${ASCEND_HOME_PATH}/lib64/libascendcl.so" || -f "${ASCEND_OPP_PATH}/lib64/libascendcl.so" ]]; then
    runtime_library="-lascendcl"
fi
runner="${build_dir}/test_vector_dot"
if ! g++ matmul/mat_mul_v3/examples/test_vector_dot.cpp \
    -std=gnu++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
    -I "${PWD}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/include/aclnnop" \
    -I "${ASCEND_HOME_PATH}/include/aclnn" \
    -I "${ASCEND_HOME_PATH}/$(uname -m)-linux/pkg_inc" \
    -L "${ASCEND_OPP_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -L "${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64" \
    "${opapi_nn}" "${opapi_math}" "${runtime_library}" \
    -lnnopbase -lregister -lopp_registry -ldl \
    -Wl,-rpath,"$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}")" \
    -o "${runner}" >>"${build_log}" 2>&1; then
    cat "${build_log}" >&2
    exit 1
fi

build_vector_kernel() {
    local kernel_name="MatMulV3_VectorDot"
    local kernel_dir="${build_dir}/vector_dot_bin"
    local object="${kernel_dir}/${kernel_name}.o"
    local metadata="${kernel_dir}/${kernel_name}.json"
    if [[ -f "${object}" && -f "${metadata}" ]] &&
       ! find matmul/mat_mul_v3/op_kernel -type f -newer "${object}" -print -quit | grep -q .; then
        printf '%s\n' "${object}"
        return 0
    fi

    local ascendc_dir="${build_dir}/tbe/ascendc"
    local dynamic_dir="${build_dir}/tbe/dynamic"
    local param_dir="${build_dir}/vector_dot_params"
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
        --deterministic=false --tiling_key=2162688 >>"${build_log}" 2>&1
    if [[ ! -f "${object}" || ! -f "${metadata}" ]]; then
        return 1
    fi
    if ! readelf -Ws "${object}" | awk \
        '$4 == "FUNC" && $5 == "GLOBAL" && $8 == "MatMulV3_VectorDot_2162688" {found=1} END {exit !found}'; then
        return 1
    fi
    printf '%s\n' "${object}"
}

vector_binary="$(build_vector_kernel)" || {
    cat "${build_log}" >&2
    exit 1
}

shapes=()
for k in 8192 10240 12288 14336 16384 20480; do
    for m in {1..16}; do
        for n in 17 18 20 22 24 28 31 32 33 36 40 44 48 52 56 60 63 64; do
            output_dots=$((m * n))
            padded_dots=$(( ((m + 15) / 16 * 16) * ((n + 15) / 16 * 16) ))
            if ((output_dots <= 20 || padded_dots * 2 <= output_dots * 5)); then
                continue
            fi
            if ((k < 16384 && output_dots >= 280)); then
                continue
            fi
            shapes+=("${m}" "${n}" "${k}")
        done
    done
done

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_VECTOR_BINARY="${vector_binary}"
export MATMUL_DISABLE_REPO=1
export MATMUL_LEGACY_COMMON_LIBRARY="${legacy_common}"
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"
exec "${runner}" "${shapes[@]}"
