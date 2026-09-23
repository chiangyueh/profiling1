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
runner="${build_dir}/test_adaptive_deterministic_v1"
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

printf '{"stage":"workload_generation","status":"begin"}\n'
mapfile -t adaptive_workloads < <(python3 - <<'PY'
import random

dtypes = ("fp16_fp16", "bf16_bf16", "fp32_fp32")
layouts = ("NN", "NT", "TN", "TT")
m_values = (1, 7, 15, 16, 17, 24, 31, 32, 33, 47, 48, 55, 63, 64, 65,
            79, 80, 95, 96, 111, 112, 127, 128, 159, 160, 191, 192, 255,
            256, 319, 320, 383, 384)
n_values = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
            1280, 1536, 1792, 2048, 2304, 2560, 3072, 3584, 4096)
k_values = (8192, 12288, 14336, 16384, 18432, 22528, 24576, 28672,
            32768, 36864, 40960, 45056, 49152, 57344, 65536)
groups = []
for group_index, (dtype, layout) in enumerate(
        (pair for dtype in dtypes for pair in ((dtype, value) for value in layouts))):
    rng = random.Random(8505 + group_index)
    rows = []
    for m in m_values:
        for n in n_values:
            bytes_per_element = 4 if dtype == "fp32_fp32" else 2
            valid_k = tuple(k for k in k_values
                            if (m + n) * k * bytes_per_element <= 512 * 1024 * 1024)
            first, second = rng.sample(valid_k, 2)
            rows.append((dtype, layout, m, n, first))
            rows.append((dtype, layout, m, n, second))
    rng.shuffle(rows)
    groups.append(rows)
for index in range(min(len(group) for group in groups)):
    for group in groups:
        for value in group[index]:
            print(value)
PY
)

printf '{"stage":"workload_generation","status":"passed","adaptive":%d}\n' \
    "$(( ${#adaptive_workloads[@]} / 5 ))"

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

run_campaign ADAPTIVE_DETERMINISTIC_SPLIT_K 480 "${runner}" "${adaptive_workloads[@]}"
printf '{"overnight_complete":true,"campaigns":1,"campaign_process_failures":%d}\n' "${campaign_failures}"
