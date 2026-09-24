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
workload_manifest="$(mktemp)"
trap 'rm -f "${build_log}" "${workload_manifest}"' EXIT

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
runner="${build_dir}/test_wide_n_shallow_k_base"
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
python3 - >"${workload_manifest}" <<'PY'
import random

anchors = (
    ("bf16_bf16", "NN", 111, 7168, 16384),
    ("bf16_bf16", "NN", 80, 7168, 22528),
    ("fp16_fp16", "NN", 33, 12288, 18432),
    ("fp16_fp16", "NN", 10, 6144, 22528),
    ("bf16_bf16", "NN", 27, 6144, 16384),
    ("fp16_fp16", "NN", 40, 6144, 24576),
    ("fp16_fp16", "NN", 80, 7168, 18432),
    ("bf16_bf16", "NN", 10, 7168, 22528),
    ("bf16_bf16", "NN", 20, 12288, 20480),
)
for row in anchors:
    print("\t".join(str(value) for value in row))

rng = random.Random(8505)
rows = []
for dtype in ("fp16_fp16", "bf16_bf16"):
    for m in range(1, 129):
        for n in range(5120, 16385, 256):
            for k in range(16384, 32769, 512):
                if (m + n) * k * 2 <= 512 * 1024 * 1024:
                    rows.append((dtype, "NN", m, n, k))
rng.shuffle(rows)
for row in rows:
    if row not in anchors:
        print("\t".join(str(value) for value in row))
PY

adaptive_count="$(wc -l <"${workload_manifest}")"
printf '{"stage":"workload_generation","status":"passed","candidates":%d}\n' "${adaptive_count}"

export MATMUL_HOST_LIBRARY="${host_library}"
export MATMUL_DISABLE_REPO=1
export LD_LIBRARY_PATH="$(dirname -- "${opapi_nn}"):$(dirname -- "${opapi_math}"):${ASCEND_OPP_PATH}/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/$(uname -m)-linux/lib64:${LD_LIBRARY_PATH:-}"

campaign_failures=0
success_target="${MATMUL_SUCCESS_TARGET:-300}"
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

run_campaign WIDE_N_SHALLOW_K_BASE "${success_target}" "${runner}" --manifest "${workload_manifest}"
printf '{"overnight_complete":true,"campaigns":1,"campaign_process_failures":%d}\n' "${campaign_failures}"
