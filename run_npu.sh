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
runner="${build_dir}/test_wide_n_panel_reuse_base"
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
    ("fp16_fp16", "NN", 17, 4865, 16384),
    ("bf16_bf16", "NN", 31, 5001, 18432),
    ("fp16_fp16", "NN", 63, 5119, 20480),
    ("bf16_bf16", "NN", 79, 5121, 22528),
    ("fp16_fp16", "NN", 95, 6145, 24576),
    ("bf16_bf16", "NN", 127, 12289, 28672),
)
for row in anchors:
    print("\t".join(str(value) for value in row))

rng = random.Random(8505)
diagnostic_failures = [
    ("fp16_fp16", "NN", 10, 6144, 22528),
    ("fp16_fp16", "NN", 84, 7424, 24576),
    ("bf16_bf16", "NN", 107, 6400, 26624),
    ("bf16_bf16", "NN", 87, 5888, 24576),
    ("fp16_fp16", "NN", 54, 5376, 16384),
    ("bf16_bf16", "NN", 61, 6400, 22016),
    ("fp16_fp16", "NN", 6, 5632, 16384),
    ("bf16_bf16", "NN", 85, 5632, 16384),
    ("bf16_bf16", "NN", 56, 6656, 16384),
    ("bf16_bf16", "NN", 73, 6144, 18944),
    ("fp16_fp16", "NN", 76, 6144, 21504),
    ("bf16_bf16", "NN", 95, 6400, 18944),
    ("fp16_fp16", "NN", 109, 7680, 19456),
    ("bf16_bf16", "NN", 88, 5632, 17408),
    ("fp16_fp16", "NN", 15, 11008, 22528),
    ("bf16_bf16", "NN", 34, 10752, 16896),
    ("bf16_bf16", "NN", 47, 6912, 26112),
    ("bf16_bf16", "NN", 19, 12800, 20480),
    ("bf16_bf16", "NN", 32, 11520, 23040),
    ("fp16_fp16", "NN", 43, 12800, 17408),
    ("bf16_bf16", "NN", 89, 6400, 18944),
    ("fp16_fp16", "NN", 58, 5632, 24064),
    ("fp16_fp16", "NN", 87, 7680, 18944),
    ("fp16_fp16", "NN", 103, 6912, 21504),
    ("fp16_fp16", "NN", 100, 6912, 27136),
    ("fp16_fp16", "NN", 73, 7168, 23552),
    ("bf16_bf16", "NN", 8, 5632, 18432),
    ("bf16_bf16", "NN", 9, 5632, 16384),
    ("bf16_bf16", "NN", 87, 5376, 16896),
    ("bf16_bf16", "NN", 78, 7424, 27136),
    ("bf16_bf16", "NN", 20, 5888, 25088),
    ("bf16_bf16", "NN", 58, 7424, 27136),
    ("bf16_bf16", "NN", 85, 6144, 17920),
    ("fp16_fp16", "NN", 31, 6656, 23040),
    ("fp16_fp16", "NN", 14, 11008, 23552),
    ("bf16_bf16", "NN", 23, 7680, 17408),
    ("bf16_bf16", "NN", 66, 6912, 23040),
]
for _ in range(3):
    rng.shuffle(diagnostic_failures)
    for row in diagnostic_failures:
        print("\t".join(str(value) for value in row))

rows = []
for dtype in ("fp16_fp16", "bf16_bf16"):
    for m in range(1, 129):
        for _ in range(96):
            n = rng.randrange(4865, 32769)
            k = rng.randrange(1024, 131073)
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

run_campaign WIDE_N_PANEL_REUSE_BASE "${success_target}" "${runner}" --manifest "${workload_manifest}"
printf '{"overnight_complete":true,"campaigns":1,"campaign_process_failures":%d}\n' "${campaign_failures}"
