#!/usr/bin/env bash
set -euo pipefail

# original begin: require the redundant "full" argument
# if [[ "$#" -ne 1 || "$1" != "full" ]]; then
#     echo "usage: bash run_npu.sh full" >&2
#     exit 2
# fi
# original end: require the redundant "full" argument

# new begin: the script has one complete execution mode and takes no arguments
if [[ "$#" -ne 0 ]]; then
    echo "usage: bash run_npu.sh" >&2
    exit 2
fi
# new end: the script has one complete execution mode and takes no arguments

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

source /usr/local/Ascend/cann-8.5.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2

# original begin: this A/B loop only linked the installed built-in operator
# for shrink_mode in 0 1; do
#     echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
#     MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
#         bash build.sh \
#         --run_example mat_mul_v3 eager \
#         --example_name=matmul
#
#     echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
# done
# original end: this A/B loop only linked the installed built-in operator

# original begin: rebuild the package and recompile the example on every A/B run
# vendor_name="matmulab"
#
# bash build.sh \
#     --pkg \
#     --soc=ascend910b \
#     --ops=mat_mul_v3 \
#     --vendor_name="${vendor_name}" \
#     -j1 \
#     -O3
#
# shopt -s nullglob
# run_packages=(build_out/cann-ops-nn-${vendor_name}-linux.*.run)
# if [[ "${#run_packages[@]}" -ne 1 ]]; then
#     echo "fatal: expected one custom operator package, found ${#run_packages[@]}" >&2
#     exit 1
# fi
#
# "${run_packages[0]}" --quiet --install-path="${ASCEND_OPP_PATH}"
# source "${ASCEND_OPP_PATH}/vendors/${vendor_name}_nn/bin/set_env.bash"
#
# for shrink_mode in 0 1; do
#     echo "CORE_SHRINK_RUN_BEGIN enabled=${shrink_mode}"
#     MATMUL_V3_SHRINK_IDLE_CORES="${shrink_mode}" \
#         bash build.sh \
#         --run_example mat_mul_v3 eager cust \
#         --vendor_name="${vendor_name}" \
#         --example_name=matmul
#     echo "CORE_SHRINK_RUN_END enabled=${shrink_mode}"
# done
# original end: rebuild the package and recompile the example on every A/B run

# new begin: rebuild only when package-relevant source content changes
vendor_name="matmulab"
installed_root="${ASCEND_OPP_PATH}/vendors/${vendor_name}_nn"
installed_env="${installed_root}/bin/set_env.bash"
installed_api="${installed_root}/op_api/lib/libcust_opapi.so"
installed_tiling="${installed_root}/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -m)/libcust_opmaster.so"
installed_marker="${installed_root}/bin/profiling1_matmul_v3.sha256"
local_marker="build_out/${vendor_name}_matmul_v3.sha256"

source_revision="$(git rev-parse HEAD)"
source_fingerprint="${source_revision}-cann-8.5.0-ascend910b"

cached_fingerprint=""
if [[ -r "${installed_marker}" ]]; then
    read -r cached_fingerprint < "${installed_marker}"
elif [[ -r "${local_marker}" ]]; then
    read -r cached_fingerprint < "${local_marker}"
fi

package_ready=0
if [[ "${cached_fingerprint}" == "${source_fingerprint}" &&
      -f "${installed_env}" &&
      -f "${installed_api}" &&
      -f "${installed_tiling}" &&
      -d "${installed_root}/op_impl/ai_core/tbe/kernel" ]]; then
    package_ready=1
fi

if [[ "${package_ready}" -eq 1 ]]; then
    echo "CUSTOM_PACKAGE_CACHE hit fingerprint=${source_fingerprint}"
else
    echo "CUSTOM_PACKAGE_CACHE miss fingerprint=${source_fingerprint}"
    bash build.sh \
        --pkg \
        --soc=ascend910b \
        --ops=mat_mul_v3 \
        --vendor_name="${vendor_name}" \
        -j1 \
        -O3

    shopt -s nullglob
    run_packages=(build_out/cann-ops-nn-${vendor_name}-linux.*.run)
    if [[ "${#run_packages[@]}" -ne 1 ]]; then
        echo "fatal: expected one custom operator package, found ${#run_packages[@]}" >&2
        exit 1
    fi

    "${run_packages[0]}" --quiet --install-path="${ASCEND_OPP_PATH}"
    mkdir -p -- "$(dirname -- "${local_marker}")"
    printf '%s\n' "${source_fingerprint}" > "${local_marker}"
    if ! printf '%s\n' "${source_fingerprint}" > "${installed_marker}"; then
        echo "warning: installed cache marker is not writable; cache remains local to this checkout" >&2
    fi
fi

source "${installed_env}"

example_source="matmul/mat_mul_v3/examples/test_aclnn_matmul.cpp"
example_binary="${PWD}/build/test_aclnn_matmul"

echo "CORE_SHRINK_RUN_BEGIN enabled=0"
if [[ ! -x "${example_binary}" || "${example_source}" -nt "${example_binary}" ]]; then
    echo "EXAMPLE_BUILD cache=miss"
    MATMUL_V3_SHRINK_IDLE_CORES=0 \
        bash build.sh \
        --run_example mat_mul_v3 eager cust \
        --vendor_name="${vendor_name}" \
        --example_name=matmul
else
    echo "EXAMPLE_BUILD cache=hit"
    MATMUL_V3_SHRINK_IDLE_CORES=0 "${example_binary}"
fi
echo "CORE_SHRINK_RUN_END enabled=0"

if [[ ! -x "${example_binary}" ]]; then
    echo "fatal: example executable was not produced: ${example_binary}" >&2
    exit 1
fi

echo "CORE_SHRINK_RUN_BEGIN enabled=1"
MATMUL_V3_SHRINK_IDLE_CORES=1 "${example_binary}"
echo "CORE_SHRINK_RUN_END enabled=1"
# new end: rebuild only when package-relevant source content changes
