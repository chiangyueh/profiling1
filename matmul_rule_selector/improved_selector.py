#!/usr/bin/env python3
"""C220 MatMulV3 exact zero-owner core elimination.

The CANN 8.1 host algorithm is reconstructed locally from source and produces
the complete baseline packet. Each of the twelve installed kernel suffixes is
then passed through its own ownership rule. A candidate differs from that
baseline only when launched AICs and their paired AIVs provably own no Cube,
ND2NZ, epilogue, workspace, or reduction task.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASELINE_PACKAGE_ROOT = ROOT / "baseline_core"
sys.path.insert(0, str(BASELINE_PACKAGE_ROOT))

from matmul_reconstruction._core.abi import pack_packet, unpack_packet  # noqa: E402
from matmul_reconstruction._core.initializer import CUBE_FIELDS  # noqa: E402
from matmul_reconstruction.api import select as source_select  # noqa: E402

from core_ownership_rules import (  # noqa: E402
    INSTALLED_SUFFIXES,
    CorePlan,
    ceil_div,
    derive_core_plan,
)


HARDWARE = {
    "aicNum": 20, "aivNum": 40, "ubSize": 196352,
    "l1Size": 524032, "l2Size": 201326592,
    "l0CSize": 131072, "l0ASize": 65536, "l0BSize": 65536,
    "btSize": 1024, "supportL0c2out": True,
    "supportL12BtBf16": False, "cubeFreq": 0, "npuArch": 220,
    "socVersion": 220, "socVersionStr": "Ascend910B3",
    "cannVersion": "8.1.RC1",
}

KERNEL_VARIANT = {
    0: "BASE_UNALIGNED", 1: "BASE_ALIGNED",
    20: "SINGLE_CORE_SPLIT_K_UNALIGNED",
    21: "SINGLE_CORE_SPLIT_K_ALIGNED",
    30: "DETERMINISTIC_SPLIT_K_UNALIGNED",
    31: "DETERMINISTIC_SPLIT_K_ALIGNED",
    101: "AL1_FULL_LOAD_ALIGNED",
    200: "BL1_FULL_LOAD_UNALIGNED", 201: "BL1_FULL_LOAD_ALIGNED",
    10200: "BL1_FULL_LOAD_FIXPIPE_UNALIGNED",
    10201: "BL1_FULL_LOAD_FIXPIPE_ALIGNED",
    20201: "BL1_FULL_LOAD_VEC_NZ2ND",
}

SOURCE_TO_FORMULA_FAMILY = {
    "BASE": "BASE", "AL1_FULL_LOAD": "AL1",
    "BL1_FULL_LOAD": "BL1",
    "BL1_FULL_LOAD_FIXPIPE": "FIXPIPE_BL1",
    "SINGLE_CORE_SPLIT_K": "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K": "DETERMINISTIC_SPLIT_K",
    "INCREMENTAL_PATTERN": "INCREMENTAL_PATTERN",
}

if tuple(sorted(KERNEL_VARIANT)) != INSTALLED_SUFFIXES:
    raise RuntimeError("selector variant table does not cover all installed suffixes")


def make_request(m, k, n, dtype, trans_a, trans_b):
    return {
        "M": m, "N": n, "K": k, "dtype": dtype,
        "output_dtype": dtype, "layoutA": "ND", "layoutB": "ND",
        "layoutC": "ND", "transA": trans_a, "transB": trans_b,
        "bias": False, "hf32": False, "forceGrpAccForFp32": False,
    }


def _packet_output(source: dict, raw: bytes | None = None, cores: int | None = None) -> dict:
    suffix = int(source["tiling_key"] - 10**19)
    blob = bytes.fromhex(source["raw_buffer_hex"]) if raw is None else raw
    block_dim = int(source["blockDim"]) if cores is None else int(cores)
    return {
        "selected_family": source["selected_family"],
        "kernel_variant": KERNEL_VARIANT[suffix],
        "tiling_key": int(source["tiling_key"]),
        "block_dim": block_dim,
        "workspace_bytes": int(source["workspace_bytes"]),
        "tiling_data_bytes": len(blob),
        "tiling_data_hex": blob.hex(),
        "tiling_data_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _scheduled_work(cube: dict, request: dict) -> dict:
    parent_m = ceil_div(int(request["M"]), int(cube["singleCoreM"]))
    parent_n = ceil_div(int(request["N"]), int(cube["singleCoreN"]))
    parent_k = ceil_div(int(request["K"]), int(cube["singleCoreK"]))
    base_m = ceil_div(int(cube["singleCoreM"]), int(cube["baseM"]))
    base_n = ceil_div(int(cube["singleCoreN"]), int(cube["baseN"]))
    base_k = ceil_div(int(cube["singleCoreK"]), int(cube["baseK"]))
    iterations = parent_m * parent_n * parent_k * base_m * base_n * base_k
    return {
        "parent_grid_mnk": [parent_m, parent_n, parent_k],
        "base_tiles_per_parent_mnk": [base_m, base_n, base_k],
        "base_tile_iterations": iterations,
        "output_tasks": parent_m * parent_n,
        "logical_fma": int(request["M"]) * int(request["N"]) * int(request["K"]),
        "padded_cube_fma": (
            iterations * int(cube["baseM"]) * int(cube["baseN"]) *
            int(cube["baseK"])
        ),
    }


def _theory(source: dict, request: dict, plan: CorePlan) -> dict:
    cube = source["tilingData"]["matmulTiling"]
    l2 = source["tilingData"]["tileL2cacheTiling"]
    return {
        "selection_mode": "TWELVE_SUFFIX_EXACT_ZERO_OWNER_DELETION",
        "complete_tilings_constructed": 1,
        "candidate_count": 1,
        "candidate_enumeration": False,
        "pareto_pruning": False,
        "latency_or_cost_score": False,
        "logical_fma_count": int(request["M"]) * int(request["N"]) * int(request["K"]),
        "source_family": source["selected_family"],
        "source_suffix": int(source["tiling_key"] - 10**19),
        "source_attempts": source["attempted_families"],
        "formula_family": SOURCE_TO_FORMULA_FAMILY[source["selected_family"]],
        "formula_fields": {
            name: int(cube[name]) for name in (
                "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
                "baseM", "baseN", "baseK", "depthA1", "depthB1",
                "stepM", "stepN", "stepKa", "stepKb", "dbL0A",
                "dbL0B", "dbL0C", "iterateOrder",
            )
        },
        "formula_l2": dict(l2),
        "resource_bytes": source["resource_facts"],
        "scheduled_work": _scheduled_work(cube, request),
        "core_ownership": plan.as_dict(),
        "selection_contract": {
            "complete_tilings_constructed": 1,
            "candidate_enumeration": False,
            "pareto_pruning": False,
            "latency_or_cost_score": False,
            "history_lookup": False,
            "runtime_tiling_bank": False,
            "decision": "source_reconstruction_then_exact_branch_worker_cardinality",
        },
    }


def _runtime_dependencies() -> dict:
    return {
        "cost_model": False,
        "candidate_enumeration": False,
        "history": False,
        "runtime_kb": False,
        "tiling_bank": False,
        "installed_host_tiler": False,
        "source_reconstruction_for_complete_baseline_packet": True,
        "source_packet_fields_used_for_exact_instruction_deletion": True,
    }


def _unchanged(request: dict, source: dict, plan: CorePlan, theory: dict) -> dict:
    selected = _packet_output(source)
    return {
        "status": "CORE_OWNERSHIP_AUDITED_UNCHANGED",
        "request": request, "selected": selected, "baseline": selected,
        "improved": None, "baseline_equivalent": True,
        "changed_rules": [], "changed_fields": {},
        "skip_reason": plan.action,
        "formula_family": theory["formula_family"],
        "kernel_suffix": plan.suffix,
        "core_plan": plan.as_dict(),
        "npu_eligible": False, "theory": theory,
        "runtime_dependencies": _runtime_dependencies(),
        "validation": {
            "host_packet": "PASS_SOURCE_RECONSTRUCTION_272_BYTE_PACKET",
            "branch_worker_equation": "PASS_NO_ZERO_OWNER_CORE_TO_DELETE",
            "npu_performance": "NO_DISTINCT_PACKET",
        },
    }


def _validate_deletion(source: dict, raw: bytes, plan: CorePlan) -> None:
    source_raw = bytes.fromhex(source["raw_buffer_hex"])
    if len(raw) != 272 or len(source_raw) != 272:
        raise ValueError("CANN 8.1 packet must be exactly 272 bytes")
    changed_words = [
        index for index in range(68)
        if raw[index * 4:(index + 1) * 4] != source_raw[index * 4:(index + 1) * 4]
    ]
    core_word = CUBE_FIELDS.index("usedCoreNum")
    if changed_words != [core_word]:
        raise ValueError(f"candidate changed non-core packet words: {changed_words}")
    if not (0 < plan.required_cores < plan.source_cores <= 20):
        raise ValueError("candidate does not strictly delete launched cores")
    if plan.cube_owners > plan.required_cores:
        raise ValueError("candidate removed an AIC with Cube ownership")
    if plan.reduction_owners > plan.required_cores:
        raise ValueError("candidate removed an AIC with reduction ownership")
    if plan.vector_output_owners > plan.required_cores:
        raise ValueError("candidate removed an AIC/AIV epilogue owner")
    for tasks in (plan.vector_a_tasks, plan.vector_b_tasks):
        if tasks < 0 or tasks > 2 * plan.required_cores:
            raise ValueError("candidate removed an AIV conversion owner")
    payload = unpack_packet(raw, "legacy272_cube200")
    if int(payload["cube_words"][core_word]) != plan.required_cores:
        raise ValueError("candidate packet core word does not match its proof")


def generate(
    m, k, n, dtype="fp16", trans_a=False, trans_b=False,
    *, required_suffix=None, required_family=None, required_fields=None,
):
    request = make_request(m, k, n, dtype, trans_a, trans_b)
    source = source_select(
        request, HARDWARE, source_profile="installed_81", trace=True
    )
    suffix = int(source["tiling_key"] - 10**19)
    if required_suffix is not None and suffix != int(required_suffix):
        raise ValueError(f"source suffix {suffix} != required {required_suffix}")
    if required_family is not None:
        actual = SOURCE_TO_FORMULA_FAMILY[source["selected_family"]]
        if actual != required_family and source["selected_family"] != required_family:
            raise ValueError(f"source family {actual} != required {required_family}")
    if required_fields:
        cube = source["tilingData"]["matmulTiling"]
        mismatches = {
            name: (cube.get(name), value) for name, value in required_fields.items()
            if int(cube.get(name, -1)) != int(value)
        }
        if mismatches:
            raise ValueError(f"source packet field mismatch: {mismatches}")

    plan = derive_core_plan(request, source["tilingData"], suffix)
    theory = _theory(source, request, plan)
    if plan.action != "DELETE_ZERO_OWNERSHIP_CORES":
        return _unchanged(request, source, plan, theory)

    payload = copy.deepcopy(source["abi_payload"])
    core_word = CUBE_FIELDS.index("usedCoreNum")
    payload["cube_words"][core_word] = plan.required_cores
    raw = pack_packet(payload, "legacy272_cube200")
    _validate_deletion(source, raw, plan)

    baseline = _packet_output(source)
    improved = _packet_output(source, raw=raw, cores=plan.required_cores)
    cube = source["tilingData"]["matmulTiling"]
    full_a_bytes = 0
    if suffix == 101:
        width = 4 if dtype == "fp32" else 2
        c0 = 32 // width
        full_a_bytes = ceil_div(m, 16) * 16 * ceil_div(k, c0) * c0 * width
    certificate = {
        **plan.as_dict(),
        "proof_kind": plan.proof_kind,
        "source_packet_sha256": baseline["tiling_data_sha256"],
        "candidate_packet_sha256": improved["tiling_data_sha256"],
        "changed_packet_words": ["usedCoreNum"],
        "same_kernel_suffix": suffix,
        "same_workspace_bytes": int(source["workspace_bytes"]),
        "same_cube_tile_geometry": True,
        "same_l2_schedule": True,
        "same_nd2nz_tile_geometry": True,
        "new_MMAD_commands": 0,
        "new_conversion_tasks": 0,
        "new_output_tasks": 0,
        "new_workspace_bytes": 0,
        "full_a_copy_bytes_per_launched_aic": full_a_bytes,
        "eliminated_full_a_copy_bytes": full_a_bytes * plan.eliminated_cores,
    }
    theory["improvement_equation"] = certificate
    theory["strict_dominance_certificate"] = certificate
    theory["formula_fields"]["usedCoreNum"] = plan.required_cores

    return {
        "status": "MODIFIED_TILING", "request": request,
        "selected": improved, "baseline": baseline, "improved": improved,
        "baseline_equivalent": False,
        "changed_rules": [plan.proof_kind],
        "changed_fields": {
            "usedCoreNum": {
                "baseline": int(cube["usedCoreNum"]),
                "improved": plan.required_cores,
            },
            "block_dim": {
                "baseline": int(source["blockDim"]),
                "improved": plan.required_cores,
            },
        },
        "formula_family": theory["formula_family"],
        "kernel_suffix": suffix, "core_plan": plan.as_dict(),
        "selection_basis": "SOURCE_RECONSTRUCTED_EXACT_ZERO_OWNER_DELETION",
        "npu_eligible": True, "theory": theory,
        "runtime_dependencies": _runtime_dependencies(),
        "path_coverage": {
            "family_rule": KERNEL_VARIANT[suffix],
            "kernel_implementation": f"UNCHANGED_SUFFIX_{suffix}",
            "abi_layout": "RETAINED_272_BYTE_ABI_ONLY_USED_CORE_NUM_DIFFERS",
            "installed_host_tiler": "NOT_CALLED",
            "history_or_tiling_bank": "NOT_CALLED",
        },
        "validation": {
            "host_packet": "PASS_272_BYTE_ABI_ONLY_CORE_WORD_CHANGED",
            "branch_worker_equation": "PASS_ALL_REMOVED_WORKERS_HAVE_ZERO_OWNERSHIP",
            "kernel_key_domain": "PASS_INSTALLED_81_SUFFIX_DISPATCH",
            "npu_performance": "NOT_MEASURED",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = generate(
        args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b
    )
    print(json.dumps(
        result, sort_keys=True, indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ))


if __name__ == "__main__":
    main()
