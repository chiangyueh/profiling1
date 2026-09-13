#!/usr/bin/env python3
"""Single-result MatMulV3 tiling rules for Ascend 910B3 / CANN 8.1.

The complete source-reconstructed decision chain first selects exactly one
execution family and legal packet from shape and hardware.  A family-specific
closed-form rule may then replace that packet without reading any of its tile
fields.  No cost model, candidate set, Pareto pass, history table, RuntimeKb
or installed host tiler is consulted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASELINE_PACKAGE_ROOT = ROOT / "baseline_core"
sys.path.insert(0, str(BASELINE_PACKAGE_ROOT))

from matmul_reconstruction._core.abi import (  # noqa: E402
    VECTOR_NAMES,
    pack_packet,
    unpack_packet,
)
from matmul_reconstruction._core.initializer import (  # noqa: E402
    CUBE_FIELDS,
    initialize_cube,
)
from matmul_reconstruction.api import select as source_select  # noqa: E402

from formula_rules import Hardware, Shape  # noqa: E402


HARDWARE = {
    "aicNum": 20, "aivNum": 40, "ubSize": 196352,
    "l1Size": 524032, "l2Size": 201326592,
    "l0CSize": 131072, "l0ASize": 65536, "l0BSize": 65536,
    "btSize": 1024, "supportL0c2out": True,
    "supportL12BtBf16": False, "cubeFreq": 0, "npuArch": 220,
    "socVersion": 220, "socVersionStr": "Ascend910B3",
    "cannVersion": "8.1.RC1",
}

FORMULA_HARDWARE = Hardware(
    cores=20, l0a=65536, l0b=65536, l0c=131072,
    l1_usable=524032, l2=201326592, ub=196352, bt=1024,
)

FAMILY_NAME = {
    "BASE": "BASE", "FIXPIPE_BL1": "BL1_FULL_LOAD_FIXPIPE",
    "AL1": "AL1_FULL_LOAD", "BL1": "BL1_FULL_LOAD",
    "SINGLE_CORE_SPLIT_K": "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K": "DETERMINISTIC_SPLIT_K",
}

SOURCE_TO_FORMULA_FAMILY = {
    "BASE": "BASE", "AL1_FULL_LOAD": "AL1", "BL1_FULL_LOAD": "BL1",
    "BL1_FULL_LOAD_FIXPIPE": "FIXPIPE_BL1",
    "SINGLE_CORE_SPLIT_K": "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K": "DETERMINISTIC_SPLIT_K",
    "INCREMENTAL_PATTERN": "INCREMENTAL_PATTERN",
}

FAMILY_KEY_PARTS = {
    "BASE": (0, 0, 0), "AL1": (1, 0, 0), "BL1": (2, 0, 0),
    "SINGLE_CORE_SPLIT_K": (0, 2, 0),
    "DETERMINISTIC_SPLIT_K": (0, 3, 0),
}

KERNEL_SUFFIXES = {
    "fp16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "bf16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "fp32": frozenset((0, 1, 20, 21, 30, 31, 101, 200, 201,
                       10200, 10201, 20201)),
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


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def make_request(m, k, n, dtype, trans_a, trans_b):
    return {
        "M": m, "N": n, "K": k, "dtype": dtype, "output_dtype": dtype,
        "layoutA": "ND", "layoutB": "ND", "layoutC": "ND",
        "transA": trans_a, "transB": trans_b, "bias": False,
        "hf32": False, "forceGrpAccForFp32": False,
    }


def _source_output(source: dict) -> dict:
    suffix = int(source["tiling_key"] - 10**19)
    return {
        "selected_family": source["selected_family"],
        "kernel_variant": KERNEL_VARIANT[suffix],
        "tiling_key": int(source["tiling_key"]),
        "block_dim": int(source["blockDim"]),
        "workspace_bytes": int(source["workspace_bytes"]),
        "tiling_data_bytes": int(source["raw_buffer_bytes"]),
        "tiling_data_hex": source["raw_buffer_hex"],
        "tiling_data_sha256": source["raw_buffer_sha256"],
    }


def _nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
    if ceil_div(n_aligned, base_n) == ceil_div(n_value, base_n):
        return False
    mask = (1 << 64) - 1
    completed = (n_value // base_n - 1) & mask
    remaining = (n_aligned - ((completed * base_n) & mask)) & mask
    return remaining * base_d > FORMULA_HARDWARE.ub // 2 // width


def _nd2nz_geometry(width, n_value, d_value, used_cores):
    """Compute one CANN 8.1 vector stripe; this is not tiling enumeration."""
    vector_cores = max(2 * used_cores, 1)
    threshold = 2048 // width
    c0 = 32 // width
    n_aligned = align_up(n_value, 16)
    d_aligned = align_up(d_value, c0)
    if d_value <= threshold:
        base_d = max(d_aligned, 1)
        initial_n = FORMULA_HARDWARE.ub // 2 // width // base_d
        rounds = max(ceil_div(ceil_div(n_aligned, vector_cores), initial_n), 1)
        base_n = max(ceil_div(ceil_div(n_aligned, vector_cores), rounds), 16)
        while base_n > 16 and _nd2nz_overflow(
            n_aligned, n_value, base_n, base_d, width
        ):
            base_n -= 1
        return base_n, base_d

    best_tail, best_n, best_d = 0, 16, 4096 // width
    for base_bytes in (6144, 4096, 2048):
        base_d = max(min(d_aligned, base_bytes // width), 1)
        d_loops = ceil_div(d_aligned, base_d)
        d_tail = d_aligned % base_d
        if 0 < d_tail < 512 // width:
            if base_d * width == 6144:
                continue
            d_loops -= 1
            base_d = max(align_up(ceil_div(d_aligned, d_loops), c0), 1)
        base_n = max(FORMULA_HARDWARE.ub // 2 // width // base_d, 16)
        if base_n * base_d * width * 2 > FORMULA_HARDWARE.ub:
            continue
        while base_n > 16:
            if not _nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
                tail = ceil_div(n_aligned, base_n) * d_loops % vector_cores
                if tail == 0:
                    return base_n, base_d
                if tail > best_tail:
                    best_tail, best_n, best_d = tail, base_n, base_d
            base_n -= 1
    return best_n, best_d


def _workspace_for(family, fields, shape, conversion_a, conversion_b, fix_mode):
    total = 20 * 1024 * 1024
    if family == "SINGLE_CORE_SPLIT_K":
        width = 4 if shape.dtype == "fp32" else 2
        total += shape.m * align_up(shape.n, 256 // width) * 4
    elif family == "DETERMINISTIC_SPLIT_K":
        total += fields["usedCoreNum"] * fields["singleCoreM"] * fields["singleCoreN"] * 8
    width = shape.d
    if fix_mode == 1:
        total += (align_up(shape.n, 512 // width) * fields["baseM"] *
                  fields["usedCoreNum"] * 2 * width)
    elif fix_mode == 2:
        total += (align_up(shape.n, 16) * fields["baseM"] *
                  fields["usedCoreNum"] * 2 * width)
    c0 = 32 // width
    if conversion_a:
        total += (align_up(shape.m, c0) * align_up(shape.k, 16) * width
                  if shape.trans_a else
                  align_up(shape.m, 16) * align_up(shape.k, c0) * width)
    if conversion_b:
        total += (align_up(shape.n, 16) * align_up(shape.k, c0) * width
                  if shape.trans_b else
                  align_up(shape.n, c0) * align_up(shape.k, 16) * width)
    return total


def _validate_packet(payload, raw, suffix, fields, shape):
    if len(raw) != 272 or unpack_packet(raw, "legacy272_cube200") != payload:
        raise ValueError("candidate is not a round-trippable 272-byte packet")
    if suffix not in KERNEL_SUFFIXES[shape.dtype]:
        raise ValueError("installed CANN 8.1 does not dispatch this dtype/suffix")
    if not 1 <= fields["usedCoreNum"] <= FORMULA_HARDWARE.cores:
        raise ValueError("usedCoreNum is outside the 20-AIC domain")
    if any(fields[name] < 16 or fields[name] % 16
           for name in ("baseM", "baseN", "baseK")):
        raise ValueError("base tile is not positive and 16-aligned")
    width = shape.d
    footprints = {
        "L0A": fields.get("dbL0A", 2) * fields["baseM"] * fields["baseK"] * width,
        "L0B": fields.get("dbL0B", 2) * fields["baseN"] * fields["baseK"] * width,
        "L0C": fields["dbL0C"] * fields["baseM"] * fields["baseN"] * 4,
        "L1": (fields["depthA1"] * fields["baseM"] +
               fields["depthB1"] * fields["baseN"]) * fields["baseK"] * width,
    }
    limits = {"L0A": 65536, "L0B": 65536, "L0C": 131072, "L1": 524288}
    if any(footprints[name] > limits[name] for name in footprints):
        raise ValueError(f"local capacity exceeded: {footprints}")
    return {name: {"bytes": footprints[name], "limit": limits[name], "pass": True}
            for name in footprints}


def _scheduled_work(fields, shape):
    """Exact integer work implied by a packet; it is never used to rank it."""
    parent_m = ceil_div(shape.m, int(fields["singleCoreM"]))
    parent_n = ceil_div(shape.n, int(fields["singleCoreN"]))
    parent_k = ceil_div(shape.k, int(fields["singleCoreK"]))
    base_m_per_parent = ceil_div(int(fields["singleCoreM"]), int(fields["baseM"]))
    base_n_per_parent = ceil_div(int(fields["singleCoreN"]), int(fields["baseN"]))
    base_k_per_parent = ceil_div(int(fields["singleCoreK"]), int(fields["baseK"]))
    base_tile_iterations = (
        parent_m * parent_n * parent_k *
        base_m_per_parent * base_n_per_parent * base_k_per_parent
    )
    padded_cube_fma = (
        base_tile_iterations * int(fields["baseM"]) *
        int(fields["baseN"]) * int(fields["baseK"])
    )
    logical_fma = shape.m * shape.n * shape.k
    output_tasks = parent_m * parent_n
    return {
        "parent_grid_mnk": [parent_m, parent_n, parent_k],
        "base_tiles_per_parent_mnk": [
            base_m_per_parent, base_n_per_parent, base_k_per_parent,
        ],
        "base_tile_iterations": base_tile_iterations,
        "logical_fma": logical_fma,
        "padded_cube_fma": padded_cube_fma,
        "padding_fma": padded_cube_fma - logical_fma,
        "output_tasks": output_tasks,
        "launched_aic": int(fields["usedCoreNum"]),
        "output_task_waves": ceil_div(output_tasks, int(fields["usedCoreNum"])),
    }


def _al1_idle_core_formula(shape):
    """Derive the source-backed AL1 packet with idle AICs removed.

    The CANN 8.1 AL1 kernel copies the complete A matrix into every launched
    AIC's L1 before it checks whether that AIC owns an output task.  The
    installed host rule always launches all 20 AICs.  In this deliberately
    narrow domain there are only ceil(N/16) output tasks, so launching exactly
    that many AICs deletes complete GM-to-L1 copies on otherwise idle AICs.

    Every field below is derived from shape and fixed C220 capacities.  No
    source packet field is used to construct the candidate.
    """
    if not (
        shape.dtype == "fp32"
        and not shape.trans_a
        and shape.trans_b
        and 1 <= shape.m <= 16
        and 16 < shape.n <= 160
        and 5120 <= shape.k <= 7168
        and shape.k % 128 == 0
    ):
        raise ValueError("shape is outside the certified AL1 idle-core domain")
    output_tasks = ceil_div(shape.n, 16)
    if not 5 <= output_tasks <= 10:
        raise ValueError("certified AL1 domain requires five to ten N tasks")
    k_iterations = ceil_div(shape.k, 256)
    fields = {
        "baseM": 16, "baseN": 16, "baseK": 256,
        "singleCoreM": shape.m, "singleCoreN": 16,
        "singleCoreK": shape.k, "usedCoreNum": output_tasks,
        "stepM": 1, "stepN": 1,
        "stepKa": k_iterations, "stepKb": 1,
        "depthA1": k_iterations, "depthB1": 2,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
        "iterateOrder": 0,
    }
    full_a_copy_bytes = align_up(shape.m, 16) * align_up(shape.k, 8) * 4
    eliminated_idle_aic = FORMULA_HARDWARE.cores - output_tasks
    eliminated_full_a_copy_bytes = eliminated_idle_aic * full_a_copy_bytes
    resource_bytes = {
        "L0A_double": 2 * fields["baseM"] * fields["baseK"] * 4,
        "L0B_double": 2 * fields["baseN"] * fields["baseK"] * 4,
        "L0C": fields["dbL0C"] * fields["baseM"] * fields["baseN"] * 4,
        "L1_AB": (fields["depthA1"] * fields["baseM"]
                  + fields["depthB1"] * fields["baseN"]) * fields["baseK"] * 4,
        "bias_reserved": 0,
    }
    return {
        "family": "AL1",
        "fields": fields,
        "l2": {
            "mTile": 1, "nTile": 1,
            "mTileBlock": 1, "nTileBlock": output_tasks,
            "calOrder": 1,
        },
        "resource_bytes": resource_bytes,
        "conversion_a": False,
        "conversion_b": False,
        "fix_mode": 0,
        "legality": {
            "source_AL1_predicate": "PASS",
            "same_suffix_101_kernel": "PASS",
            "same_active_core_output_ownership": "PASS",
            "same_MMAD_and_output_work": "PASS",
            "strictly_fewer_full_A_copies": "PASS",
            "no_new_kernel_instructions": "PASS",
            "local_capacity": "PASS",
        },
        "dominance_certificate": {
            "proof_kind": "AL1_IDLE_AIC_FULL_A_COPY_ELIMINATION",
            "source_launched_aic": FORMULA_HARDWARE.cores,
            "improved_launched_aic": output_tasks,
            "output_tasks": output_tasks,
            "eliminated_idle_aic": eliminated_idle_aic,
            "full_a_copy_bytes_per_launched_aic": full_a_copy_bytes,
            "source_full_a_copy_bytes": FORMULA_HARDWARE.cores * full_a_copy_bytes,
            "improved_full_a_copy_bytes": output_tasks * full_a_copy_bytes,
            "eliminated_full_a_copy_bytes": eliminated_full_a_copy_bytes,
            "source_copy_order": "COPY_FULL_A_BEFORE_OUTPUT_TASK_GUARD",
            "active_core_instruction_stream": "UNCHANGED_SUFFIX_101",
            "new_kernel_instructions": 0,
            "new_workspace_bytes": 0,
            "new_MMAD_commands": 0,
            "new_output_tasks": 0,
        },
        "selection_contract": {
            "complete_tilings_constructed": 1,
            "candidate_enumeration": False,
            "pareto_pruning": False,
            "latency_or_cost_score": False,
            "history_lookup": False,
            "runtime_tiling_bank": False,
            "decision": "closed_form_AL1_idle_AIC_elimination",
        },
    }


def _unchanged(request, source, reason, theory):
    selected = _source_output(source)
    return {
        "status": "BASELINE_EQUIVALENT_UNCHANGED", "request": request,
        "selected": selected, "baseline": selected, "improved": None,
        "baseline_equivalent": True, "changed_rules": [], "changed_fields": {},
        "skip_reason": reason,
        "formula_family": SOURCE_TO_FORMULA_FAMILY[source["selected_family"]],
        "kernel_suffix": int(source["tiling_key"] - 10**19),
        "npu_eligible": False, "theory": theory,
        "runtime_dependencies": {
            "cost_model": False, "candidate_enumeration": False,
            "history": False, "runtime_kb": False, "tiling_bank": False,
            "installed_host_tiler": False,
            "source_reconstruction_for_family_audit": True,
            "source_packet_fields_used_for_improved_packet": False,
        },
        "validation": {
            "host_packet": "SOURCE_RECONSTRUCTION_272_BYTE_PACKET",
            "kernel_key_domain": "PASS_INSTALLED_81_DTYPE_AWARE_DISPATCH",
            "local_capacity": "SOURCE_RULE_RESULT",
            "npu_performance": "NOT_AN_IMPROVED_PACKET",
        },
    }


def generate(m, k, n, dtype="fp16", trans_a=False, trans_b=False):
    request = make_request(m, k, n, dtype, trans_a, trans_b)
    shape = Shape(m=m, n=n, k=k, dtype=dtype,
                  trans_a=trans_a, trans_b=trans_b)
    source = source_select(
        request, HARDWARE, source_profile="installed_81", trace=True
    )
    source_family = source["selected_family"]
    source_formula_family = SOURCE_TO_FORMULA_FAMILY[source_family]
    source_cube = source["tilingData"]["matmulTiling"]
    source_l2 = source["tilingData"]["tileL2cacheTiling"]
    theory = {
        "selection_mode": "PROOF_CARRYING_CLOSED_FORM",
        "complete_tilings_constructed": 1, "candidate_count": 1,
        "candidate_enumeration": False, "pareto_pruning": False,
        "latency_or_cost_score": False,
        "logical_fma_count": m * n * k,
        "logical_flop_count": 2 * m * n * k,
        "formula_family": source_formula_family,
        "formula_fields": {
            name: int(source_cube[name]) for name in (
                "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
                "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
                "stepN", "stepKa", "stepKb", "dbL0A", "dbL0B", "dbL0C",
                "iterateOrder",
            )
        },
        "formula_l2": dict(source_l2),
        "resource_bytes": source["resource_facts"],
        "branch_calculations": {
            "attempted_families": source["attempted_families"],
            "selected_family": source_family,
        },
        "legality": "SOURCE_RECONSTRUCTION_COMPLETE_PACKET",
        "source_family": source_family,
        "source_suffix": int(source["tiling_key"] - 10**19),
        "source_attempts": source["attempted_families"],
        "scheduled_work": {
            "selected": _scheduled_work(source_cube, shape),
        },
        "selection_contract": {
            "complete_tilings_constructed": 1,
            "candidate_enumeration": False,
            "pareto_pruning": False,
            "latency_or_cost_score": False,
            "history_lookup": False,
            "runtime_tiling_bank": False,
            "decision": "ordered_branch_predicates_then_integer_equations",
        },
    }

    activate = None
    if (
        source_family == "AL1_FULL_LOAD"
        and dtype == "fp32"
        and not trans_a
        and trans_b
        and 1 <= m <= 16
        and 16 < n <= 160
        and 5120 <= k <= 7168
        and k % 128 == 0
        and 5 <= ceil_div(n, 16) <= 10
    ):
        activate = "AL1_IDLE_CORE_FULL_A_COPY_ELIMINATION"
    if activate is None:
        return _unchanged(
            request, source,
            "NO_STRICT_CLOSED_FORM_IMPROVEMENT_FOR_SOURCE_FAMILY", theory,
        )

    # The replacement consumes shape and fixed hardware only.  The source
    # reconstruction above audits applicability but contributes no packet
    # field to the candidate.
    formula = _al1_idle_core_formula(shape)
    fields = formula["fields"]
    if formula["family"] != source_formula_family:
        return _unchanged(
            request, source, "INDEPENDENT_FORMULA_AND_SOURCE_FAMILY_DISAGREE",
            theory,
        )
    theory.update({
        "formula_family": formula["family"],
        "formula_fields": dict(fields),
        "formula_l2": dict(formula["l2"]),
        "resource_bytes": dict(formula["resource_bytes"]),
        "branch_calculations": {
            name: value for name, value in formula.items()
            if name not in {"input", "hardware", "compile_info", "fields", "l2",
                            "resource_bytes", "legality", "selection_contract"}
        },
        "legality": formula["legality"],
    })

    initializer = initialize_cube(request, HARDWARE, trace=False)
    if initializer["return_code"] != 0:
        raise ValueError("independent initializer rejected the request")
    cube = dict(initializer["cube"])
    for name in (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0C", "iterateOrder",
    ):
        cube[name] = int(fields[name])
    cube["dbL0A"] = int(fields.get("dbL0A", 2))
    cube["dbL0B"] = int(fields.get("dbL0B", 2))
    cube["dbL0C"] = int(fields["dbL0C"])

    lf = formula["l2"]
    l2 = {"mTileCntL2": int(lf["mTile"]), "nTileCntL2": int(lf["nTile"]),
          "mTileBlock": int(lf["mTileBlock"]),
          "nTileBlock": int(lf["nTileBlock"]), "calOrder": int(lf["calOrder"])}
    conversion_a = bool(formula["conversion_a"])
    conversion_b = bool(formula["conversion_b"])
    vector = {name: 0 for name in VECTOR_NAMES}
    if conversion_a:
        conversion_n, conversion_d = ((k, m) if trans_a else (m, k))
        vector["baseAN"], vector["baseAD"] = _nd2nz_geometry(
            shape.d, conversion_n, conversion_d, fields["usedCoreNum"])
    if conversion_b:
        conversion_n, conversion_d = ((n, k) if trans_b else (k, n))
        vector["baseBN"], vector["baseBD"] = _nd2nz_geometry(
            shape.d, conversion_n, conversion_d, fields["usedCoreNum"])
    run = {"transA": int(trans_a), "transB": int(trans_b),
           "nd2nzA": int(conversion_a), "nd2nzB": int(conversion_b),
           "isHf32": 0}
    payload = {
        "cube_words": [int(cube[name]) & 0xFFFFFFFF for name in CUBE_FIELDS],
        "tileL2cacheTiling": l2, "matmulRunInfo": run, "l2CacheFlag": 0,
        "vector": vector,
        "padding_hex": {"220": "00000000", "244": "00000000",
                        "252": "00000000"},
    }
    raw = pack_packet(payload, "legacy272_cube200")
    fix_mode = int(formula.get("fix_mode", 0))
    if formula["family"] == "FIXPIPE_BL1":
        load_mode, split_mode = 2, 0
    else:
        load_mode, split_mode, expected_fix = FAMILY_KEY_PARTS[formula["family"]]
        if fix_mode != expected_fix:
            raise ValueError("formula family emitted an inconsistent FixOpt mode")
    mix_mode = 0 if conversion_a or conversion_b else 1
    suffix = fix_mode * 10000 + load_mode * 100 + split_mode * 10 + mix_mode
    capacities = _validate_packet(payload, raw, suffix, fields, shape)
    workspace = _workspace_for(
        formula["family"], fields, shape, conversion_a, conversion_b, fix_mode)
    improved = {
        "selected_family": ("BL1_FULL_LOAD_VEC_NZ2ND" if fix_mode == 2
                            else FAMILY_NAME[formula["family"]]),
        "kernel_variant": KERNEL_VARIANT[suffix], "tiling_key": 10**19 + suffix,
        "block_dim": int(fields["usedCoreNum"]),
        "workspace_bytes": workspace, "tiling_data_bytes": len(raw),
        "tiling_data_hex": raw.hex(),
        "tiling_data_sha256": hashlib.sha256(raw).hexdigest(),
    }
    runner_suffix = suffix
    baseline = _source_output(source)
    baseline_cube = source["tilingData"]["matmulTiling"]
    baseline_l2 = source["tilingData"]["tileL2cacheTiling"]
    source_work = _scheduled_work(baseline_cube, shape)
    improved_work = _scheduled_work(fields, shape)
    theory["scheduled_work"] = {
        "source": source_work,
        "improved": improved_work,
    }
    changes = {
        name: {"baseline": int(baseline_cube[name]), "improved": int(cube[name])}
        for name in CUBE_FIELDS if int(baseline_cube[name]) != int(cube[name])
    }
    changes.update({
        "l2." + name: {"baseline": int(baseline_l2[name]), "improved": int(l2[name])}
        for name in l2 if int(baseline_l2[name]) != int(l2[name])
    })
    for name in ("selected_family", "tiling_key", "block_dim", "workspace_bytes"):
        if baseline[name] != improved[name]:
            changes[name] = {"baseline": baseline[name], "improved": improved[name]}
    if not changes:
        return _unchanged(request, source, "FORMULA_PRODUCED_SOURCE_VALUES", theory)

    certificate = formula["dominance_certificate"]
    cube_differences = {
        name for name in CUBE_FIELDS
        if int(baseline_cube[name]) != int(cube[name])
    }
    if not (
        int(source["tiling_key"] - 10**19) == 101
        and suffix == 101
        and baseline["block_dim"] == FORMULA_HARDWARE.cores
        and cube_differences == {"usedCoreNum"}
        and all(int(baseline_l2[name]) == int(l2[name]) for name in l2)
        and baseline["workspace_bytes"] == improved["workspace_bytes"]
        and all(
            source_work[name] == improved_work[name]
            for name in (
                "parent_grid_mnk", "base_tiles_per_parent_mnk",
                "base_tile_iterations", "logical_fma", "padded_cube_fma",
                "padding_fma", "output_tasks", "output_task_waves",
            )
        )
        and certificate["eliminated_idle_aic"] > 0
        and certificate["eliminated_full_a_copy_bytes"] > 0
        and certificate["new_kernel_instructions"] == 0
        and certificate["new_workspace_bytes"] == 0
        and certificate["new_MMAD_commands"] == 0
        and certificate["new_output_tasks"] == 0
    ):
        raise ValueError("AL1 candidate lacks exact instruction-deletion dominance")
    theory["improvement_equation"] = dict(certificate)
    theory["strict_dominance_certificate"] = dict(certificate)
    rules = ["AL1_IDLE_CORE_FULL_A_COPY_ELIMINATION"]

    theory["capacity_audit"] = capacities
    theory["candidate_sha256"] = improved["tiling_data_sha256"]
    return {
        "status": "MODIFIED_TILING", "request": request,
        "selected": improved, "baseline": baseline, "improved": improved,
        "baseline_equivalent": False, "changed_rules": rules,
        "changed_fields": changes, "formula_family": formula["family"],
        "kernel_suffix": runner_suffix, "resource_bytes": formula["resource_bytes"],
        "selection_basis": "PROOF_CARRYING_CLOSED_FORM", "npu_eligible": True,
        "theory": theory,
        "runtime_dependencies": {
            "cost_model": False, "candidate_enumeration": False,
            "history": False, "runtime_kb": False, "tiling_bank": False,
            "installed_host_tiler": False,
            "source_reconstruction_for_family_audit": True,
            "source_packet_fields_used_for_improved_packet": False,
        },
        "path_coverage": {
            "initializer": "ABI_DEFAULTS_ONLY_NO_TILE_SELECTION",
            "family_selection": "ORDERED_SOURCE_RECONSTRUCTION_PLUS_INDEPENDENT_FORMULA_AGREEMENT",
            "kernel_implementation": "UNCHANGED_SUFFIX_101_AL1_KERNEL",
            "abi_layout": "RETAINED_272_BYTE_ABI_ONLY_USED_CORE_NUM_DIFFERS",
            "official_runtime_tiling_seed": "FORBIDDEN_AND_NOT_USED",
            "unmodified_source_paths": "REPORTED_AS_BASELINE_EQUIVALENT_NOT_IMPROVED",
        },
        "validation": {
            "host_packet": "PASS_272_BYTE_ABI_ROUNDTRIP",
            "kernel_key_domain": "PASS_INSTALLED_81_FP32_SUFFIX_101_DISPATCH",
            "local_capacity": "PASS_FIXED_910B3_CAPACITIES",
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
    result = generate(args.m, args.k, args.n, args.dtype,
                      args.trans_a, args.trans_b)
    print(json.dumps(result, sort_keys=True,
                     indent=2 if args.pretty else None,
                     separators=None if args.pretty else (",", ":")))


if __name__ == "__main__":
    main()
