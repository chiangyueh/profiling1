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

from formula_rules import CompileInfo, Hardware, Shape, solve  # noqa: E402
from base_schedule_theory import (  # noqa: E402
    BaseShape as TheoryBaseShape,
    analyze as analyze_base_schedule,
    pack_schedule_extension,
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


def _base_schedule_formula(shape, analysis):
    """Build the one packet admitted by the BASE ownership proof.

    Tile geometry is derived by base_schedule_theory and is not copied from
    the reconstructed source packet.  The custom kernel changes ownership;
    the packet deliberately retains the native C220 pipeline geometry.
    """
    packet = analysis["native_packet"]
    fields = {
        "baseM": int(packet["base_m"]),
        "baseN": int(packet["base_n"]),
        "baseK": int(packet["base_k"]),
        "singleCoreM": int(packet["single_core_m"]),
        "singleCoreN": int(packet["single_core_n"]),
        "singleCoreK": int(packet["single_core_k"]),
        "usedCoreNum": int(packet["used_cores"]),
        "stepM": 1,
        "stepN": 1,
        "stepKa": int(packet["step_ka"]),
        "stepKb": int(packet["step_kb"]),
        "depthA1": int(packet["depth_a1"]),
        "depthB1": int(packet["depth_b1"]),
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": int(packet["db_l0c"]),
        "iterateOrder": 0,
    }
    width = shape.d
    resource_bytes = {
        "L0A_double": 2 * fields["baseM"] * fields["baseK"] * width,
        "L0B_double": 2 * fields["baseN"] * fields["baseK"] * width,
        "L0C": fields["dbL0C"] * fields["baseM"] * fields["baseN"] * 4,
        "L1_AB": (
            fields["depthA1"] * fields["baseM"]
            + fields["depthB1"] * fields["baseN"]
        ) * fields["baseK"] * width,
        "bias_reserved": 0,
    }
    return {
        "family": "BASE",
        "fields": fields,
        "l2": {
            "mTile": 1,
            "nTile": 1,
            "mTileBlock": int(analysis["grid"]["m_count"]),
            "nTileBlock": int(analysis["grid"]["n_count"]),
            "calOrder": 0,
        },
        "resource_bytes": resource_bytes,
        "conversion_a": False,
        "conversion_b": False,
        "legality": {
            "native_C220_capacity": "PASS",
            "one_L2_rectangle": "PASS",
            "ownership_exact_cover": "PASS",
            "aggregate_issued_work_equal": "PASS",
            "all_resource_maxima_nonincreasing": "PASS",
            "multidimensional_balance_strictly_improved": "PASS",
        },
        "base_schedule_analysis": analysis,
        "selection_contract": analysis["selection_contract"],
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
    compile_info = CompileInfo(
        aicore_arch=220, support_l0c2out=True, support_l12_bt_bf16=False,
        orig_dtype_x1=dtype, orig_dtype_x2=dtype, orig_dtype_y=dtype,
        orig_dtype_bias=dtype, format_x1="ND", format_x2="ND",
        format_y="ND", total_ub_size=196352,
    )

    source = source_select(
        request, HARDWARE, source_profile="installed_81", trace=True
    )
    source_family = source["selected_family"]
    source_formula_family = SOURCE_TO_FORMULA_FAMILY[source_family]
    source_cube = source["tilingData"]["matmulTiling"]
    source_l2 = source["tilingData"]["tileL2cacheTiling"]
    theory = {
        "selection_mode": "UNIQUE_ORDERED_CLOSED_FORM",
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
    base_schedule_analysis = None
    if source_family == "BASE" and dtype in ("fp16", "bf16") and not trans_a and not trans_b:
        base_schedule_analysis = analyze_base_schedule(
            TheoryBaseShape(m=m, n=n, k=k, dtype=dtype)
        )
        native = base_schedule_analysis["native_packet"]
        expected_cube = {
            "baseM": native["base_m"], "baseN": native["base_n"],
            "baseK": native["base_k"],
            "singleCoreM": native["single_core_m"],
            "singleCoreN": native["single_core_n"],
            "singleCoreK": native["single_core_k"],
            "usedCoreNum": native["used_cores"],
            "stepM": 1, "stepN": 1,
            "stepKa": native["step_ka"], "stepKb": native["step_kb"],
            "depthA1": native["depth_a1"], "depthB1": native["depth_b1"],
            "dbL0C": native["db_l0c"], "iterateOrder": 0,
        }
        expected_l2 = {
            "mTileCntL2": 1, "nTileCntL2": 1,
            "mTileBlock": base_schedule_analysis["grid"]["m_count"],
            "nTileBlock": base_schedule_analysis["grid"]["n_count"],
            "calOrder": 0,
        }
        source_audit = {
            "cube_native_match": all(
                int(source_cube[name]) == int(value)
                for name, value in expected_cube.items()
            ),
            "one_rectangle_match": all(
                int(source_l2[name]) == int(value)
                for name, value in expected_l2.items()
            ),
            "no_head_conversion": not (
                source["tilingData"]["matmulRunInfo"]["nd2nzA"]
                or source["tilingData"]["matmulRunInfo"]["nd2nzB"]
            ),
        }
        base_schedule_analysis["source_scope_audit"] = source_audit
        theory["base_schedule_analysis"] = base_schedule_analysis
        if (
            base_schedule_analysis["decision"] == "ENABLE_NEW_SCHEDULER"
            and all(source_audit.values())
        ):
            activate = "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE"
    elif source_family == "AL1_FULL_LOAD":
        activate = "AL1_CAPACITY_DERIVED_K_GRAIN"
    elif (source_family == "BL1_FULL_LOAD_FIXPIPE" and
          int(source["tiling_key"] - 10**19) == 20201 and n > 16):
        activate = "FIXPIPE_VECTOR_MULTI_GROUP_PIPELINE"
    if activate is None:
        return _unchanged(
            request, source,
            "NO_STRICT_CLOSED_FORM_IMPROVEMENT_FOR_SOURCE_FAMILY", theory,
        )

    # The replacement formula consumes only shape, fixed hardware and compile
    # flags.  The source packet above selects the execution family but none of
    # its tile fields are arguments to solve().
    if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE":
        formula = _base_schedule_formula(shape, base_schedule_analysis)
    else:
        formula = solve(shape, FORMULA_HARDWARE, compile_info)
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
    if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE":
        raw += pack_schedule_extension(base_schedule_analysis)
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
    if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE":
        improved["selected_family"] = "BASE_MULTIDIMENSIONAL_BALANCED"
        improved["kernel_variant"] = "BASE_ALIGNED_CUSTOM_BALANCED_RANGES"
    runner_suffix = (
        901 if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE" else suffix
    )
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
    if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE":
        changes["kernel_scheduler"] = {
            "baseline": "LCM_STAGGERED",
            "improved": "FOUR_CLASS_MULTIDIMENSIONAL_BALANCED_RANGES",
        }
    if not changes:
        return _unchanged(request, source, "FORMULA_PRODUCED_SOURCE_VALUES", theory)

    if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE":
        analysis = base_schedule_analysis
        if not (
            analysis["invariants"]["aggregate_issued_work_equal"]
            and analysis["all_resource_maxima_nonincreasing"]
            and analysis["decision"] == "ENABLE_NEW_SCHEDULER"
            and fields["baseM"] == 128
            and fields["baseN"] == 256
            and fields["baseK"] == 64
            and l2["mTileCntL2"] == 1
            and l2["nTileCntL2"] == 1
            and l2["calOrder"] == 0
        ):
            raise ValueError("BASE scheduler lacks its parameter-free dominance proof")
        theory["improvement_equation"] = {
            "proof_kind": "coefficient_free_multidimensional_load_balance",
            "ownership_before": "LCM-staggered source mapping",
            "ownership_after": "four exact task classes with per-core ranges",
            "native_tile": [fields["baseM"], fields["baseN"], fields["baseK"]],
            "source_max_per_core": analysis["source_max_per_core"],
            "improved_max_per_core": analysis["proposed_max_per_core"],
            "strictly_reduced_resources": analysis["strictly_reduced_resources"],
            "source_worst_normalized_load": analysis["source_balance"][
                "worst_normalized_max_over_mean"],
            "improved_worst_normalized_load": analysis["proposed_balance"][
                "worst_normalized_max_over_mean"],
            "indivisibility_lower_bound": analysis["proposed_balance"][
                "indivisibility_lower_bound"],
            "all_resource_maxima_nonincreasing": True,
            "aggregate_issued_work_equal": True,
            "latency_weights": False,
            "logical_fma_delta": 0,
        }
        rules = ["BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE"]
    elif activate == "AL1_CAPACITY_DERIVED_K_GRAIN":
        old_k_loops = ceil_div(k, int(baseline_cube["baseK"]))
        new_k_loops = ceil_div(k, int(fields["baseK"]))
        old_tasks = ceil_div(n, int(baseline_cube["singleCoreN"]))
        new_tasks = ceil_div(n, int(fields["singleCoreN"]))
        if not (new_k_loops <= old_k_loops and new_tasks == old_tasks and
                fields["usedCoreNum"] == min(20, new_tasks) and
                improved_work["base_tile_iterations"] <
                source_work["base_tile_iterations"] and
                improved_work["padded_cube_fma"] <=
                source_work["padded_cube_fma"] and
                (new_k_loops < old_k_loops or
                 fields["usedCoreNum"] < baseline["block_dim"])):
            raise ValueError("AL1 rule lacks its declared monotonic improvement")
        theory["improvement_equation"] = {
            "proof_kind": "same_output_tasks_fewer_k_iterations_or_idle_cores",
            "source_k_iterations": old_k_loops,
            "improved_k_iterations": new_k_loops,
            "output_tasks": new_tasks,
            "source_launched_aic": baseline["block_dim"],
            "improved_launched_aic": fields["usedCoreNum"],
            "source_base_tile_iterations": source_work["base_tile_iterations"],
            "improved_base_tile_iterations": improved_work["base_tile_iterations"],
            "source_padded_cube_fma": source_work["padded_cube_fma"],
            "improved_padded_cube_fma": improved_work["padded_cube_fma"],
            "padded_cube_fma_delta": (
                improved_work["padded_cube_fma"] - source_work["padded_cube_fma"]
            ),
            "eliminated_k_loop_boundaries": (
                source_work["base_tile_iterations"] -
                improved_work["base_tile_iterations"]
            ),
            "logical_fma_delta": 0,
        }
        rules = ["AL1_CAPACITY_DERIVED_K_GRAIN"]
    else:
        output_tasks = ceil_div(m, int(fields["singleCoreM"]))
        waves = ceil_div(output_tasks, int(fields["usedCoreNum"]))
        if not (n > 16 and waves >= 2 and fields["dbL0C"] == 2 and
                fields["depthA1"] >= 2 and
                improved_work["base_tile_iterations"] ==
                source_work["base_tile_iterations"] and
                improved_work["padded_cube_fma"] ==
                source_work["padded_cube_fma"]):
            raise ValueError("Fixpipe pipeline cannot reach steady state")
        theory["improvement_equation"] = {
            "proof_kind": "multi_wave_aic_aiv_pipeline_overlap",
            "output_tasks": output_tasks, "active_aic": fields["usedCoreNum"],
            "task_waves": waves,
            "source_l0c_buffers": int(baseline_cube["dbL0C"]),
            "improved_l0c_buffers": fields["dbL0C"],
            "source_a_depth": int(baseline_cube["depthA1"]),
            "improved_a_depth": fields["depthA1"],
            "source_base_tile_iterations": source_work["base_tile_iterations"],
            "improved_base_tile_iterations": improved_work["base_tile_iterations"],
            "source_padded_cube_fma": source_work["padded_cube_fma"],
            "improved_padded_cube_fma": improved_work["padded_cube_fma"],
            "overlap_opportunities": max(output_tasks - fields["usedCoreNum"], 0),
            "single_buffer_symbolic_critical_path": "q*(Tcube+Tfix)",
            "double_buffer_symbolic_critical_path": (
                "Tcube+Tfix+(q-1)*max(Tcube,Tfix)"
            ),
            "q_max_tasks_per_active_aic": waves,
            "logical_fma_delta": 0,
        }
        rules = ["FIXPIPE_VECTOR_MULTI_GROUP_PIPELINE"]

    theory["capacity_audit"] = capacities
    theory["candidate_sha256"] = improved["tiling_data_sha256"]
    return {
        "status": "MODIFIED_TILING", "request": request,
        "selected": improved, "baseline": baseline, "improved": improved,
        "baseline_equivalent": False, "changed_rules": rules,
        "changed_fields": changes, "formula_family": formula["family"],
        "kernel_suffix": runner_suffix, "resource_bytes": formula["resource_bytes"],
        "selection_basis": "UNIQUE_ORDERED_CLOSED_FORM", "npu_eligible": True,
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
            "kernel_implementation": (
                "CUSTOM_BASE_OWNERSHIP_WITH_RETAINED_CANN81_MATMUL_PIPELINE"
                if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE" else
                "RETAINED_INSTALLED_CANN_81_BRANCH_AND_MARKED"
            ),
            "abi_layout": (
                "RETAINED_272_BYTE_BASE_PLUS_672_BYTE_BALANCE_SCHEDULE"
                if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE" else
                "RETAINED_272_BYTE_ABI_AND_MARKED"
            ),
            "official_runtime_tiling_seed": "FORBIDDEN_AND_NOT_USED",
            "unmodified_source_paths": "REPORTED_AS_BASELINE_EQUIVALENT_NOT_IMPROVED",
        },
        "validation": {
            "host_packet": (
                "PASS_272_BYTE_BASE_PLUS_AUDITED_672_BYTE_SCHEDULE"
                if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE" else
                "PASS_272_BYTE_ABI_ROUNDTRIP"
            ),
            "kernel_key_domain": (
                "CUSTOM_DIRECT_BASE_KERNEL_COMPILED_FOR_SUFFIX_1"
                if activate == "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE" else
                "PASS_INSTALLED_81_DTYPE_AWARE_DISPATCH"
            ),
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
