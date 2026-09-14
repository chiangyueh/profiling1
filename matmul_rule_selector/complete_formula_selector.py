#!/usr/bin/env python3
"""Independent closed-form CANN 8.1 MatMulV3 packet generator.

The only inputs to ``generate`` are the logical problem and fixed C220
capacity/capability values.  ``formula_rules.solve`` constructs exactly one
complete schedule.  This module maps that schedule to one of the twelve
installed CANN 8.1 kernel specializations and materializes the legacy
272-byte ABI without calling an official selector or a candidate ranker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from typing import Iterable

from formula_rules import CompileInfo, Hardware, Shape, solve


CUBE_FIELDS = (
    "usedCoreNum", "M", "N", "Ka", "Kb", "singleCoreM", "singleCoreN",
    "singleCoreK", "baseM", "baseN", "baseK", "depthA1", "depthB1",
    "stepM", "stepN", "isBias", "transLength", "iterateOrder", "shareMode",
    "shareL1Size", "shareL0CSize", "shareUbSize", "batchM", "batchN",
    "singleBatchM", "singleBatchN", "stepKa", "stepKb", "depthAL1CacheUB",
    "depthBL1CacheUB", "dbL0A", "dbL0B", "dbL0C", "ALayoutInfoB",
    "ALayoutInfoS", "ALayoutInfoN", "ALayoutInfoG", "ALayoutInfoD",
    "BLayoutInfoB", "BLayoutInfoS", "BLayoutInfoN", "BLayoutInfoG",
    "BLayoutInfoD", "CLayoutInfoB", "CLayoutInfoS1", "CLayoutInfoN",
    "CLayoutInfoG", "CLayoutInfoS2", "BatchNum", "reserved",
)

PACKET_BYTES = 272
SYSTEM_WORKSPACE_BYTES = 20 * 1024 * 1024
INSTALLED_SUFFIXES = (0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201)
SUPPORTED_SUFFIXES = {
    "fp16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "bf16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "fp32": frozenset(INSTALLED_SUFFIXES),
}

FAMILY_KEY_DIGITS = {
    "BASE": (0, 0, 0),
    "SINGLE_CORE_SPLIT_K": (0, 2, 0),
    "DETERMINISTIC_SPLIT_K": (0, 3, 0),
    "AL1": (1, 0, 0),
    "BL1": (2, 0, 0),
}


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value>=0 and divisor>0")
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _compile_info(shape: Shape, hardware: Hardware) -> CompileInfo:
    return CompileInfo(
        aicore_arch=220,
        support_l0c2out=True,
        support_l12_bt_bf16=False,
        orig_dtype_x1=shape.dtype,
        orig_dtype_x2=shape.b_dtype,
        orig_dtype_y=shape.c_dtype,
        orig_dtype_bias=shape.dtype,
        format_x1=shape.format_a,
        format_x2=shape.format_b,
        format_y=shape.format_out,
        total_ub_size=hardware.ub,
    )


def _kernel_suffix(state: dict, dtype: str) -> tuple[int, dict]:
    family = state["family"]
    mix = int(not (state["conversion_a"] or state["conversion_b"]))
    if family == "FIXPIPE_BL1":
        load, split, fix = 2, 0, int(state["fix_mode"])
    else:
        load, split, fix = FAMILY_KEY_DIGITS[family]
    suffix = fix * 10_000 + load * 100 + split * 10 + mix
    if suffix not in INSTALLED_SUFFIXES:
        raise ValueError(
            f"closed-form family maps to a non-installed CANN 8.1 suffix: "
            f"family={family} suffix={suffix}"
        )
    if suffix not in SUPPORTED_SUFFIXES[dtype]:
        raise ValueError(f"dtype={dtype} has no built kernel for suffix={suffix}")
    return suffix, {
        "load_mode": load,
        "split_core_mode": split,
        "fix_mode": fix,
        "mix_nd2nz": mix,
        "decimal_equation": (
            f"{fix}*10000+{load}*100+{split}*10+{mix}={suffix}"
        ),
    }


def _nd2nz_overflow(
    n_aligned: int,
    n_value: int,
    base_n: int,
    base_d: int,
    width: int,
) -> bool:
    if ceil_div(n_aligned, base_n) == ceil_div(n_value, base_n):
        return False
    mask = (1 << 64) - 1
    completed = (n_value // base_n - 1) & mask
    remaining = (n_aligned - ((completed * base_n) & mask)) & mask
    return remaining * base_d > 196_352 // 2 // width


def _nd2nz_base(width: int, n_value: int, d_value: int, used_cores: int) -> tuple[int, int]:
    """Closed-form port of CalcNd2NzTiling's capacity projection."""
    vector_cores = max(2 * used_cores, 1)
    base_threshold = 2048 // width
    c0 = 32 // width
    n_aligned = align_up(n_value, 16)
    d_aligned = align_up(d_value, c0)
    if d_value <= base_threshold:
        base_d = max(d_aligned, 1)
        initial_n = 196_352 // 2 // width // base_d
        rounds = max(ceil_div(ceil_div(n_aligned, vector_cores), initial_n), 1)
        base_n = max(ceil_div(ceil_div(n_aligned, vector_cores), rounds), 16)
        while base_n > 16 and _nd2nz_overflow(
            n_aligned, n_value, base_n, base_d, width
        ):
            base_n -= 1
        return base_n, base_d

    best_tail = 0
    best_n = 16
    best_d = 4096 // width
    for base_bytes in (6144, 4096, 2048):
        base_d = max(min(d_aligned, base_bytes // width), 1)
        d_loop = ceil_div(d_aligned, base_d)
        d_tail = d_aligned % base_d
        if 0 < d_tail < 512 // width:
            if base_d * width == 6144:
                continue
            d_loop -= 1
            base_d = max(align_up(ceil_div(d_aligned, d_loop), c0), 1)
        base_n = max(196_352 // 2 // width // base_d, 16)
        if base_n * base_d * width * 2 > 196_352:
            continue
        if _nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
            continue
        while base_n > 16:
            if _nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
                base_n -= 1
                continue
            tail = ceil_div(n_aligned, base_n) * d_loop % vector_cores
            if tail == 0:
                return base_n, base_d
            if tail > best_tail:
                best_tail, best_n, best_d = tail, base_n, base_d
            base_n -= 1
    return best_n, best_d


def _l2_cache_flag(shape: Shape, state: dict, suffix: int, hardware: Hardware) -> int:
    f = state["fields"]
    a_enabled = False
    b_enabled = False
    c_enabled = shape.m * shape.n * shape.dc <= hardware.l2
    bias_enabled = True
    split = (suffix // 10) % 10
    l2 = state["l2"]
    if split == 0:
        if l2["mTile"] > 1 or l2["nTile"] > 1:
            b_enabled = l2["mTile"] > 1
            a_enabled = l2["nTile"] > 1
        else:
            b_enabled = f["singleCoreM"] < shape.m
            a_enabled = f["singleCoreN"] < shape.n
            tasks = (
                ceil_div(shape.m, f["singleCoreM"])
                * ceil_div(shape.n, f["singleCoreN"])
            )
            if ceil_div(tasks, f["usedCoreNum"]) > 1:
                b_enabled = True
    elif split == 2:
        c_enabled = True
        b_enabled = f["singleCoreM"] < shape.m
        a_enabled = f["singleCoreN"] < shape.n
        m_full = f["singleCoreM"] <= f["baseM"] * f["stepM"]
        n_full = f["singleCoreN"] <= f["baseN"] * f["stepN"]
        k_full = f["singleCoreK"] <= shape.k
        if not m_full and not n_full:
            b_enabled = True
            a_enabled = not k_full
    elif split == 3:
        c_enabled = True
        b_enabled = f["singleCoreM"] < shape.m
        tasks = (
            ceil_div(shape.m, f["singleCoreM"])
            * ceil_div(shape.k, f["singleCoreK"])
        )
        one_round = ceil_div(tasks, f["usedCoreNum"]) <= 1
        n_full = f["singleCoreN"] == shape.n
        if not one_round and not n_full:
            a_enabled = True
            b_enabled = not one_round
    else:
        raise ValueError(f"unsupported split digit {split}")
    if a_enabled and b_enabled and c_enabled and bias_enabled:
        return 1
    flag = 0
    if not a_enabled:
        flag |= 1 << 1
    if not b_enabled:
        flag |= 1 << 2
    if not bias_enabled:
        flag |= 1 << 3
    if not c_enabled:
        flag |= 1 << 4
    return flag


def _workspace_bytes(shape: Shape, state: dict, suffix: int) -> tuple[int, dict]:
    f = state["fields"]
    width = shape.d
    split = (suffix // 10) % 10
    fix = suffix // 10_000
    pieces = {"system": SYSTEM_WORKSPACE_BYTES, "split": 0, "fix": 0, "nd2nz_a": 0, "nd2nz_b": 0}
    if split == 2:
        pieces["split"] = shape.m * align_up(shape.n, 256 // width) * 4
    elif split == 3:
        pieces["split"] = (
            f["usedCoreNum"] * f["singleCoreN"] * f["singleCoreM"] * 2 * 4
        )
    if fix == 1:
        pieces["fix"] = (
            align_up(shape.n, 512 // width) * f["baseM"]
            * f["usedCoreNum"] * 2 * width
        )
    elif fix == 2:
        pieces["fix"] = (
            align_up(shape.n, 16) * f["baseM"]
            * f["usedCoreNum"] * 2 * width
        )
    c0 = 32 // width
    if state["conversion_a"]:
        pieces["nd2nz_a"] = (
            align_up(shape.m, c0) * align_up(shape.k, 16) * width
            if shape.trans_a else
            align_up(shape.m, 16) * align_up(shape.k, c0) * width
        )
    if state["conversion_b"]:
        pieces["nd2nz_b"] = (
            align_up(shape.n, 16) * align_up(shape.k, c0) * width
            if shape.trans_b else
            align_up(shape.n, c0) * align_up(shape.k, 16) * width
        )
    return sum(pieces.values()), pieces


def _cube_words(shape: Shape, state: dict) -> tuple[list[int], dict]:
    f = state["fields"]
    usage = state["resource_bytes"]
    cube = {name: 0 for name in CUBE_FIELDS}
    cube.update({
        "usedCoreNum": f["usedCoreNum"],
        "M": shape.m,
        "N": shape.n,
        "Ka": shape.k,
        "Kb": shape.k,
        "singleCoreM": f["singleCoreM"],
        "singleCoreN": f["singleCoreN"],
        "singleCoreK": f["singleCoreK"],
        "baseM": f["baseM"],
        "baseN": f["baseN"],
        "baseK": f["baseK"],
        "depthA1": f["depthA1"],
        "depthB1": f["depthB1"],
        "stepM": f["stepM"],
        "stepN": f["stepN"],
        "isBias": int(shape.bias),
        "transLength": 0,
        "iterateOrder": f["iterateOrder"],
        "shareMode": 0,
        "shareL1Size": usage["L1_AB"] + usage["bias_reserved"],
        # MatmulTilingAlgorithm::GetUsedSize publishes one base-MN FP32
        # buffer here. dbL0C is a separate pipeline-depth field and must not
        # be multiplied into the shared-buffer ABI length.
        "shareL0CSize": f["baseM"] * f["baseN"] * 4,
        "shareUbSize": 0,
        "batchM": 1,
        "batchN": 1,
        "singleBatchM": 1,
        "singleBatchN": 1,
        "stepKa": f["stepKa"],
        "stepKb": f["stepKb"],
        "depthAL1CacheUB": 0,
        "depthBL1CacheUB": 0,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": f["dbL0C"],
        "BatchNum": 0,
        "reserved": 0,
    })
    words = [cube[name] for name in CUBE_FIELDS]
    if any(type(value) is not int or not 0 <= value < 2**31 for value in words):
        raise ValueError("TCubeTiling contains a value outside signed int32")
    return words, cube


def _packet(shape: Shape, state: dict, suffix: int, hardware: Hardware) -> tuple[bytes, dict]:
    cube_words, cube = _cube_words(shape, state)
    l2 = state["l2"]
    vector = {"baseAN": 0, "baseAD": 0, "baseBN": 0, "baseBD": 0}
    if state["conversion_a"]:
        an, ad = ((shape.k, shape.m) if shape.trans_a else (shape.m, shape.k))
        vector["baseAN"], vector["baseAD"] = _nd2nz_base(
            shape.d, an, ad, state["fields"]["usedCoreNum"]
        )
    if state["conversion_b"]:
        bn, bd = ((shape.n, shape.k) if shape.trans_b else (shape.k, shape.n))
        vector["baseBN"], vector["baseBD"] = _nd2nz_base(
            shape.d, bn, bd, state["fields"]["usedCoreNum"]
        )
    if state["conversion_a"] != bool(vector["baseAN"] and vector["baseAD"]):
        raise ValueError("ND2NZ-A vector geometry does not match the selected conversion path")
    if state["conversion_b"] != bool(vector["baseBN"] and vector["baseBD"]):
        raise ValueError("ND2NZ-B vector geometry does not match the selected conversion path")
    flag = _l2_cache_flag(shape, state, suffix, hardware)
    words = [0] * 68
    words[:50] = cube_words
    words[50:55] = [
        l2["mTile"], l2["nTile"], l2["mTileBlock"],
        l2["nTileBlock"], l2["calOrder"],
    ]
    words[56:61] = [
        int(shape.trans_a), int(shape.trans_b),
        int(state["conversion_a"]), int(state["conversion_b"]), 0,
    ]
    words[62] = flag
    words[64:68] = [vector[name] for name in ("baseAN", "baseAD", "baseBN", "baseBD")]
    if any(type(value) is not int or not 0 <= value < 2**32 for value in words):
        raise ValueError("legacy ABI contains a value outside uint32")
    blob = struct.pack("<68I", *words)
    if len(blob) != PACKET_BYTES or list(struct.unpack("<68I", blob)) != words:
        raise RuntimeError("legacy 272-byte packet roundtrip failed")
    return blob, {"cube": cube, "l2_cache_flag": flag, "vector": vector, "words": words}


def _ownership(shape: Shape, state: dict, suffix: int) -> dict:
    f = state["fields"]
    m_tasks = ceil_div(shape.m, f["singleCoreM"])
    n_tasks = ceil_div(shape.n, f["singleCoreN"])
    k_tasks = ceil_div(shape.k, f["singleCoreK"])
    split = (suffix // 10) % 10
    if split == 3:
        total = k_tasks
        semantics = "K_PARTIALS_REPEATED_FOR_EACH_MN_PANEL"
    else:
        total = m_tasks * n_tasks
        semantics = "DISJOINT_MN_PARENT_TILES"
    cores = f["usedCoreNum"]
    high = ceil_div(total, cores)
    low = total // cores
    return {
        "semantics": semantics,
        "m_tasks": m_tasks,
        "n_tasks": n_tasks,
        "k_tasks": k_tasks,
        "scheduled_units": total,
        "used_cores": cores,
        "minimum_units_per_core": low,
        "maximum_units_per_core": high,
        "task_count_imbalance": high - low,
        "no_zero_owner_core": total >= cores,
    }


def _schedule_facts(shape: Shape, state: dict, suffix: int) -> dict:
    f = state["fields"]
    m_tiles = ceil_div(shape.m, f["baseM"])
    n_tiles = ceil_div(shape.n, f["baseN"])
    k_tiles = ceil_div(shape.k, f["baseK"])
    padded_m = (m_tiles - 1) * f["baseM"] + align_up(
        shape.m - (m_tiles - 1) * f["baseM"], 16
    )
    padded_n = (n_tiles - 1) * f["baseN"] + align_up(
        shape.n - (n_tiles - 1) * f["baseN"], 16
    )
    padded_k = (k_tiles - 1) * f["baseK"] + align_up(
        shape.k - (k_tiles - 1) * f["baseK"], 16
    )
    logical = shape.m * shape.n * shape.k
    padded = padded_m * padded_n * padded_k
    return {
        "base_tile_count_mnk": [m_tiles, n_tiles, k_tiles],
        "logical_fma": logical,
        "edge_aligned_fma_upper_bound": padded,
        "edge_padding_fma": padded - logical,
        "edge_padding_ppm": (padded - logical) * 1_000_000 // logical,
        "ownership": _ownership(shape, state, suffix),
        "l0_l1_bytes": state["resource_bytes"],
    }


def _fnv1a64(blob: Iterable[int]) -> str:
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def generate(
    m: int,
    k: int,
    n: int,
    dtype: str = "fp16",
    trans_a: bool = False,
    trans_b: bool = False,
    *,
    hardware: Hardware | None = None,
) -> dict:
    hardware = Hardware() if hardware is None else hardware
    shape = Shape(
        m=int(m), n=int(n), k=int(k), dtype=str(dtype),
        trans_a=bool(trans_a), trans_b=bool(trans_b),
    )
    state = solve(shape, hardware, _compile_info(shape, hardware))
    suffix, key = _kernel_suffix(state, shape.dtype)
    blob, packet = _packet(shape, state, suffix, hardware)
    workspace, workspace_terms = _workspace_bytes(shape, state, suffix)
    schedule = _schedule_facts(shape, state, suffix)
    return {
        "status": "INDEPENDENT_COMPLETE_FORMULA_TILING",
        "request": {
            "M": shape.m, "N": shape.n, "K": shape.k,
            "dtype": shape.dtype,
            "transA": shape.trans_a, "transB": shape.trans_b,
        },
        "formula_family": state["family"],
        "kernel_suffix": suffix,
        "tiling_key": 10_000_000_000_000_000_000 + suffix,
        "key_derivation": key,
        "block_dim": state["fields"]["usedCoreNum"],
        "workspace_bytes": workspace,
        "workspace_terms": workspace_terms,
        "packet_layout": "legacy272_cube200",
        "packet_bytes": len(blob),
        "packet_hex": blob.hex(),
        "packet_sha256": hashlib.sha256(blob).hexdigest(),
        "packet_fnv1a64": _fnv1a64(blob),
        "tiling_fields": state["fields"],
        "l2_fields": state["l2"],
        "conversion_a": state["conversion_a"],
        "conversion_b": state["conversion_b"],
        "abi_fields": packet,
        "derivation": state,
        "schedule_facts": schedule,
        "selection_contract": {
            "complete_tilings_constructed": 1,
            "candidate_enumeration": False,
            "candidate_ranking": False,
            "latency_score": False,
            "cost_model": False,
            "official_selector": False,
            "official_packet_seed": False,
            "history_lookup": False,
            "repository_lookup": False,
            "decision": "ordered_family_predicates_and_closed_form_integer_equations",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        generate(args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b),
        sort_keys=True, indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ))


if __name__ == "__main__":
    main()
