#!/usr/bin/env python3
"""Closed-form tiler for the repository-owned C220 MatMul kernel.

The selector consumes only shape, dtype, transpose flags, and fixed 910B3
capacity constants.  It never calls the installed MatMulV3 selector, a tiling
repository, measurement history, a ranker, or a candidate search.
"""
from __future__ import annotations

import hashlib
import json
import struct


# Private packet schema used by direct_matmul/mat_mul_v3_tiling_data_280.h.
# It is repeated here deliberately: the production selector must not import
# or execute any reconstructed/installed MatMulV3 selector path.
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


def pack_packet(packet: dict) -> bytes:
    cube = packet["cube_words"]
    if len(cube) != 50 or any(type(word) is not int or not 0 <= word < 2**32 for word in cube):
        raise ValueError("packet requires exactly fifty uint32 cube words")
    blob = bytearray(280)
    struct.pack_into("<50I", blob, 0, *cube)
    l2 = packet["tileL2cacheTiling"]
    struct.pack_into("<5I", blob, 200, *(l2[name] for name in (
        "mTileCntL2", "nTileCntL2", "mTileBlock", "nTileBlock", "calOrder",
    )))
    run = packet["matmulRunInfo"]
    struct.pack_into("<7I", blob, 224, *(run[name] for name in (
        "transA", "transB", "nd2nzA", "nd2nzB", "isNzA", "isNzB", "isHf32",
    )))
    struct.pack_into("<I", blob, 256, packet["l2CacheFlag"])
    vector = packet["vector"]
    struct.pack_into("<4I", blob, 264, *(vector[name] for name in (
        "baseAN", "baseAD", "baseBN", "baseBD",
    )))
    for offset in (220, 252, 260):
        padding = bytes.fromhex(packet["padding_hex"][str(offset)])
        if len(padding) != 4:
            raise ValueError("every explicit padding region must be four bytes")
        blob[offset:offset + 4] = padding
    return bytes(blob)


def unpack_packet(blob: bytes) -> dict:
    if len(blob) != 280:
        raise ValueError("independent packet must be 280 bytes")
    l2_names = ("mTileCntL2", "nTileCntL2", "mTileBlock", "nTileBlock", "calOrder")
    run_names = ("transA", "transB", "nd2nzA", "nd2nzB", "isNzA", "isNzB", "isHf32")
    vector_names = ("baseAN", "baseAD", "baseBN", "baseBD")
    return {
        "cube_words": list(struct.unpack_from("<50I", blob, 0)),
        "tileL2cacheTiling": dict(zip(l2_names, struct.unpack_from("<5I", blob, 200))),
        "matmulRunInfo": dict(zip(run_names, struct.unpack_from("<7I", blob, 224))),
        "l2CacheFlag": struct.unpack_from("<I", blob, 256)[0],
        "vector": dict(zip(vector_names, struct.unpack_from("<4I", blob, 264))),
        "padding_hex": {str(offset): blob[offset:offset + 4].hex() for offset in (220, 252, 260)},
    }


AIC = 20
L0A = 65_536
L0B = 65_536
L0C = 131_072
L1 = 524_032
SYSTEM_WORKSPACE = 20 * 1024 * 1024
KERNEL_SUFFIX = 91_000

FAMILY_MODE = {
    "MICRO_DIRECT": 11,
    "BALANCED_MN": 12,
    "SEEDED_SPLIT_K": 13,
    "RESIDENT_B_M_STRIPE": 14,
    "RESIDENT_A_N_STRIPE": 15,
    "SEEDED_TAIL_WAVE": 16,
}


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value>=0 and divisor>0")
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def request(
    m: int, k: int, n: int, dtype: str,
    trans_a: bool, trans_b: bool,
) -> dict:
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unsupported dtype={dtype}")
    if min(m, n, k) <= 0 or max(m, n, k) >= 2**31:
        raise ValueError("M/N/K must be positive signed-32-bit dimensions")
    return {
        "M": int(m), "N": int(n), "K": int(k), "dtype": dtype,
        "transA": bool(trans_a), "transB": bool(trans_b),
    }


def dtype_bytes(req: dict) -> int:
    return 4 if req["dtype"] == "fp32" else 2


def base_grid(req: dict) -> tuple[int, int]:
    """One formula projection, not an enumeration of tile candidates."""
    m = min(128, align_up(req["M"], 16))
    n = min(128, align_up(req["N"], 16))
    # L0C is FP32.  Shrink only the longer axis until double-buffered output
    # closes; the loop is a capacity projection, not performance ranking.
    while m * n * 4 * 2 > L0C:
        if m >= n and m > 16:
            m //= 2
        elif n > 16:
            n //= 2
        else:
            raise ValueError("no legal L0C projection")
    return m, n


def base_k(req: dict, base_m: int, base_n: int) -> int:
    width = dtype_bytes(req)
    maximum = min(
        128 if width == 2 else 64,
        L0A // (base_m * width * 2),
        L0B // (base_n * width * 2),
    )
    projected = align_down(maximum, 16)
    if projected < 16:
        raise ValueError("no legal L0A/L0B K projection")
    return projected


def family_for(req: dict) -> tuple[str, dict]:
    """Disjoint decision tree derived from ownership bottlenecks.

    Precedence is structural rather than a latency score: exploit K ownership
    only when MN has fewer than two output tiles; repair an incomplete final
    wave only when at least two K owners fit; then use exact resident-operand
    capacity predicates for extreme aspect ratios; finally distinguish a
    single-output micro problem from the balanced general path.
    """
    m, n, k = req["M"], req["N"], req["K"]
    width = dtype_bytes(req)
    m_tiles = ceil_div(m, 128)
    n_tiles = ceil_div(n, 128)
    output_tiles = m_tiles * n_tiles
    k_quantum = 64 if width == 4 else 128
    k_tiles = ceil_div(k, k_quantum)
    minimum_k_per_owner = 384 if width == 4 else 512
    maximum_k_owners = min(AIC, k // minimum_k_per_owner, k_tiles)

    # Whole-output Split-K is selected only when all AICs receive a nonempty
    # interval at least as large as the dtype-specific Cube pipeline quantum.
    # This replaces a tuned K threshold with a full-occupancy proof.
    if output_tiles == 1 and maximum_k_owners == AIC:
        return "SEEDED_SPLIT_K", {
            "decision_inequality": (
                "ceil(M/128)*ceil(N/128)=1 and "
                "floor(K/minimum_k_per_owner)>=20"
            ),
            "minimum_k_per_owner": minimum_k_per_owner,
            "proven_nonempty_owners": maximum_k_owners,
            "bottleneck": "MN_PARALLELISM_BELOW_AIC_COUNT",
        }

    tail = output_tiles % AIC
    # tail<=AIC/2 guarantees at least two K owners for every tail output.
    # The second inequality guarantees every owner enough K to amortize one
    # Cube pipeline, without an empirical latency or score constant.
    if output_tiles > AIC and 1 <= tail <= AIC // 2:
        largest_group = ceil_div(AIC, tail)
        if k >= largest_group * minimum_k_per_owner:
            return "SEEDED_TAIL_WAVE", {
                "decision_inequality": (
                    "Tmn>20, 1<=Tmn mod 20<=10, "
                    "K>=ceil(20/tail)*minimum_k_per_owner"
                ),
                "minimum_k_per_owner": minimum_k_per_owner,
                "largest_owner_group": largest_group,
                "bottleneck": "UNDERFILLED_FINAL_MN_WAVE",
            }

    resident_b_bytes = align_up(k, 16) * align_up(n, 16) * width
    if m_tiles >= AIC and n_tiles <= 2 and resident_b_bytes <= L1 // 2:
        return "RESIDENT_B_M_STRIPE", {
            "decision_inequality": (
                "ceil(M/128)>=20, ceil(N/128)<=2, "
                "aligned(K)*aligned(N)*d<=L1/2"
            ),
            "resident_operand_bytes": resident_b_bytes,
            "bottleneck": "B_RELOAD_ACROSS_LONG_M",
        }

    resident_a_bytes = align_up(m, 16) * align_up(k, 16) * width
    if n_tiles >= AIC and m_tiles <= 2 and resident_a_bytes <= L1 // 2:
        return "RESIDENT_A_N_STRIPE", {
            "decision_inequality": (
                "ceil(N/128)>=20, ceil(M/128)<=2, "
                "aligned(M)*aligned(K)*d<=L1/2"
            ),
            "resident_operand_bytes": resident_a_bytes,
            "bottleneck": "A_RELOAD_ACROSS_LONG_N",
        }

    if output_tiles == 1:
        return "MICRO_DIRECT", {
            "decision_inequality": (
                "ceil(M/128)*ceil(N/128)=1 and fewer than 20 "
                "minimum-size K owners"
            ),
            "bottleneck": "LAUNCH_AND_PADDING_DOMINATED",
        }

    return "BALANCED_MN", {
        "decision_inequality": "all preceding structural predicates are false",
        "bottleneck": "GENERAL_MN_CRITICAL_PATH",
    }


def empty_cube(req: dict) -> dict:
    cube = {name: 0 for name in CUBE_FIELDS}
    cube.update({
        "M": req["M"], "N": req["N"], "Ka": req["K"], "Kb": req["K"],
        "batchM": 1, "batchN": 1, "singleBatchM": 1,
        "singleBatchN": 1, "BatchNum": 1,
    })
    return cube


def direct_geometry(req: dict, family: str) -> tuple[int, int, int]:
    if family == "MICRO_DIRECT":
        return req["M"], req["N"], 1
    if family == "RESIDENT_B_M_STRIPE":
        target = min(AIC, ceil_div(req["M"], 128))
        single_m = align_up(ceil_div(req["M"], target), 16)
        used = ceil_div(req["M"], single_m)
        return single_m, req["N"], used
    if family == "RESIDENT_A_N_STRIPE":
        target = min(AIC, ceil_div(req["N"], 128))
        single_n = align_up(ceil_div(req["N"], target), 16)
        used = ceil_div(req["N"], single_n)
        return req["M"], single_n, used
    single_m = 128
    single_n = 128
    used = min(AIC, ceil_div(req["M"], single_m) * ceil_div(req["N"], single_n))
    return single_m, single_n, used


def build_cube(req: dict, family: str) -> tuple[dict, dict]:
    bm, bn = base_grid(req)
    bk = base_k(req, bm, bn)
    mode = FAMILY_MODE[family]

    if family == "SEEDED_SPLIT_K":
        minimum_k = 384 if req["dtype"] == "fp32" else 512
        maximum_owners = min(AIC, req["K"] // minimum_k, ceil_div(req["K"], bk))
        if maximum_owners < 2:
            raise ValueError("seeded Split-K requires at least two nonempty K owners")
        maximum_k_tiles = ceil_div(ceil_div(req["K"], bk), maximum_owners)
        used = ceil_div(ceil_div(req["K"], bk), maximum_k_tiles)
        single_m, single_n = req["M"], req["N"]
        single_k = maximum_k_tiles * bk
        derivation = {
            "ownership": "DISJOINT_K_INTERVAL_PER_AIC",
            "owners": used, "k_quantum": bk,
            "maximum_k_tiles_per_owner": maximum_k_tiles,
            "output_protocol": "RANK0_DIRECT_GLOBAL_BARRIER_REMAINDER_ATOMIC",
            "removed_operations_vs_clear_all_atomic": {
                "full_output_clear_bytes": req["M"] * req["N"] * dtype_bytes(req),
                "atomic_updates_per_output_element": 1,
            },
        }
    elif family == "SEEDED_TAIL_WAVE":
        single_m, single_n, used = 128, 128, AIC
        single_k = req["K"]
        total = ceil_div(req["M"], 128) * ceil_div(req["N"], 128)
        full_rounds, tail = divmod(total, AIC)
        small_group = AIC // tail
        old_path = (full_rounds + 1) * ceil_div(req["K"], bk)
        new_path = (
            full_rounds * ceil_div(req["K"], bk) +
            ceil_div(ceil_div(req["K"], bk), small_group)
        )
        derivation = {
            "ownership": "FULL_MN_WAVES_PLUS_FINAL_WAVE_DISJOINT_K",
            "full_rounds": full_rounds, "tail_tiles": tail,
            "small_owner_group": small_group,
            "large_owner_group": ceil_div(AIC, tail),
            "same_grid_direct_critical_k_tiles": old_path,
            "new_critical_k_tiles": new_path,
            "critical_path_reduction_ppm": (old_path - new_path) * 1_000_000 // old_path,
            "output_protocol": "TAIL_RANK0_DIRECT_GLOBAL_BARRIER_REMAINDER_ATOMIC",
        }
    else:
        single_m, single_n, used = direct_geometry(req, family)
        single_k = req["K"]
        mt = ceil_div(req["M"], single_m)
        nt = ceil_div(req["N"], single_n)
        tasks = mt * nt
        derivation = {
            "ownership": "DISJOINT_MN_TILE_INTERVAL_PER_AIC",
            "output_tiles": tasks,
            "maximum_tiles_per_aic": ceil_div(tasks, used),
            "minimum_tiles_per_aic": tasks // used,
            "task_imbalance_bound": ceil_div(tasks, used) - tasks // used,
            "traversal": "CONTIGUOUS_BALANCED_SERPENTINE_M_FAST",
        }
        if family == "RESIDENT_B_M_STRIPE":
            derivation["loop_invariant"] = "ONE_N_STRIPE_AND_FULL_K_PER_AIC_WHILE_M_IS_INTERNAL"
        elif family == "RESIDENT_A_N_STRIPE":
            derivation["loop_invariant"] = "ONE_M_STRIPE_AND_FULL_K_PER_AIC_WHILE_N_IS_INTERNAL"

    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": single_m, "singleCoreN": single_n,
        "singleCoreK": single_k,
        "baseM": bm, "baseN": bn, "baseK": bk,
        "depthA1": 2, "depthB1": 2,
        "stepM": 1, "stepN": 1, "stepKa": 1, "stepKb": 1,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
        "iterateOrder": 0, "reserved": mode,
    })
    return cube, derivation


def validate_cube(req: dict, cube: dict, family: str) -> dict:
    width = dtype_bytes(req)
    checks = {
        "all_words_uint32": all(
            type(cube[name]) is int and 0 <= cube[name] < 2**32
            for name in CUBE_FIELDS
        ),
        "mode_exact": cube["reserved"] == FAMILY_MODE[family],
        "core_bound": 1 <= cube["usedCoreNum"] <= AIC,
        "base_alignment": all(
            cube[name] % 16 == 0 for name in ("baseM", "baseN", "baseK")
        ),
        "l0a_fit": cube["baseM"] * cube["baseK"] * width * cube["dbL0A"] <= L0A,
        "l0b_fit": cube["baseN"] * cube["baseK"] * width * cube["dbL0B"] <= L0B,
        "l0c_fit": cube["baseM"] * cube["baseN"] * 4 * cube["dbL0C"] <= L0C,
        "l1_fit": (
            (cube["baseM"] * cube["baseK"] * cube["depthA1"] +
             cube["baseN"] * cube["baseK"] * cube["depthB1"]) * width <= L1
        ),
    }
    if family in ("SEEDED_SPLIT_K", "SEEDED_TAIL_WAVE"):
        checks["atomic_dtype_supported_on_910b"] = req["dtype"] in ("fp16", "bf16", "fp32")
    if not all(checks.values()):
        raise ValueError(f"illegal independent tiling: {checks}")
    return checks


def payload(req: dict, cube: dict) -> dict:
    return {
        "cube_words": [cube[name] for name in CUBE_FIELDS],
        "tileL2cacheTiling": {
            "mTileCntL2": 1, "nTileCntL2": 1,
            "mTileBlock": 0, "nTileBlock": 0, "calOrder": 0,
        },
        "matmulRunInfo": {
            "transA": int(req["transA"]), "transB": int(req["transB"]),
            "nd2nzA": 0, "nd2nzB": 0, "isNzA": 0, "isNzB": 0,
            "isHf32": 0,
        },
        "l2CacheFlag": 0,
        "vector": {"baseAN": 16, "baseAD": 8, "baseBN": 16, "baseBD": 8},
        "padding_hex": {"220": "00000000", "252": "00000000", "260": "00000000"},
    }


def generate(
    m: int, k: int, n: int, dtype: str = "fp16",
    trans_a: bool = False, trans_b: bool = False,
) -> dict:
    req = request(m, k, n, dtype, trans_a, trans_b)
    family, decision = family_for(req)
    cube, derivation = build_cube(req, family)
    checks = validate_cube(req, cube, family)
    packet = payload(req, cube)
    blob = pack_packet(packet)
    if len(blob) != 280 or unpack_packet(blob) != packet:
        raise RuntimeError("independent 280-byte packet roundtrip mismatch")
    return {
        "status": "INDEPENDENT_EXECUTABLE_TILING",
        "origin": "REPOSITORY_OWNED_KERNEL_AND_SELECTOR",
        "request": req,
        "formula_family": family,
        "kernel_suffix": KERNEL_SUFFIX,
        "kernel_mode": FAMILY_MODE[family],
        "selection_basis": "DISJOINT_CLOSED_FORM_OWNERSHIP_TREE",
        "decision": decision,
        "cube": cube,
        "derivation": derivation,
        "resource_checks": checks,
        "packet_layout": "target280_cube200",
        "packet_bytes": len(blob),
        "packet_hex": blob.hex(),
        "packet_sha256": hashlib.sha256(blob).hexdigest(),
        "block_dim": cube["usedCoreNum"],
        "workspace_bytes": SYSTEM_WORKSPACE,
        "runtime_dependencies": {
            "installed_selector": False,
            "official_tiling_seed": False,
            "cost_model": False,
            "candidate_enumeration": False,
            "candidate_ranking": False,
            "history": False,
            "repository_lookup": False,
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(
        args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b,
    ), sort_keys=True, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
