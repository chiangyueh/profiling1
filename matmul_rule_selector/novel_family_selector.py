#!/usr/bin/env python3
"""Closed-form tiler for repository-owned C220 execution families."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "baseline_core"))

from matmul_reconstruction._core.abi import pack_packet, unpack_packet  # noqa: E402
from matmul_reconstruction._core.initializer import CUBE_FIELDS  # noqa: E402


AIC = 20
L0A = 65536
L0B = 65536
L0C = 131072
L1 = 524032
SYSTEM_WORKSPACE = 20 * 1024 * 1024
CUSTOM_SUFFIX = {
    "SEEDED_ATOMIC_SPLIT_K": 90001,
    "SEEDED_TAIL_WAVE_SPLIT_K": 90002,
}
CUSTOM_MODE = {
    "SEEDED_ATOMIC_SPLIT_K": 1,
    "SEEDED_TAIL_WAVE_SPLIT_K": 2,
}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def request(m: int, k: int, n: int, dtype: str,
            trans_a: bool, trans_b: bool) -> dict:
    if dtype != "fp32":
        raise ValueError("novel seeded families currently require fp32 accumulation and output")
    if min(m, n, k) <= 0 or max(m, n, k) >= 2**32:
        raise ValueError("M/N/K must be positive uint32 dimensions")
    return {
        "M": int(m), "N": int(n), "K": int(k), "dtype": dtype,
        "transA": bool(trans_a), "transB": bool(trans_b),
    }


def applicable(req: dict, family: str) -> tuple[bool, str]:
    m, n, k = req["M"], req["N"], req["K"]
    layout = not req["transA"] and req["transB"]
    if family == "SEEDED_ATOMIC_SPLIT_K":
        ok = layout and m <= 128 and n <= 128 and 8192 <= k <= 60000
        return ok, "one output tile, at least twenty 384-element K ownership stripes"
    if family == "SEEDED_TAIL_WAVE_SPLIT_K":
        tiles = ceil_div(m, 128) * ceil_div(n, 128)
        tail = tiles % AIC
        largest_group = ceil_div(AIC, tail) if tail else 0
        k_tiles = ceil_div(k, 64)
        ok = (
            layout and 20 < tiles <= 320 and 1 <= tail <= 10 and
            4096 <= k <= 60000 and k_tiles >= largest_group
        )
        return ok, (
            "complete MN waves use disjoint direct stores; the underfilled final "
            "wave has at least two K owners per output tile"
        )
    raise ValueError(f"unknown family={family}")


def select_family(req: dict, required_family: str | None = None) -> tuple[str, str]:
    matches = []
    for family in CUSTOM_SUFFIX:
        ok, reason = applicable(req, family)
        if ok:
            matches.append((family, reason))
    if required_family is not None:
        required = [entry for entry in matches if entry[0] == required_family]
        if len(required) != 1:
            raise ValueError(f"shape is not applicable to required family={required_family}")
        return required[0]
    if len(matches) != 1:
        raise ValueError(f"novel family rule must have exactly one match, got {matches}")
    return matches[0]


def empty_cube(req: dict) -> dict:
    cube = {name: 0 for name in CUBE_FIELDS}
    cube.update({
        "M": req["M"], "N": req["N"], "Ka": req["K"], "Kb": req["K"],
        "batchM": 1, "batchN": 1, "singleBatchM": 1,
        "singleBatchN": 1, "BatchNum": 1,
    })
    return cube


def common_cube(req: dict, *, used: int, single_m: int,
                single_n: int, single_k: int, mode: int) -> dict:
    base_m = min(128, align_up(min(req["M"], single_m), 16))
    base_n = min(128, align_up(min(req["N"], single_n), 16))
    base_k = 64
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": single_m, "singleCoreN": single_n,
        "singleCoreK": single_k,
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": 2, "depthB1": 2,
        "stepM": 1, "stepN": 1, "stepKa": 1, "stepKb": 1,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
        "iterateOrder": 0,
        "reserved": mode,
    })
    return cube


def seeded_atomic_cube(req: dict) -> tuple[dict, dict]:
    maximum_owners = min(AIC, req["K"] // 384)
    if maximum_owners < 2:
        raise ValueError("seeded atomic ownership needs at least two K owners")
    k_tiles = ceil_div(req["K"], 64)
    target_owners = min(maximum_owners, k_tiles)
    maximum_k_tiles = ceil_div(k_tiles, target_owners)
    used = ceil_div(k_tiles, maximum_k_tiles)
    cube = common_cube(
        req, used=used, single_m=req["M"], single_n=req["N"],
        single_k=maximum_k_tiles * 64, mode=CUSTOM_MODE["SEEDED_ATOMIC_SPLIT_K"],
    )
    output_bytes = req["M"] * req["N"] * 4
    return cube, {
        "ownership": "ALL_OWNERS_SHARE_MN_AND_OWN_DISJOINT_K_INTERVALS",
        "k_owners": used,
        "k_quantum": 64,
        "maximum_k_tiles_per_owner": maximum_k_tiles,
        "output_protocol": "OWNER_0_DIRECT_THEN_GLOBAL_BARRIER_THEN_OWNERS_1_TO_P_MINUS_1_ATOMIC",
        "official_multicore_protocol_bytes_removed": output_bytes,
        "atomic_output_transactions_removed_per_element": 1,
        "proof": (
            "each K quantum belongs to exactly one owner by floor(Q*i/P) boundaries; "
            "owner 0 initializes every C element before any atomic owner passes SyncAll"
        ),
    }


def seeded_tail_cube(req: dict) -> tuple[dict, dict]:
    tile_m = 128
    tile_n = 128
    m_tiles = ceil_div(req["M"], tile_m)
    n_tiles = ceil_div(req["N"], tile_n)
    total_tiles = m_tiles * n_tiles
    full_rounds = total_tiles // AIC
    tail_tiles = total_tiles % AIC
    small_group = AIC // tail_tiles
    large_group = ceil_div(AIC, tail_tiles)
    k_tiles = ceil_div(req["K"], 64)
    if small_group < 2 or k_tiles < large_group:
        raise ValueError("tail wave cannot assign nonempty K intervals to every owner")
    cube = common_cube(
        req, used=AIC, single_m=tile_m, single_n=tile_n,
        single_k=req["K"], mode=CUSTOM_MODE["SEEDED_TAIL_WAVE_SPLIT_K"],
    )
    baseline_critical_k_tiles = (full_rounds + 1) * k_tiles
    candidate_critical_k_tiles = full_rounds * k_tiles + ceil_div(k_tiles, small_group)
    return cube, {
        "ownership": "FULL_MN_WAVES_DIRECT_PLUS_ONLY_FINAL_WAVE_SPLIT_K",
        "m_tiles": m_tiles, "n_tiles": n_tiles,
        "full_mn_rounds": full_rounds, "tail_output_tiles": tail_tiles,
        "small_k_owner_group": small_group,
        "large_k_owner_group": large_group,
        "large_groups": AIC % tail_tiles,
        "k_quantum": 64,
        "output_protocol": "EACH_TAIL_GROUP_OWNER_0_DIRECT_THEN_BARRIER_THEN_REMAINING_OWNERS_ATOMIC",
        "same_grid_dp_critical_k_tiles": baseline_critical_k_tiles,
        "seeded_tail_critical_k_tiles": candidate_critical_k_tiles,
        "cube_work_critical_path_reduction_ppm": (
            (baseline_critical_k_tiles - candidate_critical_k_tiles) * 1_000_000
            // baseline_critical_k_tiles
        ),
        "workspace_reducer_bytes": 0,
        "proof": (
            "the first floor(T/20)*20 output tiles are disjoint; all 20 owners "
            "then partition each remaining tile's K quanta exactly once; direct seed "
            "happens before atomic owners pass SyncAll"
        ),
    }


def validate_cube(cube: dict, family: str) -> dict:
    checks = {
        "all_words_uint32": all(
            type(cube[name]) is int and 0 <= cube[name] < 2**32
            for name in CUBE_FIELDS
        ),
        "custom_mode": cube["reserved"] == CUSTOM_MODE[family],
        "core_bound": 2 <= cube["usedCoreNum"] <= AIC,
        "base_alignment": (
            cube["baseM"] % 16 == 0 and cube["baseN"] % 16 == 0 and
            cube["baseK"] % 16 == 0
        ),
        "l0a_fit": cube["baseM"] * cube["baseK"] * 4 * cube["dbL0A"] <= L0A,
        "l0b_fit": cube["baseN"] * cube["baseK"] * 4 * cube["dbL0B"] <= L0B,
        "l0c_fit": cube["baseM"] * cube["baseN"] * 4 * cube["dbL0C"] <= L0C,
        "l1_fit": (
            (cube["baseM"] * cube["baseK"] * cube["depthA1"] +
             cube["baseN"] * cube["baseK"] * cube["depthB1"]) * 4 <= L1
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"illegal novel candidate: {checks}")
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


def generate(m: int, k: int, n: int, dtype: str,
             trans_a: bool, trans_b: bool,
             *, required_family: str | None = None) -> dict:
    req = request(m, k, n, dtype, trans_a, trans_b)
    family, reason = select_family(req, required_family)
    if family == "SEEDED_ATOMIC_SPLIT_K":
        cube, derivation = seeded_atomic_cube(req)
    else:
        cube, derivation = seeded_tail_cube(req)
    checks = validate_cube(cube, family)
    packet = payload(req, cube)
    blob = pack_packet(packet, "target280_cube200")
    if len(blob) != 280 or unpack_packet(blob, "target280_cube200") != packet:
        raise RuntimeError("custom 280-byte packet roundtrip mismatch")
    return {
        "status": "NOVEL_EXECUTABLE_FAMILY_TILING",
        "origin": "REPOSITORY_OWNED_NOT_OFFICIAL_BACKPORT",
        "npu_eligible": True,
        "formula_family": family,
        "kernel_suffix": CUSTOM_SUFFIX[family],
        "selection_basis": "CLOSED_FORM_HARDWARE_OWNERSHIP_RULE",
        "applicability_reason": reason,
        "packet_layout": "target280_cube200",
        "packet_bytes": len(blob),
        "packet_hex": blob.hex(),
        "packet_sha256": hashlib.sha256(blob).hexdigest(),
        "block_dim": cube["usedCoreNum"],
        "workspace_bytes": SYSTEM_WORKSPACE,
        "cube": cube,
        "resource_checks": checks,
        "derivation": derivation,
        "runtime_dependencies": {
            "official_selector": False, "official_tiling_seed": False,
            "cost_model": False, "history": False, "repo_lookup": False,
            "candidate_search": False,
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--family")
    args = parser.parse_args()
    print(json.dumps(generate(
        args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b,
        required_family=args.family,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
