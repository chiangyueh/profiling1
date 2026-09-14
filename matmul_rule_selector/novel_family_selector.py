#!/usr/bin/env python3
"""Closed-form tiler for repository-owned C220 execution families."""
from __future__ import annotations

import hashlib
import json
import struct


# The custom path owns its complete 280-byte ABI.  Keeping the schema here is
# intentional: no official/reconstructed selector or initializer is imported.
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
L2_FIELDS = ("mTileCntL2", "nTileCntL2", "mTileBlock", "nTileBlock", "calOrder")
RUN_FIELDS = ("transA", "transB", "nd2nzA", "nd2nzB", "isNzA", "isNzB", "isHf32")
VECTOR_FIELDS = ("baseAN", "baseAD", "baseBN", "baseBD")


def pack_packet(payload: dict) -> bytes:
    required = {
        "cube_words", "tileL2cacheTiling", "matmulRunInfo",
        "l2CacheFlag", "vector", "padding_hex",
    }
    if set(payload) != required or len(payload["cube_words"]) != 50:
        raise ValueError("custom ABI requires the complete explicit payload")
    words = list(payload["cube_words"])
    words.extend(payload["tileL2cacheTiling"][name] for name in L2_FIELDS)
    words.append(0)
    words.extend(payload["matmulRunInfo"][name] for name in RUN_FIELDS)
    words.append(0)
    words.append(payload["l2CacheFlag"])
    words.append(0)
    words.extend(payload["vector"][name] for name in VECTOR_FIELDS)
    if len(words) != 70 or any(type(value) is not int or not 0 <= value < 2**32 for value in words):
        raise ValueError("custom ABI contains a non-uint32 word")
    return struct.pack("<70I", *words)


def unpack_packet(blob: bytes) -> dict:
    if len(blob) != 280:
        raise ValueError("custom packet must be 280 bytes")
    words = struct.unpack("<70I", blob)
    return {
        "cube_words": list(words[:50]),
        "tileL2cacheTiling": dict(zip(L2_FIELDS, words[50:55])),
        "matmulRunInfo": dict(zip(RUN_FIELDS, words[56:63])),
        "l2CacheFlag": words[64],
        "vector": dict(zip(VECTOR_FIELDS, words[66:70])),
        "padding_hex": {"220": blob[220:224].hex(), "252": blob[252:256].hex(), "260": blob[260:264].hex()},
    }


AIC = 20
L0A = 65536
L0B = 65536
L0C = 131072
L1 = 524032
SYSTEM_WORKSPACE = 20 * 1024 * 1024
CUSTOM_SUFFIX = {
    "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K": 90001,
    "DIRECT_INIT_TAIL_WAVE_SPLIT_K": 90002,
}
CUSTOM_MODE = {
    "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K": 1,
    "DIRECT_INIT_TAIL_WAVE_SPLIT_K": 2,
}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def request(m: int, k: int, n: int, dtype: str,
            trans_a: bool, trans_b: bool) -> dict:
    if dtype != "fp32":
        raise ValueError("direct-init Split-K families currently require fp32 accumulation and output")
    if min(m, n, k) <= 0 or max(m, n, k) >= 2**32:
        raise ValueError("M/N/K must be positive uint32 dimensions")
    return {
        "M": int(m), "N": int(n), "K": int(k), "dtype": dtype,
        "transA": bool(trans_a), "transB": bool(trans_b),
    }


def applicable(req: dict, family: str) -> tuple[bool, str]:
    m, n, k = req["M"], req["N"], req["K"]
    layout = not req["transA"] and req["transB"]
    if family == "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K":
        if not (layout and m <= 128 and n <= 128):
            return False, "requires FP32 NT with exactly one 128x128 output tile"
        pipeline = derive_inner_pipeline(
            req, single_m=m, single_n=n, maximum_task_k=k
        )
        k_blocks = ceil_div(k, pipeline["base_k"])
        ok = k_blocks // 3 >= 2
        return ok, "one output tile and at least three capacity-derived baseK blocks per K owner"
    if family == "DIRECT_INIT_TAIL_WAVE_SPLIT_K":
        tiles = ceil_div(m, 128) * ceil_div(n, 128)
        tail = tiles % AIC
        largest_group = ceil_div(AIC, tail) if tail else 0
        if not (layout and tiles > AIC and 1 <= tail <= AIC // 2):
            return False, "requires an incomplete final MN wave with at least two AICs per tail tile"
        pipeline = derive_inner_pipeline(
            req, single_m=128, single_n=128, maximum_task_k=k
        )
        k_tiles = ceil_div(k, pipeline["base_k"])
        ok = k_tiles >= largest_group
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


def derive_inner_pipeline(
    req: dict, *, single_m: int, single_n: int, maximum_task_k: int,
) -> dict:
    """Solve the complete inner Matmul tile for the custom scheduler.

    The output tile is first projected to the 16-element Cube lattice.  The K
    base then saturates the tighter double-buffered L0A/L0B capacity.  Finally
    stepK is the largest balanced K group whose two A/B L1 buffers fit.  These
    are successive projections of one tiling, not a candidate set.
    """
    width = 4
    base_m = min(128, align_up(min(req["M"], single_m), 16))
    base_n = min(128, align_up(min(req["N"], single_n), 16))
    base_k_cap = min(
        align_up(maximum_task_k, 16),
        L0A // (2 * base_m * width),
        L0B // (2 * base_n * width),
    )
    base_k = align_down(base_k_cap, 16)
    if base_k < 16:
        raise ValueError("custom family has no double-buffered L0A/L0B baseK")
    k_blocks = ceil_div(maximum_task_k, base_k)
    step_cap = L1 // (2 * (base_m + base_n) * base_k * width)
    if step_cap < 1:
        raise ValueError("custom family has no double-buffered L1 K step")
    k_groups = ceil_div(k_blocks, step_cap)
    step_k = ceil_div(k_blocks, k_groups)
    depth_a = depth_b = 2 * step_k
    db_l0c = 2 if 2 * base_m * base_n * 4 <= L0C else 1
    return {
        "base_m": base_m, "base_n": base_n, "base_k": base_k,
        "step_k": step_k, "depth_a": depth_a, "depth_b": depth_b,
        "db_l0c": db_l0c, "maximum_task_k": maximum_task_k,
        "k_blocks_per_largest_task": k_blocks,
        "l1_step_capacity": step_cap, "k_pipeline_groups": k_groups,
        "l0a_bytes": 2 * base_m * base_k * width,
        "l0b_bytes": 2 * base_n * base_k * width,
        "l0c_bytes": db_l0c * base_m * base_n * 4,
        "l1_bytes": (depth_a * base_m + depth_b * base_n) * base_k * width,
    }


def common_cube(req: dict, *, used: int, single_m: int,
                single_n: int, single_k: int, mode: int,
                pipeline: dict) -> dict:
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": single_m, "singleCoreN": single_n,
        "singleCoreK": single_k,
        "baseM": pipeline["base_m"], "baseN": pipeline["base_n"],
        "baseK": pipeline["base_k"],
        "depthA1": pipeline["depth_a"], "depthB1": pipeline["depth_b"],
        "stepM": 1, "stepN": 1,
        "stepKa": pipeline["step_k"], "stepKb": pipeline["step_k"],
        "dbL0A": 2, "dbL0B": 2, "dbL0C": pipeline["db_l0c"],
        "iterateOrder": 0,
        "shareL1Size": pipeline["l1_bytes"],
        "shareL0CSize": pipeline["l0c_bytes"],
        "reserved": mode,
    })
    return cube


def direct_init_whole_output_cube(req: dict) -> tuple[dict, dict]:
    # First solve the one-output-tile L0 geometry.  Three base-K blocks per
    # owner are required to overlap two input buffers with Cube execution.
    provisional = derive_inner_pipeline(
        req, single_m=req["M"], single_n=req["N"], maximum_task_k=req["K"]
    )
    k_tiles = ceil_div(req["K"], provisional["base_k"])
    maximum_owners = min(AIC, k_tiles // 3)
    if maximum_owners < 2:
        raise ValueError("direct-init whole-output ownership needs at least two K owners")
    maximum_k_tiles = ceil_div(k_tiles, maximum_owners)
    used = maximum_owners
    maximum_task_k = min(req["K"], maximum_k_tiles * provisional["base_k"])
    pipeline = derive_inner_pipeline(
        req, single_m=req["M"], single_n=req["N"],
        maximum_task_k=maximum_task_k,
    )
    # The second projection cannot change baseK: maximum_task_k remains at
    # least three original base blocks and the same L0 geometry is used.
    if pipeline["base_k"] != provisional["base_k"]:
        raise ValueError("atomic K ownership changed the already solved L0 baseK")
    cube = common_cube(
        req, used=used, single_m=req["M"], single_n=req["N"],
        single_k=maximum_task_k, mode=CUSTOM_MODE["DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K"],
        pipeline=pipeline,
    )
    output_bytes = req["M"] * req["N"] * 4
    return cube, {
        "ownership": "ALL_OWNERS_SHARE_MN_AND_OWN_DISJOINT_K_INTERVALS",
        "k_owners": used,
        "k_quantum": pipeline["base_k"],
        "maximum_k_tiles_per_owner": maximum_k_tiles,
        "output_protocol": "OWNER_0_DIRECT_THEN_GLOBAL_BARRIER_THEN_OWNERS_1_TO_P_MINUS_1_ATOMIC",
        "official_multicore_protocol_bytes_removed": output_bytes,
        "atomic_output_transactions_removed_per_element": 1,
        "proof": (
            "each K quantum belongs to exactly one owner by floor(Q*i/P) boundaries; "
            "owner 0 initializes every C element before any atomic owner passes SyncAll"
        ),
        "inner_pipeline": pipeline,
    }


def direct_init_tail_wave_cube(req: dict) -> tuple[dict, dict]:
    tile_m = 128
    tile_n = 128
    m_tiles = ceil_div(req["M"], tile_m)
    n_tiles = ceil_div(req["N"], tile_n)
    total_tiles = m_tiles * n_tiles
    full_rounds = total_tiles // AIC
    tail_tiles = total_tiles % AIC
    small_group = AIC // tail_tiles
    large_group = ceil_div(AIC, tail_tiles)
    # 128x128 FP32 double-buffering consumes the complete 128 KiB L0C.  The
    # matching L0A/L0B intersection gives baseK=64; derive it rather than
    # storing either value as a tuning knob.
    provisional = derive_inner_pipeline(
        req, single_m=tile_m, single_n=tile_n, maximum_task_k=req["K"]
    )
    k_tiles = ceil_div(req["K"], provisional["base_k"])
    if small_group < 2 or k_tiles < large_group:
        raise ValueError("tail wave cannot assign nonempty K intervals to every owner")
    # Full waves compute complete K, so their task is the largest task and
    # determines the common Matmul pipeline.  Tail owners reuse that legal
    # pipeline for smaller contiguous K intervals.
    pipeline = provisional
    cube = common_cube(
        req, used=AIC, single_m=tile_m, single_n=tile_n,
        single_k=req["K"], mode=CUSTOM_MODE["DIRECT_INIT_TAIL_WAVE_SPLIT_K"],
        pipeline=pipeline,
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
        "k_quantum": pipeline["base_k"],
        "output_protocol": "EACH_TAIL_GROUP_OWNER_0_DIRECT_THEN_BARRIER_THEN_REMAINING_OWNERS_ATOMIC",
        "same_grid_dp_critical_k_tiles": baseline_critical_k_tiles,
        "direct_init_tail_critical_k_tiles": candidate_critical_k_tiles,
        "cube_work_critical_path_reduction_ppm": (
            (baseline_critical_k_tiles - candidate_critical_k_tiles) * 1_000_000
            // baseline_critical_k_tiles
        ),
        "workspace_reducer_bytes": 0,
        "proof": (
            "the first floor(T/20)*20 output tiles are disjoint; all 20 owners "
            "then partition each remaining tile's K quanta exactly once; direct initialization "
            "happens before atomic owners pass SyncAll"
        ),
        "inner_pipeline": pipeline,
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
        "step_depth_integral": (
            cube["stepKa"] == cube["stepKb"] and
            cube["depthA1"] == 2 * cube["stepKa"] and
            cube["depthB1"] == 2 * cube["stepKb"]
        ),
        "single_k_positive": cube["singleCoreK"] > 0,
    }
    if family == "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K":
        k_tiles = ceil_div(cube["Ka"], cube["baseK"])
        maximum_tiles = ceil_div(k_tiles, cube["usedCoreNum"])
        minimum_tiles = k_tiles // cube["usedCoreNum"]
        checks.update({
            "single_output_tile": cube["singleCoreM"] == cube["M"] and cube["singleCoreN"] == cube["N"],
            "three_k_blocks_per_owner": minimum_tiles >= 3,
            "single_k_is_largest_owner": cube["singleCoreK"] == min(cube["Ka"], maximum_tiles * cube["baseK"]),
        })
    else:
        total_tiles = ceil_div(cube["M"], cube["singleCoreM"]) * ceil_div(cube["N"], cube["singleCoreN"])
        tail_tiles = total_tiles % cube["usedCoreNum"]
        checks.update({
            "incomplete_tail_wave": total_tiles > cube["usedCoreNum"] and 0 < tail_tiles <= cube["usedCoreNum"] // 2,
            "full_k_parent_tile": cube["singleCoreK"] == cube["Ka"],
        })
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
        "vector": {"baseAN": 0, "baseAD": 0, "baseBN": 0, "baseBD": 0},
        "padding_hex": {"220": "00000000", "252": "00000000", "260": "00000000"},
    }


def generate(m: int, k: int, n: int, dtype: str,
             trans_a: bool, trans_b: bool,
             *, required_family: str | None = None) -> dict:
    req = request(m, k, n, dtype, trans_a, trans_b)
    family, reason = select_family(req, required_family)
    if family == "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K":
        cube, derivation = direct_init_whole_output_cube(req)
    else:
        cube, derivation = direct_init_tail_wave_cube(req)
    checks = validate_cube(cube, family)
    packet = payload(req, cube)
    blob = pack_packet(packet)
    if len(blob) != 280 or unpack_packet(blob) != packet:
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
