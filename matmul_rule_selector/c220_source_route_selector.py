#!/usr/bin/env python3
"""Complete-packet adapters for C220 source routes absent from CANN 8.1."""
from __future__ import annotations

import hashlib

from c220_experimental_selector import generate as generate_later_route
from complete_formula_selector import generate as generate_installed_route
from matmul_reconstruction._core.abi import pack_packet, unpack_packet


SYSTEM_WORKSPACE = 20 * 1024 * 1024
LATER_FAMILY_BY_SUFFIX = {
    41: "MULTI_CORE_SPLIT_K",
    51: "SINGLE_CORE_NKM_SPLIT_K",
    60: "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
    61: "SINGLE_CORE_SPLIT_K_GM_TO_L1",
    121: "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD",
}
ADAPTED_FAMILY_BY_SUFFIX = {
    202: "CVP_BL1_PARALLEL_ND2NZ",
    20030: "DETERMINISTIC_SPLIT_K_VEC_NZ2ND_UNALIGNED",
    20031: "DETERMINISTIC_SPLIT_K_VEC_NZ2ND_ALIGNED",
    100001: "BASE_K_SHIFT",
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _repack_installed(result: dict) -> bytes:
    request = result["request"]
    l2 = result["l2_fields"]
    payload = {
        "cube_words": [result["abi_fields"]["cube"][name] for name in (
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
        )],
        "tileL2cacheTiling": {
            "mTileCntL2": l2["mTile"], "nTileCntL2": l2["nTile"],
            "mTileBlock": l2["mTileBlock"], "nTileBlock": l2["nTileBlock"],
            "calOrder": l2["calOrder"],
        },
        "matmulRunInfo": {
            "transA": int(request["transA"]), "transB": int(request["transB"]),
            "nd2nzA": int(result["conversion_a"]), "nd2nzB": int(result["conversion_b"]),
            "isNzA": 0, "isNzB": 0, "isHf32": 0,
        },
        "l2CacheFlag": int(result["abi_fields"]["l2_cache_flag"]),
        "vector": {name: int(result["abi_fields"]["vector"][name]) for name in (
            "baseAN", "baseAD", "baseBN", "baseBD",
        )},
        "padding_hex": {"220": "00000000", "252": "00000000", "260": "00000000"},
    }
    blob = pack_packet(payload, "target280_cube200")
    if unpack_packet(blob, "target280_cube200") != payload:
        raise RuntimeError("C220 route packet roundtrip mismatch")
    return blob


def _adapted_route(
    suffix: int, m: int, k: int, n: int, dtype: str,
    trans_a: bool, trans_b: bool,
) -> dict:
    result = generate_installed_route(m, k, n, dtype, trans_a, trans_b)
    source_suffix = int(result["kernel_suffix"])
    required_source = {
        100001: {1},
        202: {200},
        20030: {30},
        20031: {31},
    }[suffix]
    if source_suffix not in required_source:
        raise ValueError(
            f"suffix={suffix} requires geometry from {sorted(required_source)}, got {source_suffix}"
        )
    blob = _repack_installed(result)
    cube = result["abi_fields"]["cube"]
    workspace = int(result["workspace_bytes"])
    if suffix == 202:
        width = 4
        c0 = 32 // width
        resident_axis = cube["singleCoreM"] if result["conversion_a"] else cube["singleCoreN"]
        one_buffer = align_up(resident_axis, 16) * align_up(cube["singleCoreK"], c0)
        workspace += 2 * cube["usedCoreNum"] * one_buffer * width
    elif suffix in (20030, 20031):
        single_nz = align_up(cube["singleCoreM"], 16) * align_up(cube["singleCoreN"], 16)
        workspace = max(
            workspace,
            SYSTEM_WORKSPACE + 2 * cube["usedCoreNum"] * single_nz * 4,
        )
    return {
        "status": "BUNDLED_LATER_OFFICIAL_SOURCE_ROUTE_PACKET",
        "origin": "BUNDLED_LATER_OFFICIAL_MATMUL_V3_SOURCE",
        "formula_family": ADAPTED_FAMILY_BY_SUFFIX[suffix],
        "kernel_suffix": suffix,
        "packet_bytes": len(blob),
        "packet_hex": blob.hex(),
        "packet_sha256": hashlib.sha256(blob).hexdigest(),
        "block_dim": cube["usedCoreNum"],
        "workspace_bytes": workspace,
        "cube": cube,
        "source_geometry_suffix": source_suffix,
        "selection_basis": "SOURCE_ROUTE_CONTRACT_PLUS_CLOSED_FORM_PACKET_ADAPTER",
        "runtime_dependencies": {
            "official_selector": False, "official_tiling_seed": False,
            "cost_model": False, "history": False, "repo_lookup": False,
            "candidate_search": False,
        },
    }


def generate_for_suffix(
    suffix: int, m: int, k: int, n: int, dtype: str,
    trans_a: bool, trans_b: bool,
) -> dict:
    suffix = int(suffix)
    if suffix in LATER_FAMILY_BY_SUFFIX:
        result = generate_later_route(
            m, k, n, dtype, trans_a, trans_b,
            required_family=LATER_FAMILY_BY_SUFFIX[suffix],
        )
        if int(result["kernel_suffix"]) != suffix:
            raise ValueError(f"later route selector emitted suffix={result['kernel_suffix']}")
        return result
    if suffix in ADAPTED_FAMILY_BY_SUFFIX:
        return _adapted_route(suffix, m, k, n, dtype, trans_a, trans_b)
    raise ValueError(f"suffix={suffix} is not an added C220 source route")
