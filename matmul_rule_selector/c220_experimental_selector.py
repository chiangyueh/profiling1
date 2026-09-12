#!/usr/bin/env python3
"""Independent finite-rule selector for the C220 structural MatMul paths."""
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
FAMILIES = {
    "MULTI_CORE_SPLIT_K": {"fp32": 41},
    "SINGLE_CORE_NKM_SPLIT_K": {"fp32": 51},
    "SINGLE_CORE_SPLIT_K_GM_TO_L1": {"fp16": 61, "bf16": 61},
    "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED": {"fp16": 60, "bf16": 60},
    "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": {"fp16": 121, "bf16": 121},
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def request(m: int, k: int, n: int, dtype: str, trans_a: bool, trans_b: bool) -> dict:
    if dtype not in ("fp16", "bf16", "fp32"):
        raise ValueError(f"unsupported dtype={dtype}")
    if min(m, n, k) <= 0 or max(m, n, k) >= 2**32:
        raise ValueError("M/N/K must be positive uint32 dimensions")
    return {
        "M": int(m), "N": int(n), "K": int(k), "dtype": dtype,
        "transA": bool(trans_a), "transB": bool(trans_b),
    }


def applicable(req: dict, family: str) -> tuple[bool, str]:
    m, n, k, dtype = req["M"], req["N"], req["K"], req["dtype"]
    ta, tb = req["transA"], req["transB"]
    if family == "MULTI_CORE_SPLIT_K":
        ok = dtype == "fp32" and not ta and tb and m <= 128 and n <= 128 and k >= 8192
        return ok, "fp32 NT, small output, K has enough independent ownership stripes"
    if family == "SINGLE_CORE_NKM_SPLIT_K":
        ok = dtype == "fp32" and not ta and tb and n <= 64 and m >= 1920 and 27392 <= k < 65535
        return ok, "fp32 NT, narrow N, large M and split-worthy K"
    if family == "SINGLE_CORE_SPLIT_K_GM_TO_L1":
        ok = (dtype in ("fp16", "bf16") and not ta and not tb and
              n % 128 == 0 and k % 128 == 0 and k >= 27392 and max(n, k) < 65535)
        return ok, "16-bit NN with 256-byte-aligned inner axes and split-worthy K"
    if family == "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED":
        ok = (dtype in ("fp16", "bf16") and not ta and not tb and
              k >= 27392 and max(n, k) < 65535 and (n % 128 != 0 or k % 128 != 0))
        return ok, "16-bit NN with an unaligned inner axis requiring the mixed converter"
    if family == "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD":
        middle_k = 9216 <= k <= 20480 and 6144 <= n < 65536
        middle_n = 6144 <= k < 65536 and 6144 <= n < 16384
        ok = (dtype in ("fp16", "bf16") and not ta and m <= 256 and
              n % 16 == 0 and k % 16 == 0 and (middle_k or middle_n))
        return ok, "small M and large aligned N/K allow one A stripe to be reused across N"
    raise ValueError(f"unknown family={family}")


def empty_cube(req: dict) -> dict:
    cube = {name: 0 for name in CUBE_FIELDS}
    cube.update({
        "M": req["M"], "N": req["N"], "Ka": req["K"], "Kb": req["K"],
        "batchM": 1, "batchN": 1, "singleBatchM": 1, "singleBatchN": 1,
        "BatchNum": 1,
    })
    return cube


def choose_mn_grid(m: int, n: int, base_m: int, base_n: int) -> tuple[int, int, int]:
    best = None
    for m_cores in range(1, AIC + 1):
        for n_cores in range(1, AIC // m_cores + 1):
            requested_cores = m_cores * n_cores
            single_m = align_up(ceil_div(m, m_cores), 16)
            single_n = align_up(ceil_div(n, n_cores), 16)
            used = ceil_div(m, single_m) * ceil_div(n, single_n)
            waves = ceil_div(ceil_div(m, single_m) * ceil_div(n, single_n), used)
            tail = (single_m * m_cores - m) * n + (single_n * n_cores - n) * m
            score = (waves, -used, requested_cores - used, tail, single_m * single_n)
            if best is None or score < best[0]:
                best = (score, single_m, single_n, used)
    assert best is not None
    return best[1], best[2], best[3]


def sc_cube(req: dict, *, nkm: bool) -> tuple[dict, dict]:
    width = 4 if req["dtype"] == "fp32" else 2
    base_m = 128
    base_n = 64 if nkm else 128
    base_k = 32 if width == 4 else 64
    single_m, single_n, used = choose_mn_grid(req["M"], req["N"], base_m, base_n)
    step_m = max(1, min(3, ceil_div(single_m, base_m)))
    step_n = max(1, min(3, ceil_div(single_n, base_n)))
    max_k_a = (L1 // 2) // (base_m * base_k * width)
    max_k_b = (L1 // 2) // (base_n * base_k * width)
    step_k = max(1, min(max_k_a, max_k_b, req["K"] // base_k, 32))
    while step_k > 1:
        depth_a = 2 * step_m * step_k
        depth_b = 2 * step_n * step_k
        l1_bytes = (base_m * base_k * depth_a + base_n * base_k * depth_b) * width
        if l1_bytes <= L1:
            break
        step_k -= 1
    depth_a = 2 * step_m * step_k
    depth_b = 2 * step_n * step_k
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": single_m, "singleCoreN": single_n,
        "singleCoreK": step_k * base_k,
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": depth_a, "depthB1": depth_b,
        "stepM": step_m, "stepN": step_n,
        "stepKa": step_k, "stepKb": step_k,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
    })
    audit = {
        "mn_grid_used_cores": used,
        "k_stripe": step_k * base_k,
        "l0a_bytes": base_m * base_k * width * 2,
        "l0b_bytes": base_n * base_k * width * 2,
        "l0c_bytes": base_m * base_n * 4 * 2,
        "l1_bytes": (base_m * base_k * depth_a + base_n * base_k * depth_b) * width,
        "loop_order": "NKM" if nkm else "MKN",
    }
    return cube, audit


def multi_core_cube(req: dict) -> tuple[dict, dict]:
    k_parts = min(AIC, max(2, ceil_div(req["K"], 4096)))
    single_k = align_up(ceil_div(req["K"], k_parts), 16)
    k_parts = ceil_div(req["K"], single_k)
    base_m = min(align_up(req["M"], 16), 128)
    base_n = min(align_up(req["N"], 16), 128)
    base_k = 32
    while base_m * base_n * 4 * 2 > L0C:
        if base_m >= base_n:
            base_m //= 2
        else:
            base_n //= 2
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": k_parts,
        "singleCoreM": req["M"], "singleCoreN": req["N"], "singleCoreK": single_k,
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": 2, "depthB1": 2,
        "stepM": 1, "stepN": 1, "stepKa": 1, "stepKb": 1,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 2,
    })
    return cube, {
        "ownership": "disjoint_K_stripes_with_atomic_output",
        "k_partitions": k_parts, "k_stripe": single_k,
        "output_clear_bytes_per_core": align_up(ceil_div(req["M"] * req["N"] * 4, k_parts), 32),
    }


def al1_steps(req: dict, base_m: int, base_n: int, base_k: int,
             single_n: int) -> tuple[int, int, int, int, int]:
    width = 2
    block_a = base_m * base_k * width
    block_b = base_n * base_k * width
    max_step_n = max(1, ceil_div(single_n, base_n))
    for step_n in range(1, max_step_n + 1):
        best = None
        max_step_kb = min((L1 - 256) // (step_n * 2 * block_b), req["K"] // base_k)
        for step_kb in range(1, max_step_kb + 1):
            b_bytes = step_n * step_kb * 2 * block_b
            max_step_ka = min((L1 + 256 - b_bytes) // block_a, req["K"] // base_k)
            if max_step_ka < step_kb:
                continue
            step_ka = max_step_ka // step_kb * step_kb
            loops = ceil_div(req["K"], step_ka * base_k)
            align_score = 2 if step_kb * base_k * width % 512 == 0 else 1
            score = (align_score if req["transB"] else 0, -loops, b_bytes, step_ka)
            if best is None or score > best[0]:
                best = (score, step_ka, step_kb)
        if best is not None:
            step_ka, step_kb = best[1], best[2]
            return step_ka, step_kb, step_n, step_ka, step_n * step_kb * 2
    raise ValueError("no legal AL1/BL1 step combination")


def al1_cube(req: dict) -> tuple[dict, dict]:
    base_m = align_up(req["M"], 16)
    candidates = []
    for base_n in (512, 384, 256, 128, 64, 32):
        if base_m * base_n * 4 > L0C:
            continue
        if req["transB"] and req["K"] > req["N"] and base_n > 128:
            continue
        for base_k in range(256, 31, -32):
            if base_m * base_k * 2 * 2 > L0A or base_n * base_k * 2 * 2 > L0B:
                continue
            n_tiles = ceil_div(req["N"], base_n)
            used = min(n_tiles, AIC)
            if used == 0:
                continue
            remainder = n_tiles % used
            head = used if remainder == 0 else remainder
            tail = used - head
            if tail > 0 and head < tail:
                continue
            single_n = base_n * ceil_div(n_tiles, used)
            try:
                steps = al1_steps(req, base_m, base_n, base_k, single_n)
            except ValueError:
                continue
            step_ka, step_kb, step_n, depth_a, depth_b = steps
            perfect = n_tiles % used == 0
            bk_align = int(step_kb * base_k * 2 % 512 == 0)
            k_loops = ceil_div(req["K"], step_ka * base_k)
            rounds = ceil_div(single_n, base_n) * ceil_div(req["K"], base_k)
            score = (int(perfect), used, bk_align if req["transB"] else 0,
                     -k_loops, -rounds, base_n, base_k)
            candidates.append((score, base_n, base_k, single_n, used, steps))
    if not candidates:
        raise ValueError("no legal SC+AL1 finite candidate")
    score, base_n, base_k, single_n, used, steps = max(candidates)
    step_ka, step_kb, step_n, depth_a, depth_b = steps
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": req["M"], "singleCoreN": single_n, "singleCoreK": req["K"],
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": depth_a, "depthB1": depth_b,
        "stepM": 1, "stepN": step_n, "stepKa": step_ka, "stepKb": step_kb,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 1,
    })
    l1_bytes = base_m * base_k * depth_a * 2 + base_n * base_k * depth_b * 2
    return cube, {
        "finite_candidate_count": len(candidates), "candidate_score": list(score),
        "a_l1_stripe_k": step_ka * base_k,
        "a_l1_load_count": ceil_div(req["K"], step_ka * base_k),
        "l1_bytes": l1_bytes,
    }


def validate_cube(cube: dict, width: int, family: str) -> dict:
    resident_a = align_up(cube["singleCoreM"], 16) * cube["singleCoreK"] * width
    resident_b = (align_up(cube["singleCoreK"], 16) *
                  align_up(cube["singleCoreN"], 16) * width)
    checks = {
        "all_words_uint32": all(type(cube[name]) is int and 0 <= cube[name] < 2**32 for name in CUBE_FIELDS),
        "base_alignment": cube["baseM"] % 16 == 0 and cube["baseN"] % 16 == 0 and cube["baseK"] % 16 == 0,
        "core_bound": 1 <= cube["usedCoreNum"] <= AIC,
        "l0a_fit": cube["baseM"] * cube["baseK"] * width * cube["dbL0A"] <= L0A,
        "l0b_fit": cube["baseN"] * cube["baseK"] * width * cube["dbL0B"] <= L0B,
        "l0c_fit": cube["baseM"] * cube["baseN"] * 4 * cube["dbL0C"] <= L0C,
        "l1_fit": ((cube["baseM"] * cube["baseK"] * cube["depthA1"] +
                    cube["baseN"] * cube["baseK"] * cube["depthB1"]) * width <= L1 + 256),
        "gm_to_l1_resident_fit": (
            family not in (
                "SINGLE_CORE_SPLIT_K_GM_TO_L1",
                "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
            ) or resident_a + resident_b <= L1
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"illegal C220 candidate: {checks}")
    return checks


def vector_geometry(req: dict, used: int) -> dict:
    width = 4 if req["dtype"] == "fp32" else 2
    cores = max(2 * used, 1)
    c0 = 32 // width
    def geom(rows: int, cols: int) -> tuple[int, int]:
        aligned_rows = align_up(rows, 16)
        aligned_cols = align_up(cols, c0)
        base_cols = min(aligned_cols, 4096 // width)
        base_rows = max(16, min(aligned_rows, (196352 // 2 // width) // max(base_cols, 1)))
        return base_rows, base_cols
    a_rows, a_cols = (req["K"], req["M"]) if req["transA"] else (req["M"], req["K"])
    b_rows, b_cols = (req["N"], req["K"]) if req["transB"] else (req["K"], req["N"])
    base_an, base_ad = geom(a_rows, a_cols)
    base_bn, base_bd = geom(b_rows, b_cols)
    return {"baseAN": base_an, "baseAD": base_ad, "baseBN": base_bn, "baseBD": base_bd}


def generate(m: int, k: int, n: int, dtype: str, trans_a: bool, trans_b: bool,
             *, required_family: str) -> dict:
    req = request(m, k, n, dtype, trans_a, trans_b)
    if required_family not in FAMILIES:
        raise ValueError(f"unknown required_family={required_family}")
    ok, reason = applicable(req, required_family)
    if not ok:
        raise ValueError(f"{required_family} is not legal for {req}")
    if dtype not in FAMILIES[required_family]:
        raise ValueError(f"{required_family} has no {dtype} kernel")
    if required_family == "MULTI_CORE_SPLIT_K":
        cube, derivation = multi_core_cube(req)
        cal_order = 0
    elif required_family == "SINGLE_CORE_NKM_SPLIT_K":
        cube, derivation = sc_cube(req, nkm=True)
        cal_order = 1
    elif required_family in ("SINGLE_CORE_SPLIT_K_GM_TO_L1", "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED"):
        cube, derivation = sc_cube(req, nkm=False)
        cal_order = 0
    else:
        cube, derivation = al1_cube(req)
        cal_order = 0
    width = 4 if dtype == "fp32" else 2
    checks = validate_cube(cube, width, required_family)
    if required_family in (
        "SINGLE_CORE_SPLIT_K_GM_TO_L1",
        "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
    ):
        derivation.update({
            "resident_a_l1_bytes": align_up(cube["singleCoreM"], 16) * cube["singleCoreK"] * width,
            "resident_b_l1_bytes": (align_up(cube["singleCoreK"], 16) *
                                     align_up(cube["singleCoreN"], 16) * width),
            "resident_l1_limit_bytes": L1,
        })
    unaligned = required_family.endswith("UNALIGNED")
    vector = vector_geometry(req, cube["usedCoreNum"])
    payload = {
        "cube_words": [cube[name] for name in CUBE_FIELDS],
        "tileL2cacheTiling": {
            "mTileCntL2": 1, "nTileCntL2": 1, "mTileBlock": 0,
            "nTileBlock": 0, "calOrder": cal_order,
        },
        "matmulRunInfo": {
            "transA": int(trans_a), "transB": int(trans_b),
            "nd2nzA": int(unaligned and k % (256 // width) != 0),
            "nd2nzB": int(unaligned and n % (256 // width) != 0),
            "isNzA": 0, "isNzB": 0, "isHf32": 0,
        },
        "l2CacheFlag": 0,
        "vector": vector,
        "padding_hex": {"220": "00000000", "252": "00000000", "260": "00000000"},
    }
    blob = pack_packet(payload, "target280_cube200")
    if unpack_packet(blob, "target280_cube200") != payload:
        raise RuntimeError("280-byte packet roundtrip mismatch")
    needs_fp32_output = dtype != "fp32" and (
        required_family != "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD" or
        cube["stepKa"] * cube["baseK"] < k
    )
    output_workspace = align_up(m * n * 4, 512) if needs_fp32_output else 0
    conversion_workspace = align_up(m * k * width, 512) + align_up(k * n * width, 512) if unaligned else 0
    workspace = SYSTEM_WORKSPACE + output_workspace + conversion_workspace
    suffix = FAMILIES[required_family][dtype]
    return {
        "status": "EXPERIMENTAL_FAMILY_TILING",
        "npu_eligible": True,
        "formula_family": required_family,
        "kernel_suffix": suffix,
        "selection_basis": "INDEPENDENT_STRUCTURAL_RULE",
        "applicability_reason": reason,
        "packet_layout": "target280_cube200",
        "packet_bytes": len(blob),
        "packet_hex": blob.hex(),
        "packet_sha256": hashlib.sha256(blob).hexdigest(),
        "block_dim": cube["usedCoreNum"],
        "workspace_bytes": workspace,
        "cube": cube,
        "resource_checks": checks,
        "derivation": derivation,
        "runtime_dependencies": {
            "official_selector": False, "cost_model": False, "history": False,
            "repo_lookup": False, "candidate_search": False,
        },
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.m, args.k, args.n, args.dtype, args.trans_a,
                              args.trans_b, required_family=args.family), sort_keys=True))


if __name__ == "__main__":
    main()
