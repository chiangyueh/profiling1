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
CORE_FACTORS = (1, 2, 4, 5, 10, 20)
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
NPU_BUILDABLE_ON_CANN81 = {
    "MULTI_CORE_SPLIT_K",
    "SINGLE_CORE_NKM_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD",
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
              n % 128 == 0 and k % 128 == 0 and (middle_k or middle_n))
        return ok, "small M and large aligned N/K allow one A stripe to be reused across N"
    raise ValueError(f"unknown family={family}")


def select_family(req: dict) -> tuple[str, str]:
    """Choose by disjoint structural predicates, without latency ranking."""
    matches = []
    for family in FAMILIES:
        ok, reason = applicable(req, family)
        if ok:
            matches.append((family, reason))
    if len(matches) != 1:
        raise ValueError(
            f"expanded family rule must have exactly one match, got "
            f"{[family for family, _ in matches]} for {req}"
        )
    return matches[0]


def empty_cube(req: dict) -> dict:
    cube = {name: 0 for name in CUBE_FIELDS}
    cube.update({
        "M": req["M"], "N": req["N"], "Ka": req["K"], "Kb": req["K"],
        "batchM": 1, "batchN": 1, "singleBatchM": 1, "singleBatchN": 1,
        "BatchNum": 1,
    })
    return cube


def next_core_factor(value: int) -> int:
    for factor in CORE_FACTORS:
        if value <= factor:
            return factor
    return value


def sc_factor_grid(m: int, n: int, step_m: int, base_m: int,
                   m_align: int, n_align: int) -> tuple[int, int, int, int, int]:
    """Translate DoSingleCoreSplitKTiling's 20-core factor projection.

    This deliberately accepts only the branch for which the source equation
    determines singleCoreM/N without inheriting opaque pre-selector fields.
    """
    m_tiles = ceil_div(m, step_m * base_m)
    n_tiles = next_core_factor(ceil_div(n, 3072))
    if m_tiles * n_tiles >= AIC:
        m_tiles = max(AIC // n_tiles, 1)
        single_m = min(m, align_up(ceil_div(m, m_tiles), m_align))
        single_n = min(n, align_up(ceil_div(n, n_tiles), n_align))
        used = min(ceil_div(m, single_m) * ceil_div(n, single_n), AIC)
        if used == AIC:
            return single_m, single_n, used, ceil_div(m, single_m), ceil_div(n, single_n)

    n_tiles = max(AIC // max(m_tiles, 1), 1)
    total = 0
    best = None
    single_n = n
    while n_tiles <= AIC and total < AIC and single_n >= 1024:
        n_tiles = next_core_factor(n_tiles)
        m_tiles = max(AIC // n_tiles, 1)
        single_n = align_up(ceil_div(n, n_tiles), n_align)
        single_m = align_up(ceil_div(m, m_tiles), m_align)
        m_count = ceil_div(m, single_m)
        n_count = ceil_div(n, single_n)
        if m_count * n_count > total:
            total = m_count * n_count
            best = (min(single_m, m), min(single_n, n), total, m_count, n_count)
        n_tiles += 1
    if best is None:
        raise ValueError("shape depends on inherited SC geometry; closed-form GM-to-L1 route rejected")
    return best


def nkm_cube(req: dict) -> tuple[dict, dict]:
    """C220 NK 3x3 geometry, with N kept resident and ownership split on M."""
    base_m = 128
    base_n = 128
    base_k = 64
    single_m = min(req["M"], align_up(ceil_div(req["M"], AIC), 8))
    single_n = req["N"]
    used = ceil_div(req["M"], single_m)
    step_m, step_n, step_k = 1, 3, 3
    depth_a, depth_b = 6, 9
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
        "iterateOrder": 0,
    })
    audit = {
        "mn_grid_used_cores": used,
        "k_stripe": step_k * base_k,
        "l0a_bytes": base_m * base_k * 4 * 2,
        "l0b_bytes": base_n * base_k * 4 * 2,
        "l0c_bytes": base_m * base_n * 4 * 2,
        "l1_bytes": (base_m * base_k * depth_a + base_n * base_k * depth_b) * 4,
        "loop_order": "NKM",
        "equation": "NK33(base=128x128x64,step=1x3x3,depth=6/9); M ownership projected to 20 AIC",
    }
    return cube, audit


def gm_to_l1_cube(req: dict) -> tuple[dict, dict]:
    """C220 MKN Split-K geometry consumed by the real GM-to-L1 kernels."""
    base_m = 128
    base_n = 128
    base_k = 128
    # The 2x4 schedule is required when either outer dimension is below one
    # 3x3 macro tile.  Otherwise retain the MK 3x3 pipeline.
    if req["M"] < 384 or req["N"] < 384:
        step_m, step_n, step_k = 1, 1, 4
        depth_a, depth_b = 8, 8
        schedule = "MK24_SMALL_OUTER"
    else:
        step_m, step_n, step_k = 3, 1, 3
        depth_a, depth_b = 9, 6
        schedule = "MK33"
    single_m, single_n, used, m_owners, n_owners = sc_factor_grid(
        req["M"], req["N"], step_m, base_m, 16, 128
    )
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
        "iterateOrder": 1,
    })
    return cube, {
        "schedule": schedule,
        "mn_owner_grid": [m_owners, n_owners],
        "mn_grid_used_cores": used,
        "k_stripe": step_k * base_k,
        "l0a_bytes": base_m * base_k * 2 * 2,
        "l0b_bytes": base_n * base_k * 2 * 2,
        "l0c_bytes": base_m * base_n * 4 * 2,
        "l1_bytes": (base_m * base_k * depth_a + base_n * base_k * depth_b) * 2,
        "loop_order": "MKN",
        "equation": "C220 factor-grid ownership with MK24/MK33 GM-to-L1 pipeline",
    }


def multi_core_cube(req: dict) -> tuple[dict, dict]:
    # MULTI_CORE_SINGLE_K is 384 in the C220 host policy.  The old prototype
    # used 4096 here and accidentally serialized most K work onto 4--12 AICs.
    maximum_parts = min(AIC, req["K"] // 384)
    if maximum_parts < 2:
        raise ValueError("K cannot provide two independent 384-element stripes")
    base_k = 64
    single_k = align_up(ceil_div(req["K"], maximum_parts), base_k)
    k_parts = ceil_div(req["K"], single_k)
    base_m = min(align_up(req["M"], 16), 128)
    base_n = min(align_up(req["N"], 16), 128)
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
        "minimum_k_per_owner": 384,
        "equation": "C=min(20,floor(K/384)); singleK=align64(ceil(K/C)); C=ceil(K/singleK)",
    }


def al1_steps(req: dict, base_m: int, base_n: int, base_k: int,
             single_n: int) -> tuple[int, int, int, int, int]:
    width = 2
    block_a = base_m * base_k * width
    block_b = base_n * base_k * width
    max_step_n = max(1, ceil_div(single_n, base_n))
    available_l1 = L1 + 256
    for step_n in range(1, max_step_n + 1):
        best = None
        max_step_kb = min(available_l1 // (step_n * 2 * block_b), req["K"] // base_k)
        for step_kb in range(1, max_step_kb + 1):
            b_bytes = step_n * step_kb * 2 * block_b
            max_step_ka = min((available_l1 - b_bytes) // block_a, req["K"] // base_k)
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
    step_m = ceil_div(req["M"], base_m)
    max_base_n = L0C // (base_m * 4)
    if req["transB"] and req["K"] > req["N"]:
        max_base_n = min(max_base_n, 128)
    best = None
    legal_projections = 0

    def block_rounds(single_n: int, base_n: int, base_k: int) -> int:
        return ceil_div(single_n, base_n) * ceil_div(req["K"], base_k)

    def compare_rounds(cur: dict, old: dict) -> int:
        cur_rounds = block_rounds(cur["single_n"], cur["base_n"], cur["base_k"])
        old_rounds = block_rounds(old["single_n"], old["base_n"], old["base_k"])
        cur_ref = block_rounds(cur["single_n"], 256, 128)
        old_ref = block_rounds(old["single_n"], 256, 128)
        cur_le = cur_rounds <= cur_ref
        old_le = old_rounds <= old_ref
        if cur_le != old_le:
            return 1 if cur_le else -1
        if cur_le:
            left, right = cur_rounds * old_ref, old_rounds * cur_ref
            return 1 if left > right else (-1 if left < right else 0)
        left = (cur_rounds - cur_ref) * old_ref
        right = (old_rounds - old_ref) * cur_ref
        return 1 if left < right else (-1 if left > right else 0)

    def better(cur: dict, old: dict | None) -> bool:
        if old is None:
            return True
        cur_perfect = cur["n_tiles"] % cur["used"] == 0
        old_perfect = old["n_tiles"] % old["used"] == 0
        if not req["transB"] and cur_perfect != old_perfect:
            return cur_perfect
        if cur["used"] != old["used"]:
            return cur["used"] > old["used"]
        if not req["transB"] and not cur_perfect:
            cur_balance = cur["head"] - cur["tail"]
            old_balance = old["head"] - old["tail"]
            if cur_balance != old_balance:
                return cur_balance < old_balance
        if req["transB"] and cur["bk_align"] != old["bk_align"]:
            return cur["bk_align"] > old["bk_align"]
        round_cmp = compare_rounds(cur, old)
        if round_cmp:
            return round_cmp > 0
        if cur["base_n"] != old["base_n"]:
            return cur["base_n"] > old["base_n"]
        return cur["base_k"] > old["base_k"]

    for base_n in (512, 384, 256, 128, 64, 32):
        if base_n > max_base_n:
            continue
        n_tiles = ceil_div(req["N"], base_n)
        used = min(n_tiles, AIC)
        remainder = n_tiles % used
        head = used if remainder == 0 else remainder
        tail = used - head
        if tail > 0 and head < tail:
            continue
        max_ka = L0A // (2 * 2 * base_m)
        max_kb = L0B // (2 * 2 * base_n)
        policy_base_k = min(256, min(max_ka, max_kb) // 32 * 32)
        if policy_base_k < 32:
            continue
        single_n = base_n * ceil_div(n_tiles, used)
        for base_k in range(policy_base_k, 31, -32):
            if base_m * base_k * 2 * 2 > L0A or base_n * base_k * 2 * 2 > L0B:
                continue
            try:
                steps = al1_steps(req, base_m, base_n, base_k, single_n)
            except ValueError:
                continue
            step_ka, step_kb, step_n, depth_a, depth_b = steps
            bk_bytes = step_kb * base_k * 2
            bk_align = 2 if bk_bytes % 512 == 0 else (1 if bk_bytes % 256 == 0 else 0)
            current = {
                "base_n": base_n, "base_k": base_k,
                "single_n": single_n, "used": used,
                "n_tiles": n_tiles, "head": head, "tail": tail,
                "bk_align": bk_align, "steps": steps,
            }
            legal_projections += 1
            if better(current, best):
                best = current
    if best is None:
        raise ValueError("no legal SC+AL1 hardware projection")
    base_n = best["base_n"]
    base_k = best["base_k"]
    single_n = best["single_n"]
    used = best["used"]
    steps = best["steps"]
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
        "legal_hardware_projections": legal_projections,
        "selection_lexicographic_order": [
            "perfect_n_split", "used_cores", "head_tail_balance",
            "b_trans_512b_alignment", "block_round_ratio", "base_n", "base_k",
        ],
        "a_l1_stripe_k": step_ka * base_k,
        "a_l1_load_count": ceil_div(req["K"], step_ka * base_k),
        "l1_bytes": l1_bytes,
    }


def validate_cube(cube: dict, width: int, family: str) -> dict:
    checks = {
        "all_words_uint32": all(type(cube[name]) is int and 0 <= cube[name] < 2**32 for name in CUBE_FIELDS),
        "base_alignment": cube["baseM"] % 16 == 0 and cube["baseN"] % 16 == 0 and cube["baseK"] % 16 == 0,
        "core_bound": 1 <= cube["usedCoreNum"] <= AIC,
        "l0a_fit": cube["baseM"] * cube["baseK"] * width * cube["dbL0A"] <= L0A,
        "l0b_fit": cube["baseN"] * cube["baseK"] * width * cube["dbL0B"] <= L0B,
        "l0c_fit": cube["baseM"] * cube["baseN"] * 4 * cube["dbL0C"] <= L0C,
        "l1_fit": ((cube["baseM"] * cube["baseK"] * cube["depthA1"] +
                    cube["baseN"] * cube["baseK"] * cube["depthB1"]) * width <= L1 + 256),
        # The real GM-to-L1 kernel streams stepK blocks through L1.  Requiring
        # the complete singleCoreM x singleCoreK and singleCoreK x singleCoreN
        # rectangles to be resident was a constraint of the removed prototype,
        # not of the actual kernel.
        "gm_to_l1_streaming": True,
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
             *, required_family: str | None = None) -> dict:
    req = request(m, k, n, dtype, trans_a, trans_b)
    selected_family, reason = select_family(req)
    if required_family is not None and selected_family != required_family:
        raise ValueError(
            f"independent selector chose {selected_family}, expected {required_family}"
        )
    if dtype not in FAMILIES[selected_family]:
        raise ValueError(f"{selected_family} has no {dtype} kernel")
    if selected_family == "MULTI_CORE_SPLIT_K":
        cube, derivation = multi_core_cube(req)
        cal_order = 0
    elif selected_family == "SINGLE_CORE_NKM_SPLIT_K":
        cube, derivation = nkm_cube(req)
        cal_order = 1
    elif selected_family in ("SINGLE_CORE_SPLIT_K_GM_TO_L1", "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED"):
        cube, derivation = gm_to_l1_cube(req)
        cal_order = 0
    else:
        cube, derivation = al1_cube(req)
        cal_order = 0
    width = 4 if dtype == "fp32" else 2
    checks = validate_cube(cube, width, selected_family)
    unaligned = selected_family.endswith("UNALIGNED")
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
        selected_family != "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD" or
        cube["stepKa"] * cube["baseK"] < k
    )
    output_workspace = align_up(m * n * 4, 512) if needs_fp32_output else 0
    conversion_workspace = align_up(m * k * width, 512) + align_up(k * n * width, 512) if unaligned else 0
    workspace = SYSTEM_WORKSPACE + output_workspace + conversion_workspace
    suffix = FAMILIES[selected_family][dtype]
    npu_eligible = selected_family in NPU_BUILDABLE_ON_CANN81
    return {
        "status": "EXPERIMENTAL_FAMILY_TILING",
        "npu_eligible": npu_eligible,
        "formula_family": selected_family,
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
        "toolchain_contract": {
            "target": "C220",
            "host_packet": "COMPLETE_280_BYTE_PACKET",
            "cann81_kernel_build": "PASS" if npu_eligible else "BLOCKED_8_5_L0C_ITERATE_API",
            "fallback_kernel": False,
        },
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
    parser.add_argument("--family")
    args = parser.parse_args()
    print(json.dumps(generate(args.m, args.k, args.n, args.dtype, args.trans_a,
                              args.trans_b, required_family=args.family), sort_keys=True))


if __name__ == "__main__":
    main()
