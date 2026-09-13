#!/usr/bin/env python3
"""Independent finite-candidate selector for C220 MatMul family kernels."""
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
L1_QUEUE_RESERVE = 4096
L2_TO_LOCAL_BYTES_PER_CYCLE = 110.0
L1_TO_L0_BYTES_PER_CYCLE = 256.0
FIXPIPE_BYTES_PER_CYCLE = 64.0
AGGREGATE_HBM_BYTES_PER_CYCLE = 640.0
AGGREGATE_L2_BYTES_PER_CYCLE = 2200.0
CUBE_MACS_PER_CYCLE_16BIT = 4096.0
VECTOR_CAST_ELEMENTS_PER_CYCLE = 128.0
MTE2_TRANSFER_CYCLES = 347.0
MTE1_TRANSFER_CYCLES = 2.0
FIXPIPE_TRANSFER_CYCLES = 31.0
GM_TO_L1_SYNC_CYCLES = 545.0
KERNEL_LAUNCH_CYCLES = 96.0
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


def partition_by_base_tiles(total: int, base: int, groups: int) -> tuple[int, list[int]]:
    total_tiles = ceil_div(total, base)
    tiles_per_group = ceil_div(total_tiles, groups)
    single = tiles_per_group * base
    extents = []
    for start in range(0, total, single):
        extents.append(min(single, total - start))
    return single, extents


def gm_to_l1_cost(req: dict, *, base_m: int, base_n: int, base_k: int,
                  k_stripe: int, m_extents: list[int], n_extents: list[int]) -> dict:
    width = 2
    k_chunks = ceil_div(req["K"], k_stripe)
    k_tail = req["K"] - (k_chunks - 1) * k_stripe
    aligned_k_sum = (k_chunks - 1) * k_stripe + align_up(k_tail, 16)
    k_base_tiles = ((k_chunks - 1) * ceil_div(k_stripe, base_k) +
                    ceil_div(k_tail, base_k))
    core_costs = []
    core_breakdowns = []
    unique_hbm_bytes = (
        (req["M"] * req["K"] + req["K"] * req["N"]) * width +
        req["M"] * req["N"] * (4 + 4 + width)
    )
    total_l2_bytes = 0
    total_useful_macs = 0
    total_padded_macs = 0
    total_matmul_iterations = 0
    for m_extent in m_extents:
        m_tile_count = ceil_div(m_extent, base_m)
        m_tail = m_extent - (m_tile_count - 1) * base_m
        aligned_m_sum = (m_tile_count - 1) * base_m + align_up(m_tail, 16)
        for n_extent in n_extents:
            n_tile_count = ceil_div(n_extent, base_n)
            n_tail = n_extent - (n_tile_count - 1) * base_n
            aligned_n_sum = (n_tile_count - 1) * base_n + align_up(n_tail, 16)
            # Each output base tile remains in L0C while all K stripes are
            # accumulated with Iterate(enPartialSum).  A is therefore loaded
            # once per N base tile and B once per M base tile; neither operand
            # is copied from HBM for every use because GlobalTensor keeps the
            # normal L2 policy.
            gm_bytes = (
                n_tile_count * m_extent * req["K"] * width +
                m_tile_count * req["K"] * n_extent * width
            )
            matmul_iterations = m_tile_count * n_tile_count * k_chunks
            gm_copies = 2 * matmul_iterations
            l1_l0_bytes = (
                aligned_m_sum * n_tile_count * aligned_k_sum +
                m_tile_count * aligned_n_sum * aligned_k_sum
            ) * width
            l1_l0_copies = 2 * m_tile_count * n_tile_count * k_chunks
            useful = m_extent * n_extent * req["K"]
            padded = aligned_m_sum * aligned_n_sum * aligned_k_sum
            total_useful_macs += useful
            total_padded_macs += padded
            cube_macs = padded
            mad_tiles = m_tile_count * n_tile_count * k_base_tiles
            output_bytes = m_extent * n_extent * 4
            output_tile_count = m_tile_count * n_tile_count
            output_cycles = (
                output_bytes / FIXPIPE_BYTES_PER_CYCLE +
                output_tile_count * FIXPIPE_TRANSFER_CYCLES
            )
            input_cycles = gm_bytes / L2_TO_LOCAL_BYTES_PER_CYCLE + gm_copies * MTE2_TRANSFER_CYCLES
            l1_l0_cycles = (l1_l0_bytes / L1_TO_L0_BYTES_PER_CYCLE +
                            l1_l0_copies * MTE1_TRANSFER_CYCLES)
            cube_cycles = cube_macs / CUBE_MACS_PER_CYCLE_16BIT + mad_tiles * 21.0
            # Two AIVs split the rows owned by one AIC.  Conversion begins
            # after the AIC publishes its completion event.
            vector_elements = ceil_div(m_extent, 2) * n_extent
            vector_cycles = (
                vector_elements * 4 / L2_TO_LOCAL_BYTES_PER_CYCLE +
                vector_elements * width / FIXPIPE_BYTES_PER_CYCLE +
                vector_elements / VECTOR_CAST_ELEMENTS_PER_CYCLE +
                MTE2_TRANSFER_CYCLES + 25.0
            )
            core_cycles = (input_cycles + max(cube_cycles, l1_l0_cycles) +
                           output_cycles + vector_cycles)
            core_costs.append(core_cycles)
            core_breakdowns.append({
                "input_l2_to_l1_cycles": input_cycles,
                "cube_cycles": cube_cycles,
                "l1_to_l0_cycles": l1_l0_cycles,
                "cube_l1_overlap_cycles": max(cube_cycles, l1_l0_cycles),
                "output_accumulation_cycles": output_cycles,
                "vector_cast_cycles": vector_cycles,
                "matmul_iterations": matmul_iterations,
                "output_flushes": output_tile_count,
                "total_cycles": core_cycles,
            })
            total_l2_bytes += gm_bytes + m_extent * n_extent * (4 + 4 + width)
            total_matmul_iterations += matmul_iterations
    active_cores = len(core_costs)
    effective_hbm_rate = min(AGGREGATE_HBM_BYTES_PER_CYCLE, active_cores * 32.0)
    effective_l2_rate = min(AGGREGATE_L2_BYTES_PER_CYCLE,
                            active_cores * L2_TO_LOCAL_BYTES_PER_CYCLE)
    aggregate_hbm_cycles = unique_hbm_bytes / effective_hbm_rate
    aggregate_l2_cycles = total_l2_bytes / effective_l2_rate
    worst_index = max(range(len(core_costs)), key=core_costs.__getitem__)
    worst = core_breakdowns[worst_index]
    critical_cycles = (max(worst["total_cycles"], aggregate_hbm_cycles,
                           aggregate_l2_cycles) +
                       GM_TO_L1_SYNC_CYCLES + KERNEL_LAUNCH_CYCLES)
    dominant_local = max(
        ("input_l2_to_l1", "cube_l1_overlap", "output_accumulation", "vector_cast"),
        key=lambda name: worst[f"{name}_cycles"],
    )
    return {
        "critical_cycles": critical_cycles,
        "critical_path": (
            "aggregate_hbm" if aggregate_hbm_cycles == max(
                worst["total_cycles"], aggregate_hbm_cycles, aggregate_l2_cycles)
            else "aggregate_l2" if aggregate_l2_cycles > worst["total_cycles"]
            else f"worst_core:{dominant_local}"
        ),
        "worst_core_cycles": worst["total_cycles"],
        "worst_core_components": worst,
        "aggregate_hbm_cycles": aggregate_hbm_cycles,
        "aggregate_l2_cycles": aggregate_l2_cycles,
        "unique_hbm_bytes": unique_hbm_bytes,
        "total_l2_bytes": total_l2_bytes,
        "total_matmul_iterations": total_matmul_iterations,
        "active_cores": active_cores,
        "effective_hbm_bytes_per_cycle": effective_hbm_rate,
        "effective_l2_bytes_per_cycle": effective_l2_rate,
        "worst_core_matmul_iterations": max(
            item["matmul_iterations"] for item in core_breakdowns
        ),
        "useful_macs": total_useful_macs,
        "padded_macs": total_padded_macs,
        "padding_ratio": total_padded_macs / max(1, total_useful_macs),
        "k_chunks": k_chunks,
    }


def gm_to_l1_cube(req: dict) -> tuple[dict, dict]:
    width = 2
    candidates = []
    seen = set()
    for base_m in (256, 128, 64, 32):
        for base_n in (256, 128, 64, 32):
            for base_k in (128, 64, 32):
                if (base_m * base_k * width * 2 > L0A or
                        base_n * base_k * width * 2 > L0B or
                        base_m * base_n * 4 > L0C):
                    continue
                resident_per_step = (base_m + base_n) * base_k * width
                max_step = (L1 - L1_QUEUE_RESERVE) // resident_per_step
                if max_step < 1:
                    continue
                step_k = min(max_step, ceil_div(req["K"], base_k))
                k_stripe = step_k * base_k
                m_tiles = ceil_div(req["M"], base_m)
                n_tiles = ceil_div(req["N"], base_n)
                for m_groups in range(1, min(m_tiles, AIC) + 1):
                    for n_groups in range(1, min(n_tiles, AIC // m_groups) + 1):
                        single_m, m_extents = partition_by_base_tiles(
                            req["M"], base_m, m_groups)
                        single_n, n_extents = partition_by_base_tiles(
                            req["N"], base_n, n_groups)
                        used = len(m_extents) * len(n_extents)
                        if used > AIC:
                            continue
                        signature = (base_m, base_n, base_k, k_stripe, single_m, single_n, used)
                        if signature in seen:
                            continue
                        seen.add(signature)
                        cost = gm_to_l1_cost(
                            req, base_m=base_m, base_n=base_n,
                            base_k=base_k, k_stripe=k_stripe,
                            m_extents=m_extents, n_extents=n_extents)
                        # Every term before used is a hardware service term;
                        # core count is only a final tie-break, never the goal.
                        score = (
                            cost["critical_cycles"], cost["worst_core_matmul_iterations"],
                            cost["aggregate_hbm_cycles"], cost["total_l2_bytes"],
                            cost["aggregate_l2_cycles"], cost["padding_ratio"], -used,
                        )
                        candidates.append((
                            score, base_m, base_n, base_k, k_stripe,
                            single_m, single_n, used, cost,
                        ))
    if not candidates:
        raise ValueError("no legal GM-to-L1 finite candidate")

    def pareto_metrics(candidate: tuple) -> tuple[float, ...]:
        cost = candidate[8]
        return (
            cost["critical_cycles"], cost["worst_core_cycles"],
            cost["aggregate_hbm_cycles"], cost["aggregate_l2_cycles"],
            float(cost["worst_core_matmul_iterations"]), cost["padding_ratio"],
            float(cost["total_l2_bytes"]),
        )

    pareto_candidates = []
    for candidate in candidates:
        metrics = pareto_metrics(candidate)
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            other_metrics = pareto_metrics(other)
            if (all(left <= right for left, right in zip(other_metrics, metrics)) and
                    any(left < right for left, right in zip(other_metrics, metrics))):
                dominated = True
                break
        if not dominated:
            pareto_candidates.append(candidate)
    if not pareto_candidates:
        raise RuntimeError("GM-to-L1 Pareto pruning removed every legal candidate")
    (score, base_m, base_n, base_k, k_stripe,
     single_m, single_n, used, cost) = min(pareto_candidates)
    step_k = k_stripe // base_k
    cube = empty_cube(req)
    cube.update({
        "usedCoreNum": used,
        "singleCoreM": single_m, "singleCoreN": single_n,
        "singleCoreK": k_stripe,
        "baseM": base_m, "baseN": base_n, "baseK": base_k,
        "depthA1": step_k, "depthB1": step_k,
        "stepM": 1, "stepN": 1,
        "stepKa": step_k, "stepKb": step_k,
        "dbL0A": 2, "dbL0B": 2, "dbL0C": 1,
    })
    return cube, {
        "model": "gm_to_l1_l0c_resident_critical_path_v2",
        "finite_candidate_count": len(candidates),
        "pareto_candidate_count": len(pareto_candidates),
        "candidate_score": list(score),
        "critical_path_cycles": cost["critical_cycles"],
        "critical_path": cost["critical_path"],
        "worst_core_cycles": cost["worst_core_cycles"],
        "worst_core_components": cost["worst_core_components"],
        "aggregate_hbm_cycles": cost["aggregate_hbm_cycles"],
        "aggregate_l2_cycles": cost["aggregate_l2_cycles"],
        "unique_hbm_bytes": cost["unique_hbm_bytes"],
        "total_l2_bytes": cost["total_l2_bytes"],
        "total_matmul_iterations": cost["total_matmul_iterations"],
        "active_cores": cost["active_cores"],
        "effective_hbm_bytes_per_cycle": cost["effective_hbm_bytes_per_cycle"],
        "effective_l2_bytes_per_cycle": cost["effective_l2_bytes_per_cycle"],
        "worst_core_matmul_iterations": cost["worst_core_matmul_iterations"],
        "padding_ratio": cost["padding_ratio"],
        "k_chunks": cost["k_chunks"],
        "core_grid_tasks": used,
        "base_tile": [base_m, base_n, base_k],
        "k_stripe": k_stripe,
        "protocol": "l0c_resident_k_partial_sum_single_output_flush",
        "critical_path_equation": "max(worst_core,aggregate_hbm,aggregate_l2)+sync+launch",
        "worst_core_equation": "input_l2_to_l1+max(cube,l1_to_l0)+output_flush+vector_cast",
        "primitive_calibration": "frozen_c220_isolated_instruction_contract_v1",
        "service_rates": {
            "l2_to_local_bytes_per_cycle": L2_TO_LOCAL_BYTES_PER_CYCLE,
            "l1_to_l0_bytes_per_cycle": L1_TO_L0_BYTES_PER_CYCLE,
            "cube_macs_per_cycle": CUBE_MACS_PER_CYCLE_16BIT,
            "fixpipe_bytes_per_cycle": FIXPIPE_BYTES_PER_CYCLE,
            "aggregate_hbm_bytes_per_cycle": AGGREGATE_HBM_BYTES_PER_CYCLE,
            "aggregate_l2_bytes_per_cycle": AGGREGATE_L2_BYTES_PER_CYCLE,
        },
    }


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
    gm_to_l1 = family in (
        "SINGLE_CORE_SPLIT_K_GM_TO_L1",
        "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
    )
    resident_m = cube["baseM"] if gm_to_l1 else cube["singleCoreM"]
    resident_n = cube["baseN"] if gm_to_l1 else cube["singleCoreN"]
    resident_a = align_up(resident_m, 16) * cube["singleCoreK"] * width
    resident_b = (align_up(cube["singleCoreK"], 16) *
                  align_up(resident_n, 16) * width)
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
            not gm_to_l1 or resident_a + resident_b <= L1 - L1_QUEUE_RESERVE
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
        cube, derivation = gm_to_l1_cube(req)
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
            "resident_a_l1_bytes": align_up(cube["baseM"], 16) * cube["singleCoreK"] * width,
            "resident_b_l1_bytes": (align_up(cube["singleCoreK"], 16) *
                                     align_up(cube["baseN"], 16) * width),
            "resident_l1_limit_bytes": L1 - L1_QUEUE_RESERVE,
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
    # The custom family pads each base tile directly during GM->L1 ND2NZ;
    # it does not materialize whole converted A/B tensors in workspace.
    conversion_workspace = 0
    workspace = SYSTEM_WORKSPACE + output_workspace + conversion_workspace
    suffix = FAMILIES[required_family][dtype]
    return {
        "status": "EXPERIMENTAL_FAMILY_TILING",
        "npu_eligible": True,
        "formula_family": required_family,
        "kernel_suffix": suffix,
        "selection_basis": "INDEPENDENT_FINITE_CANDIDATE_CRITICAL_PATH",
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
