#!/usr/bin/env python3
"""Parameter-free BASE scheduling analysis for the 20-AIC C220 kernel.

This module compares two ownership formulas for an otherwise identical BASE
packet.  It does not estimate latency, fit coefficients, enumerate tilings, or
change baseM/baseN/baseK.  A proposed schedule is admissible only when its
per-core pipeline work is covered component-by-component by the existing
schedule and the aggregate issued work is identical.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple


PIPELINE_RESOURCES: Tuple[str, ...] = (
    "task_control",
    "gm_to_l1_a_elements",
    "gm_to_l1_b_elements",
    "l1_to_l0a_fractals",
    "l1_to_l0b_fractals",
    "cube_mmad_fractals",
    "l0c_to_gm_fragments",
)


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def lcm(left: int, right: int) -> int:
    return left // gcd(left, right) * right


@dataclass(frozen=True)
class BaseShape:
    m: int
    n: int
    k: int
    dtype: str

    @property
    def input_bytes(self) -> int:
        if self.dtype not in ("fp16", "bf16"):
            raise ValueError("BASE schedule proof currently covers fp16/bf16")
        return 2


@dataclass(frozen=True)
class NativeBasePacket:
    base_m: int
    base_n: int
    base_k: int
    single_core_m: int
    single_core_n: int
    single_core_k: int
    used_cores: int
    step_ka: int
    step_kb: int
    depth_a1: int
    depth_b1: int
    db_l0c: int


def derive_native_nn_packet(shape: BaseShape, cores: int = 20) -> NativeBasePacket:
    """Derive the C220 NN BASE geometry from local-memory constraints.

    N is the contiguous output axis, so the native inner tile uses 256
    elements.  The FP32 accumulator footprint then fixes M to 128 in 128 KiB
    L0C.  Double-buffered L0B is the binding K constraint and fixes K to 64.
    All three dimensions retain the ISA preferred 32/64 repeat granularities.
    """
    if min(shape.m, shape.n, shape.k) <= 0:
        raise ValueError("matrix dimensions must be positive")
    width = shape.input_bytes
    l0a_bytes = 64 * 1024
    l0b_bytes = 64 * 1024
    l0c_bytes = 128 * 1024
    compile_l1_bytes = 512 * 1024 - 256

    base_n = 256
    base_m = l0c_bytes // (base_n * 4)
    base_k = min(
        l0a_bytes // (2 * base_m * width),
        l0b_bytes // (2 * base_n * width),
    )
    base_k = base_k // 32 * 32
    if (base_m, base_n, base_k) != (128, 256, 64):
        raise AssertionError("unexpected C220 native BASE geometry")

    # The platform compile-info reports 512 KiB minus the 256-byte RPC
    # reservation.  MatMulV3 adds that reservation back for a standalone op.
    total_l1 = compile_l1_bytes + 256
    visible_l1 = total_l1
    depth_a = total_l1 // 2 // (base_m * base_k * width)
    depth_b = total_l1 // 2 // (base_n * base_k * width)
    if (depth_a * base_m + depth_b * base_n) * base_k * width > visible_l1:
        depth_a //= 2
    step_ka = depth_a // 2
    step_kb = depth_b // 2
    if step_ka >= step_kb:
        step_ka = step_ka // step_kb * step_kb
    else:
        step_kb = step_kb // step_ka * step_ka
    depth_a = 2 * step_ka
    depth_b = 2 * step_kb
    return NativeBasePacket(
        base_m=base_m,
        base_n=base_n,
        base_k=base_k,
        single_core_m=base_m,
        single_core_n=base_n,
        single_core_k=shape.k,
        used_cores=cores,
        step_ka=step_ka,
        step_kb=step_kb,
        depth_a1=depth_a,
        depth_b1=depth_b,
        db_l0c=1,
    )


Task = Tuple[int, int]


def source_staggered_ownership(m_count: int, n_count: int, cores: int) -> List[List[Task]]:
    """Exact one-L2-window calOrder=0 mapping in MatmulBaseBlock."""
    total = m_count * n_count
    rounds = ceil_div(total, cores)
    extra = total % cores
    common = lcm(m_count, n_count)
    owned: List[List[Task]] = [[] for _ in range(cores)]
    for core in range(cores):
        real_rounds = rounds if extra == 0 or core < extra else rounds - 1
        for round_index in range(real_rounds):
            linear = core + round_index * cores
            m_index = linear % m_count
            n_index = (linear + linear // common) % n_count
            owned[core].append((m_index, n_index))
    _validate_ownership(owned, m_count, n_count)
    return owned


def flat_round_robin_ownership(m_count: int, n_count: int, cores: int) -> List[List[Task]]:
    """Closed-form row-major ownership q=core+round*cores."""
    total = m_count * n_count
    owned: List[List[Task]] = [[] for _ in range(cores)]
    for core in range(cores):
        for linear in range(core, total, cores):
            owned[core].append((linear // n_count, linear % n_count))
    _validate_ownership(owned, m_count, n_count)
    return owned


def _validate_ownership(owned: Sequence[Sequence[Task]], m_count: int, n_count: int) -> None:
    flat = [task for core_tasks in owned for task in core_tasks]
    expected = [(m_index, n_index) for m_index in range(m_count) for n_index in range(n_count)]
    if sorted(flat) != expected:
        raise AssertionError("ownership must cover every output tile exactly once")


def _task_resources(shape: BaseShape, packet: NativeBasePacket, task: Task) -> Dict[str, int]:
    m_index, n_index = task
    actual_m = min(packet.base_m, shape.m - m_index * packet.single_core_m)
    actual_n = min(packet.base_n, shape.n - n_index * packet.single_core_n)
    if actual_m <= 0 or actual_n <= 0:
        raise AssertionError("task is outside the logical output grid")

    c0 = 16
    m_fractals = ceil_div(actual_m, 16)
    n_fractals = ceil_div(actual_n, c0)
    k_fractals = ceil_div(shape.k, 16)
    aligned_m = 16 * m_fractals
    aligned_n = c0 * n_fractals
    aligned_k = 16 * k_fractals
    return {
        "task_control": 1,
        "gm_to_l1_a_elements": aligned_m * aligned_k,
        "gm_to_l1_b_elements": aligned_k * aligned_n,
        "l1_to_l0a_fractals": m_fractals * k_fractals,
        "l1_to_l0b_fractals": n_fractals * k_fractals,
        "cube_mmad_fractals": m_fractals * n_fractals * k_fractals,
        "l0c_to_gm_fragments": m_fractals * n_fractals,
    }


def _core_resources(
    shape: BaseShape,
    packet: NativeBasePacket,
    owned: Sequence[Sequence[Task]],
) -> List[Dict[str, int]]:
    result: List[Dict[str, int]] = []
    for tasks in owned:
        totals = {name: 0 for name in PIPELINE_RESOURCES}
        for task in tasks:
            values = _task_resources(shape, packet, task)
            for name in PIPELINE_RESOURCES:
                totals[name] += values[name]
        result.append(totals)
    return result


def _aggregate(vectors: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total = {name: 0 for name in PIPELINE_RESOURCES}
    for vector in vectors:
        for name in PIPELINE_RESOURCES:
            total[name] += vector[name]
    return total


def _maxima(vectors: Sequence[Dict[str, int]]) -> Dict[str, int]:
    return {name: max(vector[name] for vector in vectors) for name in PIPELINE_RESOURCES}


def _covered_by_source(
    proposed: Sequence[Dict[str, int]], source: Sequence[Dict[str, int]]
) -> Tuple[bool, List[int]]:
    """Sufficient proof for every nonnegative pipeline weighting.

    Every proposed core vector must be componentwise no larger than at least
    one source core vector.  Therefore no proposed per-core nonnegative linear
    resource cost can exceed the source schedule's maximum.
    """
    witnesses: List[int] = []
    for vector in proposed:
        witness = next(
            (
                index
                for index, old in enumerate(source)
                if all(vector[name] <= old[name] for name in PIPELINE_RESOURCES)
            ),
            -1,
        )
        witnesses.append(witness)
    return all(index >= 0 for index in witnesses), witnesses


def analyze(shape: BaseShape, cores: int = 20, l2_bytes: int = 192 * 1024 * 1024) -> dict:
    packet = derive_native_nn_packet(shape, cores)
    m_count = ceil_div(shape.m, packet.single_core_m)
    n_count = ceil_div(shape.n, packet.single_core_n)
    source_owned = source_staggered_ownership(m_count, n_count, cores)
    proposed_owned = flat_round_robin_ownership(m_count, n_count, cores)
    source_vectors = _core_resources(shape, packet, source_owned)
    proposed_vectors = _core_resources(shape, packet, proposed_owned)
    source_total = _aggregate(source_vectors)
    proposed_total = _aggregate(proposed_vectors)
    aggregate_equal = source_total == proposed_total
    covered, witnesses = _covered_by_source(proposed_vectors, source_vectors)
    source_max = _maxima(source_vectors)
    proposed_max = _maxima(proposed_vectors)
    strict = [
        name for name in PIPELINE_RESOURCES
        if proposed_max[name] < source_max[name]
    ]

    width = shape.input_bytes
    full_problem_bytes = (
        align_up(shape.m, 16) * align_up(shape.k, 16) * width
        + align_up(shape.k, 16) * align_up(shape.n, 16) * width
        + align_up(shape.m, 16) * align_up(shape.n, 16) * width
    )
    fast_granularity = {
        "M_multiple_32": packet.base_m % 32 == 0,
        "N_multiple_64": packet.base_n % 64 == 0,
        "K_multiple_32": packet.base_k % 32 == 0,
    }
    eligible = (
        cores == 20
        and m_count * n_count > cores
        and shape.m % packet.base_m != 0
        and all(fast_granularity.values())
        and full_problem_bytes <= l2_bytes
        and aggregate_equal
        and covered
        and "cube_mmad_fractals" in strict
    )
    return {
        "model": "BASE_NATIVE_TILE_FLAT_RR_OWNERSHIP_V1",
        "shape": asdict(shape),
        "native_packet": asdict(packet),
        "grid": {"m_count": m_count, "n_count": n_count, "tasks": m_count * n_count},
        "invariants": {
            "base_tile_unchanged": True,
            "l1_pipeline_unchanged": True,
            "kernel_math_unchanged": True,
            "output_tiles_exactly_once": True,
            "aggregate_issued_work_equal": aggregate_equal,
            "fast_instruction_granularity": fast_granularity,
        },
        "cache_scope": {
            "aligned_problem_bytes": full_problem_bytes,
            "l2_bytes": l2_bytes,
            "full_problem_capacity_fit": full_problem_bytes <= l2_bytes,
            "claim": "capacity guard only; no fitted cache-hit coefficient",
        },
        "source_max_per_core": source_max,
        "proposed_max_per_core": proposed_max,
        "strictly_reduced_resources": strict,
        "proposed_vectors_covered_by_source": covered,
        "source_witness_core_for_each_proposed_core": witnesses,
        "decision": "ENABLE_NEW_SCHEDULER" if eligible else "KEEP_NATIVE_SCHEDULER",
        "reason": (
            "componentwise pipeline envelope is strictly reduced without changing the tiling geometry"
            if eligible else
            "no parameter-free strict pipeline-envelope improvement was proven"
        ),
        "selection_contract": {
            "candidate_enumeration": False,
            "latency_score": False,
            "fitted_coefficients": False,
            "history_lookup": False,
            "official_tiling_seed": False,
            "closed_form_owner": "linear_task = core + round * used_cores",
        },
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    print(json.dumps(analyze(BaseShape(args.m, args.n, args.k, args.dtype)), indent=2))
