#!/usr/bin/env python3
"""Parameter-free BASE scheduling analysis for the 20-AIC C220 kernel.

This module compares two ownership schedules for an otherwise identical BASE
packet.  It does not estimate latency, fit coefficients, enumerate tilings, or
change baseM/baseN/baseK.  Every shape is decomposed into its exact full/tail
task classes and every participating core is audited on seven additive
pipeline resources.  A proposed schedule is admissible only when no resource
maximum increases, normalized multidimensional imbalance strictly falls and
aggregate issued work is identical.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from math import gcd, sqrt
import struct
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

TASK_GROUPS: Tuple[str, ...] = (
    "FULL_M_FULL_N",
    "FULL_M_TAIL_N",
    "TAIL_M_FULL_N",
    "TAIL_M_TAIL_N",
)
BALANCED_SCHEDULE_MAGIC = 0x32425342
BALANCED_SCHEDULE_VERSION = 2
BALANCED_SCHEDULE_BYTES = 32 + 20 * len(TASK_GROUPS) * 8


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


@dataclass(frozen=True)
class TaskGroup:
    name: str
    count: int
    m_tail: bool
    n_tail: bool
    resources: Dict[str, int]


def _task_groups(
    shape: BaseShape, packet: NativeBasePacket,
) -> List[TaskGroup]:
    """Return the four exact output-tile classes for an arbitrary shape."""
    m_count = ceil_div(shape.m, packet.single_core_m)
    n_count = ceil_div(shape.n, packet.single_core_n)
    has_m_tail = shape.m % packet.single_core_m != 0
    has_n_tail = shape.n % packet.single_core_n != 0
    full_m_count = m_count - int(has_m_tail)
    full_n_count = n_count - int(has_n_tail)
    representatives = (
        ("FULL_M_FULL_N", full_m_count * full_n_count, False, False,
         (0, 0)),
        ("FULL_M_TAIL_N", full_m_count * int(has_n_tail), False, True,
         (0, n_count - 1)),
        ("TAIL_M_FULL_N", int(has_m_tail) * full_n_count, True, False,
         (m_count - 1, 0)),
        ("TAIL_M_TAIL_N", int(has_m_tail and has_n_tail), True, True,
         (m_count - 1, n_count - 1)),
    )
    return [
        TaskGroup(name, count, m_tail, n_tail,
                  _task_resources(shape, packet, representative))
        for name, count, m_tail, n_tail, representative in representatives
    ]


def _ratio_float(load: Dict[str, int], totals: Dict[str, int], cores: int) -> float:
    return max(
        load[name] * cores / totals[name]
        for name in PIPELINE_RESOURCES
    )


def _assign_group_jobs(
    loads: List[Dict[str, int]], counts: List[List[int]], group_index: int,
    group: TaskGroup, totals: Dict[str, int], cores: int,
) -> None:
    """Assign one exact task class without iterating over a large tile grid.

    For a fixed task vector, the per-core normalized load after x additional
    tasks is monotone.  Bisection finds the smallest common ceiling capable of
    accepting the whole class.  Capacity-proportional integer apportionment
    then fills that ceiling with at most 19 deterministic remainder steps.
    """
    if group.count == 0:
        return
    vector = group.resources

    def capacity(core: int, ceiling: float) -> int:
        result = group.count
        for name in PIPELINE_RESOURCES:
            room = ceiling * totals[name] / cores - loads[core][name]
            result = min(result, max(0, int((room + 1.0e-9) // vector[name])))
        return result

    low = max(_ratio_float(load, totals, cores) for load in loads)
    high = max(
        _ratio_float({
            name: load[name] + group.count * vector[name]
            for name in PIPELINE_RESOURCES
        }, totals, cores)
        for load in loads
    )
    # 32 steps resolve every uint32 task count while keeping host tiling cheap.
    for _ in range(32):
        middle = (low + high) / 2.0
        if sum(capacity(core, middle) for core in range(cores)) >= group.count:
            high = middle
        else:
            low = middle
    capacities = [capacity(core, high * (1.0 + 1.0e-12) + 1.0e-12)
                  for core in range(cores)]
    capacity_total = sum(capacities)
    if capacity_total < group.count:
        raise AssertionError("water-filling ceiling cannot contain the task group")
    assigned = [group.count * value // capacity_total for value in capacities]
    while sum(assigned) < group.count:
        available = [
            core for core in range(cores) if assigned[core] < capacities[core]
        ]
        core = min(
            available,
            key=lambda index: (
                _ratio_float({
                    name: loads[index][name]
                    + (assigned[index] + 1) * vector[name]
                    for name in PIPELINE_RESOURCES
                }, totals, cores),
                index,
            ),
        )
        assigned[core] += 1
    for core, amount in enumerate(assigned):
        counts[core][group_index] += amount
        for name in PIPELINE_RESOURCES:
            loads[core][name] += amount * vector[name]


def _group_tasks(
    group: TaskGroup, start: int, count: int,
    m_count: int, n_count: int, has_m_tail: bool, has_n_tail: bool,
) -> List[Task]:
    full_m_count = m_count - int(has_m_tail)
    full_n_count = n_count - int(has_n_tail)
    tasks: List[Task] = []
    for offset in range(start, start + count):
        if group.name == "FULL_M_FULL_N":
            task = (offset // full_n_count, offset % full_n_count)
        elif group.name == "FULL_M_TAIL_N":
            task = (offset, n_count - 1)
        elif group.name == "TAIL_M_FULL_N":
            task = (m_count - 1, offset)
        elif group.name == "TAIL_M_TAIL_N":
            task = (m_count - 1, n_count - 1)
        else:
            raise AssertionError("unknown task group")
        tasks.append(task)
    return tasks


def balanced_group_ownership(
    shape: BaseShape, packet: NativeBasePacket, cores: int,
) -> Tuple[List[List[Task]], List[List[Dict[str, int]]], List[TaskGroup]]:
    """Balance every exact tile class for any positive BASE shape.

    The returned ranges are sufficient for the kernel: each core receives at
    most one contiguous range from each of the four semantic task classes.
    No tiling alternatives, measured weights or history are consulted.
    """
    m_count = ceil_div(shape.m, packet.single_core_m)
    n_count = ceil_div(shape.n, packet.single_core_n)
    has_m_tail = shape.m % packet.single_core_m != 0
    has_n_tail = shape.n % packet.single_core_n != 0
    groups = _task_groups(shape, packet)
    totals = {
        name: sum(group.count * group.resources[name] for group in groups)
        for name in PIPELINE_RESOURCES
    }
    loads = [{name: 0 for name in PIPELINE_RESOURCES} for _ in range(cores)]
    counts = [[0 for _ in TASK_GROUPS] for _ in range(cores)]
    order = sorted(
        range(len(groups)),
        key=lambda index: (
            max(Fraction(groups[index].resources[name] * cores, totals[name])
                for name in PIPELINE_RESOURCES),
            groups[index].resources["cube_mmad_fractals"],
            -index,
        ),
        reverse=True,
    )
    for group_index in order:
        _assign_group_jobs(
            loads, counts, group_index, groups[group_index], totals, cores)

    ranges: List[List[Dict[str, int]]] = [
        [{"start": 0, "count": 0} for _ in TASK_GROUPS]
        for _ in range(cores)
    ]
    owned: List[List[Task]] = [[] for _ in range(cores)]
    for group_index, group in enumerate(groups):
        start = 0
        for core in range(cores):
            count = counts[core][group_index]
            ranges[core][group_index] = {"start": start, "count": count}
            owned[core].extend(
                _group_tasks(
                    group, start, count, m_count, n_count,
                    has_m_tail, has_n_tail,
                ))
            start += count
        if start != group.count:
            raise AssertionError("task-group ranges do not cover the group")
    _validate_ownership(owned, m_count, n_count)
    if _core_resources(shape, packet, owned) != loads:
        raise AssertionError("range reconstruction changed per-core resources")
    return owned, ranges, groups


def pack_schedule_extension(analysis: dict) -> bytes:
    """Serialize the audited four-class schedule consumed by suffix 901."""
    grid = analysis["grid"]
    ranges = analysis["schedule_ranges"]
    if len(ranges) != 20 or any(len(core) != len(TASK_GROUPS) for core in ranges):
        raise ValueError("balanced schedule must contain 20 cores and four groups")
    values = [
        BALANCED_SCHEDULE_MAGIC,
        BALANCED_SCHEDULE_VERSION,
        20,
        len(TASK_GROUPS),
        int(grid["tasks"]),
        int(grid["m_count"]),
        int(grid["n_count"]),
        0,
    ]
    for core_ranges in ranges:
        for item in core_ranges:
            values.extend((int(item["start"]), int(item["count"])))
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("balanced schedule exceeds the uint32 direct ABI")
    raw = struct.pack("<" + "I" * len(values), *values)
    if len(raw) != BALANCED_SCHEDULE_BYTES:
        raise AssertionError("balanced schedule extension has an unexpected size")
    return raw


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


def _balance_metrics(
    vectors: Sequence[Dict[str, int]], owned: Sequence[Sequence[Task]],
    groups: Sequence[TaskGroup],
) -> dict:
    participating = [
        index for index, tasks in enumerate(owned) if len(tasks) > 0
    ]
    if not participating:
        raise AssertionError("a positive shape must have a participating core")
    resources = {}
    global_worst = 1.0
    lower_bound = 1.0
    for name in PIPELINE_RESOURCES:
        values = [vectors[index][name] for index in participating]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        maximum = max(values)
        individual = max(
            (group.resources[name] for group in groups if group.count),
            default=0,
        )
        quantum = 0
        for group in groups:
            if group.count:
                quantum = gcd(quantum, group.resources[name])
        quantized_average = ceil_div(sum(values), len(values) * quantum) * quantum
        ratio = maximum / mean
        resource_lower = max(mean, individual, quantized_average) / mean
        global_worst = max(global_worst, ratio)
        lower_bound = max(lower_bound, resource_lower)
        resources[name] = {
            "min": min(values),
            "mean": mean,
            "max": maximum,
            "max_over_mean": ratio,
            "coefficient_of_variation": sqrt(variance) / mean,
            "indivisible_task_lower_bound": resource_lower,
        }
    return {
        "participating_cores": participating,
        "participating_core_count": len(participating),
        "launched_core_count": len(vectors),
        "per_core_task_count": [len(tasks) for tasks in owned],
        "per_core_resources": [dict(vector) for vector in vectors],
        "resources": resources,
        "worst_normalized_max_over_mean": global_worst,
        "indivisibility_lower_bound": lower_bound,
        "gap_above_lower_bound": global_worst - lower_bound,
        "exactly_equal": all(
            len({vectors[index][name] for index in participating}) == 1
            for name in PIPELINE_RESOURCES
        ),
    }


def _locality_metrics(owned: Sequence[Sequence[Task]]) -> dict:
    per_core = []
    for tasks in owned:
        per_core.append({
            "distinct_m_tiles": len({task[0] for task in tasks}),
            "distinct_n_tiles": len({task[1] for task in tasks}),
            "same_m_transitions": sum(
                tasks[index - 1][0] == tasks[index][0]
                for index in range(1, len(tasks))
            ),
            "same_n_transitions": sum(
                tasks[index - 1][1] == tasks[index][1]
                for index in range(1, len(tasks))
            ),
        })
    return {"per_core": per_core}


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
    proposed_owned, proposed_ranges, groups = balanced_group_ownership(
        shape, packet, cores)
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
    source_balance = _balance_metrics(source_vectors, source_owned, groups)
    proposed_balance = _balance_metrics(proposed_vectors, proposed_owned, groups)
    balance_strict = (
        proposed_balance["worst_normalized_max_over_mean"]
        < source_balance["worst_normalized_max_over_mean"] - 1.0e-12
    )
    maxima_nonincreasing = all(
        proposed_max[name] <= source_max[name]
        for name in PIPELINE_RESOURCES
    )

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
        and (shape.m % packet.base_m != 0 or shape.n % packet.base_n != 0)
        and all(fast_granularity.values())
        and full_problem_bytes <= l2_bytes
        and aggregate_equal
        and maxima_nonincreasing
        and balance_strict
    )
    if eligible:
        balance_outcome = "PROPOSED_STRICTLY_BETTER_WITHOUT_RESOURCE_MAX_REGRESSION"
    elif m_count * n_count <= cores:
        balance_outcome = "INDIVISIBLE_OUTPUT_TASK_LIMIT"
    elif not (shape.m % packet.base_m or shape.n % packet.base_n):
        balance_outcome = "EQUAL_WEIGHT_TILES_NATIVE_ALREADY_BALANCED"
    elif not balance_strict:
        balance_outcome = "NATIVE_NOT_WORSE_ON_COEFFICIENT_FREE_BALANCE"
    elif not maxima_nonincreasing:
        balance_outcome = "PROPOSED_HAS_A_PIPELINE_RESOURCE_TRADEOFF"
    else:
        balance_outcome = "OUTSIDE_PROVEN_KERNEL_SCOPE"
    return {
        "model": "BASE_NATIVE_TILE_MULTIDIMENSIONAL_BALANCE_V2",
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
        "all_resource_maxima_nonincreasing": maxima_nonincreasing,
        "proposed_vectors_covered_by_source": covered,
        "source_witness_core_for_each_proposed_core": witnesses,
        "source_balance": source_balance,
        "proposed_balance": proposed_balance,
        "source_locality": _locality_metrics(source_owned),
        "proposed_locality": _locality_metrics(proposed_owned),
        "balance_certificate": {
            "outcome": balance_outcome,
            "selected_schedule": "PROPOSED" if eligible else "NATIVE",
            "all_participating_cores_evaluated": True,
            "all_seven_pipeline_resources_evaluated": True,
            "unavoidable_exact_equality_exception": (
                "indivisible output tasks can prevent exact equality"
            ),
        },
        "task_groups": [
            {
                "name": group.name,
                "count": group.count,
                "m_tail": group.m_tail,
                "n_tail": group.n_tail,
                "resources": dict(group.resources),
            }
            for group in groups
        ],
        "schedule_ranges": proposed_ranges,
        "decision": "ENABLE_NEW_SCHEDULER" if eligible else "KEEP_NATIVE_SCHEDULER",
        "reason": (
            "all resource maxima are nonincreasing and multidimensional imbalance is strictly reduced"
            if eligible else
            balance_outcome
        ),
        "selection_contract": {
            "candidate_enumeration": False,
            "latency_score": False,
            "fitted_coefficients": False,
            "history_lookup": False,
            "official_tiling_seed": False,
            "schedule_generation": "four exact task classes plus normalized per-class water filling",
            "balance_weights": False,
            "balance_normalization": "each pipeline resource divided by its all-core mean",
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
