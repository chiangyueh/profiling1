#!/usr/bin/env python3
"""Select exactly one bounded, parameter-only MatMul tiling per workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npu_cost_model import (  # noqa: E402
    MemorySpace,
    Resource,
    TilingPlan,
    ascend_910b3,
    cann81_matmul_effective_l1_bytes,
    execution_mode_name,
    lower_plan_to_cann,
    plan_from_cann,
    simulate,
    source_kernel_suffix,
    validate_cann_tiling,
)
from npu_cost_model.operators import matmul  # noqa: E402


MAX_EVALUATIONS = 32
BASE_STAGE_LIMIT = 18
GRAPH_STAGE_LIMIT = 24
GEOMETRY_LIMIT = 6
CORE_EXPANSION_SEEDS = 4
KNOWLEDGE_FIELDS = (
    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
    "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
    "stepN", "iterateOrder", "stepKa", "stepKb", "dbL0A", "dbL0B",
    "dbL0C", "l2MTileCnt", "l2NTileCnt", "l2MTileBlock",
    "l2NTileBlock", "l2IterateOrder", "tilingEnable",
)
KNOWLEDGE_COLUMNS = {
    "usedCoreNum": "used_core_num",
    "singleCoreM": "single_core_m",
    "singleCoreN": "single_core_n",
    "singleCoreK": "single_core_k",
    "baseM": "base_m",
    "baseN": "base_n",
    "baseK": "base_k",
    "depthA1": "depth_a1",
    "depthB1": "depth_b1",
    "stepM": "step_m",
    "stepN": "step_n",
    "iterateOrder": "iterate_order",
    "stepKa": "step_ka",
    "stepKb": "step_kb",
    "dbL0A": "db_l0a",
    "dbL0B": "db_l0b",
    "dbL0C": "db_l0c",
    "l2MTileCnt": "bank_l2_m_tile_count",
    "l2NTileCnt": "bank_l2_n_tile_count",
    "l2MTileBlock": "bank_l2_m_tile_block",
    "l2NTileBlock": "bank_l2_n_tile_block",
    "l2IterateOrder": "bank_l2_iterate_order",
    "tilingEnable": "bank_tiling_enable",
}
OUTPUT_FIELDS = (
    "rank", "workload_id", "m", "n", "k", "dtype", "trans_a",
    "trans_b", "max_cores", "source", "candidate_role", "valid",
    "error", "execution_mode", *KNOWLEDGE_COLUMNS.values(),
    "new_model_cycles", "new_model_rank", "new_model_bottleneck",
    "new_model_breakdown", "model_schedule_sha256",
    "model_kernel_suffix", "model_kernel_family", "model_input_source",
    "tiling_parameter_origin", "selection_basis", "tiling_signature",
    "is_reserve", "required_successful_tilings", "candidate_budget",
    "generated_candidate_count", "legal_candidate_count",
    "candidate_generation_ms", "static_legality_ms",
    "simulator_scoring_ms", "tiling_solver_total_ms",
)


@dataclass(frozen=True)
class Workload:
    workload_id: str
    m: int
    n: int
    k: int
    dtype: str
    trans_a: bool
    trans_b: bool
    max_cores: int


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def read_workloads(path: Path) -> list[Workload]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    workloads = [
        Workload(
            workload_id=row["workload_id"],
            m=int(row["m"]),
            n=int(row["n"]),
            k=int(row["k"]),
            dtype=row["dtype"].lower(),
            trans_a=truthy(row.get("trans_a")),
            trans_b=truthy(row.get("trans_b")),
            max_cores=int(row.get("max_cores") or 20),
        )
        for row in rows
    ]
    if not workloads or len({item.workload_id for item in workloads}) != len(workloads):
        raise ValueError("workload catalog is empty or contains duplicate identities")
    return workloads


def build_hardware(args: argparse.Namespace):
    base = ascend_910b3()
    core_counts = dict(base.core_counts)
    core_counts[Resource.CUBE] = args.aic_cores
    capacities = dict(base.capacities)
    capacities.update({
        MemorySpace.L0A: args.l0a_bytes,
        MemorySpace.L0B: args.l0b_bytes,
        MemorySpace.L0C: args.l0c_bytes,
        MemorySpace.L1: cann81_matmul_effective_l1_bytes(args.l1_bytes),
        MemorySpace.L2: args.l2_bytes,
    })
    return replace(
        base,
        core_counts=core_counts,
        capacities=capacities,
        aggregate_hbm_bytes_per_cycle=(
            args.hbm_bytes_per_cycle_per_core * args.aic_cores
        ),
        aggregate_l2_bytes_per_cycle=(
            args.l2_bytes_per_cycle_per_core * args.aic_cores
        ),
    )


def knowledge_signature(knowledge: dict[str, int]) -> tuple[int, ...]:
    return tuple(int(knowledge[field]) for field in KNOWLEDGE_FIELDS)


def model_schedule_sha(workload: Workload, knowledge: dict[str, int]) -> str:
    payload = {
        "shape": [workload.m, workload.n, workload.k],
        "dtype": workload.dtype,
        "trans_a": workload.trans_a,
        "trans_b": workload.trans_b,
        "knowledge": [knowledge[field] for field in KNOWLEDGE_FIELDS],
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def geometry_priority(
    workload: Workload,
    base_m: int,
    base_n: int,
    output_capacity: int,
) -> tuple[float, ...]:
    m_tiles = ceil_div(workload.m, base_m)
    n_tiles = ceil_div(workload.n, base_n)
    padded = m_tiles * base_m * n_tiles * base_n
    useful = workload.m * workload.n
    tasks = m_tiles * n_tiles
    cores = max(1, min(workload.max_cores, tasks))
    waves = ceil_div(tasks, cores)
    wave_slots = waves * cores
    return (
        -(base_m * base_n) / output_capacity,
        -useful / padded,
        -tasks / wave_slots,
        abs(math.log2(base_m / base_n)),
        float(base_m),
        float(base_n),
    )


def frontier_geometries(workload: Workload, hardware) -> list[tuple[int, int]]:
    alignment = 16
    output_capacity = hardware.capacities[MemorySpace.L0C] // 4
    minimum_axis = 4 * alignment
    maximum_axis = min(512, align_down(output_capacity // minimum_axis, alignment))
    axis_values = range(minimum_axis, maximum_axis + alignment, alignment)
    feasible = [
        (base_m, base_n)
        for base_m in axis_values
        for base_n in axis_values
        if output_capacity // 2 <= base_m * base_n <= output_capacity
    ]
    boundary: set[tuple[int, int]] = set()
    for base_m in axis_values:
        values = [pair for pair in feasible if pair[0] == base_m]
        boundary.update(sorted(values, key=lambda pair: pair[1], reverse=True)[:2])
    for base_n in axis_values:
        values = [pair for pair in feasible if pair[1] == base_n]
        boundary.update(sorted(values, key=lambda pair: pair[0], reverse=True)[:2])
    boundary.update(sorted(
        feasible,
        key=lambda pair: (
            abs(math.log2(pair[0] / pair[1])),
            -(pair[0] * pair[1]),
        ),
    )[:4])
    ranked = sorted(
        boundary,
        key=lambda pair: geometry_priority(
            workload, pair[0], pair[1], output_capacity
        ),
    )
    return ranked[:GEOMETRY_LIMIT]


def cache_tiles(workload: Workload, base_m: int, base_n: int, l2_bytes: int):
    input_bytes = 4 if workload.dtype == "fp32" else 2
    output_bytes = input_bytes
    cache_m = workload.m
    cache_n = workload.n

    def footprint() -> int:
        return input_bytes * (
            cache_m * workload.k + workload.k * cache_n
        ) + output_bytes * cache_m * cache_n

    while footprint() > l2_bytes and (cache_m > base_m or cache_n > base_n):
        if cache_m / base_m >= cache_n / base_n and cache_m > base_m:
            cache_m = max(base_m, align_down(ceil_div(cache_m, 2), base_m))
        elif cache_n > base_n:
            cache_n = max(base_n, align_down(ceil_div(cache_n, 2), base_n))
        else:
            break
    return (("m", cache_m), ("n", cache_n), ("k", workload.k))


def make_base_plan(
    workload: Workload,
    hardware,
    base_m: int,
    base_n: int,
    base_k: int,
    traversal: tuple[str, str],
    used_cores: int | None = None,
) -> TilingPlan:
    tasks = ceil_div(workload.m, base_m) * ceil_div(workload.n, base_n)
    core_limit = min(workload.max_cores, hardware.core_count(Resource.CUBE))
    return TilingPlan(
        algorithm=0,
        axis_tiles=(("m", base_m), ("n", base_n), ("k", base_k)),
        task_tiles=(("m", base_m), ("n", base_n), ("k", workload.k)),
        cache_tiles=cache_tiles(
            workload, base_m, base_n, hardware.capacities[MemorySpace.L2]
        ),
        used_cores=min(tasks, core_limit) if used_cores is None else used_cores,
        reduction_parts=(("k", 1),),
        buffers=(
            (MemorySpace.L1, 2),
            (MemorySpace.L0A, 2),
            (MemorySpace.L0B, 2),
            (MemorySpace.L0C, 1),
        ),
        traversal=traversal,
    )


def lower_candidate(workload: Workload, hardware, plan: TilingPlan):
    knowledge = lower_plan_to_cann(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        workload.trans_a,
        workload.trans_b,
        plan,
        hardware,
    )
    violations = validate_cann_tiling(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        workload.trans_a,
        workload.trans_b,
        knowledge,
        hardware,
    )
    if violations:
        raise ValueError(";".join(violations))
    return knowledge


def base_proposals(workload: Workload, hardware) -> list[dict[str, int]]:
    proposals: dict[tuple[int, ...], dict[str, int]] = {}
    input_bytes = 4 if workload.dtype == "fp32" else 2
    k_alignment = 8 if (
        workload.dtype == "fp32" and not workload.trans_a and workload.trans_b
    ) else 16
    specifications = []
    for base_m, base_n in frontier_geometries(workload, hardware):
        maximum_k = min(
            hardware.capacities[MemorySpace.L0A] // (2 * input_bytes * base_m),
            hardware.capacities[MemorySpace.L0B] // (2 * input_bytes * base_n),
            workload.k,
        )
        maximum_k = align_down(maximum_k, k_alignment)
        k_values = [
            value for value in (256, 128, 64, 32, 16, 8)
            if value <= maximum_k and value % k_alignment == 0
        ][:3]
        for base_k in k_values:
            specifications.append((base_m, base_n, base_k, ("m", "n")))

    for base_m, base_n, base_k, traversal in specifications:
        if len(proposals) >= BASE_STAGE_LIMIT:
            break
        try:
            knowledge = lower_candidate(
                workload,
                hardware,
                make_base_plan(
                    workload, hardware, base_m, base_n, base_k, traversal
                ),
            )
        except (KeyError, ValueError):
            continue
        proposals.setdefault(knowledge_signature(knowledge), knowledge)
    return list(proposals.values())


def score_pool(workload: Workload, hardware, proposed):
    operator = matmul(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        trans_a=workload.trans_a,
        trans_b=workload.trans_b,
    )
    legal = []
    legality_ns = 0
    scoring_ns = 0
    for knowledge in proposed:
        started = time.perf_counter_ns()
        violations = validate_cann_tiling(
            workload.m,
            workload.n,
            workload.k,
            workload.dtype,
            workload.trans_a,
            workload.trans_b,
            knowledge,
            hardware,
        )
        legality_ns += time.perf_counter_ns() - started
        if violations:
            continue
        try:
            plan = plan_from_cann(
                workload.m,
                workload.n,
                workload.k,
                knowledge,
                dtype=workload.dtype,
                trans_a=workload.trans_a,
                trans_b=workload.trans_b,
            )
            started = time.perf_counter_ns()
            result = simulate(operator, plan, hardware)
            scoring_ns += time.perf_counter_ns() - started
        except (KeyError, ValueError):
            continue
        if result.valid:
            legal.append({
                "knowledge": knowledge,
                "plan": plan,
                "simulation": result,
                "family": execution_mode_name(knowledge),
            })
    legal.sort(key=lambda item: (
        item["simulation"].total_cycles,
        knowledge_signature(item["knowledge"]),
    ))
    return legal, legality_ns, scoring_ns


def merge_scored(*pools):
    unique = {}
    for pool in pools:
        for item in pool:
            unique.setdefault(
                knowledge_signature(item["knowledge"]), item
            )
    result = list(unique.values())
    result.sort(key=lambda item: (
        item["simulation"].total_cycles,
        knowledge_signature(item["knowledge"]),
    ))
    return result


def add_execution_graphs(workload: Workload, hardware, pool, proposed) -> None:
    if not pool:
        return
    operator = matmul(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        trans_a=workload.trans_a,
        trans_b=workload.trans_b,
    )
    signatures = {knowledge_signature(value) for value in proposed}
    for algorithm, graph in enumerate(operator.algorithms):
        if graph.name == "base":
            continue
        for item in pool[:1]:
            if len(proposed) >= GRAPH_STAGE_LIMIT:
                return
            prototype = item["plan"]
            parts = min(
                min(workload.max_cores, hardware.core_count(Resource.CUBE)),
                max(2, ceil_div(workload.k, max(1, 3 * prototype.tiles["k"]))),
            )
            plan = replace(
                prototype,
                algorithm=algorithm,
                reduction_parts=(("k", parts),),
            )
            try:
                knowledge = lower_candidate(workload, hardware, plan)
            except (KeyError, ValueError):
                continue
            signature = knowledge_signature(knowledge)
            if signature not in signatures:
                signatures.add(signature)
                proposed.append(knowledge)


def add_core_frontier(workload: Workload, hardware, pool, proposed) -> None:
    signatures = {knowledge_signature(value) for value in proposed}
    for item in pool[:CORE_EXPANSION_SEEDS]:
        if len(proposed) >= MAX_EVALUATIONS:
            return
        knowledge = item["knowledge"]
        tasks = (
            ceil_div(workload.m, knowledge["singleCoreM"])
            * ceil_div(workload.n, knowledge["singleCoreN"])
        )
        maximum = min(
            tasks, workload.max_cores, hardware.core_count(Resource.CUBE)
        )
        waves = ceil_div(tasks, maximum)
        filled = ceil_div(tasks, waves)
        for cores in sorted({
            maximum,
            max(1, filled - 1),
            filled,
            min(maximum, filled + 1),
        }):
            if len(proposed) >= MAX_EVALUATIONS:
                return
            candidate = dict(knowledge)
            candidate["usedCoreNum"] = cores
            try:
                violations = validate_cann_tiling(
                    workload.m,
                    workload.n,
                    workload.k,
                    workload.dtype,
                    workload.trans_a,
                    workload.trans_b,
                    candidate,
                    hardware,
                )
            except (KeyError, ValueError):
                continue
            signature = knowledge_signature(candidate)
            if not violations and signature not in signatures:
                signatures.add(signature)
                proposed.append(candidate)


def breakdown(result) -> str:
    return json.dumps({
        "critical_core_cycles": result.critical_core_cycles,
        "hbm_cycles": result.hbm_cycles,
        "l2_cycles": result.l2_cycles,
        "shared_resource_cycles": result.shared_resource_cycles,
        "active_cores": result.active_cores,
        "gm_read_bytes": result.gm_read_bytes,
        "gm_write_bytes": result.gm_write_bytes,
        "l2_bytes": result.l2_bytes,
        "workspace_bytes": result.workspace_bytes,
        "resources": {
            resource.value: cycles for resource, cycles in result.resource_cycles
        },
    }, separators=(",", ":"), sort_keys=True)


def output_row(
    workload: Workload,
    item,
    model_rank: int,
    generated_count: int,
    legal_count: int,
    generation_ms: float,
    legality_ms: float,
    scoring_ms: float,
    total_ms: float,
) -> dict[str, str]:
    knowledge = item["knowledge"]
    simulation = item["simulation"]
    suffix = source_kernel_suffix(
        workload.m,
        workload.n,
        workload.k,
        workload.dtype,
        workload.trans_a,
        workload.trans_b,
        knowledge,
    )
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update({
        "rank": str(model_rank),
        "workload_id": workload.workload_id,
        "m": str(workload.m),
        "n": str(workload.n),
        "k": str(workload.k),
        "dtype": workload.dtype,
        "trans_a": str(int(workload.trans_a)),
        "trans_b": str(int(workload.trans_b)),
        "max_cores": str(workload.max_cores),
        "source": "independent_hardware_cost_model",
        "candidate_role": "searched",
        "valid": "1",
        "execution_mode": item["family"],
        "new_model_cycles": f"{simulation.total_cycles:.12g}",
        "new_model_rank": str(model_rank),
        "new_model_bottleneck": simulation.bottleneck,
        "new_model_breakdown": breakdown(simulation),
        "model_schedule_sha256": model_schedule_sha(workload, knowledge),
        "model_kernel_suffix": str(suffix),
        "model_kernel_family": item["family"].upper(),
        "model_input_source": "shape_hardware_and_cost_model_only",
        "tiling_parameter_origin": "independent_model_generation",
        "selection_basis": "minimum_simulated_cycles_in_bounded_hardware_region",
        "tiling_signature": ":".join(str(value) for value in knowledge_signature(knowledge)),
        "is_reserve": "0",
        "required_successful_tilings": "1",
        "candidate_budget": str(MAX_EVALUATIONS),
        "generated_candidate_count": str(generated_count),
        "legal_candidate_count": str(legal_count),
        "candidate_generation_ms": f"{generation_ms:.9g}",
        "static_legality_ms": f"{legality_ms:.9g}",
        "simulator_scoring_ms": f"{scoring_ms:.9g}",
        "tiling_solver_total_ms": f"{total_ms:.9g}",
    })
    for name, column in KNOWLEDGE_COLUMNS.items():
        row[column] = str(knowledge[name])
    return row


def append_audit(stream, record: dict) -> None:
    stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l0a-bytes", type=int, required=True)
    parser.add_argument("--l0b-bytes", type=int, required=True)
    parser.add_argument("--l0c-bytes", type=int, required=True)
    parser.add_argument("--l1-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    parser.add_argument("--l2-bytes-per-cycle-per-core", type=float, required=True)
    parser.add_argument("--hbm-bytes-per-cycle-per-core", type=float, required=True)
    args = parser.parse_args()

    workloads = read_workloads(args.workloads)
    hardware = build_hardware(args)
    selected_rows = []
    scored_rows = []
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8") as audit:
        for index, workload in enumerate(workloads, 1):
            shape_started = time.perf_counter_ns()
            generation_started = time.perf_counter_ns()
            proposed = base_proposals(workload, hardware)
            base_generation_ns = time.perf_counter_ns() - generation_started
            pool, legality_ns, scoring_ns = score_pool(workload, hardware, proposed)
            graph_start = len(proposed)
            add_execution_graphs(workload, hardware, pool, proposed)
            graph_pool, legality_more_ns, scoring_more_ns = score_pool(
                workload, hardware, proposed[graph_start:]
            )
            pool = merge_scored(pool, graph_pool)
            legality_ns += legality_more_ns
            scoring_ns += scoring_more_ns
            core_start = len(proposed)
            add_core_frontier(workload, hardware, pool, proposed)
            if len(proposed) > MAX_EVALUATIONS:
                raise RuntimeError("candidate evaluation budget was exceeded")
            core_pool, legality_final_ns, scoring_final_ns = score_pool(
                workload, hardware, proposed[core_start:]
            )
            pool = merge_scored(pool, core_pool)
            legality_ns += legality_final_ns
            scoring_ns += scoring_final_ns
            if not pool:
                raise RuntimeError(f"{workload.workload_id}: no executable model tiling")
            total_ms = (time.perf_counter_ns() - shape_started) / 1e6
            generation_ms = base_generation_ns / 1e6
            legality_ms = legality_ns / 1e6
            scoring_ms = scoring_ns / 1e6
            rows = [
                output_row(
                    workload,
                    item,
                    rank,
                    len(proposed),
                    len(pool),
                    generation_ms,
                    legality_ms,
                    scoring_ms,
                    total_ms,
                )
                for rank, item in enumerate(pool, 1)
            ]
            selected_rows.append(rows[0])
            scored_rows.extend(rows)
            selected_knowledge = pool[0]["knowledge"]
            append_audit(audit, {
                "schema": "matmul_deployment_selection_v1",
                "record_type": "model_top1_frozen",
                "workload_id": workload.workload_id,
                "model_inputs": {
                    "shape": [workload.m, workload.n, workload.k],
                    "dtype": workload.dtype,
                    "trans_a": workload.trans_a,
                    "trans_b": workload.trans_b,
                    "hardware": {
                        "soc": args.soc,
                        "aic_cores": args.aic_cores,
                        "l0a_bytes": args.l0a_bytes,
                        "l0b_bytes": args.l0b_bytes,
                        "l0c_bytes": args.l0c_bytes,
                        "l1_bytes": args.l1_bytes,
                        "l2_bytes": args.l2_bytes,
                        "l2_bytes_per_cycle_per_core": args.l2_bytes_per_cycle_per_core,
                        "hbm_bytes_per_cycle_per_core": args.hbm_bytes_per_cycle_per_core,
                    },
                },
                "baseline_data_consumed": False,
                "measured_latency_consumed": False,
                "candidate_budget": MAX_EVALUATIONS,
                "generated_candidate_count": len(proposed),
                "legal_candidate_count": len(pool),
                "selection_ms": total_ms,
                "selected": {
                    "model_schedule_sha256": rows[0]["model_schedule_sha256"],
                    "predicted_cycles": pool[0]["simulation"].total_cycles,
                    "bottleneck": pool[0]["simulation"].bottleneck,
                    "execution_mode": pool[0]["family"],
                    "kernel_suffix": int(rows[0]["model_kernel_suffix"]),
                    "tiling": {
                        field: selected_knowledge[field] for field in KNOWLEDGE_FIELDS
                    },
                },
            })
            print(
                f"MODEL_TOP1 [{index}/{len(workloads)}] {workload.workload_id} "
                f"generated={len(proposed)} legal={len(pool)} "
                f"family={pool[0]['family']} cores={selected_knowledge['usedCoreNum']} "
                f"base={selected_knowledge['baseM']}x{selected_knowledge['baseN']}x"
                f"{selected_knowledge['baseK']} host_ms={total_ms:.3f}",
                flush=True,
            )
    write_csv(args.output, selected_rows)
    write_csv(args.all_output, scored_rows)
    print(
        "MATMUL_DEPLOYMENT_SELECTION "
        f"shapes={len(workloads)} selected={len(selected_rows)} "
        f"internal_scored={len(scored_rows)} max_per_shape={MAX_EVALUATIONS} "
        "model_inputs=shape_hardware_cost_model",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"fatal: {error}", flush=True)
        raise SystemExit(1)
