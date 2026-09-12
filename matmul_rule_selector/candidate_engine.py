"""Finite CANN 8.1 MatMulV3 family candidates and critical-path selection.

This module is the improved selector's decision layer.  It consumes only a
shape and a frozen hardware description.  It never calls the reconstructed
official selector, reads a RuntimeKb/tiling bank, or consumes measured
operator latency.  Generic schedules are lowered first; only the exact CANN
fields are then re-simulated and ranked.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys
from time import perf_counter_ns
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CORE = Path(__file__).resolve().parent / "baseline_core"
for candidate_path in (REPO_ROOT, BASELINE_CORE):
    if str(candidate_path) not in sys.path:
        sys.path.insert(0, str(candidate_path))

from matmul_reconstruction.incremental import generate as generate_incremental  # noqa: E402
from npu_cost_model.cann_matmul import (  # noqa: E402
    decode_execution_graph,
    kernel_suffix,
    lower_plan_to_cann,
    plan_from_cann,
    validate_cann_tiling,
)
from npu_cost_model.hardware import ascend_910b3  # noqa: E402
from npu_cost_model.ir import MemorySpace, Resource  # noqa: E402
from npu_cost_model.matmul_layout import source_layout_conversion  # noqa: E402
from npu_cost_model.operators import matmul  # noqa: E402
from npu_cost_model.schedule import SearchPolicy  # noqa: E402
from npu_cost_model.simulator import align_up, ceil_div, simulate  # noqa: E402
from npu_cost_model.solver import derive_ideal_region  # noqa: E402


GRAPH_TO_FAMILY = {
    "base": "BASE",
    "al1_full_load": "AL1",
    "bl1_full_load": "BL1",
    "bl1_full_load_fixpipe": "FIXPIPE_BL1",
    "bl1_full_load_vec_nz2nd": "FIXPIPE_BL1",
    "single_core_split_k": "SINGLE_CORE_SPLIT_K",
    "deterministic_split_k": "DETERMINISTIC_SPLIT_K",
}

NON_INSTALLED_FAMILIES = {
    "NONDETERMINISTIC_MULTI_CORE_SPLIT_K": "mode-4 kernel is not dispatched by installed CANN 8.1",
    "SINGLE_CORE_SPLIT_K_NKM": "mode-5 kernel is not dispatched by installed CANN 8.1",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1": "mode-6 kernel is not dispatched by installed CANN 8.1",
    "SINGLE_CORE_SPLIT_K_AL1": "combined SC+AL1 kernel is absent from installed CANN 8.1",
    "STREAM_K_AND_DP_STREAM_K": "tail K ownership needs the arch35 kernel and reducer",
    "ASW_AND_ASWT": "tail sub-block ownership needs the arch35 scheduler and kernel",
}

INSTALLED_SUFFIXES = frozenset(
    (0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201)
)
INSTALLED_SUFFIXES_BY_DTYPE = {
    "fp16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "bf16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "fp32": INSTALLED_SUFFIXES,
}


def _dtype_bytes(dtype: str) -> int:
    return {"fp16": 2, "bf16": 2, "fp32": 4}[dtype]


def _shape_request(shape: Any) -> dict[str, Any]:
    return {
        "M": int(shape.m),
        "N": int(shape.n),
        "K": int(shape.k),
        "dtype": str(shape.dtype),
        "output_dtype": str(shape.c_dtype),
        "transA": bool(shape.trans_a),
        "transB": bool(shape.trans_b),
        "bias": bool(shape.bias),
        **({"bias_dtype": str(shape.bias_dtype)} if shape.bias else {}),
    }


def _hardware_request(hardware: Any) -> dict[str, Any]:
    return {
        "aicNum": int(hardware.cores),
        "aivNum": 2 * int(hardware.cores),
        "ubSize": int(hardware.ub),
        "l1Size": int(hardware.l1_usable),
        "l2Size": int(hardware.l2),
        "l0CSize": int(hardware.l0c),
        "l0ASize": int(hardware.l0a),
        "l0BSize": int(hardware.l0b),
        "btSize": int(hardware.bt),
        "supportL0c2out": True,
        "supportL12BtBf16": False,
    }


def _conversion(shape: Any, graph_name: str):
    return source_layout_conversion(
        shape.m,
        shape.n,
        shape.k,
        shape.dtype,
        shape.trans_a,
        shape.trans_b,
        a_layout=shape.format_a,
        b_layout=shape.format_b,
        graph_name=graph_name,
    )


def _al1_applicable(shape: Any, hardware: Any) -> bool:
    if not (
        shape.dtype == "fp32"
        and not shape.trans_a
        and shape.trans_b
        and shape.m <= 16
        and 16 < shape.n <= 16 * hardware.cores
        and shape.k >= 4096
        and shape.k % (512 // shape.d) == 0
    ):
        return False
    a_bytes = align_up(shape.m, 16) * align_up(shape.k, 32 // shape.d) * shape.d
    b_bytes = 16 * 256 * shape.d * 2
    bias_bytes = 16 * _dtype_bytes(shape.bias_dtype) if shape.bias else 0
    return a_bytes + b_bytes + bias_bytes <= hardware.l1_usable


def _bl1_applicable(shape: Any, hardware: Any) -> bool:
    if not (shape.m > 16 * max(shape.k, shape.n) and shape.k <= 256):
        return False
    c0 = 32 // shape.d
    inner_a = shape.m if shape.trans_a else shape.k
    outer_a = shape.k if shape.trans_a else shape.m
    on_the_way = shape.n in (32, 64, 96, 128, 160, 192, 224, 256, 384)
    if shape.trans_b:
        on_the_way = on_the_way and shape.k * shape.d in (
            32, 64, 96, 128, 160, 192, 224, 256, 384
        )
    conversion_b = _conversion(shape, "bl1_full_load").b
    aligned_bl1 = on_the_way and not conversion_b
    vnchw_a = (
        shape.dtype == "fp32"
        and outer_a >= 72_368
        and 1 < inner_a <= c0
    )
    bias_bytes = 256 * 4 if shape.bias else 0
    resident_b = shape.k * shape.n * shape.d
    return aligned_bl1 or (
        vnchw_a and resident_b + bias_bytes < hardware.l1_usable // 2
    )


def _fixpipe_applicable(shape: Any, hardware: Any) -> bool:
    output_alignment = 256 // shape.dc
    c0 = 32 // shape.d
    applicable = (
        shape.n < 256
        and shape.k <= 256
        and shape.n % output_alignment != 0
        and output_alignment % shape.n != 0
        and shape.m >= hardware.cores * 512
        and not (shape.n < c0 and shape.k < c0)
    )
    if shape.dtype in ("fp16", "bf16"):
        applicable = applicable and not shape.trans_a and (
            shape.k % (256 // shape.d) == 0
            or shape.k < 256 // shape.d
        )
    return applicable


def _l2_cache_enabled(shape: Any, hardware: Any, base_m: int, base_n: int) -> bool:
    """Literal CANN 8.1 CalcTile enable flag used by the SC support gate."""

    total = (
        shape.m * shape.k * shape.d
        + shape.k * shape.n * shape.d
        + shape.m * shape.n * shape.dc
    )
    budget = hardware.l2 * 100 * 1024 * 1024 // (192 * 1024 * 1024)
    if total <= budget:
        return False

    if base_n >= base_m:
        out_base, inner_base = base_n, base_m
        out_value, inner_value = shape.n, shape.m
        out_bytes, inner_bytes = shape.d, shape.d
        inner_bad = shape.trans_a
    else:
        out_base, inner_base = base_m, base_n
        out_value, inner_value = shape.m, shape.n
        out_bytes, inner_bytes = shape.d, shape.d
        inner_bad = not shape.trans_b

    max_conflict = 5 if hardware.cores == 20 else min(hardware.cores, 6)
    min_conflict = 4 if hardware.cores == 20 else min(hardware.cores, 3)
    inner_max = min_conflict if inner_bad else max_conflict
    outer_min = max(1, hardware.cores // max_conflict)
    inner_min = max(1, hardware.cores // inner_max)
    enabled = False
    out_conflict = inner_conflict = 0
    for outer_use in range(hardware.cores, outer_min - 1, -1):
        for inner_use in range(hardware.cores, inner_min - 1, -1):
            out_tile = max(out_value // (out_base * outer_use), 1)
            inner_tile = max(inner_value // (inner_base * inner_use), 1)
            out_split = align_up(ceil_div(out_value, out_tile), out_base)
            inner_split = align_up(ceil_div(inner_value, inner_tile), inner_base)
            size = (
                out_split * shape.k * out_bytes
                + shape.k * inner_split * inner_bytes
                + out_split * inner_split * shape.dc
            )
            if size > budget:
                continue
            out_tail_value = (out_value + out_split - 1) % out_split + 1
            inner_tail_value = (inner_value + inner_split - 1) % inner_split + 1
            out_tail_count = ceil_div(out_tail_value, out_base)
            inner_tail_count = ceil_div(inner_tail_value, inner_base)
            if (
                out_tail_count * max_conflict < hardware.cores
                or inner_tail_count * inner_max < hardware.cores
            ):
                continue
            new_out_conflict = ceil_div(hardware.cores, out_tail_count)
            new_inner_conflict = ceil_div(hardware.cores, inner_tail_count)
            if not enabled or (
                out_conflict >= new_out_conflict
                and inner_conflict >= new_inner_conflict
            ):
                enabled = True
                out_conflict = new_out_conflict
                inner_conflict = new_inner_conflict
    return enabled


def _single_core_split_shape_applicable(
    shape: Any, hardware: Any, knowledge: dict[str, int]
) -> bool:
    base_m = int(knowledge["baseM"])
    base_n = int(knowledge["baseN"])
    large = (
        shape.m * shape.k >= 5 * 384 * 384
        and shape.n >= 1024
        and shape.m >= 384
        and shape.k >= 384
        and shape.m * shape.n >= 1024 * 384 * hardware.cores
        and not _l2_cache_enabled(shape, hardware, base_m, base_n)
    )
    n_256_aligned = shape.n * shape.d % 256 == 0
    tlb_mata = (
        shape.trans_a
        and not shape.trans_b
        and n_256_aligned
        and shape.k >= 11_000
        and (
            (shape.m % 8192 == 0 and shape.n >= 6144)
            or (shape.n % 8192 == 0 and shape.m >= 6144)
        )
    )
    mata_nt = (
        not shape.trans_a
        and shape.trans_b
        and shape.k % 16384 == 0
        and 1280 <= shape.m <= 8192
        and 1280 <= shape.n <= 8192
        and shape.dtype in ("fp16", "bf16")
        and hardware.cores == 24
    )
    supported = shape.k >= 27_392 or large or tlb_mata or mata_nt
    if not supported:
        return False
    single_m = int(knowledge["singleCoreM"])
    single_n = int(knowledge["singleCoreN"])
    if single_n < 512:
        return False
    occupancy = shape.m * shape.n / (
        single_m * single_n * hardware.cores
    )
    if occupancy < 0.70:
        return False
    if n_256_aligned and 896 <= shape.n <= 2048:
        if single_n <= 640 or occupancy < 0.85:
            return False
    return True


def _single_core_split_shape_domain(shape: Any, hardware: Any) -> bool:
    """Shape-only source gate before candidate-dependent SC occupancy checks."""

    return (
        shape.k >= 27_392
        or (
            shape.trans_a
            and not shape.trans_b
            and shape.n * shape.d % 256 == 0
            and shape.k >= 11_000
            and (
                (shape.m % 8192 == 0 and shape.n >= 6144)
                or (shape.n % 8192 == 0 and shape.m >= 6144)
            )
        )
        or (
            shape.m * shape.k >= 5 * 384 * 384
            and shape.n >= 1024
            and shape.m >= 384
            and shape.k >= 384
            and shape.m * shape.n >= 1024 * 384 * hardware.cores
        )
        or (
            not shape.trans_a
            and shape.trans_b
            and shape.k % 16384 == 0
            and 1280 <= shape.m <= 8192
            and 1280 <= shape.n <= 8192
            and shape.dtype in ("fp16", "bf16")
            and hardware.cores == 24
        )
    )


def _deterministic_split_shape_applicable(shape: Any, hardware: Any) -> bool:
    m_count = ceil_div(shape.m, 128)
    n_count = ceil_div(shape.n, 128)
    enough_k = shape.k >= hardware.cores * 384
    low_mn = m_count * n_count < hardware.cores // 2
    if enough_k and low_mn and not (not shape.trans_a and shape.trans_b):
        return True
    if (
        1000 <= shape.k <= 4608
        and shape.m <= 256
        and shape.n <= 256
        and shape.trans_a
        and not shape.trans_b
    ):
        return True
    if shape.k < 27_392:
        return False
    m_256_aligned = shape.m * shape.d % 256 == 0
    k_256_aligned = shape.k * shape.d % 256 == 0
    n_256_aligned = shape.n * shape.d % 256 == 0
    inner_a_aligned = m_256_aligned if shape.trans_a else k_256_aligned
    inner_b_aligned = k_256_aligned if shape.trans_b else n_256_aligned
    mte2_bound_fp32 = (
        (not inner_a_aligned or not inner_b_aligned)
        and shape.dtype == "fp32"
    )
    occupancy = (m_count * n_count) / align_up(
        m_count * n_count, hardware.cores
    )
    rejected_fixpipe_case = (
        m_count >= 4
        and n_count >= 4
        and not n_256_aligned
        and not mte2_bound_fp32
        and occupancy > 0.8
    )
    return not rejected_fixpipe_case


def _incremental_applicable(shape: Any) -> bool:
    if not (
        shape.m <= 128
        and shape.dtype in ("fp16", "bf16")
        and shape.c_dtype == shape.dtype
        and shape.b_dtype == shape.dtype
        and shape.format_a == shape.format_b == shape.format_out == "ND"
        and not shape.trans_a
        and shape.trans_b
        and not shape.bias
    ):
        return False
    return _conversion(shape, "base").disable_mix_nd2nz == 1


def _pre_applicable(graph_name: str, shape: Any, hardware: Any) -> bool:
    if graph_name == "base":
        return True
    if graph_name == "al1_full_load":
        return _al1_applicable(shape, hardware)
    if graph_name == "bl1_full_load":
        return _bl1_applicable(shape, hardware)
    if graph_name in (
        "bl1_full_load_fixpipe", "bl1_full_load_vec_nz2nd"
    ):
        if not _fixpipe_applicable(shape, hardware):
            return False
        if graph_name == "bl1_full_load_vec_nz2nd":
            conversion = _conversion(shape, graph_name)
            return (
                shape.dtype == "fp32"
                and not conversion.a
                and shape.k % 8 == 0
                and shape.n <= 192
            )
        return True
    if graph_name == "deterministic_split_k":
        return _deterministic_split_shape_applicable(shape, hardware)
    if graph_name == "single_core_split_k":
        # Its candidate-specific occupancy and L2 checks run after lowering.
        return _single_core_split_shape_domain(shape, hardware)
    return False


def family_applicability(shape: Any, hardware: Any) -> dict[str, bool]:
    """Public shape-only gates used by the audit and diagnostic output."""

    return {
        "BASE": True,
        "AL1": _al1_applicable(shape, hardware),
        "BL1": _bl1_applicable(shape, hardware),
        "FIXPIPE_BL1": _fixpipe_applicable(shape, hardware),
        # SC has a candidate-specific tail/occupancy gate.  This flag means
        # at least one source-visible shape mechanism could admit it.
        "SINGLE_CORE_SPLIT_K": _single_core_split_shape_domain(
            shape, hardware
        ),
        "DETERMINISTIC_SPLIT_K": _deterministic_split_shape_applicable(
            shape, hardware
        ),
        "INCREMENTAL_PATTERN": _incremental_applicable(shape),
    }


def non_installed_family_status(name: str) -> dict[str, str]:
    if name not in NON_INSTALLED_FAMILIES:
        raise KeyError(name)
    return {
        "family": name,
        "status": "REQUIRES_NEW_KERNEL",
        "reason": NON_INSTALLED_FAMILIES[name],
        "packet": "PROHIBITED",
    }


def _incremental_knowledge(
    shape: Any, hardware: Any, candidate: dict[str, Any]
) -> dict[str, int]:
    result = candidate["result"]
    core = candidate["core"]
    base_m = int(result["m_l0"]) * 16
    base_n = int(result["n_l0"]) * 16
    base_k = int(result["k_l0"]) * 16
    step_m = int(core["m_al1"])
    step_n = int(core["n_bl1"])
    step_ka = int(result["kal1_16"]) // int(result["k_l0"])
    step_kb = int(result["kbl1_16"]) // int(result["k_l0"])
    used = min(
        hardware.cores,
        int(result["m_dim"]) * int(result["n_dim"]),
    )
    m_tasks = ceil_div(shape.m, base_m)
    n_tasks = ceil_div(shape.n, base_n)
    return {
        "usedCoreNum": max(1, min(used, m_tasks * n_tasks)),
        "singleCoreM": base_m,
        "singleCoreN": base_n,
        "singleCoreK": shape.k,
        "baseM": base_m,
        "baseN": base_n,
        "baseK": base_k,
        "depthA1": step_m * step_ka * int(result["db_al1"]),
        "depthB1": step_n * step_kb * int(result["db_bl1"]),
        "stepM": step_m,
        "stepN": step_n,
        "iterateOrder": 0,
        "stepKa": step_ka,
        "stepKb": step_kb,
        "dbL0A": 2,
        "dbL0B": 2,
        "dbL0C": int(result["db_l0c"]),
        "l2MTileCnt": 1,
        "l2NTileCnt": 1,
        "l2MTileBlock": m_tasks,
        "l2NTileBlock": n_tasks,
        "l2IterateOrder": 0,
        "tilingEnable": 0,
        "mixNd2Nz": 1,
        "fp32Addmm": 0,
    }


def _peak_local_bytes(simulation: Any) -> int:
    return sum(
        value
        for space, value in simulation.peak_memory_bytes
        if space in (
            MemorySpace.L1,
            MemorySpace.L0A,
            MemorySpace.L0B,
            MemorySpace.L0C,
            MemorySpace.UB,
        )
    )


def _family_cost_components(
    shape: Any,
    hardware: Any,
    family: str,
    knowledge: dict[str, int],
    simulation: Any,
    resources: dict[str, float],
) -> dict[str, float | int | str]:
    """Expose the physical terms that feed each family's critical path.

    These values are diagnostics of the same exact-plan simulation used for
    ranking.  They are not an independent fitted score and they contain no
    measured operator latency.
    """

    single_m = int(knowledge["singleCoreM"])
    single_n = int(knowledge["singleCoreN"])
    single_k = int(knowledge["singleCoreK"])
    base_m = int(knowledge["baseM"])
    base_n = int(knowledge["baseN"])
    active = int(simulation.active_cores)
    m_tasks = ceil_div(shape.m, single_m)
    n_tasks = ceil_div(shape.n, single_n)
    output_tasks = m_tasks * n_tasks
    waves = ceil_div(output_tasks, max(1, active))
    k_chunks = ceil_div(shape.k, single_k)
    padded_output = (
        ceil_div(shape.m, base_m) * base_m
        * ceil_div(shape.n, base_n) * base_n
    )
    common: dict[str, float | int | str] = {
        "active_aic": active,
        "mn_tasks": output_tasks,
        "mn_waves": waves,
        "mn_tail_tasks": output_tasks % max(1, active),
        "padded_output_elements": padded_output,
        "output_padding_waste_elements": padded_output - shape.m * shape.n,
        "cube_cycles": resources[Resource.CUBE.value],
        "mte2_cycles": resources[Resource.MTE2.value],
        "mte1_cycles": resources[Resource.MTE1.value],
        "fixpipe_cycles": resources[Resource.FIXPIPE.value],
        "vector_cycles": resources[Resource.VECTOR.value],
        "atomic_cycles": resources[Resource.ATOMIC.value],
        "synchronization_cycles": resources[Resource.SYNC.value],
        "gm_read_bytes": float(simulation.gm_read_bytes),
        "gm_write_bytes": float(simulation.gm_write_bytes),
        "l2_bytes": float(simulation.l2_bytes),
        "workspace_bytes": int(simulation.workspace_bytes),
        "pipeline_fill_drain_cycles": float(simulation.pipeline_fill_cycles),
    }
    if family in ("BASE", "INCREMENTAL_PATTERN"):
        common.update({
            "dataflow": "direct_mn",
            "l2_window_m_tasks": int(knowledge["l2MTileBlock"]),
            "l2_window_n_tasks": int(knowledge["l2NTileBlock"]),
            "nd2nz_vector_cycles": resources[Resource.VECTOR.value],
        })
    elif family == "AL1":
        common.update({
            "dataflow": "resident_a_stream_b",
            "full_a_copy_bytes_per_aic": (
                align_up(shape.m, 16)
                * align_up(shape.k, 32 // shape.d) * shape.d
            ),
            "full_a_replication_bytes": (
                align_up(shape.m, 16)
                * align_up(shape.k, 32 // shape.d) * shape.d * active
            ),
            "n_tasks": n_tasks,
            "n_tail_tasks": n_tasks % max(1, active),
            "b_stream_logical_bytes": shape.k * shape.n * shape.d,
        })
    elif family == "BL1":
        common.update({
            "dataflow": "resident_b_stream_a",
            "resident_b_bytes_per_aic": (
                align_up(shape.n, 16 if shape.trans_b else 32 // shape.d)
                * align_up(shape.k, 16) * shape.d
            ),
            "resident_b_replication_bytes": (
                align_up(shape.n, 16 if shape.trans_b else 32 // shape.d)
                * align_up(shape.k, 16) * shape.d * active
            ),
            "m_tasks_per_aic_ceiling": ceil_div(m_tasks, max(1, active)),
            "a_stream_logical_bytes": shape.m * shape.k * shape.d,
        })
    elif family == "FIXPIPE_BL1":
        alignment = (
            16 if int(knowledge["tilingEnable"]) // 1000 == 2
            else 512 // shape.dc
        )
        common.update({
            "dataflow": "resident_b_vector_output",
            "resident_b_bytes_per_aic": (
                align_up(shape.n, 16) * align_up(shape.k, 16) * shape.d
            ),
            "temporary_output_alignment_elements": alignment,
            "temporary_output_padding_elements_per_task": (
                base_m * (align_up(shape.n, alignment) - shape.n)
            ),
            "aic_aiv_handshake_operations_per_output_task": 4,
        })
    elif family == "SINGLE_CORE_SPLIT_K":
        common.update({
            "dataflow": "serial_k_chunks_per_mn_owner",
            "serial_k_chunks": k_chunks,
            "repeated_accumulation_chunks": max(0, k_chunks - 1),
            "repeated_output_logical_bytes": (
                max(0, k_chunks - 1) * shape.m * shape.n * shape.dc
            ),
        })
    elif family == "DETERMINISTIC_SPLIT_K":
        producers = min(k_chunks, active)
        common.update({
            "dataflow": "parallel_k_partial_workspace_reduce",
            "logical_k_chunks": k_chunks,
            "k_partition_producers": producers,
            "k_owner_rounds_ceiling": ceil_div(k_chunks, max(1, producers)),
            "k_owner_rounds_floor": k_chunks // max(1, producers),
            "partial_fp32_logical_bytes": (
                shape.m * shape.n * 4 * producers
            ),
            "vector_reduction_adds": (
                shape.m * shape.n * max(0, producers - 1)
            ),
            "reduction_cycles": float(simulation.reduction_cycles),
        })
    else:
        raise ValueError("unknown installed family for cost components")
    return common


def _candidate_from_knowledge(
    shape: Any,
    hardware: Any,
    graph_name: str,
    family: str,
    knowledge: dict[str, int],
    source: str,
    model_hardware: Any,
    operator: Any,
    *,
    incremental_pattern: bool = False,
) -> dict[str, Any]:
    violations = validate_cann_tiling(
        shape.m,
        shape.n,
        shape.k,
        shape.dtype,
        shape.trans_a,
        shape.trans_b,
        knowledge,
        model_hardware,
        a_layout=shape.format_a,
        b_layout=shape.format_b,
        output_dtype=shape.c_dtype,
        has_bias=shape.bias,
        incremental_pattern=incremental_pattern,
    )
    if violations:
        raise ValueError(";".join(violations))
    exact_plan = plan_from_cann(
        shape.m,
        shape.n,
        shape.k,
        knowledge,
        dtype=shape.dtype,
        trans_a=shape.trans_a,
        trans_b=shape.trans_b,
        a_layout=shape.format_a,
        b_layout=shape.format_b,
        output_dtype=shape.c_dtype,
        has_bias=shape.bias,
    )
    simulation = simulate(operator, exact_plan, model_hardware)
    if not simulation.valid:
        raise ValueError("exact lowered simulation: " + simulation.error)
    observed_resources = dict(simulation.resource_cycles)
    resources = {
        resource.value: float(observed_resources.get(resource, 0.0))
        for resource in Resource
    }
    imbalance = simulation.critical_core_cycles / max(
        simulation.average_core_cycles, 1.0
    )
    peak_local = _peak_local_bytes(simulation)
    objectives = (
        float(simulation.critical_core_cycles),
        float(imbalance),
        float(resources.get(Resource.CUBE.value, 0.0)),
        float(simulation.gm_read_bytes + simulation.gm_write_bytes),
        float(simulation.l2_bytes),
        float(simulation.reduction_cycles),
        float(simulation.workspace_bytes),
        float(peak_local),
    )
    conversion = _conversion(shape, graph_name)
    graph = decode_execution_graph(knowledge)
    suffix = kernel_suffix(
        knowledge,
        aligned=bool(knowledge["mixNd2Nz"]),
    )
    if suffix not in INSTALLED_SUFFIXES_BY_DTYPE[shape.dtype]:
        raise ValueError("candidate emitted a non-installed dtype/suffix pair")
    fields = {
        name: int(knowledge[name])
        for name in (
            "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
            "baseM", "baseN", "baseK", "depthA1", "depthB1",
            "stepM", "stepN", "stepKa", "stepKb", "dbL0A",
            "dbL0B", "dbL0C", "iterateOrder",
        )
    }
    l2 = {
        "mTile": int(knowledge["l2MTileCnt"]),
        "nTile": int(knowledge["l2NTileCnt"]),
        "mTileBlock": int(knowledge["l2MTileBlock"]),
        "nTileBlock": int(knowledge["l2NTileBlock"]),
        "calOrder": int(knowledge["l2IterateOrder"]),
    }
    usage = {
        "L0A": fields["dbL0A"] * fields["baseM"] * fields["baseK"] * shape.d,
        "L0B": fields["dbL0B"] * fields["baseN"] * fields["baseK"] * shape.d,
        "L0C": fields["dbL0C"] * fields["baseM"] * fields["baseN"] * 4,
        "L1_AB": (
            fields["depthA1"] * fields["baseM"]
            + fields["depthB1"] * fields["baseN"]
        ) * fields["baseK"] * shape.d,
        "peak_local": peak_local,
    }
    return {
        "family": family,
        "graph_name": graph_name,
        "source": source,
        "knowledge": {key: int(value) for key, value in knowledge.items()},
        "fields": fields,
        "l2": l2,
        "conversion_a": bool(conversion.a),
        "conversion_b": bool(conversion.b),
        "fix_mode": int(graph.fix),
        "kernel_suffix": suffix,
        "workspace_bytes": int(simulation.workspace_bytes),
        "resource_bytes": usage,
        "cost": {
            "total_cycles": float(simulation.total_cycles),
            "critical_core_cycles": float(simulation.critical_core_cycles),
            "average_core_cycles": float(simulation.average_core_cycles),
            "critical_to_average_core_ratio": float(imbalance),
            "hbm_cycles": float(simulation.hbm_cycles),
            "l2_cycles": float(simulation.l2_cycles),
            "shared_resource_cycles": float(simulation.shared_resource_cycles),
            "pipeline_fill_cycles": float(simulation.pipeline_fill_cycles),
            "launch_cycles": float(simulation.launch_cycles),
            "reduction_cycles": float(simulation.reduction_cycles),
            "bottleneck": simulation.bottleneck,
            "active_cores": int(simulation.active_cores),
            "gm_read_bytes": float(simulation.gm_read_bytes),
            "gm_write_bytes": float(simulation.gm_write_bytes),
            "l2_bytes": float(simulation.l2_bytes),
            "workspace_bytes": int(simulation.workspace_bytes),
            "peak_local_bytes": peak_local,
            "resource_cycles": resources,
        },
        "family_cost_components": _family_cost_components(
            shape, hardware, family, knowledge, simulation, resources
        ),
        "pareto_objectives": objectives,
        "exact_lowered_resimulation": True,
        "validation": [],
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lhs = left["pareto_objectives"]
    rhs = right["pareto_objectives"]
    return all(a <= b for a, b in zip(lhs, rhs)) and any(
        a < b
        for a, b in zip(lhs, rhs)
    )


def _pareto_by_family(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["family"], []).append(candidate)
    retained: list[dict[str, Any]] = []
    for family in sorted(grouped):
        items = grouped[family]
        retained.extend(
            candidate
            for candidate in items
            if not any(
                other is not candidate and _dominates(other, candidate)
                for other in items
            )
        )
    return retained


def _candidate_identity(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["family"],
        candidate["graph_name"],
        tuple(sorted(candidate["knowledge"].items())),
    )


def _winner_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["cost"]["total_cycles"],
        candidate["family"],
        candidate["graph_name"],
        tuple(sorted(candidate["knowledge"].items())),
    )


def _knowledge_identity(knowledge: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, int(value)) for name, value in knowledge.items()))


def _set_l2_window(
    shape: Any,
    knowledge: dict[str, int],
    m_block: int,
    n_block: int,
    order: int,
) -> dict[str, int]:
    result = dict(knowledge)
    m_tasks = ceil_div(shape.m, int(result["singleCoreM"]))
    n_tasks = ceil_div(shape.n, int(result["singleCoreN"]))
    if m_block == 0 or n_block == 0:
        result.update({
            "l2MTileCnt": 1, "l2NTileCnt": 1,
            "l2MTileBlock": 0, "l2NTileBlock": 0,
            "l2IterateOrder": 0,
        })
    else:
        m_block = max(1, min(m_tasks, m_block))
        n_block = max(1, min(n_tasks, n_block))
        result.update({
            "l2MTileCnt": ceil_div(m_tasks, m_block),
            "l2NTileCnt": ceil_div(n_tasks, n_block),
            "l2MTileBlock": m_block,
            "l2NTileBlock": n_block,
            "l2IterateOrder": order,
        })
    return result


def _structural_packet_variants(
    shape: Any,
    hardware: Any,
    graph_name: str,
    original: dict[str, int],
) -> tuple[dict[str, int], ...]:
    """Expand one lowered point at every kernel-visible transition.

    This is a finite structural enumeration.  Values come from scratchpad
    capacity, divisibility, task waves and installed packet semantics; no
    measured latency, official packet, workload id or historical row enters
    the candidate set.
    """

    variants: dict[tuple[tuple[str, int], ...], dict[str, int]] = {}

    def keep(item: dict[str, int]) -> None:
        variants.setdefault(_knowledge_identity(item), item)

    keep(dict(original))
    m_tasks = ceil_div(shape.m, int(original["singleCoreM"]))
    n_tasks = ceil_div(shape.n, int(original["singleCoreN"]))

    # Cube traversal, local buffer counts and L2 windows are already full
    # dimensions of derive_ideal_region.  Do not cross-product them again at
    # the ABI boundary; doing so would duplicate the same structural point
    # and make deployment-time selection scale quadratically.

    if graph_name == "al1_full_load":
        in_bytes = _dtype_bytes(shape.dtype)
        base_m = base_n = 16
        max_base_k = min(
            shape.k,
            hardware.l0a // (2 * base_m * in_bytes),
            hardware.l0b // (2 * base_n * in_bytes),
        )
        resident_a = align_up(shape.m, 16) * shape.k * in_bytes
        opposite_double_buffer_limit = (
            hardware.l1_usable - resident_a
        ) // (2 * base_n * in_bytes)
        limits = {
            max_base_k,
            min(max_base_k, opposite_double_buffer_limit),
            max_base_k // 2,
            max_base_k // 4,
            64, 128, 256, 512,
        }
        base_k_values = {
            max(
                candidate
                for candidate in range(8, max(8, limit) + 1, 8)
                if shape.k % candidate == 0
            )
            for limit in limits
            if limit >= 8
        }
        for base_k in sorted(base_k_values):
            k_steps = shape.k // base_k
            opposite_capacity = (
                hardware.l1_usable - resident_a
            ) // (base_n * base_k * in_bytes)
            step_kb_values = {
                divisor
                for divisor in range(1, k_steps + 1)
                if k_steps % divisor == 0 and divisor <= opposite_capacity
            }
            if not step_kb_values:
                continue
            transition_steps = {
                1,
                max(step_kb_values),
                max(value for value in step_kb_values
                    if value <= max(1, max(step_kb_values) // 2)),
            }
            for step_kb in sorted(transition_steps):
                for packets in (1, 2):
                    depth_b = packets * step_kb
                    if (
                        resident_a
                        + depth_b * base_n * base_k * in_bytes
                        > hardware.l1_usable
                    ):
                        continue
                    item = dict(original)
                    item.update({
                        "usedCoreNum": min(hardware.cores, n_tasks),
                        "singleCoreM": shape.m,
                        "singleCoreN": 16,
                        "singleCoreK": shape.k,
                        "baseM": 16, "baseN": 16,
                        "baseK": base_k,
                        "depthA1": k_steps,
                        "depthB1": depth_b,
                        "stepM": 1, "stepN": 1,
                        "stepKa": k_steps, "stepKb": step_kb,
                        "dbL0A": 2, "dbL0B": 2,
                    })
                    keep(_set_l2_window(
                        shape, item, 1, n_tasks, 1
                    ))

    if graph_name in (
        "bl1_full_load_fixpipe", "bl1_full_load_vec_nz2nd"
    ):
        in_bytes = _dtype_bytes(shape.dtype)
        base_m = int(original["baseM"])
        base_n = int(original["baseN"])
        base_k = int(original["baseK"])
        b_depth = ceil_div(shape.k, base_k)
        a_capacity = (
            hardware.l1_usable // in_bytes - b_depth * base_n * base_k
        ) // (base_m * base_k)
        for a_packets in (1, 2):
            for step_ka in range(1, min(b_depth, a_capacity // a_packets) + 1):
                item = dict(original)
                item.update({
                    "stepKa": step_ka,
                    "depthA1": a_packets * step_ka,
                    "stepKb": b_depth,
                    "depthB1": b_depth,
                })
                for db_l0c in (1, 2):
                    if db_l0c * base_m * base_n * 4 > hardware.l0c:
                        continue
                    item["dbL0C"] = db_l0c
                    # Explicitly retain both the legacy disabled scheduler
                    # and installed block traversal.  This prevents either
                    # path from disappearing merely due generic projection.
                    keep(_set_l2_window(shape, item, 0, 0, 0))
                    keep(_set_l2_window(shape, item, 1, 1, 1))

    return tuple(variants.values())


def generate_and_select(
    shape: Any,
    hardware: Any,
    compile_info: Any,
    *,
    include_audit_records: bool = False,
) -> dict[str, Any]:
    """Generate the complete finite ideal region and select its global minimum."""

    del compile_info  # Input-domain validation remains owned by formula_rules.
    started = perf_counter_ns()
    model_hardware = ascend_910b3()
    expected = (
        hardware.cores,
        hardware.l0a,
        hardware.l0b,
        hardware.l0c,
        hardware.l1_usable,
        hardware.l2,
        hardware.ub,
    )
    actual = (
        model_hardware.core_count(Resource.CUBE),
        model_hardware.capacities[MemorySpace.L0A],
        model_hardware.capacities[MemorySpace.L0B],
        model_hardware.capacities[MemorySpace.L0C],
        model_hardware.capacities[MemorySpace.L1],
        model_hardware.capacities[MemorySpace.L2],
        model_hardware.capacities[MemorySpace.UB],
    )
    if expected != actual:
        raise ValueError(
            "candidate engine is pinned to the frozen Ascend910B3 profile"
        )

    complete_operator = matmul(
        shape.m,
        shape.n,
        shape.k,
        shape.dtype,
        trans_a=shape.trans_a,
        trans_b=shape.trans_b,
        a_layout=shape.format_a,
        b_layout=shape.format_b,
        output_dtype=shape.c_dtype,
        has_bias=shape.bias,
    )
    # Exact source predicates are hard legality constraints, so rejecting an
    # inapplicable graph before ideal-region construction cannot remove a
    # legal candidate.  It avoids deriving and simulating hundreds of plans
    # that the installed source would reject unconditionally.
    applicable_indices = tuple(
        index
        for index, algorithm in enumerate(complete_operator.algorithms)
        if _pre_applicable(algorithm.name, shape, hardware)
    )
    applicable_algorithms = tuple(
        complete_operator.algorithms[index] for index in applicable_indices
    )
    if not applicable_algorithms:
        raise RuntimeError("candidate engine has no applicable BASE graph")
    operator = replace(complete_operator, algorithms=applicable_algorithms)
    region = derive_ideal_region(
        operator,
        model_hardware,
        policy=SearchPolicy(max_evaluations=100_000),
    )
    if not region.exhaustive:
        raise RuntimeError(
            "ideal-region generation reached max_evaluations; partial output is forbidden"
        )

    raw_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    pareto_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    evaluated_packets: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
    for plan in region.plans:
        graph_name = operator.algorithms[plan.algorithm].name
        family = GRAPH_TO_FAMILY[graph_name]
        raw_counts[family] += 1
        try:
            full_plan = replace(
                plan, algorithm=applicable_indices[plan.algorithm]
            )
            knowledge = lower_plan_to_cann(
                shape.m,
                shape.n,
                shape.k,
                shape.dtype,
                shape.trans_a,
                shape.trans_b,
                full_plan,
                model_hardware,
                a_layout=shape.format_a,
                b_layout=shape.format_b,
                output_dtype=shape.c_dtype,
                has_bias=shape.bias,
            )
            if graph_name == "single_core_split_k" and not (
                _single_core_split_shape_applicable(
                    shape, hardware, knowledge
                )
            ):
                raise ValueError("SOURCE_SINGLE_CORE_SPLIT_K_GATE")
            lowered_variants = _structural_packet_variants(
                shape, hardware, graph_name, knowledge
            )
        except (ValueError, ZeroDivisionError) as exception:
            rejection_reasons[
                family + ":" + str(exception).split(";", 1)[0]
            ] += 1
            continue
        for knowledge_variant in lowered_variants:
            packet_identity = (
                family, _knowledge_identity(knowledge_variant)
            )
            if packet_identity in evaluated_packets:
                continue
            evaluated_packets.add(packet_identity)
            try:
                candidate = _candidate_from_knowledge(
                    shape,
                    hardware,
                    graph_name,
                    family,
                    knowledge_variant,
                    "hardware_transition_region",
                    model_hardware,
                    complete_operator,
                )
            except (ValueError, ZeroDivisionError) as exception:
                rejection_reasons[
                    family + ":" + str(exception).split(";", 1)[0]
                ] += 1
                continue
            unique.setdefault(_candidate_identity(candidate), candidate)

    incremental_source_count = 0
    incremental_converted_count = 0
    if _incremental_applicable(shape):
        incremental = generate_incremental(
            _shape_request(shape), _hardware_request(hardware), trace=False
        )
        incremental_source_count = len(incremental["candidates"])
        raw_counts["INCREMENTAL_PATTERN"] += incremental_source_count
        for source_candidate in incremental["candidates"]:
            try:
                knowledge = _incremental_knowledge(
                    shape, hardware, source_candidate
                )
                candidate = _candidate_from_knowledge(
                    shape,
                    hardware,
                    "base",
                    "INCREMENTAL_PATTERN",
                    knowledge,
                    "incremental_source_loop",
                    model_hardware,
                    complete_operator,
                    incremental_pattern=True,
                )
            except (ValueError, ZeroDivisionError) as exception:
                rejection_reasons[
                    "INCREMENTAL_PATTERN:"
                    + str(exception).split(";", 1)[0]
                ] += 1
                continue
            incremental_converted_count += 1
            unique.setdefault(_candidate_identity(candidate), candidate)

    candidates = list(unique.values())
    for candidate in candidates:
        unique_counts[candidate["family"]] += 1
    frontier = _pareto_by_family(candidates)
    for candidate in frontier:
        pareto_counts[candidate["family"]] += 1
    if not frontier:
        raise RuntimeError("candidate engine produced no legal BASE candidate")
    frontier.sort(key=_winner_key)
    winner = frontier[0]
    elapsed_us = (perf_counter_ns() - started) / 1000.0
    winner = dict(winner)
    winner["candidate_audit"] = {
        "region_exhaustive": True,
        "region_evaluated": int(region.evaluated),
        "region_plan_count": len(region.plans),
        "region_anchor_count": len(region.anchors),
        "raw_counts": dict(sorted(raw_counts.items())),
        "unique_counts": dict(sorted(unique_counts.items())),
        "pareto_counts": dict(sorted(pareto_counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "incremental_source_candidates": incremental_source_count,
        "incremental_converted_candidates": incremental_converted_count,
        "winner_is_global_minimum": _winner_key(winner) == min(
            _winner_key(candidate) for candidate in frontier
        ),
        "selection_elapsed_us": elapsed_us,
        "official_selector_called": False,
        "history_lookup": False,
        "fixed_topn_pruning": False,
        "exact_lowered_resimulation": all(
            candidate["exact_lowered_resimulation"] for candidate in candidates
        ),
        "applicability": family_applicability(shape, hardware),
        "non_installed": [
            non_installed_family_status(name)
            for name in sorted(NON_INSTALLED_FAMILIES)
        ],
    }
    if include_audit_records:
        winner["candidate_audit"]["all_candidates"] = sorted(
            candidates, key=_winner_key
        )
        winner["candidate_audit"]["pareto_frontier"] = sorted(
            frontier, key=_winner_key
        )
    return winner
