#!/usr/bin/env python3
"""Exact worker-cardinality rules for every installed C220 MatMulV3 suffix.

The rules in this file do not choose a kernel family or a tile geometry.  They
take the packet produced by the source-reconstructed CANN 8.1 selector and
prove how many AIC/AIV workers that exact packet can use.  A launch is reduced
only when every removed worker has zero Cube, conversion, output, and reduction
ownership.  This makes the transformation an instruction deletion rather than
a second tiling search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


INSTALLED_SUFFIXES = (
    0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201,
)


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div requires value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


def align_up(value: int, quantum: int) -> int:
    return ceil_div(value, quantum) * quantum


@dataclass(frozen=True)
class CorePlan:
    suffix: int
    branch: str
    source_cores: int
    required_cores: int
    cube_owners: int
    reduction_owners: int
    vector_a_tasks: int
    vector_b_tasks: int
    vector_output_owners: int
    action: str
    proof_kind: str
    reason: str

    @property
    def eliminated_cores(self) -> int:
        return self.source_cores - self.required_cores

    def as_dict(self) -> dict:
        return {
            "suffix": self.suffix,
            "branch": self.branch,
            "source_cores": self.source_cores,
            "required_cores": self.required_cores,
            "eliminated_cores": self.eliminated_cores,
            "cube_owners": self.cube_owners,
            "reduction_owners": self.reduction_owners,
            "vector_a_tasks": self.vector_a_tasks,
            "vector_b_tasks": self.vector_b_tasks,
            "vector_output_owners": self.vector_output_owners,
            "action": self.action,
            "proof_kind": self.proof_kind,
            "reason": self.reason,
        }


def _regular_nd2nz_tasks(
    *, operand: str, m: int, n: int, k: int, width: int,
    trans_a: bool, trans_b: bool, base_n: int, base_d: int,
) -> tuple[int, str]:
    """Return exact MatrixtoNZ tile count, or -1 for the VNCHW subpath."""
    c0 = 32 // width
    if operand == "A":
        ori_n = k if trans_a else m
        ori_d = m if trans_a else k
        d_value = align_up(m, c0) if trans_a else align_up(k, c0)
    elif operand == "B":
        ori_n = n if trans_b else k
        ori_d = k if trans_b else n
        d_value = align_up(k, c0) if trans_b else align_up(n, c0)
    else:
        raise ValueError(f"unknown operand {operand}")
    vnchw = (
        ori_n > 8192 and ori_d > 1 and ori_d != c0 and
        (
            ori_d * width <= 192 or
            (ori_d * width <= 384 and ori_d % 2 == 0) or
            (ori_d * width <= 512 and ori_d % 4 == 0)
        )
    )
    if vnchw:
        return -1, "VNCHW_PARTITION_DEPENDS_ON_LAUNCHED_AIV"
    if base_n <= 0 or base_d <= 0:
        raise ValueError(f"{operand} conversion has non-positive vector tile")
    return ceil_div(ori_n, base_n) * ceil_div(d_value, base_d), "MATRIX_TO_NZ_TILES"


def _common_geometry(request: dict, tiling_data: dict) -> dict:
    cube = tiling_data["matmulTiling"]
    run = tiling_data["matmulRunInfo"]
    l2 = tiling_data["tileL2cacheTiling"]
    source = int(cube["usedCoreNum"])
    if not 1 <= source <= 20:
        raise ValueError(f"source usedCoreNum is outside C220: {source}")
    output_tasks = (
        ceil_div(int(request["M"]), int(cube["singleCoreM"])) *
        ceil_div(int(request["N"]), int(cube["singleCoreN"]))
    )
    global_m = ceil_div(int(request["M"]), int(cube["singleCoreM"]))
    global_n = ceil_div(int(request["N"]), int(cube["singleCoreN"]))
    m_block = int(l2["mTileBlock"])
    n_block = int(l2["nTileBlock"])
    if m_block <= 0:
        m_block = ceil_div(global_m, int(l2["mTileCntL2"]))
    if n_block <= 0:
        n_block = ceil_div(global_n, int(l2["nTileCntL2"]))
    window_output_tasks = min(global_m, m_block) * min(global_n, n_block)
    width = 4 if request["dtype"] == "fp32" else 2
    vector_tasks = {"A": 0, "B": 0}
    vector_modes = {"A": "DISABLED", "B": "DISABLED"}
    if int(run["nd2nzA"]):
        vector_tasks["A"], vector_modes["A"] = _regular_nd2nz_tasks(
            operand="A", m=int(request["M"]), n=int(request["N"]),
            k=int(request["K"]), width=width,
            trans_a=bool(request["transA"]), trans_b=bool(request["transB"]),
            base_n=int(tiling_data["baseAN"]),
            base_d=int(tiling_data["baseAD"]),
        )
    if int(run["nd2nzB"]):
        vector_tasks["B"], vector_modes["B"] = _regular_nd2nz_tasks(
            operand="B", m=int(request["M"]), n=int(request["N"]),
            k=int(request["K"]), width=width,
            trans_a=bool(request["transA"]), trans_b=bool(request["transB"]),
            base_n=int(tiling_data["baseBN"]),
            base_d=int(tiling_data["baseBD"]),
        )
    return {
        "cube": cube,
        "source": source,
        "output_tasks": output_tasks,
        "window_output_tasks": window_output_tasks,
        "vector_tasks": vector_tasks,
        "vector_modes": vector_modes,
    }


def _finish(
    *, suffix: int, branch: str, geometry: dict, cube_owners: int,
    reduction_owners: int = 0, vector_output_owners: int = 0,
    permit_reduction: bool = True, proof_kind: str,
    branch_constraint: str,
) -> CorePlan:
    source = geometry["source"]
    vector_tasks = geometry["vector_tasks"]
    vector_modes = geometry["vector_modes"]
    requirements = [1, min(source, max(cube_owners, 1))]
    if reduction_owners:
        requirements.append(min(source, reduction_owners))
    if vector_output_owners:
        requirements.append(min(source, vector_output_owners))
    for operand in ("A", "B"):
        tasks = vector_tasks[operand]
        if tasks < 0:
            permit_reduction = False
        elif tasks:
            # C220 launches two AIV workers for each AIC worker.
            requirements.append(min(source, ceil_div(tasks, 2)))
    required = max(requirements)
    if required > source:
        raise ValueError("worker proof requested more cores than source packet")
    if not permit_reduction:
        required = source
        action = "RETAIN_COUPLED_CORE_COUNT"
    elif required < source:
        action = "DELETE_ZERO_OWNERSHIP_CORES"
    else:
        action = "ALREADY_EXACT_OR_FULLY_OCCUPIED"
    reason = (
        f"{branch_constraint}; output_tasks={geometry['output_tasks']}; "
        f"max_window_tasks={geometry['window_output_tasks']}; "
        f"vectorA={vector_tasks['A']}:{vector_modes['A']}; "
        f"vectorB={vector_tasks['B']}:{vector_modes['B']}"
    )
    return CorePlan(
        suffix=suffix, branch=branch, source_cores=source,
        required_cores=required, cube_owners=cube_owners,
        reduction_owners=reduction_owners,
        vector_a_tasks=vector_tasks["A"],
        vector_b_tasks=vector_tasks["B"],
        vector_output_owners=vector_output_owners,
        action=action, proof_kind=proof_kind, reason=reason,
    )


def _base_unaligned(request: dict, data: dict) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=0, branch="BASE_UNALIGNED", geometry=g,
        cube_owners=g["window_output_tasks"],
        proof_kind="BASE_MN_PLUS_ND2NZ_ZERO_OWNER_DELETION",
        branch_constraint="AIC owns one MN parent; pre-MMAD AIV owns one ND2NZ tile",
    )


def _base_aligned(request: dict, data: dict) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=1, branch="BASE_ALIGNED_OR_INCREMENTAL", geometry=g,
        cube_owners=g["window_output_tasks"],
        proof_kind="BASE_MN_ZERO_OWNER_DELETION",
        branch_constraint="all GM/L1/MMAD/output work is inside the MN-owner guard",
    )


def _sc_unaligned(request: dict, data: dict) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=20, branch="SINGLE_CORE_SPLIT_K_UNALIGNED", geometry=g,
        cube_owners=g["output_tasks"], vector_output_owners=g["output_tasks"],
        proof_kind="SC_SPLIT_K_MN_AND_VECTOR_OWNER_CARDINALITY",
        branch_constraint="each AIC owns one MN parent and paired AIV output/conversion work",
    )


def _sc_aligned(request: dict, data: dict) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=21, branch="SINGLE_CORE_SPLIT_K_ALIGNED", geometry=g,
        cube_owners=g["output_tasks"], vector_output_owners=g["output_tasks"],
        proof_kind="SC_SPLIT_K_MN_AND_VECTOR_OWNER_CARDINALITY",
        branch_constraint="each AIC owns one MN parent and paired AIV output work",
    )


def _deterministic(request: dict, data: dict, suffix: int) -> CorePlan:
    g = _common_geometry(request, data)
    cube = g["cube"]
    k_chunks = ceil_div(int(request["K"]), int(cube["singleCoreK"]))
    return _finish(
        suffix=suffix,
        branch=("DETERMINISTIC_SPLIT_K_UNALIGNED" if suffix == 30
                else "DETERMINISTIC_SPLIT_K_ALIGNED"),
        geometry=g, cube_owners=k_chunks, reduction_owners=k_chunks,
        vector_output_owners=g["source"],
        proof_kind="DETERMINISTIC_K_OWNER_AND_REDUCTION_CARDINALITY",
        branch_constraint=(
            "usedCoreNum is the K-partial cardinality and the reducer consumes "
            "that exact number of workspace planes"
        ),
    )


def _al1(request: dict, data: dict) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=101, branch="AL1_FULL_LOAD_ALIGNED", geometry=g,
        cube_owners=g["window_output_tasks"],
        proof_kind="AL1_PRE_GUARD_FULL_A_COPY_ZERO_OWNER_DELETION",
        branch_constraint=(
            "each launched AIC copies full A before the MN-owner guard; "
            "only MN owners are required"
        ),
    )


def _bl1(request: dict, data: dict, suffix: int) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=suffix,
        branch="BL1_FULL_LOAD_UNALIGNED" if suffix == 200 else "BL1_FULL_LOAD_ALIGNED",
        geometry=g, cube_owners=g["window_output_tasks"],
        proof_kind=("BL1_MN_PLUS_ND2NZ_ZERO_OWNER_DELETION" if suffix == 200
                    else "BL1_MN_ZERO_OWNER_DELETION"),
        branch_constraint=(
            "B residency and IterateAll execute only for MN owners; unaligned "
            "variant additionally preserves every ND2NZ vector tile"
        ),
    )


def _fixpipe(request: dict, data: dict, suffix: int) -> CorePlan:
    g = _common_geometry(request, data)
    return _finish(
        suffix=suffix,
        branch={
            10200: "BL1_FULL_LOAD_FIXPIPE_UNALIGNED",
            10201: "BL1_FULL_LOAD_FIXPIPE_ALIGNED",
            20201: "BL1_FULL_LOAD_VEC_NZ2ND",
        }[suffix],
        geometry=g, cube_owners=g["window_output_tasks"],
        vector_output_owners=g["window_output_tasks"],
        proof_kind="FIXPIPE_MN_AIV_WORKSPACE_OWNER_CARDINALITY",
        branch_constraint=(
            "MN owners, paired AIV epilogue owners, per-core workspace and any "
            "input conversion are included in the lower bound"
        ),
    )


Rule = Callable[[dict, dict], CorePlan]


RULES: dict[int, Rule] = {
    0: _base_unaligned,
    1: _base_aligned,
    20: _sc_unaligned,
    21: _sc_aligned,
    30: lambda request, data: _deterministic(request, data, 30),
    31: lambda request, data: _deterministic(request, data, 31),
    101: _al1,
    200: lambda request, data: _bl1(request, data, 200),
    201: lambda request, data: _bl1(request, data, 201),
    10200: lambda request, data: _fixpipe(request, data, 10200),
    10201: lambda request, data: _fixpipe(request, data, 10201),
    20201: lambda request, data: _fixpipe(request, data, 20201),
}


if tuple(sorted(RULES)) != INSTALLED_SUFFIXES:
    raise RuntimeError("worker-cardinality dispatch does not cover all C220 suffixes")


def derive_core_plan(request: dict, tiling_data: dict, suffix: int) -> CorePlan:
    try:
        rule = RULES[int(suffix)]
    except KeyError as error:
        raise ValueError(f"unsupported installed suffix {suffix}") from error
    plan = rule(request, tiling_data)
    if plan.suffix != int(suffix):
        raise RuntimeError("worker-cardinality rule returned the wrong suffix")
    return plan
