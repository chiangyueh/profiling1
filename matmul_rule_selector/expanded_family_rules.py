#!/usr/bin/env python3
"""Execution-family registry beyond the twelve installed CANN 8.1 suffixes."""
from __future__ import annotations


AIC = 20
L1 = 524032
SYSTEM_WORKSPACE = 20 * 1024 * 1024


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


FAMILY_REGISTRY = {
    "BASE": {
        "installed_suffixes": (0, 1),
        "ownership": "MN",
        "changed_kernel": False,
    },
    "SINGLE_CORE_SPLIT_K": {
        "installed_suffixes": (20, 21),
        "ownership": "MN_WITH_INTERNAL_K_LOOP",
        "changed_kernel": False,
    },
    "DETERMINISTIC_SPLIT_K": {
        "installed_suffixes": (30, 31),
        "ownership": "K_PARTIAL_PLUS_AIV_REDUCTION",
        "changed_kernel": False,
    },
    "AL1_FULL_LOAD": {
        "installed_suffixes": (101,),
        "ownership": "N_WITH_A_RESIDENT",
        "changed_kernel": False,
    },
    "BL1_FULL_LOAD": {
        "installed_suffixes": (200, 201),
        "ownership": "M_WITH_B_RESIDENT",
        "changed_kernel": False,
    },
    "BL1_FIXPIPE": {
        "installed_suffixes": (10200, 10201, 20201),
        "ownership": "MN_WITH_FUSED_OR_VECTOR_EPILOGUE",
        "changed_kernel": False,
    },
    "ATOMIC_MULTI_CORE_SPLIT_K": {
        "bundled_suffixes": (41,),
        "ownership": "DISJOINT_K_WITH_ATOMIC_C",
        "cann81_build": True,
        "changed_kernel": True,
    },
    "NKM_SINGLE_CORE_SPLIT_K": {
        "bundled_suffixes": (51,),
        "ownership": "M_WITH_NK_PIPELINE",
        "cann81_build": True,
        "changed_kernel": True,
    },
    "GM_TO_L1_SINGLE_CORE_SPLIT_K": {
        "bundled_suffixes": (60, 61),
        "ownership": "MN_WITH_EXPLICIT_GM_L1_PIPELINE",
        "cann81_build": False,
        "blocker": "requires CANN 8.5 Iterate(bool,LocalTensor<L0C>)",
        "changed_kernel": True,
    },
    "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": {
        "bundled_suffixes": (121,),
        "ownership": "N_WITH_AL1_RESIDENT_AND_INTERNAL_K_LOOP",
        "cann81_build": True,
        "changed_kernel": True,
    },
    "BALANCED_STREAM_K_REDUCTION": {
        "proposed_suffixes": (),
        "ownership": "LINEARIZED_MNK_WORK_UNITS_PLUS_DETERMINISTIC_REDUCTION",
        "cann81_build": False,
        "blocker": "new persistent Cube scheduler and reducer kernel required",
        "changed_kernel": True,
    },
    "CUBE_VECTOR_EDGE_SPLIT": {
        "proposed_suffixes": (),
        "ownership": "ALIGNED_CUBE_INTERIOR_PLUS_VECTOR_EDGE",
        "cann81_build": False,
        "blocker": "new dual-path edge kernel and synchronization contract required",
        "changed_kernel": True,
    },
}


def proposed_family_plans(request: dict) -> list[dict]:
    """Return analytical plans for new kernels; never mark them NPU-ready."""
    m = int(request["M"])
    n = int(request["N"])
    k = int(request["K"])
    width = 4 if request["dtype"] == "fp32" else 2
    plans = []

    m_tiles = ceil_div(m, 128)
    n_tiles = ceil_div(n, 128)
    mn_tiles = m_tiles * n_tiles
    k_tiles = ceil_div(k, 256 // width)
    if mn_tiles < AIC and k_tiles >= 2 * AIC:
        work_units = mn_tiles * k_tiles
        units_per_core = ceil_div(work_units, AIC)
        partials_per_output = ceil_div(k_tiles, units_per_core)
        workspace = m_tiles * n_tiles * partials_per_output * 128 * 128 * 4
        plans.append({
            "family": "BALANCED_STREAM_K_REDUCTION",
            "npu_eligible": False,
            "reason": "MN wave underfills AIC while K has enough independent tiles",
            "equation": {
                "mn_tiles": mn_tiles,
                "k_tiles": k_tiles,
                "linear_work_units": work_units,
                "units_per_core": units_per_core,
                "partials_per_output_tile": partials_per_output,
                "partial_workspace_bytes": workspace,
                "load_imbalance_upper_bound_work_units": 1,
            },
            "hard_reject": workspace > 64 * 1024 * 1024,
            "kernel_gap": FAMILY_REGISTRY["BALANCED_STREAM_K_REDUCTION"]["blocker"],
        })

    aligned_m = m // 128 * 128
    aligned_n = n // 128 * 128
    interior = aligned_m * aligned_n
    total = m * n
    edge = total - interior
    padded = ceil_div(m, 128) * 128 * ceil_div(n, 128) * 128 - total
    if interior and edge and padded * 5 >= total:
        edge_bytes = edge * width
        plans.append({
            "family": "CUBE_VECTOR_EDGE_SPLIT",
            "npu_eligible": False,
            "reason": "128x128 Cube padding is at least 20 percent of logical output",
            "equation": {
                "cube_interior_m": aligned_m,
                "cube_interior_n": aligned_n,
                "logical_edge_elements": edge,
                "avoided_padded_output_elements": padded,
                "edge_input_output_lower_bound_bytes": edge_bytes,
                "scratch_bytes": 0,
            },
            "hard_reject": edge_bytes > L1,
            "kernel_gap": FAMILY_REGISTRY["CUBE_VECTOR_EDGE_SPLIT"]["blocker"],
        })
    return plans


def validate_registry() -> None:
    installed = []
    for entry in FAMILY_REGISTRY.values():
        installed.extend(entry.get("installed_suffixes", ()))
    expected = [0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201]
    if sorted(installed) != expected:
        raise RuntimeError("expanded registry lost an installed suffix")


validate_registry()
