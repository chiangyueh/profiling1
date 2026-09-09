#!/usr/bin/env python3
"""Freeze a narrow R5 L2-policy experiment without invoking MatMulV3 selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


SCHEDULE_FIELDS = (
    "used_core_num", "single_core_m", "single_core_n", "single_core_k",
    "base_m", "base_n", "base_k", "depth_a1", "depth_b1", "step_m",
    "step_n", "iterate_order", "step_ka", "step_kb", "db_l0a", "db_l0b",
    "db_l0c", "bank_l2_m_tile_count", "bank_l2_n_tile_count",
    "bank_l2_m_tile_block", "bank_l2_n_tile_block",
    "bank_l2_iterate_order", "bank_tiling_enable",
)
R5_CHANGED_FIELDS = (
    "bank_l2_m_tile_count",
    "bank_l2_m_tile_block",
)


@dataclass(frozen=True)
class Case:
    workload_id: str
    m: int
    n: int
    k: int
    dtype: str
    trans_a: bool
    trans_b: bool
    kernel_suffix: int
    coverage: str
    reference_signature: tuple[int, ...]


CASES = (
    Case(
        "r5_m4096_k4096_n640_fp16_tt",
        4096, 640, 4096, "fp16", True, True, 1,
        "previously_measured_r5_pair",
        (20, 256, 128, 4096, 256, 128, 64, 8, 16, 1, 1, 0, 4, 8,
         2, 2, 1, 1, 1, 16, 5, 0, 0),
    ),
    Case(
        "r5_m6144_k3072_n384_bf16_nt",
        6144, 384, 3072, "bf16", False, True, 1,
        "unmeasured_r5_boundary",
        (20, 256, 128, 3072, 256, 128, 64, 8, 16, 1, 1, 0, 4, 8,
         2, 2, 1, 1, 1, 24, 3, 0, 0),
    ),
    Case(
        "r5_m7169_k3073_n383_bf16_nt",
        7169, 383, 3073, "bf16", False, True, 0,
        "unmeasured_r5_tail_boundary",
        (20, 256, 128, 3073, 256, 128, 64, 8, 16, 1, 1, 0, 4, 8,
         2, 2, 1, 1, 1, 29, 3, 0, 0),
    ),
)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def schedule(case: Case) -> dict[str, int]:
    if len(case.reference_signature) != len(SCHEDULE_FIELDS):
        raise ValueError(f"{case.workload_id}: invalid reference signature length")
    return dict(zip(SCHEDULE_FIELDS, case.reference_signature))


def apply_r5(
    case: Case,
    reference: dict[str, int],
    *,
    aic_cores: int,
    l2_bytes: int,
) -> tuple[dict[str, int], bool]:
    """Apply only the post-family R5 equation to an explicitly marked control."""

    result = dict(reference)
    if case.dtype not in {"fp16", "bf16"} or not case.trans_b:
        return result, False
    if not (
        result["base_m"] > result["base_n"]
        and result["single_core_m"] == result["base_m"]
        and result["single_core_n"] == result["base_n"]
    ):
        return result, False
    m_blocks = ceil_div(case.m, result["single_core_m"])
    n_blocks = ceil_div(case.n, result["single_core_n"])
    if (
        result["bank_l2_m_tile_count"],
        result["bank_l2_n_tile_count"],
        result["bank_l2_m_tile_block"],
        result["bank_l2_n_tile_block"],
    ) != (1, 1, m_blocks, n_blocks):
        return result, False
    width = 2
    total_bytes = (
        case.m * case.k * width
        + case.k * case.n * width
        + case.m * case.n * width
    )
    limit = l2_bytes / (192 * 1024 * 1024) * (100 * 1024 * 1024)
    if total_bytes > limit:
        return result, False
    max_n_blocks = 5 if aic_cores == 20 else min(aic_cores, 6)
    if (
        n_blocks > max_n_blocks
        or ceil_div(m_blocks * n_blocks, result["used_core_num"]) < 4
    ):
        return result, False
    max_m_blocks = max(1, 2 * result["used_core_num"] // n_blocks)
    m_tile_count = ceil_div(m_blocks, max_m_blocks)
    if m_tile_count <= 1:
        return result, False
    result["bank_l2_m_tile_count"] = m_tile_count
    result["bank_l2_n_tile_count"] = 1
    result["bank_l2_m_tile_block"] = ceil_div(m_blocks, m_tile_count)
    result["bank_l2_n_tile_block"] = n_blocks
    return result, True


def schedule_digest(case: Case, values: dict[str, int]) -> str:
    payload = {
        "shape": {
            "m": case.m, "n": case.n, "k": case.k, "dtype": case.dtype,
            "trans_a": case.trans_a, "trans_b": case.trans_b,
        },
        "schedule": {field: values[field] for field in SCHEDULE_FIELDS},
    }
    encoded = json.dumps(
        payload, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build(
    *,
    aic_cores: int,
    l2_bytes: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict]]:
    if aic_cores < 20:
        raise ValueError("R5 validation requires at least 20 AIC cores")
    if l2_bytes <= 0:
        raise ValueError("L2 capacity must be positive")
    workload_rows: list[dict[str, str]] = []
    candidate_rows: list[dict[str, str]] = []
    audits: list[dict] = []
    all_fields = "|".join(SCHEDULE_FIELDS)
    unchanged_r5_fields = "|".join(
        field for field in SCHEDULE_FIELDS if field not in R5_CHANGED_FIELDS
    )
    blocks = (
        ("unchanged_reference_a", False),
        ("modified_r5_a", True),
        ("modified_r5_b", True),
        ("unchanged_reference_b", False),
    )
    for case in CASES:
        reference = schedule(case)
        modified, applied = apply_r5(
            case, reference, aic_cores=aic_cores, l2_bytes=l2_bytes
        )
        changed = tuple(
            field for field in SCHEDULE_FIELDS
            if modified[field] != reference[field]
        )
        if not applied or changed != R5_CHANGED_FIELDS:
            raise ValueError(
                f"{case.workload_id}: R5 must change exactly "
                + ",".join(R5_CHANGED_FIELDS)
            )
        reference_digest = schedule_digest(case, reference)
        modified_digest = schedule_digest(case, modified)
        if reference_digest == modified_digest:
            raise ValueError(f"{case.workload_id}: unchanged experimental packet")
        workload_rows.append({
            "workload_id": case.workload_id,
            "m": str(case.m), "n": str(case.n), "k": str(case.k),
            "dtype": case.dtype,
            "trans_a": str(int(case.trans_a)),
            "trans_b": str(int(case.trans_b)),
            "max_cores": "20",
            "search_family": "r5_l2_policy_paired_validation",
            "coverage_intent": case.coverage,
            "required_successful_tilings": "4",
        })
        for rank, (measurement_block, use_modified) in enumerate(blocks, 1):
            values = modified if use_modified else reference
            role = (
                "modified_r5_l2_candidate"
                if use_modified else "unchanged_direct_reference"
            )
            source = (
                "standalone_r5_l2_equation"
                if use_modified else "frozen_unchanged_packet_control"
            )
            row = {
                "rank": str(rank),
                "workload_id": case.workload_id,
                "m": str(case.m), "n": str(case.n), "k": str(case.k),
                "dtype": case.dtype,
                "trans_a": str(int(case.trans_a)),
                "trans_b": str(int(case.trans_b)),
                "max_cores": "20",
                "source": source,
                "candidate_role": "direct_measurement",
                "comparison_role": role,
                "decision_provenance": (
                    "modified_r5_l2_geometry_only"
                    if use_modified
                    else "unchanged_reference_packet_no_performance_credit"
                ),
                "performance_credit_eligible": str(int(use_modified)),
                "original_selector_executed": "0",
                "experimental_rule_applied": str(int(use_modified)),
                "modified_fields": (
                    "|".join(R5_CHANGED_FIELDS) if use_modified else ""
                ),
                "unchanged_fields": (
                    unchanged_r5_fields if use_modified else all_fields
                ),
                "measurement_block": measurement_block,
                "kernel_path_status": "unchanged_shared_direct_kernel",
                "kernel_path_performance_credit_eligible": "0",
                "execution_mode": "base",
                **{field: str(values[field]) for field in SCHEDULE_FIELDS},
                "model_schedule_sha256": (
                    modified_digest if use_modified else reference_digest
                ),
                "model_kernel_suffix": str(case.kernel_suffix),
                "model_kernel_family": "base",
                "model_input_source": (
                    "shape_hardware_reference_packet_and_standalone_r5_equation"
                    if use_modified else "predeclared_control_packet"
                ),
                "tiling_parameter_origin": (
                    "r5_modified_two_fields_with_all_other_fields_marked_unchanged"
                    if use_modified
                    else "unchanged_reference_not_new_code"
                ),
                "selection_basis": (
                    "deterministic_r5_guard_no_search"
                    if use_modified else "paired_control_only"
                ),
                "tiling_signature": ":".join(
                    str(values[field]) for field in SCHEDULE_FIELDS
                ),
                "is_reserve": "0",
                "required_successful_tilings": "4",
                "performance_claim_scope": (
                    "r5_l2_geometry_delta_only" if use_modified else "none"
                ),
            }
            candidate_rows.append(row)
        audits.append({
            "schema": "matmul_r5_l2_pair_freeze_v1",
            "workload_id": case.workload_id,
            "record_type": "paired_packets_frozen",
            "original_selector_executed": False,
            "installed_reference_consumed_by_r5": False,
            "latency_consumed_by_r5": False,
            "experimental_rule_applied": True,
            "changed_fields": list(R5_CHANGED_FIELDS),
            "unchanged_fields": [
                field for field in SCHEDULE_FIELDS
                if field not in R5_CHANGED_FIELDS
            ],
            "reference_schedule_sha256": reference_digest,
            "modified_schedule_sha256": modified_digest,
            "execution_order": [name for name, _ in blocks],
            "performance_credit_policy":
                "only_modified_r5_rows_are_eligible_reference_rows_never_are",
        })
    return workload_rows, candidate_rows, audits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--aic-cores", type=int, required=True)
    parser.add_argument("--l2-bytes", type=int, required=True)
    args = parser.parse_args()
    workloads, candidates, audits = build(
        aic_cores=args.aic_cores, l2_bytes=args.l2_bytes
    )
    write_csv(args.workloads, workloads)
    write_csv(args.candidates, candidates)
    temporary = args.audit.with_name(args.audit.name + f".tmp.{os.getpid()}")
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8") as stream:
        for record in audits:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            stream.write("\n")
    temporary.replace(args.audit)
    print(
        "R5_PAIR_FROZEN "
        f"shapes={len(workloads)} direct_measurements={len(candidates)} "
        "original_selector_executed=0 changed_fields=2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
