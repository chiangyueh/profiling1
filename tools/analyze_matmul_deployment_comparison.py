#!/usr/bin/env python3
"""Validate and summarize one model-selected tiling versus one baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path


TILING_COLUMNS = (
    "used_core_num", "single_core_m", "single_core_n", "single_core_k",
    "base_m", "base_n", "base_k", "depth_a1", "depth_b1", "step_m",
    "step_n", "iterate_order", "step_ka", "step_kb", "db_l0a",
    "db_l0b", "db_l0c", "bank_l2_m_tile_count",
    "bank_l2_n_tile_count", "bank_l2_m_tile_block",
    "bank_l2_n_tile_block", "bank_l2_iterate_order", "bank_tiling_enable",
)


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def unique_by_id(rows, description: str):
    result = {}
    for row in rows:
        workload_id = row.get("workload_id", "")
        if not workload_id or workload_id in result:
            raise RuntimeError(f"{description} has a missing or duplicate workload identity")
        result[workload_id] = row
    return result


def positive_latency(row: dict[str, str], description: str) -> float:
    try:
        value = float(row["median_ms"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{description} has no numeric median_ms") from error
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"{description} has an invalid median_ms")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    workloads = unique_by_id(read_rows(args.workloads), "workloads")
    candidates = unique_by_id(read_rows(args.candidates), "candidates")
    direct_rows = [
        row for row in read_rows(args.profile)
        if row.get("candidate_role") == "searched"
    ]
    direct = unique_by_id(direct_rows, "direct profile")
    baseline = unique_by_id(read_rows(args.official_profile), "baseline profile")
    audit_records = [
        json.loads(line)
        for line in args.selection_audit.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = unique_by_id(audit_records, "selection audit")
    identities = set(workloads)
    if not identities or any(set(values) != identities for values in (
        candidates, direct, baseline, audit,
    )):
        raise RuntimeError("comparison inputs do not contain the same workload identities")

    observed_sha = __import__("hashlib").sha256(args.candidates.read_bytes()).hexdigest()
    if observed_sha != args.candidate_sha256:
        raise RuntimeError("candidate CSV changed after model selection was frozen")

    comparisons = []
    deltas = []
    faster = tied = slower = 0
    for workload_id in workloads:
        workload = workloads[workload_id]
        candidate = candidates[workload_id]
        measured = direct[workload_id]
        reference = baseline[workload_id]
        selection = audit[workload_id]
        if not (
            candidate.get("rank") == "1"
            and candidate.get("candidate_role") == "searched"
            and candidate.get("source") == "independent_hardware_cost_model"
            and candidate.get("model_input_source") == "shape_hardware_and_cost_model_only"
            and candidate.get("tiling_parameter_origin") == "independent_model_generation"
            and candidate.get("is_reserve") == "0"
            and candidate.get("required_successful_tilings") == "1"
            and truthy(measured.get("success"))
            and truthy(measured.get("tiling_applied"))
            and truthy(measured.get("full_output_validated"))
            and measured.get("measurement_source") == "direct_tiling_buffer"
            and measured.get("model_schedule_sha256")
                == candidate.get("model_schedule_sha256")
            and measured.get("actual_tiling_sha256")
            and measured.get("actual_block_dim") == candidate.get("used_core_num")
            and measured.get("actual_kernel_suffix")
                == candidate.get("model_kernel_suffix")
            and reference.get("source") == "installed_aclnn_matmul_public_api"
            and reference.get("candidate_role") == "installed_operator_reference"
            and truthy(reference.get("success"))
            and truthy(reference.get("preflight_passed"))
            and selection.get("record_type") == "model_top1_frozen"
            and selection.get("baseline_data_consumed") is False
            and selection.get("measured_latency_consumed") is False
            and selection.get("selected", {}).get("model_schedule_sha256")
                == candidate.get("model_schedule_sha256")
        ):
            raise RuntimeError(f"{workload_id}: execution or independence attestation failed")
        model_ms = positive_latency(measured, f"{workload_id} model")
        reference_ms = positive_latency(reference, f"{workload_id} baseline")
        delta_pct = (model_ms / reference_ms - 1.0) * 100.0
        deltas.append(delta_pct)
        if delta_pct < -1.0:
            verdict = "model_faster"
            faster += 1
        elif delta_pct > 1.0:
            verdict = "baseline_faster"
            slower += 1
        else:
            verdict = "within_one_percent"
            tied += 1
        tiling = {name: int(candidate[name]) for name in TILING_COLUMNS}
        comparison = {
            "workload_id": workload_id,
            "shape": [int(workload["m"]), int(workload["n"]), int(workload["k"])],
            "dtype": workload["dtype"],
            "trans_a": truthy(workload["trans_a"]),
            "trans_b": truthy(workload["trans_b"]),
            "model_selected": {
                "median_ms": model_ms,
                "predicted_cycles": float(candidate["new_model_cycles"]),
                "selection_ms": float(candidate["tiling_solver_total_ms"]),
                "generated_candidate_count": int(candidate["generated_candidate_count"]),
                "legal_candidate_count": int(candidate["legal_candidate_count"]),
                "model_schedule_sha256": candidate["model_schedule_sha256"],
                "actual_tiling_sha256": measured["actual_tiling_sha256"],
                "kernel_suffix": int(candidate["model_kernel_suffix"]),
                "execution_mode": candidate["execution_mode"],
                "tiling": tiling,
            },
            "installed_matmul_reference": {
                "median_ms": reference_ms,
                "measurement_source": "separate_installed_aclnn_matmul_public_api",
                "tiling_parameters_captured": False,
            },
            "model_delta_pct": delta_pct,
            "verdict": verdict,
        }
        comparisons.append(comparison)
        print(
            "DEPLOYMENT_COMPARISON "
            f"workload={workload_id} model_ms={model_ms:.9g} "
            f"baseline_ms={reference_ms:.9g} delta_pct={delta_pct:.3f} "
            f"verdict={verdict} model_base="
            f"{tiling['base_m']}x{tiling['base_n']}x{tiling['base_k']} "
            f"cores={tiling['used_core_num']}",
            flush=True,
        )

    payload = {
        "schema": "matmul_deployment_comparison_v1",
        "status": "complete",
        "candidate_sha256": observed_sha,
        "shape_count": len(workloads),
        "records": 2 * len(workloads),
        "model_candidates_per_shape": 1,
        "installed_references_per_shape": 1,
        "independence": {
            "candidate_generation_precedes_baseline_execution": True,
            "baseline_data_consumed_by_model": False,
            "measured_latency_consumed_by_model": False,
            "candidate_set_hash_verified_after_measurement": True,
        },
        "aggregate": {
            "model_faster": faster,
            "within_one_percent": tied,
            "baseline_faster": slower,
            "median_model_delta_pct": statistics.median(deltas),
            "minimum_model_delta_pct": min(deltas),
            "maximum_model_delta_pct": max(deltas),
            "total_internal_candidates_generated": sum(
                item["model_selected"]["generated_candidate_count"]
                for item in comparisons
            ),
            "total_model_selection_ms": sum(
                item["model_selected"]["selection_ms"] for item in comparisons
            ),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        "DEPLOYMENT_ANALYSIS_COMPLETE "
        f"shapes={len(workloads)} records={2 * len(workloads)} "
        f"model_faster={faster} within_one_percent={tied} baseline_faster={slower}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"fatal: {error}", flush=True)
        raise SystemExit(1)
