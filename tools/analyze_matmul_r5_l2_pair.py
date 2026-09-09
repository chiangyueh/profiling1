#!/usr/bin/env python3
"""Analyze paired R5 L2 measurements without crediting unchanged paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

from generate_matmul_r5_l2_pair import R5_CHANGED_FIELDS, SCHEDULE_FIELDS


REFERENCE_ROLE = "unchanged_direct_reference"
MODIFIED_ROLE = "modified_r5_l2_candidate"
EXPECTED_BLOCKS = (
    "unchanged_reference_a",
    "modified_r5_a",
    "modified_r5_b",
    "unchanged_reference_b",
)


def truthy(value: object) -> bool:
    return str(value or "").lower() in {"1", "true", "yes", "on"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def finite_positive(value: object) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"invalid positive latency: {value!r}")
    return result


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute median of an empty sample set")
    return float(statistics.median(values))


def relative_delta(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def classification(delta_percent: float) -> str:
    if delta_percent < -1.0:
        return "modified_r5_faster"
    if delta_percent > 1.0:
        return "modified_r5_slower"
    return "within_1pct"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--official-samples", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    actual_candidate_sha = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    if actual_candidate_sha != args.candidate_sha256:
        raise ValueError("candidate CSV changed after the pre-measurement freeze")

    workloads = read_csv(args.workloads)
    candidates = read_csv(args.candidates)
    profiles = read_csv(args.profile)
    samples = read_csv(args.samples)
    official_profiles = read_csv(args.official_profile)
    official_samples = read_csv(args.official_samples)
    audits = [
        json.loads(line) for line in args.audit.read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = [row["workload_id"] for row in workloads]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("R5 campaign must contain exactly three unique workloads")
    if len(candidates) != 12:
        raise ValueError("R5 campaign must contain four direct rows per workload")

    candidates_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        candidates_by_id[row["workload_id"]].append(row)
    profiles_by_key = {
        (row["workload_id"], row["rank"]): row
        for row in profiles if row.get("candidate_role") == "direct_measurement"
    }
    samples_by_key: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in samples:
        if row.get("candidate_role") != "direct_measurement":
            continue
        samples_by_key[(row["workload_id"], row["rank"])].append(
            finite_positive(row["latency_ms"])
        )
    official_by_id = {row["workload_id"]: row for row in official_profiles}
    official_samples_by_id: dict[str, list[float]] = defaultdict(list)
    for row in official_samples:
        official_samples_by_id[row["workload_id"]].append(
            finite_positive(row["latency_ms"])
        )
    audit_by_id = {row["workload_id"]: row for row in audits}
    if (
        set(candidates_by_id) != set(ids)
        or set(official_by_id) != set(ids)
        or set(official_samples_by_id) != set(ids)
        or set(audit_by_id) != set(ids)
    ):
        raise ValueError("measurement identities do not match the frozen workload set")

    results: list[dict] = []
    counts = {
        "modified_r5_faster": 0,
        "within_1pct": 0,
        "modified_r5_slower": 0,
    }
    for workload in workloads:
        workload_id = workload["workload_id"]
        rows = candidates_by_id[workload_id]
        rows.sort(key=lambda row: int(row["rank"]))
        if [int(row["rank"]) for row in rows] != [1, 2, 3, 4]:
            raise ValueError(f"{workload_id}: invalid paired ranks")
        if tuple(row["measurement_block"] for row in rows) != EXPECTED_BLOCKS:
            raise ValueError(f"{workload_id}: execution order is not ABBA")
        if any(
            row.get("candidate_role") != "direct_measurement"
            or truthy(row.get("original_selector_executed"))
            or row.get("kernel_path_status") != "unchanged_shared_direct_kernel"
            or truthy(row.get("kernel_path_performance_credit_eligible"))
            for row in rows
        ):
            raise ValueError(
                f"{workload_id}: selector/kernel provenance contract failed"
            )
        reference_rows = [
            row for row in rows if row["comparison_role"] == REFERENCE_ROLE
        ]
        modified_rows = [
            row for row in rows if row["comparison_role"] == MODIFIED_ROLE
        ]
        if len(reference_rows) != 2 or len(modified_rows) != 2:
            raise ValueError(f"{workload_id}: expected two rows for each paired role")
        if any(truthy(row["performance_credit_eligible"]) for row in reference_rows):
            raise ValueError(f"{workload_id}: unchanged reference was credit-eligible")
        if any(
            not truthy(row["performance_credit_eligible"])
            or not truthy(row["experimental_rule_applied"])
            for row in modified_rows
        ):
            raise ValueError(f"{workload_id}: modified rows lack R5 attestation")
        reference_schedule = {
            field: reference_rows[0][field] for field in SCHEDULE_FIELDS
        }
        modified_schedule = {
            field: modified_rows[0][field] for field in SCHEDULE_FIELDS
        }
        changed = tuple(
            field for field in SCHEDULE_FIELDS
            if reference_schedule[field] != modified_schedule[field]
        )
        if changed != R5_CHANGED_FIELDS:
            raise ValueError(
                f"{workload_id}: changed fields are {changed}, "
                f"expected {R5_CHANGED_FIELDS}"
            )
        if (
            reference_rows[0]["model_schedule_sha256"]
            != reference_rows[1]["model_schedule_sha256"]
            or modified_rows[0]["model_schedule_sha256"]
            != modified_rows[1]["model_schedule_sha256"]
            or reference_rows[0]["model_schedule_sha256"]
            == modified_rows[0]["model_schedule_sha256"]
        ):
            raise ValueError(f"{workload_id}: schedule identity contract failed")
        audit = audit_by_id[workload_id]
        if (
            audit.get("original_selector_executed") is not False
            or audit.get("changed_fields") != list(R5_CHANGED_FIELDS)
            or audit.get("reference_schedule_sha256")
                != reference_rows[0]["model_schedule_sha256"]
            or audit.get("modified_schedule_sha256")
                != modified_rows[0]["model_schedule_sha256"]
        ):
            raise ValueError(f"{workload_id}: freeze audit mismatch")

        reference_values: list[float] = []
        modified_values: list[float] = []
        block_medians: dict[str, float] = {}
        raw_tiling_hashes: dict[str, list[str]] = {
            REFERENCE_ROLE: [], MODIFIED_ROLE: [],
        }
        for row in rows:
            key = (workload_id, row["rank"])
            profile = profiles_by_key.get(key)
            values = samples_by_key.get(key, [])
            if profile is None or not truthy(profile.get("success")) or not values:
                raise ValueError(f"{workload_id}: incomplete direct measurement rank={row['rank']}")
            if (
                profile.get("comparison_role") != row["comparison_role"]
                or profile.get("measurement_block") != row["measurement_block"]
                or profile.get("model_schedule_sha256")
                    != row["model_schedule_sha256"]
                or not truthy(profile.get("tiling_applied"))
                or not truthy(profile.get("full_output_validated"))
            ):
                raise ValueError(f"{workload_id}: direct attestation mismatch rank={row['rank']}")
            block_medians[row["measurement_block"]] = median(values)
            raw_tiling_hashes[row["comparison_role"]].append(
                profile["actual_tiling_sha256"]
            )
            if row["comparison_role"] == REFERENCE_ROLE:
                reference_values.extend(values)
            else:
                modified_values.extend(values)
        if (
            len(set(raw_tiling_hashes[REFERENCE_ROLE])) != 1
            or len(set(raw_tiling_hashes[MODIFIED_ROLE])) != 1
            or raw_tiling_hashes[REFERENCE_ROLE][0]
                == raw_tiling_hashes[MODIFIED_ROLE][0]
        ):
            raise ValueError(f"{workload_id}: executed packet identity contract failed")

        reference_median = median(reference_values)
        modified_median = median(modified_values)
        official_values = official_samples_by_id[workload_id]
        official_median = median(official_values)
        official_profile = official_by_id[workload_id]
        if (
            official_profile.get("source") != "installed_aclnn_matmul_public_api"
            or not truthy(official_profile.get("success"))
        ):
            raise ValueError(f"{workload_id}: installed reference is invalid")
        delta = relative_delta(modified_median, reference_median)
        outcome = classification(delta)
        counts[outcome] += 1
        results.append({
            "workload_id": workload_id,
            "shape": {
                key: workload[key]
                for key in ("m", "n", "k", "dtype", "trans_a", "trans_b")
            },
            "changed_fields": list(R5_CHANGED_FIELDS),
            "unchanged_field_count": len(SCHEDULE_FIELDS) - len(R5_CHANGED_FIELDS),
            "original_selector_executed": False,
            "kernel_path_status": "unchanged_shared_direct_kernel",
            "kernel_path_performance_credit_eligible": False,
            "unchanged_reference_performance_credit_eligible": False,
            "modified_r5_performance_credit_eligible": True,
            "reference_direct_median_ms": reference_median,
            "modified_r5_median_ms": modified_median,
            "modified_vs_reference_delta_percent": delta,
            "classification": outcome,
            "installed_public_reference_median_ms": official_median,
            "modified_vs_installed_reference_delta_percent":
                relative_delta(modified_median, official_median),
            "installed_reference_role":
                "context_only_unknown_internal_tiling_not_used_for_generation",
            "block_medians_ms": block_medians,
            "reference_actual_tiling_sha256":
                raw_tiling_hashes[REFERENCE_ROLE][0],
            "modified_actual_tiling_sha256":
                raw_tiling_hashes[MODIFIED_ROLE][0],
        })
        print(
            "R5_PAIR_RESULT "
            f"workload={workload_id} reference_ms={reference_median:.9g} "
            f"modified_ms={modified_median:.9g} delta_percent={delta:.6g} "
            f"classification={outcome}",
            flush=True,
        )

    output = {
        "schema": "matmul_r5_l2_pair_analysis_v1",
        "status": "complete",
        "candidate_sha256": actual_candidate_sha,
        "scope": "three_shape_r5_l2_decision_validation_only",
        "global_selector_claim_allowed": False,
        "provenance_policy": {
            "original_selector_executed": False,
            "unchanged_reference_rows_executed": True,
            "unchanged_reference_rows_marked": True,
            "unchanged_reference_rows_performance_credit_eligible": False,
            "unchanged_shared_kernel_performance_credit_eligible": False,
            "installed_public_reference_used_for_generation": False,
            "only_modified_fields": list(R5_CHANGED_FIELDS),
            "unchanged_fields_explicitly_marked": True,
        },
        "counts": counts,
        "results": results,
    }
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(output, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        "R5_PAIR_ANALYSIS_COMPLETE "
        f"faster={counts['modified_r5_faster']} "
        f"within_1pct={counts['within_1pct']} "
        f"slower={counts['modified_r5_slower']} "
        "global_selector_claim_allowed=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
