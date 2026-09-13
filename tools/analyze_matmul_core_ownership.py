#!/usr/bin/env python3
"""Validate and compare paired direct packets for core ownership rules."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_BASELINE_SHAPES = 60
EXPECTED_PAIRED_SHAPES = 20
EXPECTED_SAMPLES = 15
EXPECTED_WARMUP = 3
EXPECTED_REPEAT = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def direct_results(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("DIRECT_MATMUL_RESULT "):
            continue
        result = json.loads(line.split(" ", 1)[1])
        key = result["workload_id"]
        if key in rows:
            raise RuntimeError(f"duplicate direct result {key}")
        rows[key] = result
    return rows


def finite_positive(values: list[float]) -> bool:
    return all(math.isfinite(value) and value > 0 for value in values)


def attest(result: dict, manifest: dict, role: str) -> list[float]:
    samples = [float(value) for value in result.get("samples_ms", [])]
    if not (
        result.get("status") == "success" and
        result.get("candidate_role") == role and
        result.get("measurement_source") == "direct_tiling_buffer" and
        result.get("tiling_applied") == 1 and
        result.get("full_output_validated") == 1 and
        result.get("warmup") == EXPECTED_WARMUP and
        result.get("repeat") == EXPECTED_REPEAT and
        result.get("samples") == EXPECTED_SAMPLES and
        len(samples) == EXPECTED_SAMPLES and finite_positive(samples) and
        result.get("actual_tiling_sha256") == manifest["tiling_sha256"] and
        str(result.get("actual_kernel_suffix")) == manifest["kernel_suffix"] and
        str(result.get("actual_block_dim")) == manifest["used_core_num"]
    ):
        raise RuntimeError(
            f"{manifest['workload_id']}: {role} measurement attestation failed"
        )
    median = statistics.median(samples)
    if not math.isclose(float(result["median_ms"]), median, rel_tol=1e-9):
        raise RuntimeError(f"{manifest['workload_id']}: median mismatch")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    baseline_manifest = {
        row["workload_id"]: row for row in read_csv(args.baseline_manifest)
    }
    candidate_manifest = {
        row["workload_id"]: row for row in read_csv(args.candidate_manifest)
    }
    baseline = direct_results(args.baseline_log)
    candidate = direct_results(args.candidate_log)
    selections = {
        row["workload_id"]: row for row in (
            json.loads(line) for line in args.selection.read_text().splitlines()
            if line
        )
    }
    audits = [
        json.loads(line) for line in args.audit.read_text().splitlines() if line
    ]
    identities = set(baseline_manifest)
    paired_identities = set(candidate_manifest)
    if not (
        len(identities) == EXPECTED_BASELINE_SHAPES and
        len(paired_identities) == EXPECTED_PAIRED_SHAPES and
        paired_identities < identities and set(baseline) == identities and
        set(candidate) == paired_identities and set(selections) == identities
    ):
        raise RuntimeError("baseline/paired direct identity coverage mismatch")
    audit_counts = {}
    for row in audits:
        suffix = int(row["suffix"])
        audit_counts[suffix] = audit_counts.get(suffix, 0) + 1
    if len(audit_counts) != 12 or set(audit_counts.values()) != {5}:
        raise RuntimeError("core audit does not cover all twelve suffixes")

    output_rows = []
    for workload_id in sorted(identities):
        old_manifest = baseline_manifest[workload_id]
        old_samples = attest(
            baseline[workload_id], old_manifest, "family_core_baseline"
        )
        selection = selections[workload_id]
        old_cores = int(old_manifest["used_core_num"])
        old_median = statistics.median(old_samples)
        if workload_id in paired_identities:
            new_manifest = candidate_manifest[workload_id]
            if any(
                old_manifest[name] != new_manifest[name]
                for name in ("m", "n", "k", "dtype", "trans_a", "trans_b", "kernel_suffix")
            ):
                raise RuntimeError(f"{workload_id}: paired input drift")
            new_samples = attest(
                candidate[workload_id], new_manifest, "family_core_candidate"
            )
            equation = selection["theory"]["strict_dominance_certificate"]
            new_cores = int(new_manifest["used_core_num"])
            if not (
                equation["changed_packet_words"] == ["usedCoreNum"] and
                int(equation["source_cores"]) == old_cores and
                int(equation["required_cores"]) == new_cores and
                int(equation["eliminated_cores"]) == old_cores - new_cores > 0 and
                equation["same_cube_tile_geometry"] is True and
                equation["same_l2_schedule"] is True and
                equation["same_nd2nz_tile_geometry"] is True and
                int(equation["new_MMAD_commands"]) == 0 and
                int(equation["new_conversion_tasks"]) == 0 and
                int(equation["new_output_tasks"]) == 0 and
                int(equation["new_workspace_bytes"]) == 0
            ):
                raise RuntimeError(f"{workload_id}: deletion proof failed")
            new_median = statistics.median(new_samples)
            delta = 100.0 * (new_median / old_median - 1.0)
            if max(new_samples) < min(old_samples):
                separation = "CLEAR_CANDIDATE_WINNER"
            elif min(new_samples) > max(old_samples):
                separation = "CLEAR_BASELINE_WINNER"
            else:
                separation = "OVERLAPPING_SAMPLES"
            winner = "candidate" if new_median < old_median else "baseline"
            proof_kind = equation["proof_kind"]
        else:
            if not (
                selection["status"] == "CORE_OWNERSHIP_AUDITED_UNCHANGED" and
                selection["baseline_equivalent"] is True and
                selection["improved"] is None and
                selection["core_plan"]["required_cores"] == old_cores
            ):
                raise RuntimeError(f"{workload_id}: retained-core proof failed")
            new_cores = old_cores
            new_median = None
            delta = None
            separation = "CORE_COUNT_RETAINED"
            winner = "not_applicable"
            proof_kind = selection["core_plan"]["proof_kind"]
        output_rows.append({
            "workload_id": workload_id,
            "branch": selection["core_plan"]["branch"],
            "kernel_suffix": int(old_manifest["kernel_suffix"]),
            "m": int(old_manifest["m"]), "n": int(old_manifest["n"]),
            "k": int(old_manifest["k"]), "dtype": old_manifest["dtype"],
            "trans_a": int(old_manifest["trans_a"]),
            "trans_b": int(old_manifest["trans_b"]),
            "baseline_median_ms": old_median,
            "candidate_median_ms": new_median,
            "delta_pct": delta,
            "speedup": old_median / new_median if new_median is not None else None,
            "median_winner": winner,
            "sample_separation": separation,
            "cores_before": old_cores, "cores_after": new_cores,
            "eliminated_zero_owner_cores": old_cores - new_cores,
            "proof_kind": proof_kind,
            "correctness": (
                "PASS_PAIRED_DIRECT_FULL_OUTPUT" if new_median is not None
                else "PASS_DIRECT_BASELINE_FULL_OUTPUT_AND_RETAINED_CORE_PROOF"
            ),
        })

    result = {
        "schema": "matmul_c220_twelve_suffix_core_ownership_v1",
        "comparison_basis": "same_direct_kernel_same_suffix_only_usedCoreNum_differs",
        "branch_audit": audits,
        "shapes": output_rows,
        "aggregate": {
            "audited_suffixes": len({int(row["suffix"]) for row in audits}),
            "measured_shapes": len(output_rows),
            "paired_shapes": len(paired_identities),
            "retained_core_shapes": len(output_rows) - len(paired_identities),
            "measured_changed_suffixes": len({row["kernel_suffix"] for row in output_rows if row["candidate_median_ms"] is not None}),
            "candidate_wins": sum(row["median_winner"] == "candidate" for row in output_rows),
            "baseline_wins": sum(row["median_winner"] == "baseline" for row in output_rows),
            "clear_candidate_wins": sum(row["sample_separation"] == "CLEAR_CANDIDATE_WINNER" for row in output_rows),
            "clear_baseline_wins": sum(row["sample_separation"] == "CLEAR_BASELINE_WINNER" for row in output_rows),
            "overlap": sum(row["sample_separation"] == "OVERLAPPING_SAMPLES" for row in output_rows),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        "CORE_OWNERSHIP_ANALYSIS_COMPLETE "
        f"audited_suffixes={result['aggregate']['audited_suffixes']} "
        f"measured_shapes={len(output_rows)} paired_shapes={len(paired_identities)} "
        f"candidate_wins={result['aggregate']['candidate_wins']} "
        f"baseline_wins={result['aggregate']['baseline_wins']}"
    )


if __name__ == "__main__":
    main()
