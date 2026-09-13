#!/usr/bin/env python3
"""Validate and compare paired direct packets for core ownership rules."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 20
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
    if not (
        len(identities) == EXPECTED_SHAPES and
        set(candidate_manifest) == identities and set(baseline) == identities and
        set(candidate) == identities and set(selections) == identities
    ):
        raise RuntimeError("paired direct identity coverage mismatch")
    if len({int(row["suffix"]) for row in audits}) != 12:
        raise RuntimeError("core audit does not cover all twelve suffixes")

    output_rows = []
    for workload_id in sorted(identities):
        old_manifest = baseline_manifest[workload_id]
        new_manifest = candidate_manifest[workload_id]
        if any(
            old_manifest[name] != new_manifest[name]
            for name in ("m", "n", "k", "dtype", "trans_a", "trans_b", "kernel_suffix")
        ):
            raise RuntimeError(f"{workload_id}: paired input drift")
        old_samples = attest(
            baseline[workload_id], old_manifest, "family_core_baseline"
        )
        new_samples = attest(
            candidate[workload_id], new_manifest, "family_core_candidate"
        )
        selection = selections[workload_id]
        equation = selection["theory"]["strict_dominance_certificate"]
        old_cores = int(old_manifest["used_core_num"])
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
        old_median = statistics.median(old_samples)
        new_median = statistics.median(new_samples)
        delta = 100.0 * (new_median / old_median - 1.0)
        if max(new_samples) < min(old_samples):
            separation = "CLEAR_CANDIDATE_WINNER"
        elif min(new_samples) > max(old_samples):
            separation = "CLEAR_BASELINE_WINNER"
        else:
            separation = "OVERLAPPING_SAMPLES"
        output_rows.append({
            "workload_id": workload_id,
            "branch": selection["core_plan"]["branch"],
            "kernel_suffix": int(new_manifest["kernel_suffix"]),
            "m": int(new_manifest["m"]), "n": int(new_manifest["n"]),
            "k": int(new_manifest["k"]), "dtype": new_manifest["dtype"],
            "trans_a": int(new_manifest["trans_a"]),
            "trans_b": int(new_manifest["trans_b"]),
            "baseline_median_ms": old_median,
            "candidate_median_ms": new_median,
            "delta_pct": delta, "speedup": old_median / new_median,
            "median_winner": "candidate" if new_median < old_median else "baseline",
            "sample_separation": separation,
            "cores_before": old_cores, "cores_after": new_cores,
            "eliminated_zero_owner_cores": old_cores - new_cores,
            "proof_kind": equation["proof_kind"],
            "correctness": "PASS_PAIRED_DIRECT_FULL_OUTPUT",
        })

    result = {
        "schema": "matmul_c220_twelve_suffix_core_ownership_v1",
        "comparison_basis": "same_direct_kernel_same_suffix_only_usedCoreNum_differs",
        "branch_audit": audits,
        "shapes": output_rows,
        "aggregate": {
            "audited_suffixes": len({int(row["suffix"]) for row in audits}),
            "measured_shapes": len(output_rows),
            "measured_changed_suffixes": len({row["kernel_suffix"] for row in output_rows}),
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
        f"measured_shapes={len(output_rows)} "
        f"candidate_wins={result['aggregate']['candidate_wins']} "
        f"baseline_wins={result['aggregate']['baseline_wins']}"
    )


if __name__ == "__main__":
    main()
