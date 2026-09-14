#!/usr/bin/env python3
"""Attest and compare one independent packet against public MatMul per shape."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 200
EXPECTED_SAMPLES = 15
EXPECTED_WARMUP = 3
EXPECTED_REPEAT = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sample_map(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(path):
        values[row["workload_id"]].append(float(row["latency_ms"]))
    return dict(values)


def direct_map(path: Path) -> dict[str, dict]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("DIRECT_MATMUL_RESULT "):
            continue
        row = json.loads(line.split(" ", 1)[1])
        workload_id = row["workload_id"]
        if workload_id in values:
            raise RuntimeError(f"duplicate direct result for {workload_id}")
        values[workload_id] = row
    return values


def valid_samples(values: list[float]) -> bool:
    return len(values) == EXPECTED_SAMPLES and all(
        math.isfinite(value) and value > 0 for value in values
    )


def separation(candidate: list[float], official: list[float]) -> str:
    if max(candidate) < min(official):
        return "CLEAR_CANDIDATE_WINNER"
    if min(candidate) > max(official):
        return "CLEAR_OFFICIAL_WINNER"
    return "OVERLAPPING_SAMPLES"


def attest_candidate(result: dict, manifest: dict[str, str]) -> list[float]:
    samples = [float(value) for value in result.get("samples_ms", [])]
    valid = (
        result.get("status") == "success" and
        result.get("candidate_role") == "independent_formula_tiling" and
        result.get("measurement_source") == "direct_tiling_buffer" and
        result.get("tiling_applied") == 1 and
        result.get("full_output_validated") == 1 and
        result.get("warmup") == EXPECTED_WARMUP and
        result.get("repeat") == EXPECTED_REPEAT and
        result.get("samples") == EXPECTED_SAMPLES and
        valid_samples(samples) and
        result.get("actual_tiling_sha256") == manifest["tiling_sha256"] and
        str(result.get("actual_kernel_suffix")) == manifest["kernel_suffix"] and
        str(result.get("actual_block_dim")) == manifest["used_core_num"] and
        math.isclose(float(result["median_ms"]), statistics.median(samples), rel_tol=1e-9)
    )
    if not valid:
        raise RuntimeError(f"candidate attestation failed for {manifest['workload_id']}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--official-samples", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["workload_id"]: row for row in read_csv(args.manifest)}
    official = {row["workload_id"]: row for row in read_csv(args.official_profile)}
    official_samples = sample_map(args.official_samples)
    direct = direct_map(args.runner_log)
    selection = {
        row["workload_id"]: row for row in (
            json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    identities = set(manifest)
    if not (
        len(identities) == EXPECTED_SHAPES and set(official) == identities and
        set(official_samples) == identities and set(direct) == identities and
        set(selection) == identities
    ):
        raise RuntimeError(
            "identity mismatch "
            f"manifest={len(manifest)} official={len(official)} "
            f"official_samples={len(official_samples)} direct={len(direct)} "
            f"selection={len(selection)}"
        )

    output = []
    for workload_id, expected in manifest.items():
        candidate_samples = attest_candidate(direct[workload_id], expected)
        reference = official[workload_id]
        reference_samples = official_samples[workload_id]
        if not (
            reference.get("source") == "installed_aclnn_matmul_public_api" and
            reference.get("candidate_role") == "installed_operator_reference" and
            reference.get("success") == "1" and reference.get("preflight_passed") == "1" and
            int(reference.get("warmup", 0)) == EXPECTED_WARMUP and
            int(reference.get("repeat", 0)) == EXPECTED_REPEAT and
            int(reference.get("samples", 0)) == EXPECTED_SAMPLES and
            valid_samples(reference_samples)
        ):
            raise RuntimeError(f"official attestation failed for {workload_id}")
        candidate_median = statistics.median(candidate_samples)
        official_median = statistics.median(reference_samples)
        if not math.isclose(float(reference["median_ms"]), official_median, rel_tol=1e-9):
            raise RuntimeError(f"official median mismatch for {workload_id}")
        metadata = selection[workload_id]
        output.append({
            "workload_id": workload_id,
            "family": metadata["formula_family"],
            "validation_group": metadata["validation_group"],
            "official_suffix_audit": metadata["official_suffix_audit"],
            "kernel_mode": metadata["kernel_mode"],
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]),
            "trans_b": int(expected["trans_b"]),
            "used_cores": int(expected["used_core_num"]),
            "official_ms": official_median,
            "candidate_ms": candidate_median,
            "delta_pct": 100.0 * (candidate_median / official_median - 1.0),
            "separation": separation(candidate_samples, reference_samples),
            "correctness": "PASS_BOTH_CURRENT_RUN_OUTPUTS",
        })

    result = {
        "schema": "independent_formula_tiling_validation_v1",
        "status": "complete",
        "selection_uses_measurements": False,
        "all_current_outputs_validated": True,
        "shape_count": len(output),
        "shapes": output,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print("INDEPENDENT_RESULT_ANALYSIS passed shapes=200 outputs_validated=200")


if __name__ == "__main__":
    main()
