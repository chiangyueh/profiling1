#!/usr/bin/env python3
"""Attest same-campaign official, backport-control, and novel measurements."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 200
EXPECTED_CONTROL_SHAPES = 100
EXPECTED_SAMPLES = 15
EXPECTED_WARMUP = 3
EXPECTED_REPEAT = 10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def samples_by_id(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(path):
        rows[row["workload_id"]].append(float(row["latency_ms"]))
    return dict(rows)


def direct_results(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("DIRECT_MATMUL_RESULT "):
            continue
        row = json.loads(line.split(" ", 1)[1])
        key = (row["workload_id"], int(row.get("actual_kernel_suffix", -1)))
        if key in rows:
            raise RuntimeError(f"duplicate direct result {key}")
        rows[key] = row
    return rows


def finite_positive(values: list[float]) -> bool:
    return len(values) == EXPECTED_SAMPLES and all(
        math.isfinite(value) and value > 0 for value in values
    )


def separation(candidate: list[float], reference: list[float]) -> str:
    if max(candidate) < min(reference):
        return "CLEAR_CANDIDATE_WINNER"
    if min(candidate) > max(reference):
        return "CLEAR_REFERENCE_WINNER"
    return "OVERLAPPING_SAMPLES"


def attest_direct(row: dict, manifest: dict[str, str]) -> list[float]:
    samples = [float(value) for value in row.get("samples_ms", [])]
    valid = (
        row.get("status") == "success" and
        row.get("candidate_role") == "independent_experimental_family" and
        row.get("measurement_source") == "direct_tiling_buffer" and
        row.get("tiling_applied") == 1 and row.get("full_output_validated") == 1 and
        row.get("warmup") == EXPECTED_WARMUP and
        row.get("repeat") == EXPECTED_REPEAT and
        row.get("samples") == EXPECTED_SAMPLES and finite_positive(samples) and
        row.get("actual_tiling_sha256") == manifest["tiling_sha256"] and
        str(row.get("actual_kernel_suffix")) == manifest["kernel_suffix"] and
        str(row.get("actual_block_dim")) == manifest["used_core_num"] and
        math.isclose(float(row["median_ms"]), statistics.median(samples), rel_tol=1e-9)
    )
    if not valid:
        raise RuntimeError(f"direct measurement attestation failed for {manifest['workload_id']}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--official-profile", type=Path, required=True)
    parser.add_argument("--official-samples", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["workload_id"]: row for row in read_csv(args.manifest)}
    control = {row["workload_id"]: row for row in read_csv(args.control_manifest)}
    official = {row["workload_id"]: row for row in read_csv(args.official_profile)}
    official_samples = samples_by_id(args.official_samples)
    direct = direct_results(args.runner_log)
    selections = {
        row["workload_id"]: row for row in (
            json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    ids = set(manifest)
    if not (len(ids) == EXPECTED_SHAPES and set(official) == ids and
            set(official_samples) == ids and set(selections) == ids and
            len(control) == EXPECTED_CONTROL_SHAPES):
        raise RuntimeError(
            f"identity mismatch manifest={len(manifest)} official={len(official)} "
            f"official_samples={len(official_samples)} selection={len(selections)} "
            f"control={len(control)}"
        )

    output_rows = []
    for workload_id, expected in manifest.items():
        suffix = int(expected["kernel_suffix"])
        candidate = direct.get((workload_id, suffix))
        if candidate is None:
            raise RuntimeError(f"novel direct result missing for {workload_id}")
        candidate_samples = attest_direct(candidate, expected)
        old = official[workload_id]
        old_samples = official_samples[workload_id]
        if not (
            old.get("source") == "installed_aclnn_matmul_public_api" and
            old.get("candidate_role") == "installed_operator_reference" and
            old.get("success") == "1" and old.get("preflight_passed") == "1" and
            int(old.get("warmup", 0)) == EXPECTED_WARMUP and
            int(old.get("repeat", 0)) == EXPECTED_REPEAT and
            int(old.get("samples", 0)) == EXPECTED_SAMPLES and
            finite_positive(old_samples)
        ):
            raise RuntimeError(f"public official attestation failed for {workload_id}")
        official_median = statistics.median(old_samples)
        candidate_median = statistics.median(candidate_samples)
        if not math.isclose(float(old["median_ms"]), official_median, rel_tol=1e-9):
            raise RuntimeError(f"official median mismatch for {workload_id}")

        closest_median = None
        closest_delta = None
        closest_separation = "NO_EXECUTABLE_C220_OFFICIAL_EQUIVALENT"
        if workload_id in control:
            control_manifest = control[workload_id]
            reference = direct.get((workload_id, 41))
            if reference is None:
                raise RuntimeError(f"suffix-41 control missing for {workload_id}")
            reference_samples = attest_direct(reference, control_manifest)
            closest_median = statistics.median(reference_samples)
            closest_delta = 100.0 * (candidate_median / closest_median - 1.0)
            closest_separation = separation(candidate_samples, reference_samples)

        selection = selections[workload_id]
        output_rows.append({
            "workload_id": workload_id,
            "family": selection["required_family"],
            "kernel_suffix": suffix,
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "scale_band": selection["scale_band"],
            "used_cores": int(expected["used_core_num"]),
            "official81_median_ms": official_median,
            "candidate_median_ms": candidate_median,
            "vs_official81_delta_pct": 100.0 * (candidate_median / official_median - 1.0),
            "vs_official81_separation": separation(candidate_samples, old_samples),
            "closest_official_family": "C220_MULTI_CORE_SPLIT_K_SUFFIX_41" if closest_median else "NONE_ON_C220",
            "closest_official_median_ms": "" if closest_median is None else closest_median,
            "vs_closest_official_delta_pct": "" if closest_delta is None else closest_delta,
            "vs_closest_official_separation": closest_separation,
            "correctness": "PASS_ALL_CURRENT_RUN_OUTPUTS",
        })

    result = {
        "schema": "novel_matmul_family_validation_v1",
        "status": "complete",
        "selection_uses_measurements": False,
        "all_current_outputs_validated": True,
        "shape_count": len(output_rows),
        "control_shape_count": len(control),
        "shapes": output_rows,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print("NOVEL_RESULT_ANALYSIS passed shapes=200 closest_official_controls=100")


if __name__ == "__main__":
    main()
