#!/usr/bin/env python3
"""Verify current-run evidence and summarize formula tilings by kernel suffix."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 240
EXPECTED_SAMPLES = 5
EXPECTED_WARMUP = 1
EXPECTED_REPEAT = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def direct_results(path: Path) -> dict[str, dict]:
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("DIRECT_MATMUL_RESULT "):
            continue
        row = json.loads(line.split(" ", 1)[1])
        workload_id = row["workload_id"]
        if workload_id in output:
            raise RuntimeError(f"duplicate direct result for {workload_id}")
        output[workload_id] = row
    return output


def sample_map(path: Path) -> dict[str, list[float]]:
    output = defaultdict(list)
    for row in read_csv(path):
        output[row["workload_id"]].append(float(row["latency_ms"]))
    return dict(output)


def valid_samples(values: list[float]) -> bool:
    return len(values) == EXPECTED_SAMPLES and all(
        math.isfinite(value) and value > 0 for value in values
    )


def separation(candidate: list[float], official: list[float]) -> str:
    if max(candidate) < min(official):
        return "CLEAR_FORMULA_WIN"
    if min(candidate) > max(official):
        return "CLEAR_OFFICIAL_WIN"
    return "OVERLAP"


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
    direct = direct_results(args.runner_log)
    selections = {
        row["workload_id"]: row
        for row in (
            json.loads(line)
            for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    ids = set(manifest)
    if not (
        len(ids) == EXPECTED_SHAPES
        and set(official) == ids
        and set(official_samples) == ids
        and set(direct) == ids
        and set(selections) == ids
    ):
        raise RuntimeError(
            "current-run identity mismatch: "
            f"manifest={len(manifest)} official={len(official)} "
            f"official_samples={len(official_samples)} direct={len(direct)} "
            f"selection={len(selections)}"
        )

    rows = []
    for workload_id, expected in manifest.items():
        measured = direct[workload_id]
        candidate_samples = [float(value) for value in measured.get("samples_ms", [])]
        if not (
            measured.get("status") == "success"
            and measured.get("candidate_role") in (
                "independent_formula_tiling", "independent_experimental_family"
            )
            and measured.get("measurement_source") == "direct_tiling_buffer"
            and measured.get("tiling_applied") == 1
            and measured.get("full_output_validated") == 1
            and measured.get("warmup") == EXPECTED_WARMUP
            and measured.get("repeat") == EXPECTED_REPEAT
            and measured.get("samples") == EXPECTED_SAMPLES
            and valid_samples(candidate_samples)
            and measured.get("actual_tiling_sha256") == expected["tiling_sha256"]
            and str(measured.get("actual_kernel_suffix")) == expected["kernel_suffix"]
            and str(measured.get("actual_block_dim")) == expected["used_core_num"]
        ):
            raise RuntimeError(f"{workload_id}: formula packet attestation failed")
        reference = official[workload_id]
        reference_samples = official_samples[workload_id]
        if not (
            reference.get("source") == "installed_aclnn_matmul_public_api"
            and reference.get("candidate_role") == "installed_operator_reference"
            and reference.get("success") == "1"
            and reference.get("preflight_passed") == "1"
            and int(reference.get("warmup", 0)) == EXPECTED_WARMUP
            and int(reference.get("repeat", 0)) == EXPECTED_REPEAT
            and int(reference.get("samples", 0)) == EXPECTED_SAMPLES
            and valid_samples(reference_samples)
        ):
            raise RuntimeError(f"{workload_id}: official current-run attestation failed")
        candidate_median = statistics.median(candidate_samples)
        official_median = statistics.median(reference_samples)
        if not math.isclose(float(measured["median_ms"]), candidate_median, rel_tol=1e-9):
            raise RuntimeError(f"{workload_id}: candidate median mismatch")
        if not math.isclose(float(reference["median_ms"]), official_median, rel_tol=1e-9):
            raise RuntimeError(f"{workload_id}: official median mismatch")
        selection = selections[workload_id]
        rows.append({
            "workload_id": workload_id,
            "family": selection["formula_family"],
            "suffix": int(selection["kernel_suffix"]),
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]),
            "trans_b": int(expected["trans_b"]),
            "used_cores": int(expected["used_core_num"]),
            "official_ms": official_median,
            "formula_ms": candidate_median,
            "delta_pct": 100.0 * (candidate_median / official_median - 1.0),
            "separation": separation(candidate_samples, reference_samples),
            "correctness": "PASS_BOTH_CURRENT_RUN_OUTPUTS",
        })

    result = {
        "schema": "complete_formula_tiling_current_run_v1",
        "status": "complete",
        "selection_uses_measurements": False,
        "all_current_outputs_validated": True,
        "shape_count": len(rows),
        "rows": rows,
    }
    args.output_json.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("FORMULA_RESULT_ANALYSIS passed shapes=240 outputs_validated=240")


if __name__ == "__main__":
    main()
