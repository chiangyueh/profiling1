#!/usr/bin/env python3
"""Validate direct execution attestations and compare baseline/rule medians."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_results(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    found: dict[tuple[str, str], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("DIRECT_MATMUL_RESULT "):
            continue
        result = json.loads(line.split(" ", 1)[1])
        key = (str(result.get("workload_id", "")), str(result.get("rank", "")))
        if key in found:
            raise RuntimeError(f"duplicate direct result: {key}")
        found[key] = result
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    manifest = {
        (row["workload_id"], row["rank"]): row for row in manifest_rows
    }
    if len(manifest_rows) != 8 or len(manifest) != 8:
        raise RuntimeError("the frozen campaign must contain exactly eight unique packets")

    selections = {
        row["workload_id"]: row
        for row in (
            json.loads(line)
            for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    results = read_results(args.runner_log)
    if set(results) != set(manifest):
        raise RuntimeError(
            f"direct result identity mismatch: measured={len(results)} expected={len(manifest)}"
        )

    for key, result in results.items():
        expected = manifest[key]
        samples = result.get("samples_ms", [])
        if not (
            result.get("status") == "success"
            and result.get("measurement_source") == "direct_tiling_buffer"
            and result.get("tiling_applied") == 1
            and result.get("full_output_validated") == 1
            and result.get("actual_tiling_sha256") == expected["tiling_sha256"]
            and result.get("actual_tiling_fnv1a64") == expected["tiling_fnv1a64"]
            and str(result.get("actual_kernel_suffix")) == expected["kernel_suffix"]
            and str(result.get("actual_block_dim")) == expected["used_core_num"]
            and isinstance(samples, list)
            and len(samples) == 7
            and all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0 for value in samples)
        ):
            raise RuntimeError(f"direct execution attestation failed: {key}")

    workload_ids = sorted({key[0] for key in manifest})
    rows: list[dict[str, object]] = []
    for workload_id in workload_ids:
        if workload_id not in selections:
            raise RuntimeError(f"selection trace missing for {workload_id}")
        baseline = results[(workload_id, "0")]
        improved = results[(workload_id, "1")]
        baseline_ms = float(baseline["median_ms"])
        improved_ms = float(improved["median_ms"])
        delta_pct = 100.0 * (improved_ms / baseline_ms - 1.0)
        speedup = baseline_ms / improved_ms
        selection = selections[workload_id]
        manifest_row = manifest[(workload_id, "0")]
        row = {
            "workload_id": workload_id,
            "m": int(manifest_row["m"]),
            "n": int(manifest_row["n"]),
            "k": int(manifest_row["k"]),
            "dtype": manifest_row["dtype"],
            "trans_a": int(manifest_row["trans_a"]),
            "trans_b": int(manifest_row["trans_b"]),
            "baseline_family": selection["baseline"]["selected_family"],
            "improved_family": selection["improved"]["selected_family"],
            "changed_rules": ";".join(selection["changed_rules"]),
            "baseline_packet_sha256": manifest[(workload_id, "0")]["tiling_sha256"],
            "improved_packet_sha256": manifest[(workload_id, "1")]["tiling_sha256"],
            "baseline_median_ms": baseline_ms,
            "improved_median_ms": improved_ms,
            "delta_pct": delta_pct,
            "speedup": speedup,
            "winner": "improved" if improved_ms < baseline_ms else "baseline",
            "correctness": "PASS_BOTH_FULL_OUTPUTS",
        }
        rows.append(row)
        print(
            "RULE_AB_RESULT "
            f"workload={workload_id} baseline_ms={baseline_ms:.9g} "
            f"improved_ms={improved_ms:.9g} delta_pct={delta_pct:+.3f} "
            f"winner={row['winner']}"
        )

    aggregate = {
        "shape_count": len(rows),
        "packet_count": len(manifest),
        "improved_wins": sum(row["winner"] == "improved" for row in rows),
        "baseline_wins": sum(row["winner"] == "baseline" for row in rows),
        "all_packets_applied": True,
        "all_outputs_validated": True,
        "samples_per_packet": 7,
    }
    output = {
        "schema": "matmul_rule_ab_v1",
        "status": "complete",
        "selection_source": "pure_python_rules_with_frozen_cann81_baseline",
        "measurement_source": "direct_cann81_kernel_with_exact_272_byte_packet",
        "aggregate": aggregate,
        "workloads": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "RULE_AB_COMPLETE "
        f"shapes={aggregate['shape_count']} packets={aggregate['packet_count']} "
        f"improved_wins={aggregate['improved_wins']} "
        f"baseline_wins={aggregate['baseline_wins']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
