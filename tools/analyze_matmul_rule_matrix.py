#!/usr/bin/env python3
"""Compare one improved packet per branch with the installed public MatMulV3."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def direct_results(path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DIRECT_MATMUL_RESULT "):
            result = json.loads(line.split(" ", 1)[1])
            key = result["workload_id"]
            if key in rows:
                raise RuntimeError(f"duplicate direct result {key}")
            rows[key] = result
    return rows


def main():
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
    official_samples = defaultdict(list)
    for row in read_csv(args.official_samples):
        official_samples[row["workload_id"]].append(float(row["latency_ms"]))
    direct = direct_results(args.runner_log)
    selections = {
        row["workload_id"]: row for row in (
            json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    identities = set(manifest)
    if (len(identities) != 12 or set(official) != identities or
            set(official_samples) != identities or set(direct) != identities or
            set(selections) != identities):
        raise RuntimeError(
            f"branch identity mismatch manifest={len(manifest)} official={len(official)} "
            f"direct={len(direct)} selection={len(selections)}"
        )

    output_rows = []
    for workload_id in sorted(identities, key=lambda value: int(value.split('_', 1)[0][1:])):
        expected = manifest[workload_id]
        baseline = official[workload_id]
        improved = direct[workload_id]
        samples = improved.get("samples_ms", [])
        if not (
            baseline.get("source") == "installed_aclnn_matmul_public_api" and
            baseline.get("candidate_role") == "installed_operator_reference" and
            baseline.get("execution_mode") == "opaque_public_dispatch" and
            baseline.get("success") in ("1", "true") and
            baseline.get("preflight_passed") in ("1", "true") and
            baseline.get("warmup") == "1" and baseline.get("repeat") == "1" and
            baseline.get("samples") == "7" and
            len(official_samples[workload_id]) == 7 and
            all(math.isfinite(x) and x > 0 for x in official_samples[workload_id]) and
            float(baseline.get("median_ms", 0)) > 0 and
            improved.get("status") == "success" and
            improved.get("measurement_source") == "direct_tiling_buffer" and
            improved.get("tiling_applied") == 1 and
            improved.get("full_output_validated") == 1 and
            improved.get("actual_tiling_sha256") == expected["tiling_sha256"] and
            str(improved.get("actual_kernel_suffix")) == expected["kernel_suffix"] and
            str(improved.get("actual_block_dim")) == expected["used_core_num"] and
            isinstance(samples, list) and len(samples) == 7 and
            all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in samples)
        ):
            raise RuntimeError(f"execution attestation failed for {workload_id}")
        baseline_ms = float(baseline["median_ms"])
        improved_ms = float(improved["median_ms"])
        delta_pct = 100.0 * (improved_ms / baseline_ms - 1.0)
        selection = selections[workload_id]
        row = {
            "workload_id": workload_id,
            "kernel_suffix": int(expected["kernel_suffix"]),
            "kernel_variant": selection["improved"]["kernel_variant"],
            "m": int(expected["m"]), "n": int(expected["n"]), "k": int(expected["k"]),
            "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]), "trans_b": int(expected["trans_b"]),
            "official_median_ms": baseline_ms,
            "improved_median_ms": improved_ms,
            "delta_pct": delta_pct,
            "speedup": baseline_ms / improved_ms,
            "winner": "improved" if improved_ms < baseline_ms else "official",
            "changed_rules": ";".join(selection["changed_rules"]),
            "correctness": "PASS_BOTH_OUTPUTS",
        }
        output_rows.append(row)
        print(
            f"RULE_MATRIX_RESULT suffix={row['kernel_suffix']} variant={row['kernel_variant']} "
            f"official_ms={baseline_ms:.9g} improved_ms={improved_ms:.9g} "
            f"delta_pct={delta_pct:+.3f} winner={row['winner']}"
        )

    result = {
        "schema": "matmul_rule_matrix_v2",
        "status": "complete",
        "official_reference": "installed_aclnn_matmul_public_api",
        "improved_source": "independent_shape_hardware_formula",
        "aggregate": {
            "installed_dispatch_branches": 12,
            "measured_branches": len(output_rows),
            "improved_wins": sum(row["winner"] == "improved" for row in output_rows),
            "official_wins": sum(row["winner"] == "official" for row in output_rows),
            "all_outputs_validated": True,
            "samples_per_path": 7,
        },
        "branches": output_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"RULE_MATRIX_COMPLETE branches={len(output_rows)} "
        f"improved_wins={result['aggregate']['improved_wins']} "
        f"official_wins={result['aggregate']['official_wins']}"
    )


if __name__ == "__main__":
    main()
