#!/usr/bin/env python3
"""Compare current improved packets with exact prior public MatMul measurements."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_SHAPES = 5
EXPECTED_SAMPLES = 7
EXPECTED_WARMUP = 2
EXPECTED_REPEAT = 20


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


def finite_positive(values):
    return all(
        isinstance(value, (int, float)) and math.isfinite(value) and value > 0
        for value in values
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--reference-samples", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["workload_id"]: row for row in read_csv(args.manifest)}
    references = {
        row["workload_id"]: row for row in read_csv(args.reference_profile)
    }
    reference_samples = defaultdict(list)
    for row in read_csv(args.reference_samples):
        reference_samples[row["workload_id"]].append(float(row["latency_ms"]))
    direct = direct_results(args.runner_log)
    selections = {
        row["workload_id"]: row for row in (
            json.loads(line)
            for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    identities = set(manifest)
    if (len(identities) != EXPECTED_SHAPES or set(references) != identities or
            set(reference_samples) != identities or set(direct) != identities or
            set(selections) != identities):
        raise RuntimeError(
            f"identity mismatch manifest={len(manifest)} references={len(references)} "
            f"direct={len(direct)} selection={len(selections)}"
        )

    output_rows = []
    for workload_id in manifest:
        expected = manifest[workload_id]
        reference = references[workload_id]
        improved = direct[workload_id]
        old_samples = reference_samples[workload_id]
        new_samples = improved.get("samples_ms", [])
        reference_median = statistics.median(old_samples)
        if not (
            reference.get("source") == "historical_installed_aclnn_matmul_device_event" and
            reference.get("candidate_role") == "installed_operator_reference" and
            reference.get("execution_mode") == "official_matmul_v3" and
            reference.get("success") == "1" and reference.get("preflight_passed") == "1" and
            reference.get("preflight_mode") == "numeric_signed_axes_full_v3" and
            reference.get("device") == "Ascend910B3" and
            reference.get("toolkit") == "CANN 8.1" and
            reference.get("measurement_method") == "device_event_latency" and
            int(reference.get("warmup", 0)) == EXPECTED_WARMUP and
            int(reference.get("repeat", 0)) == EXPECTED_REPEAT and
            int(reference.get("samples", 0)) == EXPECTED_SAMPLES and
            len(old_samples) == EXPECTED_SAMPLES and finite_positive(old_samples) and
            math.isclose(float(reference["median_ms"]), reference_median, rel_tol=1e-12) and
            improved.get("status") == "success" and
            improved.get("candidate_role") == "independent_improved" and
            improved.get("measurement_source") == "direct_tiling_buffer" and
            improved.get("tiling_applied") == 1 and
            improved.get("full_output_validated") == 1 and
            improved.get("warmup") == EXPECTED_WARMUP and
            improved.get("repeat") == EXPECTED_REPEAT and
            improved.get("samples") == EXPECTED_SAMPLES and
            improved.get("actual_tiling_sha256") == expected["tiling_sha256"] and
            str(improved.get("actual_kernel_suffix")) == expected["kernel_suffix"] and
            str(improved.get("actual_block_dim")) == expected["used_core_num"] and
            isinstance(new_samples, list) and len(new_samples) == EXPECTED_SAMPLES and
            finite_positive(new_samples)
        ):
            raise RuntimeError(f"execution or reference attestation failed for {workload_id}")

        improved_median = statistics.median(new_samples)
        if not math.isclose(
            float(improved["median_ms"]), improved_median, rel_tol=1e-9
        ):
            raise RuntimeError(f"direct median does not match raw samples for {workload_id}")
        delta_pct = 100.0 * (improved_median / reference_median - 1.0)
        if max(new_samples) < min(old_samples):
            separation = "CLEAR_IMPROVED"
        elif min(new_samples) > max(old_samples):
            separation = "CLEAR_REGRESSION"
        else:
            separation = "OVERLAPPING_SAMPLES"
        selection = selections[workload_id]
        row = {
            "workload_id": workload_id,
            "selection_axis": reference["selection_axis"],
            "kernel_suffix": int(expected["kernel_suffix"]),
            "kernel_variant": selection["improved"]["kernel_variant"],
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]),
            "trans_b": int(expected["trans_b"]),
            "historical_official_median_ms": reference_median,
            "current_improved_median_ms": improved_median,
            "delta_pct": delta_pct,
            "speedup": reference_median / improved_median,
            "median_winner": "improved" if improved_median < reference_median else "official",
            "sample_separation": separation,
            "changed_rules": ";".join(selection["changed_rules"]),
            "reference_record_uid": reference["record_uid"],
            "correctness": "PASS_CURRENT_IMPROVED_AND_AUDITED_REFERENCE",
        }
        output_rows.append(row)
        print(
            f"DISCRIMINATIVE_RESULT id={workload_id} axis={row['selection_axis']} "
            f"official_ms={reference_median:.9g} improved_ms={improved_median:.9g} "
            f"delta_pct={delta_pct:+.3f} separation={separation}"
        )

    result = {
        "schema": "matmul_rule_discriminative_v3",
        "status": "complete",
        "comparison_basis": "audited_historical_cann81_reference_vs_current_improved",
        "reference_remeasured": False,
        "improved_source": "independent_shape_hardware_formula",
        "measurement_contract": {
            "warmup": EXPECTED_WARMUP,
            "repeat": EXPECTED_REPEAT,
            "samples": EXPECTED_SAMPLES,
            "latency": "device_event_per_launch",
        },
        "aggregate": {
            "selected_shapes": len(output_rows),
            "current_improved_measurements": len(output_rows),
            "reused_official_references": len(output_rows),
            "clear_improvements": sum(
                row["sample_separation"] == "CLEAR_IMPROVED" for row in output_rows
            ),
            "clear_regressions": sum(
                row["sample_separation"] == "CLEAR_REGRESSION" for row in output_rows
            ),
            "overlapping_samples": sum(
                row["sample_separation"] == "OVERLAPPING_SAMPLES" for row in output_rows
            ),
            "all_current_outputs_validated": True,
        },
        "shapes": output_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(output_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"DISCRIMINATIVE_COMPLETE shapes={len(output_rows)} "
        f"clear_improvements={result['aggregate']['clear_improvements']} "
        f"clear_regressions={result['aggregate']['clear_regressions']} "
        f"overlap={result['aggregate']['overlapping_samples']}"
    )


if __name__ == "__main__":
    main()
