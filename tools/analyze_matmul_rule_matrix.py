#!/usr/bin/env python3
"""Compare MatMulV3 against the independently generated C220 GM-to-L1 family."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 4
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


def sample_map(path: Path) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for row in read_csv(path):
        result.setdefault(row["workload_id"], []).append(float(row["latency_ms"]))
    return result


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
    official = {
        row["workload_id"]: row for row in read_csv(args.official_profile)
    }
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
    identities = set(manifest)
    if not (
        len(identities) == EXPECTED_SHAPES
        and set(official) == identities
        and set(official_samples) == identities
        and set(direct) == identities
        and set(selections) == identities
    ):
        raise RuntimeError(
            "same-campaign identity mismatch: "
            f"manifest={len(manifest)} official={len(official)} "
            f"official_samples={len(official_samples)} direct={len(direct)} "
            f"selections={len(selections)}"
        )

    output_rows = []
    for workload_id, expected in manifest.items():
        old = official[workload_id]
        new = direct[workload_id]
        old_samples = official_samples[workload_id]
        new_samples = [float(value) for value in new.get("samples_ms", [])]
        if not (
            old.get("source") == "installed_aclnn_matmul_public_api"
            and old.get("candidate_role") == "installed_operator_reference"
            and old.get("success") == "1"
            and old.get("preflight_passed") == "1"
            and int(old.get("warmup", 0)) == EXPECTED_WARMUP
            and int(old.get("repeat", 0)) == EXPECTED_REPEAT
            and int(old.get("samples", 0)) == EXPECTED_SAMPLES
            and len(old_samples) == EXPECTED_SAMPLES
            and finite_positive(old_samples)
            and new.get("status") == "success"
            and new.get("candidate_role") == expected["candidate_role"]
            and expected["candidate_role"] in (
                "independent_rule_winner",
                "independent_branch_probe",
                "independent_structural_probe",
                "independent_experimental_family",
            )
            and new.get("measurement_source") == "direct_tiling_buffer"
            and new.get("tiling_applied") == 1
            and new.get("full_output_validated") == 1
            and new.get("warmup") == EXPECTED_WARMUP
            and new.get("repeat") == EXPECTED_REPEAT
            and new.get("samples") == EXPECTED_SAMPLES
            and len(new_samples) == EXPECTED_SAMPLES
            and finite_positive(new_samples)
            and new.get("actual_tiling_sha256") == expected["tiling_sha256"]
            and str(new.get("actual_kernel_suffix")) == expected["kernel_suffix"]
            and str(new.get("actual_block_dim")) == expected["used_core_num"]
        ):
            raise RuntimeError(f"measurement attestation failed for {workload_id}")

        old_median = statistics.median(old_samples)
        new_median = statistics.median(new_samples)
        if not math.isclose(float(old["median_ms"]), old_median, rel_tol=1e-9):
            raise RuntimeError(f"official median mismatch for {workload_id}")
        if not math.isclose(float(new["median_ms"]), new_median, rel_tol=1e-9):
            raise RuntimeError(f"direct median mismatch for {workload_id}")
        delta_pct = 100.0 * (new_median / old_median - 1.0)
        if max(new_samples) < min(old_samples):
            separation = "CLEAR_CANDIDATE_WINNER"
        elif min(new_samples) > max(old_samples):
            separation = "CLEAR_OFFICIAL_WINNER"
        else:
            separation = "OVERLAPPING_SAMPLES"
        selection = selections[workload_id]
        row = {
            "workload_id": workload_id,
            "selection_axis": selection["selection_axis"],
            "case_role": selection["case_role"],
            "required_applicable_family": selection["required_applicable_family"],
            "selected_family": selection["formula_family"],
            "kernel_suffix": int(expected["kernel_suffix"]),
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]),
            "trans_b": int(expected["trans_b"]),
            "official_median_ms": old_median,
            "candidate_median_ms": new_median,
            "delta_pct": delta_pct,
            "speedup": old_median / new_median,
            "median_winner": (
                "candidate" if new_median < old_median else "official"
            ),
            "sample_separation": separation,
            "correctness": "PASS_BOTH_CURRENT_RUN_OUTPUTS",
        }
        output_rows.append(row)
        print(
            "C220_GM_TO_L1_RESULT "
            f"id={workload_id} applicable={row['required_applicable_family']} "
            f"selected={row['selected_family']} suffix={row['kernel_suffix']} "
            f"role={row['case_role']} official_ms={old_median:.9g} candidate_ms={new_median:.9g} "
            f"delta_pct={delta_pct:+.3f} separation={separation}"
        )

    result = {
        "schema": "matmul_c220_gm_to_l1_l0c_v2",
        "status": "complete",
        "comparison_basis": "same_campaign_official_api_vs_independent_gm_to_l1_family",
        "reference_remeasured": True,
        "selection_uses_measurements": False,
        "measurement_contract": {
            "warmup": EXPECTED_WARMUP,
            "repeat": EXPECTED_REPEAT,
            "samples": EXPECTED_SAMPLES,
            "latency": "device_event_per_launch",
        },
        "aggregate": {
            "shapes": len(output_rows),
            "applicable_families": len({
                row["required_applicable_family"] for row in output_rows
            }),
            "selected_families": len({row["selected_family"] for row in output_rows}),
            "candidate_median_wins": sum(
                row["median_winner"] == "candidate" for row in output_rows
            ),
            "official_median_wins": sum(
                row["median_winner"] == "official" for row in output_rows
            ),
            "clear_candidate_wins": sum(
                row["sample_separation"] == "CLEAR_CANDIDATE_WINNER"
                for row in output_rows
            ),
            "clear_official_wins": sum(
                row["sample_separation"] == "CLEAR_OFFICIAL_WINNER"
                for row in output_rows
            ),
            "overlap": sum(
                row["sample_separation"] == "OVERLAPPING_SAMPLES"
                for row in output_rows
            ),
            "all_current_outputs_validated": True,
            "experimental_family_cases_executed": sum(
                row["case_role"] == "experimental_family" for row in output_rows
            ),
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
        "C220_GM_TO_L1_COMPLETE "
        f"shapes={len(output_rows)} candidate_wins={result['aggregate']['candidate_median_wins']} "
        f"official_wins={result['aggregate']['official_median_wins']} "
        f"overlap={result['aggregate']['overlap']}"
    )


if __name__ == "__main__":
    main()
