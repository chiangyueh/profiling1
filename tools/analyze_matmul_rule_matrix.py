#!/usr/bin/env python3
"""Compare buildable expanded families with same-campaign public MatMulV3."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED_SHAPES = 224
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
        row = json.loads(line.split(" ", 1)[1])
        if row["workload_id"] in rows:
            raise RuntimeError(f"duplicate direct result {row['workload_id']}")
        rows[row["workload_id"]] = row
    return rows


def sample_map(path: Path) -> dict[str, list[float]]:
    result = {}
    for row in read_csv(path):
        result.setdefault(row["workload_id"], []).append(float(row["latency_ms"]))
    return result


def finite_positive(values: list[float]) -> bool:
    return all(math.isfinite(value) and value > 0 for value in values)


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
    all_selections = {
        row["workload_id"]: row for row in (
            json.loads(line) for line in
            args.selection.read_text(encoding="utf-8").splitlines() if line
        )
    }
    selections = {
        key: row for key, row in all_selections.items() if row["npu_eligible"]
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
            and new.get("candidate_role") == "independent_experimental_family"
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
        if max(new_samples) < min(old_samples):
            separation = "CLEAR_CANDIDATE_WINNER"
        elif min(new_samples) > max(old_samples):
            separation = "CLEAR_OFFICIAL_WINNER"
        else:
            separation = "OVERLAPPING_SAMPLES"
        selection = selections[workload_id]
        output_rows.append({
            "workload_id": workload_id,
            "family": selection["formula_family"],
            "kernel_suffix": int(expected["kernel_suffix"]),
            "m": int(expected["m"]), "n": int(expected["n"]),
            "k": int(expected["k"]), "dtype": expected["dtype"],
            "trans_a": int(expected["trans_a"]),
            "trans_b": int(expected["trans_b"]),
            "scale_band": selection["scale_band"],
            "used_cores": int(expected["used_core_num"]),
            "official_median_ms": old_median,
            "candidate_median_ms": new_median,
            "delta_pct": 100.0 * (new_median / old_median - 1.0),
            "median_winner": "candidate" if new_median < old_median else "official",
            "sample_separation": separation,
            "correctness": "PASS_BOTH_CURRENT_RUN_OUTPUTS",
        })

    aggregate = {
        "shapes": len(output_rows),
        "families": len({row["family"] for row in output_rows}),
        "candidate_wins": sum(row["median_winner"] == "candidate" for row in output_rows),
        "official_wins": sum(row["median_winner"] == "official" for row in output_rows),
        "clear_candidate_wins": sum(row["sample_separation"] == "CLEAR_CANDIDATE_WINNER" for row in output_rows),
        "clear_official_wins": sum(row["sample_separation"] == "CLEAR_OFFICIAL_WINNER" for row in output_rows),
        "overlap": sum(row["sample_separation"] == "OVERLAPPING_SAMPLES" for row in output_rows),
    }
    result = {
        "schema": "matmul_expanded_family_validation_v1",
        "status": "complete",
        "selection_uses_measurements": False,
        "all_current_outputs_validated": True,
        "aggregate": aggregate,
        "shapes": output_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        "EXPANDED_FAMILY_VALIDATION_COMPLETE "
        f"shapes={aggregate['shapes']} families={aggregate['families']} "
        f"candidate_wins={aggregate['candidate_wins']} "
        f"official_wins={aggregate['official_wins']} overlap={aggregate['overlap']}"
    )


if __name__ == "__main__":
    main()
