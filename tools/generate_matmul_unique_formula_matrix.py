#!/usr/bin/env python3
"""Generate one closed-form improved MatMul tiling for each validation shape."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR_ROOT))

from improved_selector import generate  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
EXPECTED_VALIDATION_SHAPES = 8
EXPECTED_VARIANTS = {("fp32", 101), ("fp32", 20201)}
EXPECTED_SOURCE_FAMILIES = {
    "BASE", "AL1_FULL_LOAD", "BL1_FULL_LOAD", "BL1_FULL_LOAD_FIXPIPE",
    "SINGLE_CORE_SPLIT_K", "DETERMINISTIC_SPLIT_K", "INCREMENTAL_PATTERN",
}
EXPECTED_SOURCE_SUFFIXES = {
    0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201,
}


def fnv1a64(blob: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def clear_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file():
            path.unlink()


def audit_no_candidate_engine() -> None:
    forbidden = (
        "from candidate_engine", "import candidate_engine",
        "generate_and_select(", "derive_ideal_region(", "simulate(",
    )
    for path in (
        SELECTOR_ROOT / "formula_rules.py",
        SELECTOR_ROOT / "improved_selector.py",
    ):
        source = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in source]
        if hits:
            raise RuntimeError(f"{path.name}: forbidden candidate path {hits}")


def run_selector(row: dict) -> dict:
    result = generate(
        int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
        bool(row["trans_a"]), bool(row["trans_b"]),
    )
    theory = result["theory"]
    if not (
        theory["selection_mode"] == "UNIQUE_ORDERED_CLOSED_FORM"
        and theory["complete_tilings_constructed"] == 1
        and theory["candidate_count"] == 1
        and theory["candidate_enumeration"] is False
        and theory["pareto_pruning"] is False
        and theory["latency_or_cost_score"] is False
        and result["runtime_dependencies"]["cost_model"] is False
        and result["runtime_dependencies"]["candidate_enumeration"] is False
        and result["runtime_dependencies"]["history"] is False
        and result["runtime_dependencies"]["runtime_kb"] is False
        and result["runtime_dependencies"]["tiling_bank"] is False
    ):
        raise RuntimeError(f"{row['workload_id']}: unique selector contract failed")
    return result


def compact_theory(workload_id: str, result: dict) -> dict:
    theory = result["theory"]
    return {
        "workload_id": workload_id,
        "status": result["status"],
        "source_family": theory["source_family"],
        "source_suffix": theory["source_suffix"],
        "formula_family": theory["formula_family"],
        "candidate_count": theory["candidate_count"],
        "logical_fma_count": theory["logical_fma_count"],
        "formula_fields": theory["formula_fields"],
        "formula_l2": theory["formula_l2"],
        "resource_bytes": theory["resource_bytes"],
        "selection_contract": theory["selection_contract"],
        "scheduled_work": theory.get("scheduled_work"),
        "improvement_equation": theory.get("improvement_equation"),
        "skip_reason": result.get("skip_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--theory-audit", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args()

    audit_no_candidate_engine()
    validation = json.loads(
        (SELECTOR_ROOT / "unique_formula_validation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    audit_contract = json.loads(
        (SELECTOR_ROOT / "source_family_audit_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if (len(validation) != EXPECTED_VALIDATION_SHAPES or
            len({row["workload_id"] for row in validation}) != len(validation)):
        raise RuntimeError("unique validation contract was reduced or duplicated")
    if (len(audit_contract) != 13 or
            len({row["workload_id"] for row in audit_contract}) !=
            len(audit_contract)):
        raise RuntimeError("source family audit contract was reduced or duplicated")

    for directory in (args.output_dir, args.variant_dir, args.sequence_dir):
        clear_files(directory)

    audit_rows = []
    source_families, source_suffixes = set(), set()
    for row in audit_contract:
        result = run_selector(row)
        theory = result["theory"]
        if (theory["source_family"] != row["expected_family"] or
                int(theory["source_suffix"]) != int(row["expected_suffix"])):
            raise RuntimeError(
                f"{row['workload_id']}: source decision drift "
                f"family={theory['source_family']} suffix={theory['source_suffix']}"
            )
        selected = result["selected"]
        raw = bytes.fromhex(selected["tiling_data_hex"])
        if (len(raw) != 272 or selected["tiling_data_bytes"] != 272 or
                hashlib.sha256(raw).hexdigest() !=
                selected["tiling_data_sha256"]):
            raise RuntimeError(
                f"{row['workload_id']}: source path is not one complete packet"
            )
        source_families.add(theory["source_family"])
        source_suffixes.add(int(theory["source_suffix"]))
        audit_rows.append(compact_theory(row["workload_id"], result))
    if (source_families != EXPECTED_SOURCE_FAMILIES or
            source_suffixes != EXPECTED_SOURCE_SUFFIXES):
        raise RuntimeError("full source family/suffix audit coverage was reduced")

    manifest_rows = []
    selections = []
    for row in validation:
        result = run_selector(row)
        workload_id = str(row["workload_id"])
        if not (result["status"] == "MODIFIED_TILING" and
                result["npu_eligible"] and not result["baseline_equivalent"]):
            raise RuntimeError(f"{workload_id}: no distinct improved packet")
        expected_rule = (
            "AL1_CAPACITY_DERIVED_K_GRAIN"
            if row["selection_axis"] == "al1_capacity_k_grain"
            else "FIXPIPE_VECTOR_MULTI_GROUP_PIPELINE"
        )
        if result["changed_rules"] != [expected_rule]:
            raise RuntimeError(f"{workload_id}: unexpected theoretical rule")
        packet = result["improved"]
        blob = bytes.fromhex(packet["tiling_data_hex"])
        if (len(blob) != 272 or
                hashlib.sha256(blob).hexdigest() != packet["tiling_data_sha256"]):
            raise RuntimeError(f"{workload_id}: invalid 272-byte packet")
        path = args.output_dir / f"{workload_id}.bin"
        path.write_bytes(blob)
        words = struct.unpack("<68I", blob)
        manifest_rows.append({
            "workload_id": workload_id, "rank": "0",
            "candidate_role": "unique_theoretical_improvement",
            "m": str(row["m"]), "n": str(row["n"]), "k": str(row["k"]),
            "dtype": str(row["dtype"]),
            "trans_a": str(int(row["trans_a"])),
            "trans_b": str(int(row["trans_b"])),
            "used_core_num": str(packet["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(packet["workspace_bytes"]),
            "tiling_path": str(path.resolve()),
            "tiling_sha256": packet["tiling_data_sha256"],
            "tiling_fnv1a64": fnv1a64(blob),
            "model_schedule_sha256": hashlib.sha256(
                json.dumps(compact_theory(workload_id, result), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "is_reserve": "0", "l2_cache_flag": str(words[62]),
            "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
            "required_successful_tilings": "1",
        })
        result.update({
            "workload_id": workload_id,
            "selection_axis": row["selection_axis"],
            "required_applicable_family": result["theory"]["source_family"],
            "case_role": "unique_theoretical_improvement",
        })
        selections.append(result)
        audit_rows.append(compact_theory(workload_id, result))

    variants = {(row["dtype"], int(row["kernel_suffix"]))
                for row in manifest_rows}
    if variants != EXPECTED_VARIANTS:
        raise RuntimeError(f"unexpected validation variants: {sorted(variants)}")
    write_csv(args.manifest, manifest_rows)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        grouped[(row["dtype"], row["kernel_suffix"])].append(row)
    for index, ((dtype, suffix), rows) in enumerate(sorted(grouped.items())):
        name = f"{dtype}_k{suffix}"
        write_csv(args.variant_dir / f"{name}.csv", rows)
        write_csv(args.sequence_dir / f"{index:02d}__{name}.csv", rows)

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", encoding="utf-8") as stream:
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    with args.theory_audit.open("w", encoding="utf-8") as stream:
        for result in audit_rows:
            stream.write(json.dumps(result, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    print(
        "UNIQUE_FORMULA_MATRIX_GENERATED "
        f"audit_shapes={len(audit_contract)} source_families={len(source_families)} "
        f"source_suffixes={len(source_suffixes)} npu_shapes={len(manifest_rows)} "
        f"complete_tilings_per_shape=1 variants={len(grouped)}"
    )


if __name__ == "__main__":
    main()
