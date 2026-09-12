#!/usr/bin/env python3
"""Materialize one explicit C220 GM-to-L1 packet per workload."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR_ROOT))

from c220_experimental_selector import (  # noqa: E402
    GM_TO_L1_SYNC_CYCLES,
    KERNEL_LAUNCH_CYCLES,
    generate,
)


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
EXPECTED_ROWS = 4
EXPECTED_FAMILIES = {
    "SINGLE_CORE_SPLIT_K_GM_TO_L1",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
}
EXPECTED_VARIANTS = {
    ("fp16", 61), ("bf16", 61),
    ("fp16", 60), ("bf16", 60),
}


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def audit_gm_to_l1_selection(workload: dict, result: dict) -> None:
    cube = result["cube"]
    derivation = result["derivation"]
    workload_id = workload["workload_id"]
    if derivation.get("model") != "gm_to_l1_critical_path_v1":
        raise RuntimeError(f"{workload_id}: wrong formula model")
    if derivation.get("primitive_calibration") != "frozen_c220_isolated_instruction_contract_v1":
        raise RuntimeError(f"{workload_id}: missing primitive calibration attestation")
    if derivation.get("critical_path_equation") != "max(worst_core,aggregate_hbm)+sync+launch":
        raise RuntimeError(f"{workload_id}: critical-path equation drift")
    if not (derivation["finite_candidate_count"] > 0 and
            0 < derivation["pareto_candidate_count"] <= derivation["finite_candidate_count"]):
        raise RuntimeError(f"{workload_id}: invalid finite/Pareto candidate counts")
    if derivation["base_tile"] != [cube["baseM"], cube["baseN"], cube["baseK"]]:
        raise RuntimeError(f"{workload_id}: model/kernel base-tile mismatch")
    if derivation["k_stripe"] != cube["singleCoreK"]:
        raise RuntimeError(f"{workload_id}: model/kernel K-stripe mismatch")
    step_k = cube["singleCoreK"] // cube["baseK"]
    if not (
        cube["singleCoreK"] % cube["baseK"] == 0 and
        cube["stepM"] == cube["stepN"] == 1 and
        cube["stepKa"] == cube["stepKb"] == step_k and
        cube["depthA1"] == cube["depthB1"] == step_k and
        cube["dbL0A"] == cube["dbL0B"] == cube["dbL0C"] == 2
    ):
        raise RuntimeError(f"{workload_id}: packet pipeline fields do not match the model")
    if cube["singleCoreM"] % cube["baseM"] or cube["singleCoreN"] % cube["baseN"]:
        raise RuntimeError(f"{workload_id}: core tile does not partition into base tiles")
    expected_cores = (
        ceil_div(int(workload["m"]), cube["singleCoreM"]) *
        ceil_div(int(workload["n"]), cube["singleCoreN"])
    )
    if expected_cores != cube["usedCoreNum"] or expected_cores != result["block_dim"]:
        raise RuntimeError(f"{workload_id}: core-grid ownership mismatch")
    resident = derivation["resident_a_l1_bytes"] + derivation["resident_b_l1_bytes"]
    if resident > derivation["resident_l1_limit_bytes"]:
        raise RuntimeError(f"{workload_id}: GM-to-L1 queues exceed the audited L1 limit")
    components = derivation["worst_core_components"]
    rebuilt_worst = (
        components["gm_to_l1_cycles"] + components["cube_l1_overlap_cycles"] +
        components["output_accumulation_cycles"] + components["vector_cast_cycles"]
    )
    if not math.isclose(rebuilt_worst, components["total_cycles"], rel_tol=1e-12):
        raise RuntimeError(f"{workload_id}: worst-core equation mismatch")
    rebuilt_critical = max(
        components["total_cycles"], derivation["aggregate_hbm_cycles"]
    ) + GM_TO_L1_SYNC_CYCLES + KERNEL_LAUNCH_CYCLES
    if not math.isclose(rebuilt_critical, derivation["critical_path_cycles"], rel_tol=1e-12):
        raise RuntimeError(f"{workload_id}: critical-path total mismatch")
    if not all(result["resource_checks"].values()):
        raise RuntimeError(f"{workload_id}: hard resource audit did not pass")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args()

    complete_contract = json.loads(
        (SELECTOR_ROOT / "c220_validation_contract.json").read_text(encoding="utf-8")
    )
    contract = [
        row for row in complete_contract
        if row["required_family"] in EXPECTED_FAMILIES
    ]
    ids = {row["workload_id"] for row in contract}
    families = {row["required_family"] for row in contract}
    variants = {(row["dtype"], int({
        "MULTI_CORE_SPLIT_K": 41,
        "SINGLE_CORE_NKM_SPLIT_K": 51,
        "SINGLE_CORE_SPLIT_K_GM_TO_L1": 61,
        "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED": 60,
        "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": 121,
    }[row["required_family"]])) for row in contract}
    if len(contract) != EXPECTED_ROWS or len(ids) != EXPECTED_ROWS:
        raise RuntimeError(f"C220 contract must contain {EXPECTED_ROWS} unique workloads")
    if families != EXPECTED_FAMILIES or variants != EXPECTED_VARIANTS:
        raise RuntimeError("C220 contract family/dtype coverage was reduced")

    for directory in (args.output_dir, args.variant_dir, args.sequence_dir):
        clear_files(directory)

    manifest_rows = []
    selections = []
    for row in contract:
        result = generate(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
            required_family=str(row["required_family"]),
        )
        if not result["npu_eligible"] or result["formula_family"] != row["required_family"]:
            raise RuntimeError(f"{row['workload_id']}: family selection drift")
        if any(result["runtime_dependencies"].values()):
            raise RuntimeError(f"{row['workload_id']}: forbidden runtime dependency")
        audit_gm_to_l1_selection(row, result)
        blob = bytes.fromhex(result["packet_hex"])
        if len(blob) != 280 or hashlib.sha256(blob).hexdigest() != result["packet_sha256"]:
            raise RuntimeError(f"{row['workload_id']}: invalid 280-byte packet")
        path = args.output_dir / f"{row['workload_id']}.bin"
        path.write_bytes(blob)
        words = struct.unpack("<70I", blob)
        schedule = {
            "family": result["formula_family"], "suffix": result["kernel_suffix"],
            "cube": result["cube"], "derivation": result["derivation"],
        }
        manifest_rows.append({
            "workload_id": row["workload_id"], "rank": "0",
            "candidate_role": "independent_experimental_family",
            "m": str(row["m"]), "n": str(row["n"]), "k": str(row["k"]),
            "dtype": row["dtype"], "trans_a": str(int(row["trans_a"])),
            "trans_b": str(int(row["trans_b"])),
            "used_core_num": str(result["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(result["workspace_bytes"]),
            "tiling_path": str(path.resolve()), "tiling_sha256": result["packet_sha256"],
            "tiling_fnv1a64": fnv1a64(blob),
            "model_schedule_sha256": hashlib.sha256(json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "is_reserve": "0", "l2_cache_flag": str(words[64]),
            "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
            "required_successful_tilings": "1",
        })
        result.update({
            "workload_id": row["workload_id"], "selection_axis": row["selection_axis"],
            "required_applicable_family": row["required_family"],
            "case_role": "experimental_family",
        })
        selections.append(result)

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
        for row in selections:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"C220_MATRIX_GENERATED shapes={len(manifest_rows)} families={len(families)} variants={len(grouped)} packet_bytes=280")


if __name__ == "__main__":
    main()
