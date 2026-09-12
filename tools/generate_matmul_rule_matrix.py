#!/usr/bin/env python3
"""Materialize independent selector winners and explicit branch probes."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR_ROOT))

from improved_selector import generate  # noqa: E402


MANIFEST_FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
EXPECTED_FAMILIES = {
    "BASE", "AL1", "BL1", "FIXPIPE_BL1", "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K", "INCREMENTAL_PATTERN",
}
EXPECTED_SUFFIXES = {0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201}
EXPECTED_VALIDATION_ROWS = 31
EXPECTED_BRANCH_PROBES = 13


def fnv1a64(blob: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n"
        )
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

    contract = json.loads(
        (SELECTOR_ROOT / "validation_contract.json").read_text(encoding="utf-8")
    )
    identities = {row["workload_id"] for row in contract}
    declared_families = {row["required_applicable_family"] for row in contract}
    if len(contract) != EXPECTED_VALIDATION_ROWS or len(identities) != len(contract):
        raise RuntimeError(
            f"validation contract must contain exactly {EXPECTED_VALIDATION_ROWS} unique witnesses"
        )
    if declared_families != EXPECTED_FAMILIES:
        raise RuntimeError(
            "validation contract does not cover every installed candidate family"
        )
    branch_rows = [row for row in contract if row.get("case_role") == "branch_probe"]
    selector_rows = [row for row in contract if row.get("case_role") == "selector_top1"]
    if len(branch_rows) != EXPECTED_BRANCH_PROBES or len(selector_rows) != 18:
        raise RuntimeError("validation contract branch/selector row count was reduced")
    if {int(row["required_suffix"]) for row in branch_rows} != EXPECTED_SUFFIXES:
        raise RuntimeError("branch probes do not cover exactly all 12 installed suffixes")
    if not any(
        row.get("required_family") == "INCREMENTAL_PATTERN"
        for row in branch_rows
    ):
        raise RuntimeError("incremental candidate branch probe is missing")

    for directory in (args.output_dir, args.variant_dir, args.sequence_dir):
        clear_files(directory)

    manifest_rows: list[dict[str, str]] = []
    selections: list[dict] = []
    for row in contract:
        workload_id = str(row["workload_id"])
        case_role = str(row["case_role"])
        if case_role not in ("branch_probe", "selector_top1"):
            raise RuntimeError(f"{workload_id}: invalid case_role={case_role}")
        result = generate(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
            required_suffix=row.get("required_suffix"),
            required_family=row.get("required_family"),
        )
        if result["status"] != "MODIFIED_TILING" or not result["npu_eligible"]:
            raise RuntimeError(f"{workload_id}: selector did not emit an NPU packet")
        applicable_family = str(row["required_applicable_family"])
        audit = result["candidate_audit"]
        if not audit["applicability"].get(applicable_family, False):
            raise RuntimeError(
                f"{workload_id}: required family {applicable_family} is not applicable"
            )
        if audit["unique_counts"].get(applicable_family, 0) <= 0:
            raise RuntimeError(
                f"{workload_id}: applicable family {applicable_family} emitted no candidate"
            )
        if case_role == "branch_probe":
            if result["formula_family"] != row["required_family"]:
                raise RuntimeError(f"{workload_id}: branch family drift")
            if int(result["kernel_suffix"]) != int(row["required_suffix"]):
                raise RuntimeError(f"{workload_id}: branch suffix drift")
            if result["selection_basis"] != "INDEPENDENT_BRANCH_COVERAGE_MINIMUM":
                raise RuntimeError(f"{workload_id}: branch selection attestation failed")
        elif result["selection_basis"] != "INDEPENDENT_GLOBAL_HARDWARE_COST_MINIMUM":
            raise RuntimeError(f"{workload_id}: global selection attestation failed")
        if audit["official_selector_called"] or audit["history_lookup"]:
            raise RuntimeError(f"{workload_id}: candidate decision is not independent")
        if case_role == "selector_top1" and not audit["winner_is_global_minimum"]:
            raise RuntimeError(f"{workload_id}: selected candidate is not global minimum")

        packet = result["improved"]
        blob = bytes.fromhex(packet["tiling_data_hex"])
        digest = hashlib.sha256(blob).hexdigest()
        if len(blob) != 272 or digest != packet["tiling_data_sha256"]:
            raise RuntimeError(f"{workload_id}: invalid improved packet")
        packet_path = args.output_dir / f"{workload_id}.bin"
        packet_path.write_bytes(blob)
        words = struct.unpack("<68I", blob)
        manifest_rows.append({
            "workload_id": workload_id,
            "rank": "0",
            "candidate_role": (
                "independent_branch_probe"
                if case_role == "branch_probe"
                else "independent_global_winner"
            ),
            "m": str(row["m"]), "n": str(row["n"]), "k": str(row["k"]),
            "dtype": str(row["dtype"]),
            "trans_a": str(int(row["trans_a"])),
            "trans_b": str(int(row["trans_b"])),
            "used_core_num": str(packet["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(packet["workspace_bytes"]),
            "tiling_path": str(packet_path.resolve()),
            "tiling_sha256": digest,
            "tiling_fnv1a64": fnv1a64(blob),
            "model_schedule_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "workload_id": workload_id,
                        "family": result["formula_family"],
                        "suffix": result["kernel_suffix"],
                        "packet": digest,
                        "cost": result["cost"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "is_reserve": "0",
            "l2_cache_flag": str(words[62]),
            "nd2nz_a": str(words[58]),
            "nd2nz_b": str(words[59]),
            "required_successful_tilings": "1",
        })
        result["workload_id"] = workload_id
        result["selection_axis"] = row["selection_axis"]
        result["required_applicable_family"] = applicable_family
        result["case_role"] = case_role
        selections.append(result)

    write_csv(args.manifest, manifest_rows)
    variants: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        variants[(row["dtype"], row["kernel_suffix"])].append(row)
    for index, ((dtype, suffix), rows) in enumerate(sorted(variants.items())):
        name = f"{dtype}_k{suffix}"
        write_csv(args.variant_dir / f"{name}.csv", rows)
        write_csv(args.sequence_dir / f"{index:02d}__{name}.csv", rows)

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", encoding="utf-8") as stream:
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        "RULE_MATRIX_GENERATED "
        f"shapes={len(manifest_rows)} "
        f"selector_top1={sum(row['candidate_role'] == 'independent_global_winner' for row in manifest_rows)} "
        f"branch_probes={sum(row['candidate_role'] == 'independent_branch_probe' for row in manifest_rows)} "
        f"applicable_families={len(declared_families)} variants={len(variants)}"
    )


if __name__ == "__main__":
    main()
