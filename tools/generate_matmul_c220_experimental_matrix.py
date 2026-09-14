#!/usr/bin/env python3
"""Materialize one explicit C220 structural-family packet per workload."""
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

from c220_experimental_selector import generate  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
EXPECTED_ROWS = 12
EXPECTED_NPU_ROWS = 8
EXPECTED_FAMILIES = {
    "MULTI_CORE_SPLIT_K", "SINGLE_CORE_NKM_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
    "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD",
}
EXPECTED_VARIANTS = {
    ("fp32", 41), ("fp32", 51),
    ("fp16", 121), ("bf16", 121),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads((SELECTOR_ROOT / "c220_validation_contract.json").read_text(encoding="utf-8"))
    ids = {row["workload_id"] for row in contract}
    families = {row["required_family"] for row in contract}
    variants = {(row["dtype"], int({
        "MULTI_CORE_SPLIT_K": 41,
        "SINGLE_CORE_NKM_SPLIT_K": 51,
        "SINGLE_CORE_SPLIT_K_GM_TO_L1": 61,
        "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED": 60,
        "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": 121,
    }[row["required_family"]])) for row in contract
        if row["required_family"] not in (
            "SINGLE_CORE_SPLIT_K_GM_TO_L1",
            "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED",
        )}
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
        )
        if result["formula_family"] != row["required_family"]:
            raise RuntimeError(f"{row['workload_id']}: family selection drift")
        if any(result["runtime_dependencies"].values()):
            raise RuntimeError(f"{row['workload_id']}: forbidden runtime dependency")
        blob = bytes.fromhex(result["packet_hex"])
        if len(blob) != 280 or hashlib.sha256(blob).hexdigest() != result["packet_sha256"]:
            raise RuntimeError(f"{row['workload_id']}: invalid 280-byte packet")
        schedule = {
            "family": result["formula_family"], "suffix": result["kernel_suffix"],
            "cube": result["cube"], "derivation": result["derivation"],
        }
        if result["npu_eligible"]:
            path = args.output_dir / f"{row['workload_id']}.bin"
            path.write_bytes(blob)
            words = struct.unpack("<70I", blob)
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
            "case_role": "experimental_family" if result["npu_eligible"] else "host_solved_toolchain_blocked",
        })
        selections.append(result)

    if len(manifest_rows) != EXPECTED_NPU_ROWS:
        raise RuntimeError(
            f"expected {EXPECTED_NPU_ROWS} buildable packets, got {len(manifest_rows)}"
        )

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
    print(
        f"C220_MATRIX_GENERATED solved_shapes={len(selections)} "
        f"npu_shapes={len(manifest_rows)} families={len(families)} "
        f"variants={len(grouped)} packet_bytes=280"
    )


if __name__ == "__main__":
    main()
