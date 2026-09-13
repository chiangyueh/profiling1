#!/usr/bin/env python3
"""Generate paired direct-baseline/candidate packets for all core rules."""
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

from core_ownership_rules import INSTALLED_SUFFIXES  # noqa: E402
from improved_selector import generate  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)

CANDIDATE_CASES = (
    ("base0_core_00", 17, 17, 17, "fp16", False, False, 0),
    ("base0_core_01", 19, 23, 31, "fp16", False, False, 0),
    ("base0_core_02", 31, 17, 33, "fp16", False, False, 0),
    ("base0_core_03", 33, 31, 17, "fp16", False, False, 0),
    ("base0_core_04", 47, 19, 29, "fp16", False, False, 0),
    ("base1_core_00", 16, 16, 16, "fp16", False, False, 1),
    ("base1_core_01", 32, 32, 32, "fp16", False, False, 1),
    ("base1_core_02", 48, 64, 64, "fp16", False, False, 1),
    ("base1_core_03", 64, 48, 128, "fp16", False, False, 1),
    ("base1_core_04", 80, 80, 256, "fp16", False, False, 1),
    ("al1_core_00", 1, 65, 5120, "fp32", False, True, 101),
    ("al1_core_01", 4, 80, 5504, "fp32", False, True, 101),
    ("al1_core_02", 8, 96, 6144, "fp32", False, True, 101),
    ("al1_core_03", 12, 128, 6656, "fp32", False, True, 101),
    ("al1_core_04", 16, 160, 7168, "fp32", False, True, 101),
    ("bl1_core_00", 4097, 128, 16, "fp16", False, False, 201),
    ("bl1_core_01", 4352, 128, 16, "fp16", False, False, 201),
    ("bl1_core_02", 4608, 128, 16, "fp16", False, False, 201),
    ("bl1_core_03", 4864, 128, 16, "fp16", False, False, 201),
    ("bl1_core_04", 4097, 256, 16, "fp16", False, False, 201),
)


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


def manifest_row(
    *, workload_id: str, request: dict, packet: dict, suffix: int,
    role: str, path: Path, schedule: dict,
) -> dict[str, str]:
    blob = bytes.fromhex(packet["tiling_data_hex"])
    if len(blob) != 272:
        raise RuntimeError(f"{workload_id}: packet is not 272 bytes")
    path.write_bytes(blob)
    words = struct.unpack("<68I", blob)
    return {
        "workload_id": workload_id, "rank": "0", "candidate_role": role,
        "m": str(request["M"]), "n": str(request["N"]), "k": str(request["K"]),
        "dtype": str(request["dtype"]),
        "trans_a": str(int(request["transA"])),
        "trans_b": str(int(request["transB"])),
        "used_core_num": str(packet["block_dim"]),
        "kernel_suffix": str(suffix),
        "workspace_bytes": str(packet["workspace_bytes"]),
        "tiling_path": str(path.resolve()),
        "tiling_sha256": hashlib.sha256(blob).hexdigest(),
        "tiling_fnv1a64": fnv1a64(blob),
        "model_schedule_sha256": hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "is_reserve": "0", "l2_cache_flag": str(words[62]),
        "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
        "required_successful_tilings": "1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--baseline-variants", type=Path, required=True)
    parser.add_argument("--candidate-variants", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    for directory in (
        args.baseline_dir, args.candidate_dir,
        args.baseline_variants, args.candidate_variants,
    ):
        clear_files(directory)

    audit_contract = json.loads(
        (SELECTOR_ROOT / "source_family_audit_contract.json").read_text()
    )
    audit_rows = []
    seen_suffixes = set()
    for row in audit_contract:
        result = generate(
            row["m"], row["k"], row["n"], row["dtype"],
            row["trans_a"], row["trans_b"],
        )
        plan = result["core_plan"]
        if (
            result["theory"]["source_family"] != row["expected_family"] or
            int(plan["suffix"]) != int(row["expected_suffix"])
        ):
            raise RuntimeError(f"{row['workload_id']}: source route drift")
        seen_suffixes.add(int(plan["suffix"]))
        audit_rows.append({
            "workload_id": row["workload_id"],
            "source_family": row["expected_family"],
            **plan,
        })
    if tuple(sorted(seen_suffixes)) != INSTALLED_SUFFIXES:
        raise RuntimeError("audit did not execute every installed suffix rule")

    baseline_rows = []
    candidate_rows = []
    selections = []
    changed_suffixes = set()
    for workload_id, m, n, k, dtype, trans_a, trans_b, expected_suffix in CANDIDATE_CASES:
        result = generate(m, k, n, dtype, trans_a, trans_b)
        if not (
            result["status"] == "MODIFIED_TILING" and result["npu_eligible"] and
            int(result["kernel_suffix"]) == expected_suffix and
            result["core_plan"]["eliminated_cores"] > 0
        ):
            raise RuntimeError(f"{workload_id}: no strict core-deletion candidate")
        request = result["request"]
        schedule = {
            "workload_id": workload_id,
            "suffix": expected_suffix,
            "core_plan": result["core_plan"],
            "changed_words": ["usedCoreNum"],
        }
        baseline_rows.append(manifest_row(
            workload_id=workload_id, request=request, packet=result["baseline"],
            suffix=expected_suffix, role="family_core_baseline",
            path=args.baseline_dir / f"{workload_id}.bin", schedule=schedule,
        ))
        candidate_rows.append(manifest_row(
            workload_id=workload_id, request=request, packet=result["improved"],
            suffix=expected_suffix, role="family_core_candidate",
            path=args.candidate_dir / f"{workload_id}.bin", schedule=schedule,
        ))
        result["workload_id"] = workload_id
        result["selection_axis"] = result["core_plan"]["branch"]
        selections.append(result)
        changed_suffixes.add(expected_suffix)

    expected_changed = {0, 1, 101, 201}
    if changed_suffixes != expected_changed:
        raise RuntimeError(f"changed suffix coverage drift: {sorted(changed_suffixes)}")
    write_csv(args.baseline_manifest, baseline_rows)
    write_csv(args.candidate_manifest, candidate_rows)

    grouped_baseline = defaultdict(list)
    grouped_candidate = defaultdict(list)
    for row in baseline_rows:
        grouped_baseline[(row["dtype"], row["kernel_suffix"])].append(row)
    for row in candidate_rows:
        grouped_candidate[(row["dtype"], row["kernel_suffix"])].append(row)
    if set(grouped_baseline) != set(grouped_candidate):
        raise RuntimeError("paired variant coverage mismatch")
    for index, key in enumerate(sorted(grouped_candidate)):
        dtype, suffix = key
        name = f"{index:02d}__{dtype}_k{suffix}.csv"
        write_csv(args.baseline_variants / name, grouped_baseline[key])
        write_csv(args.candidate_variants / name, grouped_candidate[key])

    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", encoding="utf-8") as stream:
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    with args.audit.open("w", encoding="utf-8") as stream:
        for row in audit_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        "CORE_OWNERSHIP_MATRIX_GENERATED "
        f"audited_suffixes={len(seen_suffixes)} audit_witnesses={len(audit_rows)} "
        f"paired_shapes={len(candidate_rows)} changed_suffixes={len(changed_suffixes)} "
        f"compiled_variants={len(grouped_candidate)}"
    )


if __name__ == "__main__":
    main()
