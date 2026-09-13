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

BRANCH_CASES = (
    ("base0_core_00", 17, 17, 17, "fp16", False, False, 0),
    ("base0_core_01", 19, 23, 31, "fp16", False, False, 0),
    ("base0_core_02", 31, 27, 33, "fp16", False, False, 0),
    ("base0_core_03", 33, 31, 37, "fp16", False, False, 0),
    ("base0_core_04", 47, 19, 29, "fp16", False, False, 0),
    ("base1_core_00", 16, 16, 16, "fp16", False, False, 1),
    ("base1_core_01", 32, 32, 32, "fp16", False, False, 1),
    ("base1_core_02", 48, 64, 64, "fp16", False, False, 1),
    ("base1_core_03", 64, 48, 128, "fp16", False, False, 1),
    ("base1_core_04", 80, 80, 256, "fp16", False, False, 1),
    ("sc20_core_00", 896, 2049, 27392, "fp16", False, False, 20),
    ("sc20_core_01", 1152, 1921, 28032, "fp16", False, False, 20),
    ("sc20_core_02", 1280, 1793, 28416, "fp16", False, False, 20),
    ("sc20_core_03", 1408, 1665, 28800, "fp16", False, False, 20),
    ("sc20_core_04", 1536, 1537, 29056, "fp16", False, False, 20),
    ("sc21_core_00", 896, 2176, 30336, "fp16", False, False, 21),
    ("sc21_core_01", 1024, 2944, 31616, "fp16", False, False, 21),
    ("sc21_core_02", 1152, 2816, 28032, "fp16", False, False, 21),
    ("sc21_core_03", 1280, 2560, 33024, "fp16", False, False, 21),
    ("sc21_core_04", 1408, 2304, 29696, "fp16", False, False, 21),
    ("det30_core_00", 32, 49, 7808, "fp16", False, False, 30),
    ("det30_core_01", 48, 97, 8064, "fp16", False, False, 30),
    ("det30_core_02", 96, 113, 10368, "fp16", False, False, 30),
    ("det30_core_03", 64, 65, 13056, "fp16", False, False, 30),
    ("det30_core_04", 80, 17, 11392, "fp16", False, False, 30),
    ("det31_core_00", 80, 112, 12032, "fp16", False, False, 31),
    ("det31_core_01", 64, 96, 14208, "fp16", False, False, 31),
    ("det31_core_02", 128, 512, 28032, "fp16", False, False, 31),
    ("det31_core_03", 512, 48, 14592, "fp16", False, False, 31),
    ("det31_core_04", 112, 64, 17024, "fp16", False, False, 31),
    ("al1_core_00", 1, 65, 5120, "fp32", False, True, 101),
    ("al1_core_01", 4, 80, 5504, "fp32", False, True, 101),
    ("al1_core_02", 8, 96, 6144, "fp32", False, True, 101),
    ("al1_core_03", 12, 128, 6656, "fp32", False, True, 101),
    ("al1_core_04", 16, 160, 7168, "fp32", False, True, 101),
    ("bl1200_core_00", 74432, 32, 9, "fp32", False, False, 200),
    ("bl1200_core_01", 72848, 64, 5, "fp32", False, False, 200),
    ("bl1200_core_02", 80240, 128, 3, "fp32", False, False, 200),
    ("bl1200_core_03", 83776, 192, 11, "fp32", False, False, 200),
    ("bl1200_core_04", 88560, 256, 13, "fp32", False, False, 200),
    ("bl1201_core_00", 4109, 96, 48, "fp16", False, False, 201),
    ("bl1201_core_01", 4276, 64, 112, "fp16", False, False, 201),
    ("bl1201_core_02", 4463, 128, 32, "fp16", False, False, 201),
    ("bl1201_core_03", 4413, 256, 96, "fp16", False, False, 201),
    ("bl1201_core_04", 4381, 192, 128, "fp16", False, False, 201),
    ("fix10200_core_00", 17408, 11, 11, "fp32", False, False, 10200),
    ("fix10200_core_01", 14848, 7, 15, "fp32", False, False, 10200),
    ("fix10200_core_02", 18176, 3, 9, "fp32", False, False, 10200),
    ("fix10200_core_03", 22144, 13, 5, "fp32", False, False, 10200),
    ("fix10200_core_04", 21504, 9, 3, "fp32", False, False, 10200),
    ("fix10201_core_00", 24064, 34, 112, "fp16", False, False, 10201),
    ("fix10201_core_01", 19200, 25, 16, "fp16", False, False, 10201),
    ("fix10201_core_02", 28160, 153, 96, "fp16", False, False, 10201),
    ("fix10201_core_03", 18560, 61, 80, "fp16", False, False, 10201),
    ("fix10201_core_04", 23424, 130, 48, "fp16", False, False, 10201),
    ("vec20201_core_00", 11776, 56, 240, "fp32", False, False, 20201),
    ("vec20201_core_01", 12288, 17, 16, "fp32", False, False, 20201),
    ("vec20201_core_02", 16384, 31, 24, "fp32", False, False, 20201),
    ("vec20201_core_03", 20480, 47, 32, "fp32", False, False, 20201),
    ("vec20201_core_04", 22528, 63, 40, "fp32", False, False, 20201),
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

    baseline_rows = []
    candidate_rows = []
    selections = []
    audit_rows = []
    seen_suffixes = set()
    suffix_counts = defaultdict(int)
    suffix_shapes = defaultdict(list)
    changed_suffixes = set()
    for workload_id, m, n, k, dtype, trans_a, trans_b, expected_suffix in BRANCH_CASES:
        result = generate(m, k, n, dtype, trans_a, trans_b)
        if int(result["kernel_suffix"]) != expected_suffix:
            raise RuntimeError(
                f"{workload_id}: source suffix {result['kernel_suffix']} != {expected_suffix}"
            )
        request = result["request"]
        schedule = {
            "workload_id": workload_id,
            "suffix": expected_suffix,
            "core_plan": result["core_plan"],
            "changed_words": (
                ["usedCoreNum"] if result["status"] == "MODIFIED_TILING" else []
            ),
        }
        baseline_rows.append(manifest_row(
            workload_id=workload_id, request=request, packet=result["baseline"],
            suffix=expected_suffix, role="family_core_baseline",
            path=args.baseline_dir / f"{workload_id}.bin", schedule=schedule,
        ))
        if result["status"] == "MODIFIED_TILING":
            if not (
                result["npu_eligible"] and
                result["core_plan"]["eliminated_cores"] > 0
            ):
                raise RuntimeError(f"{workload_id}: invalid modified packet")
            candidate_rows.append(manifest_row(
                workload_id=workload_id, request=request, packet=result["improved"],
                suffix=expected_suffix, role="family_core_candidate",
                path=args.candidate_dir / f"{workload_id}.bin", schedule=schedule,
            ))
            changed_suffixes.add(expected_suffix)
        result["workload_id"] = workload_id
        result["selection_axis"] = result["core_plan"]["branch"]
        selections.append(result)
        audit_rows.append({
            "workload_id": workload_id,
            "source_family": result["theory"]["source_family"],
            **result["core_plan"],
        })
        seen_suffixes.add(expected_suffix)
        suffix_counts[expected_suffix] += 1
        suffix_shapes[expected_suffix].append((m, n, k))

    expected_changed = {0, 1, 101, 201}
    if tuple(sorted(seen_suffixes)) != INSTALLED_SUFFIXES:
        raise RuntimeError("matrix did not execute every installed suffix rule")
    if any(suffix_counts[suffix] != 5 for suffix in INSTALLED_SUFFIXES):
        raise RuntimeError(f"each suffix requires five shapes: {dict(suffix_counts)}")
    for suffix, shapes in suffix_shapes.items():
        for axis, index in (("M", 0), ("N", 1), ("K", 2)):
            if len({shape[index] for shape in shapes}) != 5:
                raise RuntimeError(
                    f"suffix {suffix} does not vary {axis} across all five shapes"
                )
    if changed_suffixes != expected_changed:
        raise RuntimeError(f"changed suffix coverage drift: {sorted(changed_suffixes)}")
    if len(baseline_rows) != 60 or len(candidate_rows) != 20:
        raise RuntimeError(
            f"expected 60 branch executions and 20 paired changes; got "
            f"{len(baseline_rows)} and {len(candidate_rows)}"
        )
    write_csv(args.baseline_manifest, baseline_rows)
    write_csv(args.candidate_manifest, candidate_rows)

    grouped_baseline = defaultdict(list)
    grouped_candidate = defaultdict(list)
    for row in baseline_rows:
        grouped_baseline[(row["dtype"], row["kernel_suffix"])].append(row)
    for row in candidate_rows:
        grouped_candidate[(row["dtype"], row["kernel_suffix"])].append(row)
    if len(grouped_baseline) != 12 or len(grouped_candidate) != 4:
        raise RuntimeError("expected twelve baseline variants and four changed variants")
    if not set(grouped_candidate).issubset(grouped_baseline):
        raise RuntimeError("candidate variant is absent from baseline matrix")
    for key in sorted(grouped_baseline):
        dtype, suffix = key
        name = f"{dtype}_k{suffix}.csv"
        write_csv(args.baseline_variants / name, grouped_baseline[key])
    for key in sorted(grouped_candidate):
        dtype, suffix = key
        name = f"{dtype}_k{suffix}.csv"
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
        f"audited_suffixes={len(seen_suffixes)} branch_shapes={len(audit_rows)} "
        f"paired_shapes={len(candidate_rows)} changed_suffixes={len(changed_suffixes)} "
        f"compiled_variants={len(grouped_baseline)}"
    )


if __name__ == "__main__":
    main()
