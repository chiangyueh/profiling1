#!/usr/bin/env python3
"""Materialize the fixed 200-shape independent-kernel campaign."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "matmul_rule_selector"))

from independent_tiled_selector import generate  # noqa: E402
from independent_validation_cases import build_cases  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
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


def manifest_row(case: dict, result: dict, path: Path, blob: bytes) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    words = struct.unpack("<70I", blob)
    schedule = {
        "family": result["formula_family"],
        "mode": result["kernel_mode"],
        "cube": result["cube"],
        "decision": result["decision"],
        "derivation": result["derivation"],
    }
    return {
        "workload_id": case["workload_id"], "rank": "0",
        "candidate_role": "independent_formula_tiling",
        "m": str(case["m"]), "n": str(case["n"]), "k": str(case["k"]),
        "dtype": case["dtype"], "trans_a": str(int(case["trans_a"])),
        "trans_b": str(int(case["trans_b"])),
        "used_core_num": str(result["block_dim"]),
        "kernel_suffix": str(result["kernel_suffix"]),
        "workspace_bytes": str(result["workspace_bytes"]),
        "tiling_path": str(path.resolve()),
        "tiling_sha256": hashlib.sha256(blob).hexdigest(),
        "tiling_fnv1a64": fnv1a64(blob),
        "model_schedule_sha256": hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "is_reserve": "0", "l2_cache_flag": str(words[64]),
        "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
        "required_successful_tilings": "1",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root
    packet_dir = output / "packets/91000"
    variant_dir = output / "variants"
    output.mkdir(parents=True, exist_ok=True)
    packet_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selections = []
    family_counts = Counter()
    dtype_counts = Counter()
    for case in build_cases():
        result = generate(
            case["m"], case["k"], case["n"], case["dtype"],
            case["trans_a"], case["trans_b"],
        )
        if any(result["runtime_dependencies"].values()):
            raise RuntimeError(f"{case['workload_id']}: forbidden runtime dependency")
        blob = bytes.fromhex(result["packet_hex"])
        if len(blob) != 280:
            raise RuntimeError(f"{case['workload_id']}: packet is not 280 bytes")
        row = manifest_row(
            case, result, packet_dir / f"{case['workload_id']}.bin", blob
        )
        rows.append(row)
        family_counts[result["formula_family"]] += 1
        dtype_counts[case["dtype"]] += 1
        selections.append({
            **case,
            "formula_family": result["formula_family"],
            "kernel_mode": result["kernel_mode"],
            "kernel_suffix": result["kernel_suffix"],
            "block_dim": result["block_dim"],
            "packet_sha256": result["packet_sha256"],
            "selection_basis": result["selection_basis"],
            "decision": result["decision"],
            "derivation": result["derivation"],
        })

    if len(rows) != 200 or len({row["workload_id"] for row in rows}) != 200:
        raise RuntimeError("campaign must contain exactly 200 unique workloads")
    if set(family_counts) != {
        "MICRO_DIRECT", "BALANCED_MN", "SEEDED_SPLIT_K",
        "RESIDENT_B_M_STRIPE", "RESIDENT_A_N_STRIPE", "SEEDED_TAIL_WAVE",
    }:
        raise RuntimeError(f"mode coverage changed: {family_counts}")
    if set(dtype_counts) != {"fp16", "bf16", "fp32"}:
        raise RuntimeError(f"dtype coverage changed: {dtype_counts}")

    write_csv(output / "manifest.csv", rows)
    for dtype in ("fp16", "bf16", "fp32"):
        write_csv(
            variant_dir / f"{dtype}_k91000.csv",
            [row for row in rows if row["dtype"] == dtype],
        )
    with (output / "selection.jsonl").open("w", encoding="utf-8") as stream:
        for selection in selections:
            stream.write(json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        "INDEPENDENT_MATRIX_GENERATED shapes=200 variants=3 packet_bytes=280 "
        + " ".join(f"{name}={count}" for name, count in sorted(family_counts.items()))
    )


if __name__ == "__main__":
    main()
