#!/usr/bin/env python3
"""Materialize novel packets and the closest official protocol control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR))

from novel_family_selector import generate  # noqa: E402
from novel_validation_cases import CASES  # noqa: E402


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


def row_for(case: dict, result: dict, suffix: int, path: Path,
            blob: bytes, protocol: str, cube_override: dict | None = None) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    words = struct.unpack("<70I", blob)
    schedule = {
        "family": result["formula_family"],
        "protocol": protocol,
        "suffix": suffix,
        "cube": result["cube"] if cube_override is None else cube_override,
        "derivation": result["derivation"],
    }
    return {
        "workload_id": case["workload_id"], "rank": "0",
        "candidate_role": "independent_experimental_family",
        "m": str(case["m"]), "n": str(case["n"]), "k": str(case["k"]),
        "dtype": case["dtype"], "trans_a": str(int(case["trans_a"])),
        "trans_b": str(int(case["trans_b"])),
        "used_core_num": str(result["block_dim"]),
        "kernel_suffix": str(suffix),
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
    output.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    reference_rows = []
    selection_rows = []
    for case in CASES:
        result = generate(
            case["m"], case["k"], case["n"], case["dtype"],
            case["trans_a"], case["trans_b"],
            required_family=case["required_family"],
        )
        if any(result["runtime_dependencies"].values()):
            raise RuntimeError(f"{case['workload_id']}: forbidden runtime dependency")
        blob = bytes.fromhex(result["packet_hex"])
        suffix = result["kernel_suffix"]
        candidate_rows.append(row_for(
            case, result, suffix,
            output / "packets" / str(suffix) / f"{case['workload_id']}.bin",
            blob, "repository_owned",
        ))

        if case["required_family"] == "SEEDED_ATOMIC_SPLIT_K":
            control = bytearray(blob)
            struct.pack_into("<I", control, 49 * 4, 0)
            control_cube = dict(result["cube"])
            control_cube["reserved"] = 0
            reference_rows.append(row_for(
                case, result, 41,
                output / "packets" / "41" / f"{case['workload_id']}.bin",
                bytes(control), "later_official_clear_then_all_atomic", control_cube,
            ))

        selection_rows.append({
            **case,
            "kernel_suffix": suffix,
            "block_dim": result["block_dim"],
            "packet_sha256": result["packet_sha256"],
            "derivation": result["derivation"],
            "origin": result["origin"],
        })

    by_suffix: dict[int, list[dict[str, str]]] = {90001: [], 90002: []}
    for row in candidate_rows:
        by_suffix[int(row["kernel_suffix"])].append(row)
    if {suffix: len(rows) for suffix, rows in by_suffix.items()} != {90001: 100, 90002: 100}:
        raise RuntimeError("novel family shape coverage changed")
    if len(reference_rows) != 100:
        raise RuntimeError("closest official protocol control must contain 100 shapes")

    write_csv(output / "manifest.csv", candidate_rows)
    write_csv(output / "variants/fp32_k90001.csv", by_suffix[90001])
    write_csv(output / "variants/fp32_k90002.csv", by_suffix[90002])
    write_csv(output / "variants/fp32_k41.csv", reference_rows)
    with (output / "selection.jsonl").open("w", encoding="utf-8") as stream:
        for row in selection_rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        "NOVEL_MATRIX_GENERATED shapes=200 seeded_atomic=100 seeded_tail=100 "
        "closest_official_controls=100 variants=3 packet_bytes=280"
    )


if __name__ == "__main__":
    main()
