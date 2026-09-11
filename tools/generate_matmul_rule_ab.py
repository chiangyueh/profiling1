#!/usr/bin/env python3
"""Freeze one exact baseline packet and one rule packet per NPU workload."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
if str(SELECTOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SELECTOR_ROOT))

from improved_selector import generate  # noqa: E402


CASES = (
    {
        "workload_id": "base_tail_m1025_k256_n768",
        "m": 1025,
        "n": 768,
        "k": 256,
        "dtype": "fp16",
        "trans_a": False,
        "trans_b": False,
        "expected_rule": "R02_WAVE_TAIL_REDISTRIBUTION",
    },
    {
        "workload_id": "bl1_m32768_k512_n512_nt",
        "m": 32768,
        "n": 512,
        "k": 512,
        "dtype": "fp16",
        "trans_a": False,
        "trans_b": True,
        "expected_rule": "R06_BL1_RESIDENCY",
    },
    {
        "workload_id": "det_splitk_m16_k4096_n16",
        "m": 16,
        "n": 16,
        "k": 4096,
        "dtype": "fp16",
        "trans_a": False,
        "trans_b": False,
        "expected_rule": "R08_DETERMINISTIC_SPLIT_K",
    },
    {
        "workload_id": "sc_splitk_m8192_k4096_n4096",
        "m": 8192,
        "n": 4096,
        "k": 4096,
        "dtype": "fp16",
        "trans_a": False,
        "trans_b": False,
        "expected_rule": "R07_SINGLE_CORE_SPLIT_K_GRID",
    },
)

MANIFEST_FIELDS = (
    "workload_id",
    "rank",
    "candidate_role",
    "m",
    "n",
    "k",
    "dtype",
    "trans_a",
    "trans_b",
    "used_core_num",
    "kernel_suffix",
    "workspace_bytes",
    "tiling_path",
    "tiling_sha256",
    "tiling_fnv1a64",
    "model_schedule_sha256",
    "is_reserve",
    "l2_cache_flag",
    "nd2nz_a",
    "nd2nz_b",
    "required_successful_tilings",
)


def fnv1a64(blob: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    selections: list[dict[str, object]] = []

    for case in CASES:
        result = generate(
            case["m"],
            case["k"],
            case["n"],
            case["dtype"],
            case["trans_a"],
            case["trans_b"],
        )
        if result["status"] != "MODIFIED_TILING":
            raise RuntimeError(
                f"{case['workload_id']}: expected a modified tiling, got "
                f"{result['status']} ({result.get('skip_reason', '')})"
            )
        if result["baseline_equivalent"] or not result["changed_fields"]:
            raise RuntimeError(f"{case['workload_id']}: rule packet equals baseline")
        if case["expected_rule"] not in result["changed_rules"]:
            raise RuntimeError(
                f"{case['workload_id']}: expected rule {case['expected_rule']} was not active"
            )

        selection = dict(result)
        selection["workload_id"] = case["workload_id"]
        selection["expected_rule"] = case["expected_rule"]
        selections.append(selection)

        for rank, role in enumerate(("baseline", "improved")):
            packet = result[role]
            blob = bytes.fromhex(packet["tiling_data_hex"])
            if len(blob) != 272:
                raise RuntimeError(f"{case['workload_id']}/{role}: ABI is not 272 bytes")
            digest = hashlib.sha256(blob).hexdigest()
            if digest != packet["tiling_data_sha256"]:
                raise RuntimeError(f"{case['workload_id']}/{role}: SHA-256 mismatch")
            suffix = int(packet["tiling_key"]) - 10**19
            if suffix not in (1, 21, 31, 201):
                raise RuntimeError(
                    f"{case['workload_id']}/{role}: unsupported kernel suffix {suffix}"
                )
            filename = f"{case['workload_id']}__{role}__k{suffix}.bin"
            packet_path = args.output_dir / filename
            packet_path.write_bytes(blob)
            words = struct.unpack("<68I", blob)
            schedule_hash = hashlib.sha256(
                (case["workload_id"] + ":" + role + ":" + digest).encode("ascii")
            ).hexdigest()
            manifest_rows.append(
                {
                    "workload_id": str(case["workload_id"]),
                    "rank": str(rank),
                    "candidate_role": "searched",
                    "m": str(case["m"]),
                    "n": str(case["n"]),
                    "k": str(case["k"]),
                    "dtype": str(case["dtype"]),
                    "trans_a": str(int(case["trans_a"])),
                    "trans_b": str(int(case["trans_b"])),
                    "used_core_num": str(packet["block_dim"]),
                    "kernel_suffix": str(suffix),
                    "workspace_bytes": str(packet["workspace_bytes"]),
                    "tiling_path": str(packet_path.resolve()),
                    "tiling_sha256": digest,
                    "tiling_fnv1a64": fnv1a64(blob),
                    "model_schedule_sha256": schedule_hash,
                    "is_reserve": "0",
                    "l2_cache_flag": str(words[62]),
                    "nd2nz_a": str(words[58]),
                    "nd2nz_b": str(words[59]),
                    "required_successful_tilings": "2",
                }
            )

    if len(manifest_rows) != 2 * len(CASES):
        raise RuntimeError("each workload must have exactly two frozen packets")
    for offset in range(0, len(manifest_rows), 2):
        baseline, improved = manifest_rows[offset : offset + 2]
        if baseline["tiling_sha256"] == improved["tiling_sha256"]:
            raise RuntimeError(f"{baseline['workload_id']}: packets are byte-identical")

    write_csv(args.manifest, manifest_rows)
    with args.selection.open("w", encoding="utf-8") as stream:
        for selection in selections:
            stream.write(json.dumps(selection, sort_keys=True, separators=(",", ":")))
            stream.write("\n")

    variants: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        variants[(row["dtype"], row["kernel_suffix"])].append(dict(row))
    args.variant_dir.mkdir(parents=True, exist_ok=True)
    for (dtype, suffix), rows in sorted(variants.items()):
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            counts[row["workload_id"]] += 1
        for row in rows:
            row["required_successful_tilings"] = str(counts[row["workload_id"]])
        write_csv(args.variant_dir / f"{dtype}_k{suffix}.csv", rows)

    args.sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(manifest_rows):
        sequenced = dict(row)
        sequenced["required_successful_tilings"] = "1"
        write_csv(
            args.sequence_dir
            / (
                f"{index:02d}__{row['workload_id']}__r{row['rank']}__"
                f"{row['dtype']}_k{row['kernel_suffix']}.csv"
            ),
            [sequenced],
        )

    print(
        "RULE_AB_GENERATED "
        f"shapes={len(CASES)} packets={len(manifest_rows)} variants={len(variants)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
