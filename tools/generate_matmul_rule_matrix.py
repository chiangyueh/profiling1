#!/usr/bin/env python3
"""Freeze exactly one independent improved packet per installed 8.1 branch."""

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
sys.path.insert(0, str(SELECTOR_ROOT))

from improved_selector import KERNEL_VARIANT, generate  # noqa: E402
from check_matmul_rule_coverage import EXPECTED_RULES  # noqa: E402


MANIFEST_FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
WORKLOAD_FIELDS = ("workload_id", "m", "n", "k", "dtype", "trans_a", "trans_b")


def fnv1a64(blob):
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(
        (SELECTOR_ROOT / "branch_contract.json").read_text(encoding="utf-8")
    )
    if {int(row["suffix"]) for row in contract} != set(KERNEL_VARIANT):
        raise RuntimeError("branch contract does not cover the installed dispatch table")
    if {rule for row in contract for rule in row["required_rules"]} != EXPECTED_RULES:
        raise RuntimeError("branch contract does not cover all modified rule groups")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    workload_rows = []
    selections = []
    for row in contract:
        suffix = int(row["suffix"])
        workload_id = f"k{suffix}_{row['family'].lower()}"
        result = generate(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
        )
        if result["status"] != "MODIFIED_TILING" or not result["npu_eligible"]:
            raise RuntimeError(
                f"{workload_id}: improved branch is not executable: "
                f"{result.get('skip_reason', result['status'])}"
            )
        if result["baseline_equivalent"] or int(result["kernel_suffix"]) != suffix:
            raise RuntimeError(f"{workload_id}: branch witness identity mismatch")
        missing = set(row["required_rules"]) - set(result["changed_rules"])
        if missing:
            raise RuntimeError(f"{workload_id}: inactive rules {sorted(missing)}")

        packet = result["improved"]
        blob = bytes.fromhex(packet["tiling_data_hex"])
        digest = hashlib.sha256(blob).hexdigest()
        if len(blob) != 272 or digest != packet["tiling_data_sha256"]:
            raise RuntimeError(f"{workload_id}: invalid improved ABI packet")
        packet_path = args.output_dir / f"{workload_id}.bin"
        packet_path.write_bytes(blob)
        words = struct.unpack("<68I", blob)
        manifest_rows.append({
            "workload_id": workload_id,
            "rank": "0",
            "candidate_role": "independent_improved",
            "m": str(row["m"]), "n": str(row["n"]), "k": str(row["k"]),
            "dtype": str(row["dtype"]),
            "trans_a": str(int(row["trans_a"])),
            "trans_b": str(int(row["trans_b"])),
            "used_core_num": str(packet["block_dim"]),
            "kernel_suffix": str(suffix),
            "workspace_bytes": str(packet["workspace_bytes"]),
            "tiling_path": str(packet_path.resolve()),
            "tiling_sha256": digest,
            "tiling_fnv1a64": fnv1a64(blob),
            "model_schedule_sha256": hashlib.sha256(
                (workload_id + ":improved:" + digest).encode("ascii")
            ).hexdigest(),
            "is_reserve": "0", "l2_cache_flag": str(words[62]),
            "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
            "required_successful_tilings": "1",
        })
        workload_rows.append({name: manifest_rows[-1][name] for name in WORKLOAD_FIELDS})
        result["workload_id"] = workload_id
        result["contract_variant"] = row["variant"]
        selections.append(result)

    if len(manifest_rows) != len(KERNEL_VARIANT):
        raise RuntimeError("campaign must contain one improved packet per branch")
    write_csv(args.manifest, MANIFEST_FIELDS, manifest_rows)
    write_csv(args.workloads, WORKLOAD_FIELDS, workload_rows)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", encoding="utf-8") as stream:
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

    variants = defaultdict(list)
    for row in manifest_rows:
        variants[(row["dtype"], row["kernel_suffix"])].append(row)
    for (dtype, suffix), rows in variants.items():
        write_csv(args.variant_dir / f"{dtype}_k{suffix}.csv", MANIFEST_FIELDS, rows)
    for index, row in enumerate(manifest_rows):
        write_csv(
            args.sequence_dir / f"{index:02d}__{row['dtype']}_k{row['kernel_suffix']}.csv",
            MANIFEST_FIELDS, [row],
        )
    print(
        f"RULE_MATRIX_GENERATED branches={len(contract)} improved_packets={len(manifest_rows)} "
        f"official_workloads={len(workload_rows)} variants={len(variants)}"
    )


if __name__ == "__main__":
    main()
