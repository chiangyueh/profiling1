#!/usr/bin/env python3
"""Freeze a small discriminative improved set with audited historical references."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import defaultdict
from pathlib import Path


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
REFERENCE_FIELDS = (
    "workload_id", "source", "candidate_role", "execution_mode", "success",
    "preflight_passed", "preflight_mode", "error", "min_ms", "mean_ms",
    "median_ms", "stddev_ms", "p95_ms", "max_ms", "warmup", "repeat",
    "samples", "device", "toolkit", "measurement_method", "record_uid",
    "source_sha256", "selection_axis",
)
SAMPLE_FIELDS = ("workload_id", "sample", "latency_ms")


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


def percentile_95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--historical-profile", type=Path, required=True)
    parser.add_argument("--historical-samples", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(
        (SELECTOR_ROOT / "validation_contract.json").read_text(encoding="utf-8")
    )
    if len(contract) != 5 or len({row["workload_id"] for row in contract}) != 5:
        raise RuntimeError("validation contract must contain five unique shapes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.variant_dir.mkdir(parents=True, exist_ok=True)
    args.sequence_dir.mkdir(parents=True, exist_ok=True)
    for directory in (args.output_dir, args.variant_dir, args.sequence_dir):
        for stale in directory.iterdir():
            if stale.is_file():
                stale.unlink()

    manifest_rows = []
    reference_rows = []
    reference_sample_rows = []
    selections = []
    for row in contract:
        workload_id = str(row["workload_id"])
        result = generate(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
        )
        if result["status"] != "MODIFIED_TILING" or not result["npu_eligible"]:
            raise RuntimeError(
                f"{workload_id}: improved tiling is not executable: "
                f"{result.get('skip_reason', result['status'])}"
            )
        if result["baseline"]["selected_family"] != row["expected_baseline_family"]:
            raise RuntimeError(f"{workload_id}: reconstructed baseline family drift")
        if result["formula_family"] != row["expected_improved_family"]:
            raise RuntimeError(f"{workload_id}: improved family drift")
        if int(result["kernel_suffix"]) != int(row["expected_suffix"]):
            raise RuntimeError(f"{workload_id}: improved suffix drift")
        missing = set(row["required_rules"]) - set(result["changed_rules"])
        if missing or result["baseline_equivalent"]:
            raise RuntimeError(f"{workload_id}: inactive discriminative rules {sorted(missing)}")

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
            "kernel_suffix": str(result["kernel_suffix"]),
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

        reference = row["reference"]
        samples = [float(value) for value in reference["samples_ms"]]
        if (len(samples) != 7 or any(not math.isfinite(value) or value <= 0 for value in samples) or
                reference["device"] != "Ascend910B3" or
                reference["toolkit"] != "CANN 8.1" or
                reference["source"] != "installed_aclnn_matmul" or
                reference["measurement_method"] != "device_event_latency" or
                reference["preflight_mode"] != "numeric_signed_axes_full_v3" or
                int(reference["warmup"]) != 2 or int(reference["repeat"]) != 20):
            raise RuntimeError(f"{workload_id}: historical reference contract is invalid")
        reference_rows.append({
            "workload_id": workload_id,
            "source": "historical_installed_aclnn_matmul_device_event",
            "candidate_role": "installed_operator_reference",
            "execution_mode": "official_matmul_v3",
            "success": "1", "preflight_passed": "1",
            "preflight_mode": reference["preflight_mode"], "error": "",
            "min_ms": min(samples), "mean_ms": statistics.fmean(samples),
            "median_ms": statistics.median(samples),
            "stddev_ms": statistics.pstdev(samples),
            "p95_ms": percentile_95(samples), "max_ms": max(samples),
            "warmup": str(reference["warmup"]),
            "repeat": str(reference["repeat"]), "samples": str(len(samples)),
            "device": reference["device"], "toolkit": reference["toolkit"],
            "measurement_method": reference["measurement_method"],
            "record_uid": reference["record_uid"],
            "source_sha256": reference["source_sha256"],
            "selection_axis": row["selection_axis"],
        })
        reference_sample_rows.extend({
            "workload_id": workload_id, "sample": str(index), "latency_ms": value,
        } for index, value in enumerate(samples))
        result["workload_id"] = workload_id
        result["selection_axis"] = row["selection_axis"]
        result["historical_reference_record_uid"] = reference["record_uid"]
        selections.append(result)

    write_csv(args.manifest, MANIFEST_FIELDS, manifest_rows)
    write_csv(args.historical_profile, REFERENCE_FIELDS, reference_rows)
    write_csv(args.historical_samples, SAMPLE_FIELDS, reference_sample_rows)
    args.selection.parent.mkdir(parents=True, exist_ok=True)
    with args.selection.open("w", encoding="utf-8") as stream:
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")

    variants = defaultdict(list)
    for manifest_row in manifest_rows:
        variants[(manifest_row["dtype"], manifest_row["kernel_suffix"])].append(manifest_row)
    for (dtype, suffix), rows in variants.items():
        write_csv(args.variant_dir / f"{dtype}_k{suffix}.csv", MANIFEST_FIELDS, rows)
    for index, manifest_row in enumerate(manifest_rows):
        write_csv(
            args.sequence_dir / f"{index:02d}__{manifest_row['dtype']}_k{manifest_row['kernel_suffix']}.csv",
            MANIFEST_FIELDS, [manifest_row],
        )
    if len(variants) != len(contract):
        raise RuntimeError("discriminative shapes must exercise distinct compiled variants")
    print(
        f"DISCRIMINATIVE_SET_GENERATED shapes={len(contract)} "
        f"improved_packets={len(manifest_rows)} reused_official_references={len(reference_rows)} "
        f"variants={len(variants)}"
    )


if __name__ == "__main__":
    main()
