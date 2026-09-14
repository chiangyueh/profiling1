#!/usr/bin/env python3
"""Prove that frozen formula packets are real tiling changes, not core edits.

The production selector has already emitted and hashed every packet before this
program imports the reconstructed 8.1 reference.  Reference results are used
only for a one-way audit and are never returned to the selector.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
BASELINE_ROOT = SELECTOR_ROOT / "baseline_core"

CUBE_FIELDS = (
    "usedCoreNum", "M", "N", "Ka", "Kb", "singleCoreM", "singleCoreN",
    "singleCoreK", "baseM", "baseN", "baseK", "depthA1", "depthB1",
    "stepM", "stepN", "isBias", "transLength", "iterateOrder", "shareMode",
    "shareL1Size", "shareL0CSize", "shareUbSize", "batchM", "batchN",
    "singleBatchM", "singleBatchN", "stepKa", "stepKb", "depthAL1CacheUB",
    "depthBL1CacheUB", "dbL0A", "dbL0B", "dbL0C", "ALayoutInfoB",
    "ALayoutInfoS", "ALayoutInfoN", "ALayoutInfoG", "ALayoutInfoD",
    "BLayoutInfoB", "BLayoutInfoS", "BLayoutInfoN", "BLayoutInfoG",
    "BLayoutInfoD", "CLayoutInfoB", "CLayoutInfoS1", "CLayoutInfoN",
    "CLayoutInfoG", "CLayoutInfoS2", "BatchNum", "reserved",
)
PACKET_FIELDS = CUBE_FIELDS + (
    "mTileCntL2", "nTileCntL2", "mTileBlock", "nTileBlock", "calOrder",
    "padding220", "transA", "transB", "nd2nzA", "nd2nzB", "isHf32",
    "padding244", "l2CacheFlag", "padding252", "baseAN", "baseAD",
    "baseBN", "baseBD",
)
SCHEDULE_FIELDS = frozenset((
    "singleCoreM", "singleCoreN", "singleCoreK", "baseM", "baseN", "baseK",
    "depthA1", "depthB1", "stepM", "stepN", "iterateOrder", "stepKa",
    "stepKb", "dbL0A", "dbL0B", "dbL0C", "mTileCntL2", "nTileCntL2",
    "mTileBlock", "nTileBlock", "calOrder", "baseAN", "baseAD", "baseBN",
    "baseBD",
))
EXPECTED_SUFFIXES = (0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201)
HARDWARE = {
    "aicNum": 20, "aivNum": 40, "ubSize": 196352,
    "l1Size": 524032, "l2Size": 201326592,
    "l0CSize": 131072, "l0ASize": 65536, "l0BSize": 65536,
    "btSize": 1024, "supportL0c2out": True,
    "supportL12BtBf16": False, "cubeFreq": 0, "npuArch": 220,
    "socVersion": 220, "socVersionStr": "Ascend910B3",
    "cannVersion": "8.1.RC1",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _request(row: dict[str, str]) -> dict:
    dtype = row["dtype"]
    return {
        "M": int(row["m"]), "N": int(row["n"]), "K": int(row["k"]),
        "dtype": dtype, "output_dtype": dtype,
        "layoutA": "ND", "layoutB": "ND", "layoutC": "ND",
        "transA": bool(int(row["trans_a"])),
        "transB": bool(int(row["trans_b"])),
        "bias": False, "hf32": False, "forceGrpAccForFp32": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    all_manifest = _read_csv(args.manifest)
    all_selections = {
        row["workload_id"]: row
        for row in (
            json.loads(line)
            for line in args.selection.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    manifest = [
        row for row in all_manifest
        if int(row["kernel_suffix"]) in EXPECTED_SUFFIXES
    ]
    selections = {
        workload_id: row for workload_id, row in all_selections.items()
        if int(row["kernel_suffix"]) in EXPECTED_SUFFIXES
    }
    if len(manifest) != 200 or len(selections) != 200:
        raise RuntimeError("reference separation audit requires 200 frozen formula packets")

    # Freeze and re-hash all candidate bytes before the reference code exists
    # in this process.  Nothing below is allowed to rewrite these files.
    frozen: dict[str, tuple[bytes, str]] = {}
    for row in manifest:
        workload_id = row["workload_id"]
        blob = Path(row["tiling_path"]).read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if len(blob) != 272 or digest != row["tiling_sha256"]:
            raise RuntimeError(f"{workload_id}: formula packet was not frozen correctly")
        if selections[workload_id]["packet_sha256"] != digest:
            raise RuntimeError(f"{workload_id}: formula selection digest mismatch")
        frozen[workload_id] = (blob, digest)

    sys.path.insert(0, str(BASELINE_ROOT))
    from matmul_reconstruction.api import select as reference_select  # noqa: E402

    rows = []
    suffix_counts = Counter()
    reference_suffix_matches = Counter()
    union_by_suffix: dict[int, set[str]] = defaultdict(set)
    minimum_by_suffix: dict[int, int] = defaultdict(lambda: 10**9)
    for row in manifest:
        workload_id = row["workload_id"]
        candidate_blob, frozen_digest = frozen[workload_id]
        reference = reference_select(
            _request(row), HARDWARE, source_profile="installed_81", trace=False
        )
        reference_blob = bytes.fromhex(reference["raw_buffer_hex"])
        if frozen[workload_id] != (candidate_blob, frozen_digest):
            raise RuntimeError(f"{workload_id}: reference audit mutated formula state")
        candidate_words = struct.unpack("<68I", candidate_blob)
        reference_words = struct.unpack("<68I", reference_blob)
        changed = [
            PACKET_FIELDS[index]
            for index, (candidate, baseline) in enumerate(
                zip(candidate_words, reference_words)
            )
            if candidate != baseline
        ]
        schedule_changed = [name for name in changed if name in SCHEDULE_FIELDS]
        if not schedule_changed:
            raise RuntimeError(
                f"{workload_id}: candidate changes no tiling schedule field; "
                f"changed={changed}"
            )
        if candidate_blob == reference_blob:
            raise RuntimeError(f"{workload_id}: candidate is the reference packet")

        suffix = int(row["kernel_suffix"])
        reference_suffix = int(reference["tiling_key"] - 10**19)
        suffix_counts[suffix] += 1
        reference_suffix_matches[suffix] += int(reference_suffix == suffix)
        union_by_suffix[suffix].update(schedule_changed)
        minimum_by_suffix[suffix] = min(minimum_by_suffix[suffix], len(schedule_changed))
        rows.append({
            "workload_id": workload_id,
            "formula_suffix": suffix,
            "reference_suffix": reference_suffix,
            "reference_suffix_same": reference_suffix == suffix,
            "changed_packet_words": changed,
            "changed_schedule_fields": schedule_changed,
            "formula_packet_sha256": frozen_digest,
            "reference_packet_sha256": hashlib.sha256(reference_blob).hexdigest(),
            "audit_order": "FORMULA_FROZEN_BEFORE_REFERENCE_EXECUTION",
        })

    if tuple(sorted(suffix_counts)) != EXPECTED_SUFFIXES:
        raise RuntimeError(f"not all twelve installed suffixes were audited: {suffix_counts}")
    by_suffix = {
        str(suffix): {
            "shapes": suffix_counts[suffix],
            "minimum_changed_schedule_fields_per_shape": minimum_by_suffix[suffix],
            "changed_schedule_field_union": sorted(union_by_suffix[suffix]),
            "reference_suffix_same_shapes": reference_suffix_matches[suffix],
        }
        for suffix in EXPECTED_SUFFIXES
    }
    result = {
        "status": "PASS",
        "formula_packets_frozen_before_reference": True,
        "reference_output_used_by_selector": False,
        "shape_count": len(rows),
        "exact_reference_packets": 0,
        "core_only_changes": 0,
        "suffixes": by_suffix,
        "rows": rows,
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "FORMULA_REFERENCE_SEPARATION passed shapes=200 suffixes=12 "
        "exact_reference_packets=0 core_only_changes=0 "
        "reference_used_by_selector=0"
    )


if __name__ == "__main__":
    main()
