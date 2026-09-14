#!/usr/bin/env python3
"""Build and audit the 240-shape complete-formula NPU campaign."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR_ROOT))

from complete_formula_selector import (  # noqa: E402
    CUBE_FIELDS,
    INSTALLED_SUFFIXES,
    PACKET_BYTES,
    generate,
)
from novel_validation_cases import CASES as NOVEL_CASES  # noqa: E402
from tiling_selector import generate as generate_production  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
TARGET_DTYPE = {
    0: "fp16", 1: "fp16", 20: "fp16", 21: "fp16",
    30: "bf16", 31: "bf16", 101: "fp32", 200: "fp32",
    201: "fp16", 10200: "fp32", 10201: "fp16", 20201: "fp32",
}
TARGET_COUNT = {
    suffix: 17 if index < 8 else 16
    for index, suffix in enumerate(INSTALLED_SUFFIXES)
}
EXPECTED_INSTALLED_SHAPES = sum(TARGET_COUNT.values())
NOVEL_PER_FAMILY = 20
EXPECTED_SHAPES = EXPECTED_INSTALLED_SHAPES + 2 * NOVEL_PER_FAMILY
MAX_FOOTPRINT = 300 * 1024 * 1024
RANDOM_SEED = 20_260_914

SMALL_VALUES = (
    8, 16, 17, 23, 31, 32, 33, 47, 48, 63, 64, 65, 80, 95, 96,
    97, 112, 127, 128, 129, 144, 160, 176, 191, 192, 193, 224,
    255, 256, 257, 288, 320, 384, 448, 512,
)
ALL_VALUES = SMALL_VALUES + (
    640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072,
    3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288, 14336,
    16384, 20480, 24576, 32768, 49152, 65536,
)


def _sample_request(target: int, rng: random.Random) -> tuple[int, int, int, str, bool, bool]:
    """Generate validation inputs; this never generates or ranks tilings."""
    dtype = TARGET_DTYPE[target]
    if target == 200:
        return (
            rng.randrange(72_400, 150_001, 16) + rng.randrange(0, 15),
            rng.randint(2, 7),
            rng.randint(17, 1024), dtype, False, rng.choice((False, True)),
        )
    if target == 101:
        return (
            rng.randint(2, 16), rng.randrange(128, 4097, 128),
            rng.randint(17, 512), dtype, False, True,
        )
    if target in (10200, 20201):
        k = rng.choice(
            (16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
            if target == 20201 else
            (17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47)
        )
        valid_n = tuple(value for value in range(17, 256) if value % 64 and 64 % value)
        return (
            rng.randrange(10_240, 60_001), k, rng.choice(valid_n),
            dtype, False, rng.choice((False, True)),
        )
    if target == 10201:
        valid_n = tuple(value for value in range(17, 256) if value % 128 and 128 % value)
        return (
            rng.randrange(10_240, 60_001),
            rng.choice((16, 32, 48, 64, 80, 96, 112, 127, 128, 144, 160, 192, 224, 256)),
            rng.choice(valid_n), dtype, False, rng.choice((False, True)),
        )
    if target in (30, 31):
        return (
            rng.randint(2, 160),
            128 * rng.randint(32, 468),
            rng.randint(2, 160), dtype,
            rng.choice((False, True)), rng.choice((False, True)),
        )
    if target in (20, 21):
        return (
            rng.choice(ALL_VALUES[20:]),
            128 * rng.randint(6, 256),
            rng.choice(ALL_VALUES), dtype,
            rng.choice((False, True)), rng.choice((False, True)),
        )
    if target == 201:
        return (
            16 * rng.randint(64, 3072),
            rng.choice((16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 384, 512, 768, 1024)),
            rng.choice((32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 320, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288)),
            dtype, rng.choice((False, True)), rng.choice((False, True)),
        )
    return (
        rng.choice(ALL_VALUES), rng.choice(ALL_VALUES), rng.choice(ALL_VALUES),
        dtype, rng.choice((False, True)), rng.choice((False, True)),
    )


def _footprint(result: dict) -> int:
    request = result["request"]
    width = 4 if request["dtype"] == "fp32" else 2
    return (
        (request["M"] * request["K"] + request["K"] * request["N"]
         + 2 * request["M"] * request["N"]) * width
        + result["workspace_bytes"]
    )


def build_cases() -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    cases = []
    global_seen = set()
    for target in INSTALLED_SUFFIXES:
        accepted = []
        axis_seen = {"m": set(), "n": set(), "k": set()}
        axis_minimum = {
            "m": 6 if target == 200 else 8,
            "n": 6 if target == 200 else 8,
            "k": 6 if target == 200 else 8,
        }
        attempts = 0
        while len(accepted) < TARGET_COUNT[target] and attempts < 100_000:
            attempts += 1
            m, k, n, dtype, trans_a, trans_b = _sample_request(target, rng)
            identity = (m, k, n, dtype, trans_a, trans_b)
            if identity in global_seen:
                continue
            try:
                result = generate(m, k, n, dtype, trans_a, trans_b)
                production = generate_production(m, k, n, dtype, trans_a, trans_b)
            except (ValueError, OverflowError):
                continue
            if (
                result["kernel_suffix"] != target
                or production["kernel_suffix"] != target
                or production["packet_sha256"] != result["packet_sha256"]
                or _footprint(result) > MAX_FOOTPRINT
            ):
                continue
            if target in (30, 31):
                fields = result["tiling_fields"]
                k_tasks = (
                    k + fields["singleCoreK"] - 1
                ) // fields["singleCoreK"]
                if fields["usedCoreNum"] >= min(20, k_tasks):
                    # This campaign measures an active formula rule, not an
                    # unchanged deterministic packet.  Keep shapes for which
                    # the same-critical-path equation removes at least one
                    # reduction owner.
                    continue
            values = {"m": m, "n": n, "k": k}
            if any(
                len(axis_seen[axis]) < axis_minimum[axis]
                and values[axis] in axis_seen[axis]
                for axis in ("m", "n", "k")
            ):
                continue
            global_seen.add(identity)
            accepted.append({
                "workload_id": f"formula_s{target}_{len(accepted):02d}",
                "m": m, "n": n, "k": k, "dtype": dtype,
                "trans_a": trans_a, "trans_b": trans_b,
                "result": result,
            })
            for axis, value in values.items():
                axis_seen[axis].add(value)
        if len(accepted) != TARGET_COUNT[target]:
            raise RuntimeError(
                f"only constructed {len(accepted)}/{TARGET_COUNT[target]} "
                f"validation inputs for suffix={target}"
            )
        cases.extend(accepted)
    return cases


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fnv1a64(blob: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in blob:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def _source_independence_audit() -> None:
    for relative in (
        "formula_rules.py", "complete_formula_selector.py",
        "novel_family_selector.py", "tiling_selector.py",
    ):
        text = (SELECTOR_ROOT / relative).read_text(encoding="utf-8")
        forbidden_imports = (
            "import npu_cost_model", "from npu_cost_model",
            "import improved_selector", "from improved_selector",
            "import matmul_reconstruction", "from matmul_reconstruction",
            "GetTilingFromRepo(", "source_select(", "candidate_cost(",
        )
        hits = [token for token in forbidden_imports if token in text]
        if hits:
            raise RuntimeError(f"{relative}: forbidden runtime dependency {hits}")


def _audit_cases(cases: list[dict]) -> dict:
    if len(cases) != EXPECTED_INSTALLED_SHAPES:
        raise RuntimeError(f"validation shape count changed: {len(cases)}")
    suffix_counts = Counter()
    packet_hashes = set()
    axis_values = defaultdict(lambda: {"m": set(), "n": set(), "k": set()})
    field_vectors = defaultdict(set)
    families = Counter()
    for case in cases:
        result = case["result"]
        suffix = int(result["kernel_suffix"])
        suffix_counts[suffix] += 1
        families[result["formula_family"]] += 1
        packet = bytes.fromhex(result["packet_hex"])
        if len(packet) != PACKET_BYTES:
            raise RuntimeError(f"{case['workload_id']}: packet size mismatch")
        if hashlib.sha256(packet).hexdigest() != result["packet_sha256"]:
            raise RuntimeError(f"{case['workload_id']}: packet digest mismatch")
        if result["packet_sha256"] in packet_hashes:
            raise RuntimeError(f"{case['workload_id']}: duplicate packet")
        packet_hashes.add(result["packet_sha256"])
        words = struct.unpack("<68I", packet)
        fields = result["tiling_fields"]
        expected = [result["abi_fields"]["cube"][name] for name in CUBE_FIELDS]
        if list(words[:50]) != expected:
            raise RuntimeError(f"{case['workload_id']}: Cube ABI mismatch")
        legality = result["derivation"].get("legality", {})
        if legality.get("visible_input_and_branch_checks") != "PASS":
            raise RuntimeError(f"{case['workload_id']}: legality attestation missing")
        contract = result["selection_contract"]
        if not (
            contract["complete_tilings_constructed"] == 1
            and not contract["candidate_enumeration"]
            and not contract["candidate_ranking"]
            and not contract["latency_score"]
            and not contract["cost_model"]
            and not contract["official_selector"]
            and not contract["official_packet_seed"]
            and not contract["history_lookup"]
            and not contract["repository_lookup"]
        ):
            raise RuntimeError(f"{case['workload_id']}: selector independence failed")
        if not result["schedule_facts"]["ownership"]["no_zero_owner_core"]:
            raise RuntimeError(f"{case['workload_id']}: launched zero-owner core")
        for axis in ("m", "n", "k"):
            axis_values[suffix][axis].add(case[axis])
        field_vectors[suffix].add(tuple(fields[name] for name in (
            "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
            "baseM", "baseN", "baseK", "depthA1", "depthB1",
            "stepM", "stepN", "stepKa", "stepKb", "dbL0C", "iterateOrder",
        )))
    if suffix_counts != Counter(TARGET_COUNT):
        raise RuntimeError(f"twelve-suffix coverage mismatch: {suffix_counts}")
    for suffix in INSTALLED_SUFFIXES:
        minimum_axis_values = 6 if suffix == 200 else 8
        if any(
            len(axis_values[suffix][axis]) < minimum_axis_values
            for axis in ("m", "n", "k")
        ):
            raise RuntimeError(f"suffix={suffix}: M/N/K validation diversity is insufficient")
        if len(field_vectors[suffix]) < 8:
            raise RuntimeError(f"suffix={suffix}: tiling field diversity is insufficient")
    return {
        "shapes": len(cases),
        "packets": len(packet_hashes),
        "suffix_counts": dict(suffix_counts),
        "family_counts": dict(families),
        "variants": len({(case["dtype"], case["result"]["kernel_suffix"]) for case in cases}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    _source_independence_audit()
    cases = build_cases()
    audit = _audit_cases(cases)

    packet_dir = args.root / "packets"
    variant_dir = args.root / "variants"
    packet_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    selections = []
    for case in cases:
        result = case["result"]
        production = generate_production(
            case["m"], case["k"], case["n"], case["dtype"],
            case["trans_a"], case["trans_b"],
        )
        if production["packet_sha256"] != result["packet_sha256"]:
            raise RuntimeError(f"{case['workload_id']}: production selector did not emit the audited installed packet")
        packet = bytes.fromhex(result["packet_hex"])
        path = packet_dir / f"{case['workload_id']}.bin"
        path.write_bytes(packet)
        schedule_digest = hashlib.sha256(json.dumps(
            {
                "request": result["request"],
                "family": result["formula_family"],
                "suffix": result["kernel_suffix"],
                "fields": result["tiling_fields"],
                "l2": result["l2_fields"],
            }, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        manifest_rows.append({
            "workload_id": case["workload_id"], "rank": "0",
            "candidate_role": "independent_formula_tiling",
            "m": str(case["m"]), "n": str(case["n"]), "k": str(case["k"]),
            "dtype": case["dtype"],
            "trans_a": str(int(case["trans_a"])),
            "trans_b": str(int(case["trans_b"])),
            "used_core_num": str(result["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(result["workspace_bytes"]),
            "tiling_path": str(path.resolve()),
            "tiling_sha256": result["packet_sha256"],
            "tiling_fnv1a64": result["packet_fnv1a64"],
            "model_schedule_sha256": schedule_digest,
            "is_reserve": "0",
            "l2_cache_flag": str(result["abi_fields"]["l2_cache_flag"]),
            "nd2nz_a": str(int(result["conversion_a"])),
            "nd2nz_b": str(int(result["conversion_b"])),
            "required_successful_tilings": "1",
        })
        selections.append({
            "workload_id": case["workload_id"],
            "formula_family": result["formula_family"],
            "kernel_suffix": result["kernel_suffix"],
            "key_derivation": result["key_derivation"],
            "tiling_fields": result["tiling_fields"],
            "l2_fields": result["l2_fields"],
            "schedule_facts": result["schedule_facts"],
            "workspace_terms": result["workspace_terms"],
            "packet_sha256": result["packet_sha256"],
            "selection_contract": result["selection_contract"],
        })

    novel_counts = Counter()
    novel_cases = []
    for family in (
        "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K",
        "DIRECT_INIT_TAIL_WAVE_SPLIT_K",
    ):
        selected = [row for row in NOVEL_CASES if row["required_family"] == family]
        novel_cases.extend(selected[:NOVEL_PER_FAMILY])
    for case in novel_cases:
        result = generate_production(
            case["m"], case["k"], case["n"], case["dtype"],
            case["trans_a"], case["trans_b"],
        )
        if result["formula_family"] != case["required_family"]:
            raise RuntimeError(f"{case['workload_id']}: production selector chose {result['formula_family']}")
        if any(result["runtime_dependencies"].values()) or not all(result["resource_checks"].values()):
            raise RuntimeError(f"{case['workload_id']}: novel tiling independence/resource audit failed")
        packet = bytes.fromhex(result["packet_hex"])
        if len(packet) != 280 or hashlib.sha256(packet).hexdigest() != result["packet_sha256"]:
            raise RuntimeError(f"{case['workload_id']}: novel packet ABI audit failed")
        packet_path = packet_dir / f"{case['workload_id']}.bin"
        packet_path.write_bytes(packet)
        cube = result["cube"]
        schedule_digest = hashlib.sha256(json.dumps({
            "request": case,
            "family": result["formula_family"],
            "suffix": result["kernel_suffix"],
            "cube": cube,
            "derivation": result["derivation"],
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest_rows.append({
            "workload_id": case["workload_id"], "rank": "0",
            "candidate_role": "independent_experimental_family",
            "m": str(case["m"]), "n": str(case["n"]), "k": str(case["k"]),
            "dtype": case["dtype"],
            "trans_a": str(int(case["trans_a"])),
            "trans_b": str(int(case["trans_b"])),
            "used_core_num": str(result["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(result["workspace_bytes"]),
            "tiling_path": str(packet_path.resolve()),
            "tiling_sha256": result["packet_sha256"],
            "tiling_fnv1a64": _fnv1a64(packet),
            "model_schedule_sha256": schedule_digest,
            "is_reserve": "0", "l2_cache_flag": "0",
            "nd2nz_a": "0", "nd2nz_b": "0",
            "required_successful_tilings": "1",
        })
        selections.append({
            "workload_id": case["workload_id"],
            "formula_family": result["formula_family"],
            "kernel_suffix": result["kernel_suffix"],
            "tiling_fields": {
                name: cube[name] for name in (
                    "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
                    "baseM", "baseN", "baseK", "depthA1", "depthB1",
                    "stepM", "stepN", "stepKa", "stepKb", "dbL0C", "iterateOrder",
                )
            },
            "l2_fields": {
                "mTile": 1, "nTile": 1, "mTileBlock": 1,
                "nTileBlock": 1, "calOrder": 0,
            },
            "schedule_facts": result["derivation"],
            "workspace_terms": {"system": result["workspace_bytes"]},
            "packet_sha256": result["packet_sha256"],
            "selection_contract": result["runtime_dependencies"],
            "origin": result["origin"],
        })
        novel_counts[result["kernel_suffix"]] += 1

    if len(manifest_rows) != EXPECTED_SHAPES or novel_counts != Counter({90001: 20, 90002: 20}):
        raise RuntimeError(
            f"combined campaign coverage mismatch rows={len(manifest_rows)} novel={novel_counts}"
        )

    _write_csv(args.root / "manifest.csv", manifest_rows)
    grouped = defaultdict(list)
    for row in manifest_rows:
        grouped[(row["dtype"], int(row["kernel_suffix"]))].append(row)
    for (dtype, suffix), rows in sorted(grouped.items()):
        _write_csv(variant_dir / f"{dtype}_k{suffix}.csv", rows)
    with (args.root / "selection.jsonl").open("w", encoding="utf-8") as stream:
        for selection in selections:
            stream.write(json.dumps(selection, sort_keys=True, separators=(",", ":")) + "\n")
    audit["novel_suffix_counts"] = dict(novel_counts)
    audit["total_shapes"] = len(manifest_rows)
    audit["total_variants"] = len(grouped)
    (args.root / "audit.json").write_text(
        json.dumps(audit, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(
        "FORMULA_MATRIX_AUDIT passed "
        f"shapes={audit['total_shapes']} installed_packets={audit['packets']} "
        f"suffixes={len(audit['suffix_counts']) + len(novel_counts)} variants={audit['total_variants']} "
        "complete_tilings_per_shape=1 field_diversity_min=8 "
        "official_seed=0 cost_model=0 candidate_search=0 history=0"
    )


if __name__ == "__main__":
    main()
