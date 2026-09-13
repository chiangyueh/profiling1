#!/usr/bin/env python3
"""Generate proof-carrying AL1 idle-core-elimination validation packets."""
from __future__ import annotations

import argparse
from collections import defaultdict
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

from improved_selector import generate  # noqa: E402


FIELDS = (
    "workload_id", "rank", "candidate_role", "m", "n", "k", "dtype",
    "trans_a", "trans_b", "used_core_num", "kernel_suffix",
    "workspace_bytes", "tiling_path", "tiling_sha256", "tiling_fnv1a64",
    "model_schedule_sha256", "is_reserve", "l2_cache_flag", "nd2nz_a",
    "nd2nz_b", "required_successful_tilings",
)
EXPECTED_VALIDATION_SHAPES = 100
EXPECTED_VARIANTS = {("fp32", 101)}
EXPECTED_SOURCE_FAMILIES = {
    "BASE", "AL1_FULL_LOAD", "BL1_FULL_LOAD", "BL1_FULL_LOAD_FIXPIPE",
    "SINGLE_CORE_SPLIT_K", "DETERMINISTIC_SPLIT_K", "INCREMENTAL_PATTERN",
}
EXPECTED_SOURCE_SUFFIXES = {
    0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201,
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


def audit_no_candidate_engine() -> None:
    forbidden = (
        "from candidate_engine", "import candidate_engine",
        "generate_and_select(", "derive_ideal_region(", "simulate(",
    )
    for path in (
        SELECTOR_ROOT / "formula_rules.py",
        SELECTOR_ROOT / "improved_selector.py",
    ):
        source = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in source]
        if hits:
            raise RuntimeError(f"{path.name}: forbidden candidate path {hits}")


def run_selector(row: dict) -> dict:
    result = generate(
        int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
        bool(row["trans_a"]), bool(row["trans_b"]),
    )
    theory = result["theory"]
    if not (
        theory["selection_mode"] == "PROOF_CARRYING_CLOSED_FORM"
        and theory["complete_tilings_constructed"] == 1
        and theory["candidate_count"] == 1
        and theory["candidate_enumeration"] is False
        and theory["pareto_pruning"] is False
        and theory["latency_or_cost_score"] is False
        and result["runtime_dependencies"]["cost_model"] is False
        and result["runtime_dependencies"]["candidate_enumeration"] is False
        and result["runtime_dependencies"]["history"] is False
        and result["runtime_dependencies"]["runtime_kb"] is False
        and result["runtime_dependencies"]["tiling_bank"] is False
    ):
        raise RuntimeError(f"{row['workload_id']}: unique selector contract failed")
    return result


def generate_validation_cases(spec: dict) -> list[tuple[dict, dict]]:
    """Create varied shapes inside the non-vacuous AL1 dominance domain."""
    required = {
        "schema", "seed", "count", "m_min", "m_max", "n_task_min",
        "n_task_max", "k_multiple_128_min", "k_multiple_128_max",
        "dtypes", "trans_a", "trans_b", "max_input_bytes", "max_attempts",
    }
    if set(spec) != required:
        raise RuntimeError("random validation specification fields changed")
    count = int(spec["count"])
    if not (1 <= count <= 100) or count != EXPECTED_VALIDATION_SHAPES:
        raise RuntimeError("NPU validation must contain 1..100 shapes")
    if spec["schema"] != "al1_idle_core_elimination_stratified_random_v1":
        raise RuntimeError("unexpected random validation schema")
    if spec["trans_a"] != [False] or spec["trans_b"] != [True]:
        raise RuntimeError("certified AL1 route requires transA=false/transB=true")
    dtypes = tuple(str(value) for value in spec["dtypes"])
    if dtypes != ("fp32",):
        raise RuntimeError("installed suffix 101 AL1 kernel is FP32-only")
    task_min = int(spec["n_task_min"])
    task_max = int(spec["n_task_max"])
    if not (5 <= task_min <= task_max <= 10):
        raise RuntimeError("AL1 certificate requires five to ten N tasks")
    strata = list(range(task_min, task_max + 1))
    rng = random.Random(int(spec["seed"]))
    rng.shuffle(strata)
    accepted: list[tuple[dict, dict]] = []
    seen_shapes: set[tuple] = set()
    attempts = 0
    while len(accepted) < count and attempts < int(spec["max_attempts"]):
        n_tasks = strata[len(accepted) % len(strata)]
        attempts += 1
        m = rng.randint(int(spec["m_min"]), int(spec["m_max"]))
        # Include aligned and unaligned N tails while preserving n_tasks.
        n = (n_tasks - 1) * 16 + rng.randint(1, 16)
        k = 128 * rng.randint(
            int(spec["k_multiple_128_min"]),
            int(spec["k_multiple_128_max"]),
        )
        identity = (m, n, k, "fp32", False, True)
        if identity in seen_shapes:
            continue
        input_bytes = ((m * k) + (k * n)) * 4
        if input_bytes > int(spec["max_input_bytes"]):
            continue
        row = {
            "workload_id": f"al1_certified_{len(accepted):03d}",
            "m": m, "n": n, "k": k, "dtype": "fp32",
            "trans_a": False, "trans_b": True,
            "selection_axis": (
                "al1_eliminate_14_15_idle_aic" if n_tasks <= 6 else
                "al1_eliminate_12_13_idle_aic" if n_tasks <= 8 else
                "al1_eliminate_10_11_idle_aic"
            ),
        }
        result = run_selector(row)
        if not (
            result["status"] == "MODIFIED_TILING"
            and result["npu_eligible"]
            and result["changed_rules"] == [
                "AL1_IDLE_CORE_FULL_A_COPY_ELIMINATION"
            ]
            and result["kernel_suffix"] == 101
            and result["theory"]["source_family"] == "AL1_FULL_LOAD"
        ):
            continue
        seen_shapes.add(identity)
        accepted.append((row, result))
    if len(accepted) != count:
        raise RuntimeError(
            f"only generated {len(accepted)} certified AL1 shapes "
            f"after {attempts} attempts"
        )

    rows = [row for row, _ in accepted]
    variation = {
        "m": len({row["m"] for row in rows}),
        "n": len({row["n"] for row in rows}),
        "k": len({row["k"] for row in rows}),
    }
    if not (
        variation["m"] >= 16
        and variation["n"] >= 50
        and variation["k"] >= 16
        and {row["dtype"] for row in rows} == {"fp32"}
        and {row["trans_b"] for row in rows} == {True}
        and {row["selection_axis"] for row in rows} == {
            "al1_eliminate_14_15_idle_aic",
            "al1_eliminate_12_13_idle_aic",
            "al1_eliminate_10_11_idle_aic",
        }
    ):
        raise RuntimeError(
            f"certified AL1 set lacks independent axis variation: {variation}"
        )
    return accepted


def compact_theory(workload_id: str, result: dict) -> dict:
    theory = result["theory"]
    return {
        "workload_id": workload_id,
        "status": result["status"],
        "source_family": theory["source_family"],
        "source_suffix": theory["source_suffix"],
        "formula_family": theory["formula_family"],
        "candidate_count": theory["candidate_count"],
        "logical_fma_count": theory["logical_fma_count"],
        "formula_fields": theory["formula_fields"],
        "formula_l2": theory["formula_l2"],
        "resource_bytes": theory["resource_bytes"],
        "selection_contract": theory["selection_contract"],
        "scheduled_work": theory.get("scheduled_work"),
        "improvement_equation": theory.get("improvement_equation"),
        "strict_dominance_certificate": theory.get(
            "strict_dominance_certificate"
        ),
        "skip_reason": result.get("skip_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--theory-audit", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    args = parser.parse_args()

    audit_no_candidate_engine()
    validation_spec = json.loads(
        (SELECTOR_ROOT / "unique_formula_validation_contract.json").read_text(
            encoding="utf-8"
        )
    )
    audit_contract = json.loads(
        (SELECTOR_ROOT / "source_family_audit_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if (len(audit_contract) != 13 or
            len({row["workload_id"] for row in audit_contract}) !=
            len(audit_contract)):
        raise RuntimeError("source family audit contract was reduced or duplicated")

    for directory in (args.output_dir, args.variant_dir, args.sequence_dir):
        clear_files(directory)

    audit_rows = []
    source_families, source_suffixes = set(), set()
    for row in audit_contract:
        result = run_selector(row)
        theory = result["theory"]
        if (theory["source_family"] != row["expected_family"] or
                int(theory["source_suffix"]) != int(row["expected_suffix"])):
            raise RuntimeError(
                f"{row['workload_id']}: source decision drift "
                f"family={theory['source_family']} suffix={theory['source_suffix']}"
            )
        selected = result["selected"]
        raw = bytes.fromhex(selected["tiling_data_hex"])
        if (len(raw) != 272 or selected["tiling_data_bytes"] != 272 or
                hashlib.sha256(raw).hexdigest() !=
                selected["tiling_data_sha256"]):
            raise RuntimeError(
                f"{row['workload_id']}: source path is not one complete packet"
            )
        source_families.add(theory["source_family"])
        source_suffixes.add(int(theory["source_suffix"]))
        audit_rows.append(compact_theory(row["workload_id"], result))
    if (source_families != EXPECTED_SOURCE_FAMILIES or
            source_suffixes != EXPECTED_SOURCE_SUFFIXES):
        raise RuntimeError("full source family/suffix audit coverage was reduced")

    validation = generate_validation_cases(validation_spec)
    manifest_rows = []
    selections = []
    for row, result in validation:
        workload_id = str(row["workload_id"])
        if not (result["status"] == "MODIFIED_TILING" and
                result["npu_eligible"] and not result["baseline_equivalent"]):
            raise RuntimeError(f"{workload_id}: no distinct improved packet")
        if result["changed_rules"] != [
                "AL1_IDLE_CORE_FULL_A_COPY_ELIMINATION"]:
            raise RuntimeError(f"{workload_id}: unexpected certified rule")
        packet = result["improved"]
        blob = bytes.fromhex(packet["tiling_data_hex"])
        if (len(blob) != 272 or result["kernel_suffix"] != 101 or
                hashlib.sha256(blob).hexdigest() != packet["tiling_data_sha256"]):
            raise RuntimeError(f"{workload_id}: invalid 272-byte AL1 packet")
        path = args.output_dir / f"{workload_id}.bin"
        path.write_bytes(blob)
        words = struct.unpack("<68I", blob[:272])
        manifest_rows.append({
            "workload_id": workload_id, "rank": "0",
            "candidate_role": "certified_instruction_deletion",
            "m": str(row["m"]), "n": str(row["n"]), "k": str(row["k"]),
            "dtype": str(row["dtype"]),
            "trans_a": str(int(row["trans_a"])),
            "trans_b": str(int(row["trans_b"])),
            "used_core_num": str(packet["block_dim"]),
            "kernel_suffix": str(result["kernel_suffix"]),
            "workspace_bytes": str(packet["workspace_bytes"]),
            "tiling_path": str(path.resolve()),
            "tiling_sha256": packet["tiling_data_sha256"],
            "tiling_fnv1a64": fnv1a64(blob),
            "model_schedule_sha256": hashlib.sha256(
                json.dumps(compact_theory(workload_id, result), sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "is_reserve": "0", "l2_cache_flag": str(words[62]),
            "nd2nz_a": str(words[58]), "nd2nz_b": str(words[59]),
            "required_successful_tilings": "1",
        })
        result.update({
            "workload_id": workload_id,
            "selection_axis": row["selection_axis"],
            "required_applicable_family": result["theory"]["source_family"],
            "case_role": "certified_instruction_deletion",
        })
        selections.append(result)
        audit_rows.append(compact_theory(workload_id, result))

    variants = {(row["dtype"], int(row["kernel_suffix"]))
                for row in manifest_rows}
    if variants != EXPECTED_VARIANTS:
        raise RuntimeError(f"unexpected validation variants: {sorted(variants)}")
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
        for result in selections:
            stream.write(json.dumps(result, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    with args.theory_audit.open("w", encoding="utf-8") as stream:
        for result in audit_rows:
            stream.write(json.dumps(result, sort_keys=True,
                                    separators=(",", ":")) + "\n")
    print(
        "CERTIFIED_AL1_MATRIX_GENERATED "
        f"audit_shapes={len(audit_contract)} source_families={len(source_families)} "
        f"source_suffixes={len(source_suffixes)} npu_shapes={len(manifest_rows)} "
        f"complete_tilings_per_shape=1 variants={len(grouped)} "
        f"random_seed={validation_spec['seed']}"
    )


if __name__ == "__main__":
    main()
