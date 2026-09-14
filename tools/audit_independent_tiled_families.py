#!/usr/bin/env python3
"""Exhaustive host audit for the fixed independent-kernel validation domain."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "matmul_rule_selector"))

from independent_tiled_selector import AIC, CUBE_FIELDS, ceil_div, generate  # noqa: E402
from independent_validation_cases import INSTALLED_SUFFIXES, build_cases  # noqa: E402


DIRECT = {
    "MICRO_DIRECT", "BALANCED_MN",
    "RESIDENT_B_M_STRIPE", "RESIDENT_A_N_STRIPE",
}


def exact_integer_cover(intervals: list[tuple[int, int]], size: int, label: str) -> None:
    cursor = 0
    for begin, end in sorted(intervals):
        if begin != cursor or end <= begin:
            raise RuntimeError(f"{label}: invalid cover at [{begin},{end}), cursor={cursor}")
        cursor = end
    if cursor != size:
        raise RuntimeError(f"{label}: cover ended at {cursor}, expected {size}")


def map_tail_owner(block: int, cores: int, tail: int) -> tuple[int, int, int]:
    small = cores // tail
    extra = cores % tail
    large = small + 1
    large_owners = extra * large
    if block < large_owners:
        return block // large, block % large, large
    local = block - large_owners
    return extra + local // small, local % small, small


def audit_ownership(workload_id: str, family: str, cube: dict) -> dict:
    cores = cube["usedCoreNum"]
    m_tiles = ceil_div(cube["M"], cube["singleCoreM"])
    n_tiles = ceil_div(cube["N"], cube["singleCoreN"])
    output_tiles = m_tiles * n_tiles
    if family in DIRECT:
        intervals = [
            (output_tiles * block // cores, output_tiles * (block + 1) // cores)
            for block in range(cores)
        ]
        exact_integer_cover(intervals, output_tiles, workload_id + ":MN")
        loads = [end - begin for begin, end in intervals]
        if min(loads) <= 0 or max(loads) - min(loads) > 1:
            raise RuntimeError(f"{workload_id}: direct owner imbalance {loads}")
        return {"direct_tiles": output_tiles, "maximum_owner_tasks": max(loads)}

    k_tiles = ceil_div(cube["Ka"], cube["baseK"])
    if family == "SEEDED_SPLIT_K":
        intervals = [
            (k_tiles * block // cores, k_tiles * (block + 1) // cores)
            for block in range(cores)
        ]
        exact_integer_cover(intervals, k_tiles, workload_id + ":K")
        if sum(begin == 0 for begin, _ in intervals) != 1:
            raise RuntimeError(f"{workload_id}: direct seed is not unique")
        return {"k_tiles": k_tiles, "k_owners": cores, "direct_seeds": 1}

    if family != "SEEDED_TAIL_WAVE":
        raise RuntimeError(f"{workload_id}: unhandled mode {family}")
    tail = output_tiles % cores
    if not 1 <= tail < cores:
        raise RuntimeError(f"{workload_id}: invalid tail size {tail}")
    groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for block in range(cores):
        tile, rank, group = map_tail_owner(block, cores, tail)
        begin = k_tiles * rank // group
        end = k_tiles * (rank + 1) // group
        groups[tile].append((rank, begin, end))
    if set(groups) != set(range(tail)):
        raise RuntimeError(f"{workload_id}: tail tile ownership is not exhaustive")
    for tile, owners in groups.items():
        if sorted(rank for rank, _, _ in owners) != list(range(len(owners))):
            raise RuntimeError(f"{workload_id}: tail tile {tile} rank gap")
        exact_integer_cover([(begin, end) for _, begin, end in owners], k_tiles,
                            f"{workload_id}:tail{tile}:K")
        if sum(rank == 0 for rank, _, _ in owners) != 1:
            raise RuntimeError(f"{workload_id}: tail tile {tile} lacks unique direct seed")
    return {
        "full_wave_tiles": output_tiles - tail,
        "tail_tiles": tail,
        "tail_k_owners": cores,
        "direct_seeds": tail,
    }


def audit_source_separation() -> None:
    selector = (ROOT / "matmul_rule_selector/independent_tiled_selector.py").read_text(
        encoding="utf-8"
    )
    kernel = (ROOT / "novel_matmul/independent_tiled_kernel.h").read_text(encoding="utf-8")
    selector_forbidden = (
        "official_source_select", "improved_selector", "formula_rules",
        "GetTilingFromRepo", "candidate_cost", "xgboost", "latency_ms",
        "matmul_reconstruction", "source_profile",
    )
    kernel_forbidden = (
        "mat_mul_v3_kernel.h", "mat_mul_v3_common.h", "MatmulBaseKernel",
        "KernelMatmul", "SplitKMatmul", "AtomicClean",
    )
    found = [token for token in selector_forbidden if token in selector]
    found += [token for token in kernel_forbidden if token in kernel]
    if found:
        raise RuntimeError(f"production source separation failed: {found}")
    if "MatmulImpl" not in kernel:
        raise RuntimeError("independent kernel does not use the public Cube matmul primitive")


def main() -> None:
    audit_source_separation()
    cases = build_cases()
    suffix_counts = Counter()
    family_counts = Counter()
    dtype_counts = Counter()
    layout_counts = Counter()
    packet_hashes = set()
    mode_checks = Counter()
    scale_counts = Counter()
    suffix_dimensions = defaultdict(lambda: {"m": set(), "n": set(), "k": set()})
    for case in cases:
        result = generate(
            case["m"], case["k"], case["n"], case["dtype"],
            case["trans_a"], case["trans_b"],
        )
        if any(result["runtime_dependencies"].values()):
            raise RuntimeError(f"{case['workload_id']}: forbidden runtime dependency")
        if result["packet_bytes"] != 280 or len(result["cube"]) != len(CUBE_FIELDS):
            raise RuntimeError(f"{case['workload_id']}: packet schema mismatch")
        if result["packet_sha256"] in packet_hashes:
            raise RuntimeError(f"{case['workload_id']}: duplicate shape packet")
        packet_hashes.add(result["packet_sha256"])
        ownership = audit_ownership(
            case["workload_id"], result["formula_family"], result["cube"]
        )
        if not ownership:
            raise RuntimeError(f"{case['workload_id']}: empty ownership proof")
        family_counts[result["formula_family"]] += 1
        dtype_counts[case["dtype"]] += 1
        layout_counts[(case["trans_a"], case["trans_b"])] += 1
        mode_checks[result["kernel_mode"]] += 1
        work = case["m"] * case["n"] * case["k"]
        scale_counts[
            "SMALL" if work < 10_000_000 else
            "MEDIUM" if work < 10_000_000_000 else "LARGE"
        ] += 1
        if case["official_suffix_audit"] != "":
            suffix = int(case["official_suffix_audit"])
            suffix_counts[suffix] += 1
            for axis in ("m", "n", "k"):
                suffix_dimensions[suffix][axis].add(case[axis])

    if suffix_counts != Counter({suffix: 10 for suffix in INSTALLED_SUFFIXES}):
        raise RuntimeError(f"installed suffix coverage mismatch: {suffix_counts}")
    if len(family_counts) != 6 or len(mode_checks) != 6:
        raise RuntimeError(f"independent mode coverage mismatch: {family_counts}")
    if set(dtype_counts) != {"fp16", "bf16", "fp32"} or len(layout_counts) != 4:
        raise RuntimeError("dtype or transpose-layout coverage incomplete")
    if set(scale_counts) != {"SMALL", "MEDIUM", "LARGE"}:
        raise RuntimeError(f"shape-scale coverage incomplete: {scale_counts}")
    for suffix, dimensions in suffix_dimensions.items():
        if any(len(values) < 5 for values in dimensions.values()):
            raise RuntimeError(f"suffix {suffix} does not vary all M/N/K axes: {dimensions}")
    print("INDEPENDENT_FAMILY_AUDIT_BEGIN")
    print(
        "SOURCE_SEPARATION passed installed_selector=0 official_seed=0 "
        "cost_model=0 candidate_search=0 history=0 repository_lookup=0"
    )
    print(
        f"VALIDATION_DOMAIN passed shapes={len(cases)} distinct_packets={len(packet_hashes)} "
        f"installed_suffixes={len(suffix_counts)} independent_modes={len(family_counts)}"
    )
    print("INSTALLED_SUFFIX_COVERAGE " + " ".join(
        f"suffix_{suffix}={suffix_counts[suffix]}" for suffix in INSTALLED_SUFFIXES
    ))
    print("INDEPENDENT_MODE_COVERAGE " + " ".join(
        f"{family}={count}" for family, count in sorted(family_counts.items())
    ))
    print("DTYPE_COVERAGE " + " ".join(
        f"{dtype}={count}" for dtype, count in sorted(dtype_counts.items())
    ))
    print("LAYOUT_COVERAGE " + " ".join(
        f"ta{int(ta)}tb{int(tb)}={count}" for (ta, tb), count in sorted(layout_counts.items())
    ))
    print("SCALE_COVERAGE " + " ".join(
        f"{scale}={scale_counts[scale]}" for scale in ("SMALL", "MEDIUM", "LARGE")
    ))
    print("SUFFIX_AXIS_DIVERSITY passed minimum_distinct_values_per_M_N_K=5")
    print("OWNERSHIP_PROOF passed direct_exact_once=1 split_k_exact_cover=1 unique_direct_seed=1")
    print("INDEPENDENT_FAMILY_AUDIT_END")


if __name__ == "__main__":
    main()
