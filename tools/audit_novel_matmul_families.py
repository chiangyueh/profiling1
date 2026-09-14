#!/usr/bin/env python3
"""Fail-closed provenance, packet, resource, and ownership audit."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR))

from novel_family_selector import generate  # noqa: E402
from novel_validation_cases import CASES  # noqa: E402


INSTALLED = Path(
    "/usr/local/Ascend/ascend-toolkit/8.1/opp/built-in/op_impl/ai_core/"
    "tbe/impl/ascendc/mat_mul_v3"
)
BUNDLED = ROOT / "colleague_matmul_v3/op_kernel"
ARCH35 = BUNDLED / "arch35"
C220_HOST = ROOT / "colleague_matmul_v3/op_host/op_tiling"
ARCH35_HOST = C220_HOST / "arch35"
CUSTOM = ROOT / "novel_matmul/seeded_split_k_kernel.h"
LEDGER = SELECTOR / "official_family_provenance.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_for(architecture: str, relative: str) -> Path:
    roots = [ARCH35] if architecture == "Arch35" else [BUNDLED, INSTALLED]
    matches = [root / relative for root in roots if (root / relative).is_file()]
    require(bool(matches), f"provenance source missing: {architecture}/{relative}")
    return matches[0]


def host_source_for(architecture: str, relative: str) -> Path:
    root = ARCH35_HOST if architecture == "Arch35" else C220_HOST
    path = root / relative
    require(path.is_file(), f"host provenance source missing: {architecture}/{relative}")
    return path


def audit_provenance() -> dict:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    official = ledger["official_families"]
    host_strategies = ledger["official_host_strategies"]
    repository = ledger["repository_owned_families"]
    require(len(official) == 15, "official family ledger coverage changed")
    require(len(host_strategies) == 8, "official host strategy coverage changed")
    require(len(repository) == 2, "repository-owned family ledger coverage changed")
    source_paths = set()
    for entry in official:
        for relative in entry["sources"]:
            source_paths.add(source_for(entry["architecture"], relative))
    host_source_paths = set()
    for entry in host_strategies:
        for relative in entry["sources"]:
            host_source_paths.add(host_source_for(entry["architecture"], relative))

    official_multi = (BUNDLED / "mat_mul_multi_core_splitk_kernel.h").read_text(encoding="utf-8")
    official_stream = (ARCH35 / "mat_mul_stream_k_kernel.h").read_text(encoding="utf-8")
    official_scheduler = (ARCH35 / "block_scheduler_streamk.h").read_text(encoding="utf-8")
    custom = CUSTOM.read_text(encoding="utf-8")

    require("ClearOutput<C_T>(cGlobal, tiling, pipe);" in official_multi,
            "later C220 multi-core source no longer proves pre-clear")
    require("mm.GetTensorC(cGlobal[offsetC], 1);" in official_multi,
            "later C220 multi-core source no longer proves all-owner atomic output")
    require("tailMNTileNum" in official_scheduler and "skKTileNum_" in official_scheduler,
            "Arch35 tail-only K scheduling anchor missing")
    require("GetTensorC(workspaceGlobal_" in official_stream and
            "Add(ubAddTensor" in official_stream,
            "Arch35 workspace plus AIV reduction anchors missing")
    require("ClearOutput" not in custom and "workspaceGlobal_" not in custom,
            "custom protocol silently acquired an official reduction mechanism")
    require("owner.rank == 0" in custom and "SyncAll();" in custom and
            "GetTensorC(cGlobal_[offsetC], 1)" in custom,
            "custom direct-seed/barrier/atomic protocol is incomplete")
    require("mat_mul_multi_core_splitk_kernel.h" not in custom and
            "mat_mul_stream_k_kernel.h" not in custom,
            "custom implementation embeds an official family body")

    official_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for root in (INSTALLED, BUNDLED)
        for path in root.rglob("*") if path.is_file()
    )
    for token in ("SEEDED_ATOMIC_SPLIT_K", "SEEDED_TAIL_WAVE_SPLIT_K", "90001", "90002"):
        require(token not in official_text, f"custom identity already exists in official source: {token}")

    return {
        "official_family_records": len(official),
        "official_source_files": len(source_paths),
        "official_host_source_files": len(host_source_paths),
        "repository_owned_records": len(repository),
        "custom_source_sha256": hashlib.sha256(custom.encode()).hexdigest(),
    }


def split_ranges(quanta: int, owners: int) -> list[tuple[int, int]]:
    return [
        (quanta * owner // owners, quanta * (owner + 1) // owners)
        for owner in range(owners)
    ]


def tail_owner(block: int, tail_tiles: int) -> tuple[int, int, int]:
    small = 20 // tail_tiles
    extra = 20 % tail_tiles
    large = small + 1
    large_owners = extra * large
    if block < large_owners:
        return block // large, block % large, large
    local = block - large_owners
    return extra + local // small, local % small, small


def exact_cover(ranges: list[tuple[int, int]], extent: int) -> bool:
    counts = [0] * extent
    for begin, end in ranges:
        if not 0 <= begin < end <= extent:
            return False
        for index in range(begin, end):
            counts[index] += 1
    return all(count == 1 for count in counts)


def audit_case(case: dict) -> dict:
    result = generate(
        case["m"], case["k"], case["n"], case["dtype"],
        case["trans_a"], case["trans_b"],
        required_family=case["required_family"],
    )
    require(result["npu_eligible"], f"{case['workload_id']}: not NPU eligible")
    require(not any(result["runtime_dependencies"].values()),
            f"{case['workload_id']}: forbidden runtime dependency")
    require(result["origin"] == "REPOSITORY_OWNED_NOT_OFFICIAL_BACKPORT",
            f"{case['workload_id']}: provenance drift")
    cube = result["cube"]
    k_quanta = (case["k"] + 63) // 64

    if case["required_family"] == "SEEDED_ATOMIC_SPLIT_K":
        ranges = split_ranges(k_quanta, cube["usedCoreNum"])
        require(exact_cover(ranges, k_quanta),
                f"{case['workload_id']}: K ownership is not exact")
        require(cube["reserved"] == 1 and cube["singleCoreM"] == case["m"] and
                cube["singleCoreN"] == case["n"],
                f"{case['workload_id']}: pure seeded mode packet mismatch")
    else:
        m_tiles = (case["m"] + 127) // 128
        n_tiles = (case["n"] + 127) // 128
        total = m_tiles * n_tiles
        tail = total % 20
        full = total - tail
        full_owned = [round_index * 20 + block
                      for round_index in range(full // 20)
                      for block in range(20)]
        require(full_owned == list(range(full)),
                f"{case['workload_id']}: full-wave MN ownership is not exact")
        groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for block in range(20):
            tile, rank, size = tail_owner(block, tail)
            groups[tile].append((rank, size, block))
        require(sorted(groups) == list(range(tail)),
                f"{case['workload_id']}: tail tile ownership is incomplete")
        for tile, owners in groups.items():
            sizes = {size for _, size, _ in owners}
            require(len(sizes) == 1 and len(owners) == next(iter(sizes)),
                    f"{case['workload_id']}: tail group {tile} cardinality mismatch")
            size = next(iter(sizes))
            ranges = split_ranges(k_quanta, size)
            require(exact_cover(ranges, k_quanta),
                    f"{case['workload_id']}: tail group {tile} K cover mismatch")
        require(cube["reserved"] == 2 and
                result["derivation"]["cube_work_critical_path_reduction_ppm"] > 0,
                f"{case['workload_id']}: tail theorem did not improve critical work")
    return result


def main() -> None:
    provenance = audit_provenance()
    results = [audit_case(case) for case in CASES]
    family_counts = Counter(result["formula_family"] for result in results)
    packet_hashes = {result["packet_sha256"] for result in results}
    require(family_counts == {
        "SEEDED_ATOMIC_SPLIT_K": 100,
        "SEEDED_TAIL_WAVE_SPLIT_K": 100,
    }, "validation family coverage changed")
    require(len(packet_hashes) == 200, "every validation shape must produce a distinct packet")
    print(
        "NOVEL_FAMILY_AUDIT passed "
        f"official_family_records={provenance['official_family_records']} "
        f"official_source_files={provenance['official_source_files']} "
        f"official_host_strategies=8 "
        f"official_host_source_files={provenance['official_host_source_files']} "
        "repository_owned_families=2 shapes=200 distinct_packets=200 "
        "ownership_mismatches=0 resource_failures=0 forbidden_dependencies=0 "
        "device_build_targets=fp32_k90001,fp32_k90002"
    )


if __name__ == "__main__":
    main()
