#!/usr/bin/env python3
"""Build the fixed 200-shape validation domain for the independent kernel.

The first 120 shapes are selected offline to cover every installed 8.1 suffix
with ten distinct shapes.  That suffix label is validation metadata only; the
independent selector receives no label or official packet.  The final 80
shapes cover all six repository-owned modes, dtypes, and transpose layouts.
"""
from __future__ import annotations

from collections import defaultdict

from independent_tiled_selector import family_for, request


MAX_FOOTPRINT = 300 * 1024 * 1024
SYSTEM_WORKSPACE = 20 * 1024 * 1024
INSTALLED_SUFFIXES = (0, 1, 20, 21, 30, 31, 101, 200, 201, 10200, 10201, 20201)

# Frozen offline witnesses.  They are test-domain metadata, never an input to
# the independent selector.  Keeping them here avoids executing either the
# installed or reconstructed MatMulV3 selector during campaign generation.
INSTALLED_SEEDS = (
    (17, 17, 17, "fp16", False, False, 0),
    (19, 23, 31, "fp16", False, False, 0),
    (31, 27, 33, "fp16", False, False, 0),
    (33, 31, 37, "fp16", False, False, 0),
    (47, 19, 29, "fp16", False, False, 0),
    (16, 16, 16, "fp16", False, False, 1),
    (32, 32, 32, "fp16", False, False, 1),
    (48, 64, 64, "fp16", False, False, 1),
    (64, 48, 128, "fp16", False, False, 1),
    (80, 80, 256, "fp16", False, False, 1),
    (896, 2049, 27392, "fp16", False, False, 20),
    (1152, 1921, 28032, "fp16", False, False, 20),
    (1280, 1793, 28416, "fp16", False, False, 20),
    (1408, 1665, 28800, "fp16", False, False, 20),
    (1536, 1537, 29056, "fp16", False, False, 20),
    (896, 2176, 30336, "fp16", False, False, 21),
    (1024, 2944, 31616, "fp16", False, False, 21),
    (1152, 2816, 28032, "fp16", False, False, 21),
    (1280, 2560, 33024, "fp16", False, False, 21),
    (1408, 2304, 29696, "fp16", False, False, 21),
    (32, 49, 7808, "fp16", False, False, 30),
    (48, 97, 8064, "fp16", False, False, 30),
    (96, 113, 10368, "fp16", False, False, 30),
    (64, 65, 13056, "fp16", False, False, 30),
    (80, 17, 11392, "fp16", False, False, 30),
    (80, 112, 12032, "fp16", False, False, 31),
    (64, 96, 14208, "fp16", False, False, 31),
    (128, 512, 28032, "fp16", False, False, 31),
    (512, 48, 14592, "fp16", False, False, 31),
    (112, 64, 17024, "fp16", False, False, 31),
    (1, 65, 5120, "fp32", False, True, 101),
    (4, 80, 5504, "fp32", False, True, 101),
    (8, 96, 6144, "fp32", False, True, 101),
    (12, 128, 6656, "fp32", False, True, 101),
    (16, 160, 7168, "fp32", False, True, 101),
    (74432, 32, 9, "fp32", False, False, 200),
    (72848, 64, 5, "fp32", False, False, 200),
    (80240, 128, 3, "fp32", False, False, 200),
    (83776, 192, 11, "fp32", False, False, 200),
    (88560, 256, 13, "fp32", False, False, 200),
    (4109, 96, 48, "fp16", False, False, 201),
    (4276, 64, 112, "fp16", False, False, 201),
    (4463, 128, 32, "fp16", False, False, 201),
    (4413, 256, 96, "fp16", False, False, 201),
    (4381, 192, 128, "fp16", False, False, 201),
    (17408, 11, 11, "fp32", False, False, 10200),
    (14848, 7, 15, "fp32", False, False, 10200),
    (18176, 3, 9, "fp32", False, False, 10200),
    (22144, 13, 5, "fp32", False, False, 10200),
    (21504, 9, 3, "fp32", False, False, 10200),
    (24064, 34, 112, "fp16", False, False, 10201),
    (19200, 25, 16, "fp16", False, False, 10201),
    (28160, 153, 96, "fp16", False, False, 10201),
    (18560, 61, 80, "fp16", False, False, 10201),
    (23424, 130, 48, "fp16", False, False, 10201),
    (11776, 56, 240, "fp32", False, False, 20201),
    (12288, 17, 16, "fp32", False, False, 20201),
    (16384, 31, 24, "fp32", False, False, 20201),
    (20480, 47, 32, "fp32", False, False, 20201),
    (22528, 63, 40, "fp32", False, False, 20201),
)


def footprint(m: int, n: int, k: int, dtype: str) -> int:
    width = 4 if dtype == "fp32" else 2
    return (m * k + k * n + 2 * m * n) * width + SYSTEM_WORKSPACE


def installed_branch_cases() -> list[dict]:
    rows = []
    group_index = defaultdict(int)
    for m, n, k, dtype, ta, tb, suffix in INSTALLED_SEEDS:
        # Two points per seed.  Across each suffix group all M/N/K dimensions
        # vary because the five fixed seeds are structurally distinct.
        variants = ((m, n, k), (m, n + 16, k) if suffix == 101 else (m + 16, n, k))
        for vm, vn, vk in variants:
            if footprint(vm, vn, vk, dtype) > MAX_FOOTPRINT:
                raise RuntimeError(f"installed suffix {suffix} witness exceeds footprint cap")
            index = group_index[suffix]
            rows.append({
                "workload_id": f"installed_{suffix}_{index:02d}",
                "m": vm, "n": vn, "k": vk, "dtype": dtype,
                "trans_a": ta, "trans_b": tb,
                "validation_group": f"INSTALLED_SUFFIX_{suffix}",
                "official_suffix_audit": suffix,
            })
            group_index[suffix] += 1
    if dict(group_index) != {suffix: 10 for suffix in INSTALLED_SUFFIXES}:
        raise RuntimeError(f"frozen installed witness coverage drift: {dict(group_index)}")
    return rows


def independent_mode_cases() -> list[dict]:
    specs: list[tuple[str, int]] = [
        ("RESIDENT_A_N_STRIPE", 20),
        ("MICRO_DIRECT", 12),
        ("BALANCED_MN", 12),
        ("SEEDED_SPLIT_K", 12),
        ("RESIDENT_B_M_STRIPE", 12),
        ("SEEDED_TAIL_WAVE", 12),
    ]
    dtypes = ("fp16", "bf16", "fp32")
    layouts = ((False, False), (False, True), (True, False), (True, True))
    rows = []
    for family, count in specs:
        made = 0
        cursor = 0
        while made < count:
            dtype = dtypes[cursor % len(dtypes)]
            ta, tb = layouts[(cursor // len(dtypes)) % len(layouts)]
            if family == "MICRO_DIRECT":
                m = 17 + 16 * (cursor % 6)
                n = 19 + 16 * ((cursor * 3) % 6)
                k = 64 + 128 * (cursor % 16)
            elif family == "SEEDED_SPLIT_K":
                m = 16 + 16 * (cursor % 8)
                n = 16 + 16 * ((cursor * 3) % 8)
                # Enough K for all 20 owners for fp16/bf16 as well as fp32.
                k = 12288 + 1024 * (cursor % 12)
            elif family == "RESIDENT_B_M_STRIPE":
                m = 4096 + 384 * cursor
                n = 16 + 16 * (cursor % 12)
                k = 32 + 32 * ((cursor * 5) % 12)
            elif family == "RESIDENT_A_N_STRIPE":
                m = 16 + 16 * (cursor % 8)
                # Twenty N tiles suppress the tail-wave predicate.
                n = 2433 + 16 * (cursor % 7)
                k = 256 + 256 * (cursor % 12)
            elif family == "SEEDED_TAIL_WAVE":
                m_tiles = 3 + cursor % 5
                n_tiles = 7 + (cursor * 2) % 7
                total = m_tiles * n_tiles
                if not (20 < total <= 320 and 1 <= total % 20 <= 10):
                    cursor += 1
                    continue
                m = (m_tiles - 1) * 128 + 17 + 16 * (cursor % 6)
                n = (n_tiles - 1) * 128 + 19 + 16 * ((cursor * 3) % 6)
                k = 4096 + 1024 * (cursor % 12)
            else:
                m = 769 + 64 * (cursor % 9)
                n = 1025 + 64 * ((cursor * 5) % 11)
                k = 512 + 256 * (cursor % 10)

            req = request(m, k, n, dtype, ta, tb)
            actual, _ = family_for(req)
            if actual == family and footprint(m, n, k, dtype) <= MAX_FOOTPRINT:
                rows.append({
                    "workload_id": f"independent_{family.lower()}_{made:02d}",
                    "m": m, "n": n, "k": k, "dtype": dtype,
                    "trans_a": ta, "trans_b": tb,
                    "validation_group": f"INDEPENDENT_{family}",
                    "official_suffix_audit": "",
                })
                made += 1
            cursor += 1
            if cursor > 20_000:
                raise RuntimeError(f"could not construct {family} validation cases")
    return rows


def build_cases() -> list[dict]:
    rows = installed_branch_cases() + independent_mode_cases()
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 validation shapes, got {len(rows)}")
    identities = {
        (r["m"], r["n"], r["k"], r["dtype"], r["trans_a"], r["trans_b"])
        for r in rows
    }
    if len(identities) != len(rows):
        raise RuntimeError("validation shapes are not unique")
    suffix_counts = defaultdict(int)
    family_counts = defaultdict(int)
    for row in rows:
        if row["official_suffix_audit"] != "":
            suffix_counts[int(row["official_suffix_audit"])] += 1
        family, _ = family_for(request(
            row["m"], row["k"], row["n"], row["dtype"],
            row["trans_a"], row["trans_b"],
        ))
        family_counts[family] += 1
    if dict(suffix_counts) != {suffix: 10 for suffix in INSTALLED_SUFFIXES}:
        raise RuntimeError(f"installed suffix coverage drift: {dict(suffix_counts)}")
    if set(family_counts) != {
        "MICRO_DIRECT", "BALANCED_MN", "SEEDED_SPLIT_K",
        "RESIDENT_B_M_STRIPE", "RESIDENT_A_N_STRIPE", "SEEDED_TAIL_WAVE",
    }:
        raise RuntimeError(f"independent family coverage drift: {dict(family_counts)}")
    return rows


if __name__ == "__main__":
    from collections import Counter
    cases = build_cases()
    print(f"VALIDATION_CASES shapes={len(cases)}")
    print("suffixes=" + str(Counter(r["official_suffix_audit"] for r in cases if r["official_suffix_audit"] != "")))
    print("families=" + str(Counter(
        family_for(request(r["m"], r["k"], r["n"], r["dtype"], r["trans_a"], r["trans_b"]))[0]
        for r in cases
    )))
