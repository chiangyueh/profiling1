#!/usr/bin/env python3

import argparse
import collections
import json
import os
import random
import subprocess
import sys


# The 21 route names below cover the installed MatMulV3 kernels.  The formula
# fallback cannot synthesize four of them for a previously unseen shape.
TARGET_BRANCHES = (
    "BASE",
    "BASE_ND2NZ",
    "AL1_FULL_LOAD",
    "BL1_FULL_LOAD",
    "BL1_FULL_LOAD_ND2NZ",
    "BL1_FULL_LOAD_FIXPIPE",
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ",
    "BL1_FULL_LOAD_VEC_NZ2ND",
    "BL1_FULL_LOAD_CVP_PARALLEL",
    "BASE_CVP_PARALLEL",
    "BASE_K_SHIFT",
    "SINGLE_CORE_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_ND2NZ",
    "SINGLE_CORE_NKM_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1_ND2NZ",
    "DETERMINISTIC_SPLIT_K",
    "DETERMINISTIC_SPLIT_K_ND2NZ",
    "DETERMINISTIC_SPLIT_K_VEC_NZ2ND",
    "DETERMINISTIC_SPLIT_K_VEC_NZ2ND_ND2NZ",
    "MULTI_CORE_SPLIT_K",
)

# The Ascend910B3 host selector has eight reachable non-Split-K routes.  The
# tuples below are the complete input/output dtype combinations accepted by
# the official MatMulV3 implementation for each route.  A and B always share
# the input dtype; the output may retain that dtype or accumulate to FP32.
ALL_IO_ROUTE_DTYPES = {
    "BASE": (
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "BASE_ND2NZ": (
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "AL1_FULL_LOAD": (("fp32", "fp32"),),
    "BL1_FULL_LOAD": (
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "BL1_FULL_LOAD_ND2NZ": (
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "BL1_FULL_LOAD_FIXPIPE": (
        ("fp16", "fp16"), ("fp16", "fp32"),
        ("bf16", "bf16"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ": (
        ("fp16", "fp32"), ("bf16", "fp32"), ("fp32", "fp32"),
    ),
    "BL1_FULL_LOAD_VEC_NZ2ND": (("fp32", "fp32"),),
}
ALL_IO_COMBINATIONS = tuple(
    (branch, input_dtype, output_dtype)
    for branch, dtype_pairs in ALL_IO_ROUTE_DTYPES.items()
    for input_dtype, output_dtype in dtype_pairs
)
assert len(ALL_IO_COMBINATIONS) == 30

# The current pass is intentionally restricted to non-Split-K routes.  The
# completed Split-K response curves remain in the checkpoint, but are neither
# selected nor printed again.
MEASUREMENT_BRANCHES = (
    "BASE",
    "BASE_ND2NZ",
    "AL1_FULL_LOAD",
    "BL1_FULL_LOAD",
    "BL1_FULL_LOAD_ND2NZ",
    "BL1_FULL_LOAD_FIXPIPE",
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ",
    "BL1_FULL_LOAD_VEC_NZ2ND",
)

# Official-dispatch discovery on CANN 8.5 found no reachable plain
# BL1_FULL_LOAD packet and AL1 occupies a very narrow selector pocket.  Those
# two routes remain recorded when observed, but cannot prevent the broad
# reachable non-Split-K route datasets from being measured.
OFFICIAL_QUOTA_BRANCHES = tuple(
    name for name in MEASUREMENT_BRANCHES
    if name not in ("AL1_FULL_LOAD", "BL1_FULL_LOAD")
)

#NEW: Only these routes currently have a source/ISA-derived rule that can
# actually lower usedCoreNum.  Discovery below enables the real shrink hook
# and admits a shape only when actual_core < official_core, so the timed quota
# cannot be filled by fallback/no-op packets.
SHRINK_ENABLED_BRANCHES = (
    "BASE",
    "AL1_FULL_LOAD",
    "BL1_FULL_LOAD_ND2NZ",
    "BL1_FULL_LOAD_FIXPIPE",
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ",
    "BL1_FULL_LOAD_VEC_NZ2ND",
    "DETERMINISTIC_SPLIT_K",
    "DETERMINISTIC_SPLIT_K_ND2NZ",
)

# Official source restrictions, not measurement shortcuts.  AL1 and the
# VNCHW/Fixpipe-ND2NZ/Vector BL1 packets are FP32-only in the installed
# implementation.  Every other shrink-enabled route is validated for all
# three API dtypes.
SHRINK_BRANCH_DTYPES = {
    "BASE": ("fp32", "fp16", "bf16"),
    "AL1_FULL_LOAD": ("fp32",),
    "BL1_FULL_LOAD_ND2NZ": ("fp32",),
    "BL1_FULL_LOAD_FIXPIPE": ("fp32", "fp16", "bf16"),
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ": ("fp32",),
    "BL1_FULL_LOAD_VEC_NZ2ND": ("fp32",),
    "DETERMINISTIC_SPLIT_K": ("fp32", "fp16", "bf16"),
    "DETERMINISTIC_SPLIT_K_ND2NZ": ("fp32", "fp16", "bf16"),
}

# result26 rejects lowering the producer count for these complete families.
# They remain in the eight-branch validation: the formula's correct decision
# is to retain the official count rather than force a harmful shrink.
CORE_RETAIN_BRANCHES = (
    "BL1_FULL_LOAD_FIXPIPE",
    "DETERMINISTIC_SPLIT_K",
    "DETERMINISTIC_SPLIT_K_ND2NZ",
)

SHRINK_CORE_VALIDATION_SUCCESS = 50
SHRINK_CORE_VALIDATION_RESERVE = 75


def balanced_dtype_targets(branch, total):
    dtypes = SHRINK_BRANCH_DTYPES[branch]
    base, remainder = divmod(total, len(dtypes))
    return {dtype: base + (1 if index < remainder else 0)
            for index, dtype in enumerate(dtypes)}


def valid_core_decision(branch, official_core, actual_core):
    if (not isinstance(official_core, int) or not isinstance(actual_core, int) or
            official_core <= 0 or actual_core <= 0):
        return False
    if branch in CORE_RETAIN_BRANCHES:
        return actual_core == official_core
    return actual_core < official_core

REMAINING_CORE_SWEEP_BRANCHES = tuple(
    branch for branch in TARGET_BRANCHES if branch not in SHRINK_ENABLED_BRANCHES
)

FRESH_FORMULA_UNREACHABLE_BRANCHES = (
    "BL1_FULL_LOAD_CVP_PARALLEL",
    "BASE_CVP_PARALLEL",
    "BASE_K_SHIFT",
    "MULTI_CORE_SPLIT_K",
)

REPO_ONLY_BRANCHES = ("BASE_K_SHIFT", "MULTI_CORE_SPLIT_K")
SOURCE_UNREACHABLE_BRANCHES = ("BL1_FULL_LOAD_CVP_PARALLEL", "BASE_CVP_PARALLEL")

def add(pool, seen, dtype, layout, m, n, k):
    item = (dtype, layout, int(m), int(n), int(k))
    if min(item[2:]) <= 0 or item in seen:
        return
    # Keep timed inputs below 320 MiB. Discovery itself is metadata-only, but
    # every accepted shape must also be executable by the measurement pass.
    element_size = 4 if dtype == "fp32" else 2
    matrix_bytes = (m * k + k * n + m * n) * element_size
    if matrix_bytes > 320 * 1024 * 1024:
        return
    seen.add(item)
    pool[(dtype, layout)].append(item)


def candidate_pool():
    pool = collections.defaultdict(list)
    seen = set()
    rng = random.Random(0x910B85)

    # These blocks mirror the narrow source predicates.  They intentionally
    # precede the broad corpus: rare routes must not depend on random hits.

    # AL1 full-load: fp32, A=N, B=T, M<=16, 16<N<=320, K aligned to
    # 512/dtype bytes, and A resident in L1.  N starts at 80 so the
    # deterministic M,N<=64 rule cannot mask AL1.
    al1_k = (5120, 6144, 6656, 7168, 8192, 10240, 12288, 14336)
    for m in range(1, 17):
        for n in (64, 128, 192, 256, 320):
            for k in al1_k:
                add(pool, seen, "fp32", "NT", m, n, k)

    # Plain BL1 full-load needs M > 16*max(K,N), K<=256 and an on-the-fly
    # supported B.  Values below avoid the earlier fixpipe predicate.  The
    # first K set is also on-the-fly supported for A (plain BL1); the second
    # deliberately makes A use head ND2NZ while B remains on-the-fly.
    bl1_n = (32, 64, 128, 192, 256, 384)
    bl1_plain_k = (8, 16, 24, 32, 40, 48, 56, 64, 96)
    bl1_nd2nz_k = (17, 18, 19, 21, 25, 33, 41, 49, 57, 65, 73, 81)
    for index in range(160):
        m = 9216 + 128 * index
        if index < 24:
            for layout in ("NN", "NT"):
                add(pool, seen, "fp32", layout, m,
                    bl1_n[index % len(bl1_n)], bl1_plain_k[(index * 5) % len(bl1_plain_k)])
        # The real dispatcher admits this family through the NN head-ND2NZ
        # route.  NT with the same tails was exhaustively unproductive.
        for layout in ("NN",):
            add(pool, seen, "fp32", layout, m,
                bl1_n[(index + 1) % len(bl1_n)], bl1_nd2nz_k[(index * 7) % len(bl1_nd2nz_k)])
            add(pool, seen, "fp32", layout, m + 64,
                bl1_n[(index + 1) % len(bl1_n)], bl1_nd2nz_k[(index * 7) % len(bl1_nd2nz_k)])

    # K=1536 is a dedicated 8.5 DeepSeek path.  It requires both M and N to
    # be 128-aligned, one axis exactly 384 and the other at least 49152.
    # The two orientations select MKN and NKM respectively.
    for index in range(180):
        long_axis = 49152 + 128 * index
        for dtype in ("fp16", "bf16"):
            for layout in ("NN", "NT", "TN", "TT"):
                add(pool, seen, dtype, layout, 384, long_axis, 1536)
                add(pool, seen, dtype, layout, long_axis, 384, 1536)

    # Generic single-core Split-K, separated by the actual flags that decide
    # the three kernels.  NT + aligned K has no input ND2NZ.  NN + unaligned
    # N retains head ND2NZ.  N%128 selects GM-to-L1; a one-element K tail in
    # NT retains ND2NZ for its GM-to-L1 companion.
    for index in range(360):
        dtype = "fp16" if index % 2 == 0 else "bf16"
        m = 512 + 128 * (index % 17)
        n_aligned = 512 + 128 * ((index * 5) % 25)
        n_unaligned = n_aligned + 1 + 2 * (index % 31)
        k_aligned = 27392 + 128 * ((index * 11) % 45)
        k_unaligned = k_aligned + 1 + 2 * (index % 13)
        add(pool, seen, dtype, "NT", m, n_unaligned, k_aligned)
        add(pool, seen, dtype, "NN", m, n_unaligned, k_aligned)
        add(pool, seen, dtype, "NT", m, n_aligned, k_aligned)
        add(pool, seen, dtype, "NT", m + 64, n_aligned, k_unaligned)

    # FP32 NKM has an explicit selector window.  Use dense multiples around
    # the lower boundary instead of correlating M/N/K with one loop index.
    for m in range(1920, 2817, 16):
        for n in (8, 16, 24, 32, 40, 48, 56, 64):
            for k in (27392, 28032, 28672, 29696, 30336, 31616, 32768):
                add(pool, seen, "fp32", "NT", m, n, k)

    # Deterministic Split-K + VEC_NZ2ND + head ND2NZ.  GetMoreMultiCore-
    # SplitKArgs clears conversion for inner axes in [192,65535], hence the
    # explicit 129..191 inner axes.  Odd N>128 selects vector NZ2ND output.
    for index in range(420):
        m = 129 + (index * 7) % 63
        n = 129 + (index * 11 + index // 63) % 63
        if n % 16 == 0:
            n += 1
        k = 9216 + 128 * ((index * 13) % 320)
        for layout in ("NN", "TN", "TT"):
            add(pool, seen, "fp32", layout, m, n, k)

    # BL1 and the three fixpipe output variants. Long M and small K/N are
    # deliberately sampled on both aligned and unaligned boundaries.
    small_n = (17, 23, 31, 40, 47, 56, 64, 73, 80, 89, 96, 111, 112, 127, 128, 143, 160, 191, 192, 223, 240)
    small_k = (17, 24, 31, 40, 48, 63, 64, 67, 72, 80, 95, 96, 111, 112, 127, 128, 160, 191, 192, 223, 240, 255)
    for i in range(2400):
        layout = "NT" if i % 2 == 0 else "NN"
        m = 9216 + 128 * ((i * 37) % 680) + (i % 5)
        add(pool, seen, "fp32", layout, m, small_n[(i * 5) % len(small_n)],
            small_k[(i * 9) % len(small_k)])

    # BASE/BASE_ND2NZ across size scales. Odd and near-alignment dimensions
    # exercise the host ND2NZ decision without reusing an MNK triple.
    base_dims = (16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 320, 384,
                 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 3072, 4096)
    base_k = (65, 68, 95, 127, 129, 192, 255, 257, 384, 511, 513, 768, 1024,
              1535, 2048, 3072, 4096, 5120, 7168, 8192, 10240, 12288)
    for i in range(7000):
        dtype = ("fp32", "fp16", "bf16")[i % 3]
        layout = ("NN", "NT", "TN", "TT")[(i // 3) % 4]
        m = base_dims[(i * 7) % len(base_dims)] + (i % 7 if i % 4 == 0 else 0)
        n = base_dims[(i * 11 + 3) % len(base_dims)] + (i % 5 if i % 6 == 0 else 0)
        k = base_k[(i * 13 + 5) % len(base_k)]
        add(pool, seen, dtype, layout, m, n, k)

    # Deterministic Split-K, including aligned, input-ND2NZ and both
    # VEC_NZ2ND output variants.
    det_m = (3, 5, 7, 8, 12, 16, 17, 24, 31, 32, 47, 48, 63, 64, 80, 96, 112, 128,
             192, 256, 384, 512, 768, 1024)
    det_n = (8, 16, 17, 31, 32, 47, 49, 63, 64, 80, 96, 112, 128, 129, 192, 257,
             384, 513, 640, 1024)
    det_k = (6144, 6656, 7168, 8192, 10240, 12288, 16384, 17024, 24576, 27392,
             28032, 32768, 49152, 59968, 61440)
    for i in range(6500):
        layout = ("NN", "NT", "TN", "TT")[i % 4]
        add(pool, seen, "fp32", layout, det_m[(i * 5) % len(det_m)],
            det_n[(i * 7 + i // len(det_m)) % len(det_n)], det_k[(i * 11) % len(det_k)])

    # Single-core Split-K. NT aligned shapes cover normal and GM-to-L1;
    # unaligned NN/TN/TT shapes retain input ND2NZ. BF16 is included for
    # large K without fp16 overflow.
    split_m = (384, 512, 640, 768, 896, 1024, 1152, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 6144)
    split_n = (385, 512, 640, 769, 896, 1024, 1025, 1152, 1280, 1536, 1793, 2048, 2304, 2816, 3072, 4096)
    split_k = (27392, 28032, 28672, 29696, 30336, 31616, 32768, 40960, 49152, 57344, 61440)
    for i in range(9000):
        dtype = "fp16" if i % 2 == 0 else "bf16"
        layout = ("NT", "NN", "TN", "TT")[(i // 2) % 4]
        m = split_m[(i * 5) % len(split_m)] + (1 if layout[0] == "T" and i % 3 == 0 else 0)
        n = split_n[(i * 7 + i // len(split_m)) % len(split_n)]
        k = split_k[(i * 11) % len(split_k)] + (1 if layout != "NT" and i % 5 == 0 else 0)
        add(pool, seen, dtype, layout, m, n, k)

    # Randomized log-stratified tail. This is intentionally last: it expands
    # diversity and catches selector pockets not represented by the explicit
    # source-derived families above.
    values = (8, 12, 16, 17, 24, 31, 32, 47, 48, 63, 64, 80, 96, 112, 127,
              128, 160, 192, 256, 320, 384, 512, 640, 768, 896, 1024, 1280,
              1536, 1920, 2048, 3072, 4096, 6144, 8192, 12288, 16384)
    kvals = (64, 95, 127, 128, 192, 255, 256, 384, 512, 768, 1024, 1536,
             2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 27392,
             32768, 49152, 61440)
    for i in range(12000):
        dtype = rng.choice(("fp32", "fp16", "bf16"))
        layout = rng.choice(("NN", "NT", "TN", "TT"))
        m = rng.choice(values) + rng.choice((0, 0, 0, 1, 3, 5))
        n = rng.choice(values) + rng.choice((0, 0, 0, 1, 3, 5))
        k = rng.choice(kvals) + rng.choice((0, 0, 0, 1))
        add(pool, seen, dtype, layout, m, n, k)
    return pool


def shrink_validation_shapes():
    """Generate source-directed candidates for the enabled closed-form rules."""
    selected = []
    seen = set()

    def put(target, dtype, layout, m, n, k):
        item = (dtype, layout, int(m), int(n), int(k), target)
        key = item[:5]
        if key in seen or min(m, n, k) <= 0:
            return
        element_size = 4 if dtype == "fp32" else 2
        if (m * k + k * n + m * n) * element_size > 320 * 1024 * 1024:
            return
        seen.add(key)
        selected.append(item)

    # AL1: M<8 avoids the small-MN deterministic Split-K precedence rule.
    # N<320 guarantees an ownership count below 20, while the K grid spans
    # several legal A-resident footprints.
    # The grid deliberately omits all previously timed AL1 witnesses.  Host
    # discovery still has to prove that each fresh point reaches AL1.
    al1_n = (48, 80, 96, 112, 144, 160, 176, 208, 224, 240, 272, 288, 304)
    al1_k = (4352, 4608, 4864, 5376, 5632, 5888, 6400, 6912, 7424, 7680, 7936)
    for i in range(180):
        put("AL1_FULL_LOAD", "fp32", "NT", 1 + i % 7,
            al1_n[(i * 5 + i // 13) % len(al1_n)],
            al1_k[(i * 7 + i // 11) % len(al1_k)])

    # Pure BASE immediately beyond the AL1 resident-A domain.  Discovery
    # rejects any candidate that is routed elsewhere or whose BASE rule is a
    # no-op, so this may deliberately over-cover the selector boundary.
    base_n = (40, 56, 72, 88, 104, 120)
    base_k = (9472, 9984, 10752, 11776, 12800, 13824, 14848, 15872)
    for i in range(220):
        put("BASE", "fp32", "NT", 3 + i % 21,
            base_n[(i * 5 + i // 17) % len(base_n)],
            base_k[(i * 3 + i // 19) % len(base_k)])

    # BL1 + A-head VNCHW ND2NZ.  The new rule preserves both the official
    # body wave count and the serialized head wave count.
    bl1_nd_k = (17, 19, 21, 25, 33, 41, 49, 57, 65, 73, 81)
    bl1_nd_n = (32, 64, 96, 128, 192, 256, 384)
    for i in range(220):
        put("BL1_FULL_LOAD_ND2NZ", "fp32", "NN", 11731 + 96 * i + (i % 5),
            bl1_nd_n[(i * 3 + i // 7) % len(bl1_nd_n)],
            bl1_nd_k[(i * 5 + i // 11) % len(bl1_nd_k)])

    # BL1 Vector NZ2ND output.  Points straddle the rule's depth/tail
    # fallbacks; discovery keeps only packets on which the production formula
    # really shrinks.
    vec_n = (23, 31, 40, 47, 56, 73, 80, 89, 96, 111, 112, 127, 143, 160, 175, 191)
    vec_k = (24, 40, 64, 80, 96, 112, 128)
    for i in range(260):
        put("BL1_FULL_LOAD_VEC_NZ2ND", "fp32", "NN", 13037 + 320 * i + (i % 13),
            vec_n[(i * 7 + i // 9) % len(vec_n)],
            vec_k[(i * 5 + i // 13) % len(vec_k)])

    # Fixpipe: transposed-A packets and the seven-wave NN packet admit one
    # additional body wave.  Head-ND2NZ is limited to narrow N/K packets.
    for i in range(160):
        put("BL1_FULL_LOAD_FIXPIPE", "fp32", "TN", 11264 + 64 * i,
            (11, 17, 31, 47, 51, 63)[i % 6],
            (96, 112, 127, 128, 160, 192, 224, 256)[(i * 3) % 8])
        put("BL1_FULL_LOAD_FIXPIPE", "fp32", "NN", 15488 + 16 * i,
            (128, 144, 160, 176, 192)[(i * 3) % 5],
            (65, 67, 69, 71, 73)[(i * 2) % 5])
        put("BL1_FULL_LOAD_FIXPIPE_ND2NZ", "fp32", "NN", 19456 + 64 * i,
            (17, 19, 23, 25, 31)[(i * 3) % 5],
            (65, 80, 96, 112, 127, 128)[(i * 5) % 6])

    # Deterministic Split-K: cover the three plain packet equations and the
    # A-head ND2NZ equations without relying on an MNK lookup table.
    for i in range(180):
        dtype = "fp16" if i % 2 == 0 else "bf16"
        put("DETERMINISTIC_SPLIT_K", dtype, "NT", 1280 + 64 * (i % 24),
            (512, 1024, 1152, 1536, 1792)[(i * 3) % 5],
            31168 + 128 * (i % 16))
        put("DETERMINISTIC_SPLIT_K", "fp32", "NN", (32, 48, 64, 80, 96, 112, 128)[i % 7],
            (16, 32, 48, 64)[(i * 3) % 4],
            (16512, 24704, 32896)[(i * 5) % 3])
        put("DETERMINISTIC_SPLIT_K", "fp32", "NT", 1536 + 64 * (i % 16),
            (16, 24, 32)[(i * 2) % 3], 26944 + 64 * (i % 20))
    for i in range(180):
        put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp16", "TN" if i % 2 == 0 else "TT",
            32 + (i * 7) % 129, 32 + (i * 11) % 161,
            (12288, 32768, 61441)[(i * 5) % 3])
    return selected


def shrink_core_validation_shapes():
    """Fresh candidates for a complete core-aware 8-branch validation."""
    selected = []
    seen = set()

    def put(target, dtype, layout, m, n, k):
        item = (dtype, layout, int(m), int(n), int(k), target)
        key = item[:5]
        if key in seen or min(key[2:]) <= 0:
            return
        element_size = 4 if dtype == "fp32" else 2
        if (m * k + k * n + m * n) * element_size > 320 * 1024 * 1024:
            return
        seen.add(key)
        selected.append(item)

    # AL1 is a narrow fp32 NT selector pocket.  N=48 is a proven official V3
    # route; dense, previously unused K values avoid result24.
    retired_al1_k = {6400, 6912}
    for m in range(1, 8):
        for k in range(4224, 8577, 128):
            if k not in retired_al1_k:
                put("AL1_FULL_LOAD", "fp32", "NT", m, 52, k)
                put("AL1_FULL_LOAD", "fp32", "NT", m, 68, k)

    # BASE stays in the proven narrow-N region but uses K values outside
    # result24's grid.
    retired_base_k = {9472, 9984, 10752, 11776, 12800, 13824, 14848, 15872}
    for m in range(17, 32):
        for n in (44, 60):
            for k in range(9216, 16641, 128):
                if k not in retired_base_k:
                    put("BASE", "fp32", "NT", m, n, k)

    # BASE itself is not FP32-only.  The two TT grids below target the
    # independently observed FP16 and BF16 L2 packets with fewer than twenty
    # tasks in the active window.  Discovery still verifies the branch and
    # that the production rule really changes usedCoreNum.
    half_base_k = (14592, 15232, 16000, 16384, 16768, 17536, 18432, 19328, 20224)
    for i in range(720):
        put("BASE", "fp16", "TT", 2368 + 128 * (i % 49),
            (7, 9, 11, 13, 17, 19, 23, 25, 29, 31)[(i * 3 + i // 17) % 10],
            half_base_k[(i * 5 + i // 29) % len(half_base_k)])
        put("BASE", "bf16", "TT", 21 + 8 * (i % 18),
            (768, 1024, 1280, 1536, 1792, 2048, 2304, 2560)[(i * 5 + i // 19) % 8],
            half_base_k[(i * 7 + i // 31) % len(half_base_k)])

    # BL1 head-ND2NZ: new M progressions preserve the source predicate while
    # making every exact shape distinct from the earlier campaign.
    bl1_n = (32, 64, 96, 128, 192, 256, 384)
    bl1_k = (17, 19, 21, 25, 33, 41, 49, 57, 65, 73, 81)
    for i in range(560):
        put("BL1_FULL_LOAD_ND2NZ", "fp32", "NN", 11829 + 112 * i + i % 11,
            bl1_n[(i * 3 + i // 17) % len(bl1_n)],
            bl1_k[(i * 5 + i // 19) % len(bl1_k)])
    # result26 validates only the 16/17-core body/head plateaus.  Supply a
    # dense fresh grid inside that source-derived M interval so discovery is
    # not forced to scan the rejected 18/19-core tail.
    for i, m in enumerate(range(11840, 17249, 64)):
        for n in (32, 64, 128, 192, 256):
            put("BL1_FULL_LOAD_ND2NZ", "fp32", "NN", m + i % 7 + 1, n,
                bl1_k[(i * 3 + n // 32) % len(bl1_k)])

    # BL1 Vector NZ2ND: retain the strict short-K packet, with fresh M values.
    vec_n = (23, 31, 40, 47, 56, 73, 80, 89, 96, 111, 112, 127, 143, 160, 175, 191)
    vec_k = (24, 40, 64, 80, 96, 112, 128)
    for i in range(560):
        put("BL1_FULL_LOAD_VEC_NZ2ND", "fp32", "NN", 13127 + 288 * i + i % 17,
            vec_n[(i * 7 + i // 13) % len(vec_n)],
            vec_k[(i * 5 + i // 23) % len(vec_k)])

    # Fixpipe is deliberately split between its transposed-A and seven-wave NN
    # subdomains because result24 showed materially different performance.
    for i in range(320):
        put("BL1_FULL_LOAD_FIXPIPE", "fp32", "TN", 11344 + 64 * i,
            (11, 17, 31, 47, 51, 63)[i % 6],
            (96, 112, 127, 128, 160, 192, 224, 256)[(i * 3) % 8])
        put("BL1_FULL_LOAD_FIXPIPE", "fp32", "NN", 15507 + 13 * i,
            (128, 144, 160, 176, 192)[(i * 3) % 5],
            (65, 67, 69, 71, 73)[(i * 2) % 5])
        put("BL1_FULL_LOAD_FIXPIPE_ND2NZ", "fp32", "NN", 19529 + 73 * i,
            (17, 19, 23, 25, 31)[(i * 3) % 5],
            (65, 80, 96, 112, 127)[(i * 2) % 5])

    # NeedSolveFixBound explicitly supports two-byte FP16/BF16 inputs.  Keep
    # these reachable plain-Fixpipe packets in validation even though the
    # corrected core decision now retains the official count.
    for i in range(900):
        m = 57363 + 128 * i + i % 23
        n = (47, 51, 63, 73, 80, 89, 95, 111, 112, 119)[(i * 3 + i // 31) % 10]
        k = (64, 80, 96, 112, 127, 128, 160, 192, 224, 256)[(i * 7 + i // 37) % 10]
        put("BL1_FULL_LOAD_FIXPIPE", "fp16", "NN", m, n, k)
        put("BL1_FULL_LOAD_FIXPIPE", "bf16", "NN", m + 37, n, k)

    # Deterministic Split-K covers both half dtypes plus the two fp32 packet
    # equations, all outside result24's exact K grid.
    for i in range(360):
        dtype = "fp16" if i % 2 == 0 else "bf16"
        put("DETERMINISTIC_SPLIT_K", dtype, "NT", 1344 + 64 * (i % 28),
            (512, 896, 1152, 1536, 1792)[(i * 3) % 5],
            33792 + 128 * (i % 32))
        put("DETERMINISTIC_SPLIT_K", "fp32", "NN",
            (35, 51, 67, 83, 99, 115, 125)[i % 7],
            (16, 32, 48, 64)[(i * 3) % 4],
            (16640, 24832, 33024)[(i * 5) % 3])
        put("DETERMINISTIC_SPLIT_K", "fp32", "NT", 1600 + 64 * (i % 18),
            (16, 24, 32)[(i * 2) % 3], 28352 + 64 * (i % 24))

    # A-head ND2NZ is reachable for FP32, FP16 and BF16.  Sample all three
    # input packets while the corrected decision retains the official count.
    for m in range(20, 158, 9):
        for n in (32, 64, 96, 128, 160, 192):
            for k in (15808, 16448, 17152):
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp32", "TN", m, n, k)
            for k in (31872, 32640, 33408, 34176):
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp16", "TN", m, n, k)
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp16", "TT", m + 1, n, k)
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "bf16", "TT", m + 2, n, k)
            for k in (12416, 61569):
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp16", "TN", m, n, k)
                put("DETERMINISTIC_SPLIT_K_ND2NZ", "fp16", "TT", m + 1, n, k)
    return selected


def remaining_validation_shapes():
    """Source-directed fresh candidates for the 13 non-shrink routes."""
    selected = []
    seen = set()

    def put(target, dtype, layout, m, n, k):
        item = (dtype, layout, int(m), int(n), int(k), target)
        key = item[:5]
        if target not in REMAINING_CORE_SWEEP_BRANCHES or key in seen or min(key[2:]) <= 0:
            return
        element_size = 4 if dtype == "fp32" else 2
        if (m * k + k * n + m * n) * element_size > 320 * 1024 * 1024:
            return
        seen.add(key)
        selected.append(item)

    # BASE with a head conversion.  Prime offsets keep these points distinct
    # from the earlier regular grids while spanning small, medium and large
    # operation counts.
    for i in range(360):
        dtype = ("fp32", "fp16", "bf16")[i % 3]
        layout = ("NN", "TN", "TT")[(i // 3) % 3]
        m = (97, 193, 385, 641, 1025, 1537, 2305, 3073)[(i * 5) % 8] + i % 11
        n = (83, 179, 353, 547, 773, 1157, 1667, 2179)[(i * 7 + i // 8) % 8]
        k = (769, 1281, 2305, 3585, 5121, 7425, 9857, 13569)[(i * 3 + i // 9) % 8]
        put("BASE_ND2NZ", dtype, layout, m, n, k)

    # Plain BL1: avoid the earlier Fixpipe selector with N >= 256 while
    # retaining the exact on-the-fly dimensions required by the BL1 formula.
    for i in range(240):
        layout = "NT" if i % 2 else "NN"
        n = (256, 384)[(i // 2) % 2]
        k = (8, 16, 24, 32, 40, 48, 56, 64, 96)[(i * 5) % 9]
        m = 32771 + 193 * i
        put("BL1_FULL_LOAD", "fp32", layout, m, n, k)

    # The two K=1536 orientations have explicit 8.5 selectors.
    for i in range(96):
        long_axis = 49280 + 128 * i
        dtype = "fp16" if i % 2 == 0 else "bf16"
        layout = ("NN", "NT", "TN", "TT")[i % 4]
        put("SINGLE_CORE_SPLIT_K", dtype, layout, 384, long_axis, 1536)
        put("SINGLE_CORE_NKM_SPLIT_K", dtype, layout, long_axis, 384, 1536)

    # Generic Split-K variants.  The aligned NT points select the GM-to-L1
    # path; head/tail conversions split the companion ND2NZ routes.
    for i in range(420):
        dtype = "fp16" if i % 2 == 0 else "bf16"
        m = 577 + 64 * (i % 37)
        n_aligned = 640 + 128 * ((i * 7 + i // 13) % 31)
        n_tail = n_aligned + 1 + 2 * (i % 29)
        k_aligned = 27776 + 128 * ((i * 11 + i // 17) % 83)
        k_tail = k_aligned + 1 + 2 * (i % 17)
        put("SINGLE_CORE_SPLIT_K_ND2NZ", dtype, "NN", m, n_tail, k_aligned)
        put("SINGLE_CORE_SPLIT_K_GM_TO_L1", dtype, "NT", m + 63, n_aligned, k_aligned)
        put("SINGLE_CORE_SPLIT_K_GM_TO_L1_ND2NZ", dtype, "NT", m + 127, n_aligned, k_tail)

    # Deterministic Split-K Vector NZ2ND output, with and without a head
    # input conversion.  Odd output N prevents the aligned fixpipe route.
    for i in range(420):
        m = 131 + (i * 7 + i // 19) % 61
        n = 131 + (i * 11 + i // 23) % 61
        if n % 2 == 0:
            n += 1
        k = 9344 + 128 * ((i * 13 + i // 29) % 337)
        put("DETERMINISTIC_SPLIT_K_VEC_NZ2ND", "fp32", "TN", m, n, k)
        put("DETERMINISTIC_SPLIT_K_VEC_NZ2ND_ND2NZ",
            "fp16" if i % 2 == 0 else "bf16", "TT", m, n, k + 1)

    # K-shift and Multi-Core Split-K are repository/AOE-only.  Broad official
    # candidates are still offered so an installed bank hit can be accepted;
    # no route name is fabricated.  The two CVP names get no candidates
    # because GetMixNd2nzType() cannot return V_PARALELL_ND2NZ in this source.
    for i in range(240):
        dtype = ("fp32", "fp16", "bf16")[i % 3]
        layout = ("NN", "NT", "TN", "TT")[(i // 3) % 4]
        m = 257 + 128 * (i % 29)
        n = 263 + 64 * ((i * 5) % 31)
        k = 4097 + 128 * ((i * 7) % 43)
        put("BASE_K_SHIFT", dtype, layout, m, n, k)
    for i in range(240):
        m = 16 * (1 + i % 16)
        n = 16 * (1 + (i * 7 + i // 16) % 16)
        k = 8320 + 128 * ((i * 11 + i // 17) % 400)
        put("MULTI_CORE_SPLIT_K", "fp32", "NT", m, n, k)
    return selected


def all_io_core_sweep_candidates():
    """Source-directed candidates for all 30 reachable route/dtype pairs."""
    selected = []
    seen = set()

    def put(branch, input_dtype, output_dtype, layout, m, n, k):
        if (input_dtype, output_dtype) not in ALL_IO_ROUTE_DTYPES[branch]:
            return
        item = (input_dtype, output_dtype, layout, int(m), int(n), int(k), branch)
        if min(item[3:6]) <= 0 or item in seen:
            return
        input_size = 4 if input_dtype == "fp32" else 2
        output_size = 4 if output_dtype == "fp32" else 2
        tensor_bytes = (m * k + k * n) * input_size + m * n * output_size
        if tensor_bytes > 320 * 1024 * 1024:
            return
        seen.add(item)
        selected.append(item)

    # Pure BASE: dense narrow-output grids stay outside AL1 while spanning
    # all transpose patterns and several operation-count tiers.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BASE"]:
        for i in range(1800):
            layout = ("NN", "NT", "TN", "TT")[i % 4]
            m = 17 + (i * 7 + i // 41) % 239
            n = (40, 56, 72, 88, 104, 120, 136, 152)[(i * 5 + i // 37) % 8]
            k = 8320 + 128 * ((i * 11 + i // 43) % 81)
            put("BASE", input_dtype, output_dtype, layout, m, n, k)

    # BASE with input head conversion.  Odd dimensions prevent accidental
    # aligned packets; M/N/K progress independently to avoid a one-line grid.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BASE_ND2NZ"]:
        for i in range(2200):
            layout = ("NN", "TN", "TT")[(i + i // 29) % 3]
            m = 193 + (i * 67 + i // 17) % 3901
            n = 179 + (i * 43 + i // 23) % 2017
            k = 769 + (i * 131 + i // 31) % 15121
            if m % 2 == 0:
                m += 1
            if n % 2 == 0:
                n += 1
            if k % 2 == 0:
                k += 1
            put("BASE_ND2NZ", input_dtype, output_dtype, layout, m, n, k)

    # AL1 is an official FP32 NT-only selector pocket.
    for i in range(900):
        m = 1 + i % 7
        n = (52, 68, 84, 100, 116, 132, 148, 164, 180, 196,
             212, 228, 244, 260, 276, 292, 308)[(i * 5 + i // 19) % 17]
        k = 4224 + 128 * ((i * 7 + i // 23) % 35)
        put("AL1_FULL_LOAD", "fp32", "fp32", "NT", m, n, k)

    # Plain BL1 has two source-defined pockets.  FP16/BF16 use the explicit
    # 512x512 core-split full-load predicate.  FP32 uses on-the-fly K/N axes
    # that avoid both Fixpipe and input ND2NZ.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BL1_FULL_LOAD"]:
        if input_dtype == "fp32":
            for i in range(1800):
                m = 9216 + 64 * i + i % 17
                n = (64, 96, 128, 160, 192, 224, 256, 384)[(i * 3 + i // 31) % 8]
                k = (8, 16, 24, 32, 40, 48, 56, 64)[(i * 5 + i // 37) % 8]
                put("BL1_FULL_LOAD", input_dtype, output_dtype, "NT", m, n, k)
        else:
            for i in range(1200):
                m = 30848 + 64 * i + i % 13
                put("BL1_FULL_LOAD", input_dtype, output_dtype, "NT", m, 512, 512)

    # BL1 with A-head ND2NZ: B remains on-the-fly while an odd K forces the
    # A conversion.  The predicate is valid for all five I/O combinations.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BL1_FULL_LOAD_ND2NZ"]:
        for i in range(2200):
            m = 11265 + 96 * i + i % 19
            n = (32, 64, 96, 128, 160, 192, 224, 256, 384)[(i * 5 + i // 29) % 9]
            k = (17, 19, 21, 25, 33, 41, 49, 57, 65, 73, 81)[(i * 7 + i // 31) % 11]
            put("BL1_FULL_LOAD_ND2NZ", input_dtype, output_dtype, "NN", m, n, k)

    # Plain Fixpipe.  TN includes the already demonstrated high-benefit FP32
    # region; NN/NT provide independent half and mixed-output packets.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BL1_FULL_LOAD_FIXPIPE"]:
        for i in range(2600):
            if input_dtype == "fp32":
                layout = "TN" if i % 2 == 0 else "NN"
                m = 11264 + 32 * i + i % 23
                n = (11, 17, 31, 47, 51, 63, 73, 80, 89, 95,
                     111, 112, 119, 127, 143, 160, 176, 191, 223)[(i * 5 + i // 41) % 19]
                k = (63, 67, 73, 95, 111, 127, 160, 192, 224, 256)[(i * 7 + i // 43) % 10]
            else:
                layout = "NT" if output_dtype == "fp32" else "NN"
                m = 12288 + 64 * i + i % 29
                n = (47, 51, 63, 73, 80, 89, 95, 111, 112, 119)[(i * 3 + i // 37) % 10]
                k = (64, 80, 96, 112, 128, 160, 192, 224, 256)[(i * 7 + i // 47) % 9]
            put("BL1_FULL_LOAD_FIXPIPE", input_dtype, output_dtype, layout, m, n, k)

    # Fixpipe plus head ND2NZ exists for FP32 output.  Odd K forces the A-head
    # conversion for half/bfloat inputs without changing the public route.
    for input_dtype, output_dtype in ALL_IO_ROUTE_DTYPES["BL1_FULL_LOAD_FIXPIPE_ND2NZ"]:
        for i in range(2600):
            m = 19457 + 64 * i + i % 31
            n = (17, 19, 23, 25, 31, 47, 51, 63, 73, 80,
                 89, 95, 111, 112, 119)[(i * 5 + i // 41) % 15]
            k = (65, 67, 69, 71, 73, 81, 95, 97, 111, 113, 127)[(i * 7 + i // 43) % 11]
            put("BL1_FULL_LOAD_FIXPIPE_ND2NZ", input_dtype, output_dtype, "NN", m, n, k)

    # Vector NZ2ND is explicitly FP32-only.
    for i in range(2600):
        m = 12289 + 256 * i + i % 37
        n = (23, 31, 40, 47, 56, 73, 80, 89, 96, 111,
             112, 127, 143, 160, 175, 191)[(i * 7 + i // 41) % 16]
        k = (24, 40, 64, 80, 96, 112, 128)[(i * 5 + i // 43) % 7]
        put("BL1_FULL_LOAD_VEC_NZ2ND", "fp32", "fp32", "NN", m, n, k)
    return selected


def all_io_runner_env(base, input_dtype, output_dtype, layout, mode, discovery=False):
    env = runner_env(base, input_dtype, layout, mode, discovery=discovery)
    env["MATMUL_OUTPUT_DATA_TYPE"] = output_dtype
    return env


def read_all_io_selected(path):
    selected = []
    if not os.path.isfile(path):
        return selected
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 7:
                continue
            input_dtype, output_dtype, layout, m, n, k, branch = fields
            if ((branch, input_dtype, output_dtype) not in ALL_IO_COMBINATIONS or
                    layout not in ("NN", "NT", "TN", "TT")):
                continue
            selected.append((input_dtype, output_dtype, layout,
                             int(m), int(n), int(k), branch))
    return selected


def write_all_io_selected(path, selected):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        for input_dtype, output_dtype, layout, m, n, k, branch in selected:
            stream.write(
                f"{input_dtype}\t{output_dtype}\t{layout}\t{m}\t{n}\t{k}\t{branch}\n")
    os.replace(temporary, path)


def select_all_io_core_sweep(args):
    """Discover 50 real official packets for each of the 30 combinations."""
    candidate_groups = collections.defaultdict(list)
    for item in all_io_core_sweep_candidates():
        candidate_groups[(item[6], item[0], item[1])].append(item)

    selected = []
    selected_keys = set()
    counts = collections.Counter()
    for item in read_all_io_selected(args.selected):
        combination = (item[6], item[0], item[1])
        if counts[combination] >= args.quota:
            continue
        key = item[:6] + (item[6],)
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        counts[combination] += 1

    def discover(items, combination):
        if not items:
            return []
        branch, input_dtype, output_dtype = combination
        layout = items[0][2]
        env = all_io_runner_env(
            os.environ, input_dtype, output_dtype, layout, "discovery", discovery=True)
        rc, records, _stderr = invoke(
            args.runner, env,
            [(input_dtype, layout, m, n, k) for _, _, _, m, n, k, _ in items],
            args.run_log)
        if rc != 0 and len(items) > 1:
            accepted = []
            for item in items:
                accepted.extend(discover([item], combination))
            return accepted
        by_shape = {
            (record.get("shape"), record.get("input_dtype", record.get("dtype")),
             record.get("output_dtype"), record.get("layout")): record
            for record in records if record.get("status") == "DISCOVERED"
        }
        accepted = []
        for item in items:
            _, _, item_layout, m, n, k, _ = item
            shape = f"M{m}_N{n}_K{k}_{item_layout}"
            record = by_shape.get((shape, input_dtype, output_dtype, item_layout))
            if record is not None and record.get("branch") == branch:
                accepted.append(item)
        return accepted

    for combination in ALL_IO_COMBINATIONS:
        if counts[combination] >= args.quota:
            continue
        candidates = [item for item in candidate_groups[combination]
                      if item[:6] + (item[6],) not in selected_keys]
        ordered = sorted(candidates, key=lambda item: item[3] * item[4] * item[5])
        cuts = ((0, (len(ordered) + 2) // 3),
                ((len(ordered) + 2) // 3, (2 * len(ordered) + 2) // 3),
                ((2 * len(ordered) + 2) // 3, len(ordered)))
        tier_targets = (args.quota // 3, args.quota // 3,
                        args.quota - 2 * (args.quota // 3))
        attempted = set()
        for (begin, end), tier_target in zip(cuts, tier_targets):
            if counts[combination] >= args.quota:
                break
            tier_accepted = 0
            by_layout = collections.defaultdict(list)
            for item in ordered[begin:end]:
                by_layout[item[2]].append(item)
            for layout in sorted(by_layout):
                values = by_layout[layout]
                for offset in range(0, len(values), args.discovery_batch):
                    if tier_accepted >= tier_target or counts[combination] >= args.quota:
                        break
                    batch = values[offset:offset + args.discovery_batch]
                    attempted.update(batch)
                    for item in discover(batch, combination):
                        key = item[:6] + (item[6],)
                        if key in selected_keys:
                            continue
                        selected.append(item)
                        selected_keys.add(key)
                        counts[combination] += 1
                        tier_accepted += 1
                        if tier_accepted >= tier_target or counts[combination] >= args.quota:
                            break
                    write_all_io_selected(args.selected, selected)

        # Sparse selector pockets are allowed to borrow unused candidates
        # from another size tier, but never from another dtype combination.
        if counts[combination] < args.quota:
            by_layout = collections.defaultdict(list)
            for item in ordered:
                if item not in attempted and item[:6] + (item[6],) not in selected_keys:
                    by_layout[item[2]].append(item)
            for layout in sorted(by_layout):
                values = by_layout[layout]
                for offset in range(0, len(values), args.discovery_batch):
                    if counts[combination] >= args.quota:
                        break
                    batch = values[offset:offset + args.discovery_batch]
                    for item in discover(batch, combination):
                        key = item[:6] + (item[6],)
                        if key in selected_keys:
                            continue
                        selected.append(item)
                        selected_keys.add(key)
                        counts[combination] += 1
                        if counts[combination] >= args.quota:
                            break
                    write_all_io_selected(args.selected, selected)

        branch, input_dtype, output_dtype = combination
        print(json.dumps({"selection": "combination",
                          "branch": branch, "input_dtype": input_dtype,
                          "output_dtype": output_dtype, "selected": counts[combination],
                          "target": args.quota}, separators=(",", ":")), file=sys.stderr)

    order = {combination: index for index, combination in enumerate(ALL_IO_COMBINATIONS)}
    selected.sort(key=lambda item: (
        order[(item[6], item[0], item[1])], item[3] * item[4] * item[5],
        item[3], item[4], item[5]))
    write_all_io_selected(args.selected, selected)
    missing = {
        f"{branch}:{input_dtype}->{output_dtype}": args.quota - counts[(branch, input_dtype, output_dtype)]
        for branch, input_dtype, output_dtype in ALL_IO_COMBINATIONS
        if counts[(branch, input_dtype, output_dtype)] < args.quota
    }
    print(json.dumps({"all_io_selection": "complete" if not missing else "partial",
                      "combinations": len(ALL_IO_COMBINATIONS),
                      "selected_shapes": len(selected), "missing": missing},
                     separators=(",", ":")), file=sys.stderr)
    # Preserve all successfully discovered work even if an official selector
    # pocket is sparser than its source-level support declaration.
    return 0 if selected else 4


def select_shrink(args):
    #NEW: First ask the real host tiler which route each candidate reaches,
    # with the production shrink hook enabled.  This is metadata-only: no NPU
    # kernel is launched.  Keep at most quota effective-shrink shapes per
    # route and stop accepting a route as soon as it is full.
    candidates = shrink_validation_shapes()
    def scenario(item):
        dtype, layout, _m, _n, _k, target = item
        if target == "BL1_FULL_LOAD_FIXPIPE":
            return "fixpipe_trans_a" if layout[0] == "T" else "fixpipe_seven_wave"
        if target == "DETERMINISTIC_SPLIT_K":
            if dtype in ("fp16", "bf16"):
                return f"deterministic_{dtype}"
            return "deterministic_fp32_small" if layout == "NN" else "deterministic_fp32_narrow"
        if target == "DETERMINISTIC_SPLIT_K_ND2NZ":
            return f"deterministic_nd2nz_{layout.lower()}"
        return target.lower()

    #NEW: This is a fresh holdout campaign.  Historical measurements do not
    # reduce any quota.  Formula subdomains get independent quotas so one easy
    # pocket cannot stand in for an entire branch.
    scenario_quota = {
        "fixpipe_trans_a": args.quota // 2,
        "fixpipe_seven_wave": args.quota - args.quota // 2,
        "deterministic_fp16": args.quota // 6,
        "deterministic_bf16": args.quota // 6,
        "deterministic_fp32_small": args.quota // 3,
        "deterministic_fp32_narrow": args.quota - 2 * (args.quota // 3),
        "deterministic_nd2nz_tn": args.quota // 2,
        "deterministic_nd2nz_tt": args.quota - args.quota // 2,
    }
    groups = collections.defaultdict(list)
    for item in candidates:
        groups[(item[5], scenario(item))].append(item[:5])

    counts = collections.Counter()
    scenario_counts = collections.Counter()
    selected = []
    for (target, scenario_name), shapes in groups.items():
        target_quota = scenario_quota.get(scenario_name, args.quota)
        #NEW: Split each formula subdomain by operation count and require an
        # equal number of small, medium, and large holdout points.
        ordered = sorted(shapes, key=lambda item: item[2] * item[3] * item[4])
        cut1 = (len(ordered) + 2) // 3
        cut2 = (2 * len(ordered) + 2) // 3
        tiers = (ordered[:cut1], ordered[cut1:cut2], ordered[cut2:])
        tier_quotas = (target_quota // 3, target_quota // 3,
                       target_quota - 2 * (target_quota // 3))
        for tier_index, tier in enumerate(tiers):
            tier_key = (target, scenario_name, tier_index)
            by_environment = collections.defaultdict(list)
            for item in tier:
                by_environment[(item[0], item[1])].append(item)
            offsets = {key: 0 for key in by_environment}
            while scenario_counts[tier_key] < tier_quotas[tier_index]:
                progressed = False
                for (dtype, layout), environment_shapes in by_environment.items():
                    begin = offsets[(dtype, layout)]
                    batch = environment_shapes[begin:begin + args.discovery_batch]
                    if not batch:
                        continue
                    progressed = True
                    offsets[(dtype, layout)] += len(batch)
                    env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
                    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "1"
                    rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                    if rc != 0:
                        continue
                    by_shape = {record.get("shape"): record for record in records
                                if record.get("status") == "DISCOVERED"}
                    for item in batch:
                        _, _, m, n, k = item
                        shape = f"M{m}_N{n}_K{k}_{layout}"
                        record = by_shape.get(shape)
                        if record is None or record.get("branch") != target:
                            continue
                        official_core = record.get("official_core")
                        actual_core = record.get("actual_core")
                        if (not isinstance(official_core, int) or not isinstance(actual_core, int) or
                                actual_core <= 0 or actual_core >= official_core):
                            continue
                        selected.append(item + (target,))
                        counts[target] += 1
                        scenario_counts[tier_key] += 1
                        if scenario_counts[tier_key] >= tier_quotas[tier_index]:
                            break
                    if scenario_counts[tier_key] >= tier_quotas[tier_index]:
                        break
                if not progressed:
                    break
    write_selected(args.selected, selected)
    missing = {branch: args.quota - counts[branch] for branch in SHRINK_ENABLED_BRANCHES
               if counts[branch] < args.quota}
    print(json.dumps({"shrink_selection": "complete" if not missing else "incomplete_no_npu_run",
                      "counts": {branch: counts[branch] for branch in SHRINK_ENABLED_BRANCHES},
                      "missing": missing}, separators=(",", ":")), file=sys.stderr)
    return 0 if not missing else 4


def select_shrink_core_validation(args):
    def scenario(item):
        dtype, layout, _m, _n, _k, target = item
        if target == "BL1_FULL_LOAD_FIXPIPE":
            suffix = "trans_a" if layout[0] == "T" else "seven_wave"
            return f"fixpipe_{dtype}_{suffix}"
        if target == "DETERMINISTIC_SPLIT_K":
            if dtype in ("fp16", "bf16"):
                return f"deterministic_{dtype}"
            return "deterministic_fp32_small" if layout == "NN" else "deterministic_fp32_narrow"
        if target == "DETERMINISTIC_SPLIT_K_ND2NZ":
            return f"deterministic_nd2nz_{dtype}_{layout.lower()}"
        return f"{target.lower()}_{dtype}"

    scenario_quotas = {
        "base_fp32": 25,
        "base_fp16": 25,
        "base_bf16": 25,
        "fixpipe_fp32_trans_a": 13,
        "fixpipe_fp32_seven_wave": 12,
        "fixpipe_fp16_seven_wave": 25,
        "fixpipe_bf16_seven_wave": 25,
        "deterministic_fp16": 25,
        "deterministic_bf16": 25,
        "deterministic_fp32_small": 13,
        "deterministic_fp32_narrow": 12,
        "deterministic_nd2nz_fp32_tn": 25,
        "deterministic_nd2nz_fp16_tn": 13,
        "deterministic_nd2nz_fp16_tt": 12,
        "deterministic_nd2nz_bf16_tt": 25,
    }
    candidates = collections.defaultdict(list)
    for item in shrink_core_validation_shapes():
        candidates[(item[5], scenario(item))].append(item[:5])
    selected = []
    selected_keys = set()
    counts = collections.Counter()
    scenario_counts = collections.Counter()

    for (target, scenario_name), scenario_candidates in candidates.items():
        scenario_key = (target, scenario_name)
        quota = scenario_quotas.get(scenario_name, SHRINK_CORE_VALIDATION_RESERVE)
        ordered = sorted(scenario_candidates, key=lambda item: item[2] * item[3] * item[4])
        cut1 = (len(ordered) + 2) // 3
        cut2 = (2 * len(ordered) + 2) // 3
        tiers = (ordered[:cut1], ordered[cut1:cut2], ordered[cut2:])
        tier_quotas = (quota // 3, quota // 3, quota - 2 * (quota // 3))
        attempted = set()
        for tier, tier_quota in zip(tiers, tier_quotas):
            accepted = 0
            by_environment = collections.defaultdict(list)
            for item in tier:
                by_environment[(item[0], item[1])].append(item)
            for dtype, layout in sorted(by_environment):
                shapes = by_environment[(dtype, layout)]
                for offset in range(0, len(shapes), args.discovery_batch):
                    if accepted >= tier_quota or scenario_counts[scenario_key] >= quota:
                        break
                    batch = shapes[offset:offset + args.discovery_batch]
                    attempted.update(batch)
                    env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
                    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "1"
                    _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                    by_shape = {record.get("shape"): record for record in records
                                if record.get("status") == "DISCOVERED"}
                    for item in batch:
                        _, _, m, n, k = item
                        record = by_shape.get(f"M{m}_N{n}_K{k}_{layout}")
                        if record is None or record.get("branch") != target or item in selected_keys:
                            continue
                        official_core = record.get("official_core")
                        actual_core = record.get("actual_core")
                        if not valid_core_decision(target, official_core, actual_core):
                            continue
                        selected.append(item + (target,))
                        selected_keys.add(item)
                        counts[target] += 1
                        scenario_counts[scenario_key] += 1
                        accepted += 1
                        if accepted >= tier_quota or scenario_counts[scenario_key] >= quota:
                            break

        # Size tiers are preferred, but a sparse selector pocket must not
        # leave a quota short when another tier has valid unseen points.
        if scenario_counts[scenario_key] < quota:
            remaining = [item for item in ordered if item not in attempted and item not in selected_keys]
            by_environment = collections.defaultdict(list)
            for item in remaining:
                by_environment[(item[0], item[1])].append(item)
            for dtype, layout in sorted(by_environment):
                shapes = by_environment[(dtype, layout)]
                for offset in range(0, len(shapes), args.discovery_batch):
                    if scenario_counts[scenario_key] >= quota:
                        break
                    batch = shapes[offset:offset + args.discovery_batch]
                    env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
                    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "1"
                    _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                    by_shape = {record.get("shape"): record for record in records
                                if record.get("status") == "DISCOVERED"}
                    for item in batch:
                        _, _, m, n, k = item
                        record = by_shape.get(f"M{m}_N{n}_K{k}_{layout}")
                        if record is None or record.get("branch") != target or item in selected_keys:
                            continue
                        official_core = record.get("official_core")
                        actual_core = record.get("actual_core")
                        if not valid_core_decision(target, official_core, actual_core):
                            continue
                        selected.append(item + (target,))
                        selected_keys.add(item)
                        counts[target] += 1
                        scenario_counts[scenario_key] += 1
                        if scenario_counts[scenario_key] >= quota:
                            break

    selected.sort(key=lambda item: (item[5], item[2] * item[3] * item[4]))
    write_selected(args.selected, selected)
    selected_dtype_counts = collections.Counter((item[5], item[0]) for item in selected)
    missing = {branch: SHRINK_CORE_VALIDATION_RESERVE - counts[branch]
               for branch in SHRINK_ENABLED_BRANCHES
               if counts[branch] < SHRINK_CORE_VALIDATION_RESERVE}
    dtype_missing = {}
    for branch in SHRINK_ENABLED_BRANCHES:
        targets = balanced_dtype_targets(branch, SHRINK_CORE_VALIDATION_RESERVE)
        missing_for_branch = {
            dtype: target - selected_dtype_counts[(branch, dtype)]
            for dtype, target in targets.items()
            if selected_dtype_counts[(branch, dtype)] < target
        }
        if missing_for_branch:
            dtype_missing[branch] = missing_for_branch
    complete = not missing and not dtype_missing
    selection_status = "complete" if complete else "incomplete_continuing_with_available"
    print(json.dumps({"core_validation_selection": selection_status,
                      "counts": {branch: counts[branch] for branch in SHRINK_ENABLED_BRANCHES},
                      "dtype_counts": {
                          branch: {dtype: selected_dtype_counts[(branch, dtype)]
                                   for dtype in SHRINK_BRANCH_DTYPES[branch]}
                          for branch in SHRINK_ENABLED_BRANCHES
                      },
                      "success_target_per_branch": SHRINK_CORE_VALIDATION_SUCCESS,
                      "reserve_target_per_branch": SHRINK_CORE_VALIDATION_RESERVE,
                      "missing": missing, "dtype_missing": dtype_missing},
                     separators=(",", ":")), file=sys.stderr)
    # Do not discard valid work from seven branches because one branch or
    # dtype pocket missed its reserve.  compare-core-validation preserves the
    # per-branch/per-dtype targets, measures only discovered candidates, and
    # still returns incomplete at the end when any target remains unmet.
    return 0 if selected else 4


def select_remaining(args):
    candidates = collections.defaultdict(list)
    for item in remaining_validation_shapes():
        candidates[item[5]].append(item[:5])

    counts = collections.Counter()
    selected = []
    selected_keys = set()
    for item in read_selected(args.selected):
        if item[5] not in REMAINING_CORE_SWEEP_BRANCHES or counts[item[5]] >= args.quota:
            continue
        key = item[:5]
        if key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(key)
        counts[item[5]] += 1

    discovery_order = tuple(
        branch for branch in REMAINING_CORE_SWEEP_BRANCHES
        if branch not in FRESH_FORMULA_UNREACHABLE_BRANCHES
    ) + FRESH_FORMULA_UNREACHABLE_BRANCHES
    for target in discovery_order:
        if counts[target] >= args.quota:
            continue
        ordered = sorted(
            (item for item in candidates[target] if item not in selected_keys),
            key=lambda item: item[2] * item[3] * item[4], reverse=True)
        attempted = set()
        cut1 = (len(ordered) + 2) // 3
        cut2 = (2 * len(ordered) + 2) // 3
        tiers = (ordered[:cut1], ordered[cut1:cut2], ordered[cut2:])
        tier_quotas = (args.quota // 3 + (1 if args.quota % 3 else 0),
                       args.quota // 3 + (1 if args.quota % 3 > 1 else 0),
                       args.quota // 3)
        for tier, tier_quota in zip(tiers, tier_quotas):
            accepted = 0
            by_environment = collections.defaultdict(list)
            for item in tier:
                by_environment[(item[0], item[1])].append(item)
            for dtype, layout in sorted(by_environment):
                environment_shapes = by_environment[(dtype, layout)]
                for offset in range(0, len(environment_shapes), args.discovery_batch):
                    if accepted >= tier_quota or counts[target] >= args.quota:
                        break
                    batch = environment_shapes[offset:offset + args.discovery_batch]
                    attempted.update(batch)
                    env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
                    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "0"
                    _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                    by_shape = {
                        (record.get("shape"), record.get("dtype"), record.get("layout")): record
                        for record in records if record.get("status") == "DISCOVERED"
                    }
                    for item in batch:
                        _, _, m, n, k = item
                        shape = f"M{m}_N{n}_K{k}_{layout}"
                        record = by_shape.get((shape, dtype, layout))
                        if record is None or record.get("branch") != target or item in selected_keys:
                            continue
                        selected.append(item + (target,))
                        selected_keys.add(item)
                        counts[target] += 1
                        accepted += 1
                        if accepted >= tier_quota or counts[target] >= args.quota:
                            break

        # A narrow selector may not exist in all three size tiers.  Preserve
        # the large-to-small preference, but do not leave the branch below 20
        # when additional untried legal candidates remain in another tier.
        if counts[target] < args.quota:
            by_environment = collections.defaultdict(list)
            for item in ordered:
                if item not in attempted and item not in selected_keys:
                    by_environment[(item[0], item[1])].append(item)
            for dtype, layout in sorted(by_environment):
                environment_shapes = by_environment[(dtype, layout)]
                for offset in range(0, len(environment_shapes), args.discovery_batch):
                    if counts[target] >= args.quota:
                        break
                    batch = environment_shapes[offset:offset + args.discovery_batch]
                    env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
                    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "0"
                    _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                    by_shape = {
                        (record.get("shape"), record.get("dtype"), record.get("layout")): record
                        for record in records if record.get("status") == "DISCOVERED"
                    }
                    for item in batch:
                        _, _, m, n, k = item
                        shape = f"M{m}_N{n}_K{k}_{layout}"
                        record = by_shape.get((shape, dtype, layout))
                        if record is None or record.get("branch") != target or item in selected_keys:
                            continue
                        selected.append(item + (target,))
                        selected_keys.add(item)
                        counts[target] += 1
                        if counts[target] >= args.quota:
                            break

    branch_index = {branch: index for index, branch in enumerate(REMAINING_CORE_SWEEP_BRANCHES)}
    selected.sort(key=lambda item: (branch_index[item[5]], -(item[2] * item[3] * item[4]),
                                    -item[2], -item[3], -item[4]))
    write_selected(args.selected, selected)
    missing = {branch: args.quota - counts[branch] for branch in REMAINING_CORE_SWEEP_BRANCHES
               if counts[branch] < args.quota}
    print(json.dumps({"selection": "complete" if not missing else "partial",
                      "counts": {branch: counts[branch] for branch in REMAINING_CORE_SWEEP_BRANCHES},
                      "missing": missing,
                      "repo_only_fresh_unreachable": [branch for branch in REPO_ONLY_BRANCHES
                                                       if counts[branch] < args.quota],
                      "source_unreachable": [branch for branch in SOURCE_UNREACHABLE_BRANCHES
                                             if counts[branch] < args.quota]},
                     separators=(",", ":")), file=sys.stderr)
    return 0 if selected else 3


def expand_witness(pool, queued, item, branch):
    """Add a bounded local stencil around a route proven by the real selector."""
    dtype, layout, m, n, k = item
    if branch == "AL1_FULL_LOAD":
        m_delta = (-4, -2, -1, 1, 2, 4)
        n_delta = (-64, -32, -16, 16, 32, 64)
        k_delta = (-1024, -512, -128, 128, 512, 1024)
    elif branch.startswith("BL1_FULL_LOAD"):
        m_delta = (-2048, -1024, -512, -256, -128, 128, 256, 512, 1024, 2048)
        n_delta = (-32, -16, 16, 32)
        k_delta = (-16, -8, 8, 16)
    elif "SPLIT_K" in branch:
        m_delta = (-512, -256, -128, -64, -16, 16, 64, 128, 256, 512)
        n_delta = (-512, -256, -128, -64, -16, 16, 64, 128, 256, 512)
        k_delta = (-2048, -1024, -640, -512, -128, 128, 512, 640, 1024, 2048)
    else:
        return
    for delta in m_delta:
        add(pool, queued, dtype, layout, m + delta, n, k)
    for delta in n_delta:
        add(pool, queued, dtype, layout, m, n + delta, k)
    for delta in k_delta:
        add(pool, queued, dtype, layout, m, n, k + delta)


def runner_env(base, dtype, layout, mode, requested_core=None, discovery=False):
    env = dict(base)
    env["MATMUL_DATA_TYPE"] = dtype
    env["MATMUL_OUTPUT_DATA_TYPE"] = dtype
    env["MATMUL_A_TRANSPOSE"] = "1" if layout[0] == "T" else "0"
    env["MATMUL_B_TRANSPOSE"] = "1" if layout[1] == "T" else "0"
    env["MATMUL_V3_MEASUREMENT_MODE"] = mode
    env["MATMUL_V3_ONLY"] = "1"
    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "0"
    env.pop("MATMUL_V3_MEASUREMENT_PLAN", None)
    if discovery:
        env["MATMUL_V3_DISCOVERY_ONLY"] = "1"
    else:
        env.pop("MATMUL_V3_DISCOVERY_ONLY", None)
    if requested_core is None:
        env.pop("MATMUL_V3_FORCE_CORE_NUM", None)
    else:
        env["MATMUL_V3_FORCE_CORE_NUM"] = str(requested_core)
    return env


def invoke(runner, env, shapes, run_log):
    args = [runner]
    for _, _, m, n, k in shapes:
        args.extend((str(m), str(n), str(k)))
    completed = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, check=False)
    if completed.stderr:
        with open(run_log, "a", encoding="utf-8", errors="replace") as stream:
            stream.write(completed.stderr)
    records = []
    for line in completed.stdout.splitlines():
        if not line.startswith('{"shape":'):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return completed.returncode, records, completed.stderr


def read_selected(path):
    selected = []
    if not os.path.isfile(path):
        return selected
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 6:
                continue
            dtype, layout, m, n, k, branch = fields
            if dtype not in ("fp32", "fp16", "bf16") or layout not in ("NN", "NT", "TN", "TT"):
                continue
            if branch not in TARGET_BRANCHES:
                continue
            selected.append((dtype, layout, int(m), int(n), int(k), branch))
    return selected


def write_selected(path, selected):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        for dtype, layout, m, n, k, branch in selected:
            stream.write(f"{dtype}\t{layout}\t{m}\t{n}\t{k}\t{branch}\n")
    os.replace(temporary, path)


def discover(args):
    pool = candidate_pool()
    queued = {item for values in pool.values() for item in values}
    offsets = {key: 0 for key in pool}
    counts = collections.Counter()
    observed = collections.Counter()
    expanded = collections.Counter()
    witnesses = collections.defaultdict(list)
    loaded_selected = read_selected(args.selected)
    selected = []
    selected_mnk = set()
    for item in loaded_selected:
        _dtype, _layout, m, n, k, branch = item
        if branch not in MEASUREMENT_BRANCHES:
            continue
        mnk = (m, n, k)
        if mnk in selected_mnk or counts[branch] >= args.quota:
            continue
        selected_mnk.add(mnk)
        counts[branch] += 1
        selected.append(item)
    if all(counts[name] >= args.quota for name in OFFICIAL_QUOTA_BRANCHES):
        print(json.dumps({"discovery": "cache_reused", "attempted": 0,
                          "selected": len(selected),
                          "counts": {name: counts[name] for name in TARGET_BRANCHES},
                          "selector_limited": {"AL1_FULL_LOAD": counts["AL1_FULL_LOAD"],
                                               "BL1_FULL_LOAD": counts["BL1_FULL_LOAD"]}},
                         separators=(",", ":")), file=sys.stderr)
        return 0
    keys = sorted(pool)
    attempts = 0
    saved_count = len(selected)
    while keys and any(counts[name] < args.quota for name in OFFICIAL_QUOTA_BRANCHES):
        next_keys = []
        for key in keys:
            begin = offsets[key]
            raw_batch = pool[key][begin:begin + args.discovery_batch]
            if not raw_batch:
                continue
            offsets[key] += len(raw_batch)
            next_keys.append(key)
            batch = [item for item in raw_batch if (item[2], item[3], item[4]) not in selected_mnk]
            if not batch:
                continue
            dtype, layout = key
            env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
            rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
            attempts += len(batch)
            if rc != 0:
                continue
            by_shape = {(record.get("shape"), record.get("dtype"), record.get("layout")): record
                        for record in records if record.get("status") == "DISCOVERED"}
            for item in batch:
                _, _, m, n, k = item
                shape = f"M{m}_N{n}_K{k}_{layout}"
                record = by_shape.get((shape, dtype, layout))
                if record is None:
                    continue
                branch = record.get("branch")
                if branch in MEASUREMENT_BRANCHES:
                    observed[branch] += 1
                    if len(witnesses[branch]) < 3:
                        witnesses[branch].append({"dtype": dtype, "layout": layout,
                                                  "m": m, "n": n, "k": k})
                    if counts[branch] < args.quota and expanded[branch] < 12:
                        expand_witness(pool, queued, item, branch)
                        expanded[branch] += 1
                mnk = (m, n, k)
                if branch not in MEASUREMENT_BRANCHES or counts[branch] >= args.quota or mnk in selected_mnk:
                    continue
                selected.append(item + (branch,))
                selected_mnk.add(mnk)
                counts[branch] += 1
        keys = next_keys
        if len(selected) != saved_count:
            write_selected(args.selected, selected)
            saved_count = len(selected)

    write_selected(args.selected, selected)
    missing_required = {
        name: args.quota - counts[name]
        for name in OFFICIAL_QUOTA_BRANCHES if counts[name] < args.quota
    }
    if not selected:
        print(json.dumps({"fatal": "no_official_v3_shapes_discovered", "attempted": attempts},
                         separators=(",", ":")), file=sys.stderr)
        return 3
    print(json.dumps({"discovery": "complete" if not missing_required else "partial_measurement_continues",
                      "attempted": attempts, "selected": len(selected),
                      "counts": {name: counts[name] for name in TARGET_BRANCHES},
                      "selector_limited": {"AL1_FULL_LOAD": counts["AL1_FULL_LOAD"],
                                           "BL1_FULL_LOAD": counts["BL1_FULL_LOAD"]},
                      "missing_required": missing_required,
                      "witnesses": {name: witnesses[name] for name in missing_required
                                    if witnesses[name]}}, separators=(",", ":")), file=sys.stderr)
    return 0


def load_selected(path, branch_quota, allowed_branches=MEASUREMENT_BRANCHES):
    groups = collections.defaultdict(list)
    branch_counts = collections.Counter()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            dtype, layout, m, n, k, branch = line.rstrip("\n").split("\t")
            if branch != "USER" and branch not in allowed_branches:
                continue
            if branch in allowed_branches and branch_counts[branch] >= branch_quota:
                continue
            branch_counts[branch] += 1
            groups[(dtype, layout)].append((dtype, layout, int(m), int(n), int(k)))
    return groups


def measurement_key(record):
    return (record.get("shape"), record.get("dtype"), record.get("layout"),
            record.get("mode"), record.get("requested_core"))


def load_checkpoint(path):
    records = {}
    if not path or not os.path.isfile(path):
        return records
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not record.get("shape"):
                continue
            records[measurement_key(record)] = record
    return records


def append_checkpoint(path, records):
    if not path or not records:
        return
    with open(path, "a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()


def all_io_measurement_key(record):
    return (record.get("shape"), record.get("input_dtype", record.get("dtype")),
            record.get("output_dtype"), record.get("layout"),
            record.get("mode"), record.get("requested_core"))


def load_all_io_checkpoint(path):
    records = {}
    if not path or not os.path.isfile(path):
        return records
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not record.get("shape"):
                continue
            records[all_io_measurement_key(record)] = record
    return records


def measure_all_io_core_sweep(args):
    """Measure one complete 4..20 response curve before changing shape."""
    selected = read_all_io_selected(args.selected)
    order = {combination: index for index, combination in enumerate(ALL_IO_COMBINATIONS)}
    selected.sort(key=lambda item: (
        order[(item[6], item[0], item[1])], item[3] * item[4] * item[5],
        item[3], item[4], item[5]))
    checkpoint = load_all_io_checkpoint(args.checkpoint)
    attempted = collections.Counter()
    succeeded = collections.Counter()
    failed = collections.Counter()

    def store(record, combination):
        checkpoint[all_io_measurement_key(record)] = record
        append_checkpoint(args.checkpoint, [record])
        print(json.dumps(record, separators=(",", ":")), flush=True)
        attempted[combination] += 1
        if record.get("status") == "OK" and record.get("correctness") == "PASS":
            succeeded[combination] += 1
        else:
            failed[combination] += 1

    def synthetic_error(shape, input_dtype, output_dtype, layout, branch,
                        mode, core, rc, stderr):
        return {
            "shape": shape,
            "dtype": input_dtype,
            "input_dtype": input_dtype,
            "output_dtype": output_dtype,
            "layout": layout,
            "branch": branch,
            "mode": mode,
            "experiment": "fixed_tiling_core_sweep",
            "requested_core": core,
            "latency_ms": None,
            "status": "RUNNER_ERROR",
            "correctness": "NOT_CHECKED",
            "failure_stage": "runner_no_record",
            "rc": rc,
            "detail": " ".join(stderr.strip().split())[-1000:],
        }

    for item in selected:
        input_dtype, output_dtype, layout, m, n, k, expected_branch = item
        combination = (expected_branch, input_dtype, output_dtype)
        shape = f"M{m}_N{n}_K{k}_{layout}"
        # Core 20 is the unchanged official-count reference.  Measure exactly
        # the requested 4..20 curve without adding redundant pre/post runs.
        modes = [("core_sweep", core) for core in range(4, 21)]
        pending = [(mode, core) for mode, core in modes
                   if (shape, input_dtype, output_dtype, layout, mode, core) not in checkpoint]
        if not pending:
            continue

        # Fast path: one process executes the complete ordered curve.  The
        # C++ loop emits an independent record for every core and continues
        # after ordinary measurement errors.
        env = all_io_runner_env(
            os.environ, input_dtype, output_dtype, layout, "core_response")
        env["MATMUL_V3_MEASUREMENT_PLAN"] = ",".join(
            mode if core is None else str(core) for mode, core in pending)
        rc, records, stderr = invoke(
            args.runner, env, [(input_dtype, layout, m, n, k)], args.run_log)
        by_key = {}
        for record in records:
            if record.get("shape") != shape:
                continue
            key = (record.get("mode"), record.get("requested_core"))
            by_key[key] = record

        retry = []
        for mode, core in pending:
            record = by_key.get((mode, core))
            if (record is None or record.get("status") != "OK" or
                    record.get("correctness") != "PASS"):
                retry.append((mode, core, record))
                continue
            if record.get("branch") != expected_branch:
                record["expected_branch"] = expected_branch
                record["status"] = "ROUTE_CHANGED"
                retry.append((mode, core, record))
                continue
            store(record, combination)

        # Slow recovery path: every missing/failed core gets a fresh process.
        # A repeated failure is checkpointed as that core's final skipped
        # result, then the next core of the same shape still runs.
        for mode, core, first_record in retry:
            token = mode if core is None else str(core)
            retry_env = all_io_runner_env(
                os.environ, input_dtype, output_dtype, layout, "core_response")
            retry_env["MATMUL_V3_MEASUREMENT_PLAN"] = token
            retry_rc, retry_records, retry_stderr = invoke(
                args.runner, retry_env, [(input_dtype, layout, m, n, k)], args.run_log)
            matching = [record for record in retry_records
                        if record.get("shape") == shape and record.get("mode") == mode and
                        record.get("requested_core") == core]
            if matching:
                record = matching[-1]
            elif first_record is not None:
                record = first_record
            else:
                record = synthetic_error(
                    shape, input_dtype, output_dtype, layout, expected_branch,
                    mode, core, retry_rc if retry_rc != 0 else rc,
                    retry_stderr or stderr)
            if record.get("branch") not in (expected_branch, "UNKNOWN", None):
                record["expected_branch"] = expected_branch
                record["status"] = "ROUTE_CHANGED"
            store(record, combination)

    summary = {}
    for branch, input_dtype, output_dtype in ALL_IO_COMBINATIONS:
        combination = (branch, input_dtype, output_dtype)
        prefix = f"{branch}:{input_dtype}->{output_dtype}"
        summary[prefix] = {
            "selected_shapes": sum(
                1 for item in selected
                if (item[6], item[0], item[1]) == combination),
            "new_attempts": attempted[combination],
            "new_successes": succeeded[combination],
            "new_failures": failed[combination],
        }
    print(json.dumps({"all_io_core_sweep": "finished_available_shapes",
                      "combinations": len(ALL_IO_COMBINATIONS),
                      "summary": summary}, separators=(",", ":")), file=sys.stderr)
    return 0


def measure(args):
    groups = load_selected(args.selected, args.quota)
    # The non-Split-K pass measures the useful unresolved range 4..20.  Each
    # shape completes the full range before advancing.
    modes = [("official_pre", None)] + [("core_sweep", core) for core in range(4, 21)] + [
        ("official_post", None)]
    checkpoint = load_checkpoint(args.checkpoint)
    for (dtype, layout), shapes in sorted(groups.items()):
        for item in shapes:
            _, _, m, n, k = item
            shape = f"M{m}_N{n}_K{k}_{layout}"
            pending_modes = [
                (mode, core) for mode, core in modes
                if (shape, dtype, layout, mode, core) not in checkpoint
            ]
            if not pending_modes:
                continue
            plan = [mode if core is None else str(core) for mode, core in pending_modes]
            env = runner_env(os.environ, dtype, layout, "core_response")
            env["MATMUL_V3_MEASUREMENT_PLAN"] = ",".join(plan)
            rc, records, stderr = invoke(args.runner, env, [item], args.run_log)
            append_checkpoint(args.checkpoint, records)
            for record in records:
                checkpoint[measurement_key(record)] = record
                print(json.dumps(record, separators=(",", ":")), flush=True)
            if rc != 0:
                detail = " ".join(stderr.strip().split())[-2000:]
                print(json.dumps({"shape": shape, "dtype": dtype, "layout": layout,
                                  "measurement_plan": plan, "status": "RUNNER_ERROR", "rc": rc,
                                  "detail": detail}, separators=(",", ":")), flush=True)
    return 0


def hardware_skip_reason(branch, requested_core, official_record):
    tiling = official_record.get("tiling") if isinstance(official_record, dict) else None
    if not isinstance(tiling, dict):
        return None
    compile_core_num = tiling.get("compile_core_num")
    if isinstance(compile_core_num, int) and requested_core > compile_core_num:
        return f"requested_core_exceeds_compile_core_num:{compile_core_num}"
    if branch != "MULTI_CORE_SPLIT_K":
        return None
    packet = tiling.get("packet")
    if not isinstance(packet, dict):
        return None
    try:
        m_count = (int(packet["M"]) + int(packet["single_core_m"]) - 1) // int(packet["single_core_m"])
        n_count = (int(packet["N"]) + int(packet["single_core_n"]) - 1) // int(packet["single_core_n"])
        k_count = (int(packet["Ka"]) + int(packet["single_core_k"]) - 1) // int(packet["single_core_k"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    task_count = m_count * n_count * k_count
    # This kernel maps one block id to exactly one (M,N,K) task and has no
    # grid-stride loop.  Fewer blocks omit work; extra blocks index past K.
    if requested_core != task_count:
        return f"multi_core_split_k_requires_exact_task_count:{task_count}"
    return None


def measure_remaining(args):
    selected = [item for item in read_selected(args.selected)
                if item[5] in REMAINING_CORE_SWEEP_BRANCHES]
    branch_index = {branch: index for index, branch in enumerate(REMAINING_CORE_SWEEP_BRANCHES)}
    selected.sort(key=lambda item: (branch_index[item[5]], -(item[2] * item[3] * item[4]),
                                    -item[2], -item[3], -item[4]))
    checkpoint = load_checkpoint(args.checkpoint)

    def store(records):
        fresh = []
        for record in records:
            key = measurement_key(record)
            if key in checkpoint:
                continue
            checkpoint[key] = record
            fresh.append(record)
        append_checkpoint(args.checkpoint, fresh)
        for record in fresh:
            print(json.dumps(record, separators=(",", ":")), flush=True)

    def run_plan(item, tokens):
        dtype, layout, _m, _n, _k, _branch = item
        env = runner_env(os.environ, dtype, layout, "core_response")
        env["MATMUL_V3_MEASUREMENT_PLAN"] = ",".join(tokens)
        return invoke(args.runner, env, [item[:5]], args.run_log)

    for item in selected:
        dtype, layout, m, n, k, branch = item
        shape = f"M{m}_N{n}_K{k}_{layout}"
        modes = [("official_pre", None)] + [("core_sweep", core) for core in range(4, 21)] + [
            ("official_post", None)]

        if branch == "MULTI_CORE_SPLIT_K":
            official_key = (shape, dtype, layout, "official_pre", None)
            if official_key not in checkpoint:
                _rc, records, _stderr = run_plan(item, ["official_pre"])
                store(records)
            official_record = checkpoint.get(official_key, {})
            skipped = []
            for core in range(4, 21):
                key = (shape, dtype, layout, "core_sweep", core)
                if key in checkpoint:
                    continue
                reason = hardware_skip_reason(branch, core, official_record)
                if reason is None:
                    continue
                skipped.append({"shape": shape, "dtype": dtype, "layout": layout,
                                "branch": branch, "mode": "core_sweep",
                                "requested_core": core, "latency_ms": None,
                                "status": "SKIPPED_HARDWARE_RULE",
                                "correctness": "NOT_RUN", "skip_reason": reason})
            store(skipped)

        pending = [(mode, core) for mode, core in modes
                   if (shape, dtype, layout, mode, core) not in checkpoint]
        if not pending:
            continue
        tokens = [mode if core is None else str(core) for mode, core in pending]
        rc, records, stderr = run_plan(item, tokens)
        store(records)

        # A per-core kernel error is already a normal JSON record and the C++
        # runner continues with the next core.  Only a process-level failure
        # can hide later records; retry exactly those missing modes one by one.
        missing = [(mode, core) for mode, core in pending
                   if (shape, dtype, layout, mode, core) not in checkpoint]
        for mode, core in missing:
            token = mode if core is None else str(core)
            retry_rc, retry_records, retry_stderr = run_plan(item, [token])
            store(retry_records)
            key = (shape, dtype, layout, mode, core)
            if key in checkpoint:
                continue
            detail = " ".join((retry_stderr or stderr).strip().split())[-1000:]
            store([{"shape": shape, "dtype": dtype, "layout": layout,
                    "branch": branch, "mode": mode, "requested_core": core,
                    "latency_ms": None, "status": "RUNNER_ERROR",
                    "correctness": "NOT_CHECKED", "failure_stage": "runner_no_record",
                    "rc": retry_rc if retry_rc != 0 else rc, "detail": detail}])
    return 0


def compare(args):
    groups = load_selected(args.selected, args.quota, SHRINK_ENABLED_BRANCHES)
    emitted = 0
    branch_counts = collections.Counter()
    for (dtype, layout), shapes in sorted(groups.items()):
        for offset in range(0, len(shapes), args.batch_size):
            batch = shapes[offset:offset + args.batch_size]
            env = runner_env(os.environ, dtype, layout, "core_shrink_compare")
            #NEW: Symmetric ABBA sampling removes the old bias that compared
            # one shrink sample with the faster of two official samples.
            env["MATMUL_V3_MEASUREMENT_PLAN"] = \
                "official_pre,shrink_pre,shrink_post,official_post"
            _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
            by_key = {(record.get("shape"), record.get("mode")): record for record in records}
            for _, _, m, n, k in batch:
                shape = f"M{m}_N{n}_K{k}_{layout}"
                pre = by_key.get((shape, "official_pre"))
                shrink_pre = by_key.get((shape, "shrink_pre"))
                shrink_post = by_key.get((shape, "shrink_post"))
                post = by_key.get((shape, "official_post"))
                quartet = (pre, shrink_pre, shrink_post, post)
                if any(record is None or record.get("status") != "OK" or
                       record.get("correctness") != "PASS" or
                       not isinstance(record.get("latency_ms"), (int, float))
                       for record in quartet):
                    continue
                branch = shrink_pre.get("branch")
                if any(record.get("branch") != branch for record in quartet):
                    continue
                if branch not in SHRINK_ENABLED_BRANCHES or branch_counts[branch] >= args.quota:
                    continue
                if (shrink_pre.get("actual_core", 0) >= shrink_pre.get("official_core", 0) or
                        shrink_post.get("actual_core", 0) >= shrink_post.get("official_core", 0)):
                    continue
                official_cores = {record.get("official_core") for record in quartet}
                shrink_cores = {shrink_pre.get("actual_core"), shrink_post.get("actual_core")}
                if (len(official_cores) != 1 or len(shrink_cores) != 1 or
                        not all(isinstance(core, int) and core > 0
                                for core in official_cores | shrink_cores)):
                    continue
                original_latency = (float(pre["latency_ms"]) + float(post["latency_ms"])) / 2.0
                shrink_latency = (float(shrink_pre["latency_ms"]) + float(shrink_post["latency_ms"])) / 2.0
                output = {
                    "shape": shape,
                    "branch": branch,
                    "official_core": official_cores.pop(),
                    "shrinked_core": shrink_cores.pop(),
                    "shrinked_latency": f"{shrink_latency:.9f}",
                    "original_latency": f"{original_latency:.9f}",
                }
                print(json.dumps(output, separators=(",", ":")), flush=True)
                emitted += 1
                branch_counts[branch] += 1
    return 0 if emitted else 3


def compare_core_validation(args):
    groups = collections.defaultdict(list)
    for dtype, layout, m, n, k, branch in read_selected(args.selected):
        if branch in SHRINK_ENABLED_BRANCHES:
            groups[(branch, dtype, layout)].append((dtype, layout, m, n, k))

    branch_counts = collections.Counter()
    dtype_counts = collections.Counter()
    for expected_branch in SHRINK_ENABLED_BRANCHES:
        environments = sorted(key for key in groups if key[0] == expected_branch)
        offsets = {key: 0 for key in environments}
        supported_dtypes = SHRINK_BRANCH_DTYPES[expected_branch]
        dtype_targets = balanced_dtype_targets(expected_branch, args.quota)
        while branch_counts[expected_branch] < args.quota:
            made_progress = False
            for _branch, dtype, layout in environments:
                key = (expected_branch, dtype, layout)
                if dtype_counts[(expected_branch, dtype)] >= dtype_targets.get(dtype, 0):
                    continue
                offset = offsets[key]
                shapes = groups[key]
                if offset >= len(shapes):
                    continue
                made_progress = True
                offsets[key] += args.batch_size
                if branch_counts[expected_branch] >= args.quota:
                    break
                batch = shapes[offset:offset + args.batch_size]
                env = runner_env(os.environ, dtype, layout, "core_shrink_compare")
                env["MATMUL_V3_MEASUREMENT_PLAN"] = \
                    "official_pre,shrink_pre,shrink_post,official_post"
                _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
                by_key = {(record.get("shape"), record.get("mode")): record for record in records}
                for _, _, m, n, k in batch:
                    if (branch_counts[expected_branch] >= args.quota or
                            dtype_counts[(expected_branch, dtype)] >= dtype_targets[dtype]):
                        break
                    shape = f"M{m}_N{n}_K{k}_{layout}"
                    pre = by_key.get((shape, "official_pre"))
                    shrink_pre = by_key.get((shape, "shrink_pre"))
                    shrink_post = by_key.get((shape, "shrink_post"))
                    post = by_key.get((shape, "official_post"))
                    quartet = (pre, shrink_pre, shrink_post, post)
                    if any(record is None or record.get("status") != "OK" or
                           record.get("correctness") != "PASS" or
                           not isinstance(record.get("latency_ms"), (int, float))
                           for record in quartet):
                        continue
                    if any(record.get("branch") != expected_branch for record in quartet):
                        continue
                    official_cores = {record.get("official_core") for record in quartet}
                    shrink_cores = {shrink_pre.get("actual_core"), shrink_post.get("actual_core")}
                    if (len(official_cores) != 1 or len(shrink_cores) != 1 or
                            not all(isinstance(core, int) and core > 0
                                    for core in official_cores | shrink_cores)):
                        continue
                    official_core = official_cores.pop()
                    shrinked_core = shrink_cores.pop()
                    if not valid_core_decision(expected_branch, official_core, shrinked_core):
                        continue
                    original_latency = (float(pre["latency_ms"]) + float(post["latency_ms"])) / 2.0
                    shrink_latency = \
                        (float(shrink_pre["latency_ms"]) + float(shrink_post["latency_ms"])) / 2.0
                    output = {
                        "shape": shape,
                        "dtype": dtype,
                        "layout": layout,
                        "branch": expected_branch,
                        "decision": "retain_official" if shrinked_core == official_core else "shrink",
                        "official_core": official_core,
                        "shrinked_core": shrinked_core,
                        "shrinked_latency": f"{shrink_latency:.9f}",
                        "original_latency": f"{original_latency:.9f}",
                    }
                    print(json.dumps(output, separators=(",", ":")), flush=True)
                    branch_counts[expected_branch] += 1
                    dtype_counts[(expected_branch, dtype)] += 1
            if not made_progress:
                break

    missing = {branch: args.quota - branch_counts[branch] for branch in SHRINK_ENABLED_BRANCHES
               if branch_counts[branch] < args.quota}
    print(json.dumps({"core_validation": "complete" if not missing else "incomplete",
                      "counts": {branch: branch_counts[branch] for branch in SHRINK_ENABLED_BRANCHES},
                      "dtype_counts": {
                          branch: {dtype: dtype_counts[(branch, dtype)]
                                   for dtype in SHRINK_BRANCH_DTYPES[branch]}
                          for branch in SHRINK_ENABLED_BRANCHES
                      },
                      "missing": missing}, separators=(",", ":")), file=sys.stderr)
    return 0 if not missing else 4


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runner", required=True)
    common.add_argument("--selected", required=True)
    common.add_argument("--run-log", required=True)
    discover_parser = sub.add_parser("discover", parents=[common])
    discover_parser.add_argument("--quota", type=int, default=20)
    discover_parser.add_argument("--discovery-batch", type=int, default=48)
    measure_parser = sub.add_parser("measure", parents=[common])
    measure_parser.add_argument("--checkpoint", required=True)
    measure_parser.add_argument("--quota", type=int, default=20)
    select_parser = sub.add_parser("select-shrink", parents=[common])
    select_parser.add_argument("--quota", type=int, default=30)
    select_parser.add_argument("--discovery-batch", type=int, default=64)
    core_validation_parser = sub.add_parser("select-shrink-core-validation", parents=[common])
    core_validation_parser.add_argument("--discovery-batch", type=int, default=64)
    remaining_select_parser = sub.add_parser("select-remaining", parents=[common])
    remaining_select_parser.add_argument("--quota", type=int, default=20)
    remaining_select_parser.add_argument("--discovery-batch", type=int, default=64)
    remaining_measure_parser = sub.add_parser("measure-remaining", parents=[common])
    remaining_measure_parser.add_argument("--checkpoint", required=True)
    compare_parser = sub.add_parser("compare", parents=[common])
    compare_parser.add_argument("--quota", type=int, default=1000)
    compare_parser.add_argument("--batch-size", type=int, default=8)
    core_compare_parser = sub.add_parser("compare-core-validation", parents=[common])
    core_compare_parser.add_argument("--quota", type=int, default=30)
    core_compare_parser.add_argument("--batch-size", type=int, default=8)
    all_io_select_parser = sub.add_parser("select-all-io-core-sweep", parents=[common])
    all_io_select_parser.add_argument("--quota", type=int, default=50)
    all_io_select_parser.add_argument("--discovery-batch", type=int, default=64)
    all_io_measure_parser = sub.add_parser("measure-all-io-core-sweep", parents=[common])
    all_io_measure_parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "discover":
        return discover(args)
    if args.command == "measure":
        return measure(args)
    if args.command == "select-shrink":
        return select_shrink(args)
    if args.command == "select-shrink-core-validation":
        return select_shrink_core_validation(args)
    if args.command == "select-remaining":
        return select_remaining(args)
    if args.command == "measure-remaining":
        return measure_remaining(args)
    if args.command == "compare-core-validation":
        return compare_core_validation(args)
    if args.command == "select-all-io-core-sweep":
        return select_all_io_core_sweep(args)
    if args.command == "measure-all-io-core-sweep":
        return measure_all_io_core_sweep(args)
    return compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
