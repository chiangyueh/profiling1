#!/usr/bin/env python3

import argparse
import collections
import json
import os
import random
import subprocess
import sys


# The 17 routes below are exactly the combinations that the Ascend 910B
# fresh-shape selector can produce without GetTilingFromRepo/AOE state.
TARGET_BRANCHES = (
    "BASE",
    "BASE_ND2NZ",
    "AL1_FULL_LOAD",
    "BL1_FULL_LOAD",
    "BL1_FULL_LOAD_ND2NZ",
    "BL1_FULL_LOAD_FIXPIPE",
    "BL1_FULL_LOAD_FIXPIPE_ND2NZ",
    "BL1_FULL_LOAD_VEC_NZ2ND",
    "SINGLE_CORE_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_ND2NZ",
    "SINGLE_CORE_NKM_SPLIT_K",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1",
    "SINGLE_CORE_SPLIT_K_GM_TO_L1_ND2NZ",
    "DETERMINISTIC_SPLIT_K",
    "DETERMINISTIC_SPLIT_K_ND2NZ",
    "DETERMINISTIC_SPLIT_K_VEC_NZ2ND",
    "DETERMINISTIC_SPLIT_K_VEC_NZ2ND_ND2NZ",
)


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
    al1_k = (4096, 5120, 5760, 6144, 6400, 6656, 6784, 7040,
              7168, 8192, 9216, 10240, 12288, 14336, 15360, 16384)
    for index in range(4096):
        add(pool, seen, "fp32", "NT",
            1 + index % 16,
            80 + 16 * ((index // 16) % 16),
            al1_k[(index // 256) % len(al1_k)])

    # Plain BL1 full-load needs M > 16*max(K,N), K<=256 and an on-the-fly
    # supported B.  Values below avoid the earlier fixpipe predicate.  The
    # first K set is also on-the-fly supported for A (plain BL1); the second
    # deliberately makes A use head ND2NZ while B remains on-the-fly.
    bl1_n = (32, 64, 128, 192, 256, 384)
    bl1_plain_k = (8, 16, 24, 32, 40, 48, 56, 64, 96)
    bl1_nd2nz_k = (17, 18, 19, 21, 25, 33, 41, 49, 57, 65, 73, 81)
    for index in range(240):
        m = 9216 + 128 * index
        for layout in ("NN", "NT"):
            add(pool, seen, "fp32", layout, m,
                bl1_n[index % len(bl1_n)], bl1_plain_k[(index * 5) % len(bl1_plain_k)])
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
    for i in range(5000):
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


def runner_env(base, dtype, layout, mode, requested_core=None, discovery=False):
    env = dict(base)
    env["MATMUL_DATA_TYPE"] = dtype
    env["MATMUL_A_TRANSPOSE"] = "1" if layout[0] == "T" else "0"
    env["MATMUL_B_TRANSPOSE"] = "1" if layout[1] == "T" else "0"
    env["MATMUL_V3_MEASUREMENT_MODE"] = mode
    env["MATMUL_V3_ONLY"] = "1"
    env["MATMUL_V3_SHRINK_IDLE_CORES"] = "0"
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
    return completed.returncode, records


def discover(args):
    pool = candidate_pool()
    offsets = {key: 0 for key in pool}
    counts = collections.Counter()
    observed = collections.Counter()
    witnesses = collections.defaultdict(list)
    selected = []
    selected_mnk = set()
    keys = sorted(pool)
    attempts = 0
    while keys and any(counts[name] < args.quota for name in TARGET_BRANCHES):
        next_keys = []
        for key in keys:
            begin = offsets[key]
            batch = pool[key][begin:begin + args.discovery_batch]
            if not batch:
                continue
            offsets[key] += len(batch)
            next_keys.append(key)
            dtype, layout = key
            env = runner_env(os.environ, dtype, layout, "discovery", discovery=True)
            rc, records = invoke(args.runner, env, batch, args.run_log)
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
                if branch in TARGET_BRANCHES:
                    observed[branch] += 1
                    if len(witnesses[branch]) < 3:
                        witnesses[branch].append({"dtype": dtype, "layout": layout,
                                                  "m": m, "n": n, "k": k})
                mnk = (m, n, k)
                if branch not in TARGET_BRANCHES or counts[branch] >= args.quota or mnk in selected_mnk:
                    continue
                selected.append(item + (branch,))
                selected_mnk.add(mnk)
                counts[branch] += 1
        keys = next_keys

    missing = {name: args.quota - counts[name] for name in TARGET_BRANCHES if counts[name] < args.quota}
    if missing:
        print(json.dumps({"fatal": "branch_discovery_exhausted", "attempted": attempts,
                          "counts": {name: counts[name] for name in TARGET_BRANCHES},
                          "observed": {name: observed[name] for name in TARGET_BRANCHES},
                          "witnesses": {name: witnesses[name] for name in TARGET_BRANCHES
                                        if witnesses[name]},
                          "missing": missing}, separators=(",", ":")), file=sys.stderr)
        return 3
    with open(args.selected, "w", encoding="utf-8") as stream:
        for dtype, layout, m, n, k, branch in selected:
            stream.write(f"{dtype}\t{layout}\t{m}\t{n}\t{k}\t{branch}\n")
    return 0


def load_selected(path):
    groups = collections.defaultdict(list)
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            dtype, layout, m, n, k, _branch = line.rstrip("\n").split("\t")
            groups[(dtype, layout)].append((dtype, layout, int(m), int(n), int(k)))
    return groups


def measure(args):
    groups = load_selected(args.selected)
    core_order = (20, 1, 19, 2, 18, 3, 17, 4, 16, 5, 15, 6, 14, 7, 13, 8, 12, 9, 11, 10)
    modes = [("official_pre", None)] + [("core_sweep", core) for core in core_order] + [("official_post", None)]
    for mode, core in modes:
        for (dtype, layout), shapes in sorted(groups.items()):
            for begin in range(0, len(shapes), args.measurement_batch):
                batch = shapes[begin:begin + args.measurement_batch]
                env = runner_env(os.environ, dtype, layout, mode, requested_core=core)
                rc, records = invoke(args.runner, env, batch, args.run_log)
                for record in records:
                    print(json.dumps(record, separators=(",", ":")), flush=True)
                if rc != 0:
                    print(json.dumps({"dtype": dtype, "layout": layout, "mode": mode,
                                      "requested_core": core, "status": "RUNNER_ERROR", "rc": rc},
                                     separators=(",", ":")), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runner", required=True)
    common.add_argument("--selected", required=True)
    common.add_argument("--run-log", required=True)
    discover_parser = sub.add_parser("discover", parents=[common])
    discover_parser.add_argument("--quota", type=int, default=100)
    discover_parser.add_argument("--discovery-batch", type=int, default=48)
    measure_parser = sub.add_parser("measure", parents=[common])
    measure_parser.add_argument("--measurement-batch", type=int, default=160)
    args = parser.parse_args()
    return discover(args) if args.command == "discover" else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
