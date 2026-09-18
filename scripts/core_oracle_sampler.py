#!/usr/bin/env python3

import argparse
import collections
import json
import os
import random
import subprocess
import sys


# The 17 routes below are exactly the combinations that the Ascend 910B
# fresh-shape tiler can produce without GetTilingFromRepo/AOE state.  The
# outer aclnn MatMul dispatcher does not necessarily admit every route.
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

# The current pass is intentionally restricted to non-Split-K routes.  The
# completed Split-K response curves remain in the checkpoint, but are neither
# selected nor printed again.
MEASUREMENT_BRANCHES = tuple(name for name in TARGET_BRANCHES if "SPLIT_K" not in name)

# Exhaustive official-dispatch discovery on CANN 8.5 found no reachable
# BL1_FULL_LOAD packet and only 21 distinct AL1 packets.  Those two routes
# remain recorded when observed, but cannot prevent the reachable non-Split-K
# route datasets from being measured.
OFFICIAL_QUOTA_BRANCHES = tuple(
    name for name in MEASUREMENT_BRANCHES
    if name not in ("AL1_FULL_LOAD", "BL1_FULL_LOAD")
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


def load_selected(path, branch_quota):
    groups = collections.defaultdict(list)
    branch_counts = collections.Counter()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            dtype, layout, m, n, k, branch = line.rstrip("\n").split("\t")
            if branch != "USER" and branch not in MEASUREMENT_BRANCHES:
                continue
            if branch in MEASUREMENT_BRANCHES and branch_counts[branch] >= branch_quota:
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
    args = parser.parse_args()
    return discover(args) if args.command == "discover" else measure(args)


if __name__ == "__main__":
    raise SystemExit(main())
