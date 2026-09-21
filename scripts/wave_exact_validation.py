#!/usr/bin/env python3

import argparse
import collections
import json
import os

from core_oracle_sampler import invoke, runner_env


TARGET_BRANCH = "WAVE_EXACT_BASE_ND2NZ"


def candidates():
    values = []
    seen = set()
    for dtype_index, dtype in enumerate(("fp16", "bf16")):
        for index in range(720):
            m = 481 + 64 * ((index * 7 + dtype_index * 3) % 33)
            n = 497 + 16 * ((index * 11 + index // 17 + dtype_index * 5) % 173)
            k = 27008 + 128 * ((index * 13 + index // 19 + dtype_index * 7) % 49)
            if m % 2 == 0:
                m += 1
            if n % 2 == 0:
                n += 1
            item = (dtype, "NN", m, n, k)
            if item in seen:
                continue
            element_bytes = 2
            if (m * k + k * n + m * n) * element_bytes > 300 * 1024 * 1024:
                continue
            seen.add(item)
            values.append(item)
    return values


def discover(args):
    grouped = collections.defaultdict(list)
    for item in candidates():
        grouped[(item[0], item[1])].append(item)
    admitted = []
    target = max(args.quota * 4, args.quota + 12)
    for (dtype, layout), values in sorted(grouped.items()):
        for offset in range(0, len(values), args.discovery_batch):
            batch = values[offset:offset + args.discovery_batch]
            env = runner_env(os.environ, dtype, layout, "wave_exact_discovery", discovery=True)
            env["MATMUL_V3_ENABLE_WAVE_EXACT_ND2NZ"] = "1"
            rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
            if rc != 0:
                continue
            for record in records:
                if (record.get("status") == "DISCOVERED" and
                        record.get("branch") == TARGET_BRANCH and
                        isinstance(record.get("tiling"), dict)):
                    shape = record["shape"]
                    fields = shape.split("_")
                    admitted.append((
                        dtype, layout, int(fields[0][1:]), int(fields[1][1:]),
                        int(fields[2][1:]), record))
            if sum(item[0] == dtype for item in admitted) >= target // 2:
                break

    if len(admitted) < args.quota:
        print(json.dumps({
            "fatal": "wave_exact_route_quota_not_reached",
            "admitted": len(admitted), "quota": args.quota,
        }, separators=(",", ":")), flush=True)
        return []

    admitted.sort(key=lambda item: item[2] * item[3] * item[4])
    cut1 = (len(admitted) + 2) // 3
    cut2 = (2 * len(admitted) + 2) // 3
    tiers = [admitted[:cut1], admitted[cut1:cut2], admitted[cut2:]]
    selected = []
    dtype_counts = collections.Counter()
    tier_counts = collections.Counter()
    dtype_quota = {"fp16": args.quota // 2, "bf16": args.quota - args.quota // 2}
    tier_quota = {
        0: args.quota // 3,
        1: args.quota // 3,
        2: args.quota - 2 * (args.quota // 3),
    }
    while len(selected) < args.quota:
        progressed = False
        for tier_index, tier in enumerate(tiers):
            if tier_counts[tier_index] >= tier_quota[tier_index]:
                continue
            for item in tier:
                if item in selected or dtype_counts[item[0]] >= dtype_quota[item[0]]:
                    continue
                selected.append(item)
                dtype_counts[item[0]] += 1
                tier_counts[tier_index] += 1
                progressed = True
                break
        if not progressed:
            break
    return selected


def valid(record):
    return (record is not None and record.get("status") == "OK" and
            record.get("correctness") == "PASS" and
            isinstance(record.get("latency_ms"), (int, float)) and
            isinstance(record.get("tiling"), dict))


def compact_tiling(record):
    tiling = record["tiling"]
    packet = tiling["packet"]
    l2 = tiling["l2"]
    return {
        "tiling_key": tiling["tiling_key"],
        "used_core_num": packet["used_core_num"],
        "single_core_m": packet["single_core_m"],
        "single_core_n": packet["single_core_n"],
        "single_core_k": packet["single_core_k"],
        "base_m": packet["base_m"],
        "base_n": packet["base_n"],
        "base_k": packet["base_k"],
        "depth_a1": packet["depth_a1"],
        "depth_b1": packet["depth_b1"],
        "step_ka": packet["step_ka"],
        "step_kb": packet["step_kb"],
        "l2_m_tiles": l2["m_tile_count"],
        "l2_n_tiles": l2["n_tile_count"],
        "l2_m_block": l2["m_tile_block"],
        "l2_n_block": l2["n_tile_block"],
    }


def critical_cube_proxy(record, m, n):
    packet = record["tiling"]["packet"]
    m_tiles = (m + packet["single_core_m"] - 1) // packet["single_core_m"]
    n_tiles = (n + packet["single_core_n"] - 1) // packet["single_core_n"]
    tasks = m_tiles * n_tiles
    waves = (tasks + packet["used_core_num"] - 1) // packet["used_core_num"]
    return waves * packet["single_core_m"] * packet["single_core_n"]


def measure(args, selected):
    groups = collections.defaultdict(list)
    for dtype, layout, m, n, k, _record in selected:
        groups[(dtype, layout)].append((dtype, layout, m, n, k))
    emitted = 0
    wins = 0
    deltas = []
    for (dtype, layout), values in sorted(groups.items()):
        for offset in range(0, len(values), args.batch_size):
            batch = values[offset:offset + args.batch_size]
            env = runner_env(os.environ, dtype, layout, "wave_exact_compare")
            env["MATMUL_V3_WARMUP"] = str(args.warmup)
            env["MATMUL_V3_REPEATS"] = str(args.repeats)
            env["MATMUL_V3_MEASUREMENT_PLAN"] = \
                "official_pre,wave_exact_pre,wave_exact_post,official_post"
            _rc, records, _stderr = invoke(args.runner, env, batch, args.run_log)
            by_key = {(record.get("shape"), record.get("mode")): record for record in records}
            for _dtype, _layout, m, n, k in batch:
                shape = f"M{m}_N{n}_K{k}_{layout}"
                official_pre = by_key.get((shape, "official_pre"))
                candidate_pre = by_key.get((shape, "wave_exact_pre"))
                candidate_post = by_key.get((shape, "wave_exact_post"))
                official_post = by_key.get((shape, "official_post"))
                quartet = (official_pre, candidate_pre, candidate_post, official_post)
                if not all(valid(record) for record in quartet):
                    continue
                if (official_pre["branch"] != "BASE_ND2NZ" or
                        official_post["branch"] != "BASE_ND2NZ" or
                        candidate_pre["branch"] != TARGET_BRANCH or
                        candidate_post["branch"] != TARGET_BRANCH):
                    continue
                official_ms = (official_pre["latency_ms"] + official_post["latency_ms"]) / 2.0
                candidate_ms = (candidate_pre["latency_ms"] + candidate_post["latency_ms"]) / 2.0
                delta = 100.0 * (candidate_ms / official_ms - 1.0)
                official_proxy = critical_cube_proxy(official_pre, m, n)
                candidate_proxy = critical_cube_proxy(candidate_pre, m, n)
                output = {
                    "shape": shape,
                    "dtype": dtype,
                    "family": TARGET_BRANCH,
                    "official_latency_ms": f"{official_ms:.9f}",
                    "candidate_latency_ms": f"{candidate_ms:.9f}",
                    "delta_pct": f"{delta:+.3f}",
                    "predicted_critical_cube_delta_pct":
                        f"{100.0 * (candidate_proxy / official_proxy - 1.0):+.3f}",
                    "winner": "candidate" if delta < 0.0 else "official",
                    "correctness": "PASS_BOTH_ABBA",
                    "official_tiling": compact_tiling(official_pre),
                    "candidate_tiling": compact_tiling(candidate_pre),
                }
                print(json.dumps(output, separators=(",", ":")), flush=True)
                emitted += 1
                wins += delta < 0.0
                deltas.append(delta)
    summary = {
        "summary": TARGET_BRANCH,
        "measured": emitted,
        "candidate_wins": wins,
        "official_wins": emitted - wins,
        "mean_delta_pct": f"{sum(deltas) / len(deltas):+.3f}" if deltas else None,
    }
    print(json.dumps(summary, separators=(",", ":")), flush=True)
    return 0 if emitted else 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True)
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--quota", type=int, default=24)
    parser.add_argument("--discovery-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    selected = discover(args)
    if not selected:
        return 3
    return measure(args, selected)


if __name__ == "__main__":
    raise SystemExit(main())
