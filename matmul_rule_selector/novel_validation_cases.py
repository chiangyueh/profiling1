"""Deterministic, shape-diverse validation domain for repository-owned families."""
from __future__ import annotations


SYSTEM_WORKSPACE = 20 * 1024 * 1024
MAX_FOOTPRINT = 300 * 1024 * 1024


def _footprint(m: int, n: int, k: int) -> int:
    return (m * k + k * n + 2 * m * n) * 4 + SYSTEM_WORKSPACE


def build_cases() -> list[dict]:
    cases: list[dict] = []
    m_small = (16, 32, 48, 64, 80, 96, 112, 128)
    n_small = (16, 32, 48, 64, 80, 96, 112, 128)
    k_small = (
        8192, 9216, 10240, 12288, 14336, 16384, 18432, 20480,
        22528, 24576, 26624, 28672, 30720, 32768, 36864, 40960,
        45056, 49152, 53248, 59968,
    )
    seen: set[tuple[int, int, int]] = set()
    cursor = 0
    while len(cases) < 100:
        m = m_small[cursor % len(m_small)]
        n = n_small[(cursor * 3 + cursor // len(m_small)) % len(n_small)]
        k = k_small[(cursor * 7 + cursor // 5) % len(k_small)]
        key = (m, n, k)
        if key not in seen:
            seen.add(key)
            cases.append({
                "workload_id": f"novel_seeded_atomic_{len(cases):03d}",
                "required_family": "SEEDED_ATOMIC_SPLIT_K",
                "m": m, "n": n, "k": k, "dtype": "fp32",
                "trans_a": False, "trans_b": True,
                "scale_band": "small_output_long_k",
            })
        cursor += 1
        if cursor > 10000:
            raise RuntimeError("could not construct unique seeded atomic cases")

    tail_m = (16, 48, 80, 112, 127)
    tail_n = (24, 56, 88, 120, 127)
    k_tail = (4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768)
    tail_candidates = []
    for m_tiles in range(2, 17):
        for n_tiles in range(2, 17):
            total = m_tiles * n_tiles
            tail = total % 20
            if total <= 20 or not 1 <= tail <= 10:
                continue
            for variant in range(5):
                m = (m_tiles - 1) * 128 + tail_m[(variant + n_tiles) % 5]
                n = (n_tiles - 1) * 128 + tail_n[(2 * variant + m_tiles) % 5]
                k = k_tail[(m_tiles * 3 + n_tiles * 5 + variant * 7) % len(k_tail)]
                if _footprint(m, n, k) > MAX_FOOTPRINT:
                    continue
                critical_reduction = (
                    1.0 - (
                        total // 20 + 1.0 / (20 // tail)
                    ) / (total // 20 + 1)
                )
                tail_candidates.append((
                    total, m_tiles + n_tiles, m, n, k, tail,
                    critical_reduction,
                ))

    # Interleave low and high tile counts so the first 100 cover medium and
    # large outputs instead of clustering at one end of the construction grid.
    tail_candidates.sort(key=lambda row: (row[0], row[2], row[3], row[4]))
    order = []
    left, right = 0, len(tail_candidates) - 1
    while left <= right:
        order.append(tail_candidates[left])
        left += 1
        if left <= right:
            order.append(tail_candidates[right])
            right -= 1

    tail_seen: set[tuple[int, int, int]] = set()
    for total, _, m, n, k, tail, reduction in order:
        key = (m, n, k)
        if key in tail_seen:
            continue
        tail_seen.add(key)
        index = len(cases) - 100
        cases.append({
            "workload_id": f"novel_seeded_tail_{index:03d}",
            "required_family": "SEEDED_TAIL_WAVE_SPLIT_K",
            "m": m, "n": n, "k": k, "dtype": "fp32",
            "trans_a": False, "trans_b": True,
            "scale_band": "medium" if total <= 80 else "large",
            "mn_tiles": total,
            "tail_tiles": tail,
            "critical_path_reduction": reduction,
        })
        if len(cases) == 200:
            break

    if len(cases) != 200:
        raise RuntimeError(f"expected 200 validation cases, got {len(cases)}")
    triples = {(row["m"], row["n"], row["k"]) for row in cases}
    if len(triples) != 200:
        raise RuntimeError("validation cases must contain 200 distinct shapes")
    return cases


CASES = build_cases()
