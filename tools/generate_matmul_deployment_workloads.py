#!/usr/bin/env python3
"""Write the fixed, deployment-style MatMul comparison panel."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Shape notation in the identifier is M x K x N.  The CSV and model use M,N,K.
SHAPES = (
    ("deploy_m256_k256_n256", 256, 256, 256, "square_small"),
    ("deploy_m512_k512_n512", 512, 512, 512, "square_small"),
    ("deploy_m1024_k1024_n1024", 1024, 1024, 1024, "square_medium"),
    ("deploy_m2048_k2048_n2048", 2048, 2048, 2048, "square_large"),
    ("deploy_m2048_k7168_n1536", 2048, 1536, 7168, "target"),
    ("deploy_m2048_k2048_n7168", 2048, 7168, 2048, "target"),
    ("deploy_m4096_k7168_n512", 4096, 512, 7168, "target"),
    ("deploy_m512_k4096_n4096", 512, 4096, 4096, "wide"),
    ("deploy_m4096_k4096_n512", 4096, 512, 4096, "tall"),
    ("deploy_m1024_k11008_n4096", 1024, 4096, 11008, "long_k"),
    ("deploy_m1024_k4096_n11008", 1024, 11008, 4096, "wide"),
    ("deploy_m1536_k5120_n2560", 1536, 2560, 5120, "non_power_of_two"),
    ("deploy_m256_k32768_n256", 256, 256, 32768, "small_output_long_k"),
    ("deploy_m4096_k128_n192", 4096, 192, 128, "short_k"),
)


def build_workloads() -> list[dict[str, str]]:
    return [
        {
            "workload_id": workload_id,
            "m": str(m),
            "n": str(n),
            "k": str(k),
            "dtype": "fp16",
            "trans_a": "0",
            "trans_b": "0",
            "max_cores": "20",
            "search_family": "independent_deployment_top1",
            "coverage_intent": coverage,
            "required_successful_tilings": "1",
        }
        for workload_id, m, n, k, coverage in SHAPES
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_workloads()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "MATMUL_DEPLOYMENT_CATALOG "
        f"shapes={len(rows)} model_candidates={len(rows)} "
        f"installed_references={len(rows)} records={2 * len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
