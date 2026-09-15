#!/usr/bin/env python3
"""Single entry point for installed routes and experimental derivatives."""
from __future__ import annotations

import argparse
import json

from complete_formula_selector import generate as generate_installed
from novel_family_selector import applicable, generate as generate_novel


DERIVATIVE_ORDER = (
    "DIRECT_INIT_WHOLE_OUTPUT_SPLIT_K",
    "DIRECT_INIT_TAIL_WAVE_SPLIT_K",
)


def generate(
    m: int, k: int, n: int, dtype: str = "fp16",
    trans_a: bool = False, trans_b: bool = False,
) -> dict:
    request = {
        "M": int(m), "N": int(n), "K": int(k), "dtype": str(dtype),
        "transA": bool(trans_a), "transB": bool(trans_b),
    }
    matches = []
    if dtype == "fp32":
        for family in DERIVATIVE_ORDER:
            ok, reason = applicable(request, family)
            if ok:
                matches.append((family, reason))
    if len(matches) > 1:
        raise RuntimeError(f"experimental derivative predicates overlap: {matches}")
    if matches:
        result = generate_novel(
            m, k, n, dtype, trans_a, trans_b,
            required_family=matches[0][0],
        )
    else:
        result = generate_installed(m, k, n, dtype, trans_a, trans_b)
    result["production_selection"] = {
        "complete_tilings_constructed": 1,
        "experimental_derivative_predicate_matches": len(matches),
        "official_selector": False,
        "official_packet_seed": False,
        "cost_model": False,
        "latency_ranker": False,
        "candidate_enumeration": False,
        "history_lookup": False,
        "repository_lookup": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        generate(args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b),
        sort_keys=True, indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ))


if __name__ == "__main__":
    main()
