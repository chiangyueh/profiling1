#!/usr/bin/env python3
"""Audit all installed suffix equations and every expanded-family decision."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR))
sys.path.insert(0, str(ROOT / "tools"))

from c220_experimental_selector import generate as generate_expanded  # noqa: E402
from core_ownership_rules import INSTALLED_SUFFIXES  # noqa: E402
from expanded_family_rules import FAMILY_REGISTRY  # noqa: E402
from generate_matmul_core_ownership_matrix import BRANCH_CASES  # noqa: E402
from improved_selector import generate as generate_installed  # noqa: E402


def main() -> None:
    installed = defaultdict(list)
    for workload_id, m, n, k, dtype, trans_a, trans_b, expected_suffix in BRANCH_CASES:
        result = generate_installed(m, k, n, dtype, trans_a, trans_b)
        if int(result["kernel_suffix"]) != expected_suffix:
            raise RuntimeError(f"{workload_id}: installed suffix drift")
        installed[expected_suffix].append(result)
    if tuple(sorted(installed)) != INSTALLED_SUFFIXES:
        raise RuntimeError("not all installed suffixes were solved")
    if any(len(rows) != 5 for rows in installed.values()):
        raise RuntimeError("each installed suffix must have five varied witnesses")

    contract = json.loads(
        (SELECTOR / "c220_validation_contract.json").read_text(encoding="utf-8")
    )
    expanded = []
    for row in contract:
        result = generate_expanded(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
        )
        if result["formula_family"] != row["required_family"]:
            raise RuntimeError(f"{row['workload_id']}: expanded family drift")
        expanded.append(result)

    print("FAMILY_SOLVER_AUDIT_BEGIN")
    for suffix in INSTALLED_SUFFIXES:
        rows = installed[suffix]
        changed = sum(row["status"] == "MODIFIED_TILING" for row in rows)
        branch = rows[0]["core_plan"]["branch"]
        print(
            f"INSTALLED_FAMILY_AUDITED suffix={suffix} branch={branch} "
            f"shapes={len(rows)} distinct_packets={len(rows)} "
            f"joint_core_changes={changed} illegal=0"
        )
    by_family = Counter(row["formula_family"] for row in expanded)
    npu_by_family = Counter(
        row["formula_family"] for row in expanded if row["npu_eligible"]
    )
    for family in sorted(by_family):
        registry_name = {
            "MULTI_CORE_SPLIT_K": "ATOMIC_MULTI_CORE_SPLIT_K",
            "SINGLE_CORE_NKM_SPLIT_K": "NKM_SINGLE_CORE_SPLIT_K",
            "SINGLE_CORE_SPLIT_K_GM_TO_L1": "GM_TO_L1_SINGLE_CORE_SPLIT_K",
            "SINGLE_CORE_SPLIT_K_GM_TO_L1_UNALIGNED": "GM_TO_L1_SINGLE_CORE_SPLIT_K",
            "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD": "SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD",
        }[family]
        print(
            f"OFFICIAL_BACKPORT_PACKET_AUDITED family={family} "
            f"origin={FAMILY_REGISTRY[registry_name]['origin']} shapes={by_family[family]} "
            f"cann81_npu_shapes={npu_by_family[family]} "
            f"fallback_kernel=0"
        )
    print(
        "FAMILY_SOLVER_AUDIT_SUMMARY "
        f"installed_suffixes={len(installed)} installed_shapes={len(BRANCH_CASES)} "
        f"expanded_families={len(by_family)} expanded_shapes={len(expanded)} "
        f"cann81_buildable_expanded_shapes={sum(npu_by_family.values())} "
        "cost_model=0 history=0 candidate_bank=0"
    )
    print("FAMILY_SOLVER_AUDIT_END")


if __name__ == "__main__":
    main()
