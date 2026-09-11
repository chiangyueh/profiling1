#!/usr/bin/env python3
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_ROOT = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR_ROOT))

from improved_selector import KERNEL_VARIANT, generate  # noqa: E402


EXPECTED_RULES = {
    f"R{index:02d}_{name}"
    for index, name in enumerate((
        "GEOMETRIC_BASE_MN",
        "WAVE_TAIL_REDISTRIBUTION",
        "BASE_K_AFTER_MN",
        "JOINT_L1_STEP_DEPTH",
        "AL1_RESIDENCY",
        "BL1_RESIDENCY",
        "SINGLE_CORE_SPLIT_K_GRID",
        "DETERMINISTIC_SPLIT_K",
        "SPLIT_K_L2_LONG_AXIS",
        "BASE_L2_MACRO",
        "TRAVERSAL_ORDER",
        "FIXPIPE_RESIDENT_B",
        "FIXPIPE_BALANCED_A_PIPELINE",
        "TWO_WAVE_L2_RESIDENCY",
    ), 1)
}


def main():
    contract_path = SELECTOR_ROOT / "branch_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = set(KERNEL_VARIANT)
    declared = {int(row["suffix"]) for row in contract}
    if declared != expected:
        raise RuntimeError(
            f"branch contract mismatch missing={sorted(expected-declared)} "
            f"extra={sorted(declared-expected)}"
        )
    contracted_rules = {
        rule for row in contract for rule in row["required_rules"]
    }
    if contracted_rules != EXPECTED_RULES:
        raise RuntimeError(
            f"rule contract mismatch missing={sorted(EXPECTED_RULES-contracted_rules)} "
            f"extra={sorted(contracted_rules-EXPECTED_RULES)}"
        )
    observed = []
    for row in contract:
        result = generate(
            int(row["m"]), int(row["k"]), int(row["n"]), str(row["dtype"]),
            bool(row["trans_a"]), bool(row["trans_b"]),
        )
        if result["status"] != "MODIFIED_TILING":
            raise RuntimeError(
                f"suffix {row['suffix']} has no distinct improved packet: "
                f"{result.get('skip_reason', result['status'])}"
            )
        if int(result["kernel_suffix"]) != int(row["suffix"]):
            raise RuntimeError(
                f"branch witness expected suffix {row['suffix']} but generated "
                f"{result['kernel_suffix']}"
            )
        if result["formula_family"] != row["family"]:
            raise RuntimeError(
                f"suffix {row['suffix']} expected family {row['family']} but got "
                f"{result['formula_family']}"
            )
        missing_rules = set(row["required_rules"]) - set(result["changed_rules"])
        if missing_rules:
            raise RuntimeError(
                f"suffix {row['suffix']} missed rules {sorted(missing_rules)}"
            )
        if not result["changed_fields"] or result["baseline_equivalent"]:
            raise RuntimeError(f"suffix {row['suffix']} did not change executable state")
        if result["path_coverage"]["official_selector_as_seed"] != "FORBIDDEN_AND_NOT_USED":
            raise RuntimeError(f"suffix {row['suffix']} leaked the official selector")
        observed.append({
            "suffix": row["suffix"],
            "variant": result["improved"]["kernel_variant"],
            "family": result["formula_family"],
            "changed_rules": result["changed_rules"],
            "changed_field_count": len(result["changed_fields"]),
            "packet_sha256": result["improved"]["tiling_data_sha256"],
        })
    print(json.dumps({
        "status": "PASS",
        "installed_cann81_dispatch_branches": len(expected),
        "modified_rule_groups": len(EXPECTED_RULES),
        "modified_branch_witnesses": len(observed),
        "branches": observed,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
