#!/usr/bin/env python3
"""Run every mandatory MatMul candidate audit in declared order.

An audit item is never skipped.  Each item emits PASS or FAIL evidence and a
single failure makes the process return non-zero.  Device timing is outside
this host-side legality and selection audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "matmul_rule_selector"
sys.path.insert(0, str(SELECTOR))
sys.path.insert(0, str(ROOT))

from candidate_engine import (  # noqa: E402
    GRAPH_TO_FAMILY,
    INSTALLED_SUFFIXES,
    INSTALLED_SUFFIXES_BY_DTYPE,
    NON_INSTALLED_FAMILIES,
    _deterministic_split_shape_applicable,
    _dominates,
    _single_core_split_shape_applicable,
    candidate_rule_key,
    family_applicability,
    generate_and_select,
    non_installed_family_status,
)
from formula_rules import Hardware, Shape, example_compile_info  # noqa: E402
from improved_selector import generate as generate_packet  # noqa: E402
from npu_cost_model.cann_matmul import (  # noqa: E402
    CANN81_MATMUL_FAMILIES,
    CANN81_MATMUL_KERNEL_SUFFIXES,
    plan_from_cann,
    validate_cann_tiling,
)
from npu_cost_model.hardware import ascend_910b3  # noqa: E402
from npu_cost_model.ir import MemorySpace, Resource  # noqa: E402
from npu_cost_model.operators import matmul  # noqa: E402
from npu_cost_model.simulator import align_up, ceil_div, simulate  # noqa: E402


SPEC_PATH = SELECTOR / "candidate_audit_spec.json"
CONTRACT_PATH = SELECTOR / "family_candidate_contract.json"
VALIDATION_PATH = SELECTOR / "validation_contract.json"

POSITIVE_WITNESSES = {
    # This BASE witness deliberately requires ND-to-NZ so the Vector and
    # conversion-workspace terms are audited rather than merely present at 0.
    "BASE": Shape(10_000, 33, 17),
    "AL1": Shape(8, 320, 4096, dtype="fp32", trans_b=True),
    "BL1": Shape(80_000, 256, 7, dtype="fp32"),
    "FIXPIPE_BL1": Shape(10_240, 65, 64, dtype="fp32"),
    "SINGLE_CORE_SPLIT_K": Shape(
        8192, 6144, 11_008, dtype="fp16", trans_a=True
    ),
    "DETERMINISTIC_SPLIT_K": Shape(128, 128, 7680, dtype="fp16"),
    "INCREMENTAL_PATTERN": Shape(
        64, 1024, 4096, dtype="fp16", trans_b=True
    ),
}

NEGATIVE_WITNESSES = {
    "AL1": Shape(17, 320, 4096, dtype="fp32", trans_b=True),
    "BL1": Shape(80_000, 256, 257, dtype="fp32"),
    "FIXPIPE_BL1": Shape(10_239, 65, 64, dtype="fp32"),
    "SINGLE_CORE_SPLIT_K": Shape(256, 256, 4096, dtype="fp16"),
    "DETERMINISTIC_SPLIT_K": Shape(32, 32, 7552, dtype="fp16"),
    "INCREMENTAL_PATTERN": Shape(
        129, 1024, 4096, dtype="fp16", trans_b=True
    ),
}

SUFFIX_WITNESSES = {
    0: ("BASE", Shape(10_000, 33, 17)),
    1: ("BASE", Shape(512, 512, 512)),
    20: ("SINGLE_CORE_SPLIT_K", Shape(384, 2561, 27_392)),
    21: ("SINGLE_CORE_SPLIT_K", Shape(640, 2560, 27_392)),
    30: ("DETERMINISTIC_SPLIT_K", Shape(16, 17, 7680)),
    31: ("DETERMINISTIC_SPLIT_K", Shape(32, 32, 8192)),
    101: ("AL1", Shape(12, 160, 7168, dtype="fp32", trans_b=True)),
    200: ("BL1", Shape(10_240, 96, 2, dtype="fp32")),
    201: ("BL1", Shape(10_240, 96, 8, dtype="fp32")),
    10200: ("FIXPIPE_BL1", Shape(10_240, 9, 2, dtype="fp32")),
    10201: ("FIXPIPE_BL1", Shape(10_240, 65, 128, dtype="fp16")),
    20201: ("FIXPIPE_BL1", Shape(12_288, 17, 16, dtype="fp32")),
}


def shape_dict(shape: Shape) -> dict[str, Any]:
    return {
        "M": shape.m,
        "N": shape.n,
        "K": shape.k,
        "dtype": shape.dtype,
        "transA": shape.trans_a,
        "transB": shape.trans_b,
    }


def candidate_id(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["family"],
        candidate["graph_name"],
        tuple(sorted(candidate["knowledge"].items())),
    )


class AuditContext:
    def __init__(self) -> None:
        self.spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
        self.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.hardware = Hardware()
        self.model_hardware = ascend_910b3()
        self._positive: dict[str, dict[str, Any]] = {}
        self._negative: dict[str, dict[str, Any]] = {}
        self._packet: dict[str, Any] | None = None
        self._suffix: dict[int, tuple[Shape, dict[str, Any]]] = {}

    def select(self, shape: Shape) -> dict[str, Any]:
        return generate_and_select(
            shape,
            self.hardware,
            example_compile_info(shape),
            include_audit_records=True,
        )

    def positive(self, name: str) -> dict[str, Any]:
        if name not in self._positive:
            self._positive[name] = self.select(POSITIVE_WITNESSES[name])
        return self._positive[name]

    def negative(self, name: str) -> dict[str, Any]:
        if name not in self._negative:
            self._negative[name] = self.select(NEGATIVE_WITNESSES[name])
        return self._negative[name]

    def packet(self) -> dict[str, Any]:
        if self._packet is None:
            shape = POSITIVE_WITNESSES["BASE"]
            self._packet = generate_packet(
                shape.m,
                shape.k,
                shape.n,
                shape.dtype,
                shape.trans_a,
                shape.trans_b,
            )
        return self._packet

    def suffix(self, suffix: int) -> tuple[Shape, dict[str, Any]]:
        if suffix not in self._suffix:
            family, shape = SUFFIX_WITNESSES[suffix]
            winner = self.select(shape)
            candidates = [
                item for item in winner["candidate_audit"]["all_candidates"]
                if item["family"] == family
                and item["kernel_suffix"] == suffix
            ]
            assert candidates, (suffix, family, winner["candidate_audit"])
            self._suffix[suffix] = (shape, min(
                candidates,
                key=candidate_rule_key,
            ))
        return self._suffix[suffix]

    def all_positive(self) -> dict[str, dict[str, Any]]:
        return {name: self.positive(name) for name in POSITIVE_WITNESSES}

    def counts(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, winner in self._positive.items():
            audit = winner["candidate_audit"]
            result[name] = {
                "shape": shape_dict(POSITIVE_WITNESSES[name]),
                "raw": audit["raw_counts"],
                "unique": audit["unique_counts"],
                "pareto": audit["pareto_counts"],
                "winner": winner["family"],
                "winner_suffix": winner["kernel_suffix"],
                "selection_elapsed_us": audit["selection_elapsed_us"],
                "rejection_reasons": audit["rejection_reasons"],
            }
        return result


def check_scope_01(ctx: AuditContext) -> dict[str, Any]:
    installed_family_names = {
        item["name"] for item in ctx.contract["families"]
    }
    expected_scored = {
        "BASE", "AL1", "BL1", "FIXPIPE_BL1",
        "SINGLE_CORE_SPLIT_K", "DETERMINISTIC_SPLIT_K",
        "INCREMENTAL_PATTERN",
    }
    assert installed_family_names == expected_scored
    assert set(GRAPH_TO_FAMILY) == set(CANN81_MATMUL_FAMILIES)
    suffixes = {
        suffix
        for item in ctx.contract["families"]
        for suffix in item["suffixes"]
    }
    assert suffixes == set(INSTALLED_SUFFIXES)
    assert suffixes == set(CANN81_MATMUL_KERNEL_SUFFIXES)
    for item in ctx.contract["families"]:
        for key in (
            "status", "source_anchors", "applicability", "candidate_axes",
            "hard_constraints", "cost_components", "selection_rule",
        ):
            assert item.get(key), (item["name"], key)
    return {
        "execution_graphs": sorted(CANN81_MATMUL_FAMILIES),
        "scored_families": sorted(expected_scored),
        "installed_suffixes": sorted(suffixes),
    }


def check_scope_02(ctx: AuditContext) -> dict[str, Any]:
    contract = {
        item["name"]: item for item in ctx.contract["non_installed_families"]
    }
    assert set(contract) == set(NON_INSTALLED_FAMILIES)
    evidence = []
    for name in sorted(contract):
        engine = non_installed_family_status(name)
        assert contract[name]["status"] == "REQUIRES_NEW_KERNEL"
        assert engine["status"] == "REQUIRES_NEW_KERNEL"
        assert engine["packet"] == "PROHIBITED"
        assert name not in GRAPH_TO_FAMILY.values()
        evidence.append(engine)
    return {"new_kernel_boundaries": evidence}


def check_indep_01(ctx: AuditContext) -> dict[str, Any]:
    del ctx
    source_path = SELECTOR / "candidate_engine.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_modules = (
        "matmul_reconstruction.api", "xgboost", "pandas", "csv",
    )
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in imported for prefix in forbidden_modules
    )
    forbidden_calls = {
        "baseline_select", "select", "read_csv", "load_model",
        "GetTilingFromRepo", "get_tiling_from_repo", "open",
    }
    assert not (set(calls) & forbidden_calls)
    formula_tree = ast.parse(
        (SELECTOR / "formula_rules.py").read_text(encoding="utf-8")
    )
    solve = next(
        node for node in formula_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    solve_calls = {
        node.func.id
        for node in ast.walk(solve)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_obsolete_single_formula" not in solve_calls
    assert "generate_and_select" in solve_calls
    return {
        "candidate_imports": imported,
        "forbidden_calls_found": sorted(set(calls) & forbidden_calls),
        "public_solve_calls_candidate_engine": True,
    }


def check_indep_02(ctx: AuditContext) -> dict[str, Any]:
    path = SELECTOR / "improved_selector.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    assert "matmul_reconstruction.api" not in imports
    assert not ({"baseline_select", "select"} & set(calls))
    package_init = ast.parse(
        (
            SELECTOR / "baseline_core" / "matmul_reconstruction" / "__init__.py"
        ).read_text(encoding="utf-8")
    )
    package_imports = [
        node.module or ""
        for node in ast.walk(package_init)
        if isinstance(node, ast.ImportFrom)
    ]
    assert "api" not in package_imports
    assert "matmul_reconstruction.api" not in sys.modules
    generate_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    generate_calls = {
        node.func.id
        for node in ast.walk(generate_node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {"solve", "initialize_cube", "pack_packet"} <= generate_calls
    for node in ast.walk(generate_node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {"baseline", "baseline_equivalent"}
    result = ctx.packet()
    assert "baseline" not in result and "baseline_equivalent" not in result
    assert result["selection_basis"] == "INDEPENDENT_HARDWARE_RULE_MINIMUM"
    harness = (ROOT / "run_npu.sh").read_text(encoding="utf-8")
    generator_markers = (
        "\npython3 tools/generate_matmul_rule_matrix.py",
        "\npython3 tools/generate_matmul_c220_experimental_matrix.py",
    )
    generation_position = min(
        harness.index(marker) for marker in generator_markers if marker in harness
    )
    measurement_position = harness.index('announce "OFFICIAL_MEASUREMENT begin')
    assert generation_position < measurement_position
    assert "--historical-profile" not in harness
    assert "--reference-profile" not in harness
    return {
        "generator_imports_official_api": False,
        "package_side_effect_imports_official_api": False,
        "generator_calls_official_selector": False,
        "generator_emits_baseline_fields": False,
        "official_measurement_after_manifest_generation": True,
    }


def check_gen_01(ctx: AuditContext) -> dict[str, Any]:
    values = ctx.all_positive()
    evidence = {}
    for name, winner in values.items():
        audit = winner["candidate_audit"]
        assert audit["region_exhaustive"] is True
        assert audit["region_evaluated"] < 100_000
        evidence[name] = {
            "evaluated": audit["region_evaluated"],
            "plans": audit["region_plan_count"],
            "anchors": audit["region_anchor_count"],
        }
    return evidence


def check_gen_02(ctx: AuditContext) -> dict[str, Any]:
    required_variation = {
        "BASE": {"baseM", "baseN", "baseK", "iterateOrder"},
        "AL1": {"baseK", "depthB1", "dbL0C"},
        "BL1": {"baseM", "baseN", "dbL0C"},
        "FIXPIPE_BL1": {"depthA1", "dbL0C", "tilingEnable"},
        "SINGLE_CORE_SPLIT_K": {"singleCoreK", "stepM", "iterateOrder"},
        "DETERMINISTIC_SPLIT_K": {"stepM", "stepN", "iterateOrder"},
        "INCREMENTAL_PATTERN": {"baseM", "baseN", "baseK"},
    }
    evidence = {}
    for family, shape in POSITIVE_WITNESSES.items():
        winner = ctx.positive(family)
        audit = winner["candidate_audit"]
        assert family_applicability(shape, ctx.hardware)[family] is True
        count = audit["unique_counts"].get(family, 0)
        assert count > 0, (family, audit["rejection_reasons"])
        candidates = [
            item for item in audit["all_candidates"]
            if item["family"] == family
        ]
        varied = {
            name for name in candidates[0]["knowledge"]
            if len({item["knowledge"][name] for item in candidates}) > 1
        }
        assert required_variation[family] <= varied, (family, varied)
        evidence[family] = {
            "candidate_count": count,
            "varied_packet_fields": sorted(varied),
        }
    return {"positive_unique_candidates": evidence}


def check_gen_03(ctx: AuditContext) -> dict[str, Any]:
    evidence = {}
    for family, shape in NEGATIVE_WITNESSES.items():
        winner = ctx.negative(family)
        count = winner["candidate_audit"]["unique_counts"].get(family, 0)
        assert family_applicability(shape, ctx.hardware)[family] is False
        assert count == 0, (family, count)
        evidence[family] = {"shape": shape_dict(shape), "candidate_count": count}
    return evidence


def check_gen_04(ctx: AuditContext) -> dict[str, Any]:
    audit = ctx.positive("INCREMENTAL_PATTERN")["candidate_audit"]
    source = audit["incremental_source_candidates"]
    converted = audit["incremental_converted_candidates"]
    assert source > 0
    assert source == converted
    return {"source_candidates": source, "converted_and_rescored": converted}


def stable_candidate_result(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    result["candidate_audit"].pop("selection_elapsed_us", None)
    return result


def check_gen_05(ctx: AuditContext) -> dict[str, Any]:
    shape = POSITIVE_WITNESSES["BASE"]
    first = ctx.positive("BASE")
    second = ctx.select(shape)
    assert stable_candidate_result(first) == stable_candidate_result(second)
    first_packet = ctx.packet()
    second_packet = generate_packet(
        shape.m, shape.k, shape.n, shape.dtype, shape.trans_a, shape.trans_b
    )
    assert first_packet["improved"]["tiling_data_sha256"] == (
        second_packet["improved"]["tiling_data_sha256"]
    )
    return {
        "ordered_candidate_count": len(
            first["candidate_audit"]["all_candidates"]
        ),
        "packet_sha256": first_packet["improved"]["tiling_data_sha256"],
    }


def check_gen_06(ctx: AuditContext) -> dict[str, Any]:
    evidence = {}
    for name, winner in ctx.all_positive().items():
        audit = winner["candidate_audit"]
        unique = sum(audit["unique_counts"].values())
        pareto = sum(audit["pareto_counts"].values())
        elapsed_us = float(audit["selection_elapsed_us"])
        assert unique <= 256, (name, unique)
        assert pareto <= 128, (name, pareto)
        assert elapsed_us < 2_000_000, (name, elapsed_us)
        assert audit["fixed_topn_pruning"] is False
        evidence[name] = {
            "unique_candidates": unique,
            "pareto_candidates": pareto,
            "selection_elapsed_ms": elapsed_us / 1000.0,
        }
    return {
        "bounds_are_failure_thresholds_not_truncation": True,
        "per_shape": evidence,
    }


def check_gen_07(ctx: AuditContext) -> dict[str, Any]:
    evidence = {}
    for suffix in sorted(SUFFIX_WITNESSES):
        shape, candidate = ctx.suffix(suffix)
        expected_family = SUFFIX_WITNESSES[suffix][0]
        assert candidate["kernel_suffix"] == suffix
        assert candidate["family"] == expected_family
        assert suffix in INSTALLED_SUFFIXES_BY_DTYPE[shape.dtype]
        evidence[str(suffix)] = {
            "family": expected_family,
            "shape": shape_dict(shape),
            "packet": candidate["knowledge"],
        }
    assert set(map(int, evidence)) == set(INSTALLED_SUFFIXES)
    incremental = ctx.positive("INCREMENTAL_PATTERN")["candidate_audit"]
    assert incremental["unique_counts"].get("INCREMENTAL_PATTERN", 0) > 0
    return {
        "installed_suffixes": evidence,
        "incremental_candidate_source": "COVERED",
        "covered_execution_branches": len(evidence) + 1,
    }


def check_gen_08(ctx: AuditContext) -> dict[str, Any]:
    del ctx
    rows = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    counts = Counter(row["case_role"] for row in rows)
    assert len(rows) == 62
    assert counts == {
        "branch_probe": 13,
        "selector_top1": 38,
        "candidate_probe": 11,
    }
    assert len({row["workload_id"] for row in rows}) == len(rows)
    axes = {row["selection_axis"] for row in rows}
    for required in (
        "bl1_parent_task_ablation",
        "det_orientation_ablation",
        "fixpipe_l0c_ablation_low",
        "fixpipe_l0c_ablation_high",
        "al1_k_grain_ablation",
        "incremental_vs_base_ablation",
        "base_aspect_ablation",
        "base_k_grain_ablation",
        "sc_buffering_ablation",
        "base_parent_task_ablation",
        "sc_vs_base_family_ablation",
    ):
        assert required in axes
    return {"rows": len(rows), "roles": dict(sorted(counts.items()))}


def check_legal_01(ctx: AuditContext) -> dict[str, Any]:
    validated = Counter()
    for witness, winner in ctx.all_positive().items():
        shape = POSITIVE_WITNESSES[witness]
        for candidate in winner["candidate_audit"]["all_candidates"]:
            violations = validate_cann_tiling(
                shape.m,
                shape.n,
                shape.k,
                shape.dtype,
                shape.trans_a,
                shape.trans_b,
                candidate["knowledge"],
                ctx.model_hardware,
                a_layout=shape.format_a,
                b_layout=shape.format_b,
                output_dtype=shape.c_dtype,
                has_bias=shape.bias,
                incremental_pattern=(
                    candidate["family"] == "INCREMENTAL_PATTERN"
                ),
            )
            assert not violations, (witness, candidate_id(candidate), violations)
            assert candidate["kernel_suffix"] in (
                INSTALLED_SUFFIXES_BY_DTYPE[shape.dtype]
            )
            assert candidate["exact_lowered_resimulation"] is True
            validated[candidate["family"]] += 1
    return {"validated_candidates": dict(sorted(validated.items()))}


def check_legal_02(ctx: AuditContext) -> dict[str, Any]:
    cases = {
        "positive": (Shape(10_240, 65, 64, dtype="fp32"), True),
        "M_below_20x512": (Shape(10_239, 65, 64, dtype="fp32"), False),
        "N_equal_256": (Shape(10_240, 256, 64, dtype="fp32"), False),
        "K_above_256": (Shape(10_240, 65, 257, dtype="fp32"), False),
        "N_output_aligned": (Shape(10_240, 64, 64, dtype="fp32"), False),
        "N_divides_output_alignment": (Shape(10_240, 32, 64, dtype="fp32"), False),
        "scalar_bound": (Shape(10_240, 7, 7, dtype="fp32"), False),
    }
    evidence = {}
    for name, (shape, expected) in cases.items():
        actual = family_applicability(shape, ctx.hardware)["FIXPIPE_BL1"]
        assert actual is expected, name
        evidence[name] = actual
    evidence["M_formula"] = "20*512"
    return evidence


def check_legal_03(ctx: AuditContext) -> dict[str, Any]:
    cases = {
        "fp32_transA": (Shape(10_240, 65, 64, dtype="fp32", trans_a=True), True),
        "fp16_transA": (Shape(10_240, 65, 64, dtype="fp16", trans_a=True), False),
        "fp16_K_127_below_quantum": (Shape(10_240, 65, 127, dtype="fp16"), True),
        "fp16_K_128_aligned": (Shape(10_240, 65, 128, dtype="fp16"), True),
        "fp16_K_129_unaligned": (Shape(10_240, 65, 129, dtype="fp16"), False),
    }
    evidence = {}
    for name, (shape, expected) in cases.items():
        actual = family_applicability(shape, ctx.hardware)["FIXPIPE_BL1"]
        assert actual is expected, name
        evidence[name] = actual
    fix_candidates = [
        item for item in ctx.positive("FIXPIPE_BL1")["candidate_audit"]["all_candidates"]
        if item["family"] == "FIXPIPE_BL1"
    ]
    suffixes = {item["kernel_suffix"] for item in fix_candidates}
    assert {10201, 20201} <= suffixes
    evidence["positive_graph_suffixes"] = sorted(suffixes)
    return evidence


def check_legal_04(ctx: AuditContext) -> dict[str, Any]:
    shape = POSITIVE_WITNESSES["AL1"]
    candidates = [
        item for item in ctx.positive("AL1")["candidate_audit"]["all_candidates"]
        if item["family"] == "AL1"
    ]
    assert candidates
    boundaries = {
        "positive_maxima": (Shape(16, 320, 4096, dtype="fp32", trans_b=True), True),
        "dtype": (Shape(16, 320, 4096, dtype="fp16", trans_b=True), False),
        "transA": (Shape(16, 320, 4096, dtype="fp32", trans_a=True, trans_b=True), False),
        "transB": (Shape(16, 320, 4096, dtype="fp32"), False),
        "M": (Shape(17, 320, 4096, dtype="fp32", trans_b=True), False),
        "N_lower": (Shape(16, 16, 4096, dtype="fp32", trans_b=True), False),
        "N_upper": (Shape(16, 321, 4096, dtype="fp32", trans_b=True), False),
        "K_lower": (Shape(16, 320, 3968, dtype="fp32", trans_b=True), False),
        "K_alignment": (Shape(16, 320, 4100, dtype="fp32", trans_b=True), False),
        "full_A_L1_capacity": (Shape(16, 320, 32768, dtype="fp32", trans_b=True), False),
    }
    checked_boundaries = {}
    for name, (boundary, expected) in boundaries.items():
        actual = family_applicability(boundary, ctx.hardware)["AL1"]
        assert actual is expected, name
        checked_boundaries[name] = actual
    for item in candidates:
        field = item["fields"]
        assert (field["baseM"], field["baseN"]) == (16, 16)
        assert shape.k % field["baseK"] == 0
        assert field["singleCoreM"] == shape.m
        assert field["singleCoreN"] == 16
        assert field["singleCoreK"] >= shape.k
        assert field["stepKa"] == shape.k // field["baseK"]
        assert field["depthA1"] == field["stepKa"]
        assert field["depthB1"] in (
            field["stepKb"], 2 * field["stepKb"]
        )
        assert item["resource_bytes"]["L1_AB"] <= ctx.hardware.l1_usable + 256
    return {
        "candidate_count": len(candidates),
        "geometry": {
            "baseMN": [16, 16],
            "baseK_values": sorted({item["fields"]["baseK"] for item in candidates}),
        },
        "boundaries": checked_boundaries,
    }


def check_legal_05(ctx: AuditContext) -> dict[str, Any]:
    shape = POSITIVE_WITNESSES["BL1"]
    candidates = [
        item for item in ctx.positive("BL1")["candidate_audit"]["all_candidates"]
        if item["family"] == "BL1"
    ]
    assert candidates
    boundaries = {
        "on_the_way": (Shape(80_000, 256, 7, dtype="fp32"), True),
        "M_strict_threshold": (Shape(4096, 256, 64, dtype="fp16"), False),
        "K_above_256": (Shape(80_000, 256, 257, dtype="fp32"), False),
        "no_route": (Shape(20_000, 65, 64, dtype="fp16"), False),
        "transB_byte_route_missing": (Shape(20_000, 256, 7, dtype="fp16", trans_b=True), False),
        "fp32_vnchw": (Shape(80_000, 65, 7, dtype="fp32"), True),
        "vnchw_outer_below_72368": (Shape(72_367, 65, 7, dtype="fp32"), False),
        "vnchw_inner_equal_1": (Shape(80_000, 65, 1, dtype="fp32"), False),
        "vnchw_resident_B_over_half_L1": (Shape(2_100_000, 120_000, 8, dtype="fp32"), False),
    }
    checked_boundaries = {}
    for name, (boundary, expected) in boundaries.items():
        actual = family_applicability(boundary, ctx.hardware)["BL1"]
        assert actual is expected, name
        checked_boundaries[name] = actual
    assert shape.m > 16 * max(shape.k, shape.n) and shape.k <= 256
    for item in candidates:
        field = item["fields"]
        assert field["singleCoreN"] >= shape.n
        assert field["singleCoreM"] == 2 * field["baseM"]
        assert field["stepN"] == ceil_div(shape.n, field["baseN"])
        assert field["depthB1"] == field["stepN"] * field["stepKb"]
        assert item["resource_bytes"]["L1_AB"] <= ctx.hardware.l1_usable + 256
    return {"candidate_count": len(candidates), "boundaries": checked_boundaries}


def check_legal_06(ctx: AuditContext) -> dict[str, Any]:
    sc_shape = POSITIVE_WITNESSES["SINGLE_CORE_SPLIT_K"]
    sc = [
        item for item in ctx.positive("SINGLE_CORE_SPLIT_K")["candidate_audit"]["all_candidates"]
        if item["family"] == "SINGLE_CORE_SPLIT_K"
    ]
    det_shape = POSITIVE_WITNESSES["DETERMINISTIC_SPLIT_K"]
    det = [
        item for item in ctx.positive("DETERMINISTIC_SPLIT_K")["candidate_audit"]["all_candidates"]
        if item["family"] == "DETERMINISTIC_SPLIT_K"
    ]
    assert sc and det
    for item in sc:
        field = item["fields"]
        assert field["singleCoreK"] == field["stepKa"] * field["baseK"]
        assert field["stepKa"] == field["stepKb"]
        assert ceil_div(sc_shape.k, field["singleCoreK"]) >= 2
        assert field["singleCoreM"] >= field["stepM"] * field["baseM"]
        assert field["singleCoreN"] >= field["stepN"] * field["baseN"]
        assert item["cost"]["resource_cycles"][Resource.ATOMIC.value] > 0
    for item in det:
        field = item["fields"]
        k_parts = ceil_div(det_shape.k, field["singleCoreK"])
        assert field["usedCoreNum"] >= 2
        assert field["usedCoreNum"] <= k_parts
        n_tail = det_shape.n - (
            ceil_div(det_shape.n, field["singleCoreN"]) - 1
        ) * field["singleCoreN"]
        assert align_up(n_tail, 64) <= field["singleCoreN"]
        assert item["workspace_bytes"] > 20 * 1024 * 1024

    sc_knowledge = dict(sc[0]["knowledge"])
    sc_n_too_small = dict(sc_knowledge, singleCoreN=511)
    assert not _single_core_split_shape_applicable(
        sc_shape, ctx.hardware, sc_n_too_small
    )
    sc_low_occupancy = dict(
        sc_knowledge, singleCoreM=sc_shape.m, singleCoreN=sc_shape.n
    )
    assert not _single_core_split_shape_applicable(
        sc_shape, ctx.hardware, sc_low_occupancy
    )
    special_shape = Shape(4096, 1024, 27_392, dtype="fp16")
    special_bad = dict(sc_knowledge, baseM=128, baseN=128,
                       singleCoreM=128, singleCoreN=640)
    special_good = dict(special_bad, singleCoreN=768)
    assert not _single_core_split_shape_applicable(
        special_shape, ctx.hardware, special_bad
    )
    assert _single_core_split_shape_applicable(
        special_shape, ctx.hardware, special_good
    )

    det_boundaries = {
        "low_MN_K_at_20x384": (Shape(32, 32, 7680), True),
        "low_MN_K_below": (Shape(32, 32, 7679), False),
        "low_MN_NT_excluded": (Shape(32, 32, 8192, trans_b=True), False),
        "small_TA_NT_lower_K": (Shape(256, 256, 1000, trans_a=True), True),
        "small_TA_NT_below_K": (Shape(256, 256, 999, trans_a=True), False),
        "small_TA_NT_above_K": (Shape(256, 256, 4609, trans_a=True), False),
        "large_K_boundary": (Shape(512, 512, 27_392), True),
        "large_K_below": (Shape(512, 512, 27_391), False),
        "large_K_fixpipe_bound_rejection": (Shape(512, 513, 27_392), False),
    }
    det_checked = {}
    for name, (boundary, expected) in det_boundaries.items():
        actual = _deterministic_split_shape_applicable(boundary, ctx.hardware)
        assert actual is expected, name
        det_checked[name] = actual
    return {
        "serial_candidates": len(sc),
        "serial_candidate_specific_boundaries": {
            "singleN_511": False,
            "occupancy_below_0.70": False,
            "N_896_2048_singleN_640": False,
            "N_896_2048_singleN_768": True,
        },
        "deterministic_candidates": len(det),
        "deterministic_boundaries": det_checked,
    }


def check_legal_07(ctx: AuditContext) -> dict[str, Any]:
    cases = {
        "fp16_positive": (Shape(64, 1024, 4096, trans_b=True), True),
        "bf16_positive": (Shape(64, 1024, 4096, dtype="bf16", trans_b=True), True),
        "M_above_128": (Shape(129, 1024, 4096, trans_b=True), False),
        "fp32": (Shape(64, 1024, 4096, dtype="fp32", trans_b=True), False),
        "output_type_mismatch": (
            Shape(64, 1024, 4096, dtype="fp16", dtype_out="bf16", trans_b=True),
            False,
        ),
        "B_type_mismatch": (
            Shape(64, 1024, 4096, dtype="fp16", dtype_b="bf16", trans_b=True),
            False,
        ),
        "transA": (Shape(64, 1024, 4096, trans_a=True, trans_b=True), False),
        "B_not_transposed": (Shape(64, 1024, 4096), False),
        "bias": (
            Shape(64, 1024, 4096, trans_b=True, bias=True,
                  bias_dtype="fp16", bias_length=1024),
            False,
        ),
        "A_non_ND": (Shape(64, 1024, 4096, trans_b=True, format_a="NZ"), False),
        "B_non_ND": (Shape(64, 1024, 4096, trans_b=True, format_b="NZ"), False),
        "output_non_ND": (Shape(64, 1024, 4096, trans_b=True, format_out="NZ"), False),
        "head_conversion_required": (Shape(64, 1024, 4095, trans_b=True), False),
    }
    evidence = {}
    for name, (shape, expected) in cases.items():
        actual = family_applicability(shape, ctx.hardware)["INCREMENTAL_PATTERN"]
        assert actual is expected, name
        evidence[name] = actual
    audit = ctx.positive("INCREMENTAL_PATTERN")["candidate_audit"]
    assert audit["incremental_source_candidates"] > 0
    assert audit["incremental_source_candidates"] == audit["incremental_converted_candidates"]
    evidence["source_candidates"] = audit["incremental_source_candidates"]
    evidence["converted_candidates"] = audit["incremental_converted_candidates"]
    return evidence


def check_legal_08(ctx: AuditContext) -> dict[str, Any]:
    """Independent arithmetic verifier, separate from producer validation."""

    checked = {}
    for suffix in sorted(SUFFIX_WITNESSES):
        shape, candidate = ctx.suffix(suffix)
        fields = candidate["fields"]
        knowledge = candidate["knowledge"]
        assert all(
            value > 0
            for name, value in fields.items()
            if name != "iterateOrder"
        )
        assert fields["iterateOrder"] in (0, 1)
        k0 = 8 if shape.dtype == "fp32" and not shape.trans_a and shape.trans_b else 16
        assert fields["baseM"] % 16 == 0
        assert fields["baseN"] % 16 == 0
        assert fields["baseK"] % k0 == 0
        assert fields["usedCoreNum"] <= ctx.hardware.cores
        assert candidate["resource_bytes"]["L0A"] <= ctx.hardware.l0a
        assert candidate["resource_bytes"]["L0B"] <= ctx.hardware.l0b
        assert candidate["resource_bytes"]["L0C"] <= ctx.hardware.l0c
        assert candidate["resource_bytes"]["L1_AB"] <= ctx.hardware.l1_usable + 256
        m_tasks = ceil_div(shape.m, fields["singleCoreM"])
        n_tasks = ceil_div(shape.n, fields["singleCoreN"])
        mb = int(knowledge["l2MTileBlock"])
        nb = int(knowledge["l2NTileBlock"])
        if mb == 0 or nb == 0:
            assert mb == nb == 0
            assert int(knowledge["l2MTileCnt"]) == 1
            assert int(knowledge["l2NTileCnt"]) == 1
        else:
            assert int(knowledge["l2MTileCnt"]) == ceil_div(m_tasks, mb)
            assert int(knowledge["l2NTileCnt"]) == ceil_div(n_tasks, nb)
        checked[str(suffix)] = {
            "family": candidate["family"],
            "capacity_and_coverage": "PASS",
        }
    return {
        "validator": "independent_arithmetic_not_validate_cann_tiling",
        "checked": checked,
    }


def check_legal_09(ctx: AuditContext) -> dict[str, Any]:
    shape = Shape(2048, 2048, 512, dtype="bf16")
    records = ctx.select(shape)["candidate_audit"]["all_candidates"]
    boundary = [
        item for item in records
        if item["family"] == "BASE"
        and item["fields"]["baseM"] == 128
        and item["fields"]["baseN"] == 256
        and item["fields"]["baseK"] == 64
        and item["fields"]["depthA1"] == 16
        and item["fields"]["depthB1"] == 8
    ]
    assert boundary
    assert boundary[0]["resource_bytes"]["L1_AB"] == 512 * 1024
    assert ctx.model_hardware.capacities[MemorySpace.L1] == 512 * 1024
    assert ctx.hardware.l1_usable + 256 == 512 * 1024
    return {
        "compile_info_usable_bytes": ctx.hardware.l1_usable,
        "source_restored_bytes": 256,
        "physical_capacity_bytes": 512 * 1024,
        "boundary_packet_count": len(boundary),
    }


def check_cost_01(ctx: AuditContext) -> dict[str, Any]:
    total = 0
    for winner in ctx.all_positive().values():
        records = winner["candidate_audit"]["all_candidates"]
        assert all(item["exact_lowered_resimulation"] for item in records)
        total += len(records)
    source = (SELECTOR / "candidate_engine.py").read_text(encoding="utf-8")
    assert source.index("plan_from_cann(") < source.index("simulation = simulate(")
    return {"exact_lowered_candidates": total}


def check_cost_02(ctx: AuditContext) -> dict[str, Any]:
    checked = 0
    bottlenecks = Counter()
    for winner in ctx.all_positive().values():
        for item in winner["candidate_audit"]["all_candidates"]:
            cost = item["cost"]
            # Summing the same per-core floating values in a different order
            # can put the arithmetic mean a few ulps above the maximum.  The
            # tolerance is relative machine-roundoff, not a model allowance.
            cycle_tolerance = max(
                1e-9, cost["critical_core_cycles"] * 1e-12
            )
            assert (
                cost["critical_core_cycles"] + cycle_tolerance
                >= cost["average_core_cycles"]
            )
            expected = max(
                cost["critical_core_cycles"], cost["hbm_cycles"],
                cost["l2_cycles"], cost["shared_resource_cycles"],
            ) + cost["launch_cycles"]
            assert abs(cost["total_cycles"] - expected) <= max(1e-9, expected * 1e-12)
            bottlenecks[cost["bottleneck"]] += 1
            checked += 1
    return {"checked_candidates": checked, "bottlenecks": dict(bottlenecks)}


def check_cost_03(ctx: AuditContext) -> dict[str, Any]:
    required = {
        "BASE": {"mn_waves", "l2_window_m_tasks", "l2_window_n_tasks", "nd2nz_vector_cycles"},
        "INCREMENTAL_PATTERN": {"mn_waves", "l2_window_m_tasks", "l2_window_n_tasks", "nd2nz_vector_cycles"},
        "AL1": {"full_a_copy_bytes_per_aic", "full_a_replication_bytes", "n_tail_tasks", "b_stream_logical_bytes"},
        "BL1": {"resident_b_bytes_per_aic", "resident_b_replication_bytes", "m_tasks_per_aic_ceiling", "a_stream_logical_bytes"},
        "FIXPIPE_BL1": {"temporary_output_padding_elements_per_task", "aic_aiv_handshake_operations_per_output_task", "vector_cycles", "synchronization_cycles", "workspace_bytes"},
        "SINGLE_CORE_SPLIT_K": {"serial_k_chunks", "repeated_accumulation_chunks", "repeated_output_logical_bytes", "atomic_cycles"},
        "DETERMINISTIC_SPLIT_K": {"logical_k_chunks", "k_partition_producers", "partial_fp32_logical_bytes", "vector_reduction_adds", "reduction_cycles", "synchronization_cycles"},
    }
    evidence = {}
    for family in POSITIVE_WITNESSES:
        candidates = [
            item for item in ctx.positive(family)["candidate_audit"]["all_candidates"]
            if item["family"] == family
        ]
        assert candidates
        assert all(required[family] <= set(item["family_cost_components"]) for item in candidates)
        assert all(item["family_cost_components"]["cube_cycles"] > 0 for item in candidates)
        maxima = {
            key: max(float(item["family_cost_components"][key]) for item in candidates)
            for key in required[family]
            if isinstance(candidates[0]["family_cost_components"][key], (int, float))
        }
        if family == "FIXPIPE_BL1":
            assert maxima["vector_cycles"] > 0 and maxima["synchronization_cycles"] > 0
        if family == "BASE":
            assert maxima["nd2nz_vector_cycles"] > 0
        if family == "INCREMENTAL_PATTERN":
            assert maxima["nd2nz_vector_cycles"] == 0
        if family == "SINGLE_CORE_SPLIT_K":
            assert maxima["atomic_cycles"] > 0 and maxima["repeated_accumulation_chunks"] >= 1
        if family == "DETERMINISTIC_SPLIT_K":
            assert maxima["reduction_cycles"] > 0 and maxima["synchronization_cycles"] > 0
        evidence[family] = {"candidate_count": len(candidates), "maxima": maxima}
    return evidence


def check_cost_04(ctx: AuditContext) -> dict[str, Any]:
    hardware = ctx.model_hardware
    assert hardware.name == "Ascend910B3"
    assert hardware.core_count(Resource.CUBE) == 20
    assert hardware.core_count(Resource.VECTOR) == 40
    assert hardware.capacities[MemorySpace.L2] == 192 * 1024 * 1024
    engine_tree = ast.parse(
        (SELECTOR / "candidate_engine.py").read_text(encoding="utf-8")
    )
    argument_names = {
        argument.arg
        for node in ast.walk(engine_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    }
    assert not ({"latency", "workload_id", "shape_id", "measured_time"} & argument_names)
    return {
        "profile": hardware.name,
        "aic": hardware.core_count(Resource.CUBE),
        "aiv": hardware.core_count(Resource.VECTOR),
        "runtime_latency_inputs": [],
    }


def check_cost_05(ctx: AuditContext) -> dict[str, Any]:
    al1_shape, _ = ctx.suffix(101)
    al1_all = ctx.select(al1_shape)["candidate_audit"]["all_candidates"]
    al1 = [item for item in al1_all if item["family"] == "AL1"]
    assert {256, 448} <= {item["fields"]["baseK"] for item in al1}
    al1_cycles = {
        base_k: min(
            item["cost"]["total_cycles"] for item in al1
            if item["fields"]["baseK"] == base_k
        )
        for base_k in (256, 448)
    }
    assert al1_cycles[256] != al1_cycles[448]

    fix_shape, _ = ctx.suffix(20201)
    fix_all = ctx.select(fix_shape)["candidate_audit"]["all_candidates"]
    vector = [
        item for item in fix_all
        if item["graph_name"] == "bl1_full_load_vec_nz2nd"
    ]
    shallow = [
        item for item in vector
        if item["fields"]["depthA1"] == 1
        and item["fields"]["dbL0C"] == 1
    ]
    pipelined = [
        item for item in vector
        if item["fields"]["depthA1"] >= 2
        and item["fields"]["dbL0C"] == 2
    ]
    assert shallow and pipelined
    shallow_cycles = min(item["cost"]["total_cycles"] for item in shallow)
    pipelined_cycles = min(item["cost"]["total_cycles"] for item in pipelined)
    assert shallow_cycles != pipelined_cycles
    fixpipe = [
        item for item in fix_all
        if item["graph_name"] == "bl1_full_load_fixpipe"
    ]
    assert fixpipe
    assert min(item["cost"]["total_cycles"] for item in fixpipe) != min(
        item["cost"]["total_cycles"] for item in vector
    )

    prologue_shape = Shape(4, 48, 4096, dtype="fp32", trans_b=True)
    prologue_candidates = [
        item for item in ctx.select(prologue_shape)["candidate_audit"]["all_candidates"]
        if item["family"] == "AL1"
    ]
    assert prologue_candidates
    knowledge = dict(prologue_candidates[0]["knowledge"])
    operator = matmul(
        prologue_shape.m, prologue_shape.n, prologue_shape.k,
        prologue_shape.dtype, trans_b=True,
    )
    traffic = {}
    for launched in (3, 20):
        plan = plan_from_cann(
            prologue_shape.m, prologue_shape.n, prologue_shape.k,
            dict(knowledge, usedCoreNum=launched),
            dtype=prologue_shape.dtype, trans_b=True,
        )
        result = simulate(operator, plan, ctx.model_hardware)
        assert result.valid and result.active_cores == launched
        traffic[launched] = result.gm_read_bytes
    assert traffic[20] > traffic[3]
    return {
        "al1_basek_cycle_sensitivity": al1_cycles,
        "fixpipe_pipeline_cycle_sensitivity": {
            "shallow": shallow_cycles, "pipelined": pipelined_cycles,
        },
        "al1_all_launched_prologue_gm_read": traffic,
    }


def check_pareto_01(ctx: AuditContext) -> dict[str, Any]:
    checked = Counter()
    for winner in ctx.all_positive().values():
        audit = winner["candidate_audit"]
        all_candidates = audit["all_candidates"]
        frontier = audit["pareto_frontier"]
        for retained in frontier:
            same_family = [
                item for item in all_candidates
                if item["family"] == retained["family"]
            ]
            assert not any(
                candidate_id(other) != candidate_id(retained)
                and tuple(other["decision"]["family_rule_key"][:-1])
                <= tuple(retained["decision"]["family_rule_key"][:-1])
                and _dominates(other, retained)
                for other in same_family
            )
            checked[retained["family"]] += 1
    return {"retained_and_checked": dict(sorted(checked.items()))}


def check_pareto_02(ctx: AuditContext) -> dict[str, Any]:
    for winner in ctx.all_positive().values():
        assert winner["candidate_audit"]["fixed_topn_pruning"] is False
    candidate_source = (SELECTOR / "candidate_engine.py").read_text(encoding="utf-8")
    solver_source = (ROOT / "npu_cost_model" / "solver.py").read_text(encoding="utf-8")
    assert "top_k=1" not in candidate_source
    assert "unique_algorithm_anchors = tuple(dict.fromkeys(feasible_direct))" in solver_source
    assert "scored[:policy.top_k]" not in candidate_source
    return {
        "fixed_candidate_quota": False,
        "all_rate_independent_direct_pareto_anchors_retained": True,
    }


def check_select_01(ctx: AuditContext) -> dict[str, Any]:
    evidence = {}
    for name, winner in ctx.all_positive().items():
        frontier = winner["candidate_audit"]["pareto_frontier"]
        expected = min(frontier, key=candidate_rule_key)
        assert candidate_id(winner) == candidate_id(expected)
        assert winner["candidate_audit"]["winner_is_rule_minimum"] is True
        assert winner["candidate_audit"]["official_selector_called"] is False
        assert winner["candidate_audit"]["history_lookup"] is False
        evidence[name] = {
            "winner_family": winner["family"],
            "cycles": winner["cost"]["total_cycles"],
            "frontier_candidates": len(frontier),
        }
    return evidence


def check_select_02(ctx: AuditContext) -> dict[str, Any]:
    cases = {
        "BL1": Shape(32768, 128, 16),
        "DET": Shape(16, 17, 7680),
        "FIX_LOW": Shape(10240, 9, 8, dtype="fp32"),
        "FIX_HIGH": Shape(12288, 17, 16, dtype="fp32"),
        "AL1": Shape(12, 160, 7168, dtype="fp32", trans_b=True),
        "INCREMENTAL": Shape(64, 1024, 4096, trans_b=True),
        "BASE": Shape(2048, 2048, 512, dtype="bf16"),
    }
    winners = {name: ctx.select(shape) for name, shape in cases.items()}
    assert winners["BL1"]["family"] == "BL1"
    assert winners["BL1"]["fields"]["baseM"] == 128
    assert winners["BL1"]["fields"]["singleCoreM"] == 256
    det = winners["DET"]
    assert det["family"] == "DETERMINISTIC_SPLIT_K"
    assert (det["fields"]["stepM"], det["fields"]["stepN"]) == (3, 1)
    assert det["fields"]["iterateOrder"] == 1
    assert all(det["fields"][name] == 2 for name in ("dbL0A", "dbL0B", "dbL0C"))
    assert det["knowledge"]["l2IterateOrder"] == 0
    fix_low = winners["FIX_LOW"]
    assert fix_low["fields"]["dbL0C"] == 1
    assert fix_low["fields"]["depthA1"] == 1
    assert fix_low["knowledge"]["l2IterateOrder"] == 0
    fix_high = winners["FIX_HIGH"]
    assert fix_high["fields"]["dbL0C"] == 2
    assert fix_high["fields"]["depthA1"] == 2
    assert fix_high["knowledge"]["l2IterateOrder"] == 1
    assert winners["AL1"]["family"] == "AL1"
    assert winners["AL1"]["fields"]["usedCoreNum"] == 10
    assert winners["AL1"]["fields"]["baseK"] != 256
    assert winners["INCREMENTAL"]["family"] == "INCREMENTAL_PATTERN"
    base = winners["BASE"]
    assert base["family"] == "BASE"
    assert (base["fields"]["baseM"], base["fields"]["baseN"]) == (128, 256)
    return {
        name: {
            "family": winner["family"],
            "fields": winner["fields"],
        }
        for name, winner in winners.items()
    }


def check_packet_01(ctx: AuditContext) -> dict[str, Any]:
    result = ctx.packet()
    assert result["status"] == "MODIFIED_TILING"
    improved = result["improved"]
    assert improved is not None
    assert improved["tiling_data_bytes"] == 272
    assert len(bytes.fromhex(improved["tiling_data_hex"])) == 272
    assert result["kernel_suffix"] in INSTALLED_SUFFIXES_BY_DTYPE[
        result["request"]["dtype"]
    ]
    assert result["validation"]["host_packet"] == "PASS_272_BYTE_ABI_ROUNDTRIP"
    return {
        "bytes": improved["tiling_data_bytes"],
        "suffix": result["kernel_suffix"],
        "sha256": improved["tiling_data_sha256"],
    }


def check_packet_02(ctx: AuditContext) -> dict[str, Any]:
    result = ctx.packet()
    winner = ctx.positive("BASE")
    improved = result["improved"]
    assert improved["block_dim"] == winner["fields"]["usedCoreNum"]
    assert improved["workspace_bytes"] == winner["workspace_bytes"]
    assert result["candidate_audit"]["official_selector_called"] is False
    return {
        "blockDim": improved["block_dim"],
        "workspace_bytes": improved["workspace_bytes"],
        "derived_from": "selected_exact_lowered_candidate",
    }


CHECKS: dict[str, Callable[[AuditContext], dict[str, Any]]] = {
    "SCOPE_01": check_scope_01,
    "SCOPE_02": check_scope_02,
    "INDEP_01": check_indep_01,
    "INDEP_02": check_indep_02,
    "GEN_01": check_gen_01,
    "GEN_02": check_gen_02,
    "GEN_03": check_gen_03,
    "GEN_04": check_gen_04,
    "GEN_05": check_gen_05,
    "GEN_06": check_gen_06,
    "GEN_07": check_gen_07,
    "GEN_08": check_gen_08,
    "LEGAL_01": check_legal_01,
    "LEGAL_02": check_legal_02,
    "LEGAL_03": check_legal_03,
    "LEGAL_04": check_legal_04,
    "LEGAL_05": check_legal_05,
    "LEGAL_06": check_legal_06,
    "LEGAL_07": check_legal_07,
    "LEGAL_08": check_legal_08,
    "LEGAL_09": check_legal_09,
    "COST_01": check_cost_01,
    "COST_02": check_cost_02,
    "COST_03": check_cost_03,
    "COST_04": check_cost_04,
    "COST_05": check_cost_05,
    "PARETO_01": check_pareto_01,
    "PARETO_02": check_pareto_02,
    "SELECT_01": check_select_01,
    "SELECT_02": check_select_02,
    "PACKET_01": check_packet_01,
    "PACKET_02": check_packet_02,
}


def run_audit() -> tuple[dict[str, Any], bool]:
    context = AuditContext()
    mandatory = context.spec["mandatory_items"]
    declared_ids = [item["id"] for item in mandatory]
    expected_ids = [*CHECKS, "REPORT_01"]
    if declared_ids != expected_ids:
        raise RuntimeError(
            "audit implementation/order differs from candidate_audit_spec.json"
        )

    started = perf_counter()
    rows = []
    for item in mandatory:
        audit_id = item["id"]
        row = {
            "id": audit_id,
            "check": item["check"],
            "requirement": item["requirement"],
        }
        item_started = perf_counter()
        try:
            if audit_id == "REPORT_01":
                assert len(rows) == len(mandatory) - 1
                assert [entry["id"] for entry in rows] == declared_ids[:-1]
                evidence = {
                    "preceding_rows": len(rows),
                    "candidate_counts_attached": True,
                    "winner_attached": True,
                    "rejection_reasons_attached": True,
                }
            else:
                evidence = CHECKS[audit_id](context)
            row.update(status="PASS", evidence=evidence)
        except Exception as exception:  # continue so no later audit is skipped
            row.update(
                status="FAIL",
                error=f"{type(exception).__name__}: {exception}",
            )
        row["elapsed_ms"] = (perf_counter() - item_started) * 1000.0
        rows.append(row)

    failed = [row["id"] for row in rows if row["status"] != "PASS"]
    packet = context._packet
    report = {
        "audit_version": context.spec["audit_version"],
        "pass_semantics": context.spec["pass_semantics"],
        "status": "PASS" if not failed else "FAIL",
        "mandatory_count": len(mandatory),
        "passed_count": len(mandatory) - len(failed),
        "failed_ids": failed,
        "rows": rows,
        "candidate_evidence": context.counts(),
        "final_packet_winner": None if packet is None else {
            "family": packet.get("formula_family"),
            "suffix": packet.get("kernel_suffix"),
            "improved": packet.get("improved"),
        },
        "total_elapsed_ms": (perf_counter() - started) * 1000.0,
    }
    return report, not failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, passed = run_audit()
    encoded = json.dumps(
        report,
        sort_keys=True,
        indent=2 if args.pretty else None,
        separators=None if args.pretty else (",", ":"),
    ) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
