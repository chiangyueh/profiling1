#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
BASELINE_PACKAGE_ROOT = ROOT / "baseline_core"
sys.path.insert(0, str(BASELINE_PACKAGE_ROOT))

from matmul_reconstruction._core.abi import (  # noqa: E402
    VECTOR_NAMES,
    pack_packet,
    unpack_packet,
)
from matmul_reconstruction._core.initializer import CUBE_FIELDS  # noqa: E402
from matmul_reconstruction._core.key import validate_primary_selection  # noqa: E402
from matmul_reconstruction.api import select as baseline_select  # noqa: E402

from formula_rules import CompileInfo, Hardware, Shape, TilingRuleError, solve  # noqa: E402


HARDWARE = {
    "aicNum": 20,
    "aivNum": 40,
    "ubSize": 196352,
    "l1Size": 524032,
    "l2Size": 201326592,
    "l0CSize": 131072,
    "l0ASize": 65536,
    "l0BSize": 65536,
    "btSize": 1024,
    "supportL0c2out": True,
    "supportL12BtBf16": False,
    "cubeFreq": 0,
    "npuArch": 220,
    "socVersion": 220,
    "socVersionStr": "Ascend910B3",
    "cannVersion": "8.1.RC1",
}

FORMULA_HARDWARE = Hardware(
    cores=20,
    l0a=65536,
    l0b=65536,
    l0c=131072,
    l1_usable=524032,
    l2=201326592,
    ub=196352,
    bt=1024,
)

FAMILY_NAME = {
    "BASE": "BASE",
    "AL1": "AL1_FULL_LOAD",
    "BL1": "BL1_FULL_LOAD",
    "SINGLE_CORE_SPLIT_K": "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K": "DETERMINISTIC_SPLIT_K",
}

FAMILY_KEY_PARTS = {
    "BASE": (0, 0),
    "BL1": (2, 0),
    "SINGLE_CORE_SPLIT_K": (0, 2),
    "DETERMINISTIC_SPLIT_K": (0, 3),
}


def align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def make_request(m, k, n, dtype, trans_a, trans_b):
    return {
        "M": m,
        "N": n,
        "K": k,
        "dtype": dtype,
        "output_dtype": dtype,
        "layoutA": "ND",
        "layoutB": "ND",
        "layoutC": "ND",
        "transA": trans_a,
        "transB": trans_b,
        "bias": False,
        "hf32": False,
        "forceGrpAccForFp32": False,
    }


def summarize(result):
    return {
        "selected_family": result["selected_family"],
        "tiling_key": result["tiling_key"],
        "block_dim": result["block_dim"],
        "workspace_bytes": result["workspace_bytes"],
        "tiling_data_bytes": result["tiling_data_bytes"],
        "tiling_data_hex": result["tiling_data_hex"],
    }


def unchanged_result(request, baseline, reason):
    baseline_output = {
        "selected_family": baseline["selected_family"],
        "tiling_key": baseline["tiling_key"],
        "block_dim": baseline["blockDim"],
        "workspace_bytes": baseline["workspace_bytes"],
        "tiling_data_bytes": baseline["raw_buffer_bytes"],
        "tiling_data_hex": baseline["raw_buffer_hex"],
        "tiling_data_sha256": baseline["raw_buffer_sha256"],
    }
    return {
        "status": "UNCHANGED_BASELINE",
        "request": request,
        "baseline": baseline_output,
        "improved": dict(baseline_output),
        "baseline_equivalent": True,
        "changed_rules": [],
        "changed_fields": {},
        "skip_reason": reason,
        "validation": {
            "host_packet": "BASELINE_EXACT",
            "kernel_key_domain": "BASELINE_EXACT",
            "local_capacity": "BASELINE_EXACT",
            "npu_performance": "NOT_MEASURED",
        },
    }


def workspace_for(family, fields, shape):
    rpc = 20 * 1024 * 1024
    if family == "SINGLE_CORE_SPLIT_K":
        width = 4 if shape.dtype == "fp32" else 2
        return rpc + shape.m * align_up(shape.n, 256 // width) * 4
    if family == "DETERMINISTIC_SPLIT_K":
        return rpc + fields["usedCoreNum"] * fields["singleCoreM"] * fields["singleCoreN"] * 8
    return rpc


def validate_complete_candidate(payload, raw, family, fields, shape):
    """Reject an internally inconsistent host packet before it reaches an NPU run."""
    if len(raw) != 272:
        raise ValueError("candidate packet is not the installed 8.1 272-byte ABI")
    if unpack_packet(raw, "legacy272_cube200") != payload:
        raise ValueError("candidate packet does not round-trip through the 8.1 ABI")

    load_mode, split_mode = FAMILY_KEY_PARTS[family]
    validate_primary_selection((load_mode, split_mode, 0, 0, 0, 0))

    if fields["usedCoreNum"] < 1 or fields["usedCoreNum"] > FORMULA_HARDWARE.cores:
        raise ValueError("usedCoreNum exceeds the fixed 20-AIC target")
    for name in ("baseM", "baseN", "baseK"):
        if fields[name] < 16 or fields[name] % 16:
            raise ValueError(name + " is not a positive 16-element multiple")
    if fields["dbL0C"] not in (1, 2):
        raise ValueError("dbL0C is outside the installed kernel domain")

    dtype_bytes = shape.d
    l0a = 2 * fields["baseM"] * fields["baseK"] * dtype_bytes
    l0b = 2 * fields["baseN"] * fields["baseK"] * dtype_bytes
    l0c = fields["dbL0C"] * fields["baseM"] * fields["baseN"] * 4
    l1 = (
        fields["depthA1"] * fields["baseM"]
        + fields["depthB1"] * fields["baseN"]
    ) * fields["baseK"] * dtype_bytes
    if l0a > FORMULA_HARDWARE.l0a or l0b > FORMULA_HARDWARE.l0b:
        raise ValueError("candidate exceeds L0A/L0B capacity")
    if l0c > FORMULA_HARDWARE.l0c:
        raise ValueError("candidate exceeds L0C capacity")
    if l1 > FORMULA_HARDWARE.l1_usable:
        raise ValueError("candidate exceeds usable L1 capacity")


def changed_rules(family, cube_changes, l2_changes, metadata_changes):
    changed = set(cube_changes)
    rules = []
    if family == "BASE" and changed & {"baseM", "baseN"}:
        rules.append("R01_GEOMETRIC_BASE_MN")
    if family == "BASE" and changed & {"baseM", "baseN", "singleCoreM", "singleCoreN", "usedCoreNum"}:
        rules.append("R02_WAVE_TAIL_REDISTRIBUTION")
    if family == "BASE" and "baseK" in changed:
        rules.append("R03_BASE_K_AFTER_MN")
    if family == "BASE" and changed & {"stepKa", "stepKb", "depthA1", "depthB1"}:
        rules.append("R04_JOINT_L1_STEP_DEPTH")
    family_rule = {
        "AL1": "R05_AL1_RESIDENCY",
        "BL1": "R06_BL1_RESIDENCY",
        "SINGLE_CORE_SPLIT_K": "R07_SINGLE_CORE_SPLIT_K_GRID",
        "DETERMINISTIC_SPLIT_K": "R08_DETERMINISTIC_SPLIT_K",
    }.get(family)
    if family_rule and (changed or metadata_changes):
        rules.append(family_rule)
    if family in ("SINGLE_CORE_SPLIT_K", "DETERMINISTIC_SPLIT_K") and changed & {"singleCoreM", "singleCoreN"}:
        rules.append("R09_SPLIT_K_L2_LONG_AXIS")
    if family == "BASE" and l2_changes:
        rules.append("R10_BASE_L2_MACRO")
    if "iterateOrder" in changed or "calOrder" in l2_changes:
        rules.append("R11_TRAVERSAL_ORDER")
    return rules


def generate(m, k, n, dtype="fp16", trans_a=False, trans_b=False):
    request = make_request(m, k, n, dtype, trans_a, trans_b)
    baseline = baseline_select(
        request,
        HARDWARE,
        source_profile="installed_81",
        trace=False,
    )

    run_info = baseline["tilingData"]["matmulRunInfo"]
    vector = {name: baseline["tilingData"][name] for name in VECTOR_NAMES}
    if run_info["nd2nzA"] or run_info["nd2nzB"] or any(vector.values()):
        return unchanged_result(request, baseline, "BASELINE_HEAD_CONVERSION")

    shape = Shape(
        m=m,
        n=n,
        k=k,
        dtype=dtype,
        trans_a=trans_a,
        trans_b=trans_b,
    )
    compile_info = CompileInfo(
        aicore_arch=220,
        support_l0c2out=True,
        support_l12_bt_bf16=False,
        orig_dtype_x1=dtype,
        orig_dtype_x2=dtype,
        orig_dtype_y=dtype,
        orig_dtype_bias=dtype,
        format_x1="ND",
        format_x2="ND",
        format_y="ND",
        total_ub_size=196352,
    )
    try:
        formula = solve(shape, FORMULA_HARDWARE, compile_info)
    except (TilingRuleError, ValueError, ZeroDivisionError) as exc:
        return unchanged_result(request, baseline, f"FORMULA_DOMAIN: {exc}")

    if formula["family"] == "AL1":
        # The 8.1 primary key set has no standalone (LOADMODE=1,
        # SPLITCOREMODE=0) kernel. LOADMODE=1 is legal only together with the
        # distinct single-core Split-K+AL1 contract (split mode 2), which these
        # formulas do not implement. Never disguise standalone AL1 fields as
        # that different kernel family.
        return unchanged_result(
            request,
            baseline,
            "FORMULA_FAMILY_NOT_EXPOSED_BY_81_KERNEL_KEY: standalone AL1",
        )

    fields = formula["fields"]
    cube = dict(baseline["initializer"]["cube"])
    for name in (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0C", "iterateOrder",
    ):
        cube[name] = int(fields[name])

    l2_formula = formula["l2"]
    if l2_formula.get("ignored_by_this_family"):
        # Do not manufacture a byte-level difference in fields that this
        # kernel family explicitly ignores.
        l2 = dict(baseline["tilingData"]["tileL2cacheTiling"])
    else:
        l2 = {
            "mTileCntL2": int(l2_formula["mTile"]),
            "nTileCntL2": int(l2_formula["nTile"]),
            "mTileBlock": int(l2_formula["mTileBlock"]),
            "nTileBlock": int(l2_formula["nTileBlock"]),
            "calOrder": int(l2_formula["calOrder"]),
        }
    run = {
        "transA": int(trans_a),
        "transB": int(trans_b),
        "nd2nzA": 0,
        "nd2nzB": 0,
        "isHf32": 0,
    }
    payload = {
        "cube_words": [int(cube[name]) & 0xFFFFFFFF for name in CUBE_FIELDS],
        "tileL2cacheTiling": l2,
        "matmulRunInfo": run,
        "l2CacheFlag": 0,
        "vector": {name: 0 for name in VECTOR_NAMES},
        "padding_hex": {"220": "00000000", "244": "00000000", "252": "00000000"},
    }
    raw = pack_packet(payload, "legacy272_cube200")
    validate_complete_candidate(payload, raw, formula["family"], fields, shape)
    load_mode, split_mode = FAMILY_KEY_PARTS[formula["family"]]
    key = 10**19 + 1 + 100 * load_mode + 10 * split_mode
    workspace = workspace_for(formula["family"], fields, shape)
    improved = {
        "selected_family": FAMILY_NAME[formula["family"]],
        "tiling_key": key,
        "block_dim": int(fields["usedCoreNum"]),
        "workspace_bytes": workspace,
        "tiling_data_bytes": len(raw),
        "tiling_data_hex": raw.hex(),
        "tiling_data_sha256": hashlib.sha256(raw).hexdigest(),
    }
    baseline_output = {
        "selected_family": baseline["selected_family"],
        "tiling_key": baseline["tiling_key"],
        "block_dim": baseline["blockDim"],
        "workspace_bytes": baseline["workspace_bytes"],
        "tiling_data_bytes": baseline["raw_buffer_bytes"],
        "tiling_data_hex": baseline["raw_buffer_hex"],
        "tiling_data_sha256": baseline["raw_buffer_sha256"],
    }

    baseline_cube = baseline["tilingData"]["matmulTiling"]
    cube_changes = {
        name: {"baseline": int(baseline_cube[name]), "improved": int(cube[name])}
        for name in CUBE_FIELDS
        if int(baseline_cube[name]) != int(cube[name])
    }
    baseline_l2 = baseline["tilingData"]["tileL2cacheTiling"]
    l2_changes = {
        name: {"baseline": int(baseline_l2[name]), "improved": int(l2[name])}
        for name in l2
        if int(baseline_l2[name]) != int(l2[name])
    }
    metadata_changes = {
        name: {"baseline": baseline_output[name], "improved": improved[name]}
        for name in ("selected_family", "tiling_key", "block_dim", "workspace_bytes")
        if baseline_output[name] != improved[name]
    }
    all_changes = dict(cube_changes)
    all_changes.update({f"l2.{name}": value for name, value in l2_changes.items()})
    all_changes.update(metadata_changes)
    equivalent = baseline["raw_buffer_hex"] == raw.hex() and not metadata_changes
    if equivalent:
        return unchanged_result(request, baseline, "FORMULAS_PRODUCED_BASELINE_VALUES")

    return {
        "status": "MODIFIED_TILING",
        "request": request,
        "baseline": baseline_output,
        "improved": improved,
        "baseline_equivalent": False,
        "changed_rules": changed_rules(
            formula["family"], cube_changes, l2_changes, metadata_changes
        ),
        "changed_fields": all_changes,
        "formula_family": formula["family"],
        "resource_bytes": formula["resource_bytes"],
        "validation": {
            "host_packet": "PASS_272_BYTE_ABI_ROUNDTRIP",
            "kernel_key_domain": "PASS_INSTALLED_81_PRIMARY_KEY",
            "local_capacity": "PASS_FIXED_910B3_CAPACITIES",
            "npu_performance": "NOT_MEASURED",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--trans-a", action="store_true")
    parser.add_argument("--trans-b", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    result = generate(args.m, args.k, args.n, args.dtype, args.trans_a, args.trans_b)
    print(json.dumps(result, sort_keys=True, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")))


if __name__ == "__main__":
    main()
