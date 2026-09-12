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
from matmul_reconstruction._core.initializer import (  # noqa: E402
    CUBE_FIELDS,
    initialize_cube,
)

from formula_rules import CompileInfo, Hardware, Shape, solve  # noqa: E402


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
    "FIXPIPE_BL1": "BL1_FULL_LOAD_FIXPIPE",
    "AL1": "AL1_FULL_LOAD",
    "BL1": "BL1_FULL_LOAD",
    "SINGLE_CORE_SPLIT_K": "SINGLE_CORE_SPLIT_K",
    "DETERMINISTIC_SPLIT_K": "DETERMINISTIC_SPLIT_K",
    "INCREMENTAL_PATTERN": "INCREMENTAL_PATTERN",
}

FAMILY_KEY_PARTS = {
    "BASE": (0, 0, 0),
    "AL1": (1, 0, 0),
    "BL1": (2, 0, 0),
    "SINGLE_CORE_SPLIT_K": (0, 2, 0),
    "DETERMINISTIC_SPLIT_K": (0, 3, 0),
    "INCREMENTAL_PATTERN": (0, 0, 0),
}

# These are the branches actually dispatched by the installed CANN 8.1
# DAV-C220 mat_mul_v3.cpp.  Later mode-4/5/6 kernels are deliberately absent.
KERNEL_SUFFIXES = {
    "fp16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "bf16": frozenset((0, 1, 20, 21, 30, 31, 200, 201, 10200, 10201)),
    "fp32": frozenset((0, 1, 20, 21, 30, 31, 101, 200, 201,
                       10200, 10201, 20201)),
}

KERNEL_VARIANT = {
    0: "BASE_UNALIGNED", 1: "BASE_ALIGNED",
    20: "SINGLE_CORE_SPLIT_K_UNALIGNED",
    21: "SINGLE_CORE_SPLIT_K_ALIGNED",
    30: "DETERMINISTIC_SPLIT_K_UNALIGNED",
    31: "DETERMINISTIC_SPLIT_K_ALIGNED",
    101: "AL1_FULL_LOAD_ALIGNED",
    200: "BL1_FULL_LOAD_UNALIGNED", 201: "BL1_FULL_LOAD_ALIGNED",
    10200: "BL1_FULL_LOAD_FIXPIPE_UNALIGNED",
    10201: "BL1_FULL_LOAD_FIXPIPE_ALIGNED",
    20201: "BL1_FULL_LOAD_VEC_NZ2ND",
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


def nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
    aligned_loops = (n_aligned + base_n - 1) // base_n
    value_loops = (n_value + base_n - 1) // base_n
    if aligned_loops == value_loops:
        return False
    mask = (1 << 64) - 1
    completed = (n_value // base_n - 1) & mask
    remaining = (n_aligned - ((completed * base_n) & mask)) & mask
    return remaining * base_d > FORMULA_HARDWARE.ub // 2 // width


def nd2nz_geometry(width, n_value, d_value, used_cores):
    """CANN 8.1 vector head geometry, evaluated from this candidate only."""
    vector_cores = max(2 * used_cores, 1)
    threshold = 2048 // width
    c0 = 32 // width
    n_aligned = align_up(n_value, 16)
    d_aligned = align_up(d_value, c0)
    if d_value <= threshold:
        base_d = max(d_aligned, 1)
        initial_n = FORMULA_HARDWARE.ub // 2 // width // base_d
        rounds = max(
            (align_up(n_aligned, vector_cores) // vector_cores + initial_n - 1)
            // initial_n,
            1,
        )
        base_n = max(
            ((n_aligned + vector_cores - 1) // vector_cores + rounds - 1)
            // rounds,
            16,
        )
        while base_n > 16 and nd2nz_overflow(
            n_aligned, n_value, base_n, base_d, width
        ):
            base_n -= 1
        return base_n, base_d

    best_tail = 0
    best_n = 16
    best_d = 4096 // width
    for base_bytes in (6144, 4096, 2048):
        base_d = max(min(d_aligned, base_bytes // width), 1)
        d_loops = (d_aligned + base_d - 1) // base_d
        d_tail = d_aligned % base_d
        if 0 < d_tail < 512 // width:
            if base_d * width == 6144:
                continue
            d_loops -= 1
            base_d = max(
                align_up((d_aligned + d_loops - 1) // d_loops, c0), 1
            )
        base_n = max(FORMULA_HARDWARE.ub // 2 // width // base_d, 16)
        if base_n * base_d * width * 2 > FORMULA_HARDWARE.ub:
            continue
        if nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
            continue
        n_loops = (n_aligned + base_n - 1) // base_n
        tail = n_loops * d_loops % vector_cores
        while base_n > 16:
            if nd2nz_overflow(n_aligned, n_value, base_n, base_d, width):
                base_n -= 1
                n_loops = (n_aligned + base_n - 1) // base_n
                tail = n_loops * d_loops % vector_cores
                continue
            if tail == 0:
                return base_n, base_d
            if tail > best_tail:
                best_tail, best_n, best_d = tail, base_n, base_d
            base_n -= 1
            n_loops = (n_aligned + base_n - 1) // base_n
            tail = n_loops * d_loops % vector_cores
    return best_n, best_d


def validate_complete_candidate(payload, raw, suffix, fields, shape):
    """Reject an internally inconsistent host packet before it reaches an NPU run."""
    if len(raw) != 272:
        raise ValueError("candidate packet is not the installed 8.1 272-byte ABI")
    if unpack_packet(raw, "legacy272_cube200") != payload:
        raise ValueError("candidate packet does not round-trip through the 8.1 ABI")

    if suffix not in KERNEL_SUFFIXES[shape.dtype]:
        raise ValueError(
            f"installed CANN 8.1 has no {shape.dtype} direct branch for suffix {suffix}"
        )

    if fields["usedCoreNum"] < 1 or fields["usedCoreNum"] > FORMULA_HARDWARE.cores:
        raise ValueError("usedCoreNum exceeds the fixed 20-AIC target")
    for name in ("baseM", "baseN", "baseK"):
        if fields[name] < 16 or fields[name] % 16:
            raise ValueError(name + " is not a positive 16-element multiple")
    if fields["dbL0C"] not in (1, 2):
        raise ValueError("dbL0C is outside the installed kernel domain")

    dtype_bytes = shape.d
    l0a = fields["dbL0A"] * fields["baseM"] * fields["baseK"] * dtype_bytes
    l0b = fields["dbL0B"] * fields["baseN"] * fields["baseK"] * dtype_bytes
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


def generate(
    m, k, n, dtype="fp16", trans_a=False, trans_b=False,
    *, required_suffix=None, required_family=None,
):
    request = make_request(m, k, n, dtype, trans_a, trans_b)
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
    # A generation or legality failure is fatal.  Falling back to the
    # official selector here would turn an absent improved result into an
    # apparently valid baseline packet and violate decision independence.
    formula = solve(shape, FORMULA_HARDWARE, compile_info)
    selection_basis = "INDEPENDENT_GLOBAL_HARDWARE_COST_MINIMUM"
    if required_suffix is not None or required_family is not None:
        required_suffix = (
            None if required_suffix is None else int(required_suffix)
        )
        matching = [
            candidate
            for candidate in formula["candidate_audit"].get(
                "all_candidates", ()
            )
            if (
                (required_suffix is None or int(candidate["kernel_suffix"]) == required_suffix)
                and (required_family is None or candidate["family"] == required_family)
            )
        ]
        if not matching:
            # solve omits bulky records for the deployment path.  Re-run the
            # same independent engine with records only for explicit branch
            # execution coverage; this still uses no official packet/data.
            from candidate_engine import generate_and_select
            audited = generate_and_select(
                shape, FORMULA_HARDWARE, compile_info,
                include_audit_records=True,
            )
            matching = [
                candidate
                for candidate in audited["candidate_audit"]["all_candidates"]
                if (
                    (required_suffix is None or int(candidate["kernel_suffix"]) == required_suffix)
                    and (required_family is None or candidate["family"] == required_family)
                )
            ]
            formula = audited
        if not matching:
            raise ValueError(
                "no legal independent candidate for requested branch "
                f"suffix={required_suffix} family={required_family}"
            )
        selected = min(
            matching,
            key=lambda candidate: (
                candidate["cost"]["total_cycles"],
                candidate["family"],
                tuple(sorted(candidate["knowledge"].items())),
            ),
        )
        emitted_is_global_winner = (
            selected["family"] == formula["family"]
            and selected["knowledge"] == formula["knowledge"]
        )
        formula = dict(formula)
        formula["candidate_audit"] = dict(formula["candidate_audit"])
        formula["candidate_audit"].update({
            "emitted_candidate_role": "branch_probe",
            "emitted_candidate_is_global_minimum": emitted_is_global_winner,
            "required_branch_suffix": required_suffix,
            "required_branch_family": required_family,
        })
        formula.update({
            "family": selected["family"],
            "fields": selected["fields"],
            "l2": selected["l2"],
            "conversion_a": selected["conversion_a"],
            "conversion_b": selected["conversion_b"],
            "fix_mode": selected["fix_mode"],
            "resource_bytes": selected["resource_bytes"],
            "workspace_bytes": selected["workspace_bytes"],
            "cost": selected["cost"],
        })
        selection_basis = "INDEPENDENT_BRANCH_COVERAGE_MINIMUM"

    fields = formula["fields"]
    initializer = initialize_cube(request, HARDWARE, trace=False)
    if initializer["return_code"] != 0:
        raise ValueError(
            "independent MultiCore initializer rejected the improved request"
        )
    cube = dict(initializer["cube"])
    for name in (
        "usedCoreNum", "singleCoreM", "singleCoreN", "singleCoreK",
        "baseM", "baseN", "baseK", "depthA1", "depthB1", "stepM",
        "stepN", "stepKa", "stepKb", "dbL0C", "iterateOrder",
    ):
        cube[name] = int(fields[name])
    # Every analytic rule in this selector explicitly budgets double-buffered
    # L0 A/B footprints.  These flags therefore follow the new fields rather
    # than being copied from the official selector output.
    cube["dbL0A"] = int(fields["dbL0A"])
    cube["dbL0B"] = int(fields["dbL0B"])
    cube["dbL0C"] = int(fields["dbL0C"])

    l2_formula = formula["l2"]
    if l2_formula.get("ignored_by_this_family"):
        l2 = {
            "mTileCntL2": 1,
            "nTileCntL2": 1,
            "mTileBlock": 1,
            "nTileBlock": 1,
            "calOrder": 0,
        }
    else:
        l2 = {
            "mTileCntL2": int(l2_formula["mTile"]),
            "nTileCntL2": int(l2_formula["nTile"]),
            "mTileBlock": int(l2_formula["mTileBlock"]),
            "nTileBlock": int(l2_formula["nTileBlock"]),
            "calOrder": int(l2_formula["calOrder"]),
        }
    conversion_a = bool(formula["conversion_a"])
    conversion_b = bool(formula["conversion_b"])
    vector = {name: 0 for name in VECTOR_NAMES}
    if conversion_a:
        conversion_n, conversion_d = ((k, m) if trans_a else (m, k))
        vector["baseAN"], vector["baseAD"] = nd2nz_geometry(
            shape.d, conversion_n, conversion_d, fields["usedCoreNum"]
        )
    if conversion_b:
        conversion_n, conversion_d = ((n, k) if trans_b else (k, n))
        vector["baseBN"], vector["baseBD"] = nd2nz_geometry(
            shape.d, conversion_n, conversion_d, fields["usedCoreNum"]
        )
    run = {
        "transA": int(trans_a),
        "transB": int(trans_b),
        "nd2nzA": int(conversion_a),
        "nd2nzB": int(conversion_b),
        "isHf32": 0,
    }
    payload = {
        "cube_words": [int(cube[name]) & 0xFFFFFFFF for name in CUBE_FIELDS],
        "tileL2cacheTiling": l2,
        "matmulRunInfo": run,
        # The installed 8.1 host writes this storage word before it computes
        # its advisory flag.  Zero is the actual packet contract, not a value
        # fixed by the installed packet ABI, not borrowed from another tiling.
        "l2CacheFlag": 0,
        "vector": vector,
        "padding_hex": {"220": "00000000", "244": "00000000", "252": "00000000"},
    }
    raw = pack_packet(payload, "legacy272_cube200")
    fix_mode = int(formula.get("fix_mode", 0))
    if formula["family"] == "FIXPIPE_BL1":
        load_mode, split_mode = 2, 0
    else:
        load_mode, split_mode, expected_fix = FAMILY_KEY_PARTS[formula["family"]]
        if fix_mode != expected_fix:
            raise ValueError("formula family emitted an inconsistent FixOpt mode")
    mix_mode = 0 if conversion_a or conversion_b else 1
    suffix = fix_mode * 10000 + load_mode * 100 + split_mode * 10 + mix_mode
    validate_complete_candidate(payload, raw, suffix, fields, shape)
    key = 10**19 + suffix
    workspace = int(formula["workspace_bytes"])
    improved = {
        "selected_family": (
            "BL1_FULL_LOAD_VEC_NZ2ND"
            if fix_mode == 2 else FAMILY_NAME[formula["family"]]
        ),
        "kernel_variant": KERNEL_VARIANT[suffix],
        "tiling_key": key,
        "block_dim": int(fields["usedCoreNum"]),
        "workspace_bytes": workspace,
        "tiling_data_bytes": len(raw),
        "tiling_data_hex": raw.hex(),
        "tiling_data_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {
        "status": "MODIFIED_TILING",
        "request": request,
        "improved": improved,
        "selection_basis": selection_basis,
        "formula_family": formula["family"],
        "kernel_suffix": suffix,
        "resource_bytes": formula["resource_bytes"],
        "cost": formula["cost"],
        "candidate_audit": formula["candidate_audit"],
        "npu_eligible": True,
        "path_coverage": {
            "initializer": "ABI_DEFAULTS_ONLY; NO_FAMILY_OR_TILE_SELECTION",
            "family_selection": "ALL_APPLICABLE_FAMILIES_SCORED_BY_EXACT_LOWERED_CRITICAL_PATH",
            "base_and_parent_geometry": "MODIFIED",
            "k_l1_pipeline": "MODIFIED",
            "l2_partition": (
                "NOT_CONSUMED_BY_KERNEL_FAMILY"
                if l2_formula.get("ignored_by_this_family") else "MODIFIED"
            ),
            "input_conversion_predicate": "RETAINED_AND_MARKED",
            "input_conversion_geometry": "RETAINED_AND_MARKED",
            "kernel_implementation": "RETAINED_CANN_81_BRANCH_AND_MARKED",
            "abi_layout": "RETAINED_272_BYTE_ABI_AND_MARKED",
            "official_selector_as_seed": "FORBIDDEN_AND_NOT_USED",
            "history_or_runtime_kb": "FORBIDDEN_AND_NOT_USED",
            "pareto_pruning": "COMPLETE_WITHIN_FAMILY_NO_FIXED_TOPN",
            "emitted_candidate_role": (
                "BRANCH_PROBE_NOT_CLAIMED_AS_GLOBAL_WINNER"
                if required_suffix is not None or required_family is not None
                else "INDEPENDENT_GLOBAL_WINNER"
            ),
        },
        "validation": {
            "host_packet": "PASS_272_BYTE_ABI_ROUNDTRIP",
            "kernel_key_domain": "PASS_INSTALLED_81_DTYPE_AWARE_DISPATCH",
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
