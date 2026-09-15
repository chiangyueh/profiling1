#!/usr/bin/env python3
"""Extract and audit every generic C220 MatMulV3 dispatch branch from source."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "colleague_matmul_v3/op_kernel/mat_mul_v3.cpp"
KEY_HEADER = ROOT / "colleague_matmul_v3/op_kernel/mat_mul_v3_tiling_key.h"

VALUES = {
    "MAT_MUL_V3_BASE_FULLLOAD": 0,
    "MAT_MUL_V3_AL1_FULLLOAD": 1,
    "MAT_MUL_V3_BL1_FULLLOAD": 2,
    "MAT_MUL_V3_BASE_SPLIT_K": 0,
    "MAT_MUL_V3_SINGLE_CORE_SPLIT_K": 2,
    "MAT_MUL_V3_DETERMINISTIC_SPLIT_K": 3,
    "MAT_MUL_V3_MULTI_CORE_SPLIT_K": 4,
    "MAT_MUL_V3_SINGLE_CORE_NKM_SPLIT_K": 5,
    "MAT_MUL_V3_SINGLE_CORE_SPLIT_K_GM_TO_L1": 6,
    "MAT_MUL_V3_BASE_FIXOPTI": 0,
    "MAT_MUL_V3_BASE_ENABLE_ALIGNOUT": 1,
    "MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT": 2,
    "MAT_MUL_V3_MIXND2NZ_TRUE": 0,
    "MAT_MUL_V3_MIXND2NZ_FALSE": 1,
    "MAT_MUL_V3_MIXND2NZ_TRUE_PARALLEL": 2,
    "MAT_MUL_V3_K_NOT_SHIFT": 0,
    "MAT_MUL_V3_K_SHIFT": 1,
}

AXES = {
    "LOADMODE": ("load", 100),
    "SPLITCOREMODE": ("split", 10),
    "FIXOPTI": ("fix", 10_000),
    "MIXND2NZ": ("mix", 1),
    "SPECIALOPT": ("special", 100_000),
}


def _generic_region(text: str) -> tuple[str, int]:
    layout_marker = "#elif defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ"
    layout_start = text.index(layout_marker)
    start = text.index("\n#else\n", layout_start) + len("\n#else\n")
    end = text.index("\n#endif\n}", start)
    return text[start:end], text[:start].count("\n") + 1


def _balanced_call(text: str, start: int) -> str:
    opening = text.index("(", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise RuntimeError("unterminated dispatch invocation")


def _first_argument(call: str) -> str:
    inner = call[call.index("(") + 1:-1]
    depth = 0
    for index, char in enumerate(inner):
        if char in "(<":
            depth += 1
        elif char in ")>":
            depth -= 1
        elif char == "," and depth == 0:
            return inner[:index].strip()
    raise RuntimeError(f"dispatch invocation lacks arguments: {call}")


def extract() -> list[dict]:
    text = SOURCE.read_text(encoding="utf-8")
    region, first_line = _generic_region(text)
    starts = [match.start() for match in re.finditer(r"\bMMV3_IMPL(?:_C_CLASS|_CLASS)?\s*\(", region)]
    records = []
    for ordinal, start in enumerate(starts):
        condition_start = region.rfind("if constexpr", 0, start)
        if condition_start < 0:
            raise RuntimeError(f"dispatch {ordinal} has no constexpr condition")
        condition = region[condition_start:start]
        call = _balanced_call(region, start)
        values = {name: 0 for name, _ in AXES.values()}
        for symbol, (name, _) in AXES.items():
            match = re.search(rf"\b{symbol}\s*==\s*(MAT_MUL_V3_[A-Z0-9_]+)", condition)
            if match:
                if match.group(1) not in VALUES:
                    raise RuntimeError(f"unknown value macro {match.group(1)}")
                values[name] = VALUES[match.group(1)]
        suffix = sum(values[name] * multiplier for name, multiplier in AXES.values())
        source_line = first_line + region[:start].count("\n")
        records.append({
            "ordinal": ordinal,
            "source_line": source_line,
            "suffix": suffix,
            **values,
            "macro": call[:call.index("(")].strip(),
            "kernel_class": _first_argument(call),
            "invocation": " ".join(call.split()),
            "preprocessor_guards": [
                line.strip() for line in condition.splitlines()
                if line.lstrip().startswith("#")
            ],
        })
    suffixes = [record["suffix"] for record in records]
    if len(suffixes) != len(set(suffixes)):
        raise RuntimeError(f"generic C220 dispatch has duplicate suffixes: {suffixes}")
    return records


def _registered_suffixes() -> set[int]:
    cmake = (ROOT / "cmake_npu/CMakeLists.txt").read_text(encoding="utf-8")
    installed = {
        int(value) for value in re.findall(
            r"register_direct_matmul\([^\s]+\s+[^\s]+\s+(\d+)\s+\d+\)", cmake
        )
    }
    bundled = {
        int(value) for value in re.findall(
            r"register_c220_direct_matmul\([^\s]+\s+[^\s]+\s+(\d+)\s+\d+\)", cmake
        )
    }
    return installed | bundled


def _build_suffixes() -> set[int]:
    script = (ROOT / "scripts/build_all.sh").read_text(encoding="utf-8")
    return {
        int(value) for value in re.findall(r"direct_matmul_kernel_[a-z0-9]+_(\d+)", script)
    }


def audit(manifest: Path | None = None) -> dict:
    records = extract()
    source_suffixes = {record["suffix"] for record in records}
    registered = _registered_suffixes()
    build = _build_suffixes()
    missing_registration = sorted(source_suffixes - registered)
    missing_build = sorted(source_suffixes - build)
    manifest_suffixes = None
    missing_manifest = []
    if manifest is not None:
        with manifest.open(newline="", encoding="utf-8") as stream:
            manifest_suffixes = {
                int(row["kernel_suffix"]) for row in csv.DictReader(stream)
            }
        missing_manifest = sorted(source_suffixes - manifest_suffixes)
    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": __import__("hashlib").sha256(SOURCE.read_bytes()).hexdigest(),
        "dispatch_branch_count": len(records),
        "kernel_class_count": len({record["kernel_class"] for record in records}),
        "source_suffixes": sorted(source_suffixes),
        "registered_suffixes": sorted(source_suffixes & registered),
        "build_suffixes": sorted(source_suffixes & build),
        "missing_registration": missing_registration,
        "missing_build": missing_build,
        "manifest_suffixes": (
            sorted(source_suffixes & manifest_suffixes)
            if manifest_suffixes is not None else None
        ),
        "missing_manifest": missing_manifest,
        "branches": records,
    }
    if missing_registration or missing_build or missing_manifest:
        raise RuntimeError(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    payload = {
        "source": str(SOURCE.relative_to(ROOT)),
        "branches": extract(),
    } if args.extract_only else audit(args.manifest)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
