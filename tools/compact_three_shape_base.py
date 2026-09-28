#!/usr/bin/env python3

import csv
import json
import sys


FIELDS = [
    "shape",
    "official_tiling",
    "candidate_tiling",
    "official_core",
    "candidate_core",
    "official_ms",
    "candidate_ms",
    "delta_pct",
    "correctness",
]


def tiling_text(value):
    if not isinstance(value, dict):
        return ""
    base = "x".join(str(value.get(name, "")) for name in ("base_m", "base_n", "base_k"))
    steps = "x".join(str(value.get(name, "")) for name in ("step_ka", "step_kb"))
    depths = "x".join(str(value.get(name, "")) for name in ("depth_a1", "depth_b1"))
    window = "x".join(str(value.get(name, "")) for name in ("l2_m_block", "l2_n_block"))
    order = str(value.get("l2_order", ""))
    return f"{base}/step={steps}/depth={depths}/l2={window}/order={order}"


writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
writer.writeheader()
sys.stdout.flush()
for raw in sys.stdin:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        print(raw.rstrip(), file=sys.stderr)
        continue
    if obj.get("summary") is True:
        prefix = "fatal:" if obj.get("passed", 0) == 0 else "# summary"
        print(
            f"{prefix} "
            f"inputs={obj.get('inputs', '')} "
            f"official_target={obj.get('official_target', '')} "
            f"candidate_selected={obj.get('candidate_selected', '')} "
            f"passed={obj.get('passed', '')} "
            f"failed={obj.get('failed', '')} "
            f"official_failed={obj.get('official_failed', '')} "
            f"candidate_tiling_failed={obj.get('candidate_tiling_failed', '')} "
            f"first_failure_stage={obj.get('first_failure_stage', '')} "
            f"first_failure_shape={obj.get('first_failure_shape', '')} "
            f"first_failure_rc={obj.get('first_failure_rc', '')}",
            file=sys.stderr,
        )
        continue
    if "shape" not in obj:
        continue
    writer.writerow({
        "shape": obj.get("shape", ""),
        "official_tiling": tiling_text(obj.get("official_tiling")),
        "candidate_tiling": tiling_text(obj.get("candidate_tiling")),
        "official_core": obj.get("official_core", ""),
        "candidate_core": obj.get("candidate_core", ""),
        "official_ms": obj.get("official_latency_ms", ""),
        "candidate_ms": obj.get("candidate_latency_ms", ""),
        "delta_pct": obj.get("delta_pct", ""),
        "correctness": obj.get("correctness", ""),
    })
    sys.stdout.flush()
