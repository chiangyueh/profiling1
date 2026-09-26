#!/usr/bin/env python3

import csv
import json
import sys


TILING_FIELDS = (
    "key", "core", "single_m", "single_n", "single_k", "base_m", "base_n", "base_k",
    "step_m", "step_n", "step_ka", "step_kb", "depth_a1", "depth_b1", "l2_m_tile",
    "l2_n_tile", "l2_m_block", "l2_n_block", "l2_order",
)

FIELDS = [
    "record_type", "manifest_index", "shape", "input_dtype", "output_dtype",
    "candidate_branch", "candidate_variant", "candidate_selected", "tiling_changed", "status", "result_code",
]
FIELDS += [f"official_{name}" for name in TILING_FIELDS]
FIELDS += [f"candidate_{name}" for name in TILING_FIELDS]
FIELDS += [
    "structural_candidates", "structural_reference_base_n", "structural_reference_base_k",
    "structural_total_k_loops", "structural_l2_max_n_block", "analytic_tasks",
    "analytic_waves", "analytic_k_iterations", "analytic_n_tail_waste", "analytic_active_cores",
    "analytic_m_tasks", "analytic_n_tasks", "analytic_l2_dimensions", "analytic_l2_m_block",
    "analytic_l2_n_block", "analytic_l2_m_windows", "analytic_l2_n_windows",
    "analytic_l2_window_bytes", "analytic_l2_estimated_traffic", "old_partial_bytes",
    "new_partial_bytes", "partial_delta_bytes", "partial_saved_bytes", "partial_saved_pct",
    "k_units", "k_quotient", "k_remainder", "balanced_k_min", "balanced_k_max",
    "official_forward_ms", "candidate_forward_ms", "candidate_reverse_ms", "official_reverse_ms",
    "official_workspace", "candidate_workspace", "official_latency_ms", "candidate_latency_ms",
    "delta_pct", "max_abs_diff", "max_rel_diff", "correctness", "inputs", "skipped_non_v3",
    "non_target_route", "official_target", "candidate_selected_count", "official_preserved",
    "target_passes", "quota_met", "passed", "failed", "wide_n_shallow_k_passed",
    "clear_candidate_wins", "false_positive_intercepts", "overlap", "stable_candidate_wins",
    "stable_official_wins", "mixed_order", "official_failed", "manifest_start_index",
    "manifest_end_index",
]


def flatten_tiling(row, prefix, tiling):
    if not isinstance(tiling, dict):
        return
    for name in TILING_FIELDS:
        row[f"{prefix}_{name}"] = tiling.get(name, "")


def convert(obj):
    if obj.get("progress") is True:
        row = {"record_type": "progress"}
    elif obj.get("summary") is True:
        row = {"record_type": "summary", "status": "COMPLETE"}
    elif "shape" in obj:
        row = {"record_type": "measurement" if "official_tiling" in obj else "failure"}
    else:
        return None

    for name in FIELDS:
        if name in obj:
            row[name] = obj[name]
    if row["record_type"] == "summary":
        row["candidate_branch"] = obj.get("campaign", "")
        row["candidate_selected_count"] = obj.get("candidate_selected", "")
    if row["record_type"] == "measurement" and not row.get("status"):
        row["status"] = obj.get("correctness", "")
    flatten_tiling(row, "official", obj.get("official_tiling"))
    flatten_tiling(row, "candidate", obj.get("candidate_tiling"))
    if "official_core" in obj:
        row["official_core"] = obj["official_core"]
    if "candidate_core" in obj:
        row["candidate_core"] = obj["candidate_core"]
    return row


def main():
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    sys.stdout.flush()
    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            print(text, file=sys.stderr)
            continue
        row = convert(obj)
        if row is None:
            continue
        writer.writerow(row)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
