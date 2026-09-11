from __future__ import annotations
from .model import score_l0c2out
from .types import CycleUsed, PreparedCandidate


def _estimate_load_repeat(p: PreparedCandidate) -> tuple[int, int]:
    r, c = p.result, p.core
    if c.al1_full_load or c.bl1_full_load:
        if c.both_full_load:
            return 1, 1
        if c.bl1_full_load:
            return c.n // r.n_l0, 1
        if c.bl1_k_full_load:
            return 1, 1
        return 1, c.m // r.m_l0
    if c.al1_k_full_load or c.bl1_k_full_load:
        if c.al1_k_full_load and c.bl1_k_full_load:
            return 1, c.m // r.m_l0
        if c.al1_k_full_load:
            return 1, c.m // r.m_l0
        return c.n // r.n_l0, 1
    return c.n // r.n_l0, c.m // r.m_l0


def _load_size(p: PreparedCandidate) -> int:
    c = p.core
    al1_full_load_size = c.batch * c.k * c.m
    bl1_full_load_size = (c.batch if p.run.b_have_batch else 1) * c.k * c.n
    a_repeat, b_repeat = _estimate_load_repeat(p)
    return al1_full_load_size * a_repeat + bl1_full_load_size * b_repeat


def _load2d_a_repeat(p: PreparedCandidate) -> int:
    r = p.result
    return 1 if (r.kal1_16 == r.k_l0 or ((not p.run.trans_a) and r.k_l0 == 1)) else r.m_l0


def _load2d_b_repeat(p: PreparedCandidate) -> int:
    r = p.result
    return 1 if (p.n_bl1 == 1 or (p.run.trans_b and r.n_l0 == 1)) else r.k_l0


def _get_load2d_times(p: PreparedCandidate) -> int:
    r, c = p.result, p.core
    load_l0_repeat = _load2d_a_repeat(p) + _load2d_b_repeat(p)
    def common_times() -> int:
        return p.m_al1 * p.n_bl1 * (c.k // r.k_l0) * load_l0_repeat
    if c.is_any_full_load():
        if c.both_full_load:
            return load_l0_repeat
        if c.is_k_any_full_load():
            return common_times()
        return r.get_load_times(load_l0_repeat)
    if c.is_k_any_full_load():
        if c.is_k_both_full_load():
            return common_times()
        return r.get_load_times(load_l0_repeat)
    return r.get_load_times(load_l0_repeat)


def cycle_used(p: PreparedCandidate, support_l0c2out: bool = True) -> CycleUsed:
    if not support_l0c2out:
        raise ValueError("Cycle selector tie fields are implemented for the current L0C->out path; use model.score_ub for legacy cycle only.")
    b = score_l0c2out(p)
    r, c = p.result, p.core
    return CycleUsed(
        cycle=b.total_cycle,
        mad_cycle=r.m_l0 * r.k_l0 * r.n_l0,
        load_size=_load_size(p),
        l0c_used=r.m_l0 * r.n_l0 * r.db_l0c,
        load_2d_times=_get_load2d_times(p),
        repeat_load_size=r.n_dim * c.m + r.m_dim * c.n,
        k_l0=r.k_l0,
        result_m_dim=r.m_dim,
        result_n_dim=r.n_dim,
        source_index=p.source_index,
        payload=p.payload,
        breakdown=b,
    )


def vendor_would_update(cur: CycleUsed, best: CycleUsed, run, support_l0c2out: bool = True) -> bool:
    update_cycle = cur.cycle < best.cycle
    cycle_equal = best.cycle == cur.cycle
    if support_l0c2out and run.unaligned_flag and cycle_equal:
        m_inner_not_align = run.trans_a and (run.ori_shape_m % 16 != 0)
        m_dim_priority = cur.result_m_dim > cur.result_n_dim and cur.result_n_dim > 1
        n_inner_not_align = (not run.trans_b) and (run.ori_shape_n % 16 != 0)
        n_dim_priority = cur.result_n_dim > cur.result_m_dim and cur.result_m_dim > 1
        if (m_inner_not_align and m_dim_priority) or (n_inner_not_align and n_dim_priority):
            update_cycle = True
    update_mad = cycle_equal and cur.mad_cycle < best.mad_cycle
    mad_equal = cur.mad_cycle == best.mad_cycle
    if not support_l0c2out:
        update_mad = False
        mad_equal = True
    update_load = cycle_equal and mad_equal and cur.load_size < best.load_size
    update_l0c = cycle_equal and mad_equal and cur.load_size == best.load_size and cur.l0c_used > best.l0c_used
    update_load2d = (
        cycle_equal and mad_equal and cur.load_size == best.load_size and cur.l0c_used == best.l0c_used
        and cur.load_2d_times < 32
        and (best.load_2d_times >= 32 or (best.load_2d_times < 32 and best.k_l0 > cur.k_l0))
    )
    update_repeat = (
        cycle_equal and mad_equal and cur.load_size == best.load_size and cur.l0c_used == best.l0c_used
        and cur.repeat_load_size < best.repeat_load_size
        and support_l0c2out and (run.dtype_a == "fp32" or run.bias_flag)
    )
    return update_cycle or update_mad or update_load or update_l0c or update_load2d or update_repeat


def select_best(candidates: list[PreparedCandidate], support_l0c2out: bool = True) -> CycleUsed:
    if not candidates:
        raise ValueError("empty candidate list")
    best = cycle_used(candidates[0], support_l0c2out=support_l0c2out)
    for prepared in candidates[1:]:
        cur = cycle_used(prepared, support_l0c2out=support_l0c2out)
        if vendor_would_update(cur, best, prepared.run, support_l0c2out=support_l0c2out):
            best = cur
    return best


def rank_candidates(candidates: list[PreparedCandidate], support_l0c2out: bool = True) -> list[CycleUsed]:
    """Repeated one-pass selection. The first item is source-faithful best; later ranks are a deterministic extension."""
    remaining = list(candidates)
    output: list[CycleUsed] = []
    while remaining:
        # Re-index so source_index is only provenance; comparator always uses common run.
        scored = [cycle_used(p, support_l0c2out=support_l0c2out) for p in remaining]
        best_pos = 0
        for pos in range(1, len(scored)):
            if vendor_would_update(scored[pos], scored[best_pos], remaining[pos].run, support_l0c2out):
                best_pos = pos
        output.append(scored[best_pos])
        del remaining[best_pos]
    return output
