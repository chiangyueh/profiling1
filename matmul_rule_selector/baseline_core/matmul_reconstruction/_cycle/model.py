from __future__ import annotations
from .types import CycleBreakdown, PreparedCandidate, UbStatus, ceil_div

K_BLOCK = 16
REDUCE_K_BLOCK = 8
FP16_BYTES = 2
FP32_BYTES = 4
DB_ON = 2


def dtype_bytes(dtype: str) -> int:
    if dtype == "fp32":
        return 4
    if dtype == "int8":
        return 1
    return 2


def bandwidth_nd2nz(burst_length: int, dtype_size: int) -> int:
    # cache_tiling_cycle_model.cc::GetBandwidthUsageNd2Nz
    remaining = burst_length * dtype_size * K_BLOCK
    packages = 0
    for shift in (9, 8, 7):
        packages += remaining >> shift
        remaining &= (1 << shift) - 1
    if remaining:
        packages += 1
    return packages * 8  # 512B / 64 B/cycle


def bandwidth_ub(burst_length: int) -> int:
    # Greedy 8/4/3/2/1 cache-line packages, each charged 8 cycles.
    result = 0
    for cache_line in (8, 4, 3, 2, 1):
        result += burst_length // cache_line
        burst_length %= cache_line
    return result * 8


def _mte1(n_burst: int, burst_length: int, bandwidth: int, cube_k: int, dsize: int) -> int:
    return n_burst * burst_length * K_BLOCK * cube_k * dsize // bandwidth


def _mte2_l0c2out_a(p: PreparedCandidate) -> int:
    run, r = p.run, p.result
    n_burst = (r.kal1_16 if run.trans_a else p.m_al1 * r.m_l0) * K_BLOCK
    burst_length = p.m_al1 * r.m_l0 if run.trans_a else r.kal1_16
    ori_inner = run.m if run.trans_a else run.k
    if not run.format_a_nd:
        n_burst = p.m_al1 * r.m_l0 if run.trans_a else r.kal1_16
        burst_length = r.kal1_16 * K_BLOCK if run.trans_a else p.m_al1 * r.m_l0 * K_BLOCK
    max_merge, min_merge = 8, 2
    if run.dtype_a == "fp32":
        max_merge >>= 1
        min_merge >>= 1
    if min_merge <= burst_length <= max_merge and ori_inner == burst_length:
        burst_length *= n_burst
        n_burst = 1
    return r.batch_l0 * n_burst * bandwidth_nd2nz(burst_length, dtype_bytes(run.dtype_a)) + 210


def _mte2_l0c2out_b(p: PreparedCandidate) -> int:
    run, r = p.run, p.result
    n_burst = (p.n_bl1 * r.n_l0 if run.trans_b else r.kbl1_16) * K_BLOCK
    burst_length = r.kbl1_16 if run.trans_b else p.n_bl1 * r.n_l0
    ori_inner = run.k if run.trans_b else run.n
    if not run.format_b_nd:
        n_burst = r.kbl1_16 if run.trans_b else p.n_bl1 * r.n_l0
        burst_length = p.n_bl1 * r.n_l0 * K_BLOCK if run.trans_b else r.kbl1_16 * K_BLOCK
    max_merge, min_merge = 8, 2
    # Source/binary: merge threshold uses dtype_b ...
    if run.dtype_b == "fp32":
        max_merge >>= 1
        min_merge >>= 1
    if min_merge <= burst_length <= max_merge and ori_inner == burst_length:
        burst_length *= n_burst
        n_burst = 1
    # ... but bandwidth lookup uses dtype_a, confirmed in source and installed binary.
    return r.batch_l0 * n_burst * bandwidth_nd2nz(burst_length, dtype_bytes(run.dtype_a)) + 210


def _fixed_pipe_l0c2out(p: PreparedCandidate, out_dtype_size: int) -> int:
    run, r, core = p.run, p.result, p.core
    l0c_to_fix = r.m_l0 * r.n_l0 * 256 * FP32_BYTES // 256
    fix_to_out = r.m_l0 * r.n_l0 * 256 * out_dtype_size // 128
    m_outer = ceil_div(core.m, r.m_l0)
    n_outer = ceil_div(core.n, r.n_l0)
    # Crucial MatMulV3 fact: InitRunParams sets is_batch_matmul_op=false, so this is 0.
    return r.batch_l0 * m_outer * n_outer * (l0c_to_fix + fix_to_out) if run.is_batch_matmul_op else 0


def _final_branch(p: PreparedCandidate, mte2_a: int, mte2_b: int, mad: int) -> tuple[int, str]:
    r, c = p.result, p.core
    m_outer = ceil_div(c.m, r.m_l0)
    n_outer = ceil_div(c.n, r.n_l0)
    if c.both_full_load:
        k_outer = ceil_div(c.k, r.k_l0)
        return mte2_a + mte2_b + m_outer * n_outer * k_outer * mad, "both_full_load"
    if c.al1_full_load:
        if c.bl1_k_full_load:
            n2 = ceil_div(c.n, r.n_l0 * p.n_bl1)
            k_outer = ceil_div(c.k, r.k_l0)
            return mte2_a + n2 * (mte2_b + m_outer * p.n_bl1 * k_outer * mad), "al1_full_bl1_k_full"
        kl1_times = ceil_div(c.k, r.kbl1_16)
        k_outer = ceil_div(r.kbl1_16, r.k_l0)
        inner = max(mte2_b, k_outer * mad) if r.db_bl1 == DB_ON else mte2_b + k_outer * mad
        return mte2_a + m_outer * n_outer * kl1_times * inner, "al1_full"
    if c.bl1_full_load:
        if c.al1_k_full_load:
            m2 = ceil_div(c.m, r.m_l0 * p.m_al1)
            k_outer = ceil_div(c.k, r.k_l0)
            return mte2_b + m2 * (mte2_a * n_outer + n_outer * p.m_al1 * k_outer * mad), "bl1_full_al1_k_full"
        kl1_times = ceil_div(c.k, r.kal1_16)
        k_outer = ceil_div(r.kal1_16, r.k_l0)
        inner = max(mte2_a, k_outer * mad) if r.db_al1 == DB_ON else mte2_a + k_outer * mad
        return mte2_b + m_outer * n_outer * kl1_times * inner, "bl1_full"
    if c.al1_k_full_load and c.bl1_k_full_load:
        m2 = ceil_div(c.m, r.m_l0 * p.m_al1)
        n2 = ceil_div(c.n, r.n_l0 * p.n_bl1)
        k_outer = ceil_div(c.k, r.k_l0)
        return m2 * (mte2_a + n2 * p.m_al1 * (mte2_b + p.n_bl1 * k_outer * mad)), "both_k_full"
    if c.al1_k_full_load:
        m2 = ceil_div(c.m, r.m_l0 * p.m_al1)
        kl1_times = ceil_div(c.k, r.kbl1_16)
        k_outer = ceil_div(r.kbl1_16, r.k_l0)
        return m2 * (mte2_a + n_outer * p.m_al1 * kl1_times * max(mte2_b, k_outer * mad)), "al1_k_full"
    if c.bl1_k_full_load:
        n2 = ceil_div(c.n, r.n_l0 * p.n_bl1)
        kl1_times = ceil_div(c.k, r.kal1_16)
        k_outer = ceil_div(r.kal1_16, r.k_l0)
        return n2 * (mte2_b + m_outer * p.n_bl1 * kl1_times * max(mte2_a, k_outer * mad)), "bl1_k_full"
    if r.kal1_16 == r.kbl1_16:
        k_outer_outer = ceil_div(c.k, r.kal1_16)
        k_outer = ceil_div(r.kal1_16, r.k_l0)
        inner = k_outer_outer * max(mte2_a + mte2_b, k_outer * mad)
    elif r.kal1_16 > r.kbl1_16:
        k_outer_outer = ceil_div(c.k, r.kal1_16)
        kl1_times = ceil_div(r.kal1_16, r.kbl1_16)
        k_outer = ceil_div(r.kbl1_16, r.k_l0)
        inner = k_outer_outer * (mte2_a + kl1_times * max(mte2_b, k_outer * mad))
    else:
        k_outer_outer = ceil_div(c.k, r.kbl1_16)
        kl1_times = ceil_div(r.kbl1_16, r.kal1_16)
        k_outer = ceil_div(r.kal1_16, r.k_l0)
        inner = k_outer_outer * (mte2_b + kl1_times * max(mte2_a, k_outer * mad))
    return m_outer * n_outer * inner, "not_full"


def score_l0c2out(p: PreparedCandidate) -> CycleBreakdown:
    run, r = p.run, p.result
    dsize = 4 if run.dtype_a == "fp32" else 2
    cube_k = 8 if run.dtype_a == "fp32" else 16
    mad_expansion = (1 if run.hf32_flag else 2) if run.dtype_a == "fp32" else 1
    out_size = 4 if run.dtype_out == "fp32" else 2
    mte2_a = _mte2_l0c2out_a(p)
    mte2_b = _mte2_l0c2out_b(p)
    mte1_a = _mte1(r.m_l0, r.k_l0, 256, cube_k, dsize) + 26
    mte1_b = _mte1(r.k_l0, r.n_l0, 128, cube_k, dsize) + 26
    mad_unit = max(r.m_l0 * r.k_l0 * r.n_l0 * mad_expansion, r.batch_l0 * max(mte1_a, mte1_b))
    base, branch = _final_branch(p, mte2_a, mte2_b, mad_unit)
    fixed = _fixed_pipe_l0c2out(p, out_size)
    multiplier = run.batch // r.batch_dim // r.batch_l0
    return CycleBreakdown(
        total_cycle=multiplier * (base + fixed), branch=branch,
        mte2_a=mte2_a, mte2_b=mte2_b, mte1_a=mte1_a, mte1_b=mte1_b,
        mad_unit=mad_unit, fixed_pipe=fixed, batch_multiplier=multiplier,
    )


def _ub_mte2_al1(p: PreparedCandidate) -> int:
    r, run = p.result, p.run
    n_burst = p.m_al1 * r.m_l0 if run.trans_a else r.kal1_16
    burst = r.kal1_16 * K_BLOCK if run.trans_a else p.m_al1 * r.m_l0 * K_BLOCK
    head = 100 if n_burst * burst < 512 else 0
    return n_burst * bandwidth_ub(burst) + head


def _ub_mte2_bl1(p: PreparedCandidate) -> int:
    r = p.result
    n_burst = ceil_div(r.kbl1_16, r.k_l0) * ceil_div(p.n_bl1, r.n_l0) * r.k_l0
    burst = r.n_l0 * K_BLOCK * K_BLOCK
    head = 100 if n_burst * burst < 512 else 0
    return n_burst * bandwidth_ub(burst) + head


def _ub_mte2_aub(p: PreparedCandidate, ub: UbStatus) -> int:
    r, run = p.result, p.run
    multi_k = ceil_div(r.kal1_16, ub.k_aub)
    multi_m = ceil_div(p.m_al1 * r.m_l0, ub.m_aub)
    n_burst = (ub.k_aub if run.trans_a else ub.m_aub) * K_BLOCK
    burst = ub.m_aub if run.trans_a else ub.k_aub
    if ((not run.trans_a and ub.k_aub == run.k) or (run.trans_a and ub.m_aub == run.m)):
        burst *= n_burst
        n_burst = 1
    head = 100 if n_burst * burst < 512 else 0
    return multi_k * multi_m * (n_burst * bandwidth_ub(burst) + head)


def _ub_mte2_bub(p: PreparedCandidate, ub: UbStatus) -> int:
    r, run = p.result, p.run
    multi_k = ceil_div(r.kbl1_16, ub.k_bub)
    multi_n = ceil_div(p.n_bl1 * r.n_l0, ub.n_bub)
    n_burst = (ub.n_bub if run.trans_b else ub.k_bub) * K_BLOCK
    burst = ub.k_bub if run.trans_b else ub.n_bub
    if ((not run.trans_b and ub.n_bub == run.n) or (run.trans_b and ub.k_bub == run.k)):
        burst *= n_burst
        n_burst = 1
    head = 100 if n_burst * burst < 512 else 0
    return multi_k * multi_n * (n_burst * bandwidth_ub(burst) + head)


def score_ub(p: PreparedCandidate, ub: UbStatus) -> CycleBreakdown:
    run, r, c = p.run, p.result, p.core
    dsize = 4 if run.dtype_a == "fp32" else 2
    cube_k = 8 if run.dtype_a == "fp32" else 16
    mad_expansion = (1 if run.hf32_flag else 2) if run.dtype_a == "fp32" else 1
    if run.is_compress_quant:
        mte2_a, mte2_b = _ub_mte2_al1(p), _ub_mte2_bl1(p)
    else:
        mte2_a, mte2_b = _ub_mte2_aub(p, ub), _ub_mte2_bub(p, ub)
    mte1_a = _mte1(r.m_l0, r.k_l0, 512, cube_k, dsize)
    mte1_b = _mte1(r.k_l0, r.n_l0, 256, cube_k, dsize)
    mad_unit = max(r.m_l0 * r.k_l0 * r.n_l0 * mad_expansion, r.batch_l0 * max(mte1_a, mte1_b))
    base, branch = _final_branch(p, mte2_a, mte2_b, mad_unit)
    n_ub_l0_time = ceil_div(r.n_l0, ub.n_cub)
    burst = r.m_l0 * ub.n_cub * K_BLOCK
    mte3 = (bandwidth_ub(burst) + (100 if burst < 512 else 0)) * n_ub_l0_time
    vector_bw = 80 if r.k_l0 == 1 and max(r.kal1_16, r.kbl1_16) >= 8 else 512
    output_vector = ((r.m_l0 * bandwidth_ub(r.n_l0) * 256 * FP16_BYTES) >> 1) // vector_bw
    preload = r.db_l0c == DB_ON and ((c.al1_full_load and not c.bl1_full_load) or (not c.al1_full_load and c.bl1_full_load))
    if preload:
        load_a = 1 if (r.kal1_16 == r.k_l0 or ((not run.trans_a) and r.k_l0 == 1)) else r.m_l0
        load_b = 1 if (p.n_bl1 == 1 or (run.trans_b and r.n_l0 == 1)) else r.k_l0
        cover = min(32 // (load_a + load_b + 1), ceil_div(c.k, r.k_l0)) * mad_unit
        l0c_to_ub = max(output_vector + mte3 - cover, 0)
    else:
        l0c_to_ub = output_vector + mte3
    base += ceil_div(c.m, r.m_l0) * ceil_div(c.n, r.n_l0) * l0c_to_ub
    multiplier = run.batch // r.batch_dim // r.batch_l0
    return CycleBreakdown(multiplier * base, branch, mte2_a, mte2_b, mte1_a, mte1_b, mad_unit, 0, multiplier)
