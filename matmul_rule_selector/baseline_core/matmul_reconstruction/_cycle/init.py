from __future__ import annotations
from dataclasses import replace
from .types import CoreStatus, PreparedCandidate, ResultInfo, RunParams, ceil_div


def matmulv3_init_run_params(
    m: int,
    n: int,
    k: int,
    dtype: str,
    trans_a: bool = False,
    trans_b: bool = False,
    has_bias: bool = False,
    pattern_flag: bool = True,
    dtype_b: str | None = None,
    dtype_out: str | None = None,
) -> RunParams:
    """Source-faithful MatMulV3 InitRunParams for the fields used by the cycle selector."""
    dtype_b = dtype if dtype_b is None else dtype_b
    dtype_out = dtype if dtype_out is None else dtype_out
    reduce = 8 if dtype == "fp32" else 16
    aligned = (m % 16 == 0) and (k % reduce == 0) and (n % 16 == 0)
    return RunParams(
        trans_a=bool(trans_a), trans_b=bool(trans_b),
        format_a_nd=True, format_b_nd=True, format_out_nd=True,
        b_have_batch=False, is_batch_matmul_mode=False, is_batch_matmul_op=False,
        bias_flag=bool(has_bias), pattern_flag=bool(pattern_flag),
        unaligned_flag=not aligned, hf32_flag=False,
        dtype_a=dtype, dtype_b=dtype_b, dtype_out=dtype_out,
        m=ceil_div(m, 16), k=ceil_div(k, reduce), n=ceil_div(n, 16), batch=1,
        ori_shape_m=m, ori_shape_k=k, ori_shape_n=n,
    )


def update_load_flags(core: CoreStatus) -> CoreStatus:
    al1_full = core.m_single_core == 1 and core.kal1_factor == 1
    al1_k = (not al1_full) and core.kal1_factor == 1
    bl1_full = core.n_single_core == 1 and core.kbl1_factor == 1
    bl1_k = (not bl1_full) and core.kbl1_factor == 1
    return replace(
        core,
        both_full_load=al1_full and bl1_full,
        al1_full_load=al1_full,
        bl1_full_load=bl1_full,
        al1_k_full_load=al1_k,
        bl1_k_full_load=bl1_k,
    )


def set_buffer_params(run: RunParams, candidate: ResultInfo, support_l0c2out: bool = True,
                      source_index: int = 0, payload=None) -> PreparedCandidate:
    """Exact GemmEstimate::SetBufferParams(ResultInfo) state transformation."""
    candidate.validate()
    m_single = ceil_div(ceil_div(run.m, candidate.m_dim), candidate.m_l1)
    n_single = ceil_div(ceil_div(run.n, candidate.n_dim), candidate.n_l1)
    kal1_factor = ceil_div(run.k, candidate.kal1_16)
    kbl1_factor = ceil_div(run.k, candidate.kbl1_16)
    core = CoreStatus(
        batch=ceil_div(run.batch, candidate.batch_dim),
        m=min(m_single * candidate.m_l1, run.m),
        k=run.k,
        n=min(n_single * candidate.n_l1, run.n),
        kal1_factor=kal1_factor,
        kbl1_factor=kbl1_factor,
        m_single_core=m_single,
        n_single_core=n_single,
    )
    if support_l0c2out and candidate.k_dim != 1:
        k_64 = ceil_div(run.k, candidate.k_dim)
        k_l1 = max(candidate.kal1_16, candidate.kbl1_16)
        k_single = ceil_div(k_64, k_l1)
        core = replace(
            core,
            kal1_factor=k_single * (k_l1 // candidate.kal1_16),
            kbl1_factor=k_single * (k_l1 // candidate.kbl1_16),
            k=min(k_single * k_l1, run.k),
        )
    result = replace(
        candidate,
        batch_dim=ceil_div(run.batch, core.batch),
        m_dim=ceil_div(run.m, core.m),
        n_dim=ceil_div(run.n, core.n),
        k_dim=ceil_div(run.k, core.k),
    )
    core = update_load_flags(core)
    return PreparedCandidate(
        run=run, result=result, core=core,
        m_al1=ceil_div(result.m_l1, result.m_l0),
        n_bl1=ceil_div(result.n_l1, result.n_l0),
        source_index=source_index, payload=payload,
    )


def prepare_external_status(run: RunParams, result: ResultInfo, core: CoreStatus,
                            m_al1: int, n_bl1: int, source_index: int = 0,
                            payload=None, emulate_constructor_bug: bool = True) -> PreparedCandidate:
    """Prepare the CoreStatus/SingleCoreStatus constructor route.

    Installed/source constructor bug: n_single_core copies core_status.m_single_core.
    The cycle equations themselves mostly use explicit full-load flags, but we preserve it.
    """
    if emulate_constructor_bug:
        core = replace(core, n_single_core=core.m_single_core)
    return PreparedCandidate(run, result, core, m_al1, n_bl1, source_index, payload)
