from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"divisor must be positive, got {b}")
    return (a + b - 1) // b


@dataclass(frozen=True)
class RunParams:
    trans_a: bool = False
    trans_b: bool = False
    format_a_nd: bool = True
    format_b_nd: bool = True
    format_out_nd: bool = True
    b_have_batch: bool = False
    is_batch_matmul_mode: bool = False
    is_batch_matmul_op: bool = False
    bias_flag: bool = False
    pattern_flag: bool = True
    unaligned_flag: bool = False
    hf32_flag: bool = False
    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_out: str = "fp16"
    m: int = 1
    k: int = 1
    n: int = 1
    batch: int = 1
    ori_shape_m: int = 1
    ori_shape_k: int = 1
    ori_shape_n: int = 1
    is_compress_quant: bool = False


@dataclass(frozen=True)
class ResultInfo:
    batch_dim: int = 1
    m_dim: int = 1
    n_dim: int = 1
    k_dim: int = 1
    m_l0: int = 1
    n_l0: int = 1
    k_l0: int = 1
    batch_l0: int = 1
    db_l0c: int = 1
    kal1_16: int = 1
    kbl1_16: int = 1
    m_l1: int = 1
    n_l1: int = 1
    db_al1: int = 1
    db_bl1: int = 1

    def validate(self) -> None:
        fields = (
            self.batch_dim, self.m_dim, self.n_dim, self.k_dim,
            self.m_l0, self.n_l0, self.k_l0, self.batch_l0,
            self.kal1_16, self.kbl1_16, self.m_l1, self.n_l1,
        )
        if min(fields) <= 0:
            raise ValueError(f"invalid non-positive ResultInfo field: {self}")

    def get_load_times(self, load_l0_repeat: int) -> int:
        return (min(self.kal1_16, self.kbl1_16) // self.k_l0) * load_l0_repeat


@dataclass(frozen=True)
class CoreStatus:
    batch: int = 1
    m: int = 1
    k: int = 1
    n: int = 1
    kal1_factor: int = 1
    kbl1_factor: int = 1
    m_single_core: int = 1
    n_single_core: int = 1
    both_full_load: bool = False
    al1_full_load: bool = False
    bl1_full_load: bool = False
    al1_k_full_load: bool = False
    bl1_k_full_load: bool = False

    def is_k_both_full_load(self) -> bool:
        return self.al1_k_full_load and self.bl1_k_full_load

    def is_k_any_full_load(self) -> bool:
        return self.al1_k_full_load or self.bl1_k_full_load

    def is_any_full_load(self) -> bool:
        return self.al1_full_load or self.bl1_full_load


@dataclass(frozen=True)
class UbStatus:
    k_aub: int = 1
    m_aub: int = 1
    k_bub: int = 1
    n_bub: int = 1
    n_cub: int = 1


@dataclass(frozen=True)
class PreparedCandidate:
    run: RunParams
    result: ResultInfo
    core: CoreStatus
    m_al1: int
    n_bl1: int
    source_index: int = 0
    payload: Any = None


@dataclass(frozen=True)
class CycleBreakdown:
    total_cycle: int
    branch: str
    mte2_a: int
    mte2_b: int
    mte1_a: int
    mte1_b: int
    mad_unit: int
    fixed_pipe: int
    batch_multiplier: int


@dataclass(frozen=True)
class CycleUsed:
    cycle: int
    mad_cycle: int
    load_size: int
    l0c_used: int
    load_2d_times: int
    repeat_load_size: int
    k_l0: int
    result_m_dim: int
    result_n_dim: int
    source_index: int = 0
    payload: Any = None
    breakdown: CycleBreakdown | None = None


@dataclass(frozen=True)
class PacketAdapterResult:
    candidate: PreparedCandidate
    exact_convert_fields: bool
    exact_current_callsite: bool
    notes: tuple[str, ...]
