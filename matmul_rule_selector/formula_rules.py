"""910B MatMulV3 主要決策欄位的整數參考實現。

保留原十二條規則；加入明確輸入／編譯適用域與無號整數邊界。
輸出不是完整 TCubeTiling：外部 SDK 未提供的 initializer 語義以未解決欄位列出。
不提供可提交裝置的序列化資料，也不呼叫官方 tiling 生成器。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import isqrt
import json
from typing import Dict, List, Optional, Tuple


def cd(x: int, y: int) -> int:
    if x < 0 or y <= 0:
        raise ValueError('ceildiv requires x>=0 and y>0')
    return (x + y - 1) // y


def up(x: int, q: int = 16) -> int:
    return cd(x, q) * q


def down(x: int, q: int = 16) -> int:
    return x // q * q


def ceil_sqrt_ratio(num: int, den: int) -> int:
    r = isqrt(cd(num, den))
    return r if r * r * den >= num else r + 1


def divisor_up(n: int, lower: int) -> Optional[int]:
    """固定 core 數的因數向上投影；不產生或比較 tiling。"""
    for d in range(max(1, lower), n + 1):
        if n % d == 0:
            return d
    return None


def divisor_down(n: int, upper: int) -> int:
    """resident K 次數的整除投影；n 受完整 resident 的 L1 容量限制。"""
    upper = min(n, upper)
    if upper < 1:
        raise ValueError('no positive resident step')
    best = 1
    for d in range(1, isqrt(n) + 1):
        if n % d == 0:
            if d <= upper:
                best = max(best, d)
            other = n // d
            if other <= upper:
                best = max(best, other)
    return best


@dataclass(frozen=True)
class Hardware:
    cores: int = 20
    l0a: int = 65536
    l0b: int = 65536
    l0c: int = 131072
    l1_usable: int = 524032
    l2: int = 192 * 1024 * 1024
    ub: int = 196352
    bt: int = 1024


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    dtype: str = 'fp16'
    trans_a: bool = False
    trans_b: bool = False
    bias: bool = False
    bias_dtype: Optional[str] = None
    bias_length: Optional[int] = None
    format_a: str = 'ND'
    format_b: str = 'ND'
    format_out: str = 'ND'
    dtype_b: Optional[str] = None
    dtype_out: Optional[str] = None
    is_hf32: bool = False
    force_group_acc: bool = False

    @property
    def d(self) -> int:
        return {'fp16': 2, 'bf16': 2, 'fp32': 4}[self.dtype]


    @property
    def b_dtype(self) -> str:
        return self.dtype if self.dtype_b is None else self.dtype_b

    @property
    def c_dtype(self) -> str:
        return self.dtype if self.dtype_out is None else self.dtype_out

    @property
    def dc(self) -> int:
        return DTYPE_BYTES[self.c_dtype]


@dataclass(frozen=True)
class CompileInfo:
    """必須來自目標的固定編譯配置；不能由 core 數推定能力或 bias 巨集。"""
    aicore_arch: int
    support_l0c2out: bool
    support_l12_bt_bf16: bool
    orig_dtype_x1: str
    orig_dtype_x2: str
    orig_dtype_y: str
    orig_dtype_bias: str
    format_x1: str
    format_x2: str
    format_y: str
    total_ub_size: int


DTYPE_BYTES = {'fp16': 2, 'bf16': 2, 'fp32': 4}
INT32_MAX = (1 << 31) - 1
UINT64_MAX = (1 << 64) - 1
# H:OpSpecificCheck；無 bias 時比較前三欄，不自行擴充組合。
SUPPORTED_TYPE_ROWS = (
    ('fp16', 'fp16', 'fp16', 'fp16'),
    ('fp16', 'fp16', 'fp16', 'fp32'),
    ('fp32', 'fp32', 'fp32', 'fp32'),
    ('bf16', 'bf16', 'bf16', 'fp32'),
    ('fp16', 'fp16', 'fp32', 'fp16'),
    ('bf16', 'bf16', 'fp32', 'bf16'),
)


class TilingRuleError(ValueError):
    """輸入超出已表示的適用域，或違反可見原碼／算術約束。"""


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise TilingRuleError(code + ': ' + detail)


def checked_u64_product(*values: int) -> int:
    product = 1
    for value in values:
        require(type(value) is int and 0 <= value <= UINT64_MAX,
                'UINT64_INPUT', 'multiplicand must be a nonnegative uint64')
        require(value == 0 or product <= UINT64_MAX // value,
                'UINT64_PRODUCT_OVERFLOW', 'multiply before using a byte count')
        product *= value
    return product


def split_k_l2_long_capacity(l2: int, used: int, k_span: int, live_panel: int,
                             panel_dtype_bytes: int, opposite_dtype_bytes: int,
                             alignment: int) -> dict:
    """精確的局部 L2 配額不等式；MK/NK 共用，無 latency 或候選排名。

    k_span*live_panel*d_panel + x*(k_span*d_opposite + 2*live_panel*4) <= l2/used
    不可容納一個對齊長邊時 capacity=0；這不是 kernel 不合法，也不宣稱 L2 駐留。
    """
    for value in (l2, used, k_span, live_panel, panel_dtype_bytes,
                  opposite_dtype_bytes, alignment):
        require(type(value) is int and 0 < value <= UINT64_MAX,
                'L2_ARGUMENT', 'all arguments must be positive uint64 values')
    quota = l2 // used
    result = dict(quota=quota, capacity=0, applied=False, fixed_bytes=None,
                  denominator=None, reason='NO_POSITIVE_REMAINDER')
    # 先用除法比較，連 fixed operand 的乘法本身也不允許溢位。
    if quota == 0 or k_span > (quota - 1) // live_panel // panel_dtype_bytes:
        return result
    fixed = checked_u64_product(k_span, live_panel, panel_dtype_bytes)
    # 上述嚴格比較保證 fixed < quota，這裡才能做 unsigned 減法。
    remainder = quota - fixed
    require(k_span <= UINT64_MAX // opposite_dtype_bytes,
            'L2_DENOMINATOR_OVERFLOW', 'k_span*d_opposite exceeds uint64')
    streamed = k_span * opposite_dtype_bytes
    require(live_panel <= (UINT64_MAX - streamed) // 8,
            'L2_DENOMINATOR_OVERFLOW', 'streamed + two FP32 panels exceeds uint64')
    denominator = streamed + 8 * live_panel
    capacity = down(remainder // denominator, alignment)
    result.update(fixed_bytes=fixed, denominator=denominator, capacity=capacity,
                  applied=capacity >= alignment,
                  reason='APPLIED' if capacity >= alignment else 'NO_ALIGNED_LONG_TILE')
    return result


def validate_input_domain(s: Shape, h: Hardware, ci: Optional[CompileInfo]) -> None:
    require(isinstance(ci, CompileInfo), 'COMPILE_INFO_REQUIRED',
            'supply architecture, capability flags and actual dtype/format macros')
    for name in ('m', 'n', 'k'):
        value = getattr(s, name)
        require(type(value) is int and 1 <= value <= INT32_MAX,
                'INPUT_DIMENSION', name + ' must be in [1, INT32_MAX]')
    require(h.cores in (20, 24), 'CORE_DOMAIN', 'reference rules cover 20 or 24 AICs')
    for name in ('l0a', 'l0b', 'l0c', 'l1_usable', 'l2', 'ub', 'bt'):
        value = getattr(h, name)
        require(type(value) is int and 0 < value <= UINT64_MAX,
                'HARDWARE_CAPACITY', name + ' must be a positive uint64')
    require(ci.aicore_arch == 220, 'ARCH_DOMAIN', 'the supported reference is arch22/220')
    require(ci.support_l0c2out is True and ci.support_l12_bt_bf16 is False,
            'HARDWARE_CAPABILITY_DOMAIN', 'requires L0C-to-out and the non-arch35 capability profile')
    require(type(ci.total_ub_size) is int and 0 < ci.total_ub_size <= h.ub,
            'UB_COMPILE_CAPACITY', 'the compiled TOTAL_UB_SIZE must fit actual UB')
    require((ci.orig_dtype_x1, ci.orig_dtype_x2, ci.orig_dtype_y) ==
            (s.dtype, s.b_dtype, s.c_dtype), 'COMPILE_DTYPE_MISMATCH',
            'ORIG_DTYPE_X1/X2/Y must match the selected tensor types')
    require(ci.orig_dtype_bias in DTYPE_BYTES, 'COMPILE_BIAS_DTYPE_REQUIRED',
            'ORIG_DTYPE_BIAS is required even with no bias tensor')
    require((ci.format_x1, ci.format_x2, ci.format_y) ==
            (s.format_a, s.format_b, s.format_out), 'COMPILE_FORMAT_MISMATCH',
            'FORMAT_X1/X2/Y must match tensor storage formats')
    dtype_tuple = (s.dtype, s.b_dtype, s.c_dtype)
    if s.bias:
        require(s.bias_dtype is not None and s.bias_length == s.n,
                'BIAS_DESCRIPTOR', 'explicit bias dtype and last dimension N are required')
        require(ci.orig_dtype_bias == s.bias_dtype, 'COMPILE_BIAS_MISMATCH',
                'the real bias tensor must match the compiled bias type')
        require(dtype_tuple + (s.bias_dtype,) in SUPPORTED_TYPE_ROWS,
                'OP_SPECIFIC_DTYPE', 'not in H:OpSpecificCheck supported quadruples')
        require(h.bt >= 1024 and h.l1_usable > 1024,
                'BIAS_CAPACITY', 'this reference retains the existing 1024-byte bias reservation')
    else:
        require(s.bias_dtype is None and s.bias_length is None,
                'UNEXPECTED_BIAS_DESCRIPTOR', 'bias=False must not describe a bias tensor')
        require(any(dtype_tuple == row[:3] for row in SUPPORTED_TYPE_ROWS),
                'OP_SPECIFIC_DTYPE', 'not in H:OpSpecificCheck supported triples')
    # 此次修正不將沒有 storage-shape / SDK 合約的路徑假裝納入完整 legality。
    require(s.format_a == s.format_b == s.format_out == 'ND',
            'REFERENCE_LAYOUT_NOT_MODELED', 'NZ/mixed layouts and NZ output are explicitly unsupported here')
    require(type(s.is_hf32) is bool and type(s.force_group_acc) is bool,
            'PRECISION_FLAG_TYPE', 'precision flags must be bool')
    require(not (s.is_hf32 and s.force_group_acc), 'PRECISION_MODE_CONFLICT',
            'MatMulV3 op_impl_mode cannot be both 0x40 and 0x4')
    require(not s.is_hf32, 'REFERENCE_HF32_NOT_MODELED',
            'HF32 is recognized but not silently evaluated with the default-precision rules')
    active_group = s.force_group_acc and s.dtype == 'fp32' and s.k >= 2048
    require(not active_group, 'REFERENCE_GROUP_ACC_NOT_MODELED',
            'H:SupportForceGrpAccForFp32 requires a separate accuracy-preserving route')


def validate_generated_rules(s: Shape, h: Hardware, ci: CompileInfo, state: dict) -> dict:
    f, usage, family = state['fields'], state['resource_bytes'], state['family']
    for key, value in f.items():
        require(type(value) is int and 0 <= value <= INT32_MAX,
                'FIELD_RANGE', key + ' exceeds the conservative signed-32-bit domain')
        require(key == 'iterateOrder' or value > 0, 'ZERO_FIELD', key)
    require(f['iterateOrder'] in (0, 1), 'ITERATE_ORDER', 'expected existing API order 0/1')
    require(1 <= f['usedCoreNum'] <= h.cores, 'CORE_COUNT', 'out-of-range usedCoreNum')
    require(all(f[k] % 16 == 0 for k in ('baseM', 'baseN', 'baseK')),
            'BASE_ALIGNMENT', 'base dimensions must satisfy the retained 16-element alignment')
    require(f['dbL0C'] in (1, 2), 'L0C_DEPTH', 'existing depth is 1 or 2')
    require(usage['L0A_double'] <= h.l0a and usage['L0B_double'] <= h.l0b and
            usage['L0C'] <= h.l0c and usage['L1_AB'] + usage['bias_reserved'] <= h.l1_usable,
            'FINAL_RESOURCE_CAPACITY', 'final values exceed the explicit per-level footprint')
    if s.bias:
        require(f['baseN'] <= 256 and 4 * f['baseN'] <= h.bt,
                'BIAS_BASE_N', 'retained host cap and BT capacity')
    for operand in ('A', 'B'):
        step = f['stepM'] if operand == 'A' else f['stepN']
        kstep = f['stepKa'] if operand == 'A' else f['stepKb']
        depth = f['depth' + operand + '1']
        require(depth >= step * kstep and depth % (step * kstep) == 0,
                'L1_BUFFER_MULTIPLICITY', operand + ' depth does not cover an integral step')
    require(max(f['stepKa'], f['stepKb']) % min(f['stepKa'], f['stepKb']) == 0,
            'K_STEP_DIVISIBILITY', 'retained full-load step relationship')
    l2 = state['l2']
    require(l2['calOrder'] in (0, 1, 2), 'L2_ORDER', 'only existing calOrder values are used')
    for key in ('mTile', 'nTile', 'mTileBlock', 'nTileBlock'):
        require(type(l2[key]) is int and 1 <= l2[key] <= INT32_MAX, 'L2_FIELD_RANGE', key)
    if family == 'BASE':
        require(f['singleCoreM'] == f['baseM'] and f['singleCoreN'] == f['baseN'] and
                f['singleCoreK'] == s.k, 'BASE_SINGLE_ITERATE', 'BASE owns exactly one MN base tile')
    elif family == 'AL1':
        require(s.dtype == 'fp32' and not s.trans_a and s.trans_b and s.m <= 16 and
                f['singleCoreM'] == s.m and f['singleCoreN'] == f['baseN'],
                'AL1_BRANCH_DOMAIN', 'float-only NT whole-M resident branch')
        require(f['singleCoreK'] == s.k and f['baseK'] * f['stepKa'] >= s.k,
                'AL1_WHOLE_K', 'resident A must cover the complete K')
    elif family == 'BL1':
        require(ci.orig_dtype_bias == ci.orig_dtype_x1, 'BL1_COMPILE_GUARD',
                'mat_mul_v3.cpp requires ORIG_DTYPE_BIAS == ORIG_DTYPE_X1')
        require(f['singleCoreK'] == s.k and f['stepN'] * f['baseN'] == f['singleCoreN'] and
                f['stepKb'] * f['baseK'] >= s.k, 'BL1_RESIDENT_DOMAIN', 'whole-K parent-N B resident')
    elif family == 'SINGLE_CORE_SPLIT_K':
        require((f['baseM'], f['baseN'], f['baseK']) == (128, 128, 256 // s.d),
                'SC_RETAINED_BASE', 'keep the existing base profile')
        require(f['singleCoreK'] == f['stepKa'] * f['baseK'],
                'SC_K_SPAN', 'the owner-local K span must match its step')
        if state['loop_order'] == 'NKM':
            require(ci.orig_dtype_bias == ci.orig_dtype_x1 and not state['conversion_a'] and
                    not state['conversion_b'], 'SC_NKM_COMPILE_GUARD',
                    'NKM requires the matching bias macro and NO_ND2NZ branch')
    elif family == 'DETERMINISTIC_SPLIT_K':
        require((f['baseM'], f['baseN'], f['baseK']) == (128, 128, 256 // s.d),
                'DET_STATIC_BASE', 'production GetMMConfig shapeParams fix these values')
        require(f['stepKa'] == f['stepKb'] == 3, 'DET_FIXED_PROFILE', 'retain the existing 33 profile')
        if state['NBuffer33_K_span_unchanged']:
            require((f['stepM'], f['stepN'], f['depthA1'], f['depthB1'], f['singleCoreK']) ==
                    (3, 1, 9, 6, 3 * f['baseK']), 'DET_NBUFFER33_VISIBLE_PROFILE',
                    'do not infer a new K span for the absent NBuffer33 policy')
    else:
        raise TilingRuleError('UNKNOWN_FAMILY: ' + family)
    return dict(visible_input_and_branch_checks='PASS',
                sdk_internal_legality='UNVERIFIED',
                actual_kernel_build='NOT_VERIFIED',
                complete_tiling=False,
                precision_domain='DEFAULT; inactive force-group flag follows the original K>=2048 condition',
                unmodeled_domains=['NZ input or output', 'head ND2NZ', 'HF32', 'active force-group accumulation',
                                  'GM-to-L1 Split-K', 'single-core Split-K+AL1',
                                  'non-deterministic multi-core Split-K', 'arch35/310P'])


def known_tcube_initializer(s: Shape, state: dict) -> dict:
    """只重建有來源的數值；不能把這個子集當作 SDK TCubeTiling 的完整宣告。"""
    values = dict(state['fields'])
    values.update(M=s.m, N=s.n, Ka=s.k, Kb=s.k, isBias=int(s.bias),
                  dbL0A=2, dbL0B=2, batchM=1, batchN=1, singleBatchM=1, singleBatchN=1)
    missing = {
        'transLength': 'the generic SDK scratch interpretation is not supplied for these dense policies',
        'shareMode': 'the dense-policy shared-buffer mode contract is not supplied',
        'shareL1Size': 'must separate caller-resident storage from API-owned storage for each family',
        'shareL0CSize': 'must establish single-buffer versus dbL0C-multiplied allocation semantics',
        'shareUbSize': 'the source snapshot does not supply the API UB sharing/scratch contract',
    }
    return dict(known_values=values, unresolved_known_fields=missing, complete=False,
                declaration_available=False,
                additional_fields='unknown until the target SDK TCubeTiling declaration is supplied',
                initialization_source='direct rules and cited non-query assignments; never mm_.GetTiling',
                zero_fill_of_unknown_fields=False)


def existing_conversion(s: Shape, h: Hardware) -> Tuple[bool, bool]:
    """H:GetMoreArgs/NeedNd2NzVnchw，限本檔宣告之預設精度域；未改其 heuristic。"""
    allowed_bytes = {32, 64, 96, 128, 160, 192, 224, 256, 384}
    def operand(inner: int, outer: int, fmt: str) -> bool:
        if fmt != 'ND':
            return False
        direct = inner * s.d in allowed_bytes
        align = inner * s.d % 256 == 0
        normal = (not align or inner > 65535) and not direct and not (s.dtype == 'fp32' and inner < 65535)
        vnchw = (outer > 8192 and inner > 1 and
                 (inner * s.d <= 192 or (inner * s.d <= 384 and inner % 2 == 0)
                  or (inner * s.d <= 512 and inner % 4 == 0)) and
                 not align and not direct and inner != 32 // s.d)
        return normal or vnchw
    a = operand(s.m if s.trans_a else s.k, s.k if s.trans_a else s.m, s.format_a)
    b = operand(s.k if s.trans_b else s.n, s.n if s.trans_b else s.k, s.format_b)
    if s.dtype != 'fp32' and s.format_b == 'ND':
        b |= (not s.trans_b and s.n % 16384 == 0 and
              (s.m > 4096 or (s.m == 4096 and h.cores >= 24)) and s.k >= 6656)
        b |= (not s.trans_a and s.trans_b and s.k == 16384 and
              4096 <= s.m <= 4480 and s.n >= 7168 and h.cores >= 24)
    return a, b


def geometry(s: Shape, h: Hardware, ca: bool, cb: bool) -> Tuple[int, int, int, int]:
    area = h.l0c // 4
    qm = 256 // s.d if s.trans_a and s.format_a == 'ND' and not ca else 16
    qn = 256 // s.d if not s.trans_b and s.format_b == 'ND' and not cb else 16
    if s.trans_a and s.trans_b:
        bm = min(256, up(s.m), up(isqrt(area), qm))
        bn = down(min(256, up(s.n), area // bm))
    else:
        bn = min(256, up(s.n), up(isqrt(area), qn))
        bm = down(min(256, up(s.m), area // bn))
    if bm == up(s.m):
        # Whole-M 窄矩形：在既有小M路徑出現過的N跨度內補足面積，而非固定N=256。
        nmax = min(256 if s.bias else 1024, up(s.n), area // bm, h.l0b // (2 * 16 * s.d))
        bn = max(16, down(nmax, qn if nmax >= qn else 16))
    if min(bm, bn) < 16:
        raise ValueError('hardware outside the stated 910B capacity domain')
    return bm, bn, qm, qn


def base_k(s: Shape, h: Hardware, bm: int, bn: int) -> int:
    return min(up(s.k), down(min(h.l0a // (2 * bm * s.d), h.l0b // (2 * bn * s.d))))


def tile_l2(s: Shape, h: Hardware, bm: int, bn: int, gm: int, gn: int) -> dict:
    kp = up(s.k)
    def f(x: int, y: int) -> int:
        return x * bm * kp * s.d + y * bn * kp * s.d + x * bm * y * bn * s.dc
    if f(gm, gn) <= h.l2:
        x, y = gm, gn
        root = None
    else:
        aa = bm * bn * s.dc
        bb = kp * s.d * (bm + bn)
        root = (isqrt(bb * bb + 4 * aa * h.l2) - bb) // (2 * aa)
        if root < 1:
            # 連一個基本 tile 的完整 K 工作集也放不進 L2；不可假造 K 方向 L2 欄位。
            x = y = 1
        else:
            x = min(gm, root)
            ycap = (h.l2 - x * bm * kp * s.d) // (bn * kp * s.d + x * bm * bn * s.dc)
            y = min(gn, ycap)
    # 以 block 數重平衡，不把元素長度寫成 macro count。
    mc, nc = cd(gm, x), cd(gn, y)
    x, y = cd(gm, mc), cd(gn, nc)
    if x == 1:
        order = 1
    elif y == 1:
        order = 2
    elif x % h.cores == 0 and y % h.cores != 0:
        order = 1
    elif y % h.cores == 0 and x % h.cores != 0:
        order = 2
    else:
        order = 0
    return dict(mTile=mc, nTile=nc, mTileBlock=x, nTileBlock=y, calOrder=order,
                mMacroTail=gm - (mc - 1) * x, nMacroTail=gn - (nc - 1) * y,
                rectangular_bytes=f(x, y), integer_root=root,
                full_k_rect_fits=f(x, y) <= h.l2)


def solve(s: Shape, h: Hardware = Hardware(), ci: Optional[CompileInfo] = None) -> dict:
    validate_input_domain(s, h, ci)
    bias_macro_matches = ci.orig_dtype_bias == ci.orig_dtype_x1
    d = s.d
    l1 = h.l1_usable - (1024 if s.bias else 0)
    ca, cb = existing_conversion(s, h)
    require(not ca and not cb, 'REFERENCE_ND2NZ_NOT_MODELED',
            'head conversion needs baseAN/AD/BN/BD and its UB contract; no executable tiling is fabricated')
    bm, bn, qm, qn = geometry(s, h, ca, cb)
    bk = base_k(s, h, bm, bn)
    if bk < 16:
        raise ValueError('no legal baseK for the stated geometry')
    gm0, gn0 = cd(s.m, bm), cd(s.n, bn)
    a, b = bm * bk * d, bn * bk * d
    streaming_cap = l1 // (2 * (a + b))
    state = dict(result_kind='DECISION_RULES_NOT_EXECUTABLE_TILING', input=asdict(s), hardware=asdict(h),
                 compile_info=asdict(ci), conversion_a=ca, conversion_b=cb,
                 geometry_base=[bm, bn, bk], geometry_grid=[gm0, gn0],
                 alignment_preference=[qm, qn], L1_after_bias=l1,
                 primitive_mn_count=cd(s.m, 16) * cd(s.n, 16),
                 geometry_streaming_k_capacity=streaming_cap * bk)
    family = 'BASE'
    sm = sn = 1
    ta = tb = 1
    single_m, single_n, single_k = bm, bn, s.k
    order = 0
    db = 1
    l2 = None
    # 每一判斷只解一個容量或合法性問題；尚未形成其他完整 tiling。
    native_nd = s.format_a == 'ND' and s.format_b == 'ND' and not ca and not cb
    sc_root = ceil_sqrt_ratio(h.cores * s.n, s.m)
    sc_pn = divisor_up(h.cores, min(h.cores, sc_root))
    sc_pm = h.cores // sc_pn
    sc_qm = 256 // d if s.trans_a else max(16, 32 // d)
    sc_qn = max(16, 256 // d)
    sc_parent_m = up(cd(s.m, sc_pm), sc_qm)
    sc_parent_n = up(cd(s.n, sc_pn), sc_qn)
    sc_grid_full = cd(s.m, sc_parent_m) * cd(s.n, sc_parent_n) == h.cores
    state['SC_aligned_grid_full'] = sc_grid_full
    al1_legal = native_nd and s.dtype == 'fp32' and not s.trans_a and s.trans_b and s.m <= 16 and s.n > 16
    ar = up(s.m) * up(s.k) * d
    if al1_legal and ar + 2 * 16 * 16 * d <= l1:
        family = 'AL1'
        bm = 16
        # full K 以16元素對齊；baseK取此長度的整除因子，避免resident額外padding。
        bn_cap = down((l1 - ar) // (2 * 16 * d))
        bn = min(256, up(s.n), max(16, down(s.n // h.cores)), bn_cap)
        kcap = down(min(up(s.k), h.l0a // (2 * bm * d), h.l0b // (2 * bn * d),
                        (l1 - ar) // (2 * bn * d)))
        bk = 16 * divisor_down(cd(s.k, 16), kcap // 16)
        tau = cd(s.k, bk)
        resident = bm * tau * bk * d
        cap = (l1 - resident) // (2 * bn * bk * d)
        tb = divisor_down(tau, cap)
        ta = tau
        da, db1 = ta, 2 * tb
        single_m, single_n = s.m, bn
        db = 2 if 8 * bm * bn <= h.l0c else 1
        l2 = dict(mTile=1, nTile=1, mTileBlock=1, nTileBlock=cd(s.n, bn), calOrder=1)
        state['resident_bytes'] = resident
        state['resident_opposite_step_cap'] = cap
    else:
        kp = up(s.k)
        ncap = down((l1 - 2 * 16 * 16 * d) // (kp * d))
        pn_min = cd(s.n, ncap) if ncap > 0 else h.cores + 1
        pn = divisor_up(h.cores, pn_min)
        bl1_legal = native_nd and bias_macro_matches and pn is not None
        if bl1_legal:
            pm = h.cores // pn
            parent_n = up(cd(s.n, pn))
            parent_m = up(cd(s.m, pm))
            bl1_legal = (parent_n <= ncap and s.m > pm * bm and
                         cd(s.m, parent_m) == pm and cd(s.n, parent_n) == pn)
        state['B_resident_N_capacity'] = ncap
        state['B_resident_required_n_cores'] = pn_min
        if bl1_legal:
            family = 'BL1'
            single_m, single_n = parent_m, parent_n
            # 由實際resident N片段反推baseN，然後回填L0C允許的baseM。
            bn = 16 * divisor_down(single_n // 16, min(bn, single_n) // 16)
            resident = single_n * kp * d
            bm = down(min(256, h.l0c // (4 * bn), (s.m - 1) // pm,
                          h.l0a // (2 * 16 * d), (l1 - resident) // (2 * 16 * d)))
            kcap = down(min(up(s.k), h.l0a // (2 * bm * d), h.l0b // (2 * bn * d),
                            (l1 - resident) // (2 * bm * d)))
            bk = 16 * divisor_down(kp // 16, kcap // 16)
            tau = kp // bk
            a = bm * bk * d
            sn = single_n // bn
            cap = (l1 - resident) // (2 * a)
            ta = divisor_down(tau, cap)
            tb = tau
            da, db1 = 2 * ta, sn * tb
            db = 2 if 8 * bm * bn <= h.l0c else 1
            order = 0
            state['resident_bytes'] = resident
            state['resident_opposite_step_cap'] = cap
            state['projected_core_grid'] = [pm, pn]
        elif (s.format_a != 'NZ' or s.format_b != 'NZ') and state['primitive_mn_count'] < h.cores and cd(s.k, 256 // d) >= h.cores:
            family = 'DETERMINISTIC_SPLIT_K'
            # 這個現存實現的 static shape 固定 128/128/256/d；不把 BASE 的公式套進來。
            bm = bn = 128
            bk = 256 // d
            nk = s.m > s.n and s.n * d % 32 == 0 and s.m >= 128
            sm, sn = (1, 3) if nk else (3, 1)
            ta = tb = 3
            da, db1 = (6, 9) if nk else (9, 6)
            order = 0 if nk else 1
            quanta = cd(s.k, bk)
            desired = min(h.cores, quanta)
            guarded_33mk = not nk and s.m <= 256
            chunk_quanta = 3 if guarded_33mk else cd(quanta, desired)
            state['NBuffer33_K_span_unchanged'] = guarded_33mk
            single_k = chunk_quanta * bk
            kcnt = cd(s.k, single_k)
            used = min(h.cores, kcnt)
            panel = (sn * bn if nk else sm * bm)
            live_panel = min(s.n if nk else s.m, panel)
            l2_capacity = split_k_l2_long_capacity(h.l2, used, single_k, live_panel, d, d, 128)
            long_cap = l2_capacity['capacity']
            other = s.m if nk else s.n
            if not l2_capacity['applied']:
                # L2 是快取而非此處的硬儲存容量；配額不成立就不使用長邊容量公式。
                other_tiled = other
            else:
                count = cd(other, long_cap)
                # 在容量以內平衡對齊塊，不能 ceil-align 後又超出 long_cap。
                other_tiled = other if count == 1 else up(cd(other, count), 128)
                require(other_tiled <= long_cap, 'L2_LONG_CAPACITY', 'aligned opposite dimension exceeds its cap')
            single_m, single_n = (other_tiled, panel) if nk else (panel, other_tiled)
            db = 2
            rounds = cd(kcnt, used)
            first_count = kcnt % used or used
            owner_ranges = []
            for core in range(used):
                count = rounds if core < first_count else rounds - 1
                start = core * rounds if core < first_count else first_count * rounds + (core - first_count) * (rounds - 1)
                owner_ranges.append(dict(core=core, first_chunk=start, chunk_count=count,
                                         K_begin=start * single_k, K_end=min(s.k, (start + count) * single_k)))
            state.update(K_quanta=quanta, K_chunk_quanta=chunk_quanta,
                         K_chunk_count=kcnt, L2_opposite_capacity=long_cap, L2_capacity_rule=l2_capacity,
                         K_owner_intervals=owner_ranges)
            l2 = dict(mTile=1, nTile=1, mTileBlock=1, nTileBlock=1, calOrder=0,
                      ignored_by_this_family=True)
        elif (s.format_a != 'NZ' or s.format_b != 'NZ') and gm0 * gn0 >= h.cores and sc_grid_full and s.k > streaming_cap * bk:
            family = 'SINGLE_CORE_SPLIT_K'
            bm = bn = 128
            bk = 256 // d
            root = ceil_sqrt_ratio(h.cores * s.n, s.m)
            pn = divisor_up(h.cores, min(h.cores, root))
            pm = h.cores // pn
            qpm = 256 // d if s.trans_a else max(16, 32 // d)
            qpn = max(16, 256 // d)
            single_m, single_n = up(cd(s.m, pm), qpm), up(cd(s.n, pn), qpn)
            nk = native_nd and bias_macro_matches and s.m > s.n and s.n * d % 32 == 0
            sm, sn = (1, min(3, cd(single_n, bn))) if nk else (min(3, cd(single_m, bm)), 1)
            ma, mb = max(2, sm), max(2, sn)
            a, b = bm * bk * d, bn * bk * d
            cap = l1 // (ma * a + mb * b)
            tau = cd(s.k, bk)
            groups = cd(tau, cap)
            ta = tb = cd(tau, groups)
            da, db1 = ma * ta, mb * tb
            single_k = ta * bk
            order = 0 if nk else 1
            panel = sn * bn if nk else sm * bm
            live_panel = min(s.n if nk else s.m, panel)
            qp = qpm if nk else qpn
            l2_capacity = split_k_l2_long_capacity(h.l2, h.cores, single_k, live_panel, d, d, qp)
            long_cap = l2_capacity['capacity']
            if l2_capacity['applied']:
                if nk:
                    single_m = min(single_m, long_cap)
                else:
                    single_n = min(single_n, long_cap)
            db = 2
            state.update(projected_core_grid=[pm, pn], grid_sqrt=root,
                         L1_step_capacity=cap, L2_opposite_capacity=long_cap, L2_capacity_rule=l2_capacity,
                         loop_order='NKM' if nk else 'MKN')
            l2 = dict(mTile=1, nTile=1, mTileBlock=cd(s.m, single_m),
                      nTileBlock=cd(s.n, single_n), calOrder=1 if nk else 0,
                      counts_ignored_order_consumed=True)
        else:
            # BASE 在同一個 wave 容量內依序改寫 M/N；沒有多套完整 tiling。
            waves = cd(gm0 * gn0, h.cores)
            shrink = []
            axes = ['M', 'N'] if not s.trans_a or not s.trans_b else ['N', 'M']
            for axis in axes:
                if axis == 'M':
                    limit = waves * h.cores // cd(s.n, bn)
                    before = bm
                    bm = min(bm, up(cd(s.m, limit), qm))
                    shrink.append(dict(axis=axis, before=before, count_limit=limit, after=bm))
                else:
                    limit = waves * h.cores // cd(s.m, bm)
                    before = bn
                    bn = min(bn, up(cd(s.n, limit), qn))
                    shrink.append(dict(axis=axis, before=before, count_limit=limit, after=bn))
            bk = base_k(s, h, bm, bn)
            a, b = bm * bk * d, bn * bk * d
            cap = l1 // (2 * (a + b))
            tau = cd(s.k, bk)
            groups = cd(tau, cap)
            ta = tb = cd(tau, groups)
            da, db1 = 2 * ta, 2 * tb
            single_m, single_n = bm, bn
            l2 = tile_l2(s, h, bm, bn, cd(s.m, bm), cd(s.n, bn))
            state.update(wave_budget=waves, axis_projection=shrink, L1_step_capacity=cap,
                         K_base_blocks=tau, K_L1_groups=groups)
    gm, gn = cd(s.m, single_m), cd(s.n, single_n)
    if family != 'DETERMINISTIC_SPLIT_K':
        used = min(h.cores, gm * gn)
    if l2 is None:
        l2 = dict(mTile=1, nTile=1, mTileBlock=gm, nTileBlock=gn, calOrder=1)
    fields = dict(baseM=bm, baseN=bn, baseK=bk, singleCoreM=single_m, singleCoreN=single_n,
                  singleCoreK=single_k, usedCoreNum=used, stepM=sm, stepN=sn,
                  stepKa=ta, stepKb=tb, depthA1=da, depthB1=db1, dbL0C=db, iterateOrder=order)
    usage = dict(L0A_double=2 * bm * bk * d, L0B_double=2 * bn * bk * d,
                 L0C=db * bm * bn * 4, L1_AB=(da * bm + db1 * bn) * bk * d,
                 bias_reserved=1024 if s.bias else 0)
    state.update(family=family, fields=fields, l2=l2, resource_bytes=usage,
                 logical_MN_grid=[gm, gn], last_M=s.m - (gm - 1) * single_m,
                 last_N=s.n - (gn - 1) * single_n,
                 last_K=s.k - (cd(s.k, single_k) - 1) * single_k,
                 core_grid_semantics='K ownership; MN are serial panels' if family == 'DETERMINISTIC_SPLIT_K' else 'MN work items')
    state['legality'] = validate_generated_rules(s, h, ci, state)
    state['tcube_initializer'] = known_tcube_initializer(s, state)
    return state


def examples() -> List[Tuple[str, Shape, Hardware]]:
    return [
        ('A_20core_tail', Shape(1025, 768, 256), Hardware(cores=20)),
        ('B_TT_tail', Shape(1024, 777, 384, trans_a=True, trans_b=True), Hardware()),
        ('C_AL1', Shape(8, 512, 4096, dtype='fp32', trans_b=True), Hardware()),
        ('D_BL1', Shape(32768, 512, 512, trans_b=True), Hardware()),
        ('E1_Deterministic_MK', Shape(32, 32, 8192, dtype='fp32'), Hardware()),
        ('E2_Deterministic_NK', Shape(256, 16, 32768), Hardware()),
        ('F_SingleCore_NKM', Shape(8192, 4096, 4096), Hardware(cores=20)),
        ('G_L2', Shape(32768, 32768, 256), Hardware()),
        ('H_bias', Shape(1025, 768, 256, bias=True, bias_dtype='fp16', bias_length=768), Hardware(cores=20)),
        ('I_small_M_NT', Shape(32, 4096, 256, trans_b=True), Hardware()),
    ]



def example_compile_info(shape: Shape) -> CompileInfo:
    """只用於本文十例的顯式配置，不是平台探測，也不代表伺服器已編譯此變體。"""
    return CompileInfo(
        aicore_arch=220, support_l0c2out=True, support_l12_bt_bf16=False,
        orig_dtype_x1=shape.dtype, orig_dtype_x2=shape.b_dtype, orig_dtype_y=shape.c_dtype,
        orig_dtype_bias=shape.bias_dtype if shape.bias else shape.dtype,
        format_x1=shape.format_a, format_x2=shape.format_b, format_y=shape.format_out,
        total_ub_size=192 * 1024)


def main() -> None:
    results = []
    for name, shape, hw in examples():
        compile_info = example_compile_info(shape)
        results.append(dict(name=name, **solve(shape, hw, compile_info)))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
