# Generated offline by tools/instrument_engine.py. Edit development/engine_plain.py.
"""Source-derived host decision projection. Original CANN notices are in NOTICE.txt.

The complete initializer is integrated by the public selector before these writes.
All primary-domain branches are explicit; unsupported domains fail at the API.
Generated engine.py instruments predicates and mutation source anchors offline.
"""
from .arithmetic import U64, MASK32, MASK64, ceil_div as cd, floor_div as fd, align_up as up, align_down as down, f32, ratio_f32, nz_sequential_size
from .constants import CALC_M_BASIC, CALC_MN_BASIC, CALC_MN_BASIC_L0C_256, CALC_ND_BASIC, SUPPORT_ND2NZ_GM2L0, SC_AL1_BASE_N, CORE_FACTORS_20, CORE_FACTORS_24, CORE_FACTORS_32, L2_SIZE_2, L2_TILE_LENGTH, L0C_SIZE_256_KB, SPLIT_K_THRES, RPC_WORKSIZE, MB_SIZE
from .trace import Record, TraceContext, sourced, plain
from .errors import UnsupportedDomain, SourceFieldOverflow

class Engine(TraceContext):

    def __init__(self, request, hardware, trace=False):
        self.init_trace(trace)
        self.request = dict(request)
        self.hardware = dict(hardware)
        self.hw = Record(self, 'compileInfo', **hardware)
        self.args = Record(self, 'args', mValue=request['M'], nValue=request['N'], kValue=request['K'], mOriValue=request['M'], nOriValue=request['N'], isATrans=request['transA'], isBTrans=request['transB'], hasBias=request['bias'], isHf32=request['hf32'], isForceGrpAccForFp32=request['forceGrpAccForFp32'], aType=request['dtype'], bType=request['dtype'], cType=request['output_dtype'], biasType=request['bias_dtype'], aFormat=request['layoutA'], bFormat=request['layoutB'], outFormat=request['layoutC'], nd2nzA=False, nd2nzB=False, isNzA=False, isNzB=False, unAlignProcessType=0, l2Ratio=0.0)
        self.run = Record(self, 'runInfo', needUpdate=False, usedCoreNum=1, singleCoreM=1, singleCoreN=1, singleCoreK=1, baseM=1, baseN=1, baseK=1, stepKa=1, stepKb=1, depthA1=1, depthB1=1, stepM=1, stepN=1, iterateOrder=0, dbL0c=0, baseAN=0, baseAD=0, baseBN=0, baseBD=0, l2Info=Record(self, 'runInfo.l2Info', mTile=1, nTile=1, mTileBlock=0, nTileBlock=0, calOrder=0))
        self.enable = Record(self, 'tilingEnable', tilingEnableSplitCore=0, tilingEnableFullLoad=0, tilingEnableFixOpti=0, tilingEnableSpecialOpti=0, tilingFp32Addmm=0)
        self.aw = U64(2)
        self.bw = U64(2)
        self.cw = U64(2)
        self.tiling_select = 0
        self.basic_m = U64(256)
        self.l2_length = U64(0)
        self.enable_cache = True
        self.l2_cache_flag = U64(0)
        self.cube_extra = {}
        self.calc_m = CALC_M_BASIC
        self.calc_mn = CALC_MN_BASIC
        self.trans = 0
        self.m_align = False
        self.ka_align = False
        self.kb_align = False
        self.n_align = False

    @sourced('GetMoreArgs', 507, 575)
    def get_more_args(self):
        self.site('matmul_v3_base_tiling.cpp', 507)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.aw = U64(4 if self.predicate('matmul_v3_base_tiling.cpp', 508, "a.aType == 'fp32'", a.aType == 'fp32', locals()) else 2)
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.bw = U64(4 if self.predicate('matmul_v3_base_tiling.cpp', 508, "a.bType == 'fp32'", a.bType == 'fp32', locals()) else 2)
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.cw = U64(4 if self.predicate('matmul_v3_base_tiling.cpp', 508, "a.cType == 'fp32'", a.cType == 'fp32', locals()) else 2)
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.m_align = self.predicate('matmul_v3_base_tiling.cpp', 508, 'a.mValue * self.aw % 256 == 0', a.mValue * self.aw % 256 == 0, locals())
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.ka_align = self.predicate('matmul_v3_base_tiling.cpp', 508, 'a.kValue * self.aw % 256 == 0', a.kValue * self.aw % 256 == 0, locals())
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.kb_align = self.predicate('matmul_v3_base_tiling.cpp', 508, 'a.kValue * self.bw % 256 == 0', a.kValue * self.bw % 256 == 0, locals())
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.n_align = self.predicate('matmul_v3_base_tiling.cpp', 508, 'a.nValue * self.bw % 256 == 0', a.nValue * self.bw % 256 == 0, locals())
        self.site('matmul_v3_base_tiling.cpp', 508)
        inner_a, outer_a, align_a = (a.kValue, a.mValue, self.ka_align)
        self.site('matmul_v3_base_tiling.cpp', 508)
        inner_b, outer_b, align_b = (a.nValue, a.kValue, self.n_align)
        self.site('matmul_v3_base_tiling.cpp', 508)
        self.trans = 0
        if self.predicate('matmul_v3_base_tiling.cpp', 524, 'a.isATrans', a.isATrans, locals()):
            self.site('matmul_v3_base_tiling.cpp', 524)
            self.trans = 1
            self.site('matmul_v3_base_tiling.cpp', 524)
            inner_a, outer_a, align_a = (a.mValue, a.kValue, self.m_align)
        if self.predicate('matmul_v3_base_tiling.cpp', 524, 'a.isBTrans', a.isBTrans, locals()):
            self.site('matmul_v3_base_tiling.cpp', 524)
            self.trans = 2
            self.site('matmul_v3_base_tiling.cpp', 524)
            inner_b, outer_b, align_b = (a.kValue, a.nValue, self.kb_align)
        if self.predicate('matmul_v3_base_tiling.cpp', 524, 'a.isATrans and a.isBTrans', a.isATrans and a.isBTrans, locals()):
            self.site('matmul_v3_base_tiling.cpp', 524)
            self.trans = 3
        self.site('matmul_v3_base_tiling.cpp', 538)
        self.calc_m = CALC_MN_BASIC_L0C_256 if self.predicate('matmul_v3_base_tiling.cpp', 538, 'h.l0CSize == L0C_SIZE_256_KB', h.l0CSize == L0C_SIZE_256_KB, locals()) else CALC_M_BASIC
        self.site('matmul_v3_base_tiling.cpp', 538)
        self.calc_mn = CALC_MN_BASIC_L0C_256 if self.predicate('matmul_v3_base_tiling.cpp', 538, 'h.l0CSize == L0C_SIZE_256_KB', h.l0CSize == L0C_SIZE_256_KB, locals()) else CALC_MN_BASIC
        self.site('matmul_v3_base_tiling.cpp', 538)
        self.l2_length = U64(L2_TILE_LENGTH)
        if self.predicate('matmul_v3_base_tiling.cpp', 538, 'not h.supportL0c2out', not h.supportL0c2out, locals()):
            self.site('matmul_v3_base_tiling.cpp', 538)
            self.tiling_select = 1
            return
        self.site('matmul_v3_base_tiling.cpp', 544)
        on_a = self.is_on_the_way(a.aFormat, inner_a, self.aw)
        self.site('matmul_v3_base_tiling.cpp', 544)
        on_b = self.is_on_the_way(a.bFormat, inner_b, self.bw)
        self.site('matmul_v3_base_tiling.cpp', 544)
        a.nd2nzA = self.predicate('matmul_v3_base_tiling.cpp', 544, "(not align_a or inner_a > 65535) and a.aFormat == 'ND' and (not on_a) and (not (a.aType == 'fp32' and (not a.isHf32) and (inner_a < 65535))) and (not (a.aType == 'fp32' and a.isHf32 and (inner_a * self.aw < 512)))", (not align_a or inner_a > 65535) and a.aFormat == 'ND' and (not on_a) and (not (a.aType == 'fp32' and (not a.isHf32) and (inner_a < 65535))) and (not (a.aType == 'fp32' and a.isHf32 and (inner_a * self.aw < 512))), locals())
        self.site('matmul_v3_base_tiling.cpp', 544)
        a.nd2nzB = self.predicate('matmul_v3_base_tiling.cpp', 544, "(not align_b or inner_b > 65535) and a.bFormat == 'ND' and (not on_b) and (not (a.bType == 'fp32' and (not a.isHf32) and (inner_b < 65535))) and (not (a.bType == 'fp32' and a.isHf32 and (inner_b * self.bw < 512)))", (not align_b or inner_b > 65535) and a.bFormat == 'ND' and (not on_b) and (not (a.bType == 'fp32' and (not a.isHf32) and (inner_b < 65535))) and (not (a.bType == 'fp32' and a.isHf32 and (inner_b * self.bw < 512))), locals())
        self.site('matmul_v3_base_tiling.cpp', 559)
        a.nd2nzA = self.predicate('matmul_v3_base_tiling.cpp', 559, 'a.nd2nzA or self.need_vnchw(outer_a, inner_a, on_a, self.aw, a.aFormat)', a.nd2nzA or self.need_vnchw(outer_a, inner_a, on_a, self.aw, a.aFormat), locals())
        self.site('matmul_v3_base_tiling.cpp', 559)
        a.nd2nzB = self.predicate('matmul_v3_base_tiling.cpp', 559, 'a.nd2nzB or self.need_vnchw(outer_b, inner_b, on_b, self.bw, a.bFormat)', a.nd2nzB or self.need_vnchw(outer_b, inner_b, on_b, self.bw, a.bFormat), locals())
        self.site('matmul_v3_base_tiling.cpp', 559)
        conflict1 = self.mata_conflict_a()
        self.site('matmul_v3_base_tiling.cpp', 559)
        conflict2 = self.mata_conflict_b()
        self.site('matmul_v3_base_tiling.cpp', 559)
        a.nd2nzB = self.predicate('matmul_v3_base_tiling.cpp', 559, 'a.nd2nzB or conflict1 or conflict2', a.nd2nzB or conflict1 or conflict2, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 570, 'a.nd2nzA and self.need_vnchw(outer_a, inner_a, on_a, self.aw, a.aFormat)', a.nd2nzA and self.need_vnchw(outer_a, inner_a, on_a, self.aw, a.aFormat), locals()):
            self.site('matmul_v3_base_tiling.cpp', 570)
            a.unAlignProcessType = 1
        else:
            self.site('matmul_v3_base_tiling.cpp', 570)
            a.unAlignProcessType = 0

    @sourced('GetNd2nzA', 483, 493)
    def mata_conflict_a(self):
        self.site('matmul_v3_base_tiling.cpp', 483)
        a, h = (self.args, self.hw)
        return self.predicate('matmul_v3_base_tiling.cpp', 483, "not a.isBTrans and a.nValue % 16384 == 0 and (a.mValue > 4096 or (a.mValue == 4096 and h.aicNum >= 24)) and (a.kValue >= 6656) and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16'))", not a.isBTrans and a.nValue % 16384 == 0 and (a.mValue > 4096 or (a.mValue == 4096 and h.aicNum >= 24)) and (a.kValue >= 6656) and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16')), locals())

    @sourced('GetNd2nzB', 495, 505)
    def mata_conflict_b(self):
        self.site('matmul_v3_base_tiling.cpp', 495)
        a, h = (self.args, self.hw)
        return self.predicate('matmul_v3_base_tiling.cpp', 495, "not a.isATrans and a.isBTrans and (a.kValue == 16384) and (4096 <= a.mValue <= 4480) and (a.nValue >= 7168) and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16')) and (h.aicNum >= 24)", not a.isATrans and a.isBTrans and (a.kValue == 16384) and (4096 <= a.mValue <= 4480) and (a.nValue >= 7168) and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16')) and (h.aicNum >= 24), locals())

    @sourced('IsOnTheWay', 588, 596)
    def is_on_the_way(self, layout, inner, width):
        if self.predicate('matmul_v3_base_tiling.cpp', 588, "layout == 'ND'", layout == 'ND', locals()):
            return self.predicate('matmul_v3_base_tiling.cpp', 588, 'inner * width in SUPPORT_ND2NZ_GM2L0', inner * width in SUPPORT_ND2NZ_GM2L0, locals())
        return False

    @sourced('NeedNd2NzVnchw', 598, 615)
    def need_vnchw(self, outer, inner, on_way, width, layout):
        if self.predicate('matmul_v3_base_tiling.cpp', 598, 'width == 0', width == 0, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 598, "layout == 'ND'", layout == 'ND', locals()):
            self.site('matmul_v3_base_tiling.cpp', 598)
            aligned = self.predicate('matmul_v3_base_tiling.cpp', 598, 'inner * width % 256 == 0', inner * width % 256 == 0, locals())
            self.site('matmul_v3_base_tiling.cpp', 598)
            fits = self.predicate('matmul_v3_base_tiling.cpp', 598, 'outer > 8192 and inner > 1 and (inner * width <= 192 or (inner * width <= 384 and inner % 2 == 0) or (inner * width <= 512 and inner % 4 == 0))', outer > 8192 and inner > 1 and (inner * width <= 192 or (inner * width <= 384 and inner % 2 == 0) or (inner * width <= 512 and inner % 4 == 0)), locals())
            self.site('matmul_v3_base_tiling.cpp', 598)
            c0_equal = self.predicate('matmul_v3_base_tiling.cpp', 598, 'inner == 32 // width', inner == 32 // width, locals())
            return self.predicate('matmul_v3_base_tiling.cpp', 598, 'fits and (not aligned) and (not on_way) and (not c0_equal)', fits and (not aligned) and (not on_way) and (not c0_equal), locals())
        return False

    @sourced('ResetBase', 577, 586)
    def reset_base(self, r):
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.baseM = 256 if self.predicate('matmul_v3_base_tiling.cpp', 577, 'self.hw.l0CSize == L0C_SIZE_256_KB', self.hw.l0CSize == L0C_SIZE_256_KB, locals()) else 128
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.baseN = 256
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.baseK = U64(128) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.stepM = 1
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.stepN = 1
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.iterateOrder = 0
        self.site('matmul_v3_base_tiling.cpp', 577)
        r.dbL0c = 1

    @sourced('SetParamsV310', 926, 938)
    def set_params_v310(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 926, 'not self.hw.supportL12BtBf16', not self.hw.supportL12BtBf16, locals()):
            return
        raise UnsupportedDomain('supportL12BtBf16 advanced architecture is outside the implemented domain')

    @sourced('SelectNZTiling', 617, 629)
    def select_nz(self):
        self.site('matmul_v3_base_tiling.cpp', 617)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 617, "a.aFormat == 'NZ' and a.bFormat == 'NZ'", a.aFormat == 'NZ' and a.bFormat == 'NZ', locals()):
            self.site('matmul_v3_base_tiling.cpp', 617)
            self.tiling_select = 1
        return True

    @sourced('SetBaseBlockTiling', 1105, 1152)
    def set_base_block(self):
        self.site('matmul_v3_base_tiling.cpp', 1105)
        a, h, r = (self.args, self.hw, self.run)
        self.site('matmul_v3_base_tiling.cpp', 1105)
        r.usedCoreNum = h.aicNum
        self.site('matmul_v3_base_tiling.cpp', 1105)
        r.baseM = 128
        self.site('matmul_v3_base_tiling.cpp', 1105)
        r.baseN = 256
        self.site('matmul_v3_base_tiling.cpp', 1105)
        r.baseK = U64(128) // self.aw
        if self.predicate('matmul_v3_base_tiling.cpp', 1105, 'h.l0CSize == L0C_SIZE_256_KB', h.l0CSize == L0C_SIZE_256_KB, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1105)
            r.baseM = 256
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 1105, 'not a.isATrans and (not a.isBTrans)', not a.isATrans and (not a.isBTrans), locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 1105, 'a.isATrans and a.isBTrans', a.isATrans and a.isBTrans, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1105)
            r.baseM = 256
            self.site('matmul_v3_base_tiling.cpp', 1105)
            r.baseN = 128
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 1125, "a.bFormat == 'NZ'", a.bFormat == 'NZ', locals()):
            return
        self.site('matmul_v3_base_tiling.cpp', 1125)
        threshold = (h.aicNum >> 1) * 256
        if self.predicate('matmul_v3_base_tiling.cpp', 1125, 'a.mValue >= threshold and a.nValue >= threshold', a.mValue >= threshold and a.nValue >= threshold, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1125)
            load128 = cd(a.mValue, 128) * a.nValue + cd(a.nValue, 256) * a.mValue
            self.site('matmul_v3_base_tiling.cpp', 1125)
            load256 = cd(a.mValue, 256) * a.nValue + cd(a.nValue, 128) * a.mValue
            if self.predicate('matmul_v3_base_tiling.cpp', 1125, 'load256 < load128', load256 < load128, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1125)
                r.baseM = 256
                self.site('matmul_v3_base_tiling.cpp', 1125)
                r.baseN = 128
        elif self.predicate('matmul_v3_base_tiling.cpp', 1125, 'a.mValue >= threshold', a.mValue >= threshold, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1125)
            r.baseM = 256
            self.site('matmul_v3_base_tiling.cpp', 1125)
            r.baseN = 128
        elif self.predicate('matmul_v3_base_tiling.cpp', 1125, 'a.nValue < threshold and a.mValue > a.nValue', a.nValue < threshold and a.mValue > a.nValue, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1125)
            r.baseM = 256
            self.site('matmul_v3_base_tiling.cpp', 1125)
            r.baseN = 128

    @sourced('DoBasicTiling', 669, 717)
    def do_basic(self):
        self.site('matmul_v3_base_tiling.cpp', 669)
        a, h, r = (self.args, self.hw, self.run)
        self.site('matmul_v3_base_tiling.cpp', 669)
        r.needUpdate = True
        self.site('matmul_v3_base_tiling.cpp', 669)
        self.basic_m = U64(256 if self.predicate('matmul_v3_base_tiling.cpp', 669, 'h.l0CSize == L0C_SIZE_256_KB', h.l0CSize == L0C_SIZE_256_KB, locals()) else 128)
        self.reset_base(r)
        self.site('matmul_v3_base_tiling.cpp', 669)
        r.singleCoreK = a.kValue
        self.site('matmul_v3_base_tiling.cpp', 669)
        aligned_m = up(a.mValue, self.basic_m)
        self.site('matmul_v3_base_tiling.cpp', 669)
        aligned_n = up(a.nValue, 256)
        self.set_base_block()
        self.site('matmul_v3_base_tiling.cpp', 679)
        small_blocks = self.predicate('matmul_v3_base_tiling.cpp', 679, 'aligned_m // r.baseM * (aligned_n // r.baseN) < r.usedCoreNum', aligned_m // r.baseM * (aligned_n // r.baseN) < r.usedCoreNum, locals())
        self.site('matmul_v3_base_tiling.cpp', 679)
        small_shape = self.predicate('matmul_v3_base_tiling.cpp', 679, 'a.mValue < 256 or a.nValue < 256', a.mValue < 256 or a.nValue < 256, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 679, '(small_blocks or small_shape) and (not h.supportL12BtBf16)', (small_blocks or small_shape) and (not h.supportL12BtBf16), locals()):
            self.do_small()
        if self.predicate('matmul_v3_base_tiling.cpp', 687, 'not h.supportL0c2out', not h.supportL0c2out, locals()):
            self.site('matmul_v3_base_tiling.cpp', 687)
            r.baseM = min(r.baseM, a.mValue)
            self.site('matmul_v3_base_tiling.cpp', 687)
            r.baseN = min(r.baseN, a.nValue)
            self.site('matmul_v3_base_tiling.cpp', 687)
            r.baseK = min(r.baseK, a.kValue)
        if self.predicate('matmul_v3_base_tiling.cpp', 687, 'h.supportL12BtBf16', h.supportL12BtBf16, locals()):
            raise UnsupportedDomain('advanced-architecture base tiling is not implemented')
        self.cal_l1()
        self.do_incre()
        self.optimize_load_balance()
        self.do_select()
        self.do_nd2nz_vector()
        if self.predicate('matmul_v3_base_tiling.cpp', 714, 'a.hasBias', a.hasBias, locals()):
            self.site('matmul_v3_base_tiling.cpp', 714)
            r = self.run
            self.site('matmul_v3_base_tiling.cpp', 714)
            r.baseN = min(U64(256), r.baseN)
            if self.predicate('matmul_v3_base_tiling.cpp', 714, 'self.enable.tilingEnableSplitCore == 0 and self.enable.tilingEnableFullLoad != 2', self.enable.tilingEnableSplitCore == 0 and self.enable.tilingEnableFullLoad != 2, locals()):
                self.site('matmul_v3_base_tiling.cpp', 714)
                r.singleCoreN = r.baseN

    @sourced('DoSmallShapeTiling', 1381, 1404)
    def do_small(self):
        self.site('matmul_v3_base_tiling.cpp', 1381)
        a, r = (self.args, self.run)
        self.formulaic()
        if self.predicate('matmul_v3_base_tiling.cpp', 1381, 'self.hw.l0CSize != L0C_SIZE_256_KB', self.hw.l0CSize != L0C_SIZE_256_KB, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1381, '128 < a.mValue < 256 and r.baseN > 128', 128 < a.mValue < 256 and r.baseN > 128, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1381)
                r.baseN = 128
            elif self.predicate('matmul_v3_base_tiling.cpp', 1381, '128 < a.nValue < 256 and r.baseM > 128', 128 < a.nValue < 256 and r.baseM > 128, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1381)
                r.baseM = 128
            elif self.predicate('matmul_v3_base_tiling.cpp', 1381, 'r.baseM * r.baseN > 32768', r.baseM * r.baseN > 32768, locals()):
                if self.predicate('matmul_v3_base_tiling.cpp', 1381, '128 < r.baseM < 256', 128 < r.baseM < 256, locals()):
                    self.site('matmul_v3_base_tiling.cpp', 1381)
                    r.baseM = 128
                if self.predicate('matmul_v3_base_tiling.cpp', 1381, '128 < r.baseN < 256', 128 < r.baseN < 256, locals()):
                    self.site('matmul_v3_base_tiling.cpp', 1381)
                    r.baseN = 128
            if self.predicate('matmul_v3_base_tiling.cpp', 1381, 'r.baseM * r.baseN > 32768', r.baseM * r.baseN > 32768, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1381)
                r.baseM = U64(32768) // r.baseN

    @sourced('CalBaseSize', 103, 115)
    def cal_base_size(self, count, cores, size, max_base):
        if self.predicate('matmul_v3_base_tiling.cpp', 103, 'count == 0', count == 0, locals()):
            self.site('matmul_v3_base_tiling.cpp', 103)
            current = U64(1)
        else:
            self.site('matmul_v3_base_tiling.cpp', 103)
            current = max(cores // count, U64(1))
        return min(up(max(size // current, U64(1)), 16), U64(max_base))

    @sourced('FormulaicTilingNoTrans', 1483, 1501)
    def formulaic_nn(self):
        self.site('matmul_v3_base_tiling.cpp', 1483)
        a, r = (self.args, self.run)
        self.site('matmul_v3_base_tiling.cpp', 1483)
        saved_ncore = cd(a.nValue, r.baseN)
        self.calc_base(self.calc_m, False)
        if self.predicate('matmul_v3_base_tiling.cpp', 1488, 'not self.ka_align and (not self.n_align)', not self.ka_align and (not self.n_align), locals()):
            self.balance_base()
        elif self.predicate('matmul_v3_base_tiling.cpp', 1488, 'not self.ka_align', not self.ka_align, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1488)
            r.baseN = 256
            self.site('matmul_v3_base_tiling.cpp', 1488)
            r.baseK = U64(128) // self.aw
            self.site('matmul_v3_base_tiling.cpp', 1488)
            r.baseM = self.cal_base_size(saved_ncore, self.hw.aicNum, a.mValue, self.basic_m)
        elif self.predicate('matmul_v3_base_tiling.cpp', 1488, 'not self.n_align', not self.n_align, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1488)
            r.baseK = U64(128) // self.aw
            self.site('matmul_v3_base_tiling.cpp', 1488)
            r.baseM, r.baseN = self.cal_mn(r.baseM, r.baseN)

    @sourced('FormulaicBaseBlockTiling', 1503, 1557)
    def formulaic(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 1503, 'self.trans == 0', self.trans == 0, locals()):
            self.formulaic_nn()
        else:
            raise UnsupportedDomain('transpose geometry is not exposed in this primary-domain implementation')
        if self.predicate('matmul_v3_base_tiling.cpp', 1503, "self.args.outFormat == 'ND' and (not self.hw.supportL0c2out)", self.args.outFormat == 'ND' and (not self.hw.supportL0c2out), locals()):
            self.site('matmul_v3_base_tiling.cpp', 1503)
            self.run.baseN = min(self.run.baseN, U64(256))

    @sourced('BalanceBaseBlockTiling', 1559, 1602)
    def balance_base(self):
        self.site('matmul_v3_base_tiling.cpp', 1559)
        a, h, r = (self.args, self.hw, self.run)
        self.reset_base(r)
        if self.predicate('matmul_v3_base_tiling.cpp', 1565, 'a.mValue >= self.basic_m', a.mValue >= self.basic_m, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1565)
            mcore = max(cd(a.mValue, self.basic_m), U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1565)
            ncore = max(h.aicNum // mcore, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1565)
            r.baseN = min(up(cd(a.nValue, ncore), 16), U64(256))
            self.site('matmul_v3_base_tiling.cpp', 1565)
            ka = h.l0ASize // 2 // self.aw // 256
            self.site('matmul_v3_base_tiling.cpp', 1565)
            kb = fd(h.l0BSize, 2 * self.bw * r.baseN)
            self.site('matmul_v3_base_tiling.cpp', 1565)
            r.baseK = down(min(ka, kb), 16)
        else:
            self.site('matmul_v3_base_tiling.cpp', 1575)
            max_n = self.basic_m if self.predicate('matmul_v3_base_tiling.cpp', 1575, 'a.mValue >= 64', a.mValue >= 64, locals()) else U64(1024)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            r.baseM = up(a.mValue, 16)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            ntile = max(a.nValue // h.aicNum, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1575)
            r.baseN = max(up(ntile, 16), max_n)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            max_by_c = down(fd(h.l0CSize, r.baseM * 4), 16)
            if self.predicate('matmul_v3_base_tiling.cpp', 1575, 'a.hasBias', a.hasBias, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1575)
                max_by_c = min(max_by_c, h.btSize // 4)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            r.baseN = min(r.baseN, max_by_c)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            ncore = cd(a.nValue, r.baseN)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            tailcore = cd(a.nValue, 256) % h.aicNum if self.predicate('matmul_v3_base_tiling.cpp', 1575, 'r.baseN > 256', r.baseN > 256, locals()) else U64(0)
            if self.predicate('matmul_v3_base_tiling.cpp', 1575, 'ncore % h.aicNum < tailcore', ncore % h.aicNum < tailcore, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1575)
                r.baseN = 256
            self.site('matmul_v3_base_tiling.cpp', 1575)
            ka = fd(h.l0ASize, 2 * self.aw * r.baseM)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            kb = fd(h.l0BSize, 2 * self.bw * r.baseN)
            self.site('matmul_v3_base_tiling.cpp', 1575)
            r.baseK = down(min(ka, kb), 16)
        if self.predicate('matmul_v3_base_tiling.cpp', 1596, 'r.baseM == 0 or r.baseN == 0 or r.baseK == 0 or r.baseM % 16 or r.baseN % 16 or r.baseK % 16', r.baseM == 0 or r.baseN == 0 or r.baseK == 0 or r.baseM % 16 or r.baseN % 16 or r.baseK % 16, locals()):
            self.reset_base(r)

    @sourced('CheckBTSize', 1604, 1610)
    def check_bt(self, base_n):
        if self.predicate('matmul_v3_base_tiling.cpp', 1604, 'not self.hw.supportL0c2out and base_n * 4 > self.hw.btSize', not self.hw.supportL0c2out and base_n * 4 > self.hw.btSize, locals()):
            return False
        return True

    @sourced('CalcBase', 1612, 1649)
    def calc_base(self, templates, mn_mode):
        self.site('matmul_v3_base_tiling.cpp', 1612)
        a, h, r = (self.args, self.hw, self.run)
        self.site('matmul_v3_base_tiling.cpp', 1612)
        computed = []
        self.site('matmul_v3_base_tiling.cpp', 1612)
        index = 0
        self.site('matmul_v3_base_tiling.cpp', 1612)
        minimum = U64(MASK64)
        for i, template in self.iterations('CalcBase.templates', enumerate(templates)):
            self.site('matmul_v3_base_tiling.cpp', 1612)
            bm, bn, kbytes = map(U64, template)
            if self.predicate('matmul_v3_base_tiling.cpp', 1612, 'not mn_mode', not mn_mode, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1612)
                ncnt = cd(a.nValue, bn)
                self.site('matmul_v3_base_tiling.cpp', 1612)
                bm = self.cal_base_size(ncnt, h.aicNum, a.mValue, bm)
                self.site('matmul_v3_base_tiling.cpp', 1612)
                mcnt = cd(a.mValue, bm)
            else:
                self.site('matmul_v3_base_tiling.cpp', 1612)
                bm, bn = self.cal_mn(bm, bn, self.calc_mn[i][0], self.calc_mn[i][1])
                self.site('matmul_v3_base_tiling.cpp', 1612)
                mcnt = cd(a.mValue, bm)
                self.site('matmul_v3_base_tiling.cpp', 1612)
                ncnt = cd(a.nValue, bn)
            computed.append((bm, bn, kbytes))
            if self.predicate('matmul_v3_base_tiling.cpp', 1635, 'a.hasBias and (not self.check_bt(bn))', a.hasBias and (not self.check_bt(bn)), locals()):
                continue
            self.site('matmul_v3_base_tiling.cpp', 1635)
            tail = mcnt * ncnt % h.aicNum
            self.site('matmul_v3_base_tiling.cpp', 1635)
            load = (bm + bn) * (mcnt * ncnt // h.aicNum)
            load += bm + bn if self.predicate('matmul_v3_base_tiling.cpp', 1635, 'tail > 0', tail > 0, locals()) else 0
            if self.predicate('matmul_v3_base_tiling.cpp', 1635, 'load < minimum', load < minimum, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1635)
                index = i
                self.site('matmul_v3_base_tiling.cpp', 1635)
                minimum = load
        self.site('matmul_v3_base_tiling.cpp', 1635)
        r.baseM, r.baseN, kbytes = computed[index]
        self.site('matmul_v3_base_tiling.cpp', 1635)
        r.baseK = kbytes // self.aw

    @sourced('CalBaseMBaseN', 1651, 1673)
    def cal_mn(self, bm, bn, max_m=128, max_n=256):
        self.site('matmul_v3_base_tiling.cpp', 1651)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_base_tiling.cpp', 1651)
        bm, bn, max_m, max_n = map(U64, (bm, bn, max_m, max_n))
        self.site('matmul_v3_base_tiling.cpp', 1651)
        mc = cd(a.mValue, max_m)
        self.site('matmul_v3_base_tiling.cpp', 1651)
        nc = cd(a.nValue, max_n)
        if self.predicate('matmul_v3_base_tiling.cpp', 1651, 'mc < nc', mc < nc, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1651)
            nc = max(h.aicNum // mc, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1651)
            bn = up(max(a.nValue // nc, U64(1)), 16)
            if self.predicate('matmul_v3_base_tiling.cpp', 1651, 'bn > max_n', bn > max_n, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1651)
                bn = max_n
        else:
            self.site('matmul_v3_base_tiling.cpp', 1651)
            mc = max(h.aicNum // nc, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1651)
            bm = up(max(a.mValue // mc, U64(1)), 16)
            if self.predicate('matmul_v3_base_tiling.cpp', 1651, 'bm > max_m', bm > max_m, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1651)
                bm = max_m
        return (up(min(a.mValue, bm), 16), up(min(a.nValue, bn), 16))

    @sourced('CalL1Tiling', 631, 667)
    def cal_l1(self):
        self.site('matmul_v3_base_tiling.cpp', 631)
        h, r = (self.hw, self.run)
        if self.predicate('matmul_v3_base_tiling.cpp', 631, 'not h.supportL0c2out', not h.supportL0c2out, locals()):
            raise UnsupportedDomain('310P L1 path is outside the supported domain')
        self.site('matmul_v3_base_tiling.cpp', 631)
        total = h.l1Size + 256
        self.site('matmul_v3_base_tiling.cpp', 631)
        r.depthA1 = total // 2 // r.baseM // r.baseK // self.aw
        self.site('matmul_v3_base_tiling.cpp', 631)
        r.depthB1 = total // 2 // r.baseN // r.baseK // self.bw
        if self.predicate('matmul_v3_base_tiling.cpp', 642, "r.depthA1 * r.baseM * r.baseK * self.aw + r.depthB1 * r.baseN * r.baseK * self.bw > total - (1024 if self.predicate('matmul_v3_base_tiling.cpp', 642, 'self.args.hasBias', self.args.hasBias, locals()) else 0)", r.depthA1 * r.baseM * r.baseK * self.aw + r.depthB1 * r.baseN * r.baseK * self.bw > total - (1024 if self.predicate('matmul_v3_base_tiling.cpp', 642, 'self.args.hasBias', self.args.hasBias, locals()) else 0), locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 642, 'r.baseM <= r.baseN', r.baseM <= r.baseN, locals()):
                self.site('matmul_v3_base_tiling.cpp', 642)
                r.depthA1 = r.depthA1 // 2
            else:
                self.site('matmul_v3_base_tiling.cpp', 642)
                r.depthB1 = r.depthB1 // 2
        self.site('matmul_v3_base_tiling.cpp', 642)
        r.stepKa = r.depthA1 // 2
        self.site('matmul_v3_base_tiling.cpp', 642)
        r.stepKb = r.depthB1 // 2
        self.site('matmul_v3_base_tiling.cpp', 642)
        r.stepKa = self.update_l1_step(r.stepKa)
        self.site('matmul_v3_base_tiling.cpp', 642)
        r.stepKb = self.update_l1_step(r.stepKb)
        if self.predicate('matmul_v3_base_tiling.cpp', 655, 'r.stepKa >= r.stepKb', r.stepKa >= r.stepKb, locals()):
            self.site('matmul_v3_base_tiling.cpp', 655)
            r.stepKa = r.stepKa // r.stepKb * r.stepKb
        else:
            self.site('matmul_v3_base_tiling.cpp', 655)
            r.stepKb = r.stepKb // r.stepKa * r.stepKa
        self.site('matmul_v3_base_tiling.cpp', 655)
        r.depthA1 = r.stepKa * 2
        self.site('matmul_v3_base_tiling.cpp', 655)
        r.depthB1 = r.stepKb * 2
        self.site('matmul_v3_base_tiling.cpp', 655)
        r.singleCoreM = r.baseM
        self.site('matmul_v3_base_tiling.cpp', 655)
        r.singleCoreN = r.baseN

    @sourced('UpdateL1TilingStepK', 1712, 1733)
    def update_l1_step(self, step):
        self.site('matmul_v3_base_tiling.cpp', 1712)
        r = self.run
        if self.predicate('matmul_v3_base_tiling.cpp', 1712, 'step * r.baseK >= self.args.kValue', step * r.baseK >= self.args.kValue, locals()):
            return step
        self.site('matmul_v3_base_tiling.cpp', 1712)
        block = r.baseK * self.aw
        if self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block > 512', step * block > 512, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block % 512 != 0 and 512 % block == 0', step * block % 512 != 0 and 512 % block == 0, locals()):
                while self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block % 512 != 0 and step > 1', step * block % 512 != 0 and step > 1, locals()):
                    self.count_loop('UpdateL1Step.512')
                    step -= 1
        elif self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block > 256', step * block > 256, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block % 256 != 0 and 256 % block == 0', step * block % 256 != 0 and 256 % block == 0, locals()):
                while self.predicate('matmul_v3_base_tiling.cpp', 1718, 'step * block % 256 != 0 and step > 1', step * block % 256 != 0 and step > 1, locals()):
                    self.count_loop('UpdateL1Step.256')
                    step -= 1
        return step

    @sourced('GetMixNd2nzType', 2287, 2298)
    def mix_type(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 2287, 'self.hw.supportL12BtBf16', self.hw.supportL12BtBf16, locals()):
            return 1
        if self.predicate('matmul_v3_base_tiling.cpp', 2287, 'self.hw.supportL0c2out', self.hw.supportL0c2out, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2287, 'self.args.nd2nzA or self.args.nd2nzB', self.args.nd2nzA or self.args.nd2nzB, locals()):
                return 0
        return 1

    @sourced('DoIncreTiling', 2243, 2285)
    def do_incre(self):
        self.site('matmul_v3_base_tiling.cpp', 2243)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2243, 'not h.supportL0c2out', not h.supportL0c2out, locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2249, 'a.mValue > 128', a.mValue > 128, locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2249, 'self.mix_type() != 1', self.mix_type() != 1, locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2257, "a.aType not in ('fp16', 'bf16') or a.bType not in ('fp16', 'bf16') or a.cType not in ('fp16', 'bf16')", a.aType not in ('fp16', 'bf16') or a.bType not in ('fp16', 'bf16') or a.cType not in ('fp16', 'bf16'), locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2257, "a.aFormat != 'ND' or a.bFormat != 'ND' or a.outFormat != 'ND'", a.aFormat != 'ND' or a.bFormat != 'ND' or a.outFormat != 'ND', locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2267, 'a.isATrans or not a.isBTrans', a.isATrans or not a.isBTrans, locals()):
            return
        if self.predicate('matmul_v3_base_tiling.cpp', 2267, 'a.hasBias', a.hasBias, locals()):
            return
        self.site('matmul_v3_base_tiling.cpp', 2275)
        self.enable.tilingEnableSplitCore = 0
        self.site('matmul_v3_base_tiling.cpp', 2275)
        self.enable.tilingEnableFullLoad = 0
        self.site('matmul_v3_base_tiling.cpp', 2275)
        self.enable.tilingEnableFixOpti = 0
        self.site('matmul_v3_base_tiling.cpp', 2275)
        self.function_counts['GetV2Tiling_UNSUPPORTED'] = self.function_counts.get('GetV2Tiling_UNSUPPORTED', 0) + 1
        raise UnsupportedDomain('GetV2Tiling -> GenTiling is unreachable for NN; NT expansion is explicitly unsupported')

    @sourced('OptimizeLoadBalanceBasicKernel', 765, 806)
    def optimize_load_balance(self):
        self.site('matmul_v3_base_tiling.cpp', 765)
        a, h, r, e = (self.args, self.hw, self.run, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 765)
        base_ok = self.predicate('matmul_v3_base_tiling.cpp', 765, 'r.baseM * r.baseN == 32768 and r.baseK == 128 // self.aw', r.baseM * r.baseN == 32768 and r.baseK == 128 // self.aw, locals())
        self.site('matmul_v3_base_tiling.cpp', 765)
        mix = self.mix_type()
        self.site('matmul_v3_base_tiling.cpp', 765)
        key_ok = self.predicate('matmul_v3_base_tiling.cpp', 765, 'e.tilingEnableFullLoad == 0 and e.tilingEnableSplitCore == 0 and (e.tilingEnableFixOpti == 0) and (mix in (0, 1))', e.tilingEnableFullLoad == 0 and e.tilingEnableSplitCore == 0 and (e.tilingEnableFixOpti == 0) and (mix in (0, 1)), locals())
        self.site('matmul_v3_base_tiling.cpp', 765)
        core_ok = self.predicate('matmul_v3_base_tiling.cpp', 765, 'h.aicNum == 24', h.aicNum == 24, locals())
        self.site('matmul_v3_base_tiling.cpp', 765)
        mc = cd(a.mValue, r.baseM)
        self.site('matmul_v3_base_tiling.cpp', 765)
        nc = cd(a.nValue, r.baseN)
        self.site('matmul_v3_base_tiling.cpp', 765)
        small_rounds = self.predicate('matmul_v3_base_tiling.cpp', 765, 'mc * nc < 4 * h.aicNum', mc * nc < 4 * h.aicNum, locals())
        self.site('matmul_v3_base_tiling.cpp', 765)
        common = self.predicate('matmul_v3_base_tiling.cpp', 765, 'key_ok and core_ok and base_ok and (not a.isATrans)', key_ok and core_ok and base_ok and (not a.isATrans), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 789, 'common and small_rounds', common and small_rounds, locals()):
            self.site('matmul_v3_base_tiling.cpp', 789)
            tasks = up(mc * nc, h.aicNum)
            self.site('matmul_v3_base_tiling.cpp', 789)
            mc = fd(tasks, nc)
            self.site('matmul_v3_base_tiling.cpp', 789)
            r.baseM = up(cd(a.mValue, mc), 16)
            self.site('matmul_v3_base_tiling.cpp', 789)
            r.singleCoreM = r.baseM
        self.site('matmul_v3_base_tiling.cpp', 797)
        small_m = self.predicate('matmul_v3_base_tiling.cpp', 797, 'a.mValue <= 672 and a.mValue > 128', a.mValue <= 672 and a.mValue > 128, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 797, 'common and (not small_rounds) and small_m', common and (not small_rounds) and small_m, locals()):
            while self.predicate('matmul_v3_base_tiling.cpp', 797, 'r.baseM > 16 and cd(a.mValue, r.baseM - 16) == mc', r.baseM > 16 and cd(a.mValue, r.baseM - 16) == mc, locals()):
                self.count_loop('OptimizeLoadBalance.reduce_m')
                r.baseM -= 16
            self.site('matmul_v3_base_tiling.cpp', 797)
            r.singleCoreM = r.baseM

    @sourced('IsPowerOfTwo', 763, 763)
    def is_power_of_two(self, x):
        return self.predicate('matmul_v3_base_tiling.cpp', 763, 'x > 0 and x & x - 1 == 0', x > 0 and x & x - 1 == 0, locals())

    @sourced('OptimizeBasicKernelStepK', 808, 881)
    def optimize_step_k(self):
        self.site('matmul_v3_base_tiling.cpp', 808)
        a, h, r, e = (self.args, self.hw, self.run, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 808)
        base_ok = self.predicate('matmul_v3_base_tiling.cpp', 808, 'r.baseM * r.baseN == 32768 and r.baseK == 64', r.baseM * r.baseN == 32768 and r.baseK == 64, locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        aligned = self.predicate('matmul_v3_base_tiling.cpp', 808, 'a.mValue % 256 == 0 and a.nValue % 256 == 0 and (a.kValue % 256 == 0) and (a.mValue >= 768) and (a.nValue >= 768) and (not self.is_power_of_two(a.kValue))', a.mValue % 256 == 0 and a.nValue % 256 == 0 and (a.kValue % 256 == 0) and (a.mValue >= 768) and (a.nValue >= 768) and (not self.is_power_of_two(a.kValue)), locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        mid_m = self.predicate('matmul_v3_base_tiling.cpp', 808, '10368 <= a.mValue <= 18000 and 1280 <= a.nValue <= 5120 and (1280 <= a.kValue <= 5120)', 10368 <= a.mValue <= 18000 and 1280 <= a.nValue <= 5120 and (1280 <= a.kValue <= 5120), locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        mid_k = self.predicate('matmul_v3_base_tiling.cpp', 808, '1280 <= a.mValue <= 5120 and 1280 <= a.nValue <= 5120 and (13788 <= a.kValue <= 19304)', 1280 <= a.mValue <= 5120 and 1280 <= a.nValue <= 5120 and (13788 <= a.kValue <= 19304), locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        big_m = self.predicate('matmul_v3_base_tiling.cpp', 808, '320000 <= a.mValue <= 380000 and 960 <= a.nValue <= 3500 and (1280 <= a.kValue <= 3328)', 320000 <= a.mValue <= 380000 and 960 <= a.nValue <= 3500 and (1280 <= a.kValue <= 3328), locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        global_mn = self.predicate('matmul_v3_base_tiling.cpp', 808, 'a.mValue * a.nValue > 32768 * h.aicNum', a.mValue * a.nValue > 32768 * h.aicNum, locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        core_ok = self.predicate('matmul_v3_base_tiling.cpp', 808, 'h.aicNum == 24', h.aicNum == 24, locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        not_mata = self.predicate('matmul_v3_base_tiling.cpp', 808, 'a.mValue % 16384 != 0 and a.nValue % 16384 != 0', a.mValue % 16384 != 0 and a.nValue % 16384 != 0, locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        dtype_ok = self.predicate('matmul_v3_base_tiling.cpp', 808, "a.aType in ('fp16', 'bf16') and a.bType in ('fp16', 'bf16') and (a.cType in ('fp16', 'bf16'))", a.aType in ('fp16', 'bf16') and a.bType in ('fp16', 'bf16') and (a.cType in ('fp16', 'bf16')), locals())
        self.site('matmul_v3_base_tiling.cpp', 808)
        self.key_arguments = self.get_key_arguments()
        if self.predicate('matmul_v3_base_tiling.cpp', 850, 'e.tilingEnableFullLoad == 0 and e.tilingEnableSplitCore == 0 and (e.tilingEnableFixOpti == 0) and (self.mix_type() == 1) and base_ok and not_mata and dtype_ok and (aligned or mid_m or mid_k or big_m) and global_mn and core_ok', e.tilingEnableFullLoad == 0 and e.tilingEnableSplitCore == 0 and (e.tilingEnableFixOpti == 0) and (self.mix_type() == 1) and base_ok and not_mata and dtype_ok and (aligned or mid_m or mid_k or big_m) and global_mn and core_ok, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 850, 'r.stepKa == 8', r.stepKa == 8, locals()):
                self.site('matmul_v3_base_tiling.cpp', 850)
                r.depthA1 = r.depthA1 // r.stepKa * 4
                self.site('matmul_v3_base_tiling.cpp', 850)
                r.stepKa = 4
            if self.predicate('matmul_v3_base_tiling.cpp', 850, 'r.stepKb == 8', r.stepKb == 8, locals()):
                self.site('matmul_v3_base_tiling.cpp', 850)
                r.depthB1 = r.depthB1 // r.stepKb * 4
                self.site('matmul_v3_base_tiling.cpp', 850)
                r.stepKb = 4

    @sourced('DoSelectTiling', 1406, 1432)
    def do_select(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.tiling_select == 0', self.tiling_select == 0, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.bl1_fixpipe()', self.bl1_fixpipe(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.al1_full()', self.al1_full(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.sc_al1_full()', self.sc_al1_full(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.bl1_full()', self.bl1_full(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.l2_tiling()', self.l2_tiling(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.single_core_split()', self.single_core_split(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.deterministic_split()', self.deterministic_split(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.l2_310p()', self.l2_310p(), locals()):
                return
        elif self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.tiling_select == 1', self.tiling_select == 1, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.l2_tiling()', self.l2_tiling(), locals()):
                return
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.l2_310p()', self.l2_310p(), locals()):
                return
        elif self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.tiling_select == 2', self.tiling_select == 2, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.single_core_split()', self.single_core_split(), locals()):
                return
        elif self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.tiling_select == 3', self.tiling_select == 3, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 1409, 'self.deterministic_split()', self.deterministic_split(), locals()):
                return

    @sourced('NeedSolveFixBound', 2052, 2091)
    def need_fix_bound(self):
        self.site('matmul_v3_base_tiling.cpp', 2052)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2052, 'h.supportL12BtBf16 or not h.supportL0c2out', h.supportL12BtBf16 or not h.supportL0c2out, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2052, 'a.nValue >= 256', a.nValue >= 256, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2063)
        fix_bound = self.predicate('matmul_v3_base_tiling.cpp', 2063, 'a.kValue <= 256 and a.nValue % (256 // self.cw) != 0 and (256 // self.cw % a.nValue != 0)', a.kValue <= 256 and a.nValue % (256 // self.cw) != 0 and (256 // self.cw % a.nValue != 0), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2063, 'not fix_bound', not fix_bound, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2063, 'a.mValue < h.aicNum * 512', a.mValue < h.aicNum * 512, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2074)
        c0 = U64(32) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 2074)
        scalar_bound = self.predicate('matmul_v3_base_tiling.cpp', 2074, 'a.nValue < c0 and a.kValue < c0', a.nValue < c0 and a.kValue < c0, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2074, 'scalar_bound', scalar_bound, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2074)
        not_k256 = self.predicate('matmul_v3_base_tiling.cpp', 2074, 'a.kValue < 256 // self.aw', a.kValue < 256 // self.aw, locals())
        self.site('matmul_v3_base_tiling.cpp', 2074)
        not_mte2 = self.predicate('matmul_v3_base_tiling.cpp', 2074, '(self.ka_align or not_k256) and (not a.isATrans)', (self.ka_align or not_k256) and (not a.isATrans), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2074, 'self.cw == 2', self.cw == 2, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2074, 'not not_mte2', not not_mte2, locals()):
                return False
            else:
                self.site('matmul_v3_base_tiling.cpp', 2074)
                a.nd2nzA = False
                self.site('matmul_v3_base_tiling.cpp', 2074)
                a.nd2nzB = False
                return True
        return True

    @sourced('SupportForceGrpAccForFp32', 2527, 2533)
    def force_group_fp32(self):
        self.site('matmul_v3_base_tiling.cpp', 2527)
        a = self.args
        return self.predicate('matmul_v3_base_tiling.cpp', 2527, "a.aType == 'fp32' and a.kValue >= 2048 and a.isForceGrpAccForFp32", a.aType == 'fp32' and a.kValue >= 2048 and a.isForceGrpAccForFp32, locals())

    @sourced('DoBL1FullloadWithFixpipeTiling', 1833, 1879)
    def bl1_fixpipe(self):
        self.site('matmul_v3_base_tiling.cpp', 1833)
        a, h, r, e = (self.args, self.hw, self.run, self.enable)
        if self.predicate('matmul_v3_base_tiling.cpp', 1833, 'not self.need_fix_bound() or self.force_group_fp32()', not self.need_fix_bound() or self.force_group_fp32(), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 1838)
        r.baseN = up(a.nValue, 16)
        self.site('matmul_v3_base_tiling.cpp', 1838)
        max_m = h.ubSize // 256 // self.cw
        self.site('matmul_v3_base_tiling.cpp', 1838)
        r.baseM = down(min(fd(h.l0CSize, r.baseN * 4), max_m), 128)
        self.site('matmul_v3_base_tiling.cpp', 1838)
        ka = h.l0ASize // 2 // self.aw // r.baseM
        self.site('matmul_v3_base_tiling.cpp', 1838)
        kb = h.l0BSize // 2 // self.bw // r.baseN
        self.site('matmul_v3_base_tiling.cpp', 1838)
        r.baseK = down(min(ka, kb), 16)
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.depthB1 = cd(a.kValue, r.baseK)
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.stepKb = r.depthB1
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.depthA1 = fd(h.l1Size // self.bw - r.depthB1 * r.baseN * r.baseK, r.baseM * r.baseK)
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.depthA1 = min(r.depthA1, r.depthB1)
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.stepKa = r.depthA1
        if self.predicate('matmul_v3_base_tiling.cpp', 1850, 'r.depthA1 >= 8', r.depthA1 >= 8, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1850)
            r.stepKa = 4
            self.site('matmul_v3_base_tiling.cpp', 1850)
            r.depthA1 = 8
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.singleCoreM = r.baseM
        self.site('matmul_v3_base_tiling.cpp', 1850)
        r.singleCoreN = r.baseN
        if self.predicate('matmul_v3_base_tiling.cpp', 1863, 'r.depthA1 == 0', r.depthA1 == 0, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 1863)
        e.tilingEnableFixOpti = 1
        self.site('matmul_v3_base_tiling.cpp', 1863)
        e.tilingEnableSplitCore = 0
        self.site('matmul_v3_base_tiling.cpp', 1863)
        e.tilingEnableFullLoad = 2
        self.site('matmul_v3_base_tiling.cpp', 1863)
        a.nd2nzB = False
        if self.predicate('matmul_v3_base_tiling.cpp', 1863, "a.aType == 'fp32'", a.aType == 'fp32', locals()):
            self.site('matmul_v3_base_tiling.cpp', 1863)
            c0 = U64(32) // self.aw
            self.site('matmul_v3_base_tiling.cpp', 1863)
            vec_out = self.predicate('matmul_v3_base_tiling.cpp', 1863, 'a.kValue % c0 == 0 and a.nValue <= 192 and (not a.nd2nzA) and (not a.isATrans)', a.kValue % c0 == 0 and a.nValue <= 192 and (not a.nd2nzA) and (not a.isATrans), locals())
            if self.predicate('matmul_v3_base_tiling.cpp', 1863, 'vec_out', vec_out, locals()):
                self.site('matmul_v3_base_tiling.cpp', 1863)
                e.tilingEnableFixOpti = 2
        return True

    @sourced('DoAL1FullLoadTiling', 1881, 1925)
    def al1_full(self):
        self.site('matmul_v3_base_tiling.cpp', 1881)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 1881, "not h.supportL0c2out or a.aType != 'fp32' or a.isATrans or (not a.isBTrans) or self.force_group_fp32()", not h.supportL0c2out or a.aType != 'fp32' or a.isATrans or (not a.isBTrans) or self.force_group_fp32(), locals()):
            return False
        raise UnsupportedDomain('FP32 NT AL1 full-load is outside the primary API domain')

    @sourced('DoBL1FullLoadTiling', 1927, 1979)
    def bl1_full(self):
        self.site('matmul_v3_base_tiling.cpp', 1927)
        a, h, r = (self.args, self.hw, self.run)
        if self.predicate('matmul_v3_base_tiling.cpp', 1927, 'self.force_group_fp32()', self.force_group_fp32(), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 1927)
        c0 = U64(32) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 1927)
        inner = a.mValue if self.predicate('matmul_v3_base_tiling.cpp', 1927, 'a.isATrans', a.isATrans, locals()) else a.kValue
        self.site('matmul_v3_base_tiling.cpp', 1927)
        outer = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 1927, 'a.isATrans', a.isATrans, locals()) else a.mValue
        self.site('matmul_v3_base_tiling.cpp', 1927)
        vnchw = self.predicate('matmul_v3_base_tiling.cpp', 1927, "a.aType == 'fp32' and outer >= 72368 and (inner <= c0) and (inner > 1)", a.aType == 'fp32' and outer >= 72368 and (inner <= c0) and (inner > 1), locals())
        self.site('matmul_v3_base_tiling.cpp', 1939)
        on_fly = self.predicate('matmul_v3_base_tiling.cpp', 1939, 'a.nValue in SUPPORT_ND2NZ_GM2L0 and (not a.isBTrans or a.kValue * self.aw in SUPPORT_ND2NZ_GM2L0)', a.nValue in SUPPORT_ND2NZ_GM2L0 and (not a.isBTrans or a.kValue * self.aw in SUPPORT_ND2NZ_GM2L0), locals())
        self.site('matmul_v3_base_tiling.cpp', 1939)
        valid_mk = self.predicate('matmul_v3_base_tiling.cpp', 1939, 'a.mValue > 16 * max(a.kValue, a.nValue) and a.kValue <= 256', a.mValue > 16 * max(a.kValue, a.nValue) and a.kValue <= 256, locals())
        self.site('matmul_v3_base_tiling.cpp', 1939)
        bias_size = r.baseN * 4 if self.predicate('matmul_v3_base_tiling.cpp', 1939, 'a.hasBias', a.hasBias, locals()) else U64(0)
        self.site('matmul_v3_base_tiling.cpp', 1939)
        aligned_k = up(a.kValue, 32 // self.bw if self.predicate('matmul_v3_base_tiling.cpp', 1939, 'a.isBTrans', a.isBTrans, locals()) else 16)
        self.site('matmul_v3_base_tiling.cpp', 1939)
        aligned_n = up(a.nValue, 16 if self.predicate('matmul_v3_base_tiling.cpp', 1939, 'a.isBTrans', a.isBTrans, locals()) else 32 // self.bw)
        self.site('matmul_v3_base_tiling.cpp', 1939)
        b_valid = self.predicate('matmul_v3_base_tiling.cpp', 1939, 'h.l1Size // 2 - bias_size > aligned_k * aligned_n * self.bw', h.l1Size // 2 - bias_size > aligned_k * aligned_n * self.bw, locals())
        self.site('matmul_v3_base_tiling.cpp', 1939)
        aligned_full = self.predicate('matmul_v3_base_tiling.cpp', 1939, 'on_fly and (not a.nd2nzB)', on_fly and (not a.nd2nzB), locals())
        self.site('matmul_v3_base_tiling.cpp', 1939)
        skip_base = self.predicate('matmul_v3_base_tiling.cpp', 1939, 'not valid_mk or (not aligned_full and (not (vnchw and b_valid)))', not valid_mk or (not aligned_full and (not (vnchw and b_valid))), locals())
        self.site('matmul_v3_base_tiling.cpp', 1953)
        tail = a.mValue // 128 % 12
        self.site('matmul_v3_base_tiling.cpp', 1953)
        tail_ok = False
        if self.predicate('matmul_v3_base_tiling.cpp', 1953, 'tail == 0', tail == 0, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1953)
            tail_ok = self.predicate('matmul_v3_base_tiling.cpp', 1953, '(a.mValue + 127) // 128 % 12 == 1', (a.mValue + 127) // 128 % 12 == 1, locals())
        elif self.predicate('matmul_v3_base_tiling.cpp', 1953, 'tail == 6', tail == 6, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1953)
            tail_ok = self.predicate('matmul_v3_base_tiling.cpp', 1953, '(a.mValue + 127) // 128 % 12 != 7', (a.mValue + 127) // 128 % 12 != 7, locals())
        elif self.predicate('matmul_v3_base_tiling.cpp', 1953, '1 <= tail < 6', 1 <= tail < 6, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1953)
            tail_ok = True
        self.site('matmul_v3_base_tiling.cpp', 1953)
        core_full = self.predicate('matmul_v3_base_tiling.cpp', 1953, "(a.aType == 'bf16' and a.bType == 'bf16' or (a.aType == 'fp16' and a.bType == 'fp16')) and 30848 <= a.mValue <= 98048 and (a.nValue == 512) and (a.kValue == 512) and tail_ok and (not a.isATrans) and a.isBTrans and (a.aFormat == 'ND') and (a.bFormat == 'ND') and (not a.hasBias)", (a.aType == 'bf16' and a.bType == 'bf16' or (a.aType == 'fp16' and a.bType == 'fp16')) and 30848 <= a.mValue <= 98048 and (a.nValue == 512) and (a.kValue == 512) and tail_ok and (not a.isATrans) and a.isBTrans and (a.aFormat == 'ND') and (a.bFormat == 'ND') and (not a.hasBias), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 1973, 'not skip_base', not skip_base, locals()):
            return self.bl1_base()
        elif self.predicate('matmul_v3_base_tiling.cpp', 1973, 'core_full', core_full, locals()):
            raise UnsupportedDomain('NT BL1 core-split full-load is not exposed in the primary API')
        return False

    @sourced('DoBL1FullLoadTilingBase', 1981, 2024)
    def bl1_base(self):
        self.site('matmul_v3_base_tiling.cpp', 1981)
        a, h, r, e = (self.args, self.hw, self.run, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 1981)
        c0 = U64(32) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 1981)
        e.tilingEnableFullLoad = 2
        self.site('matmul_v3_base_tiling.cpp', 1981)
        e.tilingEnableSplitCore = 0
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.stepM = 1
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.baseN = min(a.nValue, r.baseN)
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.baseN = up(r.baseN, 16 if self.predicate('matmul_v3_base_tiling.cpp', 1981, 'a.isBTrans', a.isBTrans, locals()) else c0)
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.stepN = cd(a.nValue, r.baseN)
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.stepKb = cd(a.kValue, r.baseK)
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.stepKa = r.stepKb
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.depthA1 = 2 * r.stepKa
        self.site('matmul_v3_base_tiling.cpp', 1981)
        r.depthB1 = r.stepN * r.stepKb
        self.site('matmul_v3_base_tiling.cpp', 1981)
        load = r.baseK * (r.depthA1 * r.baseM + r.depthB1 * r.baseN) * self.aw
        load += r.baseN * self.aw if self.predicate('matmul_v3_base_tiling.cpp', 1981, 'a.hasBias', a.hasBias, locals()) else 0
        self.site('matmul_v3_base_tiling.cpp', 2000)
        minimum_load = load - (r.baseM - 16) * r.baseK * r.depthA1 * self.aw
        if self.predicate('matmul_v3_base_tiling.cpp', 2000, 'minimum_load > h.l1Size', minimum_load > h.l1Size, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2000)
            e.tilingEnableFullLoad = 0
            return False
        while self.predicate('matmul_v3_base_tiling.cpp', 2000, 'load > h.l1Size', load > h.l1Size, locals()):
            self.count_loop('BL1FullLoad.halve_m')
            load -= r.depthA1 * r.baseM * r.baseK * self.aw
            self.site('matmul_v3_base_tiling.cpp', 2000)
            r.baseM = r.baseM // 2
            load += r.depthA1 * r.baseM * r.baseK * self.aw
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.singleCoreM = 2 * r.baseM
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.singleCoreN = a.nValue
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.dbL0c = 2 if self.predicate('matmul_v3_base_tiling.cpp', 2013, 'r.baseM * r.baseN * 4 * 2 <= h.l0CSize', r.baseM * r.baseN * 4 * 2 <= h.l0CSize, locals()) else 1
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.l2Info.mTile = 1
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.l2Info.nTile = 1
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.l2Info.mTileBlock = cd(a.mValue, r.singleCoreM)
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.l2Info.nTileBlock = 1
        self.site('matmul_v3_base_tiling.cpp', 2013)
        r.l2Info.calOrder = 1
        return True

    @sourced('InitL2SplitParams', 1735, 1758)
    def init_l2_params(self):
        self.site('matmul_v3_base_tiling.cpp', 1735)
        a, h, r = (self.args, self.hw, self.run)
        self.site('matmul_v3_base_tiling.cpp', 1735)
        p = Record(self, 'l2SplitParams', outBase=max(r.baseM, U64(1)), innerBase=max(r.baseN, U64(1)), outValue=a.mValue, innerValue=a.nValue, outDtypeSize=self.aw, innerDtypeSize=self.bw, maxConflictDim=0, minConflictDim=0, outTailCnt=0, innerTailCnt=0)
        if self.predicate('matmul_v3_base_tiling.cpp', 1735, 'r.baseN >= r.baseM', r.baseN >= r.baseM, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.outBase = r.baseN
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.innerBase = r.baseM
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.outValue = a.nValue
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.innerValue = a.mValue
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.outDtypeSize = self.bw
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.innerDtypeSize = self.aw
        self.site('matmul_v3_base_tiling.cpp', 1735)
        p.maxConflictDim = min(h.aicNum, U64(6))
        self.site('matmul_v3_base_tiling.cpp', 1735)
        p.minConflictDim = min(h.aicNum, U64(3))
        if self.predicate('matmul_v3_base_tiling.cpp', 1735, 'h.aicNum == 20', h.aicNum == 20, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.maxConflictDim = 5
            self.site('matmul_v3_base_tiling.cpp', 1735)
            p.minConflictDim = 4
        return p

    @sourced('IsTailSmall', 1760, 1773)
    def l2_tail_small(self, p, out_split, inner_split, inner_max):
        if self.predicate('matmul_v3_base_tiling.cpp', 1760, 'out_split == 0 or inner_split == 0', out_split == 0 or inner_split == 0, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 1760)
        out_tail = (p.outValue + out_split - 1) % out_split + 1
        self.site('matmul_v3_base_tiling.cpp', 1760)
        inner_tail = (p.innerValue + inner_split - 1) % inner_split + 1
        self.site('matmul_v3_base_tiling.cpp', 1760)
        p.outTailCnt = cd(out_tail, p.outBase)
        self.site('matmul_v3_base_tiling.cpp', 1760)
        p.innerTailCnt = cd(inner_tail, p.innerBase)
        return self.predicate('matmul_v3_base_tiling.cpp', 1760, 'p.outTailCnt * p.maxConflictDim < self.hw.aicNum or p.innerTailCnt * inner_max < self.hw.aicNum', p.outTailCnt * p.maxConflictDim < self.hw.aicNum or p.innerTailCnt * inner_max < self.hw.aicNum, locals())

    @sourced('GetTotalSize', 1825, 1831)
    def total_size(self, m, k, n, aw, bw):
        return U64(m) * k * aw + U64(k) * n * bw + U64(m) * n * self.cw

    @sourced('CalcTile', 1775, 1823)
    def calc_l2_tile(self, out_tile, inner_tile, out_split, inner_split, inner_bad):
        self.site('matmul_v3_base_tiling.cpp', 1775)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_base_tiling.cpp', 1775)
        p = self.init_l2_params()
        self.site('matmul_v3_base_tiling.cpp', 1775)
        inner_max = p.minConflictDim if self.predicate('matmul_v3_base_tiling.cpp', 1775, 'inner_bad', inner_bad, locals()) else p.maxConflictDim
        self.site('matmul_v3_base_tiling.cpp', 1775)
        outer_min = max(h.aicNum // p.maxConflictDim, U64(1))
        self.site('matmul_v3_base_tiling.cpp', 1775)
        inner_min = max(h.aicNum // inner_max, U64(1))
        self.site('matmul_v3_base_tiling.cpp', 1775)
        out_original = out_split
        self.site('matmul_v3_base_tiling.cpp', 1775)
        inner_original = inner_split
        self.site('matmul_v3_base_tiling.cpp', 1775)
        out_conflict = U64(0)
        self.site('matmul_v3_base_tiling.cpp', 1775)
        inner_conflict = U64(0)
        self.site('matmul_v3_base_tiling.cpp', 1775)
        enabled = False
        for out_use in self.iterations('CalcTile.outer', range(int(h.aicNum), int(outer_min) - 1, -1)):
            for inner_use in self.iterations('CalcTile.inner', range(int(h.aicNum), int(inner_min) - 1, -1)):
                self.site('matmul_v3_base_tiling.cpp', 1789)
                out_temp = max(out_original // (p.outBase * out_use), U64(1))
                self.site('matmul_v3_base_tiling.cpp', 1789)
                inner_temp = max(inner_original // (p.innerBase * inner_use), U64(1))
                self.site('matmul_v3_base_tiling.cpp', 1789)
                out_split_temp = up(cd(p.outValue, out_temp), p.outBase)
                self.site('matmul_v3_base_tiling.cpp', 1789)
                inner_split_temp = up(cd(p.innerValue, inner_temp), p.innerBase)
                self.site('matmul_v3_base_tiling.cpp', 1789)
                size = self.total_size(out_split_temp, a.kValue, inner_split_temp, p.outDtypeSize, p.innerDtypeSize)
                if self.predicate('matmul_v3_base_tiling.cpp', 1799, 'size <= a.l2Ratio * 100 * MB_SIZE', size <= a.l2Ratio * 100 * MB_SIZE, locals()):
                    if self.predicate('matmul_v3_base_tiling.cpp', 1799, 'self.l2_tail_small(p, out_split_temp, inner_split_temp, inner_max)', self.l2_tail_small(p, out_split_temp, inner_split_temp, inner_max), locals()):
                        continue
                    self.site('matmul_v3_base_tiling.cpp', 1799)
                    out_conf_temp = cd(h.aicNum, p.outTailCnt)
                    self.site('matmul_v3_base_tiling.cpp', 1799)
                    inner_conf_temp = cd(h.aicNum, p.innerTailCnt)
                    self.site('matmul_v3_base_tiling.cpp', 1799)
                    update = self.predicate('matmul_v3_base_tiling.cpp', 1799, 'not enabled or (out_conflict >= out_conf_temp and inner_conflict >= inner_conf_temp)', not enabled or (out_conflict >= out_conf_temp and inner_conflict >= inner_conf_temp), locals())
                    if self.predicate('matmul_v3_base_tiling.cpp', 1799, 'update', update, locals()):
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        enabled = True
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        out_tile = out_temp
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        inner_tile = inner_temp
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        out_split = out_split_temp
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        inner_split = inner_split_temp
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        out_conflict = out_conf_temp
                        self.site('matmul_v3_base_tiling.cpp', 1799)
                        inner_conflict = inner_conf_temp
        return (enabled, out_tile, inner_tile, out_split, inner_split)

    @sourced('DoL2CacheTiling', 2093, 2132)
    def l2_tiling(self):
        self.site('matmul_v3_base_tiling.cpp', 2093)
        a, h, r = (self.args, self.hw, self.run)
        if self.predicate('matmul_v3_base_tiling.cpp', 2095, 'h.supportL12BtBf16 or not h.supportL0c2out', h.supportL12BtBf16 or not h.supportL0c2out, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2095)
            self.enable.tilingEnableSplitCore = 0
            return False
        self.site('matmul_v3_base_tiling.cpp', 2095)
        a.l2Ratio = float(h.l2Size) / L2_SIZE_2
        self.site('matmul_v3_base_tiling.cpp', 2095)
        m_tile = U64(1)
        self.site('matmul_v3_base_tiling.cpp', 2095)
        n_tile = U64(1)
        self.site('matmul_v3_base_tiling.cpp', 2095)
        m_split = a.mValue
        self.site('matmul_v3_base_tiling.cpp', 2095)
        n_split = a.nValue
        self.site('matmul_v3_base_tiling.cpp', 2095)
        total = self.total_size(a.mValue, a.kValue, a.nValue, self.aw, self.bw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2103, 'total > a.l2Ratio * 100 * MB_SIZE', total > a.l2Ratio * 100 * MB_SIZE, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2103, 'r.baseN >= r.baseM', r.baseN >= r.baseM, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2103)
                self.enable_cache, n_tile, m_tile, n_split, m_split = self.calc_l2_tile(n_tile, m_tile, n_split, m_split, a.isATrans)
            else:
                self.site('matmul_v3_base_tiling.cpp', 2103)
                self.enable_cache, m_tile, n_tile, m_split, n_split = self.calc_l2_tile(m_tile, n_tile, m_split, n_split, not a.isBTrans)
        self.site('matmul_v3_base_tiling.cpp', 2120)
        m_block = cd(m_split, r.baseM)
        self.site('matmul_v3_base_tiling.cpp', 2120)
        n_block = cd(n_split, r.baseN)
        self.site('matmul_v3_base_tiling.cpp', 2120)
        m_tile = cd(a.mValue, m_block * r.baseM)
        self.site('matmul_v3_base_tiling.cpp', 2120)
        n_tile = cd(a.nValue, n_block * r.baseN)
        if self.predicate('matmul_v3_base_tiling.cpp', 2120, 'self.tiling_select == 0 and (not self.enable_cache)', self.tiling_select == 0 and (not self.enable_cache), locals()):
            self.site('matmul_v3_base_tiling.cpp', 2120)
            m_block = min(U64(4), cd(a.mValue, r.baseM))
            self.site('matmul_v3_base_tiling.cpp', 2120)
            n_block = min(h.aicNum // 4, cd(a.nValue, r.baseN))
            self.site('matmul_v3_base_tiling.cpp', 2120)
            m_tile = cd(a.mValue, m_block * r.baseM)
            self.site('matmul_v3_base_tiling.cpp', 2120)
            n_tile = cd(a.nValue, n_block * r.baseN)
        self.site('matmul_v3_base_tiling.cpp', 2137)
        r.l2Info.mTile = m_tile
        self.site('matmul_v3_base_tiling.cpp', 2137)
        r.l2Info.nTile = n_tile
        self.site('matmul_v3_base_tiling.cpp', 2137)
        r.l2Info.mTileBlock = m_block
        self.site('matmul_v3_base_tiling.cpp', 2137)
        r.l2Info.nTileBlock = n_block
        return False

    @sourced('DoL2CacheTiling310P', 2134, 2152)
    def l2_310p(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 2134, 'self.hw.supportL0c2out or self.hw.supportL12BtBf16', self.hw.supportL0c2out or self.hw.supportL12BtBf16, locals()):
            return False
        raise UnsupportedDomain('310P L2 branch is not exposed in the primary API')

    @sourced('IsSupportSingleCoreSplitKAL1FullLoad', 2945, 2994)
    def support_sc_al1(self):
        self.site('matmul_v3_base_tiling.cpp', 2945)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, 'not h.supportL0c2out', not h.supportL0c2out, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, 'h.supportL12BtBf16', h.supportL12BtBf16, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, 'a.isATrans', a.isATrans, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, "a.aFormat == 'NZ' or a.bFormat == 'NZ'", a.aFormat == 'NZ' or a.bFormat == 'NZ', locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, "a.aType not in ('fp16', 'bf16')", a.aType not in ('fp16', 'bf16'), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2945, 'not (a.kValue % 128 == 0 and a.nValue % 128 == 0)', not (a.kValue % 128 == 0 and a.nValue % 128 == 0), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2973)
        middle_k = self.predicate('matmul_v3_base_tiling.cpp', 2973, '9216 <= a.kValue <= 20480 and 6144 <= a.nValue < 65536', 9216 <= a.kValue <= 20480 and 6144 <= a.nValue < 65536, locals())
        self.site('matmul_v3_base_tiling.cpp', 2973)
        middle_n = self.predicate('matmul_v3_base_tiling.cpp', 2973, '6144 <= a.kValue < 65536 and 6144 <= a.nValue < 16384', 6144 <= a.kValue < 65536 and 6144 <= a.nValue < 16384, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2973, 'a.mValue > 256', a.mValue > 256, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2973, 'not middle_k and (not middle_n)', not middle_k and (not middle_n), locals()):
            return False
        return True

    @sourced('IsL0CubeFit', 41, 46, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_l0_cube_fit(self, bm, bn, bk):
        return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 41, 'bk > 0 and bn > 0 and (bm * bk * self.aw * 2 <= self.hw.l0ASize) and (bn * bk * self.bw * 2 <= self.hw.l0BSize)', bk > 0 and bn > 0 and (bm * bk * self.aw * 2 <= self.hw.l0ASize) and (bn * bk * self.bw * 2 <= self.hw.l0BSize), locals())

    @sourced('IsL0CFit', 48, 51, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_l0c_fit(self, bm, bn, db):
        return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 48, 'bm * bn * db * 4 <= self.hw.l0CSize', bm * bn * db * 4 <= self.hw.l0CSize, locals())

    @sourced('CalcPolicyBaseK', 140, 162, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_policy_bk(self, bm, bn, db):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140)
        h = self.hw
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140, 'not self.sc_l0c_fit(bm, bn, db)', not self.sc_l0c_fit(bm, bn, db), locals()):
            return U64(0)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140)
        ka = fd(h.l0ASize, 2 * self.aw * bm)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140)
        kb = fd(h.l0BSize, 2 * self.bw * bn)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140)
        maximum = min(down(min(ka, kb), 32), U64(256))
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140, 'maximum < 32', maximum < 32, locals()):
            return U64(0)
        for bk in self.iterations('ScAl1.policy_bk', range(int(maximum), 31, -32)):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140)
            bk = U64(bk)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140, 'self.sc_l0_cube_fit(bm, bn, bk)', self.sc_l0_cube_fit(bm, bn, bk), locals()):
                return bk
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 140, 'bk == 32', bk == 32, locals()):
                break
        return U64(0)

    @sourced('CalcAL1LoopCount', 60, 63, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_a_loops(self, k, step, bk):
        return U64(MASK64) if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 60, 'k == 0 or step == 0 or bk == 0', k == 0 or step == 0 or bk == 0, locals()) else cd(k, step * bk)

    @sourced('CalcBkTileAlignScore', 65, 74, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_b_align(self, byte_count):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 65, 'byte_count == 0', byte_count == 0, locals()):
            return U64(0)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 65, 'byte_count % 512 == 0', byte_count % 512 == 0, locals()):
            return U64(2)
        return U64(1 if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 65, 'byte_count % 256 == 0', byte_count % 256 == 0, locals()) else 0)

    @sourced('CompareBlockRounds', 91, 121, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_compare_rounds(self, cur, best, k):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        cr = cd(cur.singleCoreN, cur.baseN) * cd(k, cur.baseK)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        br = cd(best.singleCoreN, best.baseN) * cd(k, best.baseK)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        cref = cd(cur.singleCoreN, 256) * cd(k, 128)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        bref = cd(best.singleCoreN, 256) * cd(k, 128)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        cle = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cr <= cref', cr <= cref, locals())
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        ble = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'br <= bref', br <= bref, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cle != ble', cle != ble, locals()):
            return 1 if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cle', cle, locals()) else -1
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cle', cle, locals()):
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cr * bref > br * cref', cr * bref > br * cref, locals()):
                return 1
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'cr * bref < br * cref', cr * bref < br * cref, locals()):
                return -1
            return 0
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        ce = cr - cref
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91)
        be = br - bref
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'ce * bref < be * cref', ce * bref < be * cref, locals()):
            return 1
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 91, 'ce * bref > be * cref', ce * bref > be * cref, locals()):
            return -1
        return 0

    @sourced('CalcNTileSplit', 123, 138, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_n_split(self, n, bn, cores):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123, 'bn == 0 or cores == 0', bn == 0 or cores == 0, locals()):
            return (U64(0),) * 4
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123)
        nt = cd(n, bn)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123)
        used = min(nt, cores)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123, 'used == 0', used == 0, locals()):
            return (nt, used, U64(0), U64(0))
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123)
        rem = nt % used
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123)
        head = used if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 123, 'rem == 0', rem == 0, locals()) else rem
        return (nt, used, head, used - head)

    @sourced('EstMinAL1LoopCount', 164, 180, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_est_min_a_loop(self, total, bm, sm, bn, bk, k):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        bias = bn * 4 if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164, 'self.args.hasBias', self.args.hasBias, locals()) else U64(0)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164, 'total <= bias', total <= bias, locals()):
            return U64(MASK64)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        available = total - bias
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        ablock = bm * bk * self.aw
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        bblock = bn * bk * self.bw
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        min_b = 2 * bblock
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164, 'min_b >= available or ablock == 0 or bk == 0', min_b >= available or ablock == 0 or bk == 0, locals()):
            return U64(MASK64)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164)
        max_step = min((available - min_b) // (sm * ablock), k // bk)
        return self.sc_a_loops(k, max_step, bk) if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 164, 'max_step >= 1', max_step >= 1, locals()) else U64(MASK64)

    @sourced('IsBetterL1Step', 182, 205, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_better_l1(self, k, bk, ka, kb, bsize, best_ka, best_kb, best_bsize):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'best_ka == 0', best_ka == 0, locals()):
            return True
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'self.args.isBTrans', self.args.isBTrans, locals()):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182)
            score = self.sc_b_align(kb * bk * self.bw)
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182)
            best_score = self.sc_b_align(best_kb * bk * self.bw)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'score != best_score', score != best_score, locals()):
                return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'score > best_score', score > best_score, locals())
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182)
        loops = self.sc_a_loops(k, ka, bk)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182)
        best_loops = self.sc_a_loops(k, best_ka, bk)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'loops != best_loops', loops != best_loops, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'loops < best_loops', loops < best_loops, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'bsize != best_bsize', bsize != best_bsize, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'bsize > best_bsize', bsize > best_bsize, locals())
        return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 182, 'ka > best_ka', ka > best_ka, locals())

    @sourced('SearchBestKbForStepN', 215, 265, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_search_kb(self, sn, single_n, bn, k, bk, available, bias, sm, ablock, bblock):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'bk == 0 or sm == 0 or ablock == 0', bk == 0 or sm == 0 or ablock == 0, locals()):
            return None
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'sn * bn > single_n', sn * bn > single_n, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
        bunit = sn * 2 * bblock
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'bunit == 0', bunit == 0, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
        max_kb = min((available - bias) // bunit if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'available > bias', available > bias, locals()) else U64(0), k // bk)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'max_kb < 1', max_kb < 1, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
        best_ka = U64(0)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
        best_kb = U64(0)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
        best_bsize = U64(0)
        for kb in self.iterations('ScAl1.step_kb', range(1, int(max_kb) + 1)):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
            kb = U64(kb)
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
            bsize = sn * kb * 2 * bblock
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'bsize + bias > available', bsize + bias > available, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
            aunit = sm * ablock
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
            max_ka = min((available - bsize - bias) // aunit, k // bk)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'max_ka < kb', max_ka < kb, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
            ka = max_ka // kb * kb
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'ka < kb', ka < kb, locals()):
                continue
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'self.sc_better_l1(k, bk, ka, kb, bsize, best_ka, best_kb, best_bsize)', self.sc_better_l1(k, bk, ka, kb, bsize, best_ka, best_kb, best_bsize), locals()):
                self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
                best_ka = ka
                self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
                best_kb = kb
                self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215)
                best_bsize = bsize
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 215, 'best_ka < 1', best_ka < 1, locals()):
            return None
        return (best_ka, best_kb, sn, sm * best_ka, sn * best_kb * 2)

    @sourced('SelectL1Steps', 267, 296, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_select_l1(self, bm, sm, k, available, bias, single_n, bn, bk):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267, 'bn == 0 or bk == 0', bn == 0 or bk == 0, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267)
        ablock = bm * bk * self.aw
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267)
        bblock = bn * bk * self.bw
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267, 'ablock == 0 or bblock == 0', ablock == 0 or bblock == 0, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267)
        maximum = max(U64(1), cd(single_n, bn))
        for sn in self.iterations('ScAl1.step_n', range(1, int(maximum) + 1)):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267)
            sn = U64(sn)
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267)
            selected = self.sc_search_kb(sn, single_n, bn, k, bk, available, bias, sm, ablock, bblock)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 267, 'selected is None', selected is None, locals()):
                continue
            return selected
        return None

    @sourced('IsBetterBaseN', 298, 349, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_better_base(self, bm, sm, k, total, cur, best):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'best.baseN == 0', best.baseN == 0, locals()):
            return True
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'not self.args.isBTrans', not self.args.isBTrans, locals()):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
            cur_perfect = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.usedCoreNum > 0 and cur.nTileCnt % cur.usedCoreNum == 0', cur.usedCoreNum > 0 and cur.nTileCnt % cur.usedCoreNum == 0, locals())
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
            best_perfect = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'best.usedCoreNum > 0 and best.nTileCnt % best.usedCoreNum == 0', best.usedCoreNum > 0 and best.nTileCnt % best.usedCoreNum == 0, locals())
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur_perfect != best_perfect', cur_perfect != best_perfect, locals()):
                return cur_perfect
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.usedCoreNum != best.usedCoreNum', cur.usedCoreNum != best.usedCoreNum, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.usedCoreNum > best.usedCoreNum', cur.usedCoreNum > best.usedCoreNum, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'not self.args.isBTrans', not self.args.isBTrans, locals()):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
            cur_perfect = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.usedCoreNum > 0 and cur.nTileCnt % cur.usedCoreNum == 0', cur.usedCoreNum > 0 and cur.nTileCnt % cur.usedCoreNum == 0, locals())
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'not cur_perfect and cur.headCoreNum - cur.tailCoreNum != best.headCoreNum - best.tailCoreNum', not cur_perfect and cur.headCoreNum - cur.tailCoreNum != best.headCoreNum - best.tailCoreNum, locals()):
                return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.headCoreNum - cur.tailCoreNum < best.headCoreNum - best.tailCoreNum', cur.headCoreNum - cur.tailCoreNum < best.headCoreNum - best.tailCoreNum, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'self.args.isBTrans and cur.bkAlignScore != best.bkAlignScore', self.args.isBTrans and cur.bkAlignScore != best.bkAlignScore, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.bkAlignScore > best.bkAlignScore', cur.bkAlignScore > best.bkAlignScore, locals())
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
        cmp = self.sc_compare_rounds(cur, best, k)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cmp != 0', cmp != 0, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cmp > 0', cmp > 0, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseN != best.baseN', cur.baseN != best.baseN, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseN > best.baseN', cur.baseN > best.baseN, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseK != best.baseK', cur.baseK != best.baseK, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseK > best.baseK', cur.baseK > best.baseK, locals())
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
        loops = self.sc_est_min_a_loop(total, bm, sm, cur.baseN, cur.baseK, k)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298)
        best_loops = self.sc_est_min_a_loop(total, bm, sm, best.baseN, best.baseK, k)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'loops != best_loops', loops != best_loops, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'loops < best_loops', loops < best_loops, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseK == 0 or best.baseK == 0', cur.baseK == 0 or best.baseK == 0, locals()):
            return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'cur.baseK > best.baseK', cur.baseK > best.baseK, locals())
        return self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 298, 'bm * cur.baseN * best.baseK * (bm + best.baseN) > bm * best.baseN * cur.baseK * (bm + cur.baseN)', bm * cur.baseN * best.baseK * (bm + best.baseN) > bm * best.baseN * cur.baseK * (bm + cur.baseN), locals())

    @sourced('CalcBkAlignScoreForBaseK', 362, 385, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_score_base_k(self, bm, sm, k, total, single_n, bn, bk):
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362, 'not self.args.isBTrans', not self.args.isBTrans, locals()):
            return U64(0)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362)
        bias = bn * 4 if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362, 'self.args.hasBias', self.args.hasBias, locals()) else U64(0)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362, 'total <= bias', total <= bias, locals()):
            return None
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362)
        selected = self.sc_select_l1(bm, sm, k, total - bias, bias, single_n, bn, bk)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 362, 'selected is None', selected is None, locals()):
            return None
        return self.sc_b_align(selected[1] * bk * self.bw)

    @sourced('SearchBaseBlockCandidates', 434, 458, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_search_base(self, bm, sm, k, n, total, db, max_n):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
        best = Record(self, 'scAl1.best', baseN=0, baseK=0, bkAlignScore=0, singleCoreN=0, nTileCnt=0, usedCoreNum=0, headCoreNum=0, tailCoreNum=0)
        for bn in self.iterations('ScAl1.base_n', SC_AL1_BASE_N):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
            bn = U64(bn)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434, 'bn > max_n', bn > max_n, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
            nt, used, head, tail = self.sc_n_split(n, bn, self.hw.aicNum)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434, 'used == 0 or (tail > 0 and head < tail)', used == 0 or (tail > 0 and head < tail), locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
            policy = self.sc_policy_bk(bm, bn, db)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434, 'policy < 32', policy < 32, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
            single_n = bn * cd(nt, used)
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 434)
            best = self.sc_search_base_k(bm, sm, k, total, bn, policy, single_n, nt, used, head, tail, best)
        return best

    @sourced('SearchBaseKForBaseN', 408, 432, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_search_base_k(self, bm, sm, k, total, bn, policy, single_n, nt, used, head, tail, best):
        for bk in self.iterations('ScAl1.base_k', range(int(policy), 31, -32)):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408)
            bk = U64(bk)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408, 'not self.sc_l0_cube_fit(bm, bn, bk)', not self.sc_l0_cube_fit(bm, bn, bk), locals()):
                if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408, 'bk == 32', bk == 32, locals()):
                    break
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408)
            score = self.sc_score_base_k(bm, sm, k, total, single_n, bn, bk)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408, 'score is None', score is None, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408)
            cur = Record(self, 'scAl1.candidate', baseN=bn, baseK=bk, bkAlignScore=score, singleCoreN=single_n, nTileCnt=nt, usedCoreNum=used, headCoreNum=head, tailCoreNum=tail)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408, 'self.sc_better_base(bm, sm, k, total, cur, best)', self.sc_better_base(bm, sm, k, total, cur, best), locals()):
                self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408)
                best = cur.clone('scAl1.best')
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 408, 'bk == 32', bk == 32, locals()):
                break
        return best

    @sourced('TryBaseBlockFallback', 460, 477, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_fallback_base(self, bm, db, max_n):
        for bn in self.iterations('ScAl1.fallback_n', SC_AL1_BASE_N):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 460)
            bn = U64(bn)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 460, 'bn > max_n', bn > max_n, locals()):
                continue
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 460)
            policy = self.sc_policy_bk(bm, bn, db)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 460, 'policy >= 32', policy >= 32, locals()):
                return (bn, policy)
        return None

    @sourced('SelectBaseBlock', 479, 504, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_select_base(self, bm, sm, k, n, total, db):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 479)
        max_n = fd(self.hw.l0CSize, bm * db * 4)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 479, 'self.args.isBTrans and k > n', self.args.isBTrans and k > n, locals()):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 479)
            max_n = min(max_n, U64(128))
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 479)
        best = self.sc_search_base(bm, sm, k, n, total, db, max_n)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 479, 'best.baseN != 0', best.baseN != 0, locals()):
            return (best.baseN, best.baseK)
        return self.sc_fallback_base(bm, db, max_n)

    @sourced('AdjustL1Overflow', 506, 536, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_adjust_l1(self, total, bm, sm, bias, r):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
        a_used = r.baseM * r.baseK * r.depthA1 * self.aw
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
        b_used = r.baseN * r.baseK * r.depthB1 * self.bw
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
        used = a_used + b_used + bias
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'used <= total', used <= total, locals()):
            return True
        for i in self.iterations('ScAl1.adjust_a', range(3)):
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'not (r.stepKa > r.stepKb and used > total)', not (r.stepKa > r.stepKb and used > total), locals()):
                break
            r.stepKa -= r.stepKb
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            r.stepKa = r.stepKa // r.stepKb * r.stepKb if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'r.stepKb > 0', r.stepKb > 0, locals()) else 0
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'r.stepKa < r.stepKb', r.stepKa < r.stepKb, locals()):
                break
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            r.depthA1 = sm * r.stepKa
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            a_used = bm * r.baseK * r.depthA1 * self.aw
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            used = a_used + b_used + bias
        for i in self.iterations('ScAl1.adjust_b', range(3)):
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'not (r.stepN > 1 and used > total)', not (r.stepN > 1 and used > total), locals()):
                break
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            r.stepN = r.stepN // 2
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            r.depthB1 = r.stepN * r.stepKb * 2
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            b_used = r.baseN * r.baseK * r.depthB1 * self.bw
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506)
            used = a_used + b_used + bias
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 506, 'used > total or r.stepKa < r.stepKb or r.stepKa % r.stepKb != 0', used > total or r.stepKa < r.stepKb or r.stepKa % r.stepKb != 0, locals()):
            return False
        return True

    @sourced('MatmulV3ScSplitKAl1FullLoadTiling::DoTiling', 540, 589, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_al1_geometry(self, r):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.baseM = up(a.mValue, 16)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.dbL0c = 1
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        sm = cd(a.mValue, r.baseM)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        total = h.l1Size + 256
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        selected = self.sc_select_base(r.baseM, sm, a.kValue, a.nValue, total, r.dbL0c)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'selected is None', selected is None, locals()):
            return False
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.baseN, r.baseK = selected
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'r.baseK == 0', r.baseK == 0, locals()):
            self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
            r.baseK = self.sc_policy_bk(r.baseM, r.baseN, r.dbL0c)
            if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'r.baseK < 32', r.baseK < 32, locals()):
                return False
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        bias = r.baseN * 4 if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'a.hasBias', a.hasBias, locals()) else U64(0)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        nt = cd(a.nValue, r.baseN)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.usedCoreNum = min(nt, h.aicNum)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        single_n = r.baseN * cd(nt, r.usedCoreNum)
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        selected = self.sc_select_l1(r.baseM, sm, a.kValue, total - bias, bias, single_n, r.baseN, r.baseK)
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'selected is None', selected is None, locals()):
            return False
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.stepKa, r.stepKb, r.stepN, r.depthA1, r.depthB1 = selected
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.stepM = sm
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.singleCoreM = a.mValue
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.singleCoreN = single_n
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.singleCoreK = a.kValue
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540)
        r.iterateOrder = 0
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 540, 'not self.sc_adjust_l1(total, r.baseM, sm, bias, r)', not self.sc_adjust_l1(total, r.baseM, sm, bias, r), locals()):
            return False
        return True

    @sourced('MatmulV3ScSplitKAl1FullLoadTiling::CheckTilingOk', 591, 629, 'matmul_v3_sc_splitk_al1_fullload_tiling.cpp')
    def sc_check(self, r):
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591)
        dims = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'r.usedCoreNum >= 1 and r.singleCoreK >= r.baseK and (r.singleCoreN >= 1) and (r.depthA1 >= 1) and (r.depthB1 >= 1)', r.usedCoreNum >= 1 and r.singleCoreK >= r.baseK and (r.singleCoreN >= 1) and (r.depthA1 >= 1) and (r.depthB1 >= 1), locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'not dims', not dims, locals()):
            return False
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591)
        step_ok = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'r.stepN * r.baseN <= r.singleCoreN and r.stepKa * r.baseK <= r.singleCoreK and (r.stepKa >= r.stepKb) and (r.stepKa % r.stepKb == 0) and (r.depthB1 == r.stepN * r.stepKb * 2)', r.stepN * r.baseN <= r.singleCoreN and r.stepKa * r.baseK <= r.singleCoreK and (r.stepKa >= r.stepKb) and (r.stepKa % r.stepKb == 0) and (r.depthB1 == r.stepN * r.stepKb * 2), locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'not step_ok', not step_ok, locals()):
            return False
        self.site('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591)
        aligned = self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'r.baseK >= 32 and r.baseK % 32 == 0', r.baseK >= 32 and r.baseK % 32 == 0, locals())
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'not aligned', not aligned, locals()):
            return False
        if self.predicate('matmul_v3_sc_splitk_al1_fullload_tiling.cpp', 591, 'not self.sc_l0_cube_fit(r.baseM, r.baseN, r.baseK) or not self.sc_l0c_fit(r.baseM, r.baseN, r.dbL0c)', not self.sc_l0_cube_fit(r.baseM, r.baseN, r.baseK) or not self.sc_l0c_fit(r.baseM, r.baseN, r.dbL0c), locals()):
            return False
        return True

    @sourced('DoSingleCoreSplitKAL1FullLoadTiling', 2996, 3021)
    def sc_al1_full(self):
        if self.predicate('matmul_v3_base_tiling.cpp', 2996, 'not self.support_sc_al1()', not self.support_sc_al1(), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2996)
        temp = self.run.clone('tmpRunInfo')
        if self.predicate('matmul_v3_base_tiling.cpp', 2996, 'not self.sc_al1_geometry(temp)', not self.sc_al1_geometry(temp), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2996, 'not self.sc_check(temp)', not self.sc_check(temp), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2996)
        self.run = temp.clone('runInfo')
        self.site('matmul_v3_base_tiling.cpp', 2996)
        self.run.needUpdate = True
        self.site('matmul_v3_base_tiling.cpp', 2996)
        self.enable.tilingEnableFullLoad = 1
        self.site('matmul_v3_base_tiling.cpp', 2996)
        self.enable.tilingEnableSplitCore = 2
        return True

    @sourced('CalTileFactor', 2154, 2183)
    def core_factor(self, tile):
        self.site('matmul_v3_base_tiling.cpp', 2154)
        h = self.hw
        self.site('matmul_v3_base_tiling.cpp', 2154)
        factors = CORE_FACTORS_20
        if self.predicate('matmul_v3_base_tiling.cpp', 2154, 'h.aicNum == 24', h.aicNum == 24, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2154)
            factors = CORE_FACTORS_24
        if self.predicate('matmul_v3_base_tiling.cpp', 2154, 'h.aicNum == 32', h.aicNum == 32, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2154)
            factors = CORE_FACTORS_32
        for f in self.iterations('CalTileFactor.factors', factors):
            if self.predicate('matmul_v3_base_tiling.cpp', 2154, 'tile <= f', tile <= f, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2154)
                tile = U64(f)
                break
        if self.predicate('matmul_v3_base_tiling.cpp', 2154, 'tile > h.aicNum', tile > h.aicNum, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2154)
            tail = tile % h.aicNum
            if self.predicate('matmul_v3_base_tiling.cpp', 2154, 'tail > h.aicNum // 2', tail > h.aicNum // 2, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2154)
                tile = (tile + h.aicNum - 1) // h.aicNum
            else:
                self.site('matmul_v3_base_tiling.cpp', 2154)
                tile = tile // h.aicNum
            self.site('matmul_v3_base_tiling.cpp', 2154)
            tile = tile * h.aicNum
        return tile

    @sourced('SetBasicBlockOfMK33', 2203, 2219)
    def mk33(self, r):
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.baseM = 128
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.baseN = 128
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.baseK = U64(256) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.usedCoreNum = self.hw.aicNum
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.depthA1 = 9
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.depthB1 = 6
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.stepM = 3
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.stepN = 1
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.stepKa = r.depthA1 // r.stepM
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.stepKb = r.depthB1 // r.stepN // 2
        self.site('matmul_v3_base_tiling.cpp', 2203)
        r.iterateOrder = 1

    @sourced('SetBasicBlockOfNK33', 2185, 2201)
    def nk33(self, r):
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.baseM = 128
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.baseN = 128
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.baseK = U64(256) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.usedCoreNum = self.hw.aicNum
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.depthA1 = 6
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.depthB1 = 9
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.stepM = 1
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.stepN = 3
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.stepKa = r.depthA1 // r.stepM // 2
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.stepKb = r.depthB1 // r.stepN
        self.site('matmul_v3_base_tiling.cpp', 2185)
        r.iterateOrder = 0

    @sourced('SetBasicBlockOf24', 2221, 2241)
    def block24(self, r, mt, nt):
        if self.predicate('matmul_v3_base_tiling.cpp', 2221, 'mt * nt < self.hw.aicNum', mt * nt < self.hw.aicNum, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.depthA1 = 8
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.depthB1 = 8
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepM = 2
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepN = 1
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepKa = r.depthA1 // r.stepM
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepKb = r.depthB1 // 2
        if self.predicate('matmul_v3_base_tiling.cpp', 2221, 'self.args.mValue < 384 or self.args.nValue < 384', self.args.mValue < 384 or self.args.nValue < 384, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.depthA1 = 8
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.depthB1 = 8
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepM = 1
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepKa = 4
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepN = 1
            self.site('matmul_v3_base_tiling.cpp', 2221)
            r.stepKb = 4

    @sourced('IsSupportSingleCoreSplitSmallK', 2300, 2318)
    def support_small_k(self, x, y):
        self.site('matmul_v3_base_tiling.cpp', 2300)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_base_tiling.cpp', 2300)
        dtype_format = self.predicate('matmul_v3_base_tiling.cpp', 2300, "a.aFormat == 'ND' and a.aType in ('fp16', 'bf16', 'fp32') and (a.bType in ('fp16', 'bf16', 'fp32'))", a.aFormat == 'ND' and a.aType in ('fp16', 'bf16', 'fp32') and (a.bType in ('fp16', 'bf16', 'fp32')), locals())
        self.site('matmul_v3_base_tiling.cpp', 2300)
        small_k = self.predicate('matmul_v3_base_tiling.cpp', 2300, 'a.kValue == 1536 and a.mValue % 128 == 0 and (a.nValue % 128 == 0)', a.kValue == 1536 and a.mValue % 128 == 0 and (a.nValue % 128 == 0), locals())
        self.site('matmul_v3_base_tiling.cpp', 2300)
        large_x = self.predicate('matmul_v3_base_tiling.cpp', 2300, 'x >= 49152 and y == 384', x >= 49152 and y == 384, locals())
        self.site('matmul_v3_base_tiling.cpp', 2300)
        cache_limit = U64(float(h.l2Size) * 0.65) // 4
        self.site('matmul_v3_base_tiling.cpp', 2300)
        enough = self.predicate('matmul_v3_base_tiling.cpp', 2300, 'a.nValue * a.mValue <= cache_limit', a.nValue * a.mValue <= cache_limit, locals())
        return self.predicate('matmul_v3_base_tiling.cpp', 2300, 'dtype_format and small_k and large_x and enough', dtype_format and small_k and large_x and enough, locals())

    @sourced('IsSupportSingleCoreSplitK', 2320, 2373)
    def support_single(self):
        self.site('matmul_v3_base_tiling.cpp', 2320)
        a, h = (self.args, self.hw)
        self.site('matmul_v3_base_tiling.cpp', 2320)
        small_mkn = self.support_small_k(a.nValue, a.mValue)
        self.site('matmul_v3_base_tiling.cpp', 2320)
        small_nkm = self.support_small_k(a.mValue, a.nValue)
        if self.predicate('matmul_v3_base_tiling.cpp', 2320, 'small_mkn or small_nkm', small_mkn or small_nkm, locals()):
            return True
        if self.predicate('matmul_v3_base_tiling.cpp', 2330, 'a.isHf32 and (not self.n_align) and (a.mValue * a.nValue < 4 * MB_SIZE)', a.isHf32 and (not self.n_align) and (a.mValue * a.nValue < 4 * MB_SIZE), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2330, 'a.kValue >= SPLIT_K_THRES', a.kValue >= SPLIT_K_THRES, locals()):
            return True
        self.site('matmul_v3_base_tiling.cpp', 2330)
        large = self.predicate('matmul_v3_base_tiling.cpp', 2330, 'a.mValue * a.kValue >= 5 * 384 * 384 and a.nValue >= 1024 and (a.mValue >= 384) and (a.kValue >= 384) and (a.mValue * a.nValue >= 1024 * 384 * h.aicNum)', a.mValue * a.kValue >= 5 * 384 * 384 and a.nValue >= 1024 and (a.mValue >= 384) and (a.kValue >= 384) and (a.mValue * a.nValue >= 1024 * 384 * h.aicNum), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2330, 'not self.enable_cache and large', not self.enable_cache and large, locals()):
            return True
        self.site('matmul_v3_base_tiling.cpp', 2350)
        tlb_mata = self.predicate('matmul_v3_base_tiling.cpp', 2350, 'a.isATrans and (not a.isBTrans) and self.n_align and (a.kValue >= 11000) and (a.mValue % 8192 == 0 and a.nValue >= 6144 or (a.nValue % 8192 == 0 and a.mValue >= 6144))', a.isATrans and (not a.isBTrans) and self.n_align and (a.kValue >= 11000) and (a.mValue % 8192 == 0 and a.nValue >= 6144 or (a.nValue % 8192 == 0 and a.mValue >= 6144)), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2350, 'tlb_mata', tlb_mata, locals()):
            return True
        self.site('matmul_v3_base_tiling.cpp', 2350)
        mata = self.predicate('matmul_v3_base_tiling.cpp', 2350, "not a.isATrans and a.isBTrans and (a.kValue % 16384 == 0) and (1280 <= a.mValue <= 8192) and (1280 <= a.nValue <= 8192) and (a.aFormat == 'ND') and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16')) and (h.aicNum == 24)", not a.isATrans and a.isBTrans and (a.kValue % 16384 == 0) and (1280 <= a.mValue <= 8192) and (1280 <= a.nValue <= 8192) and (a.aFormat == 'ND') and (a.bFormat == 'ND') and (a.aType in ('fp16', 'bf16')) and (h.aicNum == 24), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2350, 'mata and (not a.nd2nzB)', mata and (not a.nd2nzB), locals()):
            return True
        return False

    @sourced('IsGmToL1ByShape', 2375, 2399)
    def gm_to_l1(self):
        self.site('matmul_v3_base_tiling.cpp', 2375)
        a, r = (self.args, self.run)
        self.site('matmul_v3_base_tiling.cpp', 2375)
        no_mix = self.predicate('matmul_v3_base_tiling.cpp', 2375, 'self.mix_type() == 1', self.mix_type() == 1, locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'no_mix', no_mix, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2375)
            axis_a = a.mValue if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'a.isATrans', a.isATrans, locals()) else a.kValue
            self.site('matmul_v3_base_tiling.cpp', 2375)
            axis_b = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'a.isBTrans', a.isBTrans, locals()) else a.nValue
        else:
            self.site('matmul_v3_base_tiling.cpp', 2375)
            axis_a = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'a.isATrans', a.isATrans, locals()) else a.mValue
            self.site('matmul_v3_base_tiling.cpp', 2375)
            axis_b = a.nValue if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'a.isBTrans', a.isBTrans, locals()) else a.kValue
        if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'not a.hasBias and axis_a <= 65535 and (axis_b <= 65535)', not a.hasBias and axis_a <= 65535 and (axis_b <= 65535), locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2375, "no_mix and a.bFormat == 'NZ'", no_mix and a.bFormat == 'NZ', locals()):
                return
            self.site('matmul_v3_base_tiling.cpp', 2375)
            self.cube_extra['shareL1Size'] = 0
            if self.predicate('matmul_v3_base_tiling.cpp', 2375, 'r.stepM == 3 and a.hasBias', r.stepM == 3 and a.hasBias, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2375)
                self.cube_extra['shareL1Size'] = 1024
            self.site('matmul_v3_base_tiling.cpp', 2375)
            self.enable.tilingEnableSplitCore = 6
            self.site('matmul_v3_base_tiling.cpp', 2375)
            self.cube_extra['shareL0CSize'] = 0

    @sourced('CheckSingleCoreSplitKEdgeCases', 2438, 2463)
    def single_edge(self, r, nkm, small_mkn, small_nkm):
        self.site('matmul_v3_base_tiling.cpp', 2438)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2438, 'not nkm and (not small_mkn) and (not small_nkm)', not nkm and (not small_mkn) and (not small_nkm), locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2438, 'r.singleCoreN < 512', r.singleCoreN < 512, locals()):
                return True
            self.site('matmul_v3_base_tiling.cpp', 2438)
            average = ratio_f32(a.mValue * a.nValue, r.singleCoreN * r.singleCoreM * h.aicNum)
            if self.predicate('matmul_v3_base_tiling.cpp', 2438, 'average < f32(0.7)', average < f32(0.7), locals()):
                return True
            if self.predicate('matmul_v3_base_tiling.cpp', 2438, 'self.n_align and 896 <= a.nValue <= 2048 and (r.singleCoreN <= 640 or average < f32(0.85))', self.n_align and 896 <= a.nValue <= 2048 and (r.singleCoreN <= 640 or average < f32(0.85)), locals()):
                return True
        return False

    @sourced('CheckSingleTilingOk', 2401, 2436)
    def check_single(self, r):
        self.site('matmul_v3_base_tiling.cpp', 2401)
        a, e = (self.args, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 2401)
        nkm = self.predicate('matmul_v3_base_tiling.cpp', 2401, "a.aType == 'fp32' and a.nValue <= 64 and (a.mValue >= 1920) and (27392 <= a.kValue < 65535) and (not a.isATrans) and a.isBTrans", a.aType == 'fp32' and a.nValue <= 64 and (a.mValue >= 1920) and (27392 <= a.kValue < 65535) and (not a.isATrans) and a.isBTrans, locals())
        self.site('matmul_v3_base_tiling.cpp', 2401)
        small_mkn = self.support_small_k(a.nValue, a.mValue)
        self.site('matmul_v3_base_tiling.cpp', 2401)
        small_nkm = self.support_small_k(a.mValue, a.nValue)
        if self.predicate('matmul_v3_base_tiling.cpp', 2401, 'self.single_edge(r, nkm, small_mkn, small_nkm)', self.single_edge(r, nkm, small_mkn, small_nkm), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2414, "small_mkn or (256 < r.singleCoreM <= 384 and (not (a.aType == 'fp32' and (not self.n_align))))", small_mkn or (256 < r.singleCoreM <= 384 and (not (a.aType == 'fp32' and (not self.n_align)))), locals()):
            self.mk33(r)
        elif self.predicate('matmul_v3_base_tiling.cpp', 2414, 'small_nkm', small_nkm, locals()):
            self.nk33(r)
        self.site('matmul_v3_base_tiling.cpp', 2414)
        r.singleCoreK = r.stepKa * r.baseK
        self.site('matmul_v3_base_tiling.cpp', 2414)
        r.dbL0c = 2
        self.site('matmul_v3_base_tiling.cpp', 2414)
        e.tilingEnableSplitCore = 2
        if self.predicate('matmul_v3_base_tiling.cpp', 2423, '(nkm or small_nkm) and (not a.nd2nzA) and (not a.nd2nzB)', (nkm or small_nkm) and (not a.nd2nzA) and (not a.nd2nzB), locals()):
            self.site('matmul_v3_base_tiling.cpp', 2423)
            e.tilingEnableSplitCore = 5
        elif self.predicate('matmul_v3_base_tiling.cpp', 2423, "a.aType != 'fp32' and a.nValue % 128 == 0 and (not small_mkn)", a.aType != 'fp32' and a.nValue % 128 == 0 and (not small_mkn), locals()):
            self.gm_to_l1()
        self.site('matmul_v3_base_tiling.cpp', 2423)
        e.tilingEnableFullLoad = 0
        self.site('matmul_v3_base_tiling.cpp', 2423)
        e.tilingEnableFixOpti = 0
        self.site('matmul_v3_base_tiling.cpp', 2423)
        self.run = r.clone('runInfo')
        self.site('matmul_v3_base_tiling.cpp', 2423)
        self.run.needUpdate = True
        self.site('matmul_v3_base_tiling.cpp', 2423)
        self.run.l2Info.calOrder = 1 if self.predicate('matmul_v3_base_tiling.cpp', 2423, 'small_nkm', small_nkm, locals()) else 0
        return True

    @sourced('DoSingleCoreSplitKTiling', 2465, 2525)
    def single_core_split(self):
        self.site('matmul_v3_base_tiling.cpp', 2465)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2465, 'not h.supportL0c2out or h.supportL12BtBf16 or (not self.support_single())', not h.supportL0c2out or h.supportL12BtBf16 or (not self.support_single()), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2465)
        r = self.run.clone('tmpRunInfo')
        self.mk33(r)
        self.site('matmul_v3_base_tiling.cpp', 2465)
        mt = cd(a.mValue, r.stepM * r.baseM)
        self.site('matmul_v3_base_tiling.cpp', 2465)
        nt = cd(a.nValue, self.l2_length)
        if self.predicate('matmul_v3_base_tiling.cpp', 2465, 'not a.hasBias', not a.hasBias, locals()):
            self.block24(r, mt, nt)
        self.site('matmul_v3_base_tiling.cpp', 2465)
        m_align = (U64(256) if self.predicate('matmul_v3_base_tiling.cpp', 2465, 'a.isATrans', a.isATrans, locals()) else U64(32)) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 2465)
        n_align = U64(256) // self.aw
        self.site('matmul_v3_base_tiling.cpp', 2465)
        mt = cd(a.mValue, r.stepM * r.baseM)
        self.site('matmul_v3_base_tiling.cpp', 2465)
        nt = self.core_factor(nt)
        if self.predicate('matmul_v3_base_tiling.cpp', 2483, 'mt * nt >= h.aicNum', mt * nt >= h.aicNum, locals()):
            if self.predicate('matmul_v3_base_tiling.cpp', 2483, 'nt == 0', nt == 0, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2483)
                mt = U64(1)
            else:
                self.site('matmul_v3_base_tiling.cpp', 2483)
                mt = max(h.aicNum // nt, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 2483)
            mt = max(h.aicNum // nt, U64(1))
            self.site('matmul_v3_base_tiling.cpp', 2483)
            r.singleCoreM = min(up(cd(a.mValue, mt), m_align), a.mValue)
            self.site('matmul_v3_base_tiling.cpp', 2483)
            r.singleCoreN = up(cd(a.nValue, nt), n_align)
            self.site('matmul_v3_base_tiling.cpp', 2483)
            r.usedCoreNum = min(cd(a.mValue, r.singleCoreM) * cd(a.nValue, r.singleCoreN), h.aicNum)
            if self.predicate('matmul_v3_base_tiling.cpp', 2483, 'r.usedCoreNum == h.aicNum', r.usedCoreNum == h.aicNum, locals()):
                return self.check_single(r)
        if self.predicate('matmul_v3_base_tiling.cpp', 2500, 'mt == 0', mt == 0, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2500)
            nt = U64(1)
        else:
            self.site('matmul_v3_base_tiling.cpp', 2500)
            nt = max(h.aicNum // mt, U64(1))
        self.site('matmul_v3_base_tiling.cpp', 2500)
        total = U64(0)
        self.site('matmul_v3_base_tiling.cpp', 2500)
        single_n = a.nValue
        while self.predicate('matmul_v3_base_tiling.cpp', 2500, 'nt <= h.aicNum and total < h.aicNum and (single_n >= 1024)', nt <= h.aicNum and total < h.aicNum and (single_n >= 1024), locals()):
            self.count_loop('SingleCoreSplit.core_factor_adjustments')
            self.site('matmul_v3_base_tiling.cpp', 2500)
            nt = self.core_factor(nt)
            self.site('matmul_v3_base_tiling.cpp', 2500)
            mt = h.aicNum // nt
            self.site('matmul_v3_base_tiling.cpp', 2500)
            single_n = up(cd(a.nValue, nt), n_align)
            self.site('matmul_v3_base_tiling.cpp', 2500)
            single_m = up(cd(a.mValue, mt), m_align)
            self.site('matmul_v3_base_tiling.cpp', 2500)
            mc = cd(a.mValue, single_m)
            self.site('matmul_v3_base_tiling.cpp', 2500)
            nc = cd(a.nValue, single_n)
            if self.predicate('matmul_v3_base_tiling.cpp', 2500, 'mc * nc > total', mc * nc > total, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2500)
                total = mc * nc
                self.site('matmul_v3_base_tiling.cpp', 2500)
                r.usedCoreNum = total
                self.site('matmul_v3_base_tiling.cpp', 2500)
                r.singleCoreM = min(single_m, a.mValue)
                self.site('matmul_v3_base_tiling.cpp', 2500)
                r.singleCoreN = min(single_n, a.nValue)
            nt += 1
        return self.check_single(r)

    @sourced('ShouldUseDeterministicMultiCoreSplitKwithSmallMN', 2648, 2655)
    def deterministic_small_mn(self):
        self.site('matmul_v3_base_tiling.cpp', 2648)
        a = self.args
        if self.predicate('matmul_v3_base_tiling.cpp', 2648, "a.aType != 'fp32' or a.isHf32", a.aType != 'fp32' or a.isHf32, locals()):
            return False
        return self.predicate('matmul_v3_base_tiling.cpp', 2648, 'a.mValue <= 64 and a.nValue <= 64 and (a.kValue >= 6144)', a.mValue <= 64 and a.nValue <= 64 and (a.kValue >= 6144), locals())

    @sourced('SupportMultiSplitK', 2535, 2583)
    def support_multi(self):
        self.site('matmul_v3_base_tiling.cpp', 2535)
        a, h = (self.args, self.hw)
        if self.predicate('matmul_v3_base_tiling.cpp', 2535, 'not h.supportL12BtBf16 and (not h.supportL0c2out)', not h.supportL12BtBf16 and (not h.supportL0c2out), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2535, 'self.force_group_fp32()', self.force_group_fp32(), locals()):
            return True
        self.site('matmul_v3_base_tiling.cpp', 2535)
        enough_k = self.predicate('matmul_v3_base_tiling.cpp', 2535, 'a.kValue >= h.aicNum * 384', a.kValue >= h.aicNum * 384, locals())
        self.site('matmul_v3_base_tiling.cpp', 2535)
        mc = cd(a.mValue, 128)
        self.site('matmul_v3_base_tiling.cpp', 2535)
        nc = cd(a.nValue, 128)
        self.site('matmul_v3_base_tiling.cpp', 2535)
        not_enough_mn = self.predicate('matmul_v3_base_tiling.cpp', 2535, 'mc * nc < h.aicNum // 2', mc * nc < h.aicNum // 2, locals())
        self.site('matmul_v3_base_tiling.cpp', 2535)
        split_scene = self.predicate('matmul_v3_base_tiling.cpp', 2535, 'enough_k and not_enough_mn and (not (not a.isATrans and a.isBTrans))', enough_k and not_enough_mn and (not (not a.isATrans and a.isBTrans)), locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2535, 'split_scene', split_scene, locals()):
            return True
        if self.predicate('matmul_v3_base_tiling.cpp', 2558, 'self.deterministic_small_mn()', self.deterministic_small_mn(), locals()):
            return True
        small_tatn_split = ((self.profile != 'installed_81' or a.aType == 'fp32') and
                            1000 <= a.kValue <= 4608 and a.mValue <= 256 and
                            a.nValue <= 256 and a.isATrans and not a.isBTrans)
        if self.predicate('matmul_v3_base_tiling.cpp', 2558, 'small_tatn_split', small_tatn_split, locals()):
            return True
        if self.predicate('matmul_v3_base_tiling.cpp', 2558, 'a.kValue < SPLIT_K_THRES', a.kValue < SPLIT_K_THRES, locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2574)
        inner_a = self.m_align if self.predicate('matmul_v3_base_tiling.cpp', 2574, 'a.isATrans', a.isATrans, locals()) else self.ka_align
        self.site('matmul_v3_base_tiling.cpp', 2574)
        inner_b = self.kb_align if self.predicate('matmul_v3_base_tiling.cpp', 2574, 'a.isBTrans', a.isBTrans, locals()) else self.n_align
        self.site('matmul_v3_base_tiling.cpp', 2574)
        mte2_float = self.predicate('matmul_v3_base_tiling.cpp', 2574, "(not inner_a or not inner_b) and a.aType == 'fp32'", (not inner_a or not inner_b) and a.aType == 'fp32', locals())
        if self.predicate('matmul_v3_base_tiling.cpp', 2574, 'mc >= 4 and nc >= 4 and (not self.n_align) and (not mte2_float) and (ratio_f32(mc * nc, up(mc * nc, h.aicNum)) > f32(0.8))', mc >= 4 and nc >= 4 and (not self.n_align) and (not mte2_float) and (ratio_f32(mc * nc, up(mc * nc, h.aicNum)) > f32(0.8)), locals()):
            return False
        return True

    @sourced('IsNkOrder', 2585, 2614)
    def nk_order(self):
        self.site('matmul_v3_base_tiling.cpp', 2585)
        a = self.args
        self.site('matmul_v3_base_tiling.cpp', 2585)
        self.aw = U64(4 if self.predicate('matmul_v3_base_tiling.cpp', 2585, "a.aType == 'fp32'", a.aType == 'fp32', locals()) else 2)
        self.site('matmul_v3_base_tiling.cpp', 2585)
        self.bw = U64(4 if self.predicate('matmul_v3_base_tiling.cpp', 2585, "a.bType == 'fp32'", a.bType == 'fp32', locals()) else 2)
        if self.predicate('matmul_v3_base_tiling.cpp', 2585, 'a.mValue <= a.nValue or a.nValue * self.bw % 32 != 0', a.mValue <= a.nValue or a.nValue * self.bw % 32 != 0, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2585, 'a.mValue < 128', a.mValue < 128, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2585, 'a.nValue * self.bw % 256 == 0 and a.mValue * self.aw % 256 != 0', a.nValue * self.bw % 256 == 0 and a.mValue * self.aw % 256 != 0, locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2585, 'a.mValue * self.aw % 256 != 0 and a.nValue * self.bw <= 256 and (a.nValue * self.bw % 32 == 0)', a.mValue * self.aw % 256 != 0 and a.nValue * self.bw <= 256 and (a.nValue * self.bw % 32 == 0), locals()):
            return False
        if self.predicate('matmul_v3_base_tiling.cpp', 2585, 'a.mValue >= 2048 and a.nValue == 16', a.mValue >= 2048 and a.nValue == 16, locals()):
            return False
        return True

    @sourced('GetMoreMultiCoreSplitKArgs', 2616, 2646)
    def more_multi_args(self):
        self.site('matmul_v3_base_tiling.cpp', 2616)
        a = self.args
        self.site('matmul_v3_base_tiling.cpp', 2616)
        ia = a.mValue if self.predicate('matmul_v3_base_tiling.cpp', 2616, 'a.isATrans', a.isATrans, locals()) else a.kValue
        self.site('matmul_v3_base_tiling.cpp', 2616)
        ib = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 2616, 'a.isBTrans', a.isBTrans, locals()) else a.nValue
        self.site('matmul_v3_base_tiling.cpp', 2616)
        oa = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 2616, 'a.isATrans', a.isATrans, locals()) else a.mValue
        self.site('matmul_v3_base_tiling.cpp', 2616)
        ob = a.nValue if self.predicate('matmul_v3_base_tiling.cpp', 2616, 'a.isBTrans', a.isBTrans, locals()) else a.kValue
        if self.predicate('matmul_v3_base_tiling.cpp', 2626, '384 // self.aw <= ia <= 65535', 384 // self.aw <= ia <= 65535, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2626)
            a.nd2nzA = False
        if self.predicate('matmul_v3_base_tiling.cpp', 2626, '384 // self.bw <= ib <= 65535', 384 // self.bw <= ib <= 65535, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2626)
            a.nd2nzB = False
        if self.predicate('matmul_v3_base_tiling.cpp', 2626, 'a.nValue > 128 and a.mValue > 128 and (a.nValue % 16 != 0)', a.nValue > 128 and a.mValue > 128 and (a.nValue % 16 != 0), locals()):
            self.site('matmul_v3_base_tiling.cpp', 2626)
            self.enable.tilingEnableFixOpti = 2
        self.site('matmul_v3_base_tiling.cpp', 2626)
        c0 = U64(32) // self.aw
        if self.predicate('matmul_v3_base_tiling.cpp', 2626, 'ia == c0 and oa % 16 == 0', ia == c0 and oa % 16 == 0, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2626)
            a.isNzA = True
        if self.predicate('matmul_v3_base_tiling.cpp', 2626, 'ib == c0 and ob % 16 == 0', ib == c0 and ob % 16 == 0, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2626)
            a.isNzB = True

    @sourced('OptCoreNumsDeterministicMultiCoreSplitK', 2657, 2667)
    def optimize_det_cores(self):
        self.site('matmul_v3_base_tiling.cpp', 2657)
        a = self.args
        if self.predicate('matmul_v3_base_tiling.cpp', 2657, 'self.deterministic_small_mn()', self.deterministic_small_mn(), locals()):
            self.site('matmul_v3_base_tiling.cpp', 2657)
            delta = fd(a.kValue - 6144, 1024) + cd(a.mValue, 16) + cd(a.nValue, 16)
            self.site('matmul_v3_base_tiling.cpp', 2657)
            self.run.usedCoreNum = min(self.hw.aicNum, 8 + delta)

    @sourced('DoDeterministicMultiCoreSplitKTiling', 2669, 2730)
    def deterministic_split(self):
        self.site('matmul_v3_base_tiling.cpp', 2669)
        a, h, r, e = (self.args, self.hw, self.run, self.enable)
        if self.predicate('matmul_v3_base_tiling.cpp', 2669, 'h.supportL12BtBf16 or not self.support_multi()', h.supportL12BtBf16 or not self.support_multi(), locals()):
            return False
        self.site('matmul_v3_base_tiling.cpp', 2669)
        e.tilingEnableSplitCore = 3
        self.site('matmul_v3_base_tiling.cpp', 2669)
        e.tilingEnableFullLoad = 0
        self.site('matmul_v3_base_tiling.cpp', 2669)
        e.tilingEnableFixOpti = 0
        self.site('matmul_v3_base_tiling.cpp', 2669)
        l2_70 = h.l2Size * 7 // 10
        if self.predicate('matmul_v3_base_tiling.cpp', 2682, 'self.nk_order()', self.nk_order(), locals()):
            self.nk33(r)
            self.site('matmul_v3_base_tiling.cpp', 2682)
            r.singleCoreM = a.mValue
            self.site('matmul_v3_base_tiling.cpp', 2682)
            r.singleCoreN = r.stepN * r.baseN
            self.site('matmul_v3_base_tiling.cpp', 2682)
            r.singleCoreK = r.stepKb * r.baseK
            self.site('matmul_v3_base_tiling.cpp', 2682)
            split = (l2_70 // h.aicNum - r.singleCoreK * min(r.singleCoreN, a.nValue) * self.bw) // (r.singleCoreK * self.aw + min(r.singleCoreN, a.nValue) * 4)
            if self.predicate('matmul_v3_base_tiling.cpp', 2682, 'a.mValue > split', a.mValue > split, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2682)
                split = up(split, 128)
                self.site('matmul_v3_base_tiling.cpp', 2682)
                count = cd(a.mValue, split)
                self.site('matmul_v3_base_tiling.cpp', 2682)
                r.singleCoreM = up(cd(a.mValue, count), 128)
        else:
            self.mk33(r)
            self.site('matmul_v3_base_tiling.cpp', 2700)
            r.singleCoreN = a.nValue
            self.site('matmul_v3_base_tiling.cpp', 2700)
            r.singleCoreM = r.stepM * r.baseM
            self.site('matmul_v3_base_tiling.cpp', 2700)
            r.singleCoreK = r.stepKa * r.baseK
            self.site('matmul_v3_base_tiling.cpp', 2700)
            split = (l2_70 // h.aicNum - r.singleCoreK * min(r.singleCoreM, a.mValue) * self.aw) // (r.singleCoreK * self.bw + min(r.singleCoreM, a.mValue) * 4)
            if self.predicate('matmul_v3_base_tiling.cpp', 2700, 'a.nValue > split', a.nValue > split, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2700)
                split = up(split, 128)
                self.site('matmul_v3_base_tiling.cpp', 2700)
                count = cd(a.nValue, split)
                self.site('matmul_v3_base_tiling.cpp', 2700)
                r.singleCoreN = up(cd(a.nValue, count), 128)
        self.site('matmul_v3_base_tiling.cpp', 2721)
        count = cd(a.kValue, r.singleCoreK)
        self.site('matmul_v3_base_tiling.cpp', 2721)
        r.usedCoreNum = min(count, h.aicNum)
        self.site('matmul_v3_base_tiling.cpp', 2721)
        r.dbL0c = 2
        self.more_multi_args()
        self.optimize_det_cores()
        return True

    @sourced('CheckUbOverFlow', 1262, 1272)
    def ub_overflow(self, n_aligned, n, bn, bd, width):
        self.site('matmul_v3_base_tiling.cpp', 1262)
        aligned_loops = cd(n_aligned, bn)
        self.site('matmul_v3_base_tiling.cpp', 1262)
        loops = cd(n, bn)
        if self.predicate('matmul_v3_base_tiling.cpp', 1262, 'bn == 0 or width == 0', bn == 0 or width == 0, locals()):
            return False
        return self.predicate('matmul_v3_base_tiling.cpp', 1262, 'aligned_loops != loops and (n_aligned - (n // bn - 1) * bn) * bd > self.hw.ubSize // 2 // width', aligned_loops != loops and (n_aligned - (n // bn - 1) * bn) * bd > self.hw.ubSize // 2 // width, locals())

    @sourced('CalcNd2NzTiling', 1274, 1360)
    def nd2nz_geometry(self, width, n, d):
        self.site('matmul_v3_base_tiling.cpp', 1274)
        a, h, r = (self.args, self.hw, self.run)
        if self.predicate('matmul_v3_base_tiling.cpp', 1281, "d % 16384 == 0 and n >= 7168 and (h.ubSize == 196352) and (a.aType in ('fp16', 'bf16')) and (h.aicNum >= 24)", d % 16384 == 0 and n >= 7168 and (h.ubSize == 196352) and (a.aType in ('fp16', 'bf16')) and (h.aicNum >= 24), locals()):
            return (U64(96), U64(512))
        if self.predicate('matmul_v3_base_tiling.cpp', 1281, 'width == 0', width == 0, locals()):
            return (U64(0), U64(0))
        self.site('matmul_v3_base_tiling.cpp', 1281)
        cores = max(2 * r.usedCoreNum, U64(1))
        self.site('matmul_v3_base_tiling.cpp', 1281)
        threshold = U64(2048) // width
        self.site('matmul_v3_base_tiling.cpp', 1281)
        c0 = U64(32) // width
        self.site('matmul_v3_base_tiling.cpp', 1281)
        na = up(n, 16)
        self.site('matmul_v3_base_tiling.cpp', 1281)
        da = up(d, c0)
        if self.predicate('matmul_v3_base_tiling.cpp', 1297, 'd <= threshold', d <= threshold, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1297)
            bd = max(up(d, c0), U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1297)
            bn = h.ubSize // 2 // width // bd
            self.site('matmul_v3_base_tiling.cpp', 1297)
            rounds = max(cd(cd(na, cores), bn), U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1297)
            bn = max(cd(cd(na, cores), rounds), U64(16))
            while self.predicate('matmul_v3_base_tiling.cpp', 1297, 'bn > 16', bn > 16, locals()):
                self.count_loop('Nd2Nz.small_d_shrink')
                if self.predicate('matmul_v3_base_tiling.cpp', 1297, 'self.ub_overflow(na, n, bn, bd, width)', self.ub_overflow(na, n, bn, bd, width), locals()):
                    bn -= 1
                    continue
                break
            return (bn, bd)
        self.site('matmul_v3_base_tiling.cpp', 1311)
        last_tail = U64(0)
        self.site('matmul_v3_base_tiling.cpp', 1311)
        best_n = U64(16)
        self.site('matmul_v3_base_tiling.cpp', 1311)
        best_d = U64(CALC_ND_BASIC[1]) // width
        for base in self.iterations('Nd2Nz.byte_templates', CALC_ND_BASIC):
            self.site('matmul_v3_base_tiling.cpp', 1311)
            bd = max(min(da, U64(base) // width), U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1311)
            dloops = cd(da, bd)
            self.site('matmul_v3_base_tiling.cpp', 1311)
            dtail = da % bd
            if self.predicate('matmul_v3_base_tiling.cpp', 1311, 'dtail > 0 and dtail < 512 // width', dtail > 0 and dtail < 512 // width, locals()):
                if self.predicate('matmul_v3_base_tiling.cpp', 1311, 'bd * width == CALC_ND_BASIC[0]', bd * width == CALC_ND_BASIC[0], locals()):
                    continue
                dloops -= 1
                self.site('matmul_v3_base_tiling.cpp', 1311)
                bd = max(up(cd(da, dloops), c0), U64(1))
            self.site('matmul_v3_base_tiling.cpp', 1311)
            bn = max(h.ubSize // 2 // width // bd, U64(16))
            if self.predicate('matmul_v3_base_tiling.cpp', 1311, 'bn * bd * width * 2 > h.ubSize', bn * bd * width * 2 > h.ubSize, locals()):
                continue
            if self.predicate('matmul_v3_base_tiling.cpp', 1311, 'self.ub_overflow(na, n, bn, bd, width)', self.ub_overflow(na, n, bn, bd, width), locals()):
                continue
            self.site('matmul_v3_base_tiling.cpp', 1311)
            nloops = cd(na, bn)
            self.site('matmul_v3_base_tiling.cpp', 1311)
            tail = nloops * dloops % cores
            while self.predicate('matmul_v3_base_tiling.cpp', 1338, 'bn > 16', bn > 16, locals()):
                self.count_loop('Nd2Nz.large_d_shrink')
                if self.predicate('matmul_v3_base_tiling.cpp', 1338, 'self.ub_overflow(na, n, bn, bd, width)', self.ub_overflow(na, n, bn, bd, width), locals()):
                    bn -= 1
                    self.site('matmul_v3_base_tiling.cpp', 1338)
                    nloops = cd(na, bn)
                    self.site('matmul_v3_base_tiling.cpp', 1338)
                    tail = nloops * dloops % cores
                    continue
                if self.predicate('matmul_v3_base_tiling.cpp', 1338, 'tail == 0', tail == 0, locals()):
                    return (bn, bd)
                if self.predicate('matmul_v3_base_tiling.cpp', 1338, 'tail > last_tail', tail > last_tail, locals()):
                    self.site('matmul_v3_base_tiling.cpp', 1338)
                    last_tail = tail
                    self.site('matmul_v3_base_tiling.cpp', 1338)
                    best_d = bd
                    self.site('matmul_v3_base_tiling.cpp', 1338)
                    best_n = bn
                bn -= 1
                self.site('matmul_v3_base_tiling.cpp', 1338)
                nloops = cd(na, bn)
                self.site('matmul_v3_base_tiling.cpp', 1338)
                tail = nloops * dloops % cores
        return (best_n, best_d)

    @sourced('DoNd2NzVectorTiling', 1362, 1379)
    def do_nd2nz_vector(self):
        self.site('matmul_v3_base_tiling.cpp', 1362)
        a, r = (self.args, self.run)
        if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.nd2nzA', a.nd2nzA, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1362)
            n = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.isATrans', a.isATrans, locals()) else a.mValue
            self.site('matmul_v3_base_tiling.cpp', 1362)
            d = a.mValue if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.isATrans', a.isATrans, locals()) else a.kValue
            self.site('matmul_v3_base_tiling.cpp', 1362)
            r.baseAN, r.baseAD = self.nd2nz_geometry(self.aw, n, d)
        if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.nd2nzB', a.nd2nzB, locals()):
            self.site('matmul_v3_base_tiling.cpp', 1362)
            n = a.nValue if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.isBTrans', a.isBTrans, locals()) else a.kValue
            self.site('matmul_v3_base_tiling.cpp', 1362)
            d = a.kValue if self.predicate('matmul_v3_base_tiling.cpp', 1362, 'a.isBTrans', a.isBTrans, locals()) else a.nValue
            self.site('matmul_v3_base_tiling.cpp', 1362)
            r.baseBN, r.baseBD = self.nd2nz_geometry(self.bw, n, d)

    @sourced('CheckMMTilingDataIsVaild', 2732, 2753)
    def check_fields(self):
        self.site('matmul_v3_base_tiling.cpp', 2732)
        r = self.run
        self.site('matmul_v3_base_tiling.cpp', 2732)
        limited = {'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN', 'baseK'}
        self.site('matmul_v3_base_tiling.cpp', 2732)
        checked = ('usedCoreNum', 'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN', 'baseK', 'depthA1', 'depthB1', 'stepM', 'stepN', 'stepKa', 'stepKb', 'iterateOrder', 'dbL0c')
        for name in checked:
            self.site('matmul_v3_base_tiling.cpp', 2732)
            maximum = MASK32 // 16 * 16 if self.predicate('matmul_v3_base_tiling.cpp', 2732, 'name in limited', name in limited, locals()) else MASK32
            if self.predicate('matmul_v3_base_tiling.cpp', 2732, 'getattr(r, name) > maximum', getattr(r, name) > maximum, locals()):
                raise SourceFieldOverflow(f'{name} exceeds the source uint32 validation limit {maximum}')
        for name in ('mTile', 'nTile', 'mTileBlock', 'nTileBlock'):
            if self.predicate('matmul_v3_base_tiling.cpp', 2732, 'getattr(r.l2Info, name) > MASK32', getattr(r.l2Info, name) > MASK32, locals()):
                raise SourceFieldOverflow(f'l2Info.{name} exceeds UINT32_MAX')

    @sourced('SetRunInfo', 948, 957)
    def set_run_info(self):
        self.site('matmul_v3_base_tiling.cpp', 948)
        a = self.args
        return {'transA': int(a.isATrans), 'transB': int(a.isBTrans), 'nd2nzA': int(a.nd2nzA), 'nd2nzB': int(a.nd2nzB), 'isNzA': int(a.isNzA), 'isNzB': int(a.isNzB), 'isHf32': int(a.isHf32)}

    @sourced('SetNd2NzInfo', 940, 946)
    def set_nd2nz_info(self):
        self.site('matmul_v3_base_tiling.cpp', 940)
        r = self.run
        return {name: int(getattr(r, name)) & MASK32 for name in ('baseAN', 'baseAD', 'baseBN', 'baseBD')}

    @sourced('DoTilingKey', 2800, 2810)
    def get_key_arguments(self):
        self.site('matmul_v3_base_tiling.cpp', 2800)
        e = self.enable
        return {'LOADMODE': int(e.tilingEnableFullLoad), 'SPLITCOREMODE': int(e.tilingEnableSplitCore), 'FIXOPTI': int(e.tilingEnableFixOpti), 'MIXND2NZ': self.mix_type(), 'SPECIALOPT': int(e.tilingEnableSpecialOpti), 'FP32ADDMM': 0}

    @sourced('L2Cache::SetL2CacheFlagBase', 30, 58, 'matmul_v3_l2_cache.cpp')
    def l2_flags_base(self):
        self.site('matmul_v3_l2_cache.cpp', 30)
        a, r = (self.args, self.run)
        if self.predicate('matmul_v3_l2_cache.cpp', 30, 'r.l2Info.mTile > 1 or r.l2Info.nTile > 1', r.l2Info.mTile > 1 or r.l2Info.nTile > 1, locals()):
            return (r.l2Info.nTile > 1, r.l2Info.mTile > 1)
        self.site('matmul_v3_l2_cache.cpp', 30)
        ae = self.predicate('matmul_v3_l2_cache.cpp', 30, 'r.singleCoreN < a.nValue', r.singleCoreN < a.nValue, locals())
        self.site('matmul_v3_l2_cache.cpp', 30)
        be = self.predicate('matmul_v3_l2_cache.cpp', 30, 'r.singleCoreM < a.mValue', r.singleCoreM < a.mValue, locals())
        self.site('matmul_v3_l2_cache.cpp', 30)
        count = cd(a.mValue, r.singleCoreM) * cd(a.nValue, r.singleCoreN)
        self.site('matmul_v3_l2_cache.cpp', 30)
        rounds = cd(count, r.usedCoreNum)
        if self.predicate('matmul_v3_l2_cache.cpp', 30, 'rounds > 1', rounds > 1, locals()):
            self.site('matmul_v3_l2_cache.cpp', 30)
            be = True
        return (ae, be)

    @sourced('L2Cache::SetL2CacheFlagSingleCoreSplitK', 60, 76, 'matmul_v3_l2_cache.cpp')
    def l2_flags_single(self):
        self.site('matmul_v3_l2_cache.cpp', 60)
        a, r = (self.args, self.run)
        self.site('matmul_v3_l2_cache.cpp', 60)
        be = self.predicate('matmul_v3_l2_cache.cpp', 60, 'r.singleCoreM < a.mValue', r.singleCoreM < a.mValue, locals())
        self.site('matmul_v3_l2_cache.cpp', 60)
        ae = self.predicate('matmul_v3_l2_cache.cpp', 60, 'r.singleCoreN < a.nValue', r.singleCoreN < a.nValue, locals())
        self.site('matmul_v3_l2_cache.cpp', 60)
        mfull = self.predicate('matmul_v3_l2_cache.cpp', 60, 'r.singleCoreM <= r.baseM * r.stepM', r.singleCoreM <= r.baseM * r.stepM, locals())
        self.site('matmul_v3_l2_cache.cpp', 60)
        nfull = self.predicate('matmul_v3_l2_cache.cpp', 60, 'r.singleCoreN <= r.baseN * r.stepN', r.singleCoreN <= r.baseN * r.stepN, locals())
        self.site('matmul_v3_l2_cache.cpp', 60)
        kfull = self.predicate('matmul_v3_l2_cache.cpp', 60, 'r.singleCoreK <= a.kValue', r.singleCoreK <= a.kValue, locals())
        if self.predicate('matmul_v3_l2_cache.cpp', 60, 'not mfull and (not nfull)', not mfull and (not nfull), locals()):
            self.site('matmul_v3_l2_cache.cpp', 60)
            be = True
            self.site('matmul_v3_l2_cache.cpp', 60)
            ae = not kfull
        return (ae, be)

    @sourced('L2Cache::SetL2CacheFlagMultiCoreSplitK', 78, 96, 'matmul_v3_l2_cache.cpp')
    def l2_flags_multi(self):
        self.site('matmul_v3_l2_cache.cpp', 78)
        a, r = (self.args, self.run)
        self.site('matmul_v3_l2_cache.cpp', 78)
        ae = False
        self.site('matmul_v3_l2_cache.cpp', 78)
        be = self.predicate('matmul_v3_l2_cache.cpp', 78, 'r.singleCoreM < a.mValue', r.singleCoreM < a.mValue, locals())
        self.site('matmul_v3_l2_cache.cpp', 78)
        count = cd(a.mValue, r.singleCoreM) * cd(a.kValue, r.singleCoreK)
        self.site('matmul_v3_l2_cache.cpp', 78)
        rounds = cd(count, r.usedCoreNum)
        if self.predicate('matmul_v3_l2_cache.cpp', 78, 'rounds > 1 and r.singleCoreN != a.nValue', rounds > 1 and r.singleCoreN != a.nValue, locals()):
            self.site('matmul_v3_l2_cache.cpp', 78)
            ae = True
            self.site('matmul_v3_l2_cache.cpp', 78)
            be = True
        return (ae, be)

    @sourced('L2Cache::SetL2CacheFlag', 126, 167, 'matmul_v3_l2_cache.cpp')
    def compute_l2_flag(self):
        self.site('matmul_v3_l2_cache.cpp', 126)
        a, r, e = (self.args, self.run, self.enable)
        self.site('matmul_v3_l2_cache.cpp', 126)
        ae = False
        self.site('matmul_v3_l2_cache.cpp', 126)
        be = False
        self.site('matmul_v3_l2_cache.cpp', 126)
        ce = self.predicate('matmul_v3_l2_cache.cpp', 126, 'a.mValue * a.nValue * self.cw <= self.hw.l2Size', a.mValue * a.nValue * self.cw <= self.hw.l2Size, locals())
        if self.predicate('matmul_v3_l2_cache.cpp', 126, 'e.tilingEnableSplitCore == 0', e.tilingEnableSplitCore == 0, locals()):
            self.site('matmul_v3_l2_cache.cpp', 126)
            ae, be = self.l2_flags_base()
        elif self.predicate('matmul_v3_l2_cache.cpp', 126, 'e.tilingEnableSplitCore == 2', e.tilingEnableSplitCore == 2, locals()):
            self.site('matmul_v3_l2_cache.cpp', 126)
            ce = True
            self.site('matmul_v3_l2_cache.cpp', 126)
            ae, be = self.l2_flags_single()
        elif self.predicate('matmul_v3_l2_cache.cpp', 126, 'e.tilingEnableSplitCore == 3', e.tilingEnableSplitCore == 3, locals()):
            self.site('matmul_v3_l2_cache.cpp', 126)
            ce = True
            self.site('matmul_v3_l2_cache.cpp', 126)
            ae, be = self.l2_flags_multi()
        if self.predicate('matmul_v3_l2_cache.cpp', 102, 'ae and be and ce', ae and be and ce, locals()):
            return self.l2_cache_flag | 1
        self.site('matmul_v3_l2_cache.cpp', 102)
        flag = self.l2_cache_flag
        if self.predicate('matmul_v3_l2_cache.cpp', 102, 'not ae', not ae, locals()):
            self.site('matmul_v3_l2_cache.cpp', 102)
            flag = flag | 2
        if self.predicate('matmul_v3_l2_cache.cpp', 102, 'not be', not be, locals()):
            self.site('matmul_v3_l2_cache.cpp', 102)
            flag = flag | 4
        if self.predicate('matmul_v3_l2_cache.cpp', 102, 'not ce', not ce, locals()):
            self.site('matmul_v3_l2_cache.cpp', 102)
            flag = flag | 16
        return flag

    @sourced('DoLibApiTiling', 2755, 2796)
    def finalize_fields(self):
        self.site('matmul_v3_base_tiling.cpp', 2755)
        a, r, e = (self.args, self.run, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 2755)
        run_fields = self.set_run_info()
        self.check_fields()
        self.site('matmul_v3_base_tiling.cpp', 2755)
        cube = {name: int(getattr(r, name)) for name in ('usedCoreNum', 'singleCoreM', 'singleCoreN', 'singleCoreK', 'baseM', 'baseN', 'baseK', 'depthA1', 'depthB1', 'stepM', 'stepN', 'stepKa', 'stepKb', 'iterateOrder')}
        if self.predicate('matmul_v3_base_tiling.cpp', 2775, 'e.tilingEnableFullLoad != 1 or e.tilingEnableSplitCore != 2', e.tilingEnableFullLoad != 1 or e.tilingEnableSplitCore != 2, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2775)
            cube['dbL0C'] = int(r.dbL0c)
        else:
            self.site('matmul_v3_base_tiling.cpp', 2775)
            cube['dbL0C'] = None
        cube.update(self.cube_extra)
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['dbL0A'] = None
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['dbL0B'] = None
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['M'] = None
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['N'] = None
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['Ka'] = None
        self.site('matmul_v3_base_tiling.cpp', 2775)
        cube['Kb'] = None
        self.site('matmul_v3_base_tiling.cpp', 2784)
        l2 = {'mTileCntL2': int(r.l2Info.mTile), 'nTileCntL2': int(r.l2Info.nTile), 'mTileBlock': int(r.l2Info.mTileBlock), 'nTileBlock': int(r.l2Info.nTileBlock), 'calOrder': int(r.l2Info.calOrder)}
        self.site('matmul_v3_base_tiling.cpp', 2784)
        packet_flag = int(self.l2_cache_flag)
        self.site('matmul_v3_base_tiling.cpp', 2784)
        vector = self.set_nd2nz_info()
        self.site('matmul_v3_base_tiling.cpp', 2784)
        self.key_arguments = self.get_key_arguments()
        self.site('matmul_v3_base_tiling.cpp', 2793)
        self.l2_cache_flag = self.compute_l2_flag()
        return {'matmulTiling': cube, 'tileL2cacheTiling': l2, 'matmulRunInfo': run_fields, 'l2cacheUseInfo': {'l2CacheFlag': packet_flag}, **vector}

    @sourced('GetDeterministicSplitKWorkspaceSize', 2812, 2829)
    def deterministic_workspace(self):
        self.site('matmul_v3_base_tiling.cpp', 2812)
        r = self.run
        self.site('matmul_v3_base_tiling.cpp', 2812)
        single_size = r.singleCoreN * r.singleCoreM
        if self.predicate('matmul_v3_base_tiling.cpp', 2812, 'self.enable.tilingEnableFixOpti == 2', self.enable.tilingEnableFixOpti == 2, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2812)
            single_size = nz_sequential_size(r.singleCoreM, r.singleCoreN, r.baseM, r.baseN)
        return r.usedCoreNum * single_size * 2 * 4 + RPC_WORKSIZE * MB_SIZE

    @sourced('GetWorkspaceSize', 2881, 2919)
    def workspace(self):
        self.site('matmul_v3_base_tiling.cpp', 2881)
        a, r, e = (self.args, self.run, self.enable)
        self.site('matmul_v3_base_tiling.cpp', 2881)
        rpc = U64(RPC_WORKSIZE) * MB_SIZE
        self.site('matmul_v3_base_tiling.cpp', 2881)
        result = rpc
        self.site('matmul_v3_base_tiling.cpp', 2881)
        parts = {'rpc': int(rpc), 'single_core_split_k': 0, 'deterministic_split_k_including_rpc': 0, 'align_out': 0, 'vec_nz2nd': 0, 'nd2nz_a': 0, 'nd2nz_b': 0}
        if self.predicate('matmul_v3_base_tiling.cpp', 2843, 'e.tilingEnableSplitCore in (2, 5, 6)', e.tilingEnableSplitCore in (2, 5, 6), locals()):
            self.site('matmul_v3_base_tiling.cpp', 2843)
            split = a.mValue * up(a.nValue, 256 // self.aw) * 4
            self.site('matmul_v3_base_tiling.cpp', 2843)
            result = split + rpc
            self.site('matmul_v3_base_tiling.cpp', 2843)
            parts['single_core_split_k'] = int(split)
        if self.predicate('matmul_v3_base_tiling.cpp', 2843, 'e.tilingEnableSplitCore == 3', e.tilingEnableSplitCore == 3, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2843)
            result = self.deterministic_workspace()
            self.site('matmul_v3_base_tiling.cpp', 2843)
            parts['deterministic_split_k_including_rpc'] = int(result)
        if self.predicate('matmul_v3_base_tiling.cpp', 2898, 'e.tilingEnableFixOpti == 1', e.tilingEnableFixOpti == 1, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2898)
            extra = up(a.nValue, 512 // self.cw) * r.baseM * r.usedCoreNum * 2 * self.cw
            result += extra
            self.site('matmul_v3_base_tiling.cpp', 2898)
            parts['align_out'] = int(extra)
        if self.predicate('matmul_v3_base_tiling.cpp', 2898, 'e.tilingEnableFixOpti == 2', e.tilingEnableFixOpti == 2, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2898)
            extra = up(a.nValue, 16) * r.baseM * r.usedCoreNum * 2 * self.cw
            result += extra
            self.site('matmul_v3_base_tiling.cpp', 2898)
            parts['vec_nz2nd'] = int(extra)
        if self.predicate('matmul_v3_base_tiling.cpp', 2858, 'self.hw.supportL0c2out', self.hw.supportL0c2out, locals()):
            self.site('matmul_v3_base_tiling.cpp', 2858)
            c0 = U64(32) // self.aw
            self.site('matmul_v3_base_tiling.cpp', 2858)
            kc0 = up(a.kValue, c0)
            self.site('matmul_v3_base_tiling.cpp', 2858)
            kn = up(a.kValue, 16)
            if self.predicate('matmul_v3_base_tiling.cpp', 2858, 'a.nd2nzA', a.nd2nzA, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2858)
                extra = up(a.mValue, c0) * kn * self.aw if self.predicate('matmul_v3_base_tiling.cpp', 2858, 'a.isATrans', a.isATrans, locals()) else up(a.mValue, 16) * kc0 * self.aw
                result += extra
                self.site('matmul_v3_base_tiling.cpp', 2858)
                parts['nd2nz_a'] = int(extra)
            if self.predicate('matmul_v3_base_tiling.cpp', 2858, 'a.nd2nzB', a.nd2nzB, locals()):
                self.site('matmul_v3_base_tiling.cpp', 2858)
                extra = up(a.nValue, 16) * kc0 * self.bw if self.predicate('matmul_v3_base_tiling.cpp', 2858, 'a.isBTrans', a.isBTrans, locals()) else up(a.nValue, c0) * kn * self.bw
                result += extra
                self.site('matmul_v3_base_tiling.cpp', 2858)
                parts['nd2nz_b'] = int(extra)
        return (int(result), parts)
