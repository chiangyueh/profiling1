"""Generated offline from immutable commit-960 C++ with typed Clang AST.
Retained from Track 01 typed translation; no runtime C++ dependency.
Source notices: NOTICE.txt and evidence/initializer/LICENSE.txt.
"""
from types import MappingProxyType
from math import sqrt, ceil, floor, pow
EOK = 0
from .initializer_runtime import (Cell, AttrRef, ItemRef, clone, i32, u32, i64, u64, f32, cdiv, cmod, assign, increment, vector_erase, vector_unique, copy_array, MathUtil)

class DataType:
    DT_FLOAT = 0
    DT_FLOAT16 = 1
    DT_INT8 = 2
    DT_INT16 = 6
    DT_UINT16 = 7
    DT_UINT8 = 4
    DT_INT32 = 3
    DT_INT64 = 9
    DT_UINT32 = 8
    DT_UINT64 = 10
    DT_BOOL = 12
    DT_DOUBLE = 11
    DT_STRING = 13
    DT_DUAL_SUB_INT8 = 14
    DT_DUAL_SUB_UINT8 = 15
    DT_COMPLEX64 = 16
    DT_COMPLEX128 = 17
    DT_QINT8 = 18
    DT_QINT16 = 19
    DT_QINT32 = 20
    DT_QUINT8 = 21
    DT_QUINT16 = 22
    DT_RESOURCE = 23
    DT_STRING_REF = 24
    DT_DUAL = 25
    DT_VARIANT = 26
    DT_BF16 = 27
    DT_UNDEFINED = 28
    DT_INT4 = 29
    DT_UINT1 = 30
    DT_INT2 = 31
    DT_UINT2 = 32
    DT_BFLOAT16 = 33
    DT_MAX = 34

class TPosition:
    GM = 0
    A1 = 1
    A2 = 2
    B1 = 3
    B2 = 4
    C1 = 5
    C2 = 6
    CO1 = 7
    CO2 = 8
    VECIN = 9
    VECOUT = 10
    VECCALC = 11
    LCM = 11
    SPM = 12
    SHM = 12
    TSCM = 13
    MAX = 14

class TilingPolicy:
    FIXED_A_TSCM = 0
    FIXED_B_TSCM = 1
    FIXED_A_B_TSCM = 2
    NO_POLICY = 3

class CubeFormat:
    ND = 0
    NZ = 1
    ZN = 2
    ZZ = 3
    NN = 4
    ND_ALIGN = 5
    SCALAR = 6
    VECTOR = 7

class MatrixTraverse:
    NOSET = 0
    FIRSTM = 1
    FIRSTN = 2

class MatrixMadType:
    NORMAL = 0
    HF32 = 1

class DequantType:
    SCALAR = 0
    TENSOR = 1

class ScheduleType:
    INNER_PRODUCT = 0
    OUTER_PRODUCT = 1

class L1TilingType:
    KAL1_16 = 0
    KBL1_16 = 1
    M_AL1 = 2
    N_BL1 = 3

UINT8_BYTES = 1
INT8_BYTES = 1
FP32_BYTES = 4
FP16_BYTES = 2
C0_SIZE = 16
C0_BYTE_SIZE = 32
BITS_PER_BYTE = 8
DB_ON = 2
DB_OFF = 1
L1_FACTORS_LEN = 6
L0PARAS_COMBO_LEN = 2
IDX_ZERO = u32(0)
IDX_ONE = u32(1)
IDX_TWO = u32(2)
IDX_THREE = u32(3)
IDX_FOUR = u32(4)
IDX_FIVE = u32(5)
IDX_SIX = u32(6)
IDX_SEVEN = u32(7)
MAX_BIAS_N = 16
MTE1_L0A_BANDWIDTH = 256
MTE1_L0B_BANDWIDTH = 128
INPUTDTYPE_BYTES = 2
MIN_MTE1_LOAD = 32
REDUCE_BLOCK_SIZE = 16
INT8_REDUCE_BLOCK_SIZE = 32
INT4_REDUCE_BLOCK_SIZE = 64
FLOAT32_REDUCE_BLOCK_SIZE = 8
MIN_FRACTAL_SIZE = i32((C0_SIZE * REDUCE_BLOCK_SIZE))
BEST_VALUE_LENGTH = u32(13)
BEST_VALUE_LIST = tuple([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
DIM_FACTOR_LENGTH = u32(4)
UINT64_TYPES = u64(8)
L0B_ALIGN_SIZE = 2
L0_FACTOR_NUM_LIMIT = 2
L1_FACTOR_NUM_LIMIT = 4
L0_FACTOR_LIMIT = 64
L1_FACTOR_LIMIT = 128
MTE1_FIXPIPE_BANDWIDTH = 128
NUM_TWO = 2
ATTACH_FLAG_ZERO = 0
ATTACH_FLAG_ONE = 1
ATTACH_FLAG_TWO = 2
INT8_ALIGN_SIZE = 32
FP32_ALIGN_SIZE = 16
INT4_ALIGN_SIZE = 64
DT_FLOAT_INVALID_BASEK = 8
DTYPE_BYTE_TAB = MappingProxyType({0:4,1:2,2:1,6:2,7:2,4:1,3:4,9:8,8:4,10:8,27:2,33:2,29:1})
DTYPE_BIT_TAB = MappingProxyType({0:32,1:16,2:8,6:16,7:16,4:8,3:32,9:64,8:32,10:64,27:16,33:16,29:4})

class SysTilingTempBufSize:
    def __init__(self, *args):
        self.ubSize = 0
        self.l1Size = 0
        self.l0cSize = 0
        for name, value in zip(('ubSize', 'l1Size', 'l0cSize'), args): setattr(self, name, clone(value))
    

class MatTilingType:
    def __init__(self, *args):
        self.pos = TPosition.GM
        self.type = CubeFormat.ND
        self.dataType = DataType.DT_FLOAT
        self.isTrans = False
        self.isDB = False
        for name, value in zip(('pos', 'type', 'dataType', 'isTrans', 'isDB'), args): setattr(self, name, clone(value))
    

class BufferPool:
    def __init__(self, *args):
        self.l1Size = 0
        self.l0CSize = 0
        self.ubSize = 0
        self.l0ASize = 0
        self.l0BSize = 0
        self.btSize = 0
        self.l1AlignSize = 0
        self.l0CAlignSize = 0
        self.l0AAlignSize = 0
        self.l0BAlignSize = 0
        self.ubAlignSize = 0
        for name, value in zip(('l1Size', 'l0CSize', 'ubSize', 'l0ASize', 'l0BSize', 'btSize', 'l1AlignSize', 'l0CAlignSize', 'l0AAlignSize', 'l0BAlignSize', 'ubAlignSize'), args): setattr(self, name, clone(value))
    

class PlatformInfo:
    def __init__(self, *args):
        self.socVersion = 0
        self.l1Size = u64(0)
        self.l0CSize = u64(0)
        self.ubSize = u64(0)
        self.l0ASize = u64(0)
        self.l0BSize = u64(0)
        for name, value in zip(('socVersion', 'l1Size', 'l0CSize', 'ubSize', 'l0ASize', 'l0BSize'), args): setattr(self, name, clone(value))
    

class MatmulConfigParams:
    def __init__(self, *args):
        self.mmConfigType = 0
        self.enableL1CacheUB = False
        self.scheduleType = 0
        self.traverse = 0
        self.enVecND2NZ = False
        for name, value in zip(('mmConfigType', 'enableL1CacheUB', 'scheduleType', 'traverse', 'enVecND2NZ'), args): setattr(self, name, clone(value))
    

class MnmAdjust:
    def __init__(self, *args):
        self.maxBaseM = 0
        self.maxBaseN = 0
        self.maxBaseK = 0
        self.minBaseM = 0
        self.minBaseN = 0
        self.minBaseK = 0
        for name, value in zip(('maxBaseM', 'maxBaseN', 'maxBaseK', 'minBaseM', 'minBaseN', 'minBaseK'), args): setattr(self, name, clone(value))
    

class MatmulTemplateCfg:
    def __init__(self, *args):
        self.l0aDB = DB_ON
        self.l0bDB = DB_ON
        self.l0cDB = DB_OFF
        self.l1DB = DB_ON
        self.factorSplit = True
        self.kSplit = False
        for name, value in zip(('l0aDB', 'l0bDB', 'l0cDB', 'l1DB', 'factorSplit', 'kSplit'), args): setattr(self, name, clone(value))
    

class L1StatusPack:
    def __init__(self, *args):
        self.kAL1 = 1
        self.kBL1 = 1
        self.mL1 = 1
        self.nL1 = 1
        self.mAL1 = 1
        self.nBL1 = 1
        self.dbAL1 = 1
        self.dbBL1 = 1
        self.aL1Size = 0
        self.bL1Size = 0
        self.aL1Times = 1
        self.bL1Times = 1
        self.allTimes = 1
        self.loadSize = 0
        self.maxMAL1 = 1
        self.maxNBL1 = 1
        self.maxKAL1 = 1
        self.maxKBL1 = 1
        self.bothFullLoad = False
        self.aL1FullLoad = False
        self.bL1FullLoad = False
        self.aL1KFullLoad = False
        self.bL1KFullLoad = False
        self.channelWiseTimes = 0
        for name, value in zip(('kAL1', 'kBL1', 'mL1', 'nL1', 'mAL1', 'nBL1', 'dbAL1', 'dbBL1', 'aL1Size', 'bL1Size', 'aL1Times', 'bL1Times', 'allTimes', 'loadSize', 'maxMAL1', 'maxNBL1', 'maxKAL1', 'maxKBL1', 'bothFullLoad', 'aL1FullLoad', 'bL1FullLoad', 'aL1KFullLoad', 'bL1KFullLoad', 'channelWiseTimes'), args): setattr(self, name, clone(value))
    
    def SetStatus(self, tmpL1Factors):
        'Source: L1StatusPack::SetStatus; lines 68-76'
        assign(AttrRef(self,'kAL1'),i32(tmpL1Factors.v[0]))
        assign(AttrRef(self,'kBL1'),i32(tmpL1Factors.v[1]))
        assign(AttrRef(self,'mAL1'),i32(tmpL1Factors.v[2]))
        assign(AttrRef(self,'nBL1'),i32(tmpL1Factors.v[3]))
        assign(AttrRef(self,'dbAL1'),i32(tmpL1Factors.v[4]))
        assign(AttrRef(self,'dbBL1'),i32(tmpL1Factors.v[5]))
    

class L0StatusPack:
    def __init__(self, *args):
        self.mL0 = 1
        self.nL0 = 1
        self.kL0 = 1
        self.batchL0 = 1
        self.l0cMultiBatch = 0
        self.dbL0A = 1
        self.dbL0B = 1
        self.dbL0C = 1
        self.dbCub = 1
        self.finalML0 = 0
        self.finalKL0 = 0
        self.finalNL0 = 0
        self.finalLoadSize = 2147483647
        self.finalL0cUse = f32(0)
        self.finalMul = 0
        self.finalMte1Loop = 2147483647
        self.finalMte1Cycles = 0
        self.maxMk = 1
        self.maxNk = 1
        self.maxMn = 1
        self.maxAxisIdx = 0
        self.maxAxisNum = 0
        self.maxAxisPnt = 1
        self.maxN = 1
        self.dtypeBias = 0
        self.load2dTimes = 0
        self.l0cUsed = 0
        self.updateUsingMte1 = False
        for name, value in zip(('mL0', 'nL0', 'kL0', 'batchL0', 'l0cMultiBatch', 'dbL0A', 'dbL0B', 'dbL0C', 'dbCub', 'finalML0', 'finalKL0', 'finalNL0', 'finalLoadSize', 'finalL0cUse', 'finalMul', 'finalMte1Loop', 'finalMte1Cycles', 'maxMk', 'maxNk', 'maxMn', 'maxAxisIdx', 'maxAxisNum', 'maxAxisPnt', 'maxN', 'dtypeBias', 'load2dTimes', 'l0cUsed', 'updateUsingMte1'), args): setattr(self, name, clone(value))
    
    def InitLoadStatus(self):
        'Source: L0StatusPack::InitLoadStatus; lines 108-119'
        assign(AttrRef(self,'finalML0'),i32(0))
        assign(AttrRef(self,'finalKL0'),i32(0))
        assign(AttrRef(self,'finalNL0'),i32(0))
        assign(AttrRef(self,'finalLoadSize'),i32(2147483647))
        assign(AttrRef(self,'finalL0cUse'),f32(f32(0)))
        assign(AttrRef(self,'finalMul'),i32(0))
        assign(AttrRef(self,'finalMte1Loop'),i32(2147483647))
        assign(AttrRef(self,'finalMte1Cycles'),i32(0))
        assign(AttrRef(self,'updateUsingMte1'),bool(False))
    

class CoreStatusPack:
    def __init__(self, *args):
        self.batch = 1
        self.m = 1
        self.k = 1
        self.n = 1
        self.batchDim = 1
        self.mDim = 1
        self.nDim = 1
        self.kDim = 1
        self.kAl1Factor = 1
        self.kBl1Factor = 1
        self.mSingleCore = 1
        self.nSingleCore = 1
        self.n0Max = 1
        self.cycle = i64(1)
        self.loadSize = 1
        self.aL1FullLoadSize = 0
        self.bL1FullLoadSize = 0
        self.madCycle = i64(1)
        self.repeatLoadSize = i64(2147483647)
        self.cycle = self.loadSize = 2147483647
        for name, value in zip(('batch', 'm', 'k', 'n', 'batchDim', 'mDim', 'nDim', 'kDim', 'kAl1Factor', 'kBl1Factor', 'mSingleCore', 'nSingleCore', 'n0Max', 'cycle', 'loadSize', 'aL1FullLoadSize', 'bL1FullLoadSize', 'madCycle', 'repeatLoadSize'), args): setattr(self, name, clone(value))
    

class SingleCoreStatus:
    def __init__(self, *args):
        self.l0Status = L0StatusPack()
        self.l1Status = L1StatusPack()
        for name, value in zip(('l0Status', 'l1Status'), args): setattr(self, name, clone(value))
    

class L0Factors:
    def __init__(self, *args):
        self.finalML0 = 0
        self.finalKL0 = 0
        self.finalNL0 = 0
        self.finalLoadSize = 2147483647
        self.finalL0cUse = f32(0)
        self.finalMul = 0
        self.finalMte1Loop = 2147483647
        self.finalMte1Cycles = 0
        for name, value in zip(('finalML0', 'finalKL0', 'finalNL0', 'finalLoadSize', 'finalL0cUse', 'finalMul', 'finalMte1Loop', 'finalMte1Cycles'), args): setattr(self, name, clone(value))
    

class MKNParasCombo:
    def __init__(self, *args):
        self.parasCombo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for name, value in zip(('parasCombo',), args): setattr(self, name, clone(value))
    

class MatmulRunParas:
    def __init__(self, *args):
        self.oriShapeM = i64(1)
        self.oriShapeN = i64(1)
        self.oriShapeKa = i64(1)
        self.oriShapeKb = i64(1)
        self.oriShapeAbatch = i64(1)
        self.oriShapeBbatch = i64(1)
        self.dtypeA = 0
        self.dtypeB = 0
        self.dtypeOut = 0
        self.dtypeBias = 0
        self.transA = False
        self.transB = False
        self.formatANd = False
        self.formatBNd = False
        self.formatOutNd = False
        self.biasFlag = False
        self.hf32Flag = bool(1)
        self.batch32 = 1
        self.m32 = 1
        self.k32 = 1
        self.n32 = 1
        self.batch32A = 1
        self.batch32B = 1
        self.mMapped = 1
        self.kMapped = 1
        self.nMapped = 1
        self.batchMapped = 1
        self.nonFactorK = False
        for name, value in zip(('oriShapeM', 'oriShapeN', 'oriShapeKa', 'oriShapeKb', 'oriShapeAbatch', 'oriShapeBbatch', 'dtypeA', 'dtypeB', 'dtypeOut', 'dtypeBias', 'transA', 'transB', 'formatANd', 'formatBNd', 'formatOutNd', 'biasFlag', 'hf32Flag', 'batch32', 'm32', 'k32', 'n32', 'batch32A', 'batch32B', 'mMapped', 'kMapped', 'nMapped', 'batchMapped', 'nonFactorK'), args): setattr(self, name, clone(value))
    

class BlockDimCalculator:
    def __init__(self, *args):
        self.batch = 1
        self.m = 1
        self.k = 1
        self.n = 1
        self.kNum = 1
        self.kBytes = 1
        self.batchDimFactor = 1
        self.mDimFactor = 1
        self.nDimFactor = 1
        self.kDimFactor = 1
        self.minLoadSize = 1
        self.coreUse = 1
        self.tmpCoreUse = 1
        self.loopNumToL0 = 1
        self.batchIdx = 0
        self.nIdx = 0
        self.batchDimCnt = 0
        self.mDimCnt = 0
        self.nDimCnt = 0
        self.kDimCnt = 0
        self.batchFactorCnt = 0
        self.oriAmatSize = 0
        self.oriBmatSize = 0
        self.amatSize = 0
        self.bmatSize = 0
        self.tmpAmatSize = 0
        self.tmpBmatSize = 0
        self.tmpLoadSize = 0
        self.totalLoadSize = 0
        self.tmpValue = 0
        self.finalValue = 0
        self.batchDimFactors = []
        self.mDimFactors = []
        self.nDimFactors = []
        self.kDimFactors = []
        self.initFlag = False
        self.bigPackage = False
        self.minLoadSize = self.loopNumToL0 = 2147483647
        for name, value in zip(('batch', 'm', 'k', 'n', 'kNum', 'kBytes', 'batchDimFactor', 'mDimFactor', 'nDimFactor', 'kDimFactor', 'minLoadSize', 'coreUse', 'tmpCoreUse', 'loopNumToL0', 'batchIdx', 'nIdx', 'batchDimCnt', 'mDimCnt', 'nDimCnt', 'kDimCnt', 'batchFactorCnt', 'oriAmatSize', 'oriBmatSize', 'amatSize', 'bmatSize', 'tmpAmatSize', 'tmpBmatSize', 'tmpLoadSize', 'totalLoadSize', 'tmpValue', 'finalValue', 'batchDimFactors', 'mDimFactors', 'nDimFactors', 'kDimFactors', 'initFlag', 'bigPackage'), args): setattr(self, name, clone(value))
    

class DimFactor:
    def __init__(self, *args):
        self.batch = 1
        self.m = 1
        self.k = 1
        self.n = 1
        self.group = 1
        for name, value in zip(('batch', 'm', 'k', 'n', 'group'), args): setattr(self, name, clone(value))
    
    def ReduceMul(self):
        'Source: DimFactor::ReduceMul; lines 277-280'
        return i32((i32((i32((self.batch * self.m)) * self.k)) * self.n))
    
    def Init(self):
        'Source: DimFactor::Init; lines 282-289'
        assign(AttrRef(self,'batch'),i32(1))
        assign(AttrRef(self,'m'),i32(1))
        assign(AttrRef(self,'k'),i32(1))
        assign(AttrRef(self,'n'),i32(1))
        assign(AttrRef(self,'group'),i32(1))
    
    def IsValid(self):
        'Source: DimFactor::IsValid; lines 291-294'
        return (((((self.group > 0) and (self.batch > 0)) and (self.m > 0)) and (self.k > 0)) and (self.n > 0))
    

class MatmulTilingAlgorithm:
    def __init__(self, tiling, trace=False):
        self.tilingIns_ = tiling
        self.singelBlockDim_ = False
        self.splitCoreFlag_ = False
        self.dbL0A_ = 2
        self.dbL0B_ = 2
        self.dbL0C_ = 1
        self.numOfBlock_ = 24
        self.cfg = MatmulTemplateCfg()
        self.trace_enabled = trace
        self.events = []
        self.counts = {}
        self.branch_counts = {}
    def _visit(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1
        if self.trace_enabled: self.events.append({"function": name})
    def _branch(self, site, value):
        value = bool(value)
        if self.trace_enabled:
            key = str(site) + (":true" if value else ":false")
            self.branch_counts[key] = self.branch_counts.get(key, 0) + 1
        return value
    def GetBestValue(self, base):
        'Source: MatmulTilingAlgorithm::GetBestValue; lines 84-93'
        self._visit('GetBestValue')
        base = Cell(i32(base))
        i = Cell(u32(u32(0)))
        _first_1 = True
        while True:
            if not _first_1:
                increment(i,1,False)
            _first_1 = False
            if not (i.v < BEST_VALUE_LENGTH): break
            if self._branch(0,((i.v == u32(0)) or (BEST_VALUE_LIST[i.v] <= base.v))):
                continue
            return BEST_VALUE_LIST[u32((i.v - u32(1)))]
        return BEST_VALUE_LIST[u32((BEST_VALUE_LENGTH - u32(1)))]
    
    def GetTwoFactors(self, res, base, dim, maxNum):
        'Source: MatmulTilingAlgorithm::GetTwoFactors; lines 95-122'
        self._visit('GetTwoFactors')
        base = Cell(i32(base))
        dim = Cell(i32(dim))
        maxNum = Cell(i32(maxNum))
        if self._branch(1,(dim.v == 1)):
            assign(ItemRef(res.v,0),i32(1))
            assign(ItemRef(res.v,1),i32(1))
            return
        assign(ItemRef(res.v,0),i32(0))
        assign(ItemRef(res.v,1),i32(0))
        cnt = Cell(i32(0))
        up = Cell(i32(i32((base.v + 1))))
        _first_2 = True
        while True:
            if not _first_2:
                increment(up,1,False)
            _first_2 = False
            if not ((up.v <= maxNum.v) and (up.v <= dim.v)): break
            if self._branch(2,(i32(cmod(dim.v,up.v)) == 0)):
                assign(ItemRef(res.v,increment(cnt,1,True)),i32(up.v))
                break
        down = Cell(i32(base.v))
        _first_3 = True
        while True:
            if not _first_3:
                increment(down,-1,False)
            _first_3 = False
            if not (down.v >= 1): break
            if self._branch(3,(i32(cmod(dim.v,down.v)) == 0)):
                assign(ItemRef(res.v,increment(cnt,1,True)),i32(down.v))
                if self._branch(4,(u64(cnt.v) == u64(cdiv(8,4)))):
                    break
    
    def GetABL1KAlignValue(self, kaAlignValue, kbAlignValue):
        'Source: MatmulTilingAlgorithm::GetABL1KAlignValue; lines 124-134'
        self._visit('GetABL1KAlignValue')
        assign(kaAlignValue,i32(1))
        assign(kbAlignValue,i32(1))
        if self._branch(5,((self.tilingIns_.aType_.dataType == DataType.DT_FLOAT) or (self.tilingIns_.bType_.dataType == DataType.DT_FLOAT))):
            assign(kaAlignValue,i32((2 if self.tilingIns_.aType_.isTrans else 1)))
            assign(kbAlignValue,i32((2 if (self.tilingIns_.aType_.isTrans or (not self.tilingIns_.bType_.isTrans)) else 1)))
    
    def GetL0StatusFromParasCombo(self, l0Status, parasCombo):
        'Source: MatmulTilingAlgorithm::GetL0StatusFromParasCombo; lines 136-151'
        self._visit('GetL0StatusFromParasCombo')
        parasCombo = Cell(parasCombo)
        l0Status.v.InitLoadStatus()
        kIdx = Cell(u64(u64(0)))
        assign(AttrRef(l0Status.v,'dbL0A'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'dbL0B'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'dbL0C'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxMk'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxNk'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxMn'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxAxisIdx'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxAxisNum'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxAxisPnt'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxN'),i32(parasCombo.v[increment(kIdx,1,True)]))
        assign(AttrRef(l0Status.v,'maxAxisPnt'),i32(min(l0Status.v.maxAxisPnt, l0Status.v.maxAxisNum)))
    
    def SetResFactors(self, resFactors, l0Status):
        'Source: MatmulTilingAlgorithm::SetResFactors; lines 153-163'
        self._visit('SetResFactors')
        assign(AttrRef(resFactors.v,'finalML0'),i32(l0Status.v.finalML0))
        assign(AttrRef(resFactors.v,'finalKL0'),i32(l0Status.v.finalKL0))
        assign(AttrRef(resFactors.v,'finalNL0'),i32(l0Status.v.finalNL0))
        assign(AttrRef(resFactors.v,'finalLoadSize'),i32(l0Status.v.finalLoadSize))
        assign(AttrRef(resFactors.v,'finalL0cUse'),f32(l0Status.v.finalL0cUse))
        assign(AttrRef(resFactors.v,'finalMte1Loop'),i32(l0Status.v.finalMte1Loop))
        assign(AttrRef(resFactors.v,'finalMul'),i32(l0Status.v.finalMul))
        assign(AttrRef(resFactors.v,'finalMte1Cycles'),i32(l0Status.v.finalMte1Cycles))
    
    def GetLoadSize(self, coreStatus, l0Status):
        'Source: MatmulTilingAlgorithm::GetLoadSize; lines 165-183'
        self._visit('GetLoadSize')
        al0FullLoad = Cell(bool((i64((i64(i32((coreStatus.v.m * coreStatus.v.k))) * i64(i32((C0_SIZE * C0_BYTE_SIZE))))) <= i64(self.tilingIns_.bufferPool_.l0ASize))))
        bl0FullLoad = Cell(bool((i64((i64(i32((coreStatus.v.n * coreStatus.v.k))) * i64(i32((C0_SIZE * C0_BYTE_SIZE))))) <= i64(self.tilingIns_.bufferPool_.l0BSize))))
        kFullLoad = Cell(bool((l0Status.v.kL0 >= coreStatus.v.k)))
        if self._branch(6,(al0FullLoad.v or bl0FullLoad.v)):
            return i32((coreStatus.v.m + coreStatus.v.n))
        else:
            if self._branch(7,kFullLoad.v):
                return min(i32((coreStatus.v.n + i32((MathUtil.CeilDivision(coreStatus.v.n, l0Status.v.nL0) * coreStatus.v.m)))), i32((coreStatus.v.m + i32((MathUtil.CeilDivision(coreStatus.v.m, l0Status.v.mL0) * coreStatus.v.n)))))
            else:
                return i32((i32((MathUtil.CeilDivision(coreStatus.v.m, l0Status.v.mL0) * coreStatus.v.n)) + i32((MathUtil.CeilDivision(coreStatus.v.n, l0Status.v.nL0) * coreStatus.v.m))))
    
    def CheckBaseMNKL1Size(self, singleCoreStatus):
        'Source: MatmulTilingAlgorithm::CheckBaseMNKL1Size; lines 185-210'
        self._visit('CheckBaseMNKL1Size')
        l0Status = AttrRef(singleCoreStatus.v,'l0Status')
        a1Length = Cell(i32(i32((i32((i32((l0Status.v.mL0 * l0Status.v.kL0)) * C0_SIZE)) * C0_BYTE_SIZE))))
        b1Length = Cell(i32(i32((i32((i32((l0Status.v.nL0 * l0Status.v.kL0)) * C0_SIZE)) * C0_BYTE_SIZE))))
        biasLength = Cell(i32(i32((u32((u32(i32((l0Status.v.nL0 * C0_SIZE))) * DTYPE_BYTE_TAB[self.tilingIns_.biasType_.dataType])) if ((self.tilingIns_.socVersion == 'ASCEND910B') or (self.tilingIns_.socVersion == 'ASCEND310B')) else u32(0)))))
        dequantSize = Cell(i32(0))
        if self._branch(8,(self.tilingIns_.deqType == DequantType.TENSOR)):
            assign(dequantSize,i32(i32(u64((u64(i32((l0Status.v.nL0 * C0_SIZE))) * UINT64_TYPES)))))
        if self._branch(9,(self.tilingIns_.aType_.pos == TPosition.TSCM)):
            assign(a1Length,i32(0))
        if self._branch(10,(self.tilingIns_.bType_.pos == TPosition.TSCM)):
            assign(b1Length,i32(0))
        if self._branch(11,((self.tilingIns_.biasType_.pos == TPosition.TSCM) or (not self.tilingIns_.isBias))):
            assign(biasLength,i32(0))
        totalLength = Cell(i32(i32((i32((i32((a1Length.v + b1Length.v)) + biasLength.v)) + dequantSize.v))))
        return (totalLength.v <= self.tilingIns_.bufferPool_.l1Size)
    
    def CheckK0Align(self, k0):
        'Source: MatmulTilingAlgorithm::CheckK0Align; lines 212-221'
        self._visit('CheckK0Align')
        k0 = Cell(i32(k0))
        if self._branch(12,((((self.tilingIns_.aType_.dataType == DataType.DT_FLOAT) and (self.tilingIns_.aType_.type == CubeFormat.NZ)) and self.tilingIns_.aType_.isTrans) or (((self.tilingIns_.bType_.dataType == DataType.DT_FLOAT) and (self.tilingIns_.bType_.type == CubeFormat.NZ)) and (not self.tilingIns_.bType_.isTrans)))):
            return (i32(cmod(k0.v,NUM_TWO)) == 0)
        return True
    
    def GetFinalMkn(self, singleCoreStatus, coreStatus, k0, majorDimFactor, minorDimFactor):
        'Source: MatmulTilingAlgorithm::GetFinalMkn; lines 223-281'
        self._visit('GetFinalMkn')
        if self._branch(13,(k0.v == 0)):
            return
        l0Status = AttrRef(singleCoreStatus.v,'l0Status')
        if self._branch(14,(l0Status.v.maxAxisIdx == 0)):
            assign(AttrRef(l0Status.v,'mL0'),i32(majorDimFactor.v))
            assign(AttrRef(l0Status.v,'nL0'),i32(minorDimFactor.v))
        else:
            assign(AttrRef(l0Status.v,'mL0'),i32(minorDimFactor.v))
            assign(AttrRef(l0Status.v,'nL0'),i32(majorDimFactor.v))
        assign(AttrRef(l0Status.v,'kL0'),i32(k0.v))
        tmpL0cUse = Cell(f32(f32(float((float((float(i32((i32((i32((i32((l0Status.v.mL0 * l0Status.v.nL0)) * l0Status.v.dbL0C)) * MIN_FRACTAL_SIZE)) * FP32_BYTES))) * 1)) / float(self.tilingIns_.bufferPool_.l0CSize))))))
        tmpMte1Cycle = Cell(i32(i32((max(i32((2 * 3)), i32(cdiv(i32((i32((i32((l0Status.v.mL0 * l0Status.v.kL0)) * C0_SIZE)) * C0_BYTE_SIZE)),MTE1_L0A_BANDWIDTH))) + max(i32((2 * 3)), i32(cdiv(i32((i32((i32((l0Status.v.kL0 * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)),MTE1_L0B_BANDWIDTH)))))))
        tmpMadCycle = Cell(i32(i32((i32((l0Status.v.mL0 * l0Status.v.kL0)) * l0Status.v.nL0))))
        tmpLoadSize = Cell(i32(self.GetLoadSize(coreStatus, l0Status)))
        tmpMte1Loop = Cell(i32(i32(((l0Status.v.kL0 if (l0Status.v.nL0 != 1) else 1) + (l0Status.v.mL0 if (l0Status.v.kL0 != 1) else 1)))))
        condition1 = Cell(bool((l0Status.v.finalML0 == 0)))
        condition2 = Cell(bool(((tmpLoadSize.v < l0Status.v.finalLoadSize) or ((tmpMte1Cycle.v < tmpMadCycle.v) and (not l0Status.v.updateUsingMte1)))))
        condition3 = Cell(bool((((tmpLoadSize.v == l0Status.v.finalLoadSize) and (tmpMadCycle.v > l0Status.v.finalMul)) and (f32((f32(tmpMadCycle.v) * tmpL0cUse.v)) >= f32((f32(l0Status.v.finalMul) * l0Status.v.finalL0cUse))))))
        condition4 = Cell(bool((((tmpMadCycle.v == l0Status.v.finalMul) and (tmpLoadSize.v == l0Status.v.finalLoadSize)) and (tmpMte1Loop.v < l0Status.v.finalMte1Loop))))
        condition5 = Cell(bool((((tmpMte1Cycle.v < tmpMadCycle.v) and l0Status.v.updateUsingMte1) or (not l0Status.v.updateUsingMte1))))
        condition6 = Cell(bool(self.CheckBaseMNKL1Size(singleCoreStatus)))
        lastReduceDim = Cell(i32((FLOAT32_REDUCE_BLOCK_SIZE if ((self.tilingIns_.aType_.dataType == DataType.DT_FLOAT) or (self.tilingIns_.bType_.dataType == DataType.DT_FLOAT)) else REDUCE_BLOCK_SIZE)))
        condition7 = Cell(bool(((self.tilingIns_.baseN != i32((-1))) or (not ((coreStatus.v.n >= lastReduceDim.v) and (l0Status.v.nL0 < lastReduceDim.v))))))
        condition8 = Cell(bool(self.CheckK0Align(l0Status.v.kL0)))
        validL0 = Cell(bool((((((((condition1.v or condition2.v) or condition3.v) or condition4.v) and condition5.v) and condition6.v) and condition7.v) and condition8.v)))
        if self._branch(15,validL0.v):
            assign(AttrRef(l0Status.v,'finalML0'),i32(l0Status.v.mL0))
            assign(AttrRef(l0Status.v,'finalKL0'),i32(l0Status.v.kL0))
            assign(AttrRef(l0Status.v,'finalNL0'),i32(l0Status.v.nL0))
            assign(AttrRef(l0Status.v,'finalLoadSize'),i32(tmpLoadSize.v))
            assign(AttrRef(l0Status.v,'finalL0cUse'),f32(tmpL0cUse.v))
            assign(AttrRef(l0Status.v,'finalMul'),i32(tmpMadCycle.v))
            assign(AttrRef(l0Status.v,'finalMte1Cycles'),i32(tmpMte1Cycle.v))
            assign(AttrRef(l0Status.v,'finalMte1Loop'),i32(tmpMte1Loop.v))
            assign(AttrRef(l0Status.v,'updateUsingMte1'),bool((l0Status.v.updateUsingMte1 or (tmpMte1Cycle.v < tmpMadCycle.v))))
    
    def GetL0bAlign(self, factors):
        'Source: MatmulTilingAlgorithm::GetL0bAlign; lines 283-292'
        self._visit('GetL0bAlign')
        alignSize = Cell(i32(2))
        if self._branch(16,((self.tilingIns_.bType_.dataType == DataType.DT_INT8) or (self.tilingIns_.bType_.dataType == DataType.DT_INT4))):
            for _item_4 in factors.v:
                num = Cell(_item_4)
                assign(num,i32(MathUtil.Align(num.v, alignSize.v)))
        return
    
    def GetL0FactorsCand(self, resFactors, coreStatus, singleCoreStatus, parasCombo, param):
        'Source: MatmulTilingAlgorithm::GetL0FactorsCand; lines 294-464'
        self._visit('GetL0FactorsCand')
        parasCombo = Cell(parasCombo)
        param.v
        l0Status = AttrRef(singleCoreStatus.v,'l0Status')
        self.GetL0StatusFromParasCombo(l0Status, parasCombo.v)
        l0bAlignSize = Cell(i32(1))
        if self._branch(17,((self.tilingIns_.bType_.dataType == DataType.DT_INT8) or (self.tilingIns_.bType_.dataType == DataType.DT_INT4))):
            assign(l0bAlignSize,i32(L0B_ALIGN_SIZE))
        majorDim = Cell(i32(coreStatus.v.m))
        minorDim = Cell(i32(MathUtil.Align(coreStatus.v.n, l0bAlignSize.v)))
        majorDimK = Cell(i32(l0Status.v.maxMk))
        minorDimK = Cell(i32(l0Status.v.maxNk))
        maxN = Cell(i32(l0Status.v.maxN))
        dimFactors = Cell([0, 0])
        if self._branch(18,(l0Status.v.maxAxisIdx != 0)):
            assign(majorDim,i32(MathUtil.Align(coreStatus.v.n, l0bAlignSize.v)))
            assign(minorDim,i32(coreStatus.v.m))
            assign(majorDimK,i32(l0Status.v.maxNk))
            assign(minorDimK,i32(l0Status.v.maxMk))
        majorDimFactors = Cell([clone(0) for _ in range(u64(DIM_FACTOR_LENGTH))])
        if self._branch(19,((self.tilingIns_.baseN != i32((-1))) and (l0Status.v.maxAxisIdx != 0))):
            assign(ItemRef(majorDimFactors.v,u64(0)),i32(MathUtil.CeilDivision(self.tilingIns_.baseN, C0_SIZE)))
        else:
            if self._branch(20,((self.tilingIns_.baseM != i32((-1))) and (l0Status.v.maxAxisIdx == 0))):
                assign(ItemRef(majorDimFactors.v,u64(0)),i32(MathUtil.CeilDivision(self.tilingIns_.baseM, C0_SIZE)))
            else:
                if self._branch(21,(((l0Status.v.maxAxisIdx != 0) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                    self.GetTwoFactors(dimFactors, min(l0Status.v.maxAxisPnt, maxN.v), majorDim.v, min(l0Status.v.maxAxisNum, maxN.v))
                else:
                    self.GetTwoFactors(dimFactors, l0Status.v.maxAxisPnt, majorDim.v, l0Status.v.maxAxisNum)
                assign(ItemRef(majorDimFactors.v,u64(0)),i32(dimFactors.v[0]))
                assign(ItemRef(majorDimFactors.v,u64(1)),i32(dimFactors.v[1]))
                majorAmend = Cell(i32(self.GetBestValue(majorDim.v)))
                if self._branch(22,(((l0Status.v.maxAxisIdx != 0) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                    self.GetTwoFactors(dimFactors, min(l0Status.v.maxAxisPnt, maxN.v), majorAmend.v, min(l0Status.v.maxAxisNum, maxN.v))
                else:
                    self.GetTwoFactors(dimFactors, l0Status.v.maxAxisPnt, majorAmend.v, l0Status.v.maxAxisNum)
                assign(ItemRef(majorDimFactors.v,u64(IDX_TWO)),i32(dimFactors.v[0]))
                assign(ItemRef(majorDimFactors.v,u64(IDX_THREE)),i32(dimFactors.v[1]))
                if self._branch(23,(l0Status.v.maxAxisIdx != 0)):
                    self.GetL0bAlign(majorDimFactors)
        majorDimFactors.v.sort(reverse=True)
        vector_erase(majorDimFactors.v,clone(vector_unique(majorDimFactors.v)), clone(len(majorDimFactors.v)))
        for _item_5 in majorDimFactors.v:
            majorDimFactor = Cell(_item_5)
            if self._branch(24,((((majorDimFactor.v == 0) or (majorDimFactor.v > l0Status.v.maxMn)) or (majorDimFactor.v > majorDimK.v)) or (majorDimFactor.v > majorDim.v))):
                continue
            minorFactorMax = Cell(i32(min(i32(cdiv(l0Status.v.maxMn,majorDimFactor.v)), minorDimK.v)))
            minorDimFactors = Cell([clone(0) for _ in range(u64(DIM_FACTOR_LENGTH))])
            if self._branch(25,((self.tilingIns_.baseN != i32((-1))) and (l0Status.v.maxAxisIdx == 0))):
                assign(ItemRef(minorDimFactors.v,u64(0)),i32(MathUtil.CeilDivision(self.tilingIns_.baseN, C0_SIZE)))
            else:
                if self._branch(26,((self.tilingIns_.baseM != i32((-1))) and (l0Status.v.maxAxisIdx != 0))):
                    assign(ItemRef(minorDimFactors.v,u64(0)),i32(MathUtil.CeilDivision(self.tilingIns_.baseM, C0_SIZE)))
                else:
                    if self._branch(27,(((l0Status.v.maxAxisIdx == 0) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                        self.GetTwoFactors(dimFactors, min(minorFactorMax.v, maxN.v), minorDim.v, min(minorFactorMax.v, maxN.v))
                    else:
                        self.GetTwoFactors(dimFactors, minorFactorMax.v, minorDim.v, minorFactorMax.v)
                    assign(ItemRef(minorDimFactors.v,u64(0)),i32(dimFactors.v[0]))
                    assign(ItemRef(minorDimFactors.v,u64(1)),i32(dimFactors.v[1]))
                    minorAmend = Cell(i32(self.GetBestValue(minorDim.v)))
                    if self._branch(28,(((l0Status.v.maxAxisIdx == 0) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                        self.GetTwoFactors(dimFactors, min(minorFactorMax.v, maxN.v), minorAmend.v, min(minorFactorMax.v, maxN.v))
                    else:
                        self.GetTwoFactors(dimFactors, minorFactorMax.v, minorAmend.v, minorFactorMax.v)
                    assign(ItemRef(minorDimFactors.v,u64(IDX_TWO)),i32(dimFactors.v[0]))
                    assign(ItemRef(minorDimFactors.v,u64(IDX_THREE)),i32(dimFactors.v[1]))
                    if self._branch(29,(l0Status.v.maxAxisIdx == 0)):
                        self.GetL0bAlign(minorDimFactors)
            minorDimFactors.v.sort(reverse=True)
            vector_erase(minorDimFactors.v,clone(vector_unique(minorDimFactors.v)), clone(len(minorDimFactors.v)))
            for _item_6 in minorDimFactors.v:
                minorDimFactor = Cell(_item_6)
                if self._branch(30,(((((minorDimFactor.v == 0) or (i32((minorDimFactor.v * majorDimFactor.v)) > l0Status.v.maxMn)) or (minorDimFactor.v > minorDimK.v)) or (minorDimFactor.v > minorDim.v)) or (minorDimFactor.v > majorDimK.v))):
                    continue
                maxN0 = Cell(i32(64))
                if self._branch(31,((self.tilingIns_.socVersion == 'ASCEND910B') or (self.tilingIns_.socVersion == 'ASCEND310B'))):
                    assign(maxN,i32(i32(cdiv(i32(cdiv(i32(cdiv(self.tilingIns_.bufferPool_.btSize,C0_SIZE)),FP32_BYTES)),l0Status.v.dbL0C))))
                if self._branch(32,(l0Status.v.maxAxisIdx != 0)):
                    if self._branch(33,(((majorDimFactor.v > maxN0.v) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                        continue
                else:
                    if self._branch(34,(((minorDimFactor.v > maxN0.v) and self.tilingIns_.isSupportL0c2Out) and self.tilingIns_.isBias)):
                        continue
                k0Max = Cell(i32(min(i32(cdiv(majorDimK.v,majorDimFactor.v)), i32(cdiv(minorDimK.v,minorDimFactor.v)))))
                k0Factors = Cell([clone(0) for _ in range(u64(DIM_FACTOR_LENGTH))])
                self.GetTwoFactors(dimFactors, k0Max.v, coreStatus.v.k, k0Max.v)
                assign(ItemRef(k0Factors.v,u64(0)),i32(dimFactors.v[0]))
                assign(ItemRef(k0Factors.v,u64(1)),i32(dimFactors.v[1]))
                kAmend = Cell(i32(self.GetBestValue(coreStatus.v.k)))
                self.GetTwoFactors(dimFactors, k0Max.v, kAmend.v, l0Status.v.maxAxisNum)
                assign(ItemRef(k0Factors.v,u64(IDX_TWO)),i32(dimFactors.v[0]))
                assign(ItemRef(k0Factors.v,u64(IDX_THREE)),i32(dimFactors.v[1]))
                k0Factors.v.sort(reverse=True)
                vector_erase(k0Factors.v,clone(vector_unique(k0Factors.v)), clone(len(k0Factors.v)))
                for _item_7 in k0Factors.v:
                    k0 = Cell(_item_7)
                    if self._branch(35,(((k0.v == 0) or (i32((minorDimFactor.v * k0.v)) > minorDimK.v)) or (i32((majorDimFactor.v * k0.v)) > majorDimK.v))):
                        continue
                    if self._branch(36,(self.tilingIns_.aType_.dataType == DataType.DT_FLOAT)):
                        mL0 = Cell(i32(majorDimFactor.v))
                        nL0 = Cell(i32(minorDimFactor.v))
                        if self._branch(37,(l0Status.v.maxAxisIdx != 0)):
                            assign(nL0,i32(majorDimFactor.v))
                            assign(mL0,i32(minorDimFactor.v))
                        l0aBufferSize = Cell(i32((i32((i32((i32((i32((MathUtil.Align(k0.v, 2) * C0_BYTE_SIZE)) * mL0.v)) * C0_SIZE)) * DB_ON)) if self.tilingIns_.aType_.isTrans else i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * mL0.v)) * C0_SIZE)) * DB_ON)))))
                        l0bBufferSize = Cell(i32((i32((i32((i32((i32((MathUtil.Align(k0.v, 2) * C0_BYTE_SIZE)) * nL0.v)) * C0_SIZE)) * DB_ON)) if (self.tilingIns_.aType_.isTrans or (not self.tilingIns_.bType_.isTrans)) else i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * nL0.v)) * C0_SIZE)) * DB_ON)))))
                        if self._branch(38,((l0aBufferSize.v > self.tilingIns_.bufferPool_.l0ASize) or (l0bBufferSize.v > self.tilingIns_.bufferPool_.l0BSize))):
                            continue
                    else:
                        if self._branch(39,((self.tilingIns_.aType_.dataType == DataType.DT_INT8) or (self.tilingIns_.aType_.dataType == DataType.DT_INT4))):
                            mL0_2 = Cell(i32(majorDimFactor.v))
                            nL0_2 = Cell(i32(minorDimFactor.v))
                            if self._branch(40,(l0Status.v.maxAxisIdx != 0)):
                                assign(nL0_2,i32(majorDimFactor.v))
                                assign(mL0_2,i32(minorDimFactor.v))
                            l0aBufferSize_2 = Cell(i32((i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * MathUtil.Align(mL0_2.v, 2))) * C0_SIZE)) * DB_ON)) if self.tilingIns_.aType_.isTrans else i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * mL0_2.v)) * C0_SIZE)) * DB_ON)))))
                            l0bBufferSize_2 = Cell(i32((i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * nL0_2.v)) * C0_SIZE)) * DB_ON)) if self.tilingIns_.bType_.isTrans else i32((i32((i32((i32((k0.v * C0_BYTE_SIZE)) * MathUtil.Align(nL0_2.v, 2))) * C0_SIZE)) * DB_ON)))))
                            if self._branch(41,((l0aBufferSize_2.v > self.tilingIns_.bufferPool_.l0ASize) or (l0bBufferSize_2.v > self.tilingIns_.bufferPool_.l0BSize))):
                                continue
                    self.GetFinalMkn(singleCoreStatus, coreStatus, k0, majorDimFactor, minorDimFactor)
        if self._branch(42,(((l0Status.v.finalML0 != 0) and (l0Status.v.finalKL0 != 0)) and (l0Status.v.finalNL0 != 0))):
            self.SetResFactors(resFactors, l0Status)
    
    def GetParasCombo(self, index, param):
        'Source: MatmulTilingAlgorithm::GetParasCombo; lines 466-488'
        self._visit('GetParasCombo')
        param.v
        parasComboMap = Cell({})
        mnMax = Cell(i32(i32(cdiv(i32(cdiv(self.tilingIns_.bufferPool_.l0CSize,i32((C0_SIZE * C0_SIZE)))),FP32_BYTES))))
        maxN = Cell(i32(64))
        if self._branch(43,((self.tilingIns_.socVersion == 'ASCEND910B') or (self.tilingIns_.socVersion == 'ASCEND310B'))):
            assign(maxN,i32(i32(cdiv(i32(cdiv(self.tilingIns_.bufferPool_.btSize,C0_SIZE)),FP32_BYTES))))
        biasBt = Cell(bool((self.tilingIns_.isSupportL0c2Out and self.tilingIns_.isBias)))
        leftSize = Cell(i32(min(self.tilingIns_.bufferPool_.l1Size, i32(cdiv(self.tilingIns_.bufferPool_.l0ASize,self.dbL0A_)))))
        rightSize = Cell(i32(min(self.tilingIns_.bufferPool_.l1Size, i32(cdiv(self.tilingIns_.bufferPool_.l0BSize,self.dbL0B_)))))
        maxMk = Cell(i32((64 if (self.tilingIns_.aType_.pos == TPosition.TSCM) else i32(cdiv(i32(cdiv(leftSize.v,C0_SIZE)),C0_BYTE_SIZE)))))
        maxNK = Cell(i32((64 if (self.tilingIns_.bType_.pos == TPosition.TSCM) else i32(cdiv(i32(cdiv(rightSize.v,C0_SIZE)),C0_BYTE_SIZE)))))
        comboZero = Cell(MKNParasCombo([2, 2, 2, maxMk.v, maxNK.v, i32(cdiv(mnMax.v,DB_ON)), 0, 64, 8, (i32(cdiv(maxN.v,DB_ON)) if biasBt.v else 64)]))
        comboOne = Cell(MKNParasCombo([self.dbL0A_, self.dbL0B_, 1, maxMk.v, maxNK.v, mnMax.v, 0, 64, 11, (maxN.v if biasBt.v else 64)]))
        assign(parasComboMap,dict([[0, comboZero.v], [1, comboOne.v]]),copy=True)
        return clone(parasComboMap.v[index.v])
    
    def GetL0cDB(self, resFactors, coreStatus, l0Status):
        'Source: MatmulTilingAlgorithm::GetL0cDB; lines 490-540'
        self._visit('GetL0cDB')
        dbAOnBOnCOnIdx = Cell(i32(0))
        dbAOnBOnCOffIdx = Cell(i32(1))
        m0L0cDbOn = Cell(i32(resFactors.v[dbAOnBOnCOnIdx.v].finalML0))
        k0L0cDbOn = Cell(i32(resFactors.v[dbAOnBOnCOnIdx.v].finalKL0))
        n0L0cDbOn = Cell(i32(resFactors.v[dbAOnBOnCOnIdx.v].finalNL0))
        loadSizeL0cDbOn = Cell(i32(resFactors.v[dbAOnBOnCOnIdx.v].finalLoadSize))
        mte1CyclesL0cDbOn = Cell(i32(resFactors.v[dbAOnBOnCOnIdx.v].finalMte1Cycles))
        m0L0cDbOff = Cell(i32(resFactors.v[dbAOnBOnCOffIdx.v].finalML0))
        k0L0cDbOff = Cell(i32(resFactors.v[dbAOnBOnCOffIdx.v].finalKL0))
        n0L0cDbOff = Cell(i32(resFactors.v[dbAOnBOnCOffIdx.v].finalNL0))
        loadSizeL0cDbOff = Cell(i32(resFactors.v[dbAOnBOnCOffIdx.v].finalLoadSize))
        mte1CyclesL0cDbOff = Cell(i32(resFactors.v[dbAOnBOnCOffIdx.v].finalMte1Cycles))
        mte3CostDbOn = Cell(i32(i32(cdiv(i32((i32((i32((i32((m0L0cDbOn.v * n0L0cDbOn.v)) * MIN_FRACTAL_SIZE)) * FP16_BYTES)) * 1)),MTE1_FIXPIPE_BANDWIDTH))))
        mte3CostDbOff = Cell(i32(i32(cdiv(i32((i32((i32((i32((m0L0cDbOff.v * n0L0cDbOff.v)) * MIN_FRACTAL_SIZE)) * FP16_BYTES)) * 1)),MTE1_FIXPIPE_BANDWIDTH))))
        madCylesDbOn = Cell(i32(max(i32((i32((m0L0cDbOn.v * k0L0cDbOn.v)) * n0L0cDbOn.v)), i32(float((float(mte1CyclesL0cDbOn.v) * 0.69999999999999996))))))
        madCylesDbOff = Cell(i32(max(i32((i32((m0L0cDbOff.v * k0L0cDbOff.v)) * n0L0cDbOff.v)), i32(float((float(mte1CyclesL0cDbOff.v) * 0.69999999999999996))))))
        dbOnPipeTime = Cell(i32(i32((i32((MathUtil.CeilDivision(coreStatus.v.m, m0L0cDbOn.v) * MathUtil.CeilDivision(coreStatus.v.n, n0L0cDbOn.v))) * i32((i32((i32((MathUtil.CeilDivision(coreStatus.v.k, k0L0cDbOn.v) - 1)) * madCylesDbOn.v)) + max(madCylesDbOn.v, mte3CostDbOn.v)))))))
        dbOffPipeTime = Cell(i32(i32((i32((MathUtil.CeilDivision(coreStatus.v.m, m0L0cDbOff.v) * MathUtil.CeilDivision(coreStatus.v.n, n0L0cDbOff.v))) * i32((i32((MathUtil.CeilDivision(coreStatus.v.k, k0L0cDbOff.v) * madCylesDbOff.v)) + mte3CostDbOff.v))))))
        assign(dbOnPipeTime,i32((2147483647 if (dbOnPipeTime.v == 0) else dbOnPipeTime.v)))
        assign(dbOffPipeTime,i32((2147483647 if (dbOffPipeTime.v == 0) else dbOffPipeTime.v)))
        if self._branch(44,((dbOffPipeTime.v < dbOnPipeTime.v) or (loadSizeL0cDbOff.v < loadSizeL0cDbOn.v))):
            assign(AttrRef(l0Status.v,'dbL0C'),i32(1))
            assign(AttrRef(l0Status.v,'dbL0A'),i32(self.dbL0A_))
            assign(AttrRef(l0Status.v,'dbL0B'),i32(self.dbL0B_))
            assign(AttrRef(l0Status.v,'mL0'),i32(m0L0cDbOff.v))
            assign(AttrRef(l0Status.v,'kL0'),i32(k0L0cDbOff.v))
            assign(AttrRef(l0Status.v,'nL0'),i32(n0L0cDbOff.v))
        else:
            assign(AttrRef(l0Status.v,'dbL0C'),i32(DB_ON))
            assign(AttrRef(l0Status.v,'dbL0A'),i32(self.dbL0A_))
            assign(AttrRef(l0Status.v,'dbL0B'),i32(self.dbL0B_))
            assign(AttrRef(l0Status.v,'mL0'),i32(m0L0cDbOn.v))
            assign(AttrRef(l0Status.v,'kL0'),i32(k0L0cDbOn.v))
            assign(AttrRef(l0Status.v,'nL0'),i32(n0L0cDbOn.v))
    
    def GetL0Factors(self, opType, param, coreStatus, singleCoreStatus):
        'Source: MatmulTilingAlgorithm::GetL0Factors; lines 542-573'
        self._visit('GetL0Factors')
        opType.v
        l0Status = AttrRef(singleCoreStatus.v,'l0Status')
        if self._branch(45,self.tilingIns_.isBias):
            assign(AttrRef(l0Status.v,'dtypeBias'),i32(i32(DTYPE_BYTE_TAB[self.tilingIns_.biasType_.dataType])))
        resFactors = Cell([L0Factors(), L0Factors()])
        i = Cell(i32(0))
        _first_8 = True
        while True:
            if not _first_8:
                increment(i,1,False)
            _first_8 = False
            if not (i.v < L0PARAS_COMBO_LEN): break
            if self._branch(46,((i.v == 0) and (self.cfg.l0cDB == DB_OFF))):
                continue
            mknParasCombo = Cell(self.GetParasCombo(i, param))
            j = Cell(i32(0))
            _first_9 = True
            while True:
                if not _first_9:
                    increment(j,1,False)
                _first_9 = False
                if not (j.v < L0PARAS_COMBO_LEN): break
                assign(ItemRef(mknParasCombo.v.parasCombo,IDX_SIX),i32(j.v))
                self.GetL0FactorsCand(ItemRef(resFactors.v,i.v), coreStatus, singleCoreStatus, mknParasCombo.v.parasCombo, param)
        if self._branch(47,(self.cfg.l0cDB == DB_OFF)):
            assign(AttrRef(l0Status.v,'dbL0C'),i32(DB_OFF))
            assign(AttrRef(l0Status.v,'dbL0A'),i32(self.dbL0A_))
            assign(AttrRef(l0Status.v,'dbL0B'),i32(self.dbL0B_))
            assign(AttrRef(l0Status.v,'mL0'),i32(resFactors.v[1].finalML0))
            assign(AttrRef(l0Status.v,'kL0'),i32(resFactors.v[1].finalKL0))
            assign(AttrRef(l0Status.v,'nL0'),i32(resFactors.v[1].finalNL0))
        else:
            self.GetL0cDB(resFactors, coreStatus, l0Status)
    
    def GetL1Size(self, l1Status, l0Status):
        'Source: MatmulTilingAlgorithm::GetL1Size; lines 575-631'
        self._visit('GetL1Size')
        curAL1Size = Cell(i32(0))
        curBL1Size = Cell(i32(0))
        channelWiseL1Size = Cell(i32(0))
        aL1Const = Cell(i32(i32((i32((C0_SIZE * C0_BYTE_SIZE)) * l1Status.v.dbAL1))))
        if self._branch(48,(self.tilingIns_.aType_.dataType == DataType.DT_FLOAT)):
            assign(aL1Const,i32(i32((aL1Const.v * NUM_TWO))))
        bL1Const = Cell(i32((i32((i32((i32((C0_SIZE * i32(cdiv(C0_BYTE_SIZE,8)))) * 5)) * l1Status.v.dbBL1)) if self.tilingIns_.isSparse_ else i32((i32((C0_SIZE * C0_BYTE_SIZE)) * l1Status.v.dbBL1)))))
        if self._branch(49,(self.tilingIns_.bType_.dataType == DataType.DT_FLOAT)):
            assign(bL1Const,i32(i32((bL1Const.v * NUM_TWO))))
        channelWiseL1Const = Cell(i32(i32((i32((i32((l1Status.v.channelWiseTimes * C0_SIZE)) * l1Status.v.dbBL1)) * l0Status.v.dtypeBias))))
        dequantSize = Cell(i32(0))
        kaAlignValue = Cell(i32(1))
        kbAlignValue = Cell(i32(1))
        self.GetABL1KAlignValue(kaAlignValue, kbAlignValue)
        if self._branch(50,(((not MathUtil.CheckMulOverflow(l1Status.v.mAL1, l0Status.v.mL0, curAL1Size)) or (not MathUtil.CheckMulOverflow(curAL1Size.v, aL1Const.v, curAL1Size))) or (not MathUtil.CheckMulOverflow(curAL1Size.v, MathUtil.Align(l1Status.v.kAL1, kaAlignValue.v), curAL1Size)))):
            return 0
        if self._branch(51,(((not MathUtil.CheckMulOverflow(l1Status.v.nBL1, l0Status.v.nL0, curBL1Size)) or (not MathUtil.CheckMulOverflow(curBL1Size.v, bL1Const.v, curBL1Size))) or (not MathUtil.CheckMulOverflow(curBL1Size.v, MathUtil.Align(l1Status.v.kBL1, kbAlignValue.v), curBL1Size)))):
            return 0
        if self._branch(52,(l1Status.v.channelWiseTimes > 0)):
            if self._branch(53,((not MathUtil.CheckMulOverflow(l1Status.v.nBL1, l0Status.v.nL0, channelWiseL1Size)) or (not MathUtil.CheckMulOverflow(channelWiseL1Size.v, channelWiseL1Const.v, channelWiseL1Size)))):
                return 0
        if self._branch(54,(self.tilingIns_.deqType == DequantType.TENSOR)):
            assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
        if self._branch(55,(self.tilingIns_.aType_.pos == TPosition.TSCM)):
            assign(curAL1Size,i32(0))
        if self._branch(56,(self.tilingIns_.bType_.pos == TPosition.TSCM)):
            assign(curBL1Size,i32(0))
        if self._branch(57,(self.tilingIns_.biasType_.pos == TPosition.TSCM)):
            assign(channelWiseL1Size,i32(0))
        totalSize = Cell(i64(i64((i64((i64((i64(curAL1Size.v) + i64(curBL1Size.v))) + i64(channelWiseL1Size.v))) + i64(dequantSize.v)))))
        return (2147483647 if (totalSize.v > i64(2147483647)) else i32(totalSize.v))
    
    def CalL1MaxLen(self, resL1Size, l1Status, l0Status, alignValue, axisName):
        'Source: MatmulTilingAlgorithm::CalL1MaxLen; lines 633-655'
        self._visit('CalL1MaxLen')
        resL1Size = Cell(i32(resL1Size))
        alignValue = Cell(i32(alignValue))
        axisName = Cell(axisName)
        axisMaxLen = Cell(i32(1))
        if self._branch(58,(axisName.v == L1TilingType.KAL1_16)):
            assign(axisMaxLen,i32(i32(cdiv(resL1Size.v,i32((i32((i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * l1Status.v.dbAL1)) * C0_SIZE)) * C0_BYTE_SIZE))))))
        if self._branch(59,(axisName.v == L1TilingType.KBL1_16)):
            assign(axisMaxLen,i32(i32(cdiv(resL1Size.v,i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * l1Status.v.dbBL1)) * C0_SIZE)) * C0_BYTE_SIZE))))))
        assign(axisMaxLen,i32(MathUtil.AlignDown(axisMaxLen.v, alignValue.v)))
        if self._branch(60,(axisName.v == L1TilingType.M_AL1)):
            assign(axisMaxLen,i32(i32(cdiv(resL1Size.v,i32((i32((i32((i32((MathUtil.Align(l1Status.v.kAL1, alignValue.v) * l0Status.v.mL0)) * l1Status.v.dbAL1)) * C0_SIZE)) * C0_BYTE_SIZE))))))
        if self._branch(61,(axisName.v == L1TilingType.N_BL1)):
            assign(axisMaxLen,i32(i32(cdiv(resL1Size.v,i32((i32((i32((i32((i32((MathUtil.Align(l1Status.v.kBL1, alignValue.v) * l0Status.v.nL0)) * l1Status.v.dbBL1)) * C0_SIZE)) * C0_BYTE_SIZE)) + i32((i32((i32((l1Status.v.channelWiseTimes * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE))))))))
        return axisMaxLen.v
    
    def GetNearestFactor(self, base, factor, capValue):
        'Source: MatmulTilingAlgorithm::GetNearestFactor; lines 662-673'
        self._visit('GetNearestFactor')
        capValue = Cell(i32(capValue))
        if self._branch(62,(not self.cfg.factorSplit)):
            return
        if self._branch(63,(capValue.v == 2147483647)):
            assign(capValue,i32(base.v))
        while ((factor.v > capValue.v) or ((factor.v > 0) and (i32(cmod(base.v,factor.v)) != 0))):
            increment(factor,-1,True)
    
    def L1StatusAl1FullLoad(self, coreStatus, l0Status, l1Status, res):
        'Source: MatmulTilingAlgorithm::L1StatusAl1FullLoad; lines 675-747'
        self._visit('L1StatusAl1FullLoad')
        res = Cell(res)
        if self._branch(64,(self.tilingIns_.bType_.pos == TPosition.TSCM)):
            return
        mRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.m, l0Status.v.mL0)))
        nRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.n, l0Status.v.nL0)))
        kaAlignValue = Cell(i32(1))
        kbAlignValue = Cell(i32(1))
        self.GetABL1KAlignValue(kaAlignValue, kbAlignValue)
        assign(AttrRef(l1Status.v,'kAL1'),i32(i32((MathUtil.CeilDivision(l1Status.v.kAL1, l0Status.v.kL0) * l0Status.v.kL0))))
        curL1Size = Cell(i32(self.GetL1Size(l1Status, l0Status)))
        a1Length = Cell(i32(self.GetAL1UbSize(l1Status, l0Status)))
        if self._branch(65,(((curL1Size.v > 0) and (curL1Size.v <= self.tilingIns_.bufferPool_.l1Size)) and (a1Length.v < self.tilingIns_.bufferPool_.ubSize))):
            assign(AttrRef(l1Status.v,'aL1FullLoad'),bool(True))
            assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((max(MathUtil.Align(coreStatus.v.k, kaAlignValue.v), MathUtil.Align(l1Status.v.kAL1, kaAlignValue.v)) * max(i32((l1Status.v.mAL1 * l0Status.v.mL0)), coreStatus.v.m))) * C0_SIZE)) * C0_BYTE_SIZE))))
            if self._branch(66,(self.tilingIns_.aType_.pos == TPosition.TSCM)):
                assign(AttrRef(l1Status.v,'bL1Size'),i32(self.tilingIns_.bufferPool_.l1Size))
            else:
                assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.aL1Size))))
            if self._branch(67,(self.cfg.l1DB == DB_ON)):
                assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_ON))
                if self._branch(68,(self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size)):
                    assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_OFF))
            biasSize = Cell(i32(i32((i32((i32((i32((i32((l1Status.v.channelWiseTimes * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * l0Status.v.dtypeBias)) * l1Status.v.dbBL1))))
            dequantSize = Cell(i32(0))
            if self._branch(69,(self.tilingIns_.deqType == DequantType.TENSOR)):
                assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
            assign(AttrRef(l1Status.v,'kBL1'),i32(min(self.CalL1MaxLen(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), l1Status, l0Status, kbAlignValue.v, L1TilingType.KBL1_16), coreStatus.v.k)))
            if self._branch(70,self.IsUbNd2Nz()):
                assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_OFF))
                b1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - a1Length.v))))
                assign(AttrRef(l1Status.v,'kBL1'),i32(min(self.CalL1MaxLen(min(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v), l1Status, l0Status, kbAlignValue.v, L1TilingType.KBL1_16), coreStatus.v.k)))
            assign(AttrRef(l1Status.v,'bL1Times'),i32(min(i32(cdiv(l1Status.v.kBL1,l0Status.v.kL0)), l1Status.v.maxKBL1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'bL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kBL1'),i32(i32((l1Status.v.bL1Times * l0Status.v.kL0))))
            if self._branch(71,(l1Status.v.kBL1 == coreStatus.v.k)):
                assign(AttrRef(l1Status.v,'nBL1'),i32(min(self.CalL1MaxLen(l1Status.v.bL1Size, l1Status, l0Status, kbAlignValue.v, L1TilingType.N_BL1), l1Status.v.maxNBL1)))
                self.GetNearestFactor(nRepeat, AttrRef(l1Status.v,'nBL1'), 2147483647)
            invalidL1Status = Cell(bool((True if ((l1Status.v.nBL1 == 0) or (l1Status.v.kBL1 == 0)) else False)))
            possibleMRepeat = Cell(i32((1 if (l1Status.v.kBL1 == coreStatus.v.k) else mRepeat.v)))
            assign(AttrRef(l1Status.v,'loadSize'),i32((2147483647 if invalidL1Status.v else i32(((0 if (self.tilingIns_.aType_.pos == TPosition.TSCM) else coreStatus.v.m) + i32((possibleMRepeat.v * coreStatus.v.n)))))))
            if self._branch(72,(((self.cfg.l1DB == DB_ON) and (l1Status.v.kBL1 == coreStatus.v.k)) and (i32((l1Status.v.nBL1 * l0Status.v.nL0)) == coreStatus.v.n))):
                assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_OFF))
            assign(ItemRef(res.v[IDX_ONE],IDX_ZERO),i32(l1Status.v.kAL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_ONE),i32(l1Status.v.mAL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_TWO),i32(l1Status.v.dbAL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_THREE),i32(l1Status.v.kBL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_FOUR),i32(l1Status.v.nBL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_FIVE),i32(l1Status.v.dbBL1))
            assign(ItemRef(res.v[IDX_ONE],IDX_SIX),i32(l1Status.v.loadSize))
    
    def L1StatusBl1FullLoad(self, coreStatus, l0Status, l1Status, res):
        'Source: MatmulTilingAlgorithm::L1StatusBl1FullLoad; lines 749-821'
        self._visit('L1StatusBl1FullLoad')
        res = Cell(res)
        if self._branch(73,(self.tilingIns_.aType_.pos == TPosition.TSCM)):
            return
        mRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.m, l0Status.v.mL0)))
        nRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.n, l0Status.v.nL0)))
        kaAlignValue = Cell(i32(1))
        kbAlignValue = Cell(i32(1))
        self.GetABL1KAlignValue(kaAlignValue, kbAlignValue)
        assign(AttrRef(l1Status.v,'kBL1'),i32(i32((MathUtil.CeilDivision(l1Status.v.kBL1, l0Status.v.kL0) * l0Status.v.kL0))))
        curL1Size = Cell(i32(self.GetL1Size(l1Status, l0Status)))
        b1Length = Cell(i32(self.GetBL1UbSize(l1Status, l0Status)))
        if self._branch(74,(((curL1Size.v > 0) and (curL1Size.v <= self.tilingIns_.bufferPool_.l1Size)) and (b1Length.v < self.tilingIns_.bufferPool_.ubSize))):
            assign(AttrRef(l1Status.v,'bL1FullLoad'),bool(True))
            assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((max(MathUtil.Align(coreStatus.v.k, kbAlignValue.v), MathUtil.Align(l1Status.v.kBL1, kbAlignValue.v)) * max(i32((l1Status.v.nBL1 * l0Status.v.nL0)), coreStatus.v.n))) * C0_SIZE)) * C0_BYTE_SIZE))))
            if self._branch(75,(self.tilingIns_.bType_.pos == TPosition.TSCM)):
                assign(AttrRef(l1Status.v,'aL1Size'),i32(self.tilingIns_.bufferPool_.l1Size))
            else:
                assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.bL1Size))))
            if self._branch(76,(self.cfg.l1DB == DB_ON)):
                assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_ON))
                if self._branch(77,(self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size)):
                    assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_OFF))
            dequantSize = Cell(i32(0))
            if self._branch(78,(self.tilingIns_.deqType == DequantType.TENSOR)):
                assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
            biasSize = Cell(i32(i32((i32((i32((i32((i32((l1Status.v.channelWiseTimes * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * l0Status.v.dtypeBias)) * l1Status.v.dbBL1))))
            assign(AttrRef(l1Status.v,'kAL1'),i32(min(self.CalL1MaxLen(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), l1Status, l0Status, kaAlignValue.v, L1TilingType.KAL1_16), coreStatus.v.k)))
            if self._branch(79,self.IsUbNd2Nz()):
                assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_OFF))
                a1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - b1Length.v))))
                assign(AttrRef(l1Status.v,'kAL1'),i32(min(self.CalL1MaxLen(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length.v), l1Status, l0Status, kaAlignValue.v, L1TilingType.KAL1_16), coreStatus.v.k)))
            assign(AttrRef(l1Status.v,'aL1Times'),i32(min(i32(cdiv(l1Status.v.kAL1,l0Status.v.kL0)), l1Status.v.maxKAL1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'aL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kAL1'),i32(i32((l1Status.v.aL1Times * l0Status.v.kL0))))
            if self._branch(80,(l1Status.v.kAL1 == coreStatus.v.k)):
                assign(AttrRef(l1Status.v,'mAL1'),i32(min(self.CalL1MaxLen(i32((l1Status.v.aL1Size - biasSize.v)), l1Status, l0Status, kaAlignValue.v, L1TilingType.M_AL1), l1Status.v.maxMAL1)))
                self.GetNearestFactor(mRepeat, AttrRef(l1Status.v,'mAL1'), 2147483647)
            invalidL1Status = Cell(bool((True if ((l1Status.v.mAL1 == 0) or (l1Status.v.kAL1 == 0)) else False)))
            possibleNRepeat = Cell(i32((1 if (l1Status.v.kAL1 == coreStatus.v.k) else nRepeat.v)))
            assign(AttrRef(l1Status.v,'loadSize'),i32((2147483647 if invalidL1Status.v else i32(((0 if (self.tilingIns_.bType_.pos == TPosition.TSCM) else coreStatus.v.n) + i32((possibleNRepeat.v * coreStatus.v.m)))))))
            if self._branch(81,(((self.cfg.l1DB == DB_ON) and (l1Status.v.kAL1 == coreStatus.v.k)) and (i32((l1Status.v.mAL1 * l0Status.v.mL0)) == coreStatus.v.m))):
                assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_OFF))
            assign(ItemRef(res.v[IDX_TWO],IDX_ZERO),i32(l1Status.v.kAL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_ONE),i32(l1Status.v.mAL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_TWO),i32(l1Status.v.dbAL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_THREE),i32(l1Status.v.kBL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_FOUR),i32(l1Status.v.nBL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_FIVE),i32(l1Status.v.dbBL1))
            assign(ItemRef(res.v[IDX_TWO],IDX_SIX),i32(l1Status.v.loadSize))
    
    def L1StatusBothFullLoad(self, coreStatus, l0Status, l1Status, res):
        'Source: MatmulTilingAlgorithm::L1StatusBothFullLoad; lines 823-849'
        self._visit('L1StatusBothFullLoad')
        res = Cell(res)
        assign(AttrRef(l1Status.v,'kAL1'),i32(i32((MathUtil.CeilDivision(l1Status.v.kAL1, l0Status.v.kL0) * l0Status.v.kL0))))
        assign(AttrRef(l1Status.v,'kBL1'),i32(i32((MathUtil.CeilDivision(l1Status.v.kBL1, l0Status.v.kL0) * l0Status.v.kL0))))
        curL1Size = Cell(i32(self.GetL1Size(l1Status, l0Status)))
        a1Length = Cell(i32(self.GetAL1UbSize(l1Status, l0Status)))
        b1Length = Cell(i32(self.GetBL1UbSize(l1Status, l0Status)))
        if self._branch(82,((self.tilingIns_.aType_.pos == TPosition.TSCM) and (self.tilingIns_.bType_.pos == TPosition.TSCM))):
            assign(AttrRef(l1Status.v,'mAL1'),i32(1))
            assign(AttrRef(l1Status.v,'nBL1'),i32(1))
        if self._branch(83,((((curL1Size.v > 0) and (curL1Size.v <= self.tilingIns_.bufferPool_.l1Size)) and (i32((a1Length.v + b1Length.v)) <= self.tilingIns_.bufferPool_.ubSize)) or ((self.tilingIns_.aType_.pos == TPosition.TSCM) and (self.tilingIns_.bType_.pos == TPosition.TSCM)))):
            assign(AttrRef(l1Status.v,'bothFullLoad'),bool(True))
            assign(AttrRef(l1Status.v,'loadSize'),i32(i32(((0 if (self.tilingIns_.aType_.pos == TPosition.TSCM) else coreStatus.v.m) + (0 if (self.tilingIns_.bType_.pos == TPosition.TSCM) else coreStatus.v.n)))))
            assign(ItemRef(res.v[IDX_ZERO],IDX_ZERO),i32(l1Status.v.kAL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_ONE),i32(l1Status.v.mAL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_TWO),i32(l1Status.v.dbAL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_THREE),i32(l1Status.v.kBL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_FOUR),i32(l1Status.v.nBL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_FIVE),i32(l1Status.v.dbBL1))
            assign(ItemRef(res.v[IDX_ZERO],IDX_SIX),i32(l1Status.v.loadSize))
    
    def NeitherFullLoadDb(self, coreStatus, l0Status, l1Status, kbl1Db):
        'Source: MatmulTilingAlgorithm::NeitherFullLoadDb; lines 850-880'
        self._visit('NeitherFullLoadDb')
        tmpKbl116 = Cell(i32(l1Status.v.kBL1))
        assign(AttrRef(l1Status.v,'kBL1'),i32(kbl1Db.v))
        if self._branch(84,((self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size) or (i32((self.GetAL1UbSize(l1Status, l0Status) + self.GetBL1UbSize(l1Status, l0Status))) > self.tilingIns_.bufferPool_.ubSize))):
            assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_OFF))
            if self._branch(85,((self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size) or (i32((self.GetAL1UbSize(l1Status, l0Status) + self.GetBL1UbSize(l1Status, l0Status))) > self.tilingIns_.bufferPool_.ubSize))):
                assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_OFF))
        assign(AttrRef(l1Status.v,'kBL1'),i32(coreStatus.v.k))
        bothDoubleBuffer = Cell(bool((((coreStatus.v.m != l0Status.v.mL0) and (coreStatus.v.k > l0Status.v.kL0)) and ((self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size) or (i32((self.GetAL1UbSize(l1Status, l0Status) + self.GetBL1UbSize(l1Status, l0Status))) > self.tilingIns_.bufferPool_.ubSize)))))
        assign(AttrRef(l1Status.v,'kBL1'),i32(tmpKbl116.v))
        if self._branch(86,bothDoubleBuffer.v):
            assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_ON))
            assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_ON))
            if self._branch(87,((self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size) or (i32((self.GetAL1UbSize(l1Status, l0Status) + self.GetBL1UbSize(l1Status, l0Status))) > self.tilingIns_.bufferPool_.ubSize))):
                assign(AttrRef(l1Status.v,'dbBL1'),i32(DB_OFF))
                if self._branch(88,((self.GetL1Size(l1Status, l0Status) > self.tilingIns_.bufferPool_.l1Size) or (i32((self.GetAL1UbSize(l1Status, l0Status) + self.GetBL1UbSize(l1Status, l0Status))) > self.tilingIns_.bufferPool_.ubSize))):
                    assign(AttrRef(l1Status.v,'dbAL1'),i32(DB_OFF))
    
    def NeitherFullLoadMN(self, coreStatus, l0Status, l1Status):
        'Source: MatmulTilingAlgorithm::NeitherFullLoadMN; lines 882-1026'
        self._visit('NeitherFullLoadMN')
        mRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.m, l0Status.v.mL0)))
        nRepeat = Cell(i32(MathUtil.CeilDivision(coreStatus.v.n, l0Status.v.nL0)))
        if self._branch(89,((l0Status.v.dtypeBias == FP32_BYTES) and (l1Status.v.channelWiseTimes > 0))):
            increment(AttrRef(l1Status.v,'channelWiseTimes'),1,True)
        biasSize = Cell(i32(i32((i32((i32((i32((i32((l1Status.v.channelWiseTimes * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * FP16_BYTES)) * l1Status.v.dbBL1))))
        dequantSize = Cell(i32(0))
        if self._branch(90,(self.tilingIns_.deqType == DequantType.TENSOR)):
            assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
        kaAlignValue = Cell(i32(1))
        kbAlignValue = Cell(i32(1))
        self.GetABL1KAlignValue(kaAlignValue, kbAlignValue)
        l1Mfirst = Cell(L1StatusPack())
        l1Nfirst = Cell(L1StatusPack())
        err = Cell(i32(copy_array(l1Mfirst.v,l1Status.v,84)))
        if self._branch(91,(err.v != EOK)):
            0
            return
        assign(err,i32(copy_array(l1Nfirst.v,l1Status.v,84)))
        if self._branch(92,(err.v != EOK)):
            0
        assign(AttrRef(l1Mfirst.v,'bL1Size'),i32(i32((i32((i32((i32((MathUtil.Align(l1Mfirst.v.kBL1, kbAlignValue.v) * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Mfirst.v.dbBL1))))
        assign(AttrRef(l1Mfirst.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Mfirst.v.bL1Size))))
        a1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetBL1UbSize(l1Mfirst, l0Status)))))
        assign(AttrRef(l1Mfirst.v,'mAL1'),i32(max(min(min(self.CalL1MaxLen(i32((i32((l1Mfirst.v.aL1Size - biasSize.v)) - dequantSize.v)), l1Mfirst, l0Status, kaAlignValue.v, L1TilingType.M_AL1), l1Mfirst.v.maxMAL1), mRepeat.v), 1)))
        if self._branch(93,self.IsUbNd2Nz()):
            assign(AttrRef(l1Mfirst.v,'mAL1'),i32(max(min(min(self.CalL1MaxLen(min(i32((i32((l1Mfirst.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length.v), l1Mfirst, l0Status, kaAlignValue.v, L1TilingType.M_AL1), l1Mfirst.v.maxMAL1), mRepeat.v), 1)))
        self.GetNearestFactor(mRepeat, AttrRef(l1Mfirst.v,'mAL1'), 2147483647)
        assign(AttrRef(l1Mfirst.v,'aL1Size'),i32(i32((i32((i32((i32((i32((MathUtil.Align(l1Mfirst.v.kAL1, kaAlignValue.v) * l1Mfirst.v.mAL1)) * l0Status.v.mL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Mfirst.v.dbAL1))))
        assign(AttrRef(l1Mfirst.v,'bL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Mfirst.v.aL1Size))))
        b1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetAL1UbSize(l1Mfirst, l0Status)))))
        assign(AttrRef(l1Mfirst.v,'nBL1'),i32(max(min(min(self.CalL1MaxLen(i32((i32((l1Mfirst.v.bL1Size - biasSize.v)) - dequantSize.v)), l1Mfirst, l0Status, kbAlignValue.v, L1TilingType.N_BL1), l1Mfirst.v.maxNBL1), nRepeat.v), 1)))
        if self._branch(94,self.IsUbNd2Nz()):
            assign(AttrRef(l1Mfirst.v,'nBL1'),i32(max(min(min(self.CalL1MaxLen(min(i32((i32((l1Mfirst.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v), l1Mfirst, l0Status, kbAlignValue.v, L1TilingType.N_BL1), l1Mfirst.v.maxNBL1), nRepeat.v), 1)))
        self.GetNearestFactor(nRepeat, AttrRef(l1Mfirst.v,'nBL1'), 2147483647)
        assign(AttrRef(l1Mfirst.v,'loadSize'),i32(i32((coreStatus.v.m + i32((coreStatus.v.n * MathUtil.CeilDivision(coreStatus.v.m, i32((l1Mfirst.v.mAL1 * l0Status.v.mL0)))))))))
        assign(AttrRef(l1Nfirst.v,'aL1Size'),i32(i32((i32((i32((i32((MathUtil.Align(l1Nfirst.v.kAL1, kaAlignValue.v) * l0Status.v.mL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Nfirst.v.dbAL1))))
        assign(AttrRef(l1Nfirst.v,'bL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Nfirst.v.aL1Size))))
        assign(b1Length,i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetAL1UbSize(l1Nfirst, l0Status)))))
        assign(AttrRef(l1Nfirst.v,'nBL1'),i32(max(min(min(self.CalL1MaxLen(i32((i32((l1Nfirst.v.bL1Size - biasSize.v)) - dequantSize.v)), l1Nfirst, l0Status, kbAlignValue.v, L1TilingType.N_BL1), l1Nfirst.v.maxNBL1), nRepeat.v), 1)))
        if self._branch(95,self.IsUbNd2Nz()):
            assign(AttrRef(l1Nfirst.v,'nBL1'),i32(max(min(min(self.CalL1MaxLen(min(i32((i32((l1Nfirst.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v), l1Nfirst, l0Status, kbAlignValue.v, L1TilingType.N_BL1), l1Nfirst.v.maxNBL1), nRepeat.v), 1)))
        self.GetNearestFactor(nRepeat, AttrRef(l1Nfirst.v,'nBL1'), 2147483647)
        assign(AttrRef(l1Nfirst.v,'bL1Size'),i32(i32((i32((i32((i32((i32((MathUtil.Align(coreStatus.v.k, kbAlignValue.v) * l1Nfirst.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Nfirst.v.dbBL1))))
        assign(AttrRef(l1Nfirst.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Nfirst.v.bL1Size))))
        assign(a1Length,i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetBL1UbSize(l1Nfirst, l0Status)))))
        assign(biasSize,i32(i32((biasSize.v * l1Nfirst.v.nBL1))))
        assign(AttrRef(l1Nfirst.v,'mAL1'),i32(max(min(min(self.CalL1MaxLen(i32((i32((l1Nfirst.v.aL1Size - biasSize.v)) - dequantSize.v)), l1Nfirst, l0Status, kaAlignValue.v, L1TilingType.M_AL1), l1Nfirst.v.maxMAL1), mRepeat.v), 1)))
        if self._branch(96,self.IsUbNd2Nz()):
            assign(AttrRef(l1Nfirst.v,'mAL1'),i32(max(min(min(self.CalL1MaxLen(min(i32((i32((l1Nfirst.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length.v), l1Nfirst, l0Status, kaAlignValue.v, L1TilingType.M_AL1), l1Nfirst.v.maxMAL1), mRepeat.v), 1)))
        self.GetNearestFactor(mRepeat, AttrRef(l1Nfirst.v,'mAL1'), 2147483647)
        assign(AttrRef(l1Nfirst.v,'loadSize'),i32(i32((i32((coreStatus.v.m * MathUtil.CeilDivision(coreStatus.v.n, i32((l1Nfirst.v.nBL1 * l0Status.v.nL0))))) + coreStatus.v.n))))
        if self._branch(97,((l1Status.v.kAL1 >= coreStatus.v.k) and (l1Status.v.kBL1 >= coreStatus.v.k))):
            if self._branch(98,(l1Nfirst.v.loadSize > l1Mfirst.v.loadSize)):
                errnoT = Cell(i32(copy_array(l1Status.v,l1Mfirst.v,84)))
                if self._branch(99,(errnoT.v != EOK)):
                    0
                    return
            else:
                errnoT_2 = Cell(i32(copy_array(l1Status.v,l1Nfirst.v,84)))
                if self._branch(100,(errnoT_2.v != EOK)):
                    0
                    return
        if self._branch(101,((l1Status.v.kAL1 >= coreStatus.v.k) and (l1Status.v.kBL1 < coreStatus.v.k))):
            assign(AttrRef(l1Mfirst.v,'nBL1'),i32(1))
            errnoT_3 = Cell(i32(copy_array(l1Status.v,l1Mfirst.v,84)))
            if self._branch(102,(errnoT_3.v != EOK)):
                0
                return
        if self._branch(103,((l1Status.v.kAL1 < coreStatus.v.k) and (l1Status.v.kBL1 >= coreStatus.v.k))):
            assign(AttrRef(l1Nfirst.v,'mAL1'),i32(1))
            errnoT_4 = Cell(i32(copy_array(l1Status.v,l1Nfirst.v,84)))
            if self._branch(104,(errnoT_4.v != EOK)):
                0
                return
        if self._branch(105,((l1Status.v.kAL1 < coreStatus.v.k) and (l1Status.v.kBL1 < coreStatus.v.k))):
            assign(AttrRef(l1Status.v,'mAL1'),i32(1))
            assign(AttrRef(l1Status.v,'nBL1'),i32(1))
            assign(AttrRef(l1Status.v,'loadSize'),i32(i32((i32((coreStatus.v.m * MathUtil.CeilDivision(coreStatus.v.n, i32((l1Mfirst.v.nBL1 * l0Status.v.nL0))))) + i32((coreStatus.v.n * MathUtil.CeilDivision(coreStatus.v.m, i32((l1Mfirst.v.mAL1 * l0Status.v.mL0)))))))))
    
    def NeitherFullLoadKforNZ(self, coreStatus, l0Status, l1Status):
        'Source: MatmulTilingAlgorithm::NeitherFullLoadKforNZ; lines 1028-1090'
        self._visit('NeitherFullLoadKforNZ')
        assign(AttrRef(l1Status.v,'kBL1'),i32(coreStatus.v.k))
        biasSize = Cell(i32(i32((i32((i32((i32((i32((l1Status.v.channelWiseTimes * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * l0Status.v.dtypeBias)) * l1Status.v.dbBL1))))
        dequantSize = Cell(i32(0))
        if self._branch(106,(self.tilingIns_.deqType == DequantType.TENSOR)):
            assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
        kaAlignValue = Cell(i32(1))
        kbAlignValue = Cell(i32(1))
        self.GetABL1KAlignValue(kaAlignValue, kbAlignValue)
        if self._branch(107,((self.GetL1Size(l1Status, l0Status) > 0) and (self.GetL1Size(l1Status, l0Status) <= self.tilingIns_.bufferPool_.l1Size))):
            assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((i32((i32((MathUtil.Align(coreStatus.v.k, kbAlignValue.v) * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbBL1))))
            assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.bL1Size))))
            a1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetBL1UbSize(l1Status, l0Status)))))
            assign(AttrRef(l1Status.v,'kAL1'),i32(min(self.CalL1MaxLen(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), l1Status, l0Status, kaAlignValue.v, L1TilingType.KAL1_16), coreStatus.v.k)))
            if self._branch(108,self.IsUbNd2Nz()):
                assign(AttrRef(l1Status.v,'kAL1'),i32(min(self.CalL1MaxLen(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length.v), l1Status, l0Status, kaAlignValue.v, L1TilingType.KAL1_16), coreStatus.v.k)))
            assign(AttrRef(l1Status.v,'aL1Times'),i32(max(min(i32(cdiv(l1Status.v.kAL1,l0Status.v.kL0)), l1Status.v.maxKAL1), 1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'aL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kAL1'),i32(i32((l1Status.v.aL1Times * l0Status.v.kL0))))
        else:
            perK = Cell(i32(min(i32((i32(cdiv(i32(cdiv(i32((i32((self.tilingIns_.bufferPool_.l1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((i32((l0Status.v.mL0 * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbAL1)) + i32((i32((i32((C0_SIZE * l0Status.v.nL0)) * C0_BYTE_SIZE)) * l1Status.v.dbBL1)))))),l0Status.v.kL0)) * l0Status.v.kL0)), coreStatus.v.k)))
            if self._branch(109,self.IsUbNd2Nz()):
                assign(perK,i32(min(i32((i32(cdiv(i32(cdiv(min(i32((i32((self.tilingIns_.bufferPool_.l1Size - biasSize.v)) - dequantSize.v)), self.tilingIns_.bufferPool_.ubSize),i32((i32((i32((i32((l0Status.v.mL0 * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbAL1)) + i32((i32((i32((C0_SIZE * l0Status.v.nL0)) * C0_BYTE_SIZE)) * l1Status.v.dbBL1)))))),l0Status.v.kL0)) * l0Status.v.kL0)), coreStatus.v.k)))
            biasFactor = Cell(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) if self.tilingIns_.isBias else 0)))
            aAlignedPerK = Cell(i32(MathUtil.Align(perK.v, kaAlignValue.v)))
            bAlignedPerK = Cell(i32(MathUtil.Align(perK.v, kbAlignValue.v)))
            if self._branch(110,((self.tilingIns_.aType_.dataType == DataType.DT_FLOAT) and (not self.CheckL1Size(i32((i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * aAlignedPerK.v)) * l1Status.v.dbAL1)), i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * bAlignedPerK.v)) * l1Status.v.dbBL1)), i32((i32((i32((i32((biasFactor.v * C0_SIZE)) * l0Status.v.dtypeBias)) * l1Status.v.dbBL1)) + dequantSize.v)))))):
                assign(perK,i32(i32((perK.v - 1))))
            perTimes = Cell(i32(min(i32(cdiv(perK.v,l0Status.v.kL0)), max(l1Status.v.maxKAL1, l1Status.v.maxKBL1))))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), perTimes, 2147483647)
            assign(perTimes,i32(min(perTimes.v, l1Status.v.allTimes)))
            assign(perK,i32(i32((perTimes.v * l0Status.v.kL0))))
            assign(AttrRef(l1Status.v,'kAL1'),i32(perK.v))
            assign(AttrRef(l1Status.v,'kBL1'),i32(perK.v))
    
    def CheckL1Size(self, amat, bmat, curBiasL1Size):
        'Source: MatmulTilingAlgorithm::CheckL1Size; lines 1092-1097'
        self._visit('CheckL1Size')
        amat = Cell(i32(amat))
        bmat = Cell(i32(bmat))
        curBiasL1Size = Cell(i32(curBiasL1Size))
        loadSizeBytes = Cell(i64(i64((i64((i64((i64(i32((amat.v + bmat.v))) * i64(C0_SIZE))) * i64(C0_BYTE_SIZE))) + i64(curBiasL1Size.v)))))
        return (loadSizeBytes.v <= i64(self.tilingIns_.bufferPool_.l1Size))
    
    def NeitherFullLoadKforND(self, coreStatus, l0Status, l1Status, kMaxAxis):
        'Source: MatmulTilingAlgorithm::NeitherFullLoadKforND; lines 1099-1239'
        self._visit('NeitherFullLoadKforND')
        biasSize = Cell(i32(i32((i32((i32((i32((i32((l1Status.v.channelWiseTimes * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * l0Status.v.dtypeBias)) * l1Status.v.dbBL1))))
        dequantSize = Cell(i32(0))
        if self._branch(111,(self.tilingIns_.deqType == DequantType.TENSOR)):
            assign(dequantSize,i32(i32(u64((u64(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE))) * UINT64_TYPES)))))
        alignValue = Cell(i32(FP32_ALIGN_SIZE))
        if self._branch(112,(self.tilingIns_.aType_.dataType == DataType.DT_INT8)):
            assign(alignValue,i32(INT8_ALIGN_SIZE))
        else:
            if self._branch(113,(self.tilingIns_.aType_.dataType == DataType.DT_INT4)):
                assign(alignValue,i32(INT4_ALIGN_SIZE))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        alignM = Cell(i32(i32((MathUtil.CeilDivision(i32((l1Status.v.mAL1 * C0_SIZE)), alignValue.v) * alignValue.v))))
        alignN = Cell(i32(i32((MathUtil.CeilDivision(i32((l1Status.v.nBL1 * C0_SIZE)), alignValue.v) * alignValue.v))))
        alignK = Cell(i32(i32(u32(cdiv(u32((u32(i32((MathUtil.CeilDivision(i32((l0Status.v.kL0 * reduceSize.v)), alignValue.v) * alignValue.v))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
        if self._branch(114,(kMaxAxis.v == 1)):
            assign(AttrRef(l1Status.v,'kBL1'),i32(l0Status.v.kL0))
            if self._branch(115,((self.tilingIns_.bType_.dataType == DataType.DT_FLOAT) or (self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_INT8)))):
                assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kBL1 * l0Status.v.nL0)) * C0_SIZE)) * alignK.v)) * l1Status.v.nBL1)) * l1Status.v.dbBL1))))
            else:
                if self._branch(116,((not self.tilingIns_.bType_.isTrans) and ((self.tilingIns_.bType_.dataType == DataType.DT_INT8) or (self.tilingIns_.bType_.dataType == DataType.DT_INT4)))):
                    assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((i32((l1Status.v.kBL1 * l0Status.v.nL0)) * alignK.v)) * alignN.v)) * l1Status.v.dbBL1))))
                else:
                    assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kBL1 * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbBL1))))
            assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.bL1Size))))
            a1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetBL1UbSize(l1Status, l0Status)))))
            factor = Cell(i32(i32((i32((i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * C0_SIZE)) * l1Status.v.dbAL1)) * C0_BYTE_SIZE))))
            assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor.v == 0) else min(i32(cdiv(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)),factor.v)), coreStatus.v.k))))
            if self._branch(117,self.IsUbNd2Nz()):
                assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor.v == 0) else min(i32(cdiv(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length.v),factor.v)), coreStatus.v.k))))
            assign(AttrRef(l1Status.v,'aL1Times'),i32(max(i32(cdiv(l1Status.v.kAL1,l0Status.v.kL0)), 1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'aL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kAL1'),i32(i32((l1Status.v.aL1Times * l0Status.v.kL0))))
            assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kAL1 * l1Status.v.mAL1)) * l0Status.v.mL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbAL1))))
            assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.aL1Size))))
            b1Length = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetAL1UbSize(l1Status, l0Status)))))
            if self._branch(118,((self.tilingIns_.bType_.dataType == DataType.DT_FLOAT) or (self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_INT8)))):
                assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * alignK.v)))), coreStatus.v.k)))
                if self._branch(119,self.IsUbNd2Nz()):
                    assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(min(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * alignK.v)))), coreStatus.v.k)))
            else:
                if self._branch(120,((not self.tilingIns_.bType_.isTrans) and ((self.tilingIns_.bType_.dataType == DataType.DT_INT8) or (self.tilingIns_.bType_.dataType == DataType.DT_INT4)))):
                    assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((alignN.v * l0Status.v.nL0)) * l1Status.v.dbBL1)) * alignK.v)))), coreStatus.v.k)))
                    if self._branch(121,self.IsUbNd2Nz()):
                        assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(min(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v),i32((i32((i32((alignN.v * l0Status.v.nL0)) * l1Status.v.dbBL1)) * alignK.v)))), coreStatus.v.k)))
                else:
                    assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * C0_BYTE_SIZE)))), coreStatus.v.k)))
                    if self._branch(122,self.IsUbNd2Nz()):
                        assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(min(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length.v),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * C0_BYTE_SIZE)))), coreStatus.v.k)))
            assign(AttrRef(l1Status.v,'bL1Times'),i32(max(min(i32(cdiv(l1Status.v.kBL1,l0Status.v.kL0)), l1Status.v.maxKBL1), 1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'bL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kBL1'),i32(i32((l1Status.v.bL1Times * l0Status.v.kL0))))
        if self._branch(123,(kMaxAxis.v == NUM_TWO)):
            assign(AttrRef(l1Status.v,'kAL1'),i32(l0Status.v.kL0))
            if self._branch(124,((self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_FLOAT)) or ((not self.tilingIns_.aType_.isTrans) and (self.tilingIns_.aType_.dataType == DataType.DT_INT8)))):
                assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kAL1 * l1Status.v.mAL1)) * l0Status.v.mL0)) * C0_SIZE)) * alignK.v)) * l1Status.v.dbAL1))))
            else:
                if self._branch(125,(self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_INT8))):
                    assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((i32((l1Status.v.kAL1 * alignM.v)) * l0Status.v.mL0)) * alignK.v)) * l1Status.v.dbAL1))))
                else:
                    assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kAL1 * l1Status.v.mAL1)) * l0Status.v.mL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbAL1))))
            assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.aL1Size))))
            b1Length_2 = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetAL1UbSize(l1Status, l0Status)))))
            assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * C0_BYTE_SIZE)))), coreStatus.v.k)))
            if self._branch(126,self.IsUbNd2Nz()):
                assign(AttrRef(l1Status.v,'kBL1'),i32(min(i32(cdiv(min(i32((i32((l1Status.v.bL1Size - biasSize.v)) - dequantSize.v)), b1Length_2.v),i32((i32((i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)) * l1Status.v.dbBL1)) * C0_BYTE_SIZE)))), coreStatus.v.k)))
            assign(AttrRef(l1Status.v,'bL1Times'),i32(max(i32(cdiv(l1Status.v.kBL1,l0Status.v.kL0)), 1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'bL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kBL1'),i32(i32((l1Status.v.bL1Times * l0Status.v.kL0))))
            assign(AttrRef(l1Status.v,'bL1Size'),i32(i32((i32((i32((i32((i32((l1Status.v.kBL1 * l1Status.v.nBL1)) * l0Status.v.nL0)) * C0_SIZE)) * C0_BYTE_SIZE)) * l1Status.v.dbBL1))))
            assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - l1Status.v.bL1Size))))
            a1Length_2 = Cell(i32(i32((self.tilingIns_.bufferPool_.ubSize - self.GetBL1UbSize(l1Status, l0Status)))))
            if self._branch(127,((self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_FLOAT)) or ((not self.tilingIns_.aType_.isTrans) and ((self.tilingIns_.aType_.dataType == DataType.DT_INT8) or (self.tilingIns_.aType_.dataType == DataType.DT_INT4))))):
                factor_2 = Cell(i32(i32((i32((i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * C0_SIZE)) * l1Status.v.dbAL1)) * alignK.v))))
                assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor_2.v == 0) else min(i32(cdiv(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)),factor_2.v)), coreStatus.v.k))))
                if self._branch(128,self.IsUbNd2Nz()):
                    assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor_2.v == 0) else min(i32(cdiv(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length_2.v),factor_2.v)), coreStatus.v.k))))
            else:
                if self._branch(129,(self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_INT8))):
                    assign(AttrRef(l1Status.v,'kAL1'),i32(min(i32(cdiv(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)),i32((i32((i32((alignM.v * l0Status.v.mL0)) * l1Status.v.dbAL1)) * alignK.v)))), coreStatus.v.k)))
                    if self._branch(130,self.IsUbNd2Nz()):
                        assign(AttrRef(l1Status.v,'kAL1'),i32(min(i32(cdiv(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length_2.v),i32((i32((i32((alignM.v * l0Status.v.mL0)) * l1Status.v.dbAL1)) * alignK.v)))), coreStatus.v.k)))
                    assign(AttrRef(l1Status.v,'aL1Size'),i32(i32((i32((i32((i32((l1Status.v.kAL1 * alignM.v)) * l0Status.v.mL0)) * alignK.v)) * l1Status.v.dbAL1))))
                else:
                    factor_3 = Cell(i32(i32((i32((i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * C0_SIZE)) * l1Status.v.dbAL1)) * C0_BYTE_SIZE))))
                    assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor_3.v == 0) else min(i32(cdiv(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)),factor_3.v)), coreStatus.v.k))))
                    if self._branch(131,self.IsUbNd2Nz()):
                        assign(AttrRef(l1Status.v,'kAL1'),i32((coreStatus.v.k if (factor_3.v == 0) else min(i32(cdiv(min(i32((i32((l1Status.v.aL1Size - biasSize.v)) - dequantSize.v)), a1Length_2.v),factor_3.v)), coreStatus.v.k))))
            assign(AttrRef(l1Status.v,'aL1Times'),i32(max(min(i32(cdiv(l1Status.v.kAL1,l0Status.v.kL0)), l1Status.v.maxKAL1), 1)))
            self.GetNearestFactor(AttrRef(l1Status.v,'allTimes'), AttrRef(l1Status.v,'aL1Times'), 2147483647)
            assign(AttrRef(l1Status.v,'kAL1'),i32(i32((l1Status.v.aL1Times * l0Status.v.kL0))))
    
    def NeitherFullLoadK(self, coreStatus, l0Status, l1Status):
        'Source: MatmulTilingAlgorithm::NeitherFullLoadK; lines 1240-1277'
        self._visit('NeitherFullLoadK')
        if self._branch(132,(l0Status.v.kL0 == coreStatus.v.k)):
            return
        kMaxAxis = Cell(i32(0))
        if self._branch(133,((not self.tilingIns_.aType_.isTrans) and (not self.tilingIns_.bType_.isTrans))):
            assign(kMaxAxis,i32(1))
        if self._branch(134,(self.tilingIns_.aType_.isTrans and self.tilingIns_.bType_.isTrans)):
            assign(kMaxAxis,i32(2))
        if self._branch(135,((not self.tilingIns_.aType_.isTrans) and self.tilingIns_.bType_.isTrans)):
            assign(kMaxAxis,i32((1 if (l0Status.v.mL0 > l0Status.v.nL0) else 2)))
        if self._branch(136,(kMaxAxis.v != 0)):
            self.NeitherFullLoadKforND(coreStatus, l0Status, l1Status, kMaxAxis)
        else:
            self.NeitherFullLoadKforNZ(coreStatus, l0Status, l1Status)
        if self._branch(137,self.cfg.factorSplit):
            if self._branch(138,((l1Status.v.kAL1 > l1Status.v.kBL1) and (i32(cmod(l1Status.v.kAL1,l1Status.v.kBL1)) != 0))):
                while ((i32(cmod(l1Status.v.kAL1,l1Status.v.kBL1)) != 0) or ((l1Status.v.kAL1 != l1Status.v.kBL1) and (i32(cmod(coreStatus.v.k,l1Status.v.kAL1)) != 0))):
                    assign(AttrRef(l1Status.v,'kAL1'),i32(i32((l1Status.v.kAL1 - 1))))
            if self._branch(139,((l1Status.v.kAL1 < l1Status.v.kBL1) and (i32(cmod(l1Status.v.kBL1,l1Status.v.kAL1)) != 0))):
                while ((i32(cmod(l1Status.v.kBL1,l1Status.v.kAL1)) != 0) or ((l1Status.v.kAL1 != l1Status.v.kBL1) and (i32(cmod(coreStatus.v.k,l1Status.v.kBL1)) != 0))):
                    assign(AttrRef(l1Status.v,'kBL1'),i32(i32((l1Status.v.kBL1 - 1))))
    
    def L1StatusNeitherFullLoad(self, coreStatus, l0Status, l1Status, res):
        'Source: MatmulTilingAlgorithm::L1StatusNeitherFullLoad; lines 1279-1299'
        self._visit('L1StatusNeitherFullLoad')
        res = Cell(res)
        if self._branch(140,((self.tilingIns_.aType_.pos == TPosition.TSCM) or (self.tilingIns_.bType_.pos == TPosition.TSCM))):
            return
        if self._branch(141,(self.cfg.l1DB == DB_ON)):
            self.NeitherFullLoadDb(coreStatus, l0Status, l1Status, Cell(DB_ON))
        self.NeitherFullLoadK(coreStatus, l0Status, l1Status)
        self.NeitherFullLoadMN(coreStatus, l0Status, l1Status)
        assign(ItemRef(res.v[IDX_THREE],IDX_ZERO),i32(l1Status.v.kAL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_ONE),i32(l1Status.v.mAL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_TWO),i32(l1Status.v.dbAL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_THREE),i32(l1Status.v.kBL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_FOUR),i32(l1Status.v.nBL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_FIVE),i32(l1Status.v.dbBL1))
        assign(ItemRef(res.v[IDX_THREE],IDX_SIX),i32(l1Status.v.loadSize))
    
    def GetL1Factors(self, opType, param, coreStatus, l0Status, l1Status):
        'Source: MatmulTilingAlgorithm::GetL1Factors; lines 1301-1386'
        self._visit('GetL1Factors')
        opType.v
        param.v
        mte1Loop = Cell(i32(i32(cdiv(MIN_MTE1_LOAD,i32(((1 if (l0Status.v.nL0 == 1) else l0Status.v.kL0) + (1 if (l0Status.v.kL0 == 1) else l0Status.v.mL0)))))))
        res = Cell([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
        assign(AttrRef(l1Status.v,'allTimes'),i32(MathUtil.CeilDivision(coreStatus.v.k, l0Status.v.kL0)))
        assign(AttrRef(l1Status.v,'maxMAL1'),i32(i32(cdiv(i32((i32((coreStatus.v.m + l0Status.v.mL0)) - 1)),l0Status.v.mL0))))
        assign(AttrRef(l1Status.v,'maxNBL1'),i32(i32(cdiv(i32((i32((coreStatus.v.n + l0Status.v.nL0)) - 1)),l0Status.v.nL0))))
        assign(AttrRef(l1Status.v,'maxKAL1'),i32(max(mte1Loop.v, i32(cdiv(i32((i32((i32(cdiv(i32((i32((MIN_MTE1_LOAD + l0Status.v.mL0)) - 1)),l0Status.v.mL0)) + l0Status.v.kL0)) - 1)),l0Status.v.kL0)))))
        assign(AttrRef(l1Status.v,'maxKBL1'),i32(max(mte1Loop.v, i32(cdiv(i32((i32((i32(cdiv(i32((i32((MIN_MTE1_LOAD + l0Status.v.nL0)) - 1)),l0Status.v.nL0)) + l0Status.v.kL0)) - 1)),l0Status.v.kL0)))))
        if self._branch(142,(self.tilingIns_.isSupportL0c2Out and self.tilingIns_.isBias)):
            increment(AttrRef(l1Status.v,'channelWiseTimes'),1,True)
        bothFullLoadFactors = Cell([coreStatus.v.k, coreStatus.v.k, l1Status.v.maxMAL1, l1Status.v.maxNBL1, DB_OFF, DB_OFF])
        l1Status.v.SetStatus(bothFullLoadFactors)
        self.L1StatusBothFullLoad(coreStatus, l0Status, l1Status, res.v)
        al1FullLoadFactors = Cell([coreStatus.v.k, l0Status.v.kL0, l1Status.v.maxMAL1, 1, DB_OFF, DB_OFF])
        l1Status.v.SetStatus(al1FullLoadFactors)
        self.L1StatusAl1FullLoad(coreStatus, l0Status, l1Status, res.v)
        bl1FullLoadFactors = Cell([l0Status.v.kL0, coreStatus.v.k, 1, l1Status.v.maxNBL1, DB_OFF, DB_OFF])
        l1Status.v.SetStatus(bl1FullLoadFactors)
        self.L1StatusBl1FullLoad(coreStatus, l0Status, l1Status, res.v)
        assign(ItemRef(res.v[IDX_THREE],IDX_SIX),i32(2147483647))
        neitherFullLoadFactors = Cell([l0Status.v.kL0, l0Status.v.kL0, 1, 1, DB_ON, DB_ON])
        l1Status.v.SetStatus(neitherFullLoadFactors)
        self.L1StatusNeitherFullLoad(coreStatus, l0Status, l1Status, res.v)
        tmpFactors = Cell(res.v[IDX_THREE])
        tmpLoadSize = Cell(i32(tmpFactors.v[IDX_SIX]))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        kAl1FactorOne = Cell(i32((MathUtil.CeilDivision(MathUtil.CeilDivision(self.tilingIns_.singleK, reduceSize.v), i32((coreStatus.v.kDim * res.v[IDX_ONE][IDX_ZERO]))) if (res.v[IDX_ONE][IDX_ZERO] > 0) else 1)))
        kBl1FactorTwo = Cell(i32((MathUtil.CeilDivision(MathUtil.CeilDivision(self.tilingIns_.singleK, reduceSize.v), i32((coreStatus.v.kDim * res.v[IDX_TWO][IDX_THREE]))) if (res.v[IDX_TWO][IDX_THREE] > 0) else 1)))
        kAl1FactorZero = Cell(i32((MathUtil.CeilDivision(MathUtil.CeilDivision(self.tilingIns_.singleK, reduceSize.v), i32((coreStatus.v.kDim * res.v[IDX_ZERO][IDX_ZERO]))) if (res.v[IDX_ZERO][IDX_ZERO] > 0) else 1)))
        kBl1FactorZero = Cell(i32((MathUtil.CeilDivision(MathUtil.CeilDivision(self.tilingIns_.singleK, reduceSize.v), i32((coreStatus.v.kDim * res.v[IDX_ZERO][IDX_THREE]))) if (res.v[IDX_ZERO][IDX_THREE] > 0) else 1)))
        al1FullLoad = Cell(bool(((l1Status.v.aL1FullLoad and (kAl1FactorOne.v == 1)) if ((self.tilingIns_.aType_.type == CubeFormat.ND) and (self.tilingIns_.bType_.type == CubeFormat.ND)) else l1Status.v.aL1FullLoad)))
        bl1FullLoad = Cell(bool(((l1Status.v.bL1FullLoad and (kBl1FactorTwo.v == 1)) if ((self.tilingIns_.aType_.type == CubeFormat.ND) and (self.tilingIns_.bType_.type == CubeFormat.ND)) else l1Status.v.bL1FullLoad)))
        bothFullLoad = Cell(bool((((l1Status.v.bothFullLoad and (kAl1FactorZero.v == 1)) and (kBl1FactorZero.v == 1)) if ((self.tilingIns_.aType_.type == CubeFormat.ND) and (self.tilingIns_.bType_.type == CubeFormat.ND)) else l1Status.v.bothFullLoad)))
        if self._branch(143,(al1FullLoad.v and ((res.v[IDX_ONE][IDX_SIX] < tmpLoadSize.v) or ((res.v[IDX_ONE][IDX_SIX] == tmpLoadSize.v) and (i32((res.v[IDX_ONE][IDX_ONE] + res.v[IDX_ONE][IDX_FOUR])) >= i32((tmpFactors.v[IDX_ONE] + tmpFactors.v[IDX_FOUR]))))))):
            assign(tmpFactors,res.v[IDX_ONE])
            assign(tmpLoadSize,i32(tmpFactors.v[IDX_SIX]))
            0
        if self._branch(144,(bl1FullLoad.v and ((res.v[IDX_TWO][IDX_SIX] < tmpLoadSize.v) or ((res.v[IDX_TWO][IDX_SIX] == tmpLoadSize.v) and (i32((res.v[IDX_TWO][IDX_ONE] + res.v[IDX_TWO][IDX_FOUR])) >= i32((tmpFactors.v[IDX_ONE] + tmpFactors.v[IDX_FOUR]))))))):
            assign(tmpFactors,res.v[IDX_TWO])
            assign(tmpLoadSize,i32(tmpFactors.v[IDX_SIX]))
            0
        if self._branch(145,(bothFullLoad.v and ((res.v[IDX_ZERO][IDX_SIX] < tmpLoadSize.v) or ((res.v[IDX_ZERO][IDX_SIX] == tmpLoadSize.v) and (i32((res.v[IDX_ZERO][IDX_ONE] + res.v[IDX_ZERO][IDX_FOUR])) >= i32((tmpFactors.v[IDX_ONE] + tmpFactors.v[IDX_FOUR]))))))):
            assign(tmpFactors,res.v[IDX_ZERO])
            0
        resL1Factors = Cell([tmpFactors.v[IDX_ZERO], tmpFactors.v[IDX_THREE], tmpFactors.v[IDX_ONE], tmpFactors.v[IDX_FOUR], tmpFactors.v[IDX_TWO], tmpFactors.v[IDX_FIVE]])
        l1Status.v.SetStatus(resL1Factors)
    
    def GetUsedSize(self, l1Size, l0cSize, ubSize, a1LengthCache, b1LengthCache):
        'Source: MatmulTilingAlgorithm::GetUsedSize; lines 1388-1484'
        self._visit('GetUsedSize')
        a1LengthCache = Cell(i32(a1LengthCache))
        b1LengthCache = Cell(i32(b1LengthCache))
        aTypeSize = Cell(u32(DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType]))
        bTypeSize = Cell(u32(DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType]))
        cTypeSize = Cell(u32(DTYPE_BYTE_TAB[self.tilingIns_.cType_.dataType]))
        biasTypeSize = Cell(u32(DTYPE_BYTE_TAB[self.tilingIns_.biasType_.dataType]))
        a1Length = Cell(i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseM * self.tilingIns_.tiling_.baseK))) * aTypeSize.v)),u32(BITS_PER_BYTE))))))
        b1Length = Cell(i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseN * self.tilingIns_.tiling_.baseK))) * bTypeSize.v)),u32(BITS_PER_BYTE))))))
        c1Length = Cell(i32(i32((i32((self.tilingIns_.tiling_.baseN * self.tilingIns_.tiling_.baseM)) * FP32_BYTES))))
        if self._branch(146,(self.tilingIns_.aType_.pos != TPosition.TSCM)):
            assign(l1Size,i32(i32((l1Size.v + i32((self.tilingIns_.tiling_.depthA1 * a1Length.v))))))
            if self._branch(147,(self.tilingIns_.enableL1CacheUB and (self.tilingIns_.socVersion == 'ASCEND310P'))):
                assign(l1Size,i32(i32((l1Size.v + i32((self.tilingIns_.tiling_.depthAL1CacheUB * a1LengthCache.v))))))
        if self._branch(148,(self.tilingIns_.bType_.pos != TPosition.TSCM)):
            assign(l1Size,i32(i32((l1Size.v + i32((self.tilingIns_.tiling_.depthB1 * b1Length.v))))))
            if self._branch(149,(self.tilingIns_.enableL1CacheUB and (self.tilingIns_.socVersion == 'ASCEND310P'))):
                assign(l1Size,i32(i32((l1Size.v + i32((self.tilingIns_.tiling_.depthBL1CacheUB * b1LengthCache.v))))))
        assign(l0cSize,i32(i32((l0cSize.v + c1Length.v))))
        if self._branch(150,bool(self.tilingIns_.tiling_.isBias)):
            if self._branch(151,(((self.tilingIns_.socVersion == 'ASCEND910B') or (self.tilingIns_.socVersion == 'ASCEND310B')) and (self.tilingIns_.biasType_.pos != TPosition.TSCM))):
                assign(l1Size,i32(u32((l1Size.v + u32((u32(self.tilingIns_.tiling_.baseN) * biasTypeSize.v))))))
        if self._branch(152,(((self.tilingIns_.socVersion == 'ASCEND910') or (self.tilingIns_.socVersion == 'ASCEND310P')) or (self.tilingIns_.socVersion == 'ASCEND310B'))):
            aUbLength = Cell(i32(0))
            bUbLength = Cell(i32(0))
            if self._branch(153,((not self.tilingIns_.aType_.isTrans) and (u32(cmod(u32(cdiv(u32((u32(self.tilingIns_.tiling_.singleCoreK) * aTypeSize.v)),u32(BITS_PER_BYTE))),u32(C0_BYTE_SIZE))) != u32(0)))):
                assign(aUbLength,i32(i32((self.tilingIns_.tiling_.baseM * C0_BYTE_SIZE))))
            if self._branch(154,(self.tilingIns_.aType_.isTrans and (u32(cmod(u32(cdiv(u32((u32(self.tilingIns_.tiling_.singleCoreM) * aTypeSize.v)),u32(BITS_PER_BYTE))),u32(C0_BYTE_SIZE))) != u32(0)))):
                assign(aUbLength,i32(i32((self.tilingIns_.tiling_.baseK * C0_BYTE_SIZE))))
            if self._branch(155,((not self.tilingIns_.bType_.isTrans) and (u32(cmod(u32(cdiv(u32((u32(self.tilingIns_.tiling_.singleCoreN) * bTypeSize.v)),u32(BITS_PER_BYTE))),u32(C0_BYTE_SIZE))) != u32(0)))):
                assign(bUbLength,i32(i32((self.tilingIns_.tiling_.baseK * C0_BYTE_SIZE))))
            if self._branch(156,(self.tilingIns_.bType_.isTrans and (u32(cmod(u32(cdiv(u32((u32(self.tilingIns_.tiling_.singleCoreK) * bTypeSize.v)),u32(BITS_PER_BYTE))),u32(C0_BYTE_SIZE))) != u32(0)))):
                assign(bUbLength,i32(i32((self.tilingIns_.tiling_.baseN * C0_BYTE_SIZE))))
            if self._branch(157,(self.tilingIns_.aType_.pos == TPosition.TSCM)):
                assign(aUbLength,i32(0))
            if self._branch(158,(self.tilingIns_.bType_.pos == TPosition.TSCM)):
                assign(bUbLength,i32(0))
            if self._branch(159,((self.tilingIns_.aType_.type == CubeFormat.ND) or (self.tilingIns_.bType_.type == CubeFormat.ND))):
                assign(ubSize,i32(i32((ubSize.v + max(aUbLength.v, bUbLength.v)))))
            if self._branch(160,(self.tilingIns_.socVersion == 'ASCEND310B')):
                return
            if self._branch(161,(self.tilingIns_.cType_.pos == TPosition.GM)):
                assign(ubSize,i32(u32((ubSize.v + u32((u32(i32((self.tilingIns_.tiling_.baseM * self.tilingIns_.tiling_.baseN))) * cTypeSize.v))))))
                if self._branch(162,((self.tilingIns_.cType_.type == CubeFormat.ND) and (u32(cmod(u32((u32(self.tilingIns_.tiling_.singleCoreN) * cTypeSize.v)),u32(C0_BYTE_SIZE))) != u32(0)))):
                    assign(ubSize,i32(i32((ubSize.v + C0_BYTE_SIZE))))
            if self._branch(163,((self.tilingIns_.cType_.pos == TPosition.VECCALC) and (self.tilingIns_.cType_.type != CubeFormat.NZ))):
                assign(ubSize,i32(u32((ubSize.v + u32((u32(i32((self.tilingIns_.tiling_.baseM * self.tilingIns_.tiling_.baseN))) * cTypeSize.v))))))
            if self._branch(164,((self.tilingIns_.deqType == DequantType.TENSOR) and (self.tilingIns_.cType_.type == CubeFormat.NZ))):
                assign(ubSize,i32(i32((ubSize.v + i32(u32((u32(self.tilingIns_.tiling_.baseN) * DTYPE_BYTE_TAB[DataType.DT_UINT64])))))))
        return
    
    def GetBankConflictSize_4(self, l1Status, l0Status, length, isAMatrix):
        'Source: MatmulTilingAlgorithm::GetBankConflictSize; lines 1486-1540'
        self._visit('GetBankConflictSize')
        isAMatrix = Cell(bool(isAMatrix))
        blockSize = Cell(i32(32))
        bankLen = Cell(i32(512))
        isBankConflict = Cell(bool(False))
        bankConflictSize = Cell(i32(0))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        if self._branch(165,isAMatrix.v):
            if self._branch(166,self.tilingIns_.aType_.isTrans):
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((i32((l1Status.v.mAL1 * l0Status.v.mL0)) * C0_SIZE)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.kL0 * reduceSize.v)) * C0_SIZE)) * MathUtil.CeilDivision(l1Status.v.kAL1, l0Status.v.kL0)))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
            else:
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((i32((MathUtil.CeilDivision(l1Status.v.kAL1, l0Status.v.kL0) * l0Status.v.kL0)) * reduceSize.v)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.mL0 * C0_SIZE)) * C0_SIZE)) * l1Status.v.mAL1))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
        else:
            if self._branch(167,self.tilingIns_.bType_.isTrans):
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((i32((MathUtil.CeilDivision(l1Status.v.kBL1, l0Status.v.kL0) * l0Status.v.kL0)) * reduceSize.v)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.nL0 * C0_SIZE)) * C0_SIZE)) * l1Status.v.nBL1))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
            else:
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((i32((l1Status.v.nBL1 * l0Status.v.nL0)) * C0_SIZE)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.kL0 * reduceSize.v)) * C0_SIZE)) * MathUtil.CeilDivision(l1Status.v.kBL1, l0Status.v.kL0)))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
        if self._branch(168,isBankConflict.v):
            assign(length,i32(i32((length.v + bankConflictSize.v))))
    
    def GetBankConflictSize_2(self, length, isAMatrix):
        'Source: MatmulTilingAlgorithm::GetBankConflictSize; lines 1542-1592'
        self._visit('GetBankConflictSize')
        isAMatrix = Cell(bool(isAMatrix))
        blockSize = Cell(i32(32))
        bankLen = Cell(i32(512))
        isBankConflict = Cell(bool(False))
        bankConflictSize = Cell(i32(0))
        if self._branch(169,isAMatrix.v):
            if self._branch(170,self.tilingIns_.aType_.isTrans):
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((self.tilingIns_.tiling_.stepM * self.tilingIns_.tiling_.baseM)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((self.tilingIns_.tiling_.baseK * C0_SIZE)) * self.tilingIns_.tiling_.stepKa))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
            else:
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((self.tilingIns_.tiling_.stepKa * self.tilingIns_.tiling_.baseK)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((self.tilingIns_.tiling_.baseM * C0_SIZE)) * self.tilingIns_.tiling_.stepM))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
        else:
            if self._branch(171,self.tilingIns_.bType_.isTrans):
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((self.tilingIns_.tiling_.stepKb * self.tilingIns_.tiling_.baseK)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((self.tilingIns_.tiling_.baseN * C0_SIZE)) * self.tilingIns_.tiling_.stepN))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
            else:
                assign(isBankConflict,bool((True if (i32(cmod(i32((MathUtil.CeilDivision(i32((self.tilingIns_.tiling_.stepN * self.tilingIns_.tiling_.baseN)), C0_SIZE) * blockSize.v)),bankLen.v)) == 0) else False)))
                assign(bankConflictSize,i32(i32(u32(cdiv(u32((u32(i32((i32((self.tilingIns_.tiling_.baseK * C0_SIZE)) * self.tilingIns_.tiling_.stepKb))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
        if self._branch(172,isBankConflict.v):
            assign(length,i32(i32((length.v + bankConflictSize.v))))
    
    def GetAL1UbSize(self, l1Status, l0Status):
        'Source: MatmulTilingAlgorithm::GetAL1UbSize; lines 1594-1611'
        self._visit('GetAL1UbSize')
        a1Length = Cell(i32(0))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        if self._branch(173,self.IsUbNd2Nz()):
            if self._branch(174,(self.tilingIns_.aType_.type == CubeFormat.ND)):
                assign(a1Length,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.mL0 * C0_SIZE)) * l0Status.v.kL0)) * reduceSize.v))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
                if self._branch(175,(self.tilingIns_.mmConfigType == 1)):
                    assign(a1Length,i32(i32((i32((a1Length.v * MathUtil.CeilDivision(l1Status.v.kAL1, l0Status.v.kL0))) * l1Status.v.mAL1))))
                self.GetBankConflictSize_4(l1Status, l0Status, a1Length, True)
        return a1Length.v
    
    def GetBL1UbSize(self, l1Status, l0Status):
        'Source: MatmulTilingAlgorithm::GetBL1UbSize; lines 1613-1630'
        self._visit('GetBL1UbSize')
        b1Length = Cell(i32(0))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        if self._branch(176,self.IsUbNd2Nz()):
            if self._branch(177,(self.tilingIns_.bType_.type == CubeFormat.ND)):
                assign(b1Length,i32(i32(u32(cdiv(u32((u32(i32((i32((i32((l0Status.v.nL0 * C0_SIZE)) * l0Status.v.kL0)) * reduceSize.v))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
                if self._branch(178,(self.tilingIns_.mmConfigType == 1)):
                    assign(b1Length,i32(i32((i32((b1Length.v * MathUtil.CeilDivision(l1Status.v.kBL1, l0Status.v.kL0))) * l1Status.v.nBL1))))
                self.GetBankConflictSize_4(l1Status, l0Status, b1Length, False)
        return b1Length.v
    
    def IsUbNd2Nz(self):
        'Source: MatmulTilingAlgorithm::IsUbNd2Nz; lines 1632-1639'
        self._visit('IsUbNd2Nz')
        if self._branch(179,((self.tilingIns_.enVecND2NZ and (self.tilingIns_.mmConfigType == 1)) and (self.tilingIns_.socVersion == 'ASCEND310P'))):
            return True
        return False
    
    def GetTransLength(self, transLength):
        'Source: MatmulTilingAlgorithm::GetTransLength; lines 1641-1689'
        self._visit('GetTransLength')
        a1Length = Cell(i32(0))
        b1Length = Cell(i32(0))
        c1Length = Cell(i32(0))
        biasLength = Cell(i32(0))
        if self._branch(180,(((self.tilingIns_.socVersion == 'ASCEND910') or (self.tilingIns_.socVersion == 'ASCEND310P')) or (self.tilingIns_.socVersion == 'ASCEND310B'))):
            if self._branch(181,(self.tilingIns_.aType_.type == CubeFormat.ND)):
                assign(a1Length,i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseM * self.tilingIns_.tiling_.baseK))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
                if self._branch(182,(self.tilingIns_.mmConfigType == 1)):
                    assign(a1Length,i32(i32((i32((a1Length.v * self.tilingIns_.tiling_.stepKa)) * self.tilingIns_.tiling_.stepM))))
                self.GetBankConflictSize_2(a1Length, True)
            if self._branch(183,((self.tilingIns_.bType_.type == CubeFormat.ND) or (((self.tilingIns_.bType_.dataType == DataType.DT_INT8) and (self.tilingIns_.bType_.type == CubeFormat.NZ)) and (i32(self.tilingIns_.bType_.isTrans) == i32(False))))):
                assign(b1Length,i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseN * self.tilingIns_.tiling_.baseK))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
                if self._branch(184,(self.tilingIns_.mmConfigType == 1)):
                    assign(b1Length,i32(i32((i32((b1Length.v * self.tilingIns_.tiling_.stepKb)) * self.tilingIns_.tiling_.stepN))))
                self.GetBankConflictSize_2(b1Length, False)
            if self._branch(185,((self.tilingIns_.cType_.type == CubeFormat.ND) or (self.tilingIns_.cType_.pos == TPosition.GM))):
                assign(c1Length,i32(i32(u32((u32(i32((self.tilingIns_.tiling_.baseN * self.tilingIns_.tiling_.baseM))) * DTYPE_BYTE_TAB[self.tilingIns_.cType_.dataType])))))
            if self._branch(186,(self.tilingIns_.isBias and (self.tilingIns_.biasType_.pos != TPosition.VECCALC))):
                assign(biasLength,i32(i32(u32((u32(self.tilingIns_.tiling_.baseN) * DTYPE_BYTE_TAB[self.tilingIns_.biasType_.dataType])))))
            if self._branch(187,(self.tilingIns_.aType_.dataType == DataType.DT_INT8)):
                quantLength = Cell(i32(i32(u64((u64(self.tilingIns_.tiling_.baseN) * 8)))))
                assign(biasLength,i32(max(quantLength.v, biasLength.v)))
        assign(transLength,i32(max(max(a1Length.v, b1Length.v), max(c1Length.v, biasLength.v))))
    
    def CheckBaseMN(self):
        'Source: MatmulTilingAlgorithm::CheckBaseMN; lines 1691-1713'
        self._visit('CheckBaseMN')
        if self._branch(188,(((((self.tilingIns_.socVersion == 'ASCEND910B') or (self.tilingIns_.socVersion == 'ASCEND310B')) and self.tilingIns_.isBias) and (self.tilingIns_.baseN > i32((MAX_BIAS_N * C0_SIZE)))) and self.tilingIns_.isSupportL0c2Out)):
            return False
        if self._branch(189,((self.tilingIns_.baseM != i32((-1))) and (self.tilingIns_.baseN != i32((-1))))):
            return (((i32((i32((self.tilingIns_.baseM * self.tilingIns_.baseN)) * FP32_BYTES)) <= self.tilingIns_.bufferPool_.l0CSize) and (i32((self.tilingIns_.baseM * C0_BYTE_SIZE)) <= self.tilingIns_.bufferPool_.l0ASize)) and (i32((self.tilingIns_.baseN * C0_BYTE_SIZE)) <= self.tilingIns_.bufferPool_.l0BSize))
        if self._branch(190,(self.tilingIns_.baseM != i32((-1)))):
            return ((i32((i32((self.tilingIns_.baseM * C0_SIZE)) * FP32_BYTES)) <= self.tilingIns_.bufferPool_.l0CSize) and (i32((self.tilingIns_.baseM * C0_BYTE_SIZE)) <= self.tilingIns_.bufferPool_.l0ASize))
        if self._branch(191,(self.tilingIns_.baseN != i32((-1)))):
            return ((i32((i32((self.tilingIns_.baseN * C0_SIZE)) * FP32_BYTES)) <= self.tilingIns_.bufferPool_.l0CSize) and (i32((self.tilingIns_.baseN * C0_BYTE_SIZE)) <= self.tilingIns_.bufferPool_.l0BSize))
        return True
    
    def GetIteratorOrder(self, singleCoreStatus, singleCoreM, singleCoreN, singleCoreK):
        'Source: MatmulTilingAlgorithm::GetIteratorOrder; lines 1715-1744'
        self._visit('GetIteratorOrder')
        singleCoreM = Cell(i32(singleCoreM))
        singleCoreN = Cell(i32(singleCoreN))
        singleCoreK = Cell(i32(singleCoreK))
        if self._branch(192,(self.tilingIns_.traverse_ != MatrixTraverse.NOSET)):
            return i32((i32(self.tilingIns_.traverse_) - 1))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        fullkAL1Load = Cell(bool((False if (float(f32((f32(singleCoreK.v) / f32(i32((singleCoreStatus.v.l1Status.kAL1 * reduceSize.v)))))) > 1) else True)))
        fullkBL1Load = Cell(bool((False if (float(f32((f32(singleCoreK.v) / f32(i32((singleCoreStatus.v.l1Status.kBL1 * reduceSize.v)))))) > 1) else True)))
        if self._branch(193,((not fullkAL1Load.v) and (not fullkBL1Load.v))):
            return i32((i32(MatrixTraverse.FIRSTM) - 1))
        else:
            if self._branch(194,(fullkAL1Load.v and (not fullkBL1Load.v))):
                return i32((i32(MatrixTraverse.FIRSTN) - 1))
            else:
                if self._branch(195,((not fullkAL1Load.v) and fullkBL1Load.v)):
                    return i32((i32(MatrixTraverse.FIRSTM) - 1))
                else:
                    mLoop = Cell(i32(MathUtil.CeilDivision(singleCoreM.v, i32((i32((singleCoreStatus.v.l1Status.mAL1 * singleCoreStatus.v.l0Status.mL0)) * C0_SIZE)))))
                    nLoop = Cell(i32(MathUtil.CeilDivision(singleCoreN.v, i32((i32((singleCoreStatus.v.l1Status.nBL1 * singleCoreStatus.v.l0Status.nL0)) * C0_SIZE)))))
                    aL1LoadSize = Cell(i32(i32((singleCoreM.v + i32((singleCoreN.v * mLoop.v))))))
                    bL1LoadSize = Cell(i32(i32((singleCoreN.v + i32((singleCoreM.v * nLoop.v))))))
                    return (1 if (aL1LoadSize.v < bL1LoadSize.v) else 0)
    
    def UpdateBlockDimCalculator(self, blockDimRes):
        'Source: MatmulTilingAlgorithm::UpdateBlockDimCalculator; lines 1746-1754'
        self._visit('UpdateBlockDimCalculator')
        if self._branch(196,(blockDimRes.v.totalLoadSize > blockDimRes.v.tmpLoadSize)):
            assign(AttrRef(blockDimRes.v,'bmatSize'),i32(blockDimRes.v.tmpBmatSize))
            assign(AttrRef(blockDimRes.v,'amatSize'),i32(blockDimRes.v.tmpAmatSize))
            assign(AttrRef(blockDimRes.v,'totalLoadSize'),i32(blockDimRes.v.tmpLoadSize))
            assign(AttrRef(blockDimRes.v,'tmpValue'),i32(0))
    
    def CalcLoadSize(self, blockDims, coreStatus, blockDimRes, params):
        'Source: MatmulTilingAlgorithm::CalcLoadSize; lines 1756-1854'
        self._visit('CalcLoadSize')
        assign(AttrRef(blockDimRes.v,'totalLoadSize'),i32(2147483647))
        totalSize = Cell(i32(i32((blockDimRes.v.amatSize + blockDimRes.v.bmatSize))))
        minMNSize = Cell(i32(16))
        minKSize = Cell(i32(64))
        minTotalSize = Cell(i32(128))
        n0 = Cell(i32(min(minMNSize.v, coreStatus.v.n)))
        m0 = Cell(i32(min(minMNSize.v, (0 if (n0.v == 0) else min(coreStatus.v.m, i32(cdiv(minTotalSize.v,n0.v)))))))
        k0 = Cell(i32((min(min(i32(cdiv(minKSize.v,m0.v)), i32(cdiv(minKSize.v,n0.v))), coreStatus.v.k) if ((m0.v != 0) and (n0.v != 0)) else coreStatus.v.k)))
        dbBuffer = Cell(i32(2))
        bothFullLoad = Cell(bool((i64((i64(totalSize.v) * i64(blockDimRes.v.kBytes))) <= i64(self.tilingIns_.bufferPool_.l1Size))))
        afulloadPlsBKFullLoad = Cell(bool((i64((i64(i32((blockDimRes.v.amatSize + i32((n0.v * dbBuffer.v))))) * i64(blockDimRes.v.kBytes))) <= i64(self.tilingIns_.bufferPool_.l1Size))))
        bfulloadPlsaKFullLoad = Cell(bool((i64((i64(i32((blockDimRes.v.bmatSize + i32((m0.v * dbBuffer.v))))) * i64(blockDimRes.v.kBytes))) <= i64(self.tilingIns_.bufferPool_.l1Size))))
        if self._branch(197,((afulloadPlsBKFullLoad.v or bfulloadPlsaKFullLoad.v) or bothFullLoad.v)):
            assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriAmatSize * blockDims.v.n))))
            assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriBmatSize * blockDims.v.m))))
            assign(AttrRef(blockDimRes.v,'tmpLoadSize'),i32(i32((blockDimRes.v.tmpAmatSize + blockDimRes.v.tmpBmatSize))))
            self.UpdateBlockDimCalculator(blockDimRes)
            return
        aKNotfulloadPlsbKNotFullLoad = Cell(bool(((i32((i32((i32((n0.v * blockDimRes.v.kBytes)) + i32((i32((i32((m0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)))) * dbBuffer.v)) > self.tilingIns_.bufferPool_.l1Size) and (i32((i32((i32((m0.v * blockDimRes.v.kBytes)) + i32((i32((i32((n0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)))) * dbBuffer.v)) > self.tilingIns_.bufferPool_.l1Size))))
        if self._branch(198,aKNotfulloadPlsbKNotFullLoad.v):
            assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriAmatSize * MathUtil.CeilDivision(params.v.n32, n0.v)))))
            assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriBmatSize * MathUtil.CeilDivision(params.v.m32, m0.v)))))
            assign(AttrRef(blockDimRes.v,'tmpLoadSize'),i32(i32((blockDimRes.v.tmpAmatSize + blockDimRes.v.tmpBmatSize))))
            self.UpdateBlockDimCalculator(blockDimRes)
            return
        aKfulloadPlsbKFullLoad = Cell(bool((i32((i32((i32((m0.v + n0.v)) * blockDimRes.v.kBytes)) * dbBuffer.v)) <= self.tilingIns_.bufferPool_.l1Size)))
        if self._branch(199,aKfulloadPlsbKFullLoad.v):
            m1 = Cell(i32(i32((MathUtil.CeilDivision(i32((self.tilingIns_.bufferPool_.l1Size - i32((i32((n0.v * blockDimRes.v.kBytes)) * dbBuffer.v)))), i32((i32((blockDimRes.v.kBytes * dbBuffer.v)) * m0.v))) * m0.v))))
            n1 = Cell(i32(i32((MathUtil.CeilDivision(i32((self.tilingIns_.bufferPool_.l1Size - i32((i32((m0.v * blockDimRes.v.kBytes)) * dbBuffer.v)))), i32((i32((blockDimRes.v.kBytes * dbBuffer.v)) * n0.v))) * n0.v))))
            mfirstLoad = Cell(i32(i32((i32((blockDimRes.v.oriAmatSize * blockDims.v.n)) + i32((blockDimRes.v.oriBmatSize * MathUtil.CeilDivision(params.v.m32, m1.v)))))))
            nfirstLoad = Cell(i32(i32((i32((blockDimRes.v.oriBmatSize * blockDims.v.m)) + i32((blockDimRes.v.oriAmatSize * MathUtil.CeilDivision(params.v.n32, n1.v)))))))
            if self._branch(200,(mfirstLoad.v < nfirstLoad.v)):
                assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriAmatSize * blockDims.v.n))))
                assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriBmatSize * MathUtil.CeilDivision(params.v.m32, m1.v)))))
            else:
                assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriAmatSize * MathUtil.CeilDivision(params.v.n32, n1.v)))))
                assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriBmatSize * blockDims.v.m))))
            assign(AttrRef(blockDimRes.v,'tmpLoadSize'),i32(i32((blockDimRes.v.tmpAmatSize + blockDimRes.v.tmpBmatSize))))
            self.UpdateBlockDimCalculator(blockDimRes)
            return
        afulloadPlsbKNotFullLoad = Cell(bool((i32((i32((blockDimRes.v.amatSize * blockDimRes.v.kBytes)) + i32((i32((i32((i32((n0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)) * dbBuffer.v)))) <= self.tilingIns_.bufferPool_.l1Size)))
        aKfulloadPlsbKNotFullLoad = Cell(bool((i32((i32((i32((m0.v * blockDimRes.v.kBytes)) * dbBuffer.v)) + i32((i32((i32((i32((n0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)) * dbBuffer.v)))) <= self.tilingIns_.bufferPool_.l1Size)))
        if self._branch(201,(afulloadPlsbKNotFullLoad.v or aKfulloadPlsbKNotFullLoad.v)):
            assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriAmatSize * blockDims.v.n))))
            assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriBmatSize * MathUtil.CeilDivision(params.v.m32, m0.v)))))
            assign(AttrRef(blockDimRes.v,'tmpLoadSize'),i32(i32((blockDimRes.v.tmpAmatSize + blockDimRes.v.tmpBmatSize))))
            self.UpdateBlockDimCalculator(blockDimRes)
        aKNotfulloadPlsbFullLoad = Cell(bool((i32((i32((blockDimRes.v.bmatSize * blockDimRes.v.kBytes)) + i32((i32((i32((i32((m0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)) * dbBuffer.v)))) <= self.tilingIns_.bufferPool_.l1Size)))
        aKNotfulloadPlsbKFullLoad = Cell(bool((i32((i32((i32((n0.v * blockDimRes.v.kBytes)) * dbBuffer.v)) + i32((i32((i32((i32((m0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE)) * dbBuffer.v)))) <= self.tilingIns_.bufferPool_.l1Size)))
        if self._branch(202,(aKNotfulloadPlsbFullLoad.v or aKNotfulloadPlsbKFullLoad.v)):
            assign(AttrRef(blockDimRes.v,'tmpAmatSize'),i32(i32((blockDimRes.v.oriBmatSize * blockDims.v.m))))
            assign(AttrRef(blockDimRes.v,'tmpBmatSize'),i32(i32((blockDimRes.v.oriAmatSize * MathUtil.CeilDivision(params.v.n32, n0.v)))))
            assign(AttrRef(blockDimRes.v,'tmpLoadSize'),i32(i32((blockDimRes.v.tmpAmatSize + blockDimRes.v.tmpBmatSize))))
            self.UpdateBlockDimCalculator(blockDimRes)
    
    def LoopNumFromSingleCoreToL0(self, coreStatus, blockDimsFactor):
        'Source: MatmulTilingAlgorithm::LoopNumFromSingleCoreToL0; lines 1856-1874'
        self._visit('LoopNumFromSingleCoreToL0')
        if self._branch(203,(not blockDimsFactor.v.IsValid())):
            return 0
        minTotalSize = Cell(i32(128))
        minSize = Cell(i32(64))
        minN0Size = Cell(i32(16))
        n0 = Cell(i32(min(min(minN0Size.v, coreStatus.v.n), minSize.v)))
        m0 = Cell(i32((0 if (n0.v == 0) else min(min(coreStatus.v.m, i32(cdiv(minTotalSize.v,n0.v))), minSize.v))))
        assign(n0,i32((0 if (m0.v == 0) else min(min(coreStatus.v.n, i32(cdiv(minTotalSize.v,m0.v))), minSize.v))))
        assign(m0,i32((0 if (n0.v == 0) else min(min(coreStatus.v.m, i32(cdiv(minTotalSize.v,n0.v))), minSize.v))))
        k0 = Cell(i32((min(min(i32(cdiv(minSize.v,m0.v)), i32(cdiv(minSize.v,n0.v))), coreStatus.v.k) if ((m0.v != 0) and (n0.v != 0)) else coreStatus.v.k)))
        loopNum = Cell(i32(i32((i32((MathUtil.CeilDivision(coreStatus.v.m, m0.v) * MathUtil.CeilDivision(coreStatus.v.n, n0.v))) * MathUtil.CeilDivision(coreStatus.v.k, k0.v)))))
        return loopNum.v
    
    def GetBigPackageCondition(self, coreStatus, blockDimRes, params):
        'Source: MatmulTilingAlgorithm::GetBigPackageCondition; lines 1876-1904'
        self._visit('GetBigPackageCondition')
        if self._branch(204,((i32(self.tilingIns_.bType_.isTrans) == i32(True)) and (i32(self.tilingIns_.aType_.isTrans) == i32(False)))):
            return ATTACH_FLAG_ZERO
        minSize = Cell(i32(16))
        flag = Cell(bool(True))
        if self._branch(205,(i32(self.tilingIns_.bType_.isTrans) == i32(False))):
            if self._branch(206,((params.v.n32 >= minSize.v) and (coreStatus.v.n < minSize.v))):
                assign(flag,bool(False))
        if self._branch(207,self.tilingIns_.aType_.isTrans):
            if self._branch(208,((params.v.m32 >= minSize.v) and (coreStatus.v.m < minSize.v))):
                assign(flag,bool(False))
        if self._branch(209,((not blockDimRes.v.bigPackage) and (not flag.v))):
            return ATTACH_FLAG_ZERO
        else:
            if self._branch(210,((not blockDimRes.v.bigPackage) and flag.v)):
                return ATTACH_FLAG_TWO
            else:
                if self._branch(211,(blockDimRes.v.bigPackage and (not flag.v))):
                    return ATTACH_FLAG_ONE
                else:
                    return ATTACH_FLAG_ZERO
    
    def GetBlockDimHelper(self, blockDim, coreStatus, blockDimRes, params):
        'Source: MatmulTilingAlgorithm::GetBlockDimHelper; lines 1906-1972'
        self._visit('GetBlockDimHelper')
        assign(AttrRef(blockDimRes.v,'kNum'),i32((0 if (blockDim.v.k == 0) else i32((i32((i32(cdiv(params.v.k32,blockDim.v.k)) * C0_SIZE)) * REDUCE_BLOCK_SIZE)))))
        assign(AttrRef(blockDimRes.v,'kBytes'),i32(i32((blockDimRes.v.kNum * INPUTDTYPE_BYTES))))
        assign(AttrRef(coreStatus.v,'batch'),i32(MathUtil.CeilDivision(params.v.batch32, blockDim.v.batch)))
        assign(AttrRef(coreStatus.v,'m'),i32(MathUtil.CeilDivision(params.v.m32, blockDim.v.m)))
        assign(AttrRef(coreStatus.v,'n'),i32(MathUtil.CeilDivision(params.v.n32, blockDim.v.n)))
        assign(AttrRef(coreStatus.v,'k'),i32((0 if (blockDim.v.k == 0) else i32(cdiv(params.v.k32,blockDim.v.k)))))
        if self._branch(212,self.tilingIns_.enableSplitK_):
            if self._branch(213,(params.v.kMapped != params.v.k32)):
                assign(AttrRef(blockDimRes.v,'kNum'),i32(i32((i32((i32((i32(cdiv(params.v.kMapped,blockDim.v.k)) * NUM_TWO)) * C0_SIZE)) * REDUCE_BLOCK_SIZE))))
                assign(AttrRef(coreStatus.v,'k'),i32(i32((i32(cdiv(params.v.kMapped,blockDim.v.k)) * NUM_TWO))))
        assign(AttrRef(blockDimRes.v,'oriAmatSize'),i32(i32((params.v.batch32 * params.v.m32))))
        assign(AttrRef(blockDimRes.v,'oriBmatSize'),i32((i32((params.v.batch32 * params.v.n32)) if (params.v.oriShapeBbatch > i64(1)) else params.v.n32)))
        assign(AttrRef(blockDimRes.v,'amatSize'),i32(i32((coreStatus.v.batch * coreStatus.v.m))))
        assign(AttrRef(blockDimRes.v,'bmatSize'),i32((i32((coreStatus.v.batch * coreStatus.v.n)) if (params.v.oriShapeBbatch > i64(1)) else coreStatus.v.n)))
        assign(AttrRef(blockDimRes.v,'tmpValue'),i32(0))
        self.CalcLoadSize(blockDim, coreStatus, blockDimRes, params)
        if self._branch(214,self.tilingIns_.enableSplitK_):
            assign(AttrRef(blockDimRes.v,'totalLoadSize'),i32(i32((blockDimRes.v.totalLoadSize * coreStatus.v.k))))
        bigpackageFlag = Cell(i32(self.GetBigPackageCondition(coreStatus, blockDimRes, params)))
        updateConditionBp = Cell(bool((False if (bigpackageFlag.v == 0) else True)))
        updateConditionBp2 = Cell(bool((True if (bigpackageFlag.v == 2) else False)))
        updateConditionBp3 = Cell(bool((False if (bigpackageFlag.v == 1) else True)))
        loopNum = Cell(i32(self.LoopNumFromSingleCoreToL0(coreStatus, blockDim)))
        updateConditionCoreUsed = Cell(bool(((not updateConditionBp.v) and ((loopNum.v < blockDimRes.v.loopNumToL0) or ((blockDim.v.ReduceMul() > blockDimRes.v.coreUse) and (loopNum.v == blockDimRes.v.loopNumToL0))))))
        updateConditionLoadsize = Cell(bool((((not updateConditionCoreUsed.v) and (blockDim.v.ReduceMul() == blockDimRes.v.coreUse)) and (blockDimRes.v.totalLoadSize < blockDimRes.v.minLoadSize))))
        orgBatchM = Cell(i32((blockDimRes.v.batchDimFactor if (params.v.oriShapeAbatch > i64(1)) else blockDimRes.v.mDimFactor)))
        curBatchM = Cell(i32((blockDim.v.batch if (params.v.oriShapeAbatch > i64(1)) else blockDim.v.m)))
        updateConditionBatchNDim = Cell(bool(((((not updateConditionCoreUsed.v) and (blockDim.v.ReduceMul() == blockDimRes.v.coreUse)) and (blockDimRes.v.totalLoadSize == blockDimRes.v.minLoadSize)) and ((i32((blockDimRes.v.nDimFactor * orgBatchM.v)) < i32((curBatchM.v * blockDim.v.n))) or ((i32((blockDimRes.v.nDimFactor * orgBatchM.v)) == i32((curBatchM.v * blockDim.v.n))) and (blockDimRes.v.batchDimFactor < blockDim.v.batch))))))
        policyCondition = Cell(bool(self.UserPolicy((TilingPolicy.FIXED_B_TSCM if (self.tilingIns_.bType_.pos == TPosition.TSCM) else TilingPolicy.NO_POLICY), coreStatus, blockDimRes)))
        if self._branch(215,(((((updateConditionBp2.v or updateConditionCoreUsed.v) or updateConditionLoadsize.v) or updateConditionBatchNDim.v) and policyCondition.v) and updateConditionBp3.v)):
            assign(AttrRef(blockDimRes.v,'minLoadSize'),i32(blockDimRes.v.totalLoadSize))
            assign(AttrRef(blockDimRes.v,'nDimFactor'),i32(blockDim.v.n))
            assign(AttrRef(blockDimRes.v,'batchDimFactor'),i32(blockDim.v.batch))
            assign(AttrRef(blockDimRes.v,'mDimFactor'),i32(blockDim.v.m))
            assign(AttrRef(blockDimRes.v,'kDimFactor'),i32(blockDim.v.k))
            assign(AttrRef(blockDimRes.v,'coreUse'),i32(blockDim.v.ReduceMul()))
            assign(AttrRef(blockDimRes.v,'loopNumToL0'),i32(loopNum.v))
            assign(AttrRef(blockDimRes.v,'finalValue'),i32(blockDimRes.v.tmpValue))
            minSize = Cell(i32(16))
            assign(AttrRef(blockDimRes.v,'bigPackage'),bool(((((coreStatus.v.n >= minSize.v) if (not self.tilingIns_.bType_.isTrans) else True) and ((coreStatus.v.m >= minSize.v) if self.tilingIns_.aType_.isTrans else True)) and (i32((i32((blockDim.v.n * blockDim.v.m)) * blockDim.v.k)) > 1))))
            assign(AttrRef(self,'splitCoreFlag_'),bool(True))
    
    def UserPolicy(self, policy, coreStatus, blockDimRes):
        'Source: MatmulTilingAlgorithm::UserPolicy; lines 1974-2008'
        self._visit('UserPolicy')
        policy = Cell(policy)
        minMNSize = Cell(i32(16))
        minKSize = Cell(i32(64))
        minTotalSize = Cell(i32(128))
        n0 = Cell(i32(min(minMNSize.v, coreStatus.v.n)))
        m0 = Cell(i32(min(minMNSize.v, (0 if (n0.v == 0) else min(coreStatus.v.m, i32(cdiv(minTotalSize.v,n0.v)))))))
        k0 = Cell(i32((min(min(i32(cdiv(minKSize.v,m0.v)), i32(cdiv(minKSize.v,n0.v))), coreStatus.v.k) if ((m0.v != 0) and (n0.v != 0)) else coreStatus.v.k)))
        if self._branch(216,(policy.v == TilingPolicy.FIXED_B_TSCM)):
            alignFactor = Cell(i32(MathUtil.CeilDivision(self.tilingIns_.alignSingleN, C0_SIZE)))
            if self._branch(217,(coreStatus.v.n < alignFactor.v)):
                return False
            alignNLength = Cell(i32(MathUtil.Align(coreStatus.v.n, alignFactor.v)))
            bMatrixSize = Cell(i32(i32((i32((alignNLength.v * blockDimRes.v.kBytes)) * 2))))
            aMatrixSize = Cell(i32(i32((i32((i32((m0.v * k0.v)) * C0_SIZE)) * C0_BYTE_SIZE))))
            biasSize = Cell(i32(0))
            if self._branch(218,(self.tilingIns_.isSupportL0c2Out and self.tilingIns_.isBias)):
                assign(biasSize,i32(i32(u32((u32(i32((alignNLength.v * C0_SIZE))) * DTYPE_BYTE_TAB[self.tilingIns_.biasType_.dataType])))))
            if self._branch(219,(i32((i32((bMatrixSize.v + aMatrixSize.v)) + biasSize.v)) <= self.tilingIns_.bufferPool_.l1Size)):
                return True
            else:
                return False
        else:
            if self._branch(220,(policy.v == TilingPolicy.FIXED_A_TSCM)):
                return False
            else:
                if self._branch(221,(policy.v == TilingPolicy.FIXED_A_B_TSCM)):
                    return False
                else:
                    return True
    
    def PreProcessMiniShape(self, opType, coreStatus, params, coreNum, splitKFlag):
        'Source: MatmulTilingAlgorithm::PreProcessMiniShape; lines 2010-2048'
        self._visit('PreProcessMiniShape')
        splitKFlag = Cell(bool(splitKFlag))
        opType.v
        miniL0cThreshold = Cell(i32(i32(cdiv(i32(cdiv(self.tilingIns_.bufferPool_.l0CSize,MIN_FRACTAL_SIZE)),FP32_BYTES))))
        miniL0abThreshold = Cell(i32(i32(cdiv(self.tilingIns_.bufferPool_.l0ASize,i32((C0_SIZE * C0_BYTE_SIZE))))))
        specialScenario = Cell(bool(False))
        if self._branch(222,(params.v.n32 > MIN_MTE1_LOAD)):
            assign(specialScenario,bool((specialScenario.v or (splitKFlag.v and (u32((u32(params.v.nMapped) & u32(i32((MIN_MTE1_LOAD - 1))))) != u32(0))))))
        if self._branch(223,(params.v.m32 > MIN_MTE1_LOAD)):
            assign(specialScenario,bool((specialScenario.v or (splitKFlag.v and (u32((u32(params.v.mMapped) & u32(i32((MIN_MTE1_LOAD - 1))))) != u32(0))))))
        if self._branch(224,(((((i32((i32((params.v.batch32 * params.v.n32)) * params.v.m32)) <= coreNum.v) and (i32((params.v.m32 * params.v.k32)) <= miniL0abThreshold.v)) and (i32((params.v.n32 * params.v.k32)) <= miniL0abThreshold.v)) and (i32((params.v.m32 * params.v.n32)) <= miniL0cThreshold.v)) and (not specialScenario.v))):
            assign(AttrRef(coreStatus.v,'batchDim'),i32(params.v.batch32))
            assign(AttrRef(coreStatus.v,'nDim'),i32((1 if (params.v.n32 <= MIN_MTE1_LOAD) else i32(cdiv(params.v.nMapped,MIN_MTE1_LOAD)))))
            assign(AttrRef(coreStatus.v,'mDim'),i32((1 if (params.v.m32 <= MIN_MTE1_LOAD) else i32(cdiv(params.v.mMapped,MIN_MTE1_LOAD)))))
            kDimCandidate = Cell([0, 0])
            self.GetTwoFactors(kDimCandidate, coreStatus.v.kDim, params.v.k32, coreNum.v)
            assign(AttrRef(coreStatus.v,'kDim'),i32((1 if ((params.v.k32 <= MIN_MTE1_LOAD) or (not splitKFlag.v)) else (kDimCandidate.v[1] if (kDimCandidate.v[1] > 1) else kDimCandidate.v[0]))))
            assign(AttrRef(coreStatus.v,'batch'),i32(1))
            assign(AttrRef(coreStatus.v,'n'),i32((params.v.n32 if (coreStatus.v.nDim == 1) else MathUtil.CeilDivision(params.v.nMapped, coreStatus.v.nDim))))
            assign(AttrRef(coreStatus.v,'m'),i32((params.v.m32 if (coreStatus.v.mDim == 1) else MathUtil.CeilDivision(params.v.mMapped, coreStatus.v.mDim))))
            assign(AttrRef(coreStatus.v,'k'),i32((params.v.k32 if (coreStatus.v.kDim == 1) else MathUtil.CeilDivision(params.v.kMapped, coreStatus.v.kDim))))
            assign(AttrRef(params.v,'nonFactorK'),bool((False if (coreStatus.v.kDim == 0) else (False if (i32(cmod(params.v.k32,coreStatus.v.kDim)) == 0) else True))))
            return True
        return False
    
    def UpdateMultiCore(self, opType, params, coreStatus, blockDimRes):
        'Source: MatmulTilingAlgorithm::UpdateMultiCore; lines 2050-2066'
        self._visit('UpdateMultiCore')
        opType.v
        assign(AttrRef(coreStatus.v,'batchDim'),i32(min(MathUtil.CeilDivision(params.v.batch32, coreStatus.v.batch), self.numOfBlock_)))
        assign(AttrRef(coreStatus.v,'nDim'),i32(min(MathUtil.CeilDivision(params.v.n32, coreStatus.v.n), self.numOfBlock_)))
        assign(AttrRef(coreStatus.v,'mDim'),i32(min(MathUtil.CeilDivision(params.v.m32, coreStatus.v.m), self.numOfBlock_)))
        if self._branch(225,self.tilingIns_.enableSplitK_):
            assign(AttrRef(coreStatus.v,'kDim'),i32(min(MathUtil.CeilDivision(params.v.k32, coreStatus.v.k), self.numOfBlock_)))
        else:
            assign(AttrRef(coreStatus.v,'kDim'),i32(blockDimRes.v.kDimFactor))
        self.UpdateBufferSize((TilingPolicy.FIXED_B_TSCM if (self.tilingIns_.bType_.pos == TPosition.TSCM) else TilingPolicy.NO_POLICY), coreStatus)
    
    def UpdateBufferSize(self, policy, coreStatus):
        'Source: MatmulTilingAlgorithm::UpdateBufferSize; lines 2068-2083'
        self._visit('UpdateBufferSize')
        policy = Cell(policy)
        if self._branch(226,(policy.v == TilingPolicy.NO_POLICY)):
            return
        else:
            if self._branch(227,(policy.v == TilingPolicy.FIXED_B_TSCM)):
                bMatrixSize = Cell(i32(i32((i32((i32((i32((MathUtil.Align(coreStatus.v.n, MathUtil.CeilDivision(self.tilingIns_.alignSingleN, C0_SIZE)) * coreStatus.v.k)) * C0_SIZE)) * C0_BYTE_SIZE)) * 2))))
                assign(AttrRef(self.tilingIns_.bufferPool_,'l1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - bMatrixSize.v))))
            else:
                if self._branch(228,(policy.v == TilingPolicy.FIXED_A_TSCM)):
                    aMatrixSize = Cell(i32(i32((i32((i32((i32((coreStatus.v.m * coreStatus.v.k)) * C0_SIZE)) * C0_BYTE_SIZE)) * 2))))
                    assign(AttrRef(self.tilingIns_.bufferPool_,'l1Size'),i32(i32((self.tilingIns_.bufferPool_.l1Size - aMatrixSize.v))))
                else:
                    return
    
    def IsInvalidFactor(self, factor):
        'Source: MatmulTilingAlgorithm::IsInvalidFactor; lines 2085-2088'
        self._visit('IsInvalidFactor')
        factor = Cell(i32(factor))
        return ((factor.v > self.numOfBlock_) or (factor.v <= 0))
    
    def AddOptimalFactors(self, opType, params, blockDimRes):
        'Source: MatmulTilingAlgorithm::AddOptimalFactors; lines 2090-2106'
        self._visit('AddOptimalFactors')
        opType.v
        coreNum = Cell(i32(self.numOfBlock_))
        mnCore = Cell(i32(MathUtil.CeilDivision(coreNum.v, params.v.batch32)))
        if self._branch(229,(mnCore.v > 1)):
            optPoint = Cell(f32(sqrt(f32((f32((f32((f32(params.v.m32) + 0)) / f32(params.v.n32))) * f32(mnCore.v))))))
            mdim = Cell(i32(i32(ceil(optPoint.v))))
            ndim = Cell(i32(i32(ceil(f32((f32(mnCore.v) / optPoint.v))))))
            MathUtil.AddFactor(AttrRef(blockDimRes.v,'mDimFactors'), mdim.v)
            MathUtil.AddFactor(AttrRef(blockDimRes.v,'mDimFactors'), (1 if (ndim.v == 0) else i32(cdiv(mnCore.v,ndim.v))))
            MathUtil.AddFactor(AttrRef(blockDimRes.v,'nDimFactors'), ndim.v)
            MathUtil.AddFactor(AttrRef(blockDimRes.v,'nDimFactors'), (1 if (mdim.v == 0) else i32(cdiv(mnCore.v,mdim.v))))
    
    def GenBlockDimsMapFactors(self, opType, params, blockDimRes):
        'Source: MatmulTilingAlgorithm::GenBlockDimsMapFactors; lines 2108-2128'
        self._visit('GenBlockDimsMapFactors')
        coreNum = Cell(i32(self.numOfBlock_))
        None
        None
        None
        None
        MathUtil.GetBlockFactors(AttrRef(blockDimRes.v,'batchDimFactors'), params.v.batch32, params.v.batchMapped, coreNum.v, min(coreNum.v, params.v.batch32))
        MathUtil.GetBlockFactors(AttrRef(blockDimRes.v,'mDimFactors'), params.v.m32, params.v.mMapped, coreNum.v, min(coreNum.v, params.v.m32))
        MathUtil.GetBlockFactors(AttrRef(blockDimRes.v,'nDimFactors'), params.v.n32, params.v.nMapped, coreNum.v, min(coreNum.v, params.v.n32))
        if self._branch(230,(not self.tilingIns_.enableSplitK_)):
            blockDimRes.v.kDimFactors.append(1)
            assign(AttrRef(params.v,'kMapped'),i32(params.v.k32))
        else:
            MathUtil.GetBlockFactors(AttrRef(blockDimRes.v,'kDimFactors'), params.v.k32, params.v.kMapped, coreNum.v, coreNum.v)
        self.AddOptimalFactors(opType, params, blockDimRes)
    
    def GetBlockDim(self, opType, params, coreStatus, blockDimRes):
        'Source: MatmulTilingAlgorithm::GetBlockDim; lines 2130-2191'
        self._visit('GetBlockDim')
        if self._branch(231,self.PreProcessMiniShape(opType, coreStatus, params, AttrRef(self,'numOfBlock_'), self.tilingIns_.enableSplitK_)):
            assign(AttrRef(coreStatus.v,'batchDim'),i32(MathUtil.CeilDivision(params.v.batch32, coreStatus.v.batch)))
            assign(AttrRef(coreStatus.v,'nDim'),i32(MathUtil.CeilDivision(params.v.n32, coreStatus.v.n)))
            assign(AttrRef(coreStatus.v,'mDim'),i32(MathUtil.CeilDivision(params.v.m32, coreStatus.v.m)))
            assign(AttrRef(coreStatus.v,'kDim'),i32(MathUtil.CeilDivision(params.v.k32, coreStatus.v.k)))
            self.UpdateBufferSize((TilingPolicy.FIXED_B_TSCM if (self.tilingIns_.bType_.pos == TPosition.TSCM) else TilingPolicy.NO_POLICY), coreStatus)
            assign(AttrRef(self,'splitCoreFlag_'),bool(True))
            return
        self.GenBlockDimsMapFactors(opType, params, blockDimRes)
        for _item_10 in blockDimRes.v.batchDimFactors:
            bFactor = Cell(_item_10)
            for _item_11 in blockDimRes.v.nDimFactors:
                nFactor = Cell(_item_11)
                if self._branch(232,self.IsInvalidFactor(i32((bFactor.v * nFactor.v)))):
                    continue
                for _item_12 in blockDimRes.v.mDimFactors:
                    mFactor = Cell(_item_12)
                    if self._branch(233,self.IsInvalidFactor(i32((i32((bFactor.v * nFactor.v)) * mFactor.v)))):
                        continue
                    for _item_13 in blockDimRes.v.kDimFactors:
                        kFactor = Cell(_item_13)
                        if self._branch(234,self.IsInvalidFactor(i32((i32((i32((bFactor.v * nFactor.v)) * mFactor.v)) * kFactor.v)))):
                            continue
                        blockDim = Cell(DimFactor(bFactor.v, mFactor.v, kFactor.v, nFactor.v))
                        self.GetBlockDimHelper(blockDim, coreStatus, blockDimRes, params)
        assign(AttrRef(coreStatus.v,'batch'),i32(MathUtil.CeilDivision(params.v.batch32, blockDimRes.v.batchDimFactor)))
        assign(AttrRef(coreStatus.v,'n'),i32(MathUtil.CeilDivision(params.v.n32, blockDimRes.v.nDimFactor)))
        assign(AttrRef(coreStatus.v,'m'),i32(MathUtil.CeilDivision(params.v.m32, blockDimRes.v.mDimFactor)))
        assign(AttrRef(coreStatus.v,'k'),i32(MathUtil.CeilDivision(params.v.k32, blockDimRes.v.kDimFactor)))
        if self._branch(235,self.cfg.factorSplit):
            n = Cell(i32(MathUtil.FindBestSingleCore(params.v.n32, params.v.nMapped, blockDimRes.v.nDimFactor, False)))
            m = Cell(i32(MathUtil.FindBestSingleCore(params.v.m32, params.v.mMapped, blockDimRes.v.mDimFactor, False)))
            k = Cell(i32(MathUtil.FindBestSingleCore(params.v.k32, params.v.kMapped, blockDimRes.v.kDimFactor, True)))
            needCoreNum = Cell(i32(i32((i32((i32((MathUtil.CeilDivision(params.v.batch32, coreStatus.v.batch) * MathUtil.CeilDivision(params.v.n32, n.v))) * MathUtil.CeilDivision(params.v.m32, m.v))) * MathUtil.CeilDivision(params.v.k32, k.v)))))
            if self._branch(236,(i32(self.IsInvalidFactor(needCoreNum.v)) == i32(False))):
                assign(AttrRef(coreStatus.v,'n'),i32(n.v))
                assign(AttrRef(coreStatus.v,'m'),i32(m.v))
                assign(AttrRef(coreStatus.v,'k'),i32(k.v))
        assign(AttrRef(params.v,'nonFactorK'),bool((False if (params.v.k32 == params.v.kMapped) else True)))
        self.UpdateMultiCore(opType, params, coreStatus, blockDimRes)
    
    def NonFactorMap(self, opType, param, blockDimRes):
        'Source: MatmulTilingAlgorithm::NonFactorMap; lines 2193-2222'
        self._visit('NonFactorMap')
        opType.v
        assign(AttrRef(param.v,'batchMapped'),i32(param.v.batch32))
        assign(AttrRef(param.v,'mMapped'),i32(param.v.m32))
        assign(AttrRef(param.v,'kMapped'),i32(param.v.k32))
        assign(AttrRef(param.v,'nMapped'),i32(param.v.n32))
        if self._branch(237,self.tilingIns_.enableSplitK_):
            kFactorLess64Cnt = Cell(i32(0))
            kFactorLess1024Cnt = Cell(i32(0))
            MathUtil.GetFactorCnt(param.v.k32, kFactorLess64Cnt, 1, L0_FACTOR_LIMIT)
            MathUtil.GetFactorCnt(param.v.k32, kFactorLess1024Cnt, i32((L0_FACTOR_LIMIT + 1)), L1_FACTOR_LIMIT)
            if self._branch(238,(((param.v.k32 > L0_FACTOR_LIMIT) and (kFactorLess64Cnt.v <= L0_FACTOR_NUM_LIMIT)) or ((param.v.k32 > L1_FACTOR_LIMIT) and (i32((kFactorLess64Cnt.v + kFactorLess1024Cnt.v)) <= L1_FACTOR_NUM_LIMIT)))):
                assign(AttrRef(param.v,'kMapped'),i32(MathUtil.MapShape(param.v.k32, False)))
        else:
            MathUtil.GetFactorCnt(param.v.batch32, AttrRef(blockDimRes.v,'batchFactorCnt'), 1, self.numOfBlock_)
            if self._branch(239,((param.v.batch32 > 1) and (blockDimRes.v.batchFactorCnt <= L0_FACTOR_NUM_LIMIT))):
                assign(AttrRef(param.v,'batchMapped'),i32(MathUtil.MapShape(param.v.batch32, True)))
            assign(AttrRef(param.v,'mMapped'),i32(MathUtil.MapShape(param.v.m32, True)))
            assign(AttrRef(param.v,'nMapped'),i32(MathUtil.MapShape(param.v.n32, True)))
    
    def FillParam(self, param):
        'Source: MatmulTilingAlgorithm::FillParam; lines 2224-2255'
        self._visit('FillParam')
        assign(AttrRef(param.v,'oriShapeM'),i64(i64(self.tilingIns_.orgM)))
        assign(AttrRef(param.v,'oriShapeN'),i64(i64(self.tilingIns_.orgN)))
        assign(AttrRef(param.v,'oriShapeKa'),i64(i64(self.tilingIns_.orgKa)))
        assign(AttrRef(param.v,'oriShapeKb'),i64(i64(self.tilingIns_.orgKb)))
        realM = Cell(i32(1))
        realN = Cell(i32(1))
        realK = Cell(i32(1))
        if self._branch(240,(((self.tilingIns_.singleCoreM != i32((-1))) or (self.tilingIns_.singleCoreK != i32((-1)))) or (self.tilingIns_.singleCoreN != i32((-1))))):
            assign(realM,i32((self.tilingIns_.singleCoreM if (self.tilingIns_.singleCoreM != i32((-1))) else self.tilingIns_.singleM)))
            assign(realK,i32((self.tilingIns_.singleCoreK if (self.tilingIns_.singleCoreK != i32((-1))) else self.tilingIns_.singleK)))
            assign(realN,i32((self.tilingIns_.singleCoreN if (self.tilingIns_.singleCoreN != i32((-1))) else self.tilingIns_.singleN)))
            assign(AttrRef(self,'singelBlockDim_'),bool(True))
            assign(AttrRef(self,'numOfBlock_'),i32(1))
        else:
            assign(realM,i32(self.tilingIns_.singleM))
            assign(realK,i32(self.tilingIns_.singleK))
            assign(realN,i32(self.tilingIns_.singleN))
            assign(AttrRef(self,'singelBlockDim_'),bool(False))
            assign(AttrRef(self,'numOfBlock_'),i32(self.tilingIns_.blockDim))
        reduceBlockSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        assign(AttrRef(param.v,'k32'),i32(MathUtil.CeilDivision(realK.v, reduceBlockSize.v)))
        assign(AttrRef(param.v,'m32'),i32(MathUtil.CeilDivision(realM.v, C0_SIZE)))
        assign(AttrRef(param.v,'n32'),i32(MathUtil.CeilDivision(realN.v, C0_SIZE)))
        assign(AttrRef(param.v,'mMapped'),i32(MathUtil.MapShape(param.v.m32, True)))
        assign(AttrRef(param.v,'kMapped'),i32(MathUtil.MapShape(param.v.k32, True)))
        assign(AttrRef(param.v,'nMapped'),i32(MathUtil.MapShape(param.v.n32, True)))
    
    def CheckFinaleParams(self, coreStatus):
        'Source: MatmulTilingAlgorithm::CheckFinaleParams; lines 2257-2297'
        self._visit('CheckFinaleParams')
        coreStatus.v
        stepM = Cell(i32(self.tilingIns_.tiling_.stepM))
        stepN = Cell(i32(self.tilingIns_.tiling_.stepN))
        depthA1 = Cell(i32(self.tilingIns_.tiling_.depthA1))
        depthB1 = Cell(i32(self.tilingIns_.tiling_.depthB1))
        l1Size = Cell(i32(self.tilingIns_.tiling_.shareL1Size))
        l0CSize = Cell(i32(self.tilingIns_.tiling_.shareL0CSize))
        uBSize = Cell(i32(self.tilingIns_.tiling_.shareUbSize))
        if self._branch(241,((((stepM.v == 0) or (stepN.v == 0)) or (depthA1.v == 0)) or (depthB1.v == 0))):
            0
            return False
        if self._branch(242,((stepM.v > depthA1.v) or (stepN.v > depthB1.v))):
            0
            return False
        if self._branch(243,(((l1Size.v > self.tilingIns_.bufferPool_.l1Size) or (l0CSize.v > self.tilingIns_.bufferPool_.l0CSize)) or (uBSize.v > self.tilingIns_.bufferPool_.ubSize))):
            0
            return False
        dateDtypeSize = Cell(i32(i32(DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])))
        if self._branch(244,((self.tilingIns_.tiling_.BatchNum > 0) and (i32((i32(cdiv(i32((i32((i32((i32((self.tilingIns_.tiling_.singleCoreM * self.tilingIns_.tiling_.singleCoreK)) + i32((self.tilingIns_.tiling_.singleCoreN * self.tilingIns_.tiling_.singleCoreK)))) * self.tilingIns_.tiling_.BatchNum)) * dateDtypeSize.v)),BITS_PER_BYTE)) + i32(cdiv(i32((i32((self.tilingIns_.tiling_.singleCoreN * self.tilingIns_.tiling_.BatchNum)) * dateDtypeSize.v)),BITS_PER_BYTE)))) > self.tilingIns_.bufferPool_.l1Size))):
            0
            return False
        return True
    
    def AdjustSparseL0Factors(self, singleCoreStatus):
        'Source: MatmulTilingAlgorithm::AdjustSparseL0Factors; lines 2299-2332'
        self._visit('AdjustSparseL0Factors')
        if self._branch(245,(not self.tilingIns_.isSparse_)):
            0
            return
        baseK = Cell(i32(i32(u32((u32(singleCoreStatus.v.l0Status.kL0) * u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))))
        sparseBaseKFac = Cell(i32(64))
        if self._branch(246,(baseK.v <= sparseBaseKFac.v)):
            assign(baseK,i32(sparseBaseKFac.v))
        else:
            assign(baseK,i32(MathUtil.AlignDown(baseK.v, sparseBaseKFac.v)))
        assign(AttrRef(singleCoreStatus.v.l0Status,'kL0'),i32(i32(u32(cdiv(u32(baseK.v),u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))))
        baseM = Cell(i32(i32((singleCoreStatus.v.l0Status.mL0 * C0_SIZE))))
        baseN = Cell(i32(i32((singleCoreStatus.v.l0Status.nL0 * C0_SIZE))))
        if self._branch(247,(i32((baseM.v * baseK.v)) > i32(cdiv(self.tilingIns_.bufferPool_.l0ASize,DB_ON)))):
            assign(AttrRef(singleCoreStatus.v.l0Status,'dbL0A'),i32(DB_OFF))
        if self._branch(248,(i32((baseN.v * baseK.v)) > i32(cdiv(self.tilingIns_.bufferPool_.l0BSize,DB_ON)))):
            assign(AttrRef(singleCoreStatus.v.l0Status,'dbL0B'),i32(DB_OFF))
        if self._branch(249,(i32((baseM.v * baseN.v)) > i32(cdiv(self.tilingIns_.bufferPool_.l0CSize,DB_ON)))):
            assign(AttrRef(singleCoreStatus.v.l0Status,'dbL0C'),i32(DB_OFF))
    
    def PreprocessL0DB(self):
        'Source: MatmulTilingAlgorithm::PreprocessL0DB; lines 2334-2358'
        self._visit('PreprocessL0DB')
        assign(AttrRef(self,'dbL0A_'),i32(self.cfg.l0aDB))
        assign(AttrRef(self,'dbL0B_'),i32(self.cfg.l0bDB))
        assign(AttrRef(self,'dbL0C_'),i32(self.cfg.l0cDB))
        if self._branch(250,(self.tilingIns_.baseM != i32((-1)))):
            baseLeftSize = Cell(i32(i32((self.tilingIns_.baseM * C0_BYTE_SIZE))))
            if self._branch(251,(baseLeftSize.v > i32(cdiv(self.tilingIns_.bufferPool_.l0ASize,DB_ON)))):
                assign(AttrRef(self,'dbL0A_'),i32(DB_OFF))
        if self._branch(252,(self.tilingIns_.baseN != i32((-1)))):
            baseRightSize = Cell(i32(i32((self.tilingIns_.baseN * C0_BYTE_SIZE))))
            if self._branch(253,(baseRightSize.v > i32(cdiv(self.tilingIns_.bufferPool_.l0BSize,DB_ON)))):
                assign(AttrRef(self,'dbL0B_'),i32(DB_OFF))
        if self._branch(254,((self.tilingIns_.baseM != i32((-1))) and (self.tilingIns_.baseN != i32((-1))))):
            baseMatrixSize = Cell(i32(i32((i32((self.tilingIns_.baseM * self.tilingIns_.baseN)) * C0_BYTE_SIZE))))
            if self._branch(255,(baseMatrixSize.v > i32(cdiv(self.tilingIns_.bufferPool_.l0CSize,DB_ON)))):
                assign(AttrRef(self,'dbL0C_'),i32(DB_OFF))
        return
    
    def SetDepthL1CacheUBParams(self, a1LengthCache, b1LengthCache):
        'Source: MatmulTilingAlgorithm::SetDepthL1CacheUBParams; lines 2360-2440'
        self._visit('SetDepthL1CacheUBParams')
        if self._branch(256,((not self.tilingIns_.enableL1CacheUB) or (self.tilingIns_.socVersion != 'ASCEND310P'))):
            return
        a1Length = Cell(i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseM * self.tilingIns_.tiling_.baseK))) * DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])),u32(BITS_PER_BYTE))))))
        b1Length = Cell(i32(i32(u32(cdiv(u32((u32(i32((self.tilingIns_.tiling_.baseN * self.tilingIns_.tiling_.baseK))) * DTYPE_BIT_TAB[self.tilingIns_.bType_.dataType])),u32(BITS_PER_BYTE))))))
        assign(a1LengthCache,i32(i32((i32((a1Length.v * self.tilingIns_.tiling_.stepKa)) * self.tilingIns_.tiling_.stepM))))
        assign(b1LengthCache,i32(i32((i32((b1Length.v * self.tilingIns_.tiling_.stepKb)) * self.tilingIns_.tiling_.stepN))))
        freeL1Size = Cell(i32(i32((i32((self.tilingIns_.bufferPool_.l1Size - i32((self.tilingIns_.tiling_.depthA1 * a1Length.v)))) - i32((self.tilingIns_.tiling_.depthB1 * b1Length.v))))))
        if self._branch(257,(freeL1Size.v <= 0)):
            return
        splitNum = Cell(i32(2))
        aOrgShapeSize = Cell(i32(i32((self.tilingIns_.tiling_.singleCoreM * self.tilingIns_.tiling_.singleCoreK))))
        bOrgShapeSize = Cell(i32(i32((self.tilingIns_.tiling_.singleCoreN * self.tilingIns_.tiling_.singleCoreK))))
        if self._branch(258,(((self.tilingIns_.aType_.type == CubeFormat.ND) and (self.tilingIns_.aType_.pos != TPosition.TSCM)) and ((self.tilingIns_.bType_.type == CubeFormat.ND) and (self.tilingIns_.bType_.pos != TPosition.TSCM)))):
            aFullLoad = Cell(bool(False))
            bFullLoad = Cell(bool(False))
            assign(aFullLoad,bool(((aOrgShapeSize.v > 0) and (aOrgShapeSize.v < i32(cdiv(freeL1Size.v,splitNum.v))))))
            assign(bFullLoad,bool(((bOrgShapeSize.v > 0) and (bOrgShapeSize.v < i32(cdiv(freeL1Size.v,splitNum.v))))))
            if self._branch(259,(aFullLoad.v and bFullLoad.v)):
                assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(1))
                assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(1))
                assign(a1LengthCache,i32(aOrgShapeSize.v))
                assign(b1LengthCache,i32(bOrgShapeSize.v))
            else:
                if self._branch(260,aFullLoad.v):
                    assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(1))
                    assign(a1LengthCache,i32(aOrgShapeSize.v))
                    depthL1CacheUB = Cell(i32((i32(cdiv(i32((freeL1Size.v - aOrgShapeSize.v)),b1LengthCache.v)) if (b1LengthCache.v > 0) else 0)))
                    assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(depthL1CacheUB.v))
                else:
                    if self._branch(261,bFullLoad.v):
                        assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(1))
                        assign(b1LengthCache,i32(bOrgShapeSize.v))
                        depthL1CacheUB_2 = Cell(i32((i32(cdiv(i32((freeL1Size.v - bOrgShapeSize.v)),a1LengthCache.v)) if (a1LengthCache.v > 0) else 0)))
                        assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(depthL1CacheUB_2.v))
                    else:
                        if self._branch(262,(a1LengthCache.v > freeL1Size.v)):
                            depthBL1CacheUB = Cell(i32((i32(cdiv(freeL1Size.v,b1LengthCache.v)) if (b1LengthCache.v > 0) else 0)))
                            assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(depthBL1CacheUB.v))
                        else:
                            if self._branch(263,(b1LengthCache.v > freeL1Size.v)):
                                depthAL1CacheUB = Cell(i32((i32(cdiv(freeL1Size.v,a1LengthCache.v)) if (a1LengthCache.v > 0) else 0)))
                                assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(depthAL1CacheUB.v))
                            else:
                                if self._branch(264,((a1LengthCache.v <= i32(cdiv(freeL1Size.v,splitNum.v))) and (b1LengthCache.v <= i32(cdiv(freeL1Size.v,splitNum.v))))):
                                    depthAL1CacheUB_2 = Cell(i32((i32(cdiv(i32(cdiv(freeL1Size.v,splitNum.v)),a1LengthCache.v)) if (a1LengthCache.v > 0) else 0)))
                                    depthBL1CacheUB_2 = Cell(i32((i32(cdiv(i32(cdiv(freeL1Size.v,splitNum.v)),b1LengthCache.v)) if (b1LengthCache.v > 0) else 0)))
                                    assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(depthAL1CacheUB_2.v))
                                    assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(depthBL1CacheUB_2.v))
                                else:
                                    if self._branch(265,(a1LengthCache.v <= b1LengthCache.v)):
                                        assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(i32(cdiv(freeL1Size.v,a1LengthCache.v))))
                                    else:
                                        assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(i32(cdiv(freeL1Size.v,b1LengthCache.v))))
        else:
            if self._branch(266,((self.tilingIns_.aType_.type == CubeFormat.ND) and (self.tilingIns_.aType_.pos != TPosition.TSCM))):
                if self._branch(267,((aOrgShapeSize.v > 0) and (aOrgShapeSize.v < freeL1Size.v))):
                    assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(1))
                    assign(a1LengthCache,i32(aOrgShapeSize.v))
                else:
                    if self._branch(268,(a1LengthCache.v > 0)):
                        assign(AttrRef(self.tilingIns_.tiling_,'depthAL1CacheUB'),i32(i32(cdiv(freeL1Size.v,a1LengthCache.v))))
            else:
                if self._branch(269,((self.tilingIns_.bType_.type == CubeFormat.ND) and (self.tilingIns_.bType_.pos != TPosition.TSCM))):
                    if self._branch(270,((bOrgShapeSize.v > 0) and (bOrgShapeSize.v < freeL1Size.v))):
                        assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(1))
                        assign(b1LengthCache,i32(bOrgShapeSize.v))
                    else:
                        if self._branch(271,(b1LengthCache.v > 0)):
                            assign(AttrRef(self.tilingIns_.tiling_,'depthBL1CacheUB'),i32(i32(cdiv(freeL1Size.v,b1LengthCache.v))))
                else:
                    return
    
    def UpdateDepthB1(self, singleCoreStatus):
        'Source: MatmulTilingAlgorithm::UpdateDepthB1; lines 2442-2465'
        self._visit('UpdateDepthB1')
        depthB1 = Cell(i32(i32((i32((MathUtil.CeilDivision(singleCoreStatus.v.l1Status.kBL1, singleCoreStatus.v.l0Status.kL0) * singleCoreStatus.v.l1Status.nBL1)) * singleCoreStatus.v.l1Status.dbBL1))))
        if self._branch(272,((self.tilingIns_.bType_.dataType != DataType.DT_FLOAT) or (self.tilingIns_.socVersion != 'ASCEND910B'))):
            return depthB1.v
        alignedBaseK = Cell(i32((MathUtil.CeilDivision(self.tilingIns_.baseK, FP32_ALIGN_SIZE) * FP32_ALIGN_SIZE)))
        alignedBaseKN = Cell(i32((i32(alignedBaseK.v) * self.tilingIns_.baseN)))
        alignedBaseKM = Cell(i32((self.tilingIns_.baseK * self.tilingIns_.baseM)))
        if self._branch(273,(self.tilingIns_.aType_.isTrans and (self.tilingIns_.aType_.dataType == DataType.DT_FLOAT))):
            assign(alignedBaseKM,i32((i32(alignedBaseK.v) * self.tilingIns_.baseM)))
        if self._branch(274,(u64((u64(i32((i32((self.tilingIns_.tiling_.depthA1 * i32(alignedBaseKM.v))) + i32((i32(alignedBaseKN.v) * depthB1.v))))) * 4)) > u64(self.tilingIns_.bufferPool_.l1Size))):
            assign(depthB1,i32(i32(cdiv(i32((i32((self.tilingIns_.baseN * self.tilingIns_.baseK)) * depthB1.v)),i32(alignedBaseKN.v)))))
            assign(depthB1,i32((1 if (depthB1.v < 1) else depthB1.v)))
        return depthB1.v
    
    def Process(self):
        'Source: MatmulTilingAlgorithm::Process; lines 2467-2669'
        self._visit('Process')
        self.PreprocessL0DB()
        if self._branch(275,(not self.CheckBaseMN())):
            0
            return i64(i32((-1)))
        assign(AttrRef(self,'singelBlockDim_'),bool(False))
        assign(AttrRef(self,'splitCoreFlag_'),bool(False))
        coreStatus = Cell(CoreStatusPack())
        singleCoreStatus = Cell(SingleCoreStatus())
        param = Cell(MatmulRunParas())
        blockDimRes = Cell(BlockDimCalculator())
        self.FillParam(param)
        opType = Cell('MatMul')
        if self._branch(276,(self.numOfBlock_ != 1)):
            self.NonFactorMap(opType, param, blockDimRes)
            self.GetBlockDim(opType, param, coreStatus, blockDimRes)
        else:
            if self._branch(277,(not self.cfg.factorSplit)):
                assign(AttrRef(coreStatus.v,'m'),i32(param.v.m32))
                assign(AttrRef(coreStatus.v,'k'),i32(param.v.k32))
                assign(AttrRef(coreStatus.v,'n'),i32(param.v.n32))
            else:
                assign(AttrRef(coreStatus.v,'m'),i32(MathUtil.FindBestSingleCore(param.v.m32, param.v.mMapped, 1, False)))
                assign(AttrRef(coreStatus.v,'k'),i32(MathUtil.FindBestSingleCore(param.v.k32, param.v.kMapped, 1, False)))
                assign(AttrRef(coreStatus.v,'n'),i32(MathUtil.FindBestSingleCore(param.v.n32, param.v.nMapped, 1, False)))
            assign(AttrRef(coreStatus.v,'batchDim'),i32(1))
            assign(AttrRef(coreStatus.v,'mDim'),i32(1))
            assign(AttrRef(coreStatus.v,'kDim'),i32(1))
            assign(AttrRef(coreStatus.v,'nDim'),i32(1))
        if self._branch(278,((self.numOfBlock_ != 1) and (self.tilingIns_.bType_.pos == TPosition.TSCM))):
            if self._branch(279,(not self.splitCoreFlag_)):
                0
                return i64(1)
        self.GetL0Factors(opType, param, coreStatus, singleCoreStatus)
        self.AdjustSparseL0Factors(singleCoreStatus)
        if self._branch(280,(((singleCoreStatus.v.l0Status.mL0 == 0) or (singleCoreStatus.v.l0Status.nL0 == 0)) or (singleCoreStatus.v.l0Status.kL0 == 0))):
            0
            return i64(i32((-1)))
        self.GetL1Factors(opType, param, coreStatus, AttrRef(singleCoreStatus.v,'l0Status'), AttrRef(singleCoreStatus.v,'l1Status'))
        coreUse = Cell(i32(i32((i32((i32((coreStatus.v.batchDim * coreStatus.v.mDim)) * coreStatus.v.kDim)) * coreStatus.v.nDim))))
        singleCoreM = Cell(i32(MathUtil.CeilDivision(self.tilingIns_.singleM, coreStatus.v.mDim)))
        singleCoreN = Cell(i32(MathUtil.CeilDivision(self.tilingIns_.singleN, coreStatus.v.nDim)))
        singleCoreK = Cell(i32(MathUtil.CeilDivision(self.tilingIns_.singleK, coreStatus.v.kDim)))
        if self._branch(281,self.singelBlockDim_):
            assign(coreUse,i32(self.tilingIns_.blockDim))
            assign(singleCoreM,i32((self.tilingIns_.singleCoreM if (self.tilingIns_.singleCoreM != i32((-1))) else self.tilingIns_.singleM)))
            assign(singleCoreN,i32((self.tilingIns_.singleCoreN if (self.tilingIns_.singleCoreN != i32((-1))) else self.tilingIns_.singleN)))
            assign(singleCoreK,i32((self.tilingIns_.singleCoreK if (self.tilingIns_.singleCoreK != i32((-1))) else self.tilingIns_.singleK)))
        if self._branch(282,(self.numOfBlock_ > 1)):
            assign(singleCoreM,i32(i32((MathUtil.CeilDivision(param.v.m32, coreStatus.v.mDim) * C0_SIZE))))
            assign(singleCoreN,i32(i32((MathUtil.CeilDivision(param.v.n32, coreStatus.v.nDim) * C0_SIZE))))
            if self._branch(283,self.tilingIns_.enableSplitK_):
                if self._branch(284,((self.tilingIns_.aType_.dataType == DataType.DT_FLOAT) or (self.tilingIns_.bType_.dataType == DataType.DT_FLOAT))):
                    assign(singleCoreK,i32(i32((MathUtil.CeilDivision(param.v.k32, coreStatus.v.kDim) * FLOAT32_REDUCE_BLOCK_SIZE))))
                else:
                    if self._branch(285,((self.tilingIns_.aType_.dataType == DataType.DT_INT8) or (self.tilingIns_.bType_.dataType == DataType.DT_INT8))):
                        assign(singleCoreK,i32(i32((MathUtil.CeilDivision(param.v.k32, coreStatus.v.kDim) * INT8_REDUCE_BLOCK_SIZE))))
                    else:
                        if self._branch(286,((self.tilingIns_.aType_.dataType == DataType.DT_INT4) or (self.tilingIns_.bType_.dataType == DataType.DT_INT4))):
                            assign(singleCoreK,i32(i32((MathUtil.CeilDivision(param.v.k32, coreStatus.v.kDim) * INT4_REDUCE_BLOCK_SIZE))))
                        else:
                            assign(singleCoreK,i32(i32((MathUtil.CeilDivision(param.v.k32, coreStatus.v.kDim) * REDUCE_BLOCK_SIZE))))
            else:
                assign(singleCoreK,i32(self.tilingIns_.singleK))
        assign(AttrRef(self.tilingIns_.tiling_,'usedCoreNum'),i32(coreUse.v))
        assign(AttrRef(self.tilingIns_.tiling_,'M'),i32(self.tilingIns_.orgM))
        assign(AttrRef(self.tilingIns_.tiling_,'N'),i32(self.tilingIns_.orgN))
        assign(AttrRef(self.tilingIns_.tiling_,'Ka'),i32(self.tilingIns_.orgKa))
        assign(AttrRef(self.tilingIns_.tiling_,'Kb'),i32(self.tilingIns_.orgKb))
        if self._branch(287,((self.tilingIns_.socVersion == 'ASCEND910') or (self.tilingIns_.socVersion == 'ASCEND310P'))):
            if self._branch(288,(((self.tilingIns_.cType_.pos == TPosition.VECCALC) and (self.tilingIns_.cType_.type == CubeFormat.ND)) and (u32(cmod(u32((u32(singleCoreN.v) * DTYPE_BYTE_TAB[self.tilingIns_.cType_.dataType])),u32(C0_BYTE_SIZE))) != u32(0)))):
                0
                return i64(i32((-1)))
        assign(AttrRef(self.tilingIns_.tiling_,'singleCoreM'),i32(singleCoreM.v))
        assign(AttrRef(self.tilingIns_.tiling_,'singleCoreN'),i32(singleCoreN.v))
        assign(AttrRef(self.tilingIns_.tiling_,'singleCoreK'),i32(singleCoreK.v))
        assign(AttrRef(self.tilingIns_.tiling_,'baseM'),i32(i32((singleCoreStatus.v.l0Status.mL0 * C0_SIZE))))
        assign(AttrRef(self.tilingIns_.tiling_,'baseN'),i32(i32((singleCoreStatus.v.l0Status.nL0 * C0_SIZE))))
        reduceSize = Cell(i32(i32(u32((u32(cdiv(u32(C0_BYTE_SIZE),DTYPE_BIT_TAB[self.tilingIns_.aType_.dataType])) * u32(BITS_PER_BYTE))))))
        assign(AttrRef(self.tilingIns_.tiling_,'baseK'),i32(i32((singleCoreStatus.v.l0Status.kL0 * reduceSize.v))))
        if self._branch(289,(((self.tilingIns_.scheduleType == ScheduleType.OUTER_PRODUCT) and (self.tilingIns_.tiling_.baseK < self.tilingIns_.tiling_.singleCoreK)) and ((self.tilingIns_.mmConfigType == 1) or ((self.tilingIns_.mmConfigType == 0) and (self.tilingIns_.batchNum != 0))))):
            0
            return i64(i32((-1)))
        assign(AttrRef(self.tilingIns_.tiling_,'iterateOrder'),i32(self.GetIteratorOrder(singleCoreStatus, singleCoreM.v, singleCoreN.v, singleCoreK.v)))
        newBaseM = Cell(i32(i32((singleCoreStatus.v.l0Status.mL0 * C0_SIZE))))
        newBaseN = Cell(i32(i32((singleCoreStatus.v.l0Status.nL0 * C0_SIZE))))
        if self._branch(290,(self.tilingIns_.scheduleType == ScheduleType.OUTER_PRODUCT)):
            isL0CFullUsed = Cell(bool((True if (i32((i32((i32((newBaseM.v * newBaseN.v)) * NUM_TWO)) * i32(DTYPE_BYTE_TAB[self.tilingIns_.cType_.dataType]))) > self.tilingIns_.bufferPool_.l0CSize) else False)))
            if self._branch(291,(isL0CFullUsed.v and (self.tilingIns_.tiling_.iterateOrder == 0))):
                assign(newBaseN,i32(MathUtil.Align(i32(cdiv(newBaseN.v,NUM_TWO)), C0_SIZE)))
            else:
                if self._branch(292,(isL0CFullUsed.v and (self.tilingIns_.tiling_.iterateOrder == 1))):
                    assign(newBaseM,i32(MathUtil.Align(i32(cdiv(newBaseM.v,NUM_TWO)), C0_SIZE)))
            assign(AttrRef(self.tilingIns_.tiling_,'baseM'),i32(newBaseM.v))
            assign(AttrRef(self.tilingIns_.tiling_,'baseN'),i32(newBaseN.v))
        assign(AttrRef(self.tilingIns_,'baseM'),i32(self.tilingIns_.tiling_.baseM))
        assign(AttrRef(self.tilingIns_,'baseN'),i32(self.tilingIns_.tiling_.baseN))
        assign(AttrRef(self.tilingIns_,'baseK'),i32(self.tilingIns_.tiling_.baseK))
        assign(AttrRef(self.tilingIns_.tiling_,'depthA1'),i32(i32((i32((MathUtil.CeilDivision(singleCoreStatus.v.l1Status.kAL1, singleCoreStatus.v.l0Status.kL0) * singleCoreStatus.v.l1Status.mAL1)) * singleCoreStatus.v.l1Status.dbAL1))))
        newDepthB1 = Cell(i32(self.UpdateDepthB1(singleCoreStatus)))
        assign(AttrRef(self.tilingIns_.tiling_,'depthB1'),i32(newDepthB1.v))
        assign(AttrRef(self.tilingIns_.tiling_,'stepM'),i32(singleCoreStatus.v.l1Status.mAL1))
        assign(AttrRef(self.tilingIns_.tiling_,'stepN'),i32(singleCoreStatus.v.l1Status.nBL1))
        assign(AttrRef(self.tilingIns_.tiling_,'stepKa'),i32(MathUtil.CeilDivision(singleCoreStatus.v.l1Status.kAL1, singleCoreStatus.v.l0Status.kL0)))
        assign(AttrRef(self.tilingIns_.tiling_,'stepKb'),i32(MathUtil.CeilDivision(singleCoreStatus.v.l1Status.kBL1, singleCoreStatus.v.l0Status.kL0)))
        if self._branch(293,(DTYPE_BYTE_TAB[self.tilingIns_.bType_.dataType] == DTYPE_BYTE_TAB[DataType.DT_FLOAT])):
            if self._branch(294,(self.tilingIns_.tiling_.baseK == DT_FLOAT_INVALID_BASEK)):
                assign(AttrRef(self.tilingIns_.tiling_,'stepKb'),i32(1))
                assign(AttrRef(self.tilingIns_.tiling_,'depthB1'),i32(i32((singleCoreStatus.v.l1Status.nBL1 * singleCoreStatus.v.l1Status.dbBL1))))
        assign(AttrRef(self.tilingIns_.tiling_,'isBias'),i32((1 if self.tilingIns_.isBias else 0)))
        transLength = Cell(i32(0))
        self.GetTransLength(transLength)
        a1LengthCache = Cell(i32(0))
        b1LengthCache = Cell(i32(0))
        self.SetDepthL1CacheUBParams(a1LengthCache, b1LengthCache)
        assign(AttrRef(self.tilingIns_.tiling_,'transLength'),i32(transLength.v))
        assign(AttrRef(self.tilingIns_.tiling_,'shareMode'),i32(0))
        assign(AttrRef(self.tilingIns_.tiling_,'dbL0A'),i32(singleCoreStatus.v.l0Status.dbL0A))
        assign(AttrRef(self.tilingIns_.tiling_,'dbL0B'),i32(singleCoreStatus.v.l0Status.dbL0B))
        assign(AttrRef(self.tilingIns_.tiling_,'dbL0C'),i32(singleCoreStatus.v.l0Status.dbL0C))
        l1Size = Cell(i32(0))
        l0cSize = Cell(i32(0))
        ubSize = Cell(i32(0))
        self.GetUsedSize(l1Size, l0cSize, ubSize, a1LengthCache.v, b1LengthCache.v)
        assign(AttrRef(self.tilingIns_.tiling_,'shareL1Size'),i32(l1Size.v))
        assign(AttrRef(self.tilingIns_.tiling_,'shareL0CSize'),i32(l0cSize.v))
        assign(AttrRef(self.tilingIns_.tiling_,'shareUbSize'),i32(ubSize.v))
        assign(AttrRef(self.tilingIns_.tiling_,'batchM'),i32(self.tilingIns_.batchM))
        assign(AttrRef(self.tilingIns_.tiling_,'batchN'),i32(self.tilingIns_.batchN))
        assign(AttrRef(self.tilingIns_.tiling_,'singleBatchM'),i32(self.tilingIns_.singleBatchM))
        assign(AttrRef(self.tilingIns_.tiling_,'singleBatchN'),i32(self.tilingIns_.singleBatchN))
        assign(AttrRef(self.tilingIns_.tiling_,'ALayoutInfoB'),i32(self.tilingIns_.aLayoutInfoB))
        assign(AttrRef(self.tilingIns_.tiling_,'ALayoutInfoS'),i32(self.tilingIns_.aLayoutInfoS))
        assign(AttrRef(self.tilingIns_.tiling_,'ALayoutInfoN'),i32(self.tilingIns_.aLayoutInfoN))
        assign(AttrRef(self.tilingIns_.tiling_,'ALayoutInfoG'),i32(self.tilingIns_.aLayoutInfoG))
        assign(AttrRef(self.tilingIns_.tiling_,'ALayoutInfoD'),i32(self.tilingIns_.aLayoutInfoD))
        assign(AttrRef(self.tilingIns_.tiling_,'BLayoutInfoB'),i32(self.tilingIns_.bLayoutInfoB))
        assign(AttrRef(self.tilingIns_.tiling_,'BLayoutInfoS'),i32(self.tilingIns_.bLayoutInfoS))
        assign(AttrRef(self.tilingIns_.tiling_,'BLayoutInfoN'),i32(self.tilingIns_.bLayoutInfoN))
        assign(AttrRef(self.tilingIns_.tiling_,'BLayoutInfoG'),i32(self.tilingIns_.bLayoutInfoG))
        assign(AttrRef(self.tilingIns_.tiling_,'BLayoutInfoD'),i32(self.tilingIns_.bLayoutInfoD))
        assign(AttrRef(self.tilingIns_.tiling_,'CLayoutInfoB'),i32(self.tilingIns_.cLayoutInfoB))
        assign(AttrRef(self.tilingIns_.tiling_,'CLayoutInfoS1'),i32(self.tilingIns_.cLayoutInfoS1))
        assign(AttrRef(self.tilingIns_.tiling_,'CLayoutInfoN'),i32(self.tilingIns_.cLayoutInfoN))
        assign(AttrRef(self.tilingIns_.tiling_,'CLayoutInfoG'),i32(self.tilingIns_.cLayoutInfoG))
        assign(AttrRef(self.tilingIns_.tiling_,'CLayoutInfoS2'),i32(self.tilingIns_.cLayoutInfoS2))
        assign(AttrRef(self.tilingIns_.tiling_,'BatchNum'),i32(self.tilingIns_.batchNum))
        ans = Cell(bool(self.CheckFinaleParams(coreStatus)))
        return i64((0 if ans.v else i32((-1))))
    
