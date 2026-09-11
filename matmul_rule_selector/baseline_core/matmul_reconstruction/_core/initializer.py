"""Source-derived default MultiCoreMatmulTiling initializer for the public domain.

The target owns a default-constructed MultiCoreMatmulTiling, not a platform-bound
one. Its ignored SetBufferSpace return is preserved, including atomic rejection
of oversized capacity requests. No host captures enter this computation.
"""
from types import SimpleNamespace
from .initializer_algorithm import (MatmulTilingAlgorithm, MatTilingType, BufferPool,
    MnmAdjust, DataType, CubeFormat, TPosition, MatrixTraverse, MatrixMadType,
    ScheduleType, DequantType)
from .initializer_runtime import clone

CUBE_FIELDS = (
    'usedCoreNum','M','N','Ka','Kb','singleCoreM','singleCoreN','singleCoreK',
    'baseM','baseN','baseK','depthA1','depthB1','stepM','stepN','isBias',
    'transLength','iterateOrder','shareMode','shareL1Size','shareL0CSize',
    'shareUbSize','batchM','batchN','singleBatchM','singleBatchN','stepKa','stepKb',
    'depthAL1CacheUB','depthBL1CacheUB','dbL0A','dbL0B','dbL0C',
    'ALayoutInfoB','ALayoutInfoS','ALayoutInfoN','ALayoutInfoG','ALayoutInfoD',
    'BLayoutInfoB','BLayoutInfoS','BLayoutInfoN','BLayoutInfoG','BLayoutInfoD',
    'CLayoutInfoB','CLayoutInfoS1','CLayoutInfoN','CLayoutInfoG','CLayoutInfoS2',
    'BatchNum','reserved')

class SourceTilingBase:
    """Reachable constructor/setter state, base.cpp and bmm_tiling.cpp commit 960."""
    def __init__(self, request, hardware):
        self.tiling_=SimpleNamespace(**{name:0 for name in CUBE_FIELDS})
        for name in ('aType_','bType_','cType_','biasType_'):
            value=MatTilingType();value.isDB=True;setattr(self,name,value)
        for name in ('aType_','bType_','cType_'):
            value=getattr(self,name)
            value.pos=TPosition.GM;value.type=CubeFormat.ND;value.dataType=DataType.DT_FLOAT16
        types={'fp16':DataType.DT_FLOAT16,'bf16':DataType.DT_BF16,'fp32':DataType.DT_FLOAT}
        self.aType_.dataType=self.bType_.dataType=types[request['dtype']]
        self.cType_.dataType=types[request['output_dtype']]
        self.aType_.isTrans=request['transA'];self.bType_.isTrans=request['transB']
        self.isBias=request['bias'];self.isSupportL0c2Out=True
        if self.isBias:
            self.biasType_.pos=TPosition.GM;self.biasType_.type=CubeFormat.ND
            self.biasType_.dataType=types[request['bias_dtype']]
        self.madType_=MatrixMadType.NORMAL;self.traverse_=MatrixTraverse.NOSET
        for name in ('singleM','singleN','singleK','singleCoreM','singleCoreN',
                     'singleCoreK','orgM','orgN','orgKa','orgKb','baseM','baseN','baseK'):
            setattr(self,name,-1)
        self.adjust_=MnmAdjust(2147483647,2147483647,2147483647,16,16,16)
        self.oriBufferPool_=BufferPool(524032,131072,196352,65536,65536,1024,32,32,32,32,32)
        self.bufferPool_=clone(self.oriBufferPool_)
        self.blockDim=1;self.batchM=self.batchN=self.singleBatchM=self.singleBatchN=1
        for name in ('aLayoutInfoB','aLayoutInfoS','aLayoutInfoN','aLayoutInfoG','aLayoutInfoD',
                     'bLayoutInfoB','bLayoutInfoS','bLayoutInfoN','bLayoutInfoG','bLayoutInfoD',
                     'cLayoutInfoB','cLayoutInfoS1','cLayoutInfoN','cLayoutInfoG','cLayoutInfoS2',
                     'batchNum','maxSingleM','maxSingleN','maxSingleK',
                     'minSingleM','minSingleN','minSingleK'):
            setattr(self,name,0)
        self.alignSingleM=self.alignSingleN=self.alignSingleK=1
        self.scheduleType=ScheduleType.INNER_PRODUCT
        self.transND2NZ_=self.transNZ2ND_=self.isSparse_=False
        self.deqType=DequantType.SCALAR;self.enableSplitK_=False;self.socVersion='ASCEND910B'
        self.mmConfigType=1;self.enableL1CacheUB=self.enVecND2NZ=False
        self.blockDim=hardware['aicNum']
        self.singleM=self.orgM=request['M'];self.singleN=self.orgN=request['N']
        self.singleK=self.orgKa=self.orgKb=request['K']
        self.buffer_return_code=self.set_buffer_space(hardware['l1Size'],hardware['l0CSize'],hardware['ubSize'])
    def set_buffer_space(self,l1,l0c,ub):
        pairs=(('l1Size',l1,'l1AlignSize'),('l0CSize',l0c,'l0CAlignSize'),('ubSize',ub,'ubAlignSize'))
        if any(value>getattr(self.bufferPool_,name) or value < -1 for name,value,_ in pairs):
            return -1
        for name,value,align_name in pairs:
            if value!=-1:
                alignment=getattr(self.bufferPool_,align_name)
                setattr(self.bufferPool_,name,value//alignment*alignment)
        return 0

def initialize_cube(request,hardware,*,trace=False):
    state=SourceTilingBase(request,hardware)
    algorithm=MatmulTilingAlgorithm(state,trace=trace)
    status=algorithm.Process()
    # SetFinalTiling copies 49 fields; the reserved word belongs to zero-initialized storage.
    cube={name:getattr(state.tiling_,name) for name in CUBE_FIELDS}
    cube['reserved']=0
    return {'return_code':status,'buffer_return_code':state.buffer_return_code,
            'cube':cube,'cube_words':[cube[name] for name in CUBE_FIELDS],
            'effective_buffer_pool':vars(state.bufferPool_).copy(),
            'method_counts':algorithm.counts,'branch_counts':algorithm.branch_counts,'trace':algorithm.events}
