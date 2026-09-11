"""Value/reference and finite-width helpers for the offline source translation.

No source parsing, foreign code, file access, environment access, or fitted data
is involved at inference. C++ references and object copies are explicit.
"""
from copy import deepcopy
from struct import pack, unpack

class Cell:
    __slots__=('v',)
    def __init__(self, value): self.v=value

class AttrRef:
    __slots__=('obj','name')
    def __init__(self,obj,name): self.obj,self.name=obj,name
    @property
    def v(self):return getattr(self.obj,self.name)
    @v.setter
    def v(self,value):setattr(self.obj,self.name,value)

class ItemRef:
    __slots__=('obj','index')
    def __init__(self,obj,index):self.obj,self.index=obj,index
    @property
    def v(self):return self.obj[self.index]
    @v.setter
    def v(self,value):self.obj[self.index]=value

def clone(x):return deepcopy(x)
def i32(x):
    x=int(x)&0xffffffff
    return x-(1<<32) if x>0x7fffffff else x
def u32(x):return int(x)&0xffffffff
def i64(x):
    x=int(x)&0xffffffffffffffff
    return x-(1<<64) if x>0x7fffffffffffffff else x
def u64(x):return int(x)&0xffffffffffffffff
def f32(x):return unpack('<f',pack('<f',float(x)))[0]
def cdiv(a,b):
    if b==0:raise ZeroDivisionError('C++ integer division by zero')
    q=abs(a)//abs(b)
    return -q if (a<0)!=(b<0) else q
def cmod(a,b):return a-cdiv(a,b)*b

def assign(ref,value,copy=False):
    ref.v=clone(value) if copy else value
    return ref.v

def increment(ref,delta,post=False):
    old=ref.v;ref.v=i32(old+delta)
    return old if post else ref.v

def vector_erase(vec,start,end=None):
    del vec[start:len(vec) if end is None else end]
    return start

def vector_unique(vec):
    dst=0
    for x in vec[:]:
        if dst==0 or vec[dst-1]!=x:vec[dst]=x;dst+=1
    return dst

def copy_array(dst,src,size):
    if isinstance(dst,list):dst[:size//4]=clone(src[:size//4])
    else:dst.__dict__.update(clone(src.__dict__))
    return 0

class MathUtil:
    @staticmethod
    def IsEqual(a,b):return abs(f32(f32(a)-f32(b)))<=2**-23
    @staticmethod
    def CeilDivision(a,b):return 0 if b==0 else i32(cdiv(int(a)+int(b)-1,int(b)))
    @staticmethod
    def Align(a,b):return i32(MathUtil.CeilDivision(a,b)*b)
    @staticmethod
    def AlignDown(a,b):return 0 if b==0 else i32(cdiv(a,b)*b)
    @staticmethod
    def CheckMulOverflow(a,b,c):
        if a<=0 or b<=0 or a>2147483647//b:return False
        c.v=a*b;return True
    @staticmethod
    def MapShape(shape,roundUpFlag=True):
        seed=16
        if shape<seed:return shape
        while seed<1024:
            if seed<shape and seed*2>=shape:break
            seed*=2
        if roundUpFlag:seed*=2
        return seed
    @staticmethod
    def GetFactors(factors,srcNum,*bounds):
        a=factors.v
        if len(bounds)==1:
            a.extend(i for i in range(1,min(srcNum,bounds[0])+1) if srcNum%i==0)
        else:
            a.extend(i for i in range(bounds[1],bounds[0]-1,-1) if srcNum%i==0)
    @staticmethod
    def GetFactorCnt(shape,count,start,end):
        count.v+=sum(shape%i==0 for i in range(start,min(shape,end)+1))
    @staticmethod
    def GetFactorLayerCnt(shape,count,start,end):
        fs=Cell([]);MathUtil.GetFactors(fs,shape,start,end)
        for f in fs.v:
            fc=Cell(0);MathUtil.GetFactorCnt(f,fc,1,f+1)
            count.v=max(count.v,fc.v)
    @staticmethod
    def AddFactor(factors,dim):factors.v[:]=sorted(set(factors.v+[dim]))
    @staticmethod
    def GetNonFactorMap(factors,srcNum,maxFactor):
        count=Cell(0);MathUtil.GetFactorLayerCnt(srcNum,count,1,maxFactor)
        mapped=MathUtil.MapShape(srcNum,True) if srcNum>1 and count.v<=4 else srcNum
        MathUtil.GetFactors(factors,mapped,maxFactor)
        return mapped
    @staticmethod
    def GetBlockFactors(factors,oriShape,mpShape,coreNum,maxNum):
        factors.v.extend(i for i in range(1,maxNum+1) if oriShape%i==0 or mpShape%i==0 or coreNum%i==0)
    @staticmethod
    def CheckFactorNumSatisfy(dim):
        if dim<=8:return True
        a=Cell(0);b=Cell(0)
        MathUtil.GetFactorLayerCnt(dim,a,1,64)
        if dim>128:MathUtil.GetFactorLayerCnt(dim,b,65,128)
        return not (a.v<=2 or (dim>128 and a.v+b.v<=4))
    @staticmethod
    def FindBestSingleCore(oriShape,mappedShape,coreNum,isKDim):
        real=MathUtil.CeilDivision(oriShape,coreNum)
        mapped=MathUtil.CeilDivision(mappedShape,coreNum)
        if isKDim:return MathUtil.CeilDivision(oriShape if oriShape%coreNum==0 else mappedShape,coreNum)
        if coreNum==1 and MathUtil.CheckFactorNumSatisfy(oriShape):return oriShape
        best=real
        while best!=mapped:
            if MathUtil.CheckFactorNumSatisfy(best):return best
            best+=1 if best<mapped else -1
        return best
