"""Source port: gemm/common/cache_tiling_request_bytes.cc, all four public routines.

Counts are vendor-model requests, not an invented physical memory-traffic model.
The source's srcD%384 remainder test and missing 96-byte merge case are preserved.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Counts:
    c512: int = 0
    c256: int = 0
    c128: int = 0
    def __add__(self, other):
        return Counts(self.c512+other.c512,self.c256+other.c256,self.c128+other.c128)
    def __mul__(self, n):
        return Counts(self.c512*n,self.c256*n,self.c128*n)
    def size(self): return self.c512+self.c256+self.c128
    def align_sum(self): return self.c512*512+self.c256*256+self.c128*128

def c512(d):
    if d<=0: return Counts()
    return Counts(d//512,d%512//256,(d%256+127)//128)

def c256(d):
    if d<=0: return Counts()
    if d<256: return Counts(0,d//256,(d%256+127)//128)
    return c512(d-256)+Counts(0,1,0)

def c384(d): return c512(d-128)+Counts(0,0,1) if d>0 else Counts()
def c128(d): return c256(d-128)+Counts(0,0,1) if d>0 else Counts()
def out256(d): return Counts(0,d//256,(d%256+127)//128) if d>0 else Counts()
def out128(d): return Counts(0,0,(d+127)//128) if d>0 else Counts()
READ_COUNTS=(c512,c128,c256,c384)

def address512(d):
    if d<=128 or d==256 or d>=512: return 1
    return 2 if d<=384 else 3

def address384(d): return 1 if d<=128 else address512(d-128)
def address256(d): return 1 if d<=128 else 2 if d<256 else address512(d-256)
def address128(d):
    if d<=128: return 1
    if d<=256: return 2
    if d<384: return 3
    return max(2,address512(d-384))
READ_ADDRESS=(address512,address128,address256,address384)

def out_address(d):
    if d<=128 or d==256: return 1
    if d<=384 or d==512: return 2
    return 3 if d<512 else 2

def out_unaligned_address(d):
    return 2 if d<=128 else 3 if d<=256 else 5 if d<=384 else 6

def normalized(cube, output=False):
    n,d,ori,width=cube
    if n<1 or d<1 or ori<1 or width not in (1,2,4): raise ValueError('invalid DMA geometry')
    ds=d*width; os=ori*width
    if d==ori and ((ds<=256) if output else ds in (32,64,128,160,192,224,256)):
        ds*=n;os*=n;n=1
    return n,ds,os

def request_nd2nz(cube):
    n,ds,os=normalized(cube); nn=n-1; first=c512(ds)
    if os%128==0:
        idx=os%512//128
        if idx==0: return c512(ds)*nn+first
        if idx in (1,3):
            total=Counts()
            for f in READ_COUNTS: total+=f(ds)
            total*=nn//4
            end=nn%4; start=4-end if os%384==0 else 1
            for i in range(end): total+=READ_COUNTS[i+start](ds)
            return total+first
        return c512(ds)*(nn//2)+c256(ds)*(nn//2+nn%2)+first
    total=Counts()
    for i in range(1,n):
        addr=i*os; res=128-addr%128; row=Counts()
        if res<128: row+=Counts(0,0,1)
        else: res=0
        last=ds-res
        if last>0: row+=READ_COUNTS[(addr+res)%512//128](last)
        total+=row
    return total+first

def request_nz2nd(cube):
    n,ds,os=normalized(cube,True);nn=n-1;first=out256(ds)
    if os%128==0:
        if os%256==0: return out256(ds)*nn+first
        return out256(ds)*(nn//2)+out128(ds)*(nn//2+nn%2)+first
    total=Counts(); n256=ds//256;end=ds%256
    for i in range(1,n):
        addr=i*os
        if addr%256==0: total+=out256(ds)
        elif addr%128==0: total+=out128(ds)
        else:
            extra=0 if end==0 else (end-(128-addr%128)+127)//128+1
            total+=Counts(0,0,n256*3+extra)
    return total+first

def same_nd2nz(cube):
    n,ds,os=normalized(cube); value=address512(ds)
    if n==1: return value
    start,end,shift=1,4,1
    if os%128==0:
        shift=0;idx=os%512//128;repeat=min(4,n)
        start,end=((1,1),(1,repeat),(2,3),(4-repeat,4))[idx]
    for i in range(start,end): value=max(value,READ_ADDRESS[i](ds))
    return value+shift

def same_nz2nd(cube):
    n,ds,os=normalized(cube,True)
    return max(out_address(ds),out_address(ds-512)) if os%128==0 else out_unaligned_address(ds)
