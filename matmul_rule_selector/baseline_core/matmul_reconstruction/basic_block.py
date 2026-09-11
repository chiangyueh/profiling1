"""GemmBBCalculator and GenTilingFromBasicBlock; supplied common-source profile.

Only MatMul run semantics (batch=1) are exposed. No measured outputs are used.
Preserves std::set lexicographic cut traversal and the signed decimal comparator.
"""
from dataclasses import dataclass, asdict, replace
from math import sqrt, gcd

BB_IDX=(2,3,4,7,8,11,14,15,18,19,22,25,28,31,34,37,40,41,42,43,44)
BB_TILES=((64,64,32),(64,64,64),(64,64,256),(64,64,512),(64,64,1024),
 (64,128,64),(64,128,128),(64,128,256),(64,128,512),(64,256,64),(64,256,128),(64,256,256),
 (128,64,64),(128,64,128),(128,64,256),(128,64,512),(128,128,64),(128,128,128),(128,128,256),(128,128,512),
 (256,64,64),(256,64,128),(256,64,256),(96,320,64),(96,320,128),(96,320,256),
 (128,192,64),(128,192,128),(128,192,256),(128,256,64),(128,256,128),(128,256,256),
 (192,128,64),(192,128,128),(192,128,256),(256,128,64),(256,128,128),(256,128,256),
 (320,96,64),(320,96,128),(320,96,256),(320,64,256),(64,320,256),(512,64,128),(64,512,128),
 (16,256,64),(16,256,128),(16,256,256),(16,256,512),(16,64,32),(256,256,256),(128,512,128))


def cd(a,b):
    if b<=0: raise ValueError('nonpositive source divisor')
    return (a+b-1)//b

def up(a,b): return cd(a,b)*b

def largest(x,y,default=1):
    for v in range(y,1,-1):
        if x%v==0:return v
    return default

def smallest(x,lo,hi,default=1,step=1):
    for v in range(lo,hi+1,step):
        if x%v==0:return v
    return default

def factors(n):
    s=set()
    for i in range(1,int(sqrt(n))+1):
        if n%i==0:s.update((i,n//i))
    return sorted(s)

def bounds(cut,cores):
    if cut<1:return (1,1,cores,cores)
    if cut>=cores:return (cores,cores,1,1)
    last=-1
    for i in factors(cores):
        if i>=cut:
            return (1,1,cores,cores) if last==-1 else (i,last,cores//i,cores//last)
        last=i
    return (cores,cores,1,1)

def kl0(k1,kmax,cube):
    val=min(kmax,k1)
    if val!=k1:
        return largest(cd(k1,cube),val//cube)
    return cd(val,cube)

@dataclass
class GemmResult:
    batch_dim:int=1
    m_dim:int=1
    k_dim:int=1
    n_dim:int=1
    batch_l0:int=1
    m_l0:int=0
    k_l0:int=0
    n_l0:int=0
    m_l1:int=0
    n_l1:int=0
    kal1_16:int=0
    kbl1_16:int=0
    db_al1:int=2
    db_bl1:int=2
    db_l0c:int=1

@dataclass
class Load:
    bb_counts:float=2147483647
    active_ratio:float=0.0
    core_diff:int=1
    full_score:float=0.0
    min_k:int=1
    load_state:int=0

@dataclass
class Cut:
    m_cut:int
    n_cut:int
    m_dim:int
    m_tile:int
    n_dim:int
    n_tile:int

class Calculator:
    source='gemm/cache_tiling_basic_block_calc.cc'
    def __init__(self,r,h,trace=True):
        self.r=r;self.h=h;self.events=[];self.trace=trace
        self.M,self.N,self.K=r['M'],r['N'],r['K']
        self.ta,self.tb=r.get('transA',False),r.get('transB',False)
        self.bias=bool(r.get('bias',False))
        self.w={'fp16':2,'bf16':2,'fp32':4}[r['dtype']]
        self.ow={'fp16':2,'bf16':2,'fp32':4}[r.get('output_dtype',r['dtype'])]
        self.cube=8 if self.w==4 else 16
        self.rm,self.rn,self.rk=cd(self.M,16),cd(self.N,16),cd(self.K,self.cube)
        self.cores=h['aicNum'];self.l2=h['l2Size'];self.l1=h['l1Size']
        self.k_cut=1;self.batch_cut=1;self.quant=0
    def emit(self,function,**data):
        if self.trace:self.events.append({'source':self.source,'function':function,**data})
    def init(self,idx):
        bm,bn,bk=BB_TILES[idx]
        if self.bias and self.N>self.h['btSize']//4 and bn>self.h['btSize']//4:return False
        self.result=GemmResult(db_l0c=2 if idx in (4,19) else 1)
        self.fullbw=128 if self.h['supportL12BtBf16'] else 512
        self.align=128 if self.h['supportL12BtBf16'] else 256
        self.sweet_up=140.0 if self.cores==24 else 117.0 if self.l2==192*1024**2 else 195.0
        self.sweet_low=73.0 if self.cores==24 else 61.0 if self.l2==192*1024**2 else 101.0
        if self.w==4:
            scale=2 if self.r.get('hf32',False) else 4
            self.sweet_up/=scale;self.sweet_low/=scale
        self.alpha=0.4
        if self.w!=4:
            self.alpha*=0.6 if self.M*self.N*self.w>=self.l2 else 0.8 if self.M*self.N*self.w>=self.l2//2 else 1
        self.kt=up(min(bk*2//self.w,self.K),self.cube)
        self.mt=up(min(bm,self.M),16);self.nt=up(min(bn,self.N),16)
        ai=(self.M if self.ta else self.K)*self.w;bi=(self.K if self.tb else self.N)*self.w
        def eff(x):return 1.0 if x%self.fullbw==0 else 2.0 if x>self.fullbw else 2.5
        self.aeff=eff(ai);self.beff=eff(bi);self.effratio=self.aeff/self.beff
        self.aeff*=self.M;self.beff*=self.N
        self.result.m_dim=cd(self.M,self.mt);self.result.n_dim=cd(self.N,self.nt)
        self.m_divided=self.rm*16%self.result.m_dim==0
        self.n_divided=self.rn*16%self.result.n_dim==0
        self.k_aligned=self.rk*self.cube*self.w%self.align==0
        self.align_cube=self.w==4 and (self.ta or not self.tb)
        self.reserve=self.K%16!=0 and self.align_cube
        self.must_divided=self.r['dtype']=='fp16' and self.r.get('output_dtype',self.r['dtype'])=='fp32' and (self.ta or not self.tb)
        threshold=1.0*self.l2/(self.K*self.w)
        self.sweet_bound=self.sweet_low
        self.active_bound=0.8 if self.cores==24 else 1.0 if self.l2==192*1024**2 else 0.75
        if (ai%self.fullbw!=0 or self.M>threshold) and (bi%self.fullbw!=0 or self.N>threshold):
            self.sweet_bound=self.sweet_up
            self.active_bound=0.7 if self.cores==24 else 0.85 if self.l2==192*1024**2 else 0.5
        self.is_l2_fit=threshold>=min(self.M,self.N)
        self.k_cut=1;self.kpc_tmp=0;self.kpc=1
        self.emit('Init',bb_idx=idx,seed=BB_TILES[idx],m_tile=self.mt,n_tile=self.nt,k_tile=self.kt,
                  initial_dimensions=(self.result.m_dim,self.result.n_dim),sweet_up=self.sweet_up,sweet_low=self.sweet_low)
        return True
    def available(self,m,n):return self.l1-2*16*(m+n)*self.w if self.reserve else self.l1
    def gen_cuts(self,cores):
        md,nd=self.result.m_dim,self.result.n_dim
        mg,ng=gcd(md,cores),gcd(nd,cores)
        cuts={(mg,min(nd,cores//mg)),(min(md,cores//ng),ng)}
        if mg==1:
            tmp=max(1,min(md,cores));cuts.add((tmp,cores//tmp))
        if ng==1:
            tmp=max(1,min(nd,cores));cuts.add((cores//tmp,tmp))
        roots=[sqrt(1.0*self.effratio*cores*self.nt/self.mt),sqrt(1.0*cores*md*self.mt/(nd*self.nt)/self.effratio)]
        if mg==cores or ng==cores:roots.append(sqrt(1.0*cores))
        for root in roots:
            a,b,c,d=bounds(root,cores)
            cuts.add((min(a,md),min(c,nd)));cuts.add((min(b,md),min(d,nd)))
        if self.divided:cuts={(smallest(md,a,md,md),smallest(nd,b,nd,nd))for a,b in cuts}
        return sorted(cuts)
    def adjust_tile(self,dim,tile,ori,cut):
        active=cd(dim,cd(dim,cut));core=cd(dim,cut);omap=up(ori,16)
        ta=cd(omap,core*cut*16)*16;da=cd(ori,ta);ca=cd(da,cd(da,cut))
        if self.divided and omap%(da*ta)!=0:return dim,tile
        return (da,ta) if ca>=active else (dim,tile)
    def load_for(self,c,cores):
        mc,nc=cd(c.m_dim,c.m_cut),cd(c.n_dim,c.n_cut)
        bb=mc*nc*max(1.0,1.0*c.n_cut*c.m_cut/cores)
        ma,na=cd(c.m_dim,mc),cd(c.n_dim,nc);active=ma*na
        ratio=1.0*active/(cd(active,cores)*cores);diff=abs(ma-na)
        tile=(c.m_tile*self.kt+c.n_tile*(self.quant+int(self.bias)+self.kt))*self.w*2
        av=self.available(c.m_tile,c.n_tile)
        mf,nf=c.m_dim<=c.m_cut,c.n_dim<=c.n_cut
        ae=self.aeff*(1.2 if c.n_cut==1 else 1);be=self.beff*(1.2 if c.m_cut==1 else 1)
        mm=1 if self.is_l2_fit else cd(c.n_dim,c.n_cut)
        nm=1 if self.is_l2_fit else cd(c.m_dim,c.m_cut)
        def score(base,mrep,nrep):return base+1.0/((ae*(mrep-mm)+be*(nrep-nm))+2.9*(self.M*mm+self.N*nm))
        if self.kpc_tmp<=self.kt and tile<=av:
            return Load(bb,ratio,diff,score(3,c.n_cut,c.m_cut if mf or nf else c.m_dim),self.kt,3)
        def mink(i):
            mk=self.kpc_tmp//self.cube;sq=int(sqrt(mk))
            if i<sq:
                for j in range(i,sq):
                    if mk%j==0:mk=j;break
            else:
                for j in range(mk//i,0,-1):
                    if mk%j==0:mk=mk//j;break
            return mk*self.cube
        if mf:
            mk=mink(cd(65536,c.n_tile*self.w*self.cube))
            tile=(self.kpc_tmp*c.m_tile+(mk+int(self.bias)+self.quant)*c.n_tile*2)*self.w
            if tile<=av:return Load(bb,ratio,diff,score(1,c.n_cut,c.m_cut),mk,1)
        if nf:
            mk=mink(cd(65536,c.m_tile*self.w*self.cube))
            tile=((self.kpc_tmp+int(self.bias)+self.quant)*c.n_tile+mk*c.m_tile*2)*self.w
            if tile<=av:return Load(bb,ratio,diff,score(1,c.n_cut,c.m_cut),mk,2)
        return Load(bb,ratio,diff,score(0,up(c.n_dim,c.n_cut),up(c.m_dim,c.m_cut)),self.kt,0)
    def comparison(self,old,new,cur_ai,in_ai):
        def cmp(a,b):return (a>b)-(a<b)
        if cur_ai>=self.sweet_up:
            ds=[cmp(old.bb_counts,new.bb_counts),cmp(new.active_ratio,old.active_ratio),cmp(old.core_diff,new.core_diff)]
        else:
            active=min(new.active_ratio,old.active_ratio)
            if active>=self.active_bound and in_ai<self.sweet_bound*active*active:
                ds=[cmp(new.full_score,old.full_score),cmp(old.active_ratio,new.active_ratio)]
            else:ds=[cmp(new.active_ratio,old.active_ratio),cmp(new.full_score,old.full_score)]
            ds.extend((cmp(old.core_diff,new.core_diff),cmp(old.bb_counts,new.bb_counts)))
        score=0
        for d in ds:score=score*10+d
        return score
    def adjust_dims(self,cores,cur_ai,in_ai):
        self.kpc_tmp=cd(self.rk,self.k_cut)*self.cube
        best=Cut(1,1,self.result.m_dim,self.mt,self.result.n_dim,self.nt)
        bestload=Load(core_diff=cores,min_k=self.kt)
        for mcut,ncut in self.gen_cuts(cores):
            can=Cut(mcut,ncut,self.result.m_dim,self.mt,self.result.n_dim,self.nt)
            if not self.is_adjust and in_ai>=self.sweet_low:
                if not self.ta:can.m_dim,can.m_tile=self.adjust_tile(can.m_dim,can.m_tile,self.M,can.m_cut)
                if self.tb:can.n_dim,can.n_tile=self.adjust_tile(can.n_dim,can.n_tile,self.N,can.n_cut)
            load=self.load_for(can,cores); comparison=self.comparison(bestload,load,cur_ai,in_ai)
            self.emit('AdjustDimTile',candidate=asdict(can),load=asdict(load),decimal_comparison=comparison,replace=comparison>0,cur_AI=cur_ai,in_AI=in_ai)
            if comparison>0:best,bestload=can,load
        self.result.m_dim=min(best.m_cut,best.m_dim);self.mt=best.m_tile
        self.result.n_dim=min(best.n_cut,best.n_dim);self.nt=best.n_tile
        self.load=bestload
        self.m_divided=self.rm*16%(self.result.m_dim*self.mt)==0
        self.n_divided=self.rn*16%(self.result.n_dim*self.nt)==0
    def adjust_kcut(self):
        ai=2.0*self.M*self.N/(self.aeff*self.result.n_dim+self.beff*self.result.m_dim)/self.w
        cur=min(2.0*self.K/self.ow,ai)
        if not self.is_adjust or self.k_cut>1:
            self.divided=self.k_cut>1 or self.must_divided
            self.adjust_dims(max(1,self.cores//self.k_cut),cur,ai)
        if (self.must_divided or self.k_cut>1) and (not self.m_divided or not self.n_divided):
            cores=max(1,self.cores//self.k_cut)
            if not self.m_divided:
                tmp=largest(self.rm,self.mt//16);self.mt=tmp*16
                self.result.m_dim=smallest(self.rm//tmp,self.result.m_dim,cores)
            if not self.n_divided:
                tmp=largest(self.rn,self.nt//16);self.nt=tmp*16
                self.result.n_dim=smallest(self.rn//tmp,self.result.n_dim,2*cores//self.result.m_dim,self.rn//tmp)
            self.m_divided=self.n_divided=True
    def ktiles_none(self):
        av=self.available(self.mt,self.nt)-2*(int(self.bias)+self.quant)*self.nt
        mk=up(cd(65536,max(self.mt,self.nt)*self.w),self.cube)
        shr=smallest(self.kt,mk,self.kt,self.kt,self.cube)
        ash=self.kt
        for i in range(shr,self.kt+1,self.cube):
            if i%(self.fullbw//self.w)==0:ash=i;break
        a=shr if self.ta else ash;b=ash if self.tb else shr
        ea=False if self.ta else not self.k_aligned
        eb=not self.k_aligned if self.tb else False
        if not self.ta and self.tb:ea=eb=True
        if ea and eb:
            mx=min(self.rk*self.cube,av//((self.mt+self.nt)*2*self.w));a=b=mx//ash*ash
        if ea:
            mx=min(self.rk*self.cube,(av//self.w-b*self.nt*2)//(self.mt*2))
            if mx>=a:
                a=mx//a*a
                bf=min(self.rk,(av//self.w-a*self.mt*2)//self.nt//2//self.cube)
                b=self.cube*largest(a//self.cube,bf)
        elif eb:
            mx=min(self.rk*self.cube,(av//self.w-a*self.mt*2)//(self.nt*2))
            if mx>=b:
                b=mx//b*b
                af=min(self.rk,(av//self.w-b*self.nt*2)//self.mt//2//self.cube)
                a=self.cube*largest(b//self.cube,af)
        elif av!=self.l1 and (self.mt*a+self.nt*b)*self.w*2>av:
            self.kt=max(av//self.w//2//(self.mt+self.nt)//self.cube,1)
            a=b=self.kt
        return a,b
    def ktiles_full(self,which):
        av=self.l1-16*(self.mt+2*self.nt)*self.w if self.reserve else self.l1
        bw=self.fullbw//self.w
        if which==1:
            av-=2*int(self.bias)*self.nt;a=self.kpc_tmp;b=self.load.min_k
            if self.kpc<self.kpc_tmp:
                tmp=cd(65536,self.nt*self.w*self.cube)*self.cube
                minb=smallest(self.kpc,tmp,self.kpc,self.kpc,self.cube)
                if (self.kpc*self.mt+(minb+int(self.bias))*2*self.nt)*self.w<=av:a,b=self.kpc,minb
            if self.tb and a%bw==0:
                mx=min(self.rk*self.cube,(av//self.w-a*self.mt)//self.nt//2)
                for k in range(self.load.min_k,mx+1,self.cube):
                    if a%k==0 and k%bw==0:b=k;break
        else:
            av-=int(self.bias)*self.nt;b=self.kpc_tmp;a=self.load.min_k
            if self.kpc<self.kpc_tmp:
                tmp=cd(65536,self.nt*self.w*self.cube)*self.cube
                mina=smallest(self.kpc,tmp,self.kpc,self.kpc,self.cube)
                if (mina*self.mt*2+(self.kpc+int(self.bias))*self.nt)*self.w<=av:b,a=self.kpc,mina
            if not self.ta and b%bw==0:
                mx=min(self.rk*self.cube,(av//self.w-b*self.nt)//self.mt//2)
                for k in range(self.load.min_k,mx+1,self.cube):
                    if b%k==0 and k%bw==0:a=k;break
        return a,b
    def ktiles(self):
        a=b=self.kt;self.kpc=cd(self.rk,self.k_cut)*self.cube
        if cd(self.K,self.kt)>self.k_cut:
            if self.load.load_state==0:a,b=self.ktiles_none()
            elif self.load.load_state in (1,2):a,b=self.ktiles_full(self.load.load_state)
        kmax=min(self.h['l0ASize']//self.w//2//self.mt,self.h['l0BSize']//self.w//2//self.nt)
        if self.align_cube:kmax=kmax//16*16
        k0=kl0(min(a,b),kmax,self.cube)
        if self.must_divided and self.k_cut==1:
            if a>b:
                aa=largest(self.rk,a//self.cube);bb=largest(aa,b//self.cube);a,b=aa*self.cube,bb*self.cube;k0=largest(bb,kmax//self.cube)
            elif b>a:
                bb=largest(self.rk,b//self.cube);aa=largest(bb,a//self.cube);a,b=aa*self.cube,bb*self.cube;k0=largest(aa,kmax//self.cube)
            else:
                kk=largest(self.rk,a//self.cube);a=b=kk*self.cube;k0=largest(kk,kmax//self.cube)
        return a,b,k0
    def fit(self):
        self.is_adjust=False;self.k_cut=1;self.load=Load(core_diff=self.cores,min_k=self.kt)
        if not self.m_divided or not self.n_divided:
            ai=2.0*self.M*self.N/(self.aeff*self.result.n_dim+self.beff*self.result.m_dim)/self.w
            self.divided=False
            self.adjust_dims(self.cores,min(ai,2*self.K*self.alpha/self.ow),ai);self.is_adjust=True
        self.adjust_kcut();a,b,k0=self.ktiles();s=self.result
        s.k_l0=k0;s.k_dim=self.k_cut;s.batch_dim=1
        s.kal1_16=cd(a,self.cube);s.kbl1_16=cd(b,self.cube)
        s.m_l0=s.m_l1=cd(self.mt,16);s.n_l0=s.n_l1=cd(self.nt,16)
        if s.m_dim*self.mt>=self.M and a*self.k_cut>=self.K:s.db_al1=1
        if s.n_dim*self.nt>=self.N and b*self.k_cut>=self.K:s.db_bl1=1
        self.emit('BasicBlockFit',result=asdict(s),k_cut_invariant=1,batch_dim_invariant=1)
        return replace(s)

def applicability(r,h):
    checks=[]
    def check(name,value):checks.append({'predicate':name,'value':bool(value)});return bool(value)
    dt=r['dtype'];od=r.get('output_dtype',dt);width=4 if dt=='fp32' else 2
    if not check('CheckDtype: matching fp16/bf16/fp32 inputs and outputs',dt in ('fp16','bf16','fp32') and od==dt):return False,checks
    if not check('support_l0c2out and not support_fix_pipe_l0c2ub',h['supportL0c2out'] and not h.get('supportFixPipeL0c2ub',False)):return False,checks
    if not check('ND or weight-NZ with ND output',r.get('layoutA','ND')=='ND' and r.get('layoutB','ND') in ('ND','NZ') and r.get('layoutC','ND')=='ND'):return False,checks
    if not check('M>256 and N>256 and K>256',min(r['M'],r['N'],r['K'])>256):return False,checks
    if not check('K*(M+N)*2*dsize/core_num > L1',r['K']*(r['M']+r['N'])*2*width//h['aicNum']>h['l1Size']):return False,checks
    ma=(r['M'] if r.get('transA') else r['K'])*width%32==0
    mb=(r['K'] if r.get('transB') else r['N'])*width%32==0
    return check('source inner A and B axes are 32-byte aligned',ma and mb),checks

def generate(r,h,trace=True):
    ok,checks=applicability(r,h)
    if not ok:return {'status':'GEN_TILING_EAGAIN','checks':checks,'candidates':[],'selected':None}
    from .basic_estimate import estimate
    candidates=[];best=None;bestscore=float('inf');calc=Calculator(r,h,trace)
    for idx in BB_IDX+((50,51) if h['supportL12BtBf16'] else ()):
        calc.events=[]
        if not calc.init(idx):continue
        result=calc.fit()
        if min(result.m_l0,result.n_l0,result.k_l0)<=0:continue
        score=estimate(r,h,result);replace_best=score['score']<bestscore
        item={'insertion_index':len(candidates),'bb_idx':idx,'seed':BB_TILES[idx],'result':asdict(result),
              'estimate':score,'comparison':{'incumbent_score':None if best is None else bestscore,'strictly_less':replace_best,'equal_keeps_earlier':score['score']==bestscore},'construction_trace':calc.events}
        candidates.append(item)
        if replace_best:best=item['insertion_index'];bestscore=score['score']
    return {'status':'GEN_TILING_EOF' if best is not None else 'GEN_TILING_EAGAIN','checks':checks,
            'candidates':candidates,'selected':best,'pattern_flag':best is not None,
            'source':'gemm/cache_tiling_basic_block.cc:134-185'}
