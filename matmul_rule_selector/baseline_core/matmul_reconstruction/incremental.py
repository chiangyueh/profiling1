"""Actual DoIncreTiling NT closure: BB rejection -> FastFindParams -> cycle winner.

Specialized by the caller's proven predicates: batch one, half/bfloat16 matching
output, no bias/quantization, A=N/B=T, ND, arch220 L0C2OUT. No history or table is
consulted. The original mutable scratch candidate is reused within each family.
"""
from dataclasses import asdict,replace
from .basic_block import GemmResult,cd,up,generate as basic_generate
from .conversion import normalize_buffers,set_double_buffer,load_flags
from ._cycle.types import ResultInfo,CoreStatus
from ._cycle.init import matmulv3_init_run_params,prepare_external_status
from ._cycle.selector import cycle_used,vendor_would_update
from ._core.errors import UnsupportedDomain


def generate(r,h,trace=True):
    if not(r['M']<=128 and r['dtype']in ('fp16','bf16')and not r['transA']and r['transB']and not r.get('bias',False)):
        raise UnsupportedDomain('Incremental closure requires its exact MatMulV3 caller gate')
    if h['aicNum'] < 20 or h['l1Size'] < 262144:
        raise UnsupportedDomain('Incremental source-reference hardware domain: aicNum>=20 and L1>=262144 bytes; ensures pattern stream is nonempty')
    bb=basic_generate(r,h,trace=trace)
    assert bb['status']=='GEN_TILING_EAGAIN' # M<=128 contradicts BB M>256
    m,n,k=cd(r['M'],16),cd(r['N'],16),cd(r['K'],16)
    unaligned=bool(r['M']%16 or r['N']%16 or r['K']%16);cores=h['aicNum']
    output=[];attempts=[];incumbent=None;best=None
    run=matmulv3_init_run_params(r['M'],r['N'],r['K'],r['dtype'],False,True)
    def fits(a,b):return(a+b)*512<=h['l1Size']
    def expand(shape,dim,value):
        if dim==1:return value
        maximum=max(cd(shape,dim-1)-1,value);aligned=up(value,16)
        return aligned if aligned<=maximum else None
    def finish(s):
        normalized,c=normalize_buffers(r,s);s.__dict__.update(normalized.__dict__)
        limit=3 if unaligned else 1.35
        if not(s.m_l1*c['m_single_core']*s.m_dim/m<limit and s.n_l1*c['n_single_core']*s.n_dim/n<limit):return None
        if not(fits(s.m_l1*s.kal1_16,s.n_l1*s.kbl1_16)and s.k_l0*s.m_l0<=64 and s.k_l0*s.n_l0<=64 and s.m_l0*s.n_l0<=128):return None
        return c
    def add(s,c,family,search_dims):
        nonlocal incumbent,best
        selected,db=set_double_buffer(r,h,s,c);s.__dict__.update(selected.__dict__)
        f=load_flags(c)
        core=CoreStatus(batch=1,m=c['m'],n=c['n'],k=c['k'],kal1_factor=c['kal1_factor'],kbl1_factor=c['kbl1_factor'],
          m_single_core=c['m_single_core'],n_single_core=c['n_single_core'],al1_full_load=f['A_full'],bl1_full_load=f['B_full'],
          both_full_load=f['A_full']and f['B_full'],al1_k_full_load=f['A_K_full'],bl1_k_full_load=f['B_K_full'])
        i=len(output);prepared=prepare_external_status(run,ResultInfo(**asdict(s)),core,c['m_al1'],c['n_bl1'],i)
        score=cycle_used(prepared)
        update=incumbent is None or vendor_would_update(score,incumbent,run)
        item={'insertion_index':i,'family':family,'search_dims':list(search_dims),'result':asdict(s),
              'core':c.copy(),'double_buffer':db,'estimate':asdict(score),
              'comparison':{'previous_winner':best,'update':update,'exact_tie_preserves_earlier':bool(incumbent and not update and score.cycle==incumbent.cycle)}}
        output.append(item)
        if update:best=i;incumbent=score
    def attempt(s,family,md,nd,fn):
        s.m_dim,s.n_dim=md,nd
        c=fn(s)
        if trace:attempts.append({'family':family,'m_dim':md,'n_dim':nd,'accepted':c is not None,'scratch':asdict(s)})
        if c is not None:add(s,c,family,(md,nd))
    def set_a(s):
        s.m_l1=cd(m,s.m_dim)
        ka,kb=s.kal1_16,s.kbl1_16  # Source intentionally snapshots old kbl1_bound here.
        expanded=expand(m,s.m_dim,s.m_l1)
        if expanded is None:return None
        s.m_l1=expanded
        if not fits(s.m_l1*ka,1):return None
        amat=s.m_l1*ka
        if s.m_l1<=16:s.m_l0=s.m_l1
        elif s.m_l1%8==0:s.m_l0=8
        else:return None
        s.k_l0=min(64//s.m_l0,k);s.n_l0=min(cd(n,s.n_dim),16)
        while s.kal1_16%s.k_l0 or s.k_l0*s.m_l0>64 or s.k_l0*s.n_l0>64:
            s.k_l0-=1
            if not s.k_l0:return None
        s.n_l1=s.n_l0;s.kbl1_16=s.k_l0
        if k%16==0 and 16%s.k_l0==0 and fits(amat,16*s.n_l1):s.kbl1_16=16
        if not fits(s.m_l1*ka,kb*s.n_l1*2):return None
        return finish(s)
    def set_b(s):
        s.n_l1=cd(n,s.n_dim)
        expanded=expand(n,s.n_dim,s.n_l1)
        if expanded is None:return None
        s.n_l1=expanded;ka,kb=s.kal1_16,s.kbl1_16
        if not fits(1,kb*s.n_l1):return None
        if s.n_l1<=16:s.n_l0=s.n_l1
        elif s.n_l1%16==0:s.n_l0=16
        else:return None
        bmat=kb*s.n_l1;cm=cd(m,s.m_dim);s.m_l0=min(cm,8);s.k_l0=min(64//s.n_l0,k)
        while s.kbl1_16%s.k_l0 or s.k_l0*s.n_l0>64 or s.k_l0*s.m_l0>64:
            s.k_l0-=1
            if not s.k_l0:return None
        s.m_l1=s.m_l0;s.kal1_16=s.k_l0
        if k%16==0 and 16%s.k_l0==0 and fits(s.m_l1*16*2,bmat):s.kal1_16=16
        ka,kb=s.kal1_16,s.kbl1_16
        if cm>=16 and fits(16*ka*2,bmat):
            if s.kal1_16==k:s.m_l1=16
            if 16*s.k_l0<=64 and 16*s.n_l0<=128:s.m_l1=s.m_l0=16
        if not fits(s.m_l1*ka*2,bmat):return None
        return finish(s)
    def l1_shape(shape,dim,preferred):
        cs=cd(shape,dim);maximum=max(cs,cs if dim==1 else cd(shape,dim-1)-1);v=min(cs,preferred)
        while dim!=1 and up(cs,v)>maximum:v-=1
        return v
    def set_notfull(s):
        kk=k
        if unaligned and 8<kk<16:kk=16
        s.n_l1=l1_shape(n,s.n_dim,8);s.m_l1=l1_shape(m,s.m_dim,8)
        s.kal1_16=min(kk,8);s.kbl1_16=min(kk,4)
        preferred=32 if unaligned and kk>=32 else 16
        if kk>=preferred:s.kal1_16=s.kbl1_16=preferred
        if s.kbl1_16>s.kal1_16:s.kbl1_16=s.kbl1_16//s.kal1_16*s.kal1_16
        if s.kal1_16>s.kbl1_16:s.kal1_16=s.kal1_16//s.kbl1_16*s.kbl1_16
        s.m_l0=s.m_l1;s.n_l0=s.n_l1;s.k_l0=4 if min(s.kal1_16,s.kbl1_16)>=4 else kk
        if s.kal1_16==kk:
            lower=max(cd(64,s.kal1_16),s.m_l1);upper=min(lower*2,cd(m,s.m_dim))
            for i in range(lower,upper+1,s.m_l0):
                if m%i==0:s.m_l1=i;break
        maximum=min(cd(64,s.m_l0),cd(64,s.n_l0))
        for i in range(s.k_l0+1,maximum+1):
            if maximum%i==0 and s.kal1_16%i==0:s.k_l0=i;s.kbl1_16=max(s.k_l0,s.kbl1_16);break
        while s.kbl1_16%s.k_l0:s.k_l0-=1
        return finish(s)
    if fits(k*cd(m,cores),1):
        s=GemmResult(kal1_16=k)
        for md in range(1,min(cores,m)+1):
            if m<=8 and md>1:break
            for nd in range(cd(16,md),min(cores//md,n)+1):attempt(s,'PATTERN_AL1_FULL',md,nd,set_a)
    if fits(1,k*cd(n,cores)):
        s=GemmResult(kbl1_16=k)
        for nd in range(1,min(cores,n)+1):
            if n<=16 and nd>1:break
            for md in range(cd(16,nd),min(cores//nd,m)+1):attempt(s,'PATTERN_BL1_FULL',md,nd,set_b)
    s=GemmResult()
    for nd in range(1,min(cores,n)+1):
        md=min(cores//nd,m)
        if md!=m:
            cs=cd(m,md);aligned=up(cs,8)
            if aligned//cs>=2:aligned=up(cs,4)
            md=cd(m,aligned)
        attempt(s,'PATTERN_NOT_FULL',md,nd,set_notfull)
    if best is None:
        raise AssertionError('Nonempty-pattern domain invariant violated')
    selected=output[best]
    return {'basic_block_gate':bb,'estimator':'CYCLE_ESTIMATE_TYPE','attempts':attempts,'candidates':output,
            'selected':best,'postprocess':{k:selected[k]for k in ('result','core','double_buffer')},
            'source':'cache_tiling.cc:TilingPatternProcess/FastFindParams',
            'candidate_constructor_semantics':'GemmEstCoreStatus copies n_single_core from m_single_core; full-load flags retained'}
