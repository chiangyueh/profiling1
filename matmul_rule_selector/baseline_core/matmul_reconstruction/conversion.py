"""Source SetBufferParams, load flags, double-buffer choice and MatMul conversion.

The estimator's private normalized candidate is not the output candidate. These
post-selection operations are applied to the original selected GemmResultInfo.
"""
from dataclasses import asdict, replace
from .basic_block import GemmResult, cd, up


def normalize_buffers(r, candidate):
    s=replace(candidate)
    unit=8 if r['dtype']=='fp32' else 16
    m,n,k=cd(r['M'],16),cd(r['N'],16),cd(r['K'],unit)
    ms,ns=cd(cd(m,s.m_dim),s.m_l1),cd(cd(n,s.n_dim),s.n_l1)
    ak,bk=cd(k,s.kal1_16),cd(k,s.kbl1_16)
    cm,cn,ck=min(ms*s.m_l1,m),min(ns*s.n_l1,n),k
    if s.k_dim!=1:
        kl=max(s.kal1_16,s.kbl1_16);ks=cd(cd(k,s.k_dim),kl)
        ak,bk=ks*(kl//s.kal1_16),ks*(kl//s.kbl1_16);ck=min(ks*kl,k)
    s.batch_dim=1;s.m_dim,s.n_dim,s.k_dim=cd(m,cm),cd(n,cn),cd(k,ck)
    core={'m':cm,'n':cn,'k':ck,'m_single_core':ms,'n_single_core':ns,
          'kal1_factor':ak,'kbl1_factor':bk,'m_al1':cd(s.m_l1,s.m_l0),'n_bl1':cd(s.n_l1,s.n_l0)}
    return s,core


def load_flags(core):
    ms,ns,ak,bk=(core[n]for n in ('m_single_core','n_single_core','kal1_factor','kbl1_factor'))
    return {'A_full':ms==1 and ak==1,'B_full':ns==1 and bk==1,
            'A_K_full':ms!=1 and ak==1,'B_K_full':ns!=1 and bk==1}


def k_bounds(r,s):
    ka,kb=1,1;unit=8 if r['dtype']=='fp32' else 16
    if r['dtype']=='fp32':ka=2 if r['transA'] else 1;kb=2 if r['transA']or not r['transB']else 1
    a,b=up(s.kal1_16,ka),up(s.kbl1_16,kb)
    if r['dtype']=='fp32':
        rk=cd(r['K'],unit)
        if (r['K']-s.kal1_16*unit)%(ka*unit)!=0 and rk%kb!=0:a=up(s.kal1_16+1,ka)
        if (r['K']-s.kbl1_16*unit)%(kb*unit)!=0 and rk%kb!=0:b=up(s.kbl1_16+1,kb)
    return a,b


def set_double_buffer(r,h,s,core):
    s=replace(s);flags=load_flags(core);ka,kb=k_bounds(r,s)
    w=4 if r['dtype']=='fp32' else 2;unit=8 if w==4 else 16
    bw={'fp16':2,'bf16':2,'fp32':4}[r.get('bias_dtype',r.get('output_dtype',r['dtype']))]
    channels=cd(bw,2) if r.get('bias',False)else 0
    bias_l1=channels*core['n_bl1']*s.n_l0*16*2
    bias_bt=s.n_l0*16*4 if r.get('bias',False)else 0
    s.db_l0c=2 if s.batch_l0*s.m_l0*s.n_l0*2<=128 and bias_bt*2<=h['btSize']else 1
    a=core['m_al1']*s.m_l0*ka;b=core['n_bl1']*s.n_l0*kb
    fits=lambda x,y,z:(x+y)*16*unit*w+z<=h['l1Size']
    s.db_al1=s.db_bl1=1;branch='single'
    if flags['A_full']or flags['B_full']:
        if flags['A_full']and flags['B_full']:branch='both_full_single'
        elif flags['A_full']and fits(a,b*2,bias_l1*2):s.db_bl1=2;branch='A_full_B_double'
        elif flags['B_full']and fits(a*2,b,bias_l1):s.db_al1=2;branch='B_full_A_double'
    elif fits(a*2,b*2,bias_l1*2):s.db_al1=s.db_bl1=2;branch='both_double'
    elif fits(a*2,b,bias_l1):s.db_al1=2;branch='A_double'
    elif fits(a,b*2,bias_l1*2):s.db_bl1=2;branch='B_double'
    return s,{'load_flags':flags,'k_bounds':[ka,kb],'L1_single_blocks':[a,b],
              'L1_channel_bytes':bias_l1,'L0C_factor_limit':128,'branch':branch}


def postprocess(r,h,candidate):
    if isinstance(candidate,dict):candidate=GemmResult(**candidate)
    s,core=normalize_buffers(r,candidate);s,db=set_double_buffer(r,h,s,core)
    return {'selected_before_postprocess':asdict(candidate),'result':asdict(s),'core':core,
            'double_buffer':db,'source':'cache_tiling.cc:SetBufferParams/UpdateL1LoadFlag/SetDoubleBuffer'}


def convert_to_run(engine,post):
    """Literal Convert2AscendCTiling for pattern_flag=true, batch=1.

    Its baseK multiplier is 16 even for a direct FP32 subroutine probe. The
    actual MatMulV3 incremental caller is FP16/BF16 only. Do not silently replace
    that multiplier by the calculator's FP32 reduce block size of 8.
    """
    s=post['result'];c=post['core'];r=engine.run;a=engine.args
    r.usedCoreNum=min(s['m_dim']*s['n_dim'],engine.hw.aicNum)
    r.baseM=s['m_l0']*16;r.baseN=s['n_l0']*16;r.baseK=s['k_l0']*16
    r.singleCoreM=r.baseM;r.singleCoreN=r.baseN;r.singleCoreK=a.kValue
    r.stepM=c['m_al1'];r.stepN=c['n_bl1'];r.iterateOrder=0
    r.stepKa=s['kal1_16']//s['k_l0'];r.stepKb=s['kbl1_16']//s['k_l0']
    r.depthA1=r.stepM*r.stepKa*s['db_al1'];r.depthB1=r.stepN*r.stepKb*s['db_bl1']
    r.dbL0c=s['db_l0c'];r.l2Info.mTile=r.l2Info.nTile=1;r.l2Info.mTileBlock=r.l2Info.nTileBlock=1
    r.needUpdate=True
