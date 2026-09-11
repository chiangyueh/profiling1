"""Versioned MatmulV3BaseTiling decisions for non-advanced ND MatMul.

Each legacy override is tied to the tested-repository source. CANN's default ALL
path has the same bodies; its extra forced-route cases are intentionally absent.
"""
from ._core.engine import Engine as SourceEngine
from ._core.arithmetic import U64, MASK64, ceil_div as cd, align_up as up, align_down as down, ratio_f32, f32
from ._core.trace import sourced, plain
from ._core.constants import SUPPORT_ND2NZ_GM2L0, SPLIT_K_THRES, MB_SIZE, CALC_ND_BASIC
from ._core.errors import UnsupportedDomain, SourceFieldOverflow

FAMILY_METHODS = (
 ('BL1_FULL_LOAD_FIXPIPE','bl1_fixpipe','DoBL1FullloadWithFixpipeTiling'),
 ('AL1_FULL_LOAD','al1_full','DoAL1FullLoadTiling'),
 ('SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD','sc_al1_full','DoSingleCoreSplitKAL1FullLoadTiling'),
 ('BL1_FULL_LOAD','bl1_full','DoBL1FullLoadTiling'),
 ('L2_CACHE','l2_tiling','DoL2CacheTiling'),
 ('SINGLE_CORE_SPLIT_K','single_core_split','DoSingleCoreSplitKTiling'),
 ('DETERMINISTIC_SPLIT_K','deterministic_split','DoDeterministicMultiCoreSplitKTiling'),
 ('L2_CACHE_310P','l2_310p','DoL2CacheTiling310P'))

class VersionedEngine(SourceEngine):
    def __init__(self,request,hardware,profile='p0',trace=True,route='ALL'):
        self.profile=profile;self.legacy=profile!='p0';self.route=route
        self.attempted_families=[];self.candidate_stream=[];self.chosen_source_family='BASE'
        self.symbol_stack=[]
        super().__init__(request,hardware,trace)
    def emit(self,event,**payload):
        if event=='enter':self.symbol_stack.append(payload.get('symbol',''))
        super().emit(event,source_profile=self.profile,active_symbol=self.symbol_stack[-1] if self.symbol_stack else '',**payload)
        if event=='exit' and self.symbol_stack:self.symbol_stack.pop()
    def iterations(self,name,iterable):
        for item in super().iterations(name,iterable):
            if self.trace_enabled:self.emit('loop_trial_begin',loop=name,item=plain(item),state=self.snapshot())
            yield item
            if self.trace_enabled:self.emit('loop_trial_end',loop=name,item=plain(item),state=self.snapshot())
    def predicate(self,file,line,expression,value,variables):
        result=super().predicate(file,line,expression,value,variables)
        self._last_predicate={'expression':expression,'value':bool(value),'implementation_source':f'{file}:{line}'}
        return result
    def gate(self,name,value,**components):
        self._last_predicate={'expression':name,'value':bool(value),'components':plain(components)}
        if self.trace_enabled:self.emit('predicate',**self._last_predicate)
        return bool(value)
    @sourced('FormulaicBaseBlockTiling',1503,1557)
    def formulaic(self):
        a,r,h=self.args,self.run,self.hw
        mc,nc=cd(a.mValue,r.baseM),cd(a.nValue,r.baseN)
        if self.trans==0:self.formulaic_nn()
        elif self.trans==1:
            if not self.m_align and not self.n_align:self.balance_base()
            elif not self.m_align:r.baseM=self.cal_base_size(nc,h.aicNum,a.mValue,self.basic_m)
            elif not self.n_align:r.baseN=self.cal_base_size(mc,h.aicNum,a.nValue,256)
        elif self.trans==2:
            self.calc_base(self.calc_mn,True)
            if not self.ka_align:self.balance_base()
        else:
            r.baseN=self.cal_base_size(mc,h.aicNum,a.nValue,256)
            if not self.m_align and self.kb_align:self.balance_base()
            elif not self.m_align:r.baseM,r.baseN=self.cal_mn(r.baseM,r.baseN)
    @sourced('CalcBase',1612,1649)
    def calc_base(self,templates,mn_mode):
        a,h,r=self.args,self.hw,self.run
        computed=[];index=0;minimum=U64(MASK64)
        for i,template in self.iterations('CalcBase.templates',enumerate(templates)):
            bm,bn,kbytes=map(U64,template)
            if not mn_mode:
                nc=cd(a.nValue,bn);bm=self.cal_base_size(nc,h.aicNum,a.mValue,bm);mc=cd(a.mValue,bm)
            else:
                bm,bn=self.cal_mn(bm,bn,self.calc_mn[i][0],self.calc_mn[i][1]);mc=cd(a.mValue,bm);nc=cd(a.nValue,bn)
            computed.append((bm,bn,kbytes))
            legal=not a.hasBias or self.check_bt(bn)
            load=(bm+bn)*(mc*nc//h.aicNum)+(bm+bn if mc*nc%h.aicNum else 0)
            better=legal and load<minimum
            candidate={'stage':'CalcBase','index':i,'baseMNK':[int(bm),int(bn),int(kbytes//self.aw)],
              'source_load_proxy':int(load),'legal':bool(legal),'incumbent':None if minimum==MASK64 else int(minimum),
              'strictly_less':bool(better),'tie_keeps_earlier':bool(legal and load==minimum)}
            self.candidate_stream.append(candidate)
            if self.trace_enabled:self.emit('candidate',**candidate)
            if better:index=i;minimum=load
        r.baseM,r.baseN,kbytes=computed[index];r.baseK=kbytes//self.aw
    @sourced('GetMoreArgs',507,575)
    def get_more_args(self):
        super().get_more_args()
        if self.legacy:self.args.unAlignProcessType=0
    def mata_conflict_a(self):
        if not self.legacy:return super().mata_conflict_a()
        a=self.args
        return (not a.isBTrans and a.nValue%16384==0 and a.mValue>4096 and a.kValue>=6656
                and a.bFormat=='ND' and a.aType in ('fp16','bf16'))
    def mata_conflict_b(self):
        return False if self.legacy else super().mata_conflict_b()
    @sourced('DoIncreTiling',2243,2285)
    def do_incre(self):
        a=self.args
        gate=(self.hw.supportL0c2out and a.mValue<=128 and self.mix_type()==1 and
              a.aType in ('fp16','bf16') and a.cType in ('fp16','bf16') and
              a.aFormat==a.bFormat==a.outFormat=='ND' and not a.isATrans and a.isBTrans and not a.hasBias)
        self.incre_gate=gate
        self.gate('DoIncreTiling complete gate',gate,M=a.mValue,transA=a.isATrans,transB=a.isBTrans,
                  mix_type=self.mix_type(),dtype=a.aType,output_dtype=a.cType,bias=a.hasBias)
        self.attempted_families.append({'order':len(self.attempted_families),'family':'INCREMENTAL_PATTERN','accepted':bool(gate),'reason':self._last_predicate})
        if gate:
            from .incremental import generate
            from .conversion import convert_to_run
            request={'M':int(a.mValue),'N':int(a.nValue),'K':int(a.kValue),'dtype':a.aType,'output_dtype':a.cType,'bias':False,'transA':False,'transB':True}
            self.incremental_result=generate(request,plain(self.hw),trace=self.trace_enabled)
            self.enable.tilingEnableSplitCore=self.enable.tilingEnableFullLoad=self.enable.tilingEnableFixOpti=0
            convert_to_run(self,self.incremental_result['postprocess'])
            self.candidate_stream.extend(self.incremental_result['candidates'])
            self.chosen_source_family='INCREMENTAL_PATTERN'
    def optimize_load_balance(self):
        if not self.legacy:return super().optimize_load_balance()
    def optimize_step_k(self):
        if not self.legacy:return super().optimize_step_k()
    @sourced('DoSelectTiling',980,1007)
    def do_select(self):
        choices=[f for f in FAMILY_METHODS if not (self.legacy and f[0]=='SINGLE_CORE_SPLIT_K_AL1_FULL_LOAD')]
        if self.route=='BASE':choices=[f for f in choices if f[0] in ('L2_CACHE','L2_CACHE_310P')]
        elif self.route!='ALL':choices=[f for f in choices if f[0]==self.route]
        for family,method,symbol in choices:
            before=self.snapshot();start=len(self.trace);self._last_predicate=None
            accepted=bool(getattr(self,method)())
            attempt={'order':len(self.attempted_families),'family':family,'symbol':symbol,
                'accepted':accepted,'reason':self._last_predicate or {'expression':symbol+' return','value':accepted},
                'state_before':before,'state_after':self.snapshot(),
                'continues_after_mutation':not accepted and before!=self.snapshot(),
                'trace_range':[start,len(self.trace)]}
            self.attempted_families.append(attempt)
            if accepted:
                self.chosen_source_family=family
                break
    @sourced('DoAL1FullLoadTiling',1881,1925)
    def al1_full(self):
        a,h,r,e=self.args,self.hw,self.run,self.enable
        front=h.supportL0c2out and a.aType=='fp32' and not a.isATrans and a.isBTrans
        if not self.legacy:front=front and not self.force_group_fp32() and not(self.deterministic_small_mn() and a.mValue>=8)
        if not self.gate('AL1: hardware/FP32/NT and profile exclusions',front):return False
        aligned=a.kValue%(512//self.aw)==0
        valid=a.mValue<=16 and 16<a.nValue<=16*h.aicNum and a.kValue>=4096 and aligned
        abytes=up(a.mValue,16)*up(a.kValue,32//self.aw)*self.aw
        bbytes=16*256*self.bw*2
        biasbytes=16*(4 if a.biasType=='fp32' else 2) if a.hasBias else U64(0)
        if not self.gate('AL1: shape and full A plus double B fit',valid and abytes<=h.l1Size-bbytes-biasbytes,
                         valid_shape=valid,A_bytes=abytes,B_bytes=bbytes,bias_bytes=biasbytes,L1=h.l1Size):return False
        r.baseM=16;r.baseN=16;r.baseK=256;r.stepM=1;r.stepN=1
        r.stepKa=cd(a.kValue,r.baseK);r.stepKb=1;r.depthA1=r.stepKa;r.depthB1=2
        r.singleCoreM=a.mValue;r.singleCoreN=r.baseN;e.tilingEnableFullLoad=1;e.tilingEnableSplitCore=0
        r.dbL0c=2 if r.baseM*r.baseN*4*2<=h.l0CSize else 1
        r.l2Info.mTileBlock=1;r.l2Info.nTileBlock=cd(a.nValue,r.singleCoreN);r.l2Info.calOrder=1
        return True
    def bl1_fixpipe(self):
        if not self.legacy:return super().bl1_fixpipe()
        # Legacy has no forced FP32 accumulation exclusion, and the FP32 VEC
        # specialization lacks p0's final !isATrans conjunct.
        result=super().bl1_fixpipe()
        if result and self.args.aType=='fp32':
            a=self.args
            if a.kValue%(32//self.aw)==0 and a.nValue<=192 and not a.nd2nzA:self.enable.tilingEnableFixOpti=2
        return result
    @sourced('DoBL1FullLoadTiling',1927,1979)
    def bl1_full(self):
        a,h,r=self.args,self.hw,self.run
        if not self.legacy and self.force_group_fp32():return False
        c0=U64(32)//self.aw;inner=a.mValue if a.isATrans else a.kValue;outer=a.kValue if a.isATrans else a.mValue
        vnchw=a.aType=='fp32' and outer>=72368 and 1<inner<=c0
        onfly=a.nValue in SUPPORT_ND2NZ_GM2L0 and (not a.isBTrans or a.kValue*self.aw in SUPPORT_ND2NZ_GM2L0)
        mk=a.mValue>16*max(a.kValue,a.nValue) and a.kValue<=256
        bias=r.baseN*4 if a.hasBias else U64(0)
        bk=a.kValue if self.legacy else up(a.kValue,32//self.bw if a.isBTrans else 16)
        bn=a.nValue if self.legacy else up(a.nValue,16 if a.isBTrans else 32//self.bw)
        fits=h.l1Size//2-bias>bk*bn*self.bw
        base=mk and ((onfly and not a.nd2nzB) or (vnchw and fits))
        self.gate('BL1: full-load base predicate',base,MK_condition=mk,on_the_way=onfly,nd2nzB=a.nd2nzB,vnchw=vnchw,B_fits=fits)
        if base:return self.bl1_base()
        if self.legacy:return False
        tail=a.mValue//128%12
        tailok=((a.mValue+127)//128%12==1) if tail==0 else ((a.mValue+127)//128%12!=7) if tail==6 else 1<=tail<6
        special=(a.aType in ('fp16','bf16') and 30848<=a.mValue<=98048 and a.nValue==512 and a.kValue==512
                 and tailok and not a.isATrans and a.isBTrans and a.aFormat==a.bFormat=='ND' and not a.hasBias)
        if not self.gate('BL1: p0 NT core-split full-load predicate',special,tail=tail,tail_ok=tailok):return False
        e=self.enable;e.tilingEnableFullLoad=2;e.tilingEnableSplitCore=0
        r.baseN=a.nValue//2;r.singleCoreN=r.baseN;r.stepM=1;r.stepN=1
        r.stepKb=cd(a.kValue,r.baseK);r.stepKa=r.stepKb;r.depthA1=2*r.stepKa;r.depthB1=r.stepKb
        r.baseM=128;r.singleCoreM=up(cd(a.mValue,12),r.baseM)
        r.dbL0c=2 if r.baseM*r.baseN*4*2<=h.l0CSize else 1
        r.l2Info.mTile=1;r.l2Info.nTile=1;r.l2Info.mTileBlock=cd(a.mValue,r.singleCoreM);r.l2Info.nTileBlock=1;r.l2Info.calOrder=1
        return True
    @sourced('DoBL1FullLoadTilingBase',1981,2024)
    def bl1_base(self):
        if not self.legacy:return super().bl1_base()
        a,h,r,e=self.args,self.hw,self.run,self.enable
        e.tilingEnableFullLoad=2;e.tilingEnableSplitCore=0;r.stepM=1
        r.baseN=up(min(a.nValue,r.baseN),16 if a.isBTrans else 32//self.aw)
        r.stepN=cd(a.nValue,r.baseN);r.stepKb=cd(a.kValue,r.baseK);r.stepKa=r.stepKb
        r.depthA1=2*r.stepKa;r.depthB1=r.stepN*r.stepKb
        load=r.baseK*(r.depthA1*r.baseM+r.depthB1*r.baseN)*self.aw+(r.baseN*self.aw if a.hasBias else 0)
        while load>h.l1Size:
            if r.baseM==0:raise SourceFieldOverflow('Legacy BL1 source halving loop has no terminating feasible baseM')
            load-=r.depthA1*r.baseM*r.baseK*self.aw;r.baseM=r.baseM//2;load+=r.depthA1*r.baseM*r.baseK*self.aw
        r.singleCoreM=2*r.baseM;r.singleCoreN=a.nValue
        r.dbL0c=2 if r.baseM*r.baseN*4*2<=h.l0CSize else 1
        r.l2Info.mTile=1;r.l2Info.nTile=1;r.l2Info.mTileBlock=cd(a.mValue,r.singleCoreM);r.l2Info.nTileBlock=1;r.l2Info.calOrder=1
        return True
    def support_small_k(self,x,y):
        return False if self.legacy else super().support_small_k(x,y)
    def force_group_fp32(self):
        return False if self.legacy else super().force_group_fp32()
    def support_single(self):
        if not self.legacy:return super().support_single()
        a,h=self.args,self.hw
        if a.isHf32 and not self.n_align and a.mValue*a.nValue<4*MB_SIZE:return False
        if a.kValue>=SPLIT_K_THRES:return True
        large=a.mValue*a.kValue>=5*384*384 and a.nValue>=1024 and a.mValue>=384 and a.kValue>=384 and a.mValue*a.nValue>=1024*384*h.aicNum
        if not self.enable_cache and large:return True
        tlb=a.isATrans and not a.isBTrans and self.n_align and a.kValue>=11000 and ((a.mValue%8192==0 and a.nValue>=6144) or (a.nValue%8192==0 and a.mValue>=6144))
        if tlb:return True
        return (not a.isATrans and a.isBTrans and a.kValue%16384==0 and 1280<=a.mValue<=8192 and 1280<=a.nValue<=8192
                and a.aFormat==a.bFormat=='ND' and a.aType in ('fp16','bf16') and h.aicNum==24)
    @sourced('CheckSingleTilingOk',2401,2436)
    def check_single(self,r):
        if not self.legacy:return super().check_single(r)
        a,h,e=self.args,self.hw,self.enable
        if not self.gate('singleCoreN >= 512',r.singleCoreN>=512,singleCoreN=r.singleCoreN):return False
        avg=ratio_f32(a.mValue*a.nValue,r.singleCoreN*r.singleCoreM*h.aicNum)
        if not self.gate('average occupancy >= float32(0.7)',avg>=f32(.7),occupancy=avg):return False
        reject=self.n_align and 896<=a.nValue<=2048 and (r.singleCoreN<=640 or avg<f32(.85))
        if self.gate('prefer multicore split-K edge',reject):return False
        if 256<r.singleCoreM<=384 and not(a.aType=='fp32' and not self.n_align):self.mk33(r)
        r.singleCoreK=r.stepKa*r.baseK;r.dbL0c=2
        if not h.supportL0c2out or h.supportL12BtBf16 or not self.support_single():return False
        e.tilingEnableSplitCore=2;e.tilingEnableFullLoad=0;e.tilingEnableFixOpti=0
        self.run=r.clone('runInfo');self.run.needUpdate=True
        return True
    def deterministic_small_mn(self):
        return False if self.legacy else super().deterministic_small_mn()
    def more_multi_args(self):
        if not self.legacy:
            return super().more_multi_args()
        a=self.args;ia=a.mValue if a.isATrans else a.kValue;ib=a.kValue if a.isBTrans else a.nValue
        threshold = 384 if self.profile == 'installed_81' else 512
        if threshold//self.aw<=ia<=65535:a.nd2nzA=False
        if threshold//self.bw<=ib<=65535:a.nd2nzB=False
    def optimize_det_cores(self):
        if not self.legacy:return super().optimize_det_cores()
    def nd2nz_geometry(self,width,n,d):
        if not self.legacy:return super().nd2nz_geometry(width,n,d)
        # The only non-guard difference is p0's MATA 96x512 specialization.
        # Execute the shared old body directly, without changing hardware state.
        vc=max(2*self.run.usedCoreNum,U64(1));threshold=U64(2048)//width;c0=U64(32)//width
        na=up(n,16);da=up(d,c0)
        if d<=threshold:
            bd=max(up(d,c0),U64(1));bn=self.hw.ubSize//2//width//bd
            rounds=max(cd(cd(na,vc),bn),U64(1));bn=max(cd(cd(na,vc),rounds),U64(16))
            while bn>16 and self.ub_overflow(na,n,bn,bd,width):bn-=1
            return bn,bd
        last=U64(0);bestn=U64(16);bestd=U64(CALC_ND_BASIC[1])//width
        for base in CALC_ND_BASIC:
            bd=max(min(da,U64(base)//width),U64(1));dl=cd(da,bd);dt=da%bd
            if 0<dt<512//width:
                if bd*width==CALC_ND_BASIC[0]:continue
                dl-=1;bd=max(up(cd(da,dl),c0),U64(1))
            bn=max(self.hw.ubSize//2//width//bd,U64(16))
            if bn*bd*width*2>self.hw.ubSize or self.ub_overflow(na,n,bn,bd,width):continue
            nl=cd(na,bn);tail=nl*dl%vc
            while bn>16:
                if self.ub_overflow(na,n,bn,bd,width):
                    bn-=1;nl=cd(na,bn);tail=nl*dl%vc;continue
                if tail==0:return bn,bd
                if tail>last:last=tail;bestd=bd;bestn=bn
                bn-=1;nl=cd(na,bn);tail=nl*dl%vc
        return bestn,bestd
    def get_key_arguments(self):
        e=self.enable
        return {'LOADMODE':int(e.tilingEnableFullLoad),'SPLITCOREMODE':int(e.tilingEnableSplitCore),
                'FIXOPTI':int(e.tilingEnableFixOpti),'MIXND2NZ':self.mix_type(),
                'SPECIALOPT':int(e.tilingEnableSpecialOpti),'FP32ADDMM':0}
    def set_run_info(self):
        values=super().set_run_info()
        if self.legacy:values={k:v for k,v in values.items() if k not in ('isNzA','isNzB')}
        return values

    def deterministic_workspace(self):
        if not self.legacy:return super().deterministic_workspace()
        return self.run.usedCoreNum*self.run.singleCoreN*self.run.singleCoreM*8+20*MB_SIZE
