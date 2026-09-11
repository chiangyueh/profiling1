"""Input-only source selector. No environment, filesystem, SDK or history access."""
import hashlib
from .profiles import PROFILES, COMMON_HASHES, INITIALIZER_HASHES
from .family_engine import VersionedEngine
from ._core.initializer import initialize_cube,CUBE_FIELDS
from ._core.initializer_runtime import i32
from ._core.arithmetic import ceil_div,align_up
from ._core.trace import plain
from ._core.abi import pack_packet,LAYOUTS,VECTOR_NAMES
from ._core.errors import InvalidRequest,UnsupportedDomain
from .basic_block import generate as generate_basic
from .trace_output import locations,attach_sources,candidates_for_family,resource_facts
from .conversion import postprocess, convert_to_run

PROFILES['installed_81'] = {
    'hashes': {},
    'abi': 'legacy272_cube200',
    'key_encoding': 'decimal_10^19',
}

HARDWARE_INTS=('aicNum','aivNum','ubSize','l1Size','l2Size','l0CSize','l0ASize','l0BSize','btSize')
BASE_DEFAULTS={'layoutA':'ND','layoutB':'ND','layoutC':'ND','transA':False,'transB':False,
 'bias':False,'hf32':False,'forceGrpAccForFp32':False,'dtype':'fp16'}

def normalize(request,hardware):
    if not isinstance(request,dict) or not isinstance(hardware,dict):raise InvalidRequest('request and hardware must be dictionaries')
    extra=set(request)-set(BASE_DEFAULTS)-{'M','N','K','output_dtype','bias_dtype'}
    if extra:raise InvalidRequest('Unknown request keys: '+str(sorted(extra)))
    r={**BASE_DEFAULTS,**request};r.setdefault('output_dtype',r['dtype']);r.setdefault('bias_dtype','fp32' if r['dtype']=='bf16' else r['output_dtype'])
    for name in ('M','N','K'):
        if type(r.get(name)) is not int or not 1<=r[name]<=2147483647:raise InvalidRequest(name+' must be in [1,INT32_MAX]')
    for name in ('transA','transB','bias','hf32','forceGrpAccForFp32'):
        if type(r[name]) is not bool:raise InvalidRequest(name+' must be boolean')
    if r['dtype'] not in ('fp16','bf16','fp32') or r['output_dtype']!=r['dtype']:
        raise UnsupportedDomain('Source-reference domain has matching FP16/BF16/FP32 inputs and output')
    if r['bias_dtype'] not in ('fp16','bf16','fp32'):raise UnsupportedDomain('Unsupported bias type')
    if r['bias'] and r['bias_dtype'] not in ({'fp16','fp32'} if r['dtype']=='fp16' else {'fp32'}):
        raise UnsupportedDomain('Source OpSpecificCheck bias dtype: FP16 allows FP16/FP32; BF16/FP32 require FP32')
    if any(r[n]!='ND' for n in ('layoutA','layoutB','layoutC')):raise UnsupportedDomain('Source-reference domain is GM ND/ND/ND')
    if r['dtype']!='fp32' and (r['hf32'] or r['forceGrpAccForFp32']):raise InvalidRequest('HF32/group accumulation only applies to FP32')
    allowed=set(HARDWARE_INTS)|{'supportL0c2out','supportL12BtBf16','cubeFreq','npuArch','socVersion','socVersionStr','cannVersion','profile_provenance'}
    if set(hardware)-allowed:raise InvalidRequest('Unknown hardware keys: '+str(sorted(set(hardware)-allowed)))
    h=dict(hardware)
    for n in HARDWARE_INTS:
        if type(h.get(n)) is not int or not 1<=h[n]<2**63:raise InvalidRequest('Explicit positive integer hardware field required: '+n)
    for n in ('supportL0c2out','supportL12BtBf16'):
        if type(h.get(n)) is not bool:raise InvalidRequest('Explicit hardware boolean required: '+n)
    if h['supportL12BtBf16'] or not h['supportL0c2out']:raise UnsupportedDomain('Selected domain is L0C2OUT=true, L12BtBf16=false')
    if not 1<=h['aicNum']<=32:raise UnsupportedDomain('Core budget must be within [1,32]')
    if h.get('npuArch',220)!=220:raise UnsupportedDomain('Architecture 220 is the selected domain')
    if not(32768<=h['l0ASize']<=262144 and 32768<=h['l0BSize']<=262144 and h['l0CSize'] in (131072,262144)
        and 131072<=h['l1Size']<=1048576 and 32768<=h['ubSize']<=1048576 and h['aivNum']<=128):
        raise UnsupportedDomain('Hardware capacity outside bounded source-reference domain')
    return r,h

def _key(args,profile):
    if profile!='p0':return 10**19+int(args['MIXND2NZ']==1)+10*args['SPLITCOREMODE']+100*args['LOADMODE']+10000*args['FIXOPTI']
    values=[args[n] for n in ('LOADMODE','SPLITCOREMODE','FIXOPTI','MIXND2NZ','SPECIALOPT','FP32ADDMM')]
    return sum(value<<shift for value,shift in zip(values,(0,4,12,16,20,24)))

def select(request,hardware,*,source_profile='p0',trace=True,entry='matmul_v3',route='ALL'):
    """Return a complete source packet, or an explicit source-domain rejection.

    entry='matmul_v3' executes the source MatMulV3 default/explicit route.
    entry='basic_block' exposes the common generator as a controlled subroutine
    experiment. It is NOT labelled as the default MatMulV3 path.
    """
    if source_profile not in PROFILES:raise InvalidRequest('Unknown source profile')
    if entry not in ('matmul_v3','basic_block'):raise InvalidRequest('Unknown entry')
    if route not in ('ALL','BASE','SINGLE_CORE_SPLIT_K','DETERMINISTIC_SPLIT_K','AL1_FULL_LOAD','BL1_FULL_LOAD','BL1_FULL_LOAD_FIXPIPE'):
        raise InvalidRequest('Unknown source route')
    if source_profile=='p0' and route in ('AL1_FULL_LOAD','BL1_FULL_LOAD','BL1_FULL_LOAD_FIXPIPE'):
        raise UnsupportedDomain('This explicit forced route exists only in tested_repository')
    if source_profile in ('cann_ops_snapshot','installed_81') and route in ('AL1_FULL_LOAD','BL1_FULL_LOAD','BL1_FULL_LOAD_FIXPIPE'):
        raise UnsupportedDomain('This snapshot does not declare the tested forced-route extension')
    r,h=normalize(request,hardware)
    e=VersionedEngine(r,h,source_profile,trace=trace,route=route)
    e.get_more_args()
    initial=initialize_cube(r,h,trace=trace)
    if initial['return_code']!=0:raise UnsupportedDomain('Original MultiCore initializer rejected this input, status='+str(initial['return_code']))
    e.set_run_info();e.set_params_v310()
    if not e.legacy:e.select_nz()
    basic=None
    if entry=='matmul_v3':
        e.do_basic()
        if not e.legacy:e.optimize_step_k()
    else:
        basic=generate_basic(r,h,trace=trace)
        if basic.get('selected') is None:raise UnsupportedDomain('Common BASIC_BLOCK generator not applicable or generated no valid candidate')
        # Converter below explicitly projects the BASIC_BLOCK subroutine result.
        # It is never substituted into MatMulV3's DoIncreTiling gate.
        basic['postprocess']=postprocess(r,h,basic['candidates'][basic['selected']]['result'])
        convert_to_run(e,basic['postprocess'])
        e.candidate_stream=basic['candidates']
        e.attempted_families=[{'family':'BASIC_BLOCK_SUBROUTINE','accepted':True,'reason':basic['checks']}] 
        e.chosen_source_family='BASIC_BLOCK_SUBROUTINE'
    e.set_nd2nz_info();packet=e.finalize_fields();workspace,parts=e.workspace()
    cube=dict(initial['cube']);provenance={n:'MultiCore initializer.SetFinalTiling' for n in CUBE_FIELDS}
    provenance['reserved']='zero-initialized reserved storage'
    for n,v in packet['matmulTiling'].items():
        if v is not None:cube[n]=i32(v);provenance[n]='MatmulV3 DoLibApiTiling / family write'
    packet['matmulTiling']=cube
    abi=PROFILES[source_profile]['abi'];spec=LAYOUTS[abi]
    words=[int(cube[n])&0xffffffff for n in CUBE_FIELDS]
    payload={'cube_words':words,'tileL2cacheTiling':dict(packet['tileL2cacheTiling']),
      'matmulRunInfo':dict(packet['matmulRunInfo']),'l2CacheFlag':packet['l2cacheUseInfo']['l2CacheFlag'],
      'vector':{n:packet[n] for n in VECTOR_NAMES},'padding_hex':{str(n):'00000000' for n in spec['padding_offsets']}}
    raw=pack_packet(payload,abi);args=e.get_key_arguments();key=_key(args,source_profile)
    for attempted in e.attempted_families:
        attempted['source_anchors']=locations(source_profile,attempted.get('symbol','DoIncreTiling'))
    attach_sources(source_profile,e.trace)
    return {'source_profile':source_profile,'source_hashes':PROFILES[source_profile]['hashes'],
      'common_source_hashes':COMMON_HASHES,'initializer_source_hashes':INITIALIZER_HASHES,'entry':entry,'route':route,'request':r,'hardware':h,
      'attempted_families':e.attempted_families,'candidate_stream':e.candidate_stream,
      'winning_family_candidates':candidates_for_family(e,packet),
      'resource_facts':resource_facts(r,h,packet,e.chosen_source_family),
      'selected_family':e.chosen_source_family,'template_arguments':args,'tilingEnable':plain(e.enable),
      'initializer':initial,'runInfo':plain(e.run),'argument_state':plain(e.args),'tilingData':packet,
      'field_provenance':provenance,'cube_words':words,'cube_words_signed':[cube[n] for n in CUBE_FIELDS],
      'abi_profile':abi,'abi_payload':payload,'raw_buffer_hex':raw.hex(),'raw_buffer_bytes':len(raw),
      'raw_buffer_sha256':hashlib.sha256(raw).hexdigest(),'tiling_key':key,
      'blockDim':int(e.run.usedCoreNum),'workspace_bytes':workspace,'workspace_components':parts,'scheduleMode':1,
      'computed_l2_cache_flag_after_packet_write':int(e.l2_cache_flag),'basic_block':basic,
      'incremental':getattr(e,'incremental_result',None),
      'trace':e.trace,'function_counts':dict(e.function_counts),'predicate_counts':dict(e.predicate_counts),
      'binary_metadata':{'installed_build_binding':None,'kernel_suffix':None},
      'storage_policy':'Explicit zero initialization for ABI padding and reserved storage; source writes otherwise preserved'}
