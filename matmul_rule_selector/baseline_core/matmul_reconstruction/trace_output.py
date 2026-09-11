"""Attach profile-correct source locations without changing decisions."""
from .source_anchors import ANCHORS

def locations(profile,symbol):
    found=ANCHORS.get(profile,{}).get(symbol,[])
    if not found:found=ANCHORS.get(profile,{}).get(symbol.split('::')[-1],[])
    if not found:found=ANCHORS['gemm'].get(symbol,[]) or ANCHORS['initializer'].get(symbol,[])
    return found

def attach_sources(profile,events):
    for e in events:
        symbol=e.get('active_symbol')or e.get('symbol','')
        exact=locations(profile,symbol)
        if 'source'in e:e['implementation_anchor']=e.pop('source')
        if 'source_range'in e:e['implementation_range']=e.pop('source_range')
        e['source_anchors']=exact
    return events

def candidates_for_family(engine,packet):
    selected=engine.chosen_source_family
    if selected=='INCREMENTAL_PATTERN':
        return engine.incremental_result['candidates']
    if selected=='BASIC_BLOCK_SUBROUTINE':return engine.candidate_stream
    attempts=[a for a in engine.attempted_families if a['accepted'] and a['family']==selected]
    if selected=='BASE':
        streams=[a for a in engine.candidate_stream if a.get('stage')=='CalcBase']
    else:streams=[]
    # Formula families have one final proposal, with ordered internal trials and
    # comparisons represented by the per-attempt trace, NOT a made-up cost score.
    result={'stage':'final_source_proposal','family':selected,'packet_fields':packet,
      'comparison':{'type':'first_accepted_family'if attempts else 'fallback_after_family_rejections',
                    'numeric_cost':None,'cross_family_argmin':False},
      'source_anchors':locations(engine.profile,attempts[0].get('symbol','DoBasicTiling')if attempts else'DoBasicTiling')}
    return streams+[result]

def resource_facts(r,h,packet,family):
    c=packet['matmulTiling'];l=packet['tileL2cacheTiling'];w=4 if r['dtype']=='fp32'else 2
    def padded(x,u):return ((x+u-1)//u)*u
    bm,bn,bk=(c[n]for n in('baseM','baseN','baseK'))
    l0a=padded(bm,16)*padded(bk,32//w)*w
    l0b=padded(bn,16)*padded(bk,32//w)*w
    l0c=padded(bm,16)*padded(bn,16)*4
    return {'L0A_tile_bytes':l0a,'L0B_tile_bytes':l0b,'L0C_accumulator_tile_bytes':l0c,
      'L0C_configured_buffer_bytes':l0c*c['dbL0C'],
      'L1_depth_formula_bytes':bm*bk*w*c['depthA1']+bn*bk*w*c['depthB1'],
      'L1_profile_available_bytes':h['l1Size'],'L1_depth_formula_is_not_allocation_proof':True,
      'launched_cores':c['usedCoreNum'],'logical_output_tiles':((r['M']+bm-1)//bm)*((r['N']+bn-1)//bn),
      'idle_cores_allowed':True,'logical_tail_alignment_required':False,
      'generic_L2_tiling_filter_applicable':family!='DETERMINISTIC_SPLIT_K',
      'allocation_validation':'family source checks; facts are not a replacement generic capacity gate'}
