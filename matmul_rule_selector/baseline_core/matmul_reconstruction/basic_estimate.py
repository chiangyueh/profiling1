"""BASIC_BLOCK_ESTIMATE_TYPE, not GemmCycleEstimate.

A transcription of the supplied GemmBBEstimate and its GemmEstimate base. Model
constants are inherited source constants, not newly fitted hardware estimates.
Scoring normalizes a private copy; callers must retain the original candidate.
"""
from dataclasses import asdict, replace
from math import exp
from .basic_block import cd, up
from .request_bytes import request_nd2nz, request_nz2nd, same_nd2nz, same_nz2nd


def estimate(r, h, candidate):
    """Return the source score and its full numeric decomposition for MatMul."""
    s = replace(candidate)
    M, N, K = r['M'], r['N'], r['K']
    w = {'fp16': 2, 'bf16': 2, 'fp32': 4}[r['dtype']]
    ow = {'fp16': 2, 'bf16': 2, 'fp32': 4}[r.get('output_dtype', r['dtype'])]
    cube = 8 if w == 4 else 16
    rm, rn, rk = cd(M, 16), cd(N, 16), cd(K, cube)
    ms = cd(cd(rm, s.m_dim), s.m_l1)
    ns = cd(cd(rn, s.n_dim), s.n_l1)
    akf, bkf = cd(rk, s.kal1_16), cd(rk, s.kbl1_16)
    cm, cn, ck = min(ms*s.m_l1, rm), min(ns*s.n_l1, rn), rk
    if h['supportL0c2out'] and s.k_dim != 1:
        k1 = max(s.kal1_16, s.kbl1_16)
        ks = cd(cd(rk, s.k_dim), k1)
        akf, bkf = ks*(k1//s.kal1_16), ks*(k1//s.kbl1_16)
        ck = min(ks*k1, rk)
    s.batch_dim = 1
    s.m_dim, s.n_dim, s.k_dim = cd(rm, cm), cd(rn, cn), cd(rk, ck)
    afull, bfull = ms == 1 and akf == 1, ns == 1 and bkf == 1
    akfull, bkfull = not afull and akf == 1, not bfull and bkf == 1
    l0ra, l0rb = cn//s.n_l0, cm//s.m_l0
    if afull or bfull:
        if afull and bfull: l1ra, l1rb = 1, 1
        elif bfull: l1ra, l1rb = cn//s.n_l0, 1
        elif bkfull: l1ra, l1rb = 1, 1
        else: l1ra, l1rb = 1, cm//s.m_l0
    elif akfull or bkfull:
        l1ra, l1rb = (1, cm//s.m_l0) if akfull else (cn//s.n_l0, 1)
    else:
        l1ra, l1rb = l0ra, l0rb
    mp, np, kp = cd(M, s.m_dim), cd(N, s.n_dim), cd(K, s.k_dim)
    m, n, k = up(mp, 16), up(np, 16), up(kp, cube)
    lm, lak, lbk, ln = s.m_l1*16, s.kal1_16*cube, s.kbl1_16*cube, s.n_l1*16
    m0, k0, n0 = s.m_l0*16, s.k_l0*cube, s.n_l0*16
    active = min(s.m_dim*s.n_dim*s.k_dim, h['aicNum'])
    mr = min(s.m_dim, active)
    nr = min(s.n_dim, active//mr)
    kr = min(s.k_dim, active//(mr*nr))
    blockext = 8 if s.m_dim*s.n_dim > 24 else max(mr, nr)
    kshift = ((mp != 1 or np != 1) and s.k_dim == 1 and blockext > 2
              and cd(K, lak) > blockext//2)
    inward = mp*w % 32 == 0 and kp*w % 32 == 0 and np*w % 32 == 0
    b2 = h['l2Size'] == 192*1024**2
    freq, ddrro, ddrphy = (1.8, 850., 37.) if b2 else (1.5, 530., 27.)
    requests, bw = {}, {}

    def bandwidth(tag, geometry):
        rows, cols, stride, size = geometry
        out = tag == 'C'
        counts = request_nz2nd(geometry) if out else request_nd2nz(geometry)
        same = same_nz2nd(geometry) if out else same_nd2nz(geometry)
        payload = rows*cols*size
        req = counts.size()
        if req == 0:
            raise ValueError('source request model produced zero requests')
        dma_cores = nr if tag == 'A' else mr if tag == 'B' else min(s.k_dim, active)
        dma_bytes = dma_cores*payload

        def biu(latency, ots):
            base = 1.0*payload*ots/(latency*req)
            eff = (dma_bytes/base)/(dma_bytes/base+latency/2.0)
            if cols != stride or cols*size != 32:
                eff *= .75
            return base*eff

        samecores = 1
        if tag == 'A':
            samecores = min(active, nr)
            if kshift: samecores = 1 if nr > mr else nr
        elif tag == 'B':
            samecores = min(active, mr)
            if kshift: samecores = 1 if mr >= nr else mr
        tilesize = 1.0*samecores*rows*max(cols*w, 128)/1000000
        mata_eff = 1-.91*exp(-.053*tilesize)
        penalty = samecores*(same-1)*req/mata_eff*.03
        mata_cycles = 1.0*req*active/8*2/freq+penalty
        mata = 1.0*payload/mata_cycles
        ddrmata = min(1.0*128*8*2/active/freq*(1.0*payload/counts.align_sum()), mata)
        reqdetail = {'geometry': geometry, 'requests': asdict(counts), 'payload_bytes': payload,
                     'aligned_bytes': counts.align_sum(), 'same_address_count': same,
                     'same_address_cores': samecores, 'mata_efficiency': mata_eff,
                     'same_address_penalty': penalty, 'mata_bw': mata, 'ddr_mata_bw': ddrmata}
        if out:
            b = biu(96, 48)
            bw['l0C_l2'] = min(128., 2550./active, b, 128., mata)
            if s.k_dim > 1: bw['l0C_l2'] *= .4
            reqdetail['l2_biu_bw'] = b
        else:
            l2b, ddrb = biu(236, 64), biu(347, 64)
            bw['l2_l1'+tag] = min(2550./active, 256., l2b, 128., mata)
            bw['ddr_l1'+tag] = min(ddrro/active, 256., ddrb, ddrphy, ddrmata)
            bw['l1_l0'+tag] = 256. if tag == 'A' else 128.
            reqdetail.update(l2_biu_bw=l2b, ddr_biu_bw=ddrb)
        requests[tag] = reqdetail

    am, ak = (lm, lak) if inward else (min(mp, lm), min(kp, lak))
    bk, bn = (lbk, ln) if inward else (min(kp, lbk), min(np, ln))
    bandwidth('A', (ak, am, M, w) if r.get('transA', False) else (am, ak, K, w))
    bandwidth('B', (bn, bk, K, w) if r.get('transB', False) else (bk, bn, N, w))
    bandwidth('C', (lm, ln, N, ow) if inward else (min(mp, lm), min(np, ln), N, ow))
    compute_k = (8 if r.get('hf32', False) else 4) if w == 4 else cube
    cube_tile = 1.0*s.m_l0*(k0//compute_k)*s.n_l0
    gamma = max((1.0*m0*k0*w/256)/cube_tile, (1.0*n0*k0*w/128)/cube_tile, 1.)
    cube_cycle = 1.0*up(m, lm)*up(k, lbk)*up(n, ln)/(16*compute_k*16)*gamma
    a_bytes, b_bytes = 1.0*mp*kp*w, 1.0*np*kp*w
    if inward:
        a_bytes, b_bytes = 1.0*up(mp, lm)*up(kp, lak)*w, 1.0*up(np, ln)*up(kp, lbk)*w
    fits = not kshift and (M*K*w+K*N*w+M*N*ow) <= h['l2Size']
    ha = hb = None
    if fits:
        ac = a_bytes/bw['ddr_l1A']+a_bytes*(l1ra-1)/bw['l2_l1A']
        bc = b_bytes/bw['ddr_l1B']+b_bytes*(l1rb-1)/bw['l2_l1B']
    else:
        la, lb = cd(K, lak), cd(K, lbk)
        af, bf = M*K*w*la, K*N*w*lb
        aks, bks = mr*lm*kr*lak*w*la, nr*ln*kr*lbk*w*lb
        ma, mb = 1.0/nr, 1.0/mr
        if af+bf <= h['l2Size'] or af+bks <= h['l2Size']:
            ma, mb = 1.0/(l1ra*nr), 1.0/(l1rb*mr)
        elif bf+aks <= h['l2Size'] or aks+bks <= h['l2Size']:
            mb = 1.0/(l1rb*mr)
        ha, hb = 1-ma, 1-mb
        abw = 1.0/(ha/bw['l2_l1A']+(1-ha)/bw['ddr_l1A'])
        bbw = 1.0/(hb/bw['l2_l1B']+(1-hb)/bw['ddr_l1B'])
        ac, bc = a_bytes*l1ra/abw, b_bytes*l1rb/bbw
    mte2 = ac+bc
    mte2t = 1.0*lm*lak*w/bw['ddr_l1A']+1.0*ln*lbk*w/bw['ddr_l1B']
    mte1 = 1.0*m*k*w*l0ra/bw['l1_l0A']+1.0*n*k*w*l0rb/bw['l1_l0B']
    mte1t = 1.0*m0*k0*w/bw['l1_l0A']+1.0*n0*k0*w/bw['l1_l0B']
    fix = 1.0*mp*np*ow/bw['l0C_l2']
    fixt = 1.0*m0*n0*ow/bw['l0C_l2']
    bound = max(cube_cycle, mte1, mte2+fix)
    unit = s.db_l0c == 1 and s.k_dim == 1 and not kshift
    if abs(cube_cycle-bound) <= 2**-23:
        branch = 'cube'
        if s.db_l0c == 2:
            pipe = bound+mte2t+mte1t+fixt; branch += ':db'
        elif unit:
            pipe = bound+mte2t+mte1t+max(fix-cube_tile, fixt); branch += ':unit'
        else:
            pipe = bound+mte2t+mte1t+fix; branch += ':plain'
    elif abs(mte2+fix-bound) <= 2**-23:
        branch = 'mte2_fix'
        if s.db_l0c == 2:
            pipe = bound+cube_tile+mte1t; branch += ':db'
        elif unit:
            pipe = bound+mte1t+max(cube_tile, cube_cycle-cube_tile-mte2); branch += ':unit'
        else:
            pipe = bound+mte1t+max(cube_tile, cube_cycle-mte2); branch += ':plain'
    else:
        pipe = bound+cube_tile+mte2t+fix; branch = 'mte1'
    starts = cd(s.n_dim*s.m_dim*s.k_dim, h['aicNum'])
    return {'score': pipe*starts, 'normalized_scoring_copy': asdict(s),
            'core_shape_blocks': [cm, cn, ck], 'patch': [mp, np, kp],
            'active_cores': active, 'run': [mr, nr, kr],
            'load_flags': {'A_full': afull, 'B_full': bfull, 'A_K_full': akfull, 'B_K_full': bkfull},
            'l0_repeat': [l0ra, l0rb], 'l1_repeat': [l1ra, l1rb],
            'k_shuffle': kshift, 'shift_inwards': inward, 'l2_fits': fits,
            'l2_hit_rates': [ha, hb], 'requests': requests, 'bandwidth': bw,
            'cycles': {'cube': cube_cycle, 'cube_tile': cube_tile, 'mte1': mte1, 'mte1_tile': mte1t,
                       'mte2': mte2, 'mte2_tile': mte2t, 'fix': fix, 'fix_tile': fixt},
            'pipeline_branch': branch, 'pipeline_cycle': pipe, 'kernel_start': starts,
            'arithmetic': 'source double; FLT_EPSILON; first strict score minimum',
            'source': 'gemm/estimate/cache_tiling_basic_block_est.cc'}
