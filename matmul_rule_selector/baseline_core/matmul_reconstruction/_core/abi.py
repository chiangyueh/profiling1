"""Lossless explicit-payload codecs. No guessed TCube words or padding.

Named source profiles explicitly select their 272-byte or 280-byte container.
"""
import hashlib, struct
from types import MappingProxyType
from .errors import InvalidRequest, MissingDefinition

LAYOUTS={
 'legacy272_cube200':{'size':272,'run_names':('transA','transB','nd2nzA','nd2nzB','isHf32'),'flag_offset':248,'vector_offset':256,'padding_offsets':(220,244,252)},
 'reference280_cube200':{'size':280,'run_names':('transA','transB','nd2nzA','nd2nzB','isNzA','isNzB','isHf32'),'flag_offset':256,'vector_offset':264,'padding_offsets':(220,252,260)}}
TARGET_PROFILE='target280_cube200'
LAYOUTS[TARGET_PROFILE]=dict(LAYOUTS['reference280_cube200'])
LAYOUTS=MappingProxyType({k:MappingProxyType(v) for k,v in LAYOUTS.items()})
L2_NAMES=('mTileCntL2','nTileCntL2','mTileBlock','nTileBlock','calOrder')
VECTOR_NAMES=('baseAN','baseAD','baseBN','baseBD')

def u32(x):
    if type(x) is not int or not 0<=x<2**32:raise InvalidRequest('Every ABI word must be an explicit uint32 integer')
    return x

def layout(name):
    if name not in LAYOUTS:raise InvalidRequest('An explicit supported conditional layout is required')
    return LAYOUTS[name]

def pack_packet(payload,layout_id):
    spec=layout(layout_id)
    required={'cube_words','tileL2cacheTiling','matmulRunInfo','l2CacheFlag','vector','padding_hex'}
    if not isinstance(payload,dict) or set(payload)!=required:raise InvalidRequest('Exact ABI payload requires precisely '+', '.join(sorted(required)))
    cube=payload['cube_words']
    if not isinstance(cube,list) or len(cube)!=50:raise InvalidRequest('All 50 TCube storage words must be supplied; no defaults')
    b=bytearray(spec['size']);struct.pack_into('<50I',b,0,*map(u32,cube))
    for name,names,offset in [('tileL2cacheTiling',L2_NAMES,200),('matmulRunInfo',spec['run_names'],224),('vector',VECTOR_NAMES,spec['vector_offset'])]:
        d=payload[name]
        if not isinstance(d,dict) or set(d)!=set(names):raise InvalidRequest('Missing/extra ABI fields in '+name)
        struct.pack_into('<'+'I'*len(names),b,offset,*[u32(d[n])for n in names])
    struct.pack_into('<I',b,spec['flag_offset'],u32(payload['l2CacheFlag']))
    pad=payload['padding_hex']
    if not isinstance(pad,dict) or set(pad)!=set(map(str,spec['padding_offsets'])):raise InvalidRequest('Every padding region must be supplied explicitly')
    for off in spec['padding_offsets']:
        try:part=bytes.fromhex(pad[str(off)])
        except (ValueError,TypeError):raise InvalidRequest('Invalid padding hex') from None
        if len(part)!=4:raise InvalidRequest('Padding region must be exactly four bytes')
        b[off:off+4]=part
    return bytes(b)

def unpack_packet(raw,layout_id):
    spec=layout(layout_id)
    if not isinstance(raw,bytes) or len(raw)!=spec['size']:raise InvalidRequest('Raw buffer length does not match selected layout')
    out={'cube_words':list(struct.unpack_from('<50I',raw,0)),'l2CacheFlag':struct.unpack_from('<I',raw,spec['flag_offset'])[0],
         'padding_hex':{str(o):raw[o:o+4].hex()for o in spec['padding_offsets']}}
    for name,names,off in [('tileL2cacheTiling',L2_NAMES,200),('matmulRunInfo',spec['run_names'],224),('vector',VECTOR_NAMES,spec['vector_offset'])]:
        out[name]=dict(zip(names,struct.unpack_from('<'+'I'*len(names),raw,off)))
    return out

