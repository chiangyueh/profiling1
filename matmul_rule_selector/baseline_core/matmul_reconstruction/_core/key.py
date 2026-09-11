"""Primary-domain template rules, not a shape/tiling lookup table.

Values are declared by mat_mul_v3_tiling_key.h. UINT fields encode declaration
indices. Here (and only because the lists are dense, ordered from zero) the
indices equal the values. Kernel metadata is not a seventh numeric parameter.
"""
from .errors import InvalidRequest, UnsupportedDomain
NAMES=('LOADMODE','SPLITCOREMODE','FIXOPTI','MIXND2NZ','SPECIALOPT','FP32ADDMM')
WIDTHS=(4,8,4,4,4,4)
DECLARED_VALUES=(tuple(range(3)),tuple(range(7)),tuple(range(3)),tuple(range(3)),(0,1),(0,1))

def validate_primary_selection(values):
    if not isinstance(values,(tuple,list)) or len(values)!=6:
        raise InvalidRequest('Exactly six numeric template arguments are required')
    for value,domain in zip(values,DECLARED_VALUES):
        if type(value) is not int or value not in domain:
            raise InvalidRequest('Template argument outside its declared integer domain')
    load,split,fix,mix,special,fp32=values
    legal=False
    if fp32==0:
        if load==0 and split==0 and fix==0:
            legal=(special==0 and mix in (0,1)) or (special==1 and mix==1)
        elif load==2 and split==0 and fix in (0,1):
            legal=special==0 and mix in (0,1)
        elif load==0 and split in (2,6) and fix==0:
            legal=special==0 and mix in (0,1)
        elif ((load==1 and split==2) or (load==0 and split==5)) and fix==0:
            legal=special==0 and mix==1
        elif load==0 and split==3 and fix in (0,2):
            legal=special==0 and mix in (0,1)
    if not legal:raise UnsupportedDomain('Tuple is not selected by the primary FP16/ND architecture-220 target key header')
    return tuple(values)

def encode_indices(values,widths,domains):
    """Generic list-index packing, used to test non-identity declarations too."""
    if len(values)!=len(widths) or len(values)!=len(domains):
        raise InvalidRequest('Key declarations and values must have equal lengths')
    result=0;shift=0
    for value,width,domain in zip(values,widths,domains):
        if type(width) is not int or width<1 or shift+width>64:
            raise InvalidRequest('Invalid numeric-key bit widths')
        if not domain or len(set(domain))!=len(domain):raise InvalidRequest('Empty or repeated declaration values')
        if type(value) is not int or value not in domain:raise InvalidRequest('Value absent from declaration')
        index=domain.index(value)
        if index>=1<<width:raise InvalidRequest('Declaration index does not fit its bit width')
        result|=index<<shift;shift+=width
    return result

def numeric_key(arguments):
    if not isinstance(arguments,dict) or set(arguments)!=set(NAMES):
        raise InvalidRequest('Exactly the six named numeric template arguments are required')
    values=validate_primary_selection(tuple(arguments[name] for name in NAMES))
    return encode_indices(values,WIDTHS,DECLARED_VALUES)

def kernel_metadata(arguments):
    values=validate_primary_selection(tuple(arguments[name] for name in NAMES))
    load,split,fix,mix,special,fp32=values
    return 'ASCENDC_TPL_AIC_ONLY' if split==0 and fix==0 and mix==1 else 'ASCENDC_TPL_MIX_AIC_1_2'
