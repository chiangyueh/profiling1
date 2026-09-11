"""Deterministic trace records; no filesystem, clock, environment or network reads."""
from functools import wraps
from .arithmetic import U64


def plain(value):
    if isinstance(value, Record): return {k: plain(v) for k, v in sorted(value.values.items())}
    if isinstance(value, U64): return int(value)
    if isinstance(value, dict): return {str(k): plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)): return [plain(x) for x in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    return '<non-data>'


class Record:
    __slots__ = ('owner', 'name', 'values')
    def __init__(self, owner, name, **values):
        object.__setattr__(self, 'owner', owner)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'values', {})
        for k,v in values.items(): self.values[k] = U64(v) if type(v) is int else v
    def __getattr__(self, key):
        try: return self.values[key]
        except KeyError: raise AttributeError(key) from None
    def __setattr__(self, key, value):
        if key in ('owner', 'name', 'values'): return object.__setattr__(self, key, value)
        if type(value) is int: value = U64(value)
        before = self.values.get(key)
        self.values[key] = value
        if self.owner.trace_enabled:
            self.owner.emit('mutation', field=self.name + '.' + key,
                            before=plain(before), after=plain(value))
    def clone(self, name):
        values={k: (v.clone(name+'.'+k) if isinstance(v, Record) else v)
                for k,v in self.values.items()}
        return Record(self.owner, name, **values)


def sourced(symbol, first, last, file='matmul_v3_base_tiling.cpp'):
    def decorate(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            previous = self.source_site
            self.source_site = f'{file}:{first}'
            self.function_counts[symbol] = self.function_counts.get(symbol, 0) + 1
            if self.trace_enabled:
                self.emit('enter', symbol=symbol, source_range=f'{file}:{first}-{last}',
                          arguments=plain(args), keyword_arguments=plain(kwargs), state=self.snapshot())
            try:
                result = fn(self, *args, **kwargs)
                if self.trace_enabled:
                    self.emit('exit', symbol=symbol, result=plain(result), state=self.snapshot())
                return result
            finally:
                self.source_site = previous
        wrapper.source_symbol=symbol
        wrapper.source_file=file
        wrapper.source_first=first
        wrapper.source_last=last
        return wrapper
    return decorate


class TraceContext:
    def init_trace(self, enabled):
        self.trace_enabled=bool(enabled)
        self.trace=[]
        self.source_site='entry:0'
        self.function_counts={}
        self.loop_counts={}
        self.predicate_counts={}
    def site(self, file, line):
        self.source_site=f'{file}:{line}'
    def emit(self, event, **payload):
        self.trace.append({'sequence':len(self.trace),'event':event,'source':self.source_site,**payload})
    def predicate(self, file, line, expression, value, variables):
        self.source_site=f'{file}:{line}'
        key=self.source_site+'|'+expression
        outcome=bool(value)
        count=self.predicate_counts.setdefault(key, [0,0])
        count[int(outcome)]+=1
        if self.trace_enabled:
            inputs={k:plain(v) for k,v in sorted(variables.items()) if k!='self' and
                    (v is None or isinstance(v,(Record,str,int,float,bool,list,tuple,dict)))}
            self.emit('predicate', expression=expression, inputs=inputs, result=outcome)
        return value
    def iterations(self, name, iterable):
        for item in iterable:
            self.loop_counts[name]=self.loop_counts.get(name,0)+1
            yield item
    def count_loop(self, name):
        self.loop_counts[name]=self.loop_counts.get(name,0)+1
    def snapshot(self):
        return {name:plain(getattr(self,name)) for name in ('args','run','enable') if hasattr(self,name)}
