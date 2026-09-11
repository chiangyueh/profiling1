"""Unsigned source semantics from common_host_tiling/op_util.h:24-69."""
import struct

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

class U64(int):
    """Keep uint64_t intermediate arithmetic, including subtraction, modulo 2**64."""
    def __new__(cls, value=0):
        return int.__new__(cls, int(value) & MASK64)
    def __add__(self, other):
        if isinstance(other, float): return float(self) + other
        return U64(int(self) + int(other))
    __radd__ = __add__
    def __sub__(self, other):
        if isinstance(other, float): return float(self) - other
        return U64(int(self) - int(other))
    def __rsub__(self, other):
        if isinstance(other, float): return other - float(self)
        return U64(int(other) - int(self))
    def __mul__(self, other):
        if isinstance(other, float): return float(self) * other
        return U64(int(self) * int(other))
    __rmul__ = __mul__
    def __floordiv__(self, other): return U64(int(self) // int(other))
    def __rfloordiv__(self, other): return U64(int(other) // int(self))
    def __mod__(self, other): return U64(int(self) % int(other))
    def __rmod__(self, other): return U64(int(other) % int(self))
    def __lshift__(self, other): return U64(int(self) << int(other))
    def __rshift__(self, other): return U64(int(self) >> int(other))
    def __and__(self, other): return U64(int(self) & int(other))
    __rand__ = __and__
    def __or__(self, other): return U64(int(self) | int(other))
    __ror__ = __or__
    def __xor__(self, other): return U64(int(self) ^ int(other))
    __rxor__ = __xor__
    def __neg__(self): return U64(-int(self))


def ceil_div(x, y):
    """op_util.h:37: denominator zero returns x (not an exception)."""
    x, y = U64(x), U64(y)
    return x // y + (1 if x % y else 0) if x and y else x


def floor_div(x, y):
    return U64(x) if not y else U64(x) // y


def align_up(x, alignment):
    return ceil_div(x, alignment) * U64(alignment)


def align_down(x, alignment):
    return U64(x) // alignment * alignment if alignment else U64(0)


def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]


def ratio_f32(numerator, denominator):
    return f32(f32(numerator) / f32(denominator))


def nz_sequential_size(m, n, base_m, base_n):
    """Exact factorisation of base.cpp:125-142, modulo uint64_t; O(1)."""
    if not base_m or not base_n:
        return U64(0)
    def axis_sum(size, base):
        size, base = U64(size), U64(base)
        return (size // base) * align_up(base, 16) + align_up(size % base, 16)
    return axis_sum(m, base_m) * axis_sum(n, base_n)
