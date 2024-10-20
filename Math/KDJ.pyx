# cython: language_level=3
import cython
from libc.math cimport fmax, fmin

cdef class KDJ_Item:
    cdef public double k, d, j

    def __init__(self, double k, double d, double j):
        self.k = k
        self.d = d
        self.j = j

cdef class KDJ:
    cdef:
        list arr
        int period
        KDJ_Item pre_kdj

    def __init__(self, int period=9):
        self.arr = []
        self.period = period
        self.pre_kdj = KDJ_Item(50.0, 50.0, 50.0)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef KDJ_Item add(self, double high, double low, double close):
        cdef:
            dict item
            double hn, ln, cn, rsv
            double cur_k, cur_d, cur_j
            KDJ_Item cur_kdj

        self.arr.append({
            'high': high,
            'low': low,
        })
        if len(self.arr) > self.period:
            self.arr.pop(0)

        hn = max([x['high'] for x in self.arr])
        ln = min([x['low'] for x in self.arr])
        cn = close
        rsv = 100.0 * (cn - ln) / (hn - ln) if hn != ln else 0.0

        cur_k = 2.0 / 3.0 * self.pre_kdj.k + 1.0 / 3.0 * rsv
        cur_d = 2.0 / 3.0 * self.pre_kdj.d + 1.0 / 3.0 * cur_k
        cur_j = 3.0 * cur_k - 2.0 * cur_d
        cur_kdj = KDJ_Item(cur_k, cur_d, cur_j)
        self.pre_kdj = cur_kdj

        return cur_kdj
