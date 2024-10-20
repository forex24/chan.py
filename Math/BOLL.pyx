# cython: language_level=3
import cython
from libc.math cimport sqrt

cdef double _truncate(double x):
    return x if x != 0 else 1e-7

cdef class BOLL_Metric:
    cdef public double theta, UP, DOWN, MID

    def __init__(self, double ma, double theta):
        self.theta = _truncate(theta)
        self.UP = ma + 2*theta
        self.DOWN = _truncate(ma - 2*theta)
        self.MID = ma

cdef class BollModel:
    cdef public int N
    cdef list arr

    def __init__(self, int N=20):
        assert N > 1
        self.N = N
        self.arr = []

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef BOLL_Metric add(self, double value):
        cdef int i
        cdef double ma, theta, diff
        cdef int arr_len

        self.arr.append(value)
        arr_len = len(self.arr)
        if arr_len > self.N:
            self.arr = self.arr[-self.N:]
            arr_len = self.N

        ma = sum(self.arr) / arr_len
        theta = 0.0
        for i in range(arr_len):
            diff = self.arr[i] - ma
            theta += diff * diff
        theta = sqrt(theta / arr_len)

        return BOLL_Metric(ma, theta)
