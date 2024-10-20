# cython: language_level=3
import cython
from libc.stdlib cimport malloc, free
from libc.math cimport fmax, fmin

from Common.CEnum import TREND_TYPE
from Common.ChanException import CChanException, ErrCode

cdef class CTrendModel:
    cdef:
        int T
        double* arr
        int arr_size
        int arr_capacity
        TREND_TYPE trend_type

    def __cinit__(self, TREND_TYPE trend_type, int T):
        self.T = T
        self.trend_type = trend_type
        self.arr_capacity = T
        self.arr = <double*>malloc(self.arr_capacity * sizeof(double))
        if not self.arr:
            raise MemoryError()
        self.arr_size = 0

    def __dealloc__(self):
        if self.arr is not NULL:
            free(self.arr)

    cdef void _resize_if_necessary(self):
        cdef:
            double* new_arr
            int new_capacity
            int i

        if self.arr_size == self.arr_capacity:
            new_capacity = self.arr_capacity * 2
            new_arr = <double*>malloc(new_capacity * sizeof(double))
            if not new_arr:
                raise MemoryError()

            for i in range(self.arr_size):
                new_arr[i] = self.arr[i]

            free(self.arr)
            self.arr = new_arr
            self.arr_capacity = new_capacity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double add(self, double value):
        cdef:
            int i
            double result

        self._resize_if_necessary()
        self.arr[self.arr_size] = value
        self.arr_size += 1

        if self.arr_size > self.T:
            for i in range(self.T - 1):
                self.arr[i] = self.arr[i + 1]
            self.arr_size = self.T

        if self.trend_type == TREND_TYPE.MEAN:
            result = 0
            for i in range(self.arr_size):
                result += self.arr[i]
            return result / self.arr_size
        elif self.trend_type == TREND_TYPE.MAX:
            result = self.arr[0]
            for i in range(1, self.arr_size):
                result = fmax(result, self.arr[i])
            return result
        elif self.trend_type == TREND_TYPE.MIN:
            result = self.arr[0]
            for i in range(1, self.arr_size):
                result = fmin(result, self.arr[i])
            return result
        else:
            raise CChanException(f"Unknown trendModel Type = {self.trend_type}", ErrCode.PARA_ERROR)
