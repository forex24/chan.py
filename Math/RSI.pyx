# cython: language_level=3
import cython
from libc.stdlib cimport malloc, free
from libc.math cimport fmax

cdef class RSI:
    cdef:
        double* close_arr
        int close_arr_size
        int close_arr_capacity
        int period
        double* diff
        int diff_size
        double* up
        double* down
        int up_down_size

    def __cinit__(self, int period=14):
        self.period = period
        self.close_arr_capacity = 100  # Initial capacity
        self.close_arr = <double*>malloc(self.close_arr_capacity * sizeof(double))
        self.diff = <double*>malloc(self.close_arr_capacity * sizeof(double))
        self.up = <double*>malloc(self.close_arr_capacity * sizeof(double))
        self.down = <double*>malloc(self.close_arr_capacity * sizeof(double))
        
        if not self.close_arr or not self.diff or not self.up or not self.down:
            raise MemoryError()
        
        self.close_arr_size = 0
        self.diff_size = 0
        self.up_down_size = 0

    def __dealloc__(self):
        if self.close_arr is not NULL:
            free(self.close_arr)
        if self.diff is not NULL:
            free(self.diff)
        if self.up is not NULL:
            free(self.up)
        if self.down is not NULL:
            free(self.down)

    cdef void _resize_if_necessary(self):
        cdef:
            double* new_arr
            int new_capacity
        
        if self.close_arr_size == self.close_arr_capacity:
            new_capacity = self.close_arr_capacity * 2
            new_arr = <double*>malloc(new_capacity * sizeof(double))
            if not new_arr:
                raise MemoryError()

            for i in range(self.close_arr_size):
                new_arr[i] = self.close_arr[i]

            free(self.close_arr)
            self.close_arr = new_arr
            self.close_arr_capacity = new_capacity

            # Resize other arrays
            self.diff = <double*>realloc(self.diff, new_capacity * sizeof(double))
            self.up = <double*>realloc(self.up, new_capacity * sizeof(double))
            self.down = <double*>realloc(self.down, new_capacity * sizeof(double))

            if not self.diff or not self.up or not self.down:
                raise MemoryError()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double add(self, double close):
        cdef:
            double upval, downval, rs, rsi
            int i

        self._resize_if_necessary()

        self.close_arr[self.close_arr_size] = close
        self.close_arr_size += 1

        if self.close_arr_size == 1:
            return 50.0

        self.diff[self.diff_size] = self.close_arr[self.close_arr_size - 1] - self.close_arr[self.close_arr_size - 2]
        self.diff_size += 1

        if self.diff_size < self.period:
            upsum = 0.0
            downsum = 0.0
            for i in range(self.diff_size):
                if self.diff[i] > 0:
                    upsum += self.diff[i]
                else:
                    downsum -= self.diff[i]
            self.up[self.up_down_size] = upsum / self.period
            self.down[self.up_down_size] = downsum / self.period
        else:
            if self.diff[self.diff_size - 1] > 0:
                upval = self.diff[self.diff_size - 1]
                downval = 0.0
            else:
                upval = 0.0
                downval = -self.diff[self.diff_size - 1]
            self.up[self.up_down_size] = (self.up[self.up_down_size - 1] * (self.period - 1) + upval) / self.period
            self.down[self.up_down_size] = (self.down[self.up_down_size - 1] * (self.period - 1) + downval) / self.period

        self.up_down_size += 1

        rs = self.up[self.up_down_size - 1] / fmax(self.down[self.up_down_size - 1], 1e-7)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        return rsi
