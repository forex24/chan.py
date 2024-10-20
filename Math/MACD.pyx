# cython: language_level=3
import cython
from libc.stdlib cimport malloc, free

cdef struct CMACD_item:
    double fast_ema
    double slow_ema
    double DIF
    double DEA
    double macd

cdef class CMACD:
    cdef:
        CMACD_item* macd_info
        int macd_info_size
        int macd_info_capacity
        int fastperiod
        int slowperiod
        int signalperiod

    def __cinit__(self, int fastperiod=12, int slowperiod=26, int signalperiod=9):
        self.macd_info = NULL
        self.macd_info_size = 0
        self.macd_info_capacity = 10  # Initial capacity
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod

        self.macd_info = <CMACD_item*>malloc(self.macd_info_capacity * sizeof(CMACD_item))
        if not self.macd_info:
            raise MemoryError()

    def __dealloc__(self):
        if self.macd_info is not NULL:
            free(self.macd_info)

    cdef void _resize_if_necessary(self):
        cdef:
            CMACD_item* new_macd_info
            int new_capacity

        if self.macd_info_size == self.macd_info_capacity:
            new_capacity = self.macd_info_capacity * 2
            new_macd_info = <CMACD_item*>malloc(new_capacity * sizeof(CMACD_item))
            if not new_macd_info:
                raise MemoryError()

            for i in range(self.macd_info_size):
                new_macd_info[i] = self.macd_info[i]

            free(self.macd_info)
            self.macd_info = new_macd_info
            self.macd_info_capacity = new_capacity

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef CMACD_item add(self, double value):
        cdef:
            double _fast_ema, _slow_ema, _dif, _dea
            CMACD_item new_item

        self._resize_if_necessary()

        if self.macd_info_size == 0:
            new_item = CMACD_item(
                fast_ema=value,
                slow_ema=value,
                DIF=0,
                DEA=0,
                macd=0
            )
        else:
            _fast_ema = (2 * value + (self.fastperiod - 1) * self.macd_info[self.macd_info_size - 1].fast_ema) / (self.fastperiod + 1)
            _slow_ema = (2 * value + (self.slowperiod - 1) * self.macd_info[self.macd_info_size - 1].slow_ema) / (self.slowperiod + 1)
            _dif = _fast_ema - _slow_ema
            _dea = (2 * _dif + (self.signalperiod - 1) * self.macd_info[self.macd_info_size - 1].DEA) / (self.signalperiod + 1)
            new_item = CMACD_item(
                fast_ema=_fast_ema,
                slow_ema=_slow_ema,
                DIF=_dif,
                DEA=_dea,
                macd=2 * (_dif - _dea)
            )

        self.macd_info[self.macd_info_size] = new_item
        self.macd_info_size += 1

        return new_item

    def __getitem__(self, int index):
        if index < 0:
            index += self.macd_info_size
        if index < 0 or index >= self.macd_info_size:
            raise IndexError("Index out of range")
        return self.macd_info[index]

    def __len__(self):
        return self.macd_info_size

    @property
    def data(self):
        return [self.macd_info[i] for i in range(self.macd_info_size)]
