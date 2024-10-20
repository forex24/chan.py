# cython: language_level=3
from Common.CEnum cimport LEFT_SEG_METHOD
from Common.ChanException cimport CChanException, ErrCode

cdef class CSegConfig:
    cdef:
        public str seg_algo
        public LEFT_SEG_METHOD left_method

    def __cinit__(self, str seg_algo="chan", str left_method="peak"):
        self.seg_algo = seg_algo
        if left_method == "all":
            self.left_method = LEFT_SEG_METHOD.ALL
        elif left_method == "peak":
            self.left_method = LEFT_SEG_METHOD.PEAK
        else:
            raise CChanException(f"unknown left_seg_method={left_method}", ErrCode.PARA_ERROR)

    def __str__(self):
        return f"CSegConfig(seg_algo={self.seg_algo}, left_method={self.left_method})"

    def __repr__(self):
        return self.__str__()

    cpdef CSegConfig copy(self):
        return CSegConfig(
            seg_algo=self.seg_algo,
            left_method="all" if self.left_method == LEFT_SEG_METHOD.ALL else "peak"
        )
