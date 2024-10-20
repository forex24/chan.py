# cython: language_level=3
cimport cython
from Bi.Bi cimport CBi
from Combiner.KLine_Combiner cimport CKLine_Combiner
from Common.CEnum cimport BI_DIR, FX_TYPE

cdef class CEigen(CKLine_Combiner[CBi]):
    cdef:
        public bint gap

    def __cinit__(self, CBi bi, BI_DIR _dir):
        super().__init__(bi, _dir)
        self.gap = False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_fx(self, CEigen _pre, CEigen _next, bint exclude_included=False, object allow_top_equal=None):
        super().update_fx(_pre, _next, exclude_included, allow_top_equal)
        if (self.fx == FX_TYPE.TOP and _pre.high < self.low) or \
           (self.fx == FX_TYPE.BOTTOM and _pre.low > self.high):
            self.gap = True

    def __str__(self):
        return f"{self.lst[0].idx}~{self.lst[-1].idx} gap={self.gap} fx={self.fx}"

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int GetPeakBiIdx(self):
        assert self.fx != FX_TYPE.UNKNOWN
        cdef BI_DIR bi_dir = self.lst[0].dir
        if bi_dir == BI_DIR.UP:  # 下降线段
            return self.get_peak_klu(is_high=False).idx - 1
        else:
            return self.get_peak_klu(is_high=True).idx - 1
