# cython: language_level=3
from libc.math cimport fabs
from Common.CEnum cimport BI_DIR, BI_TYPE, DATA_FIELD, FX_TYPE, MACD_ALGO
from KLine.KLine cimport CKLine
from KLine.KLine_Unit cimport CKLine_Unit

cdef class CBi:
    cdef:
        CKLine __begin_klc
        CKLine __end_klc
        BI_DIR __dir
        int __idx
        BI_TYPE __type
        bint __is_sure
        list __sure_end
        int __seg_idx
        public object parent_seg
        public object bsp
        public CBi next
        public CBi pre
        dict _memoize_cache

    cpdef void clean_cache(self)
    cpdef void check(self)
    cpdef void set(self, CKLine begin_klc, CKLine end_klc)
    cpdef double get_begin_val(self)
    cpdef double get_end_val(self)
    cpdef CKLine_Unit get_begin_klu(self)
    cpdef CKLine_Unit get_end_klu(self)
    cpdef double amp(self)
    cpdef int get_klu_cnt(self)
    cpdef int get_klc_cnt(self)
    cpdef double _high(self)
    cpdef double _low(self)
    cpdef double _mid(self)
    cpdef bint is_down(self)
    cpdef bint is_up(self)
    cpdef void update_virtual_end(self, CKLine new_klc)
    cpdef void restore_from_virtual_end(self, CKLine sure_end)
    cpdef void append_sure_end(self, CKLine klc)
    cpdef void update_new_end(self, CKLine new_klc)
    cpdef double cal_macd_metric(self, MACD_ALGO macd_algo, bint is_reverse)
    cpdef double Cal_Rsi(self)
    cpdef double Cal_MACD_area(self)
    cpdef double Cal_MACD_peak(self)
    cpdef double Cal_MACD_half(self, bint is_reverse)
    cpdef double Cal_MACD_half_obverse(self)
    cpdef double Cal_MACD_half_reverse(self)
    cpdef double Cal_MACD_diff(self)
    cpdef double Cal_MACD_slope(self)
    cpdef double Cal_MACD_amp(self)
    cpdef double Cal_MACD_trade_metric(self, str metric, bint cal_avg=*)
    cpdef void set_seg_idx(self, int idx)
