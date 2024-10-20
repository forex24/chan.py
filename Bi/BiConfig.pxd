# cython: language_level=3
from Common.CEnum cimport FX_CHECK_METHOD

cdef class CBiConfig:
    cdef:
        public str bi_algo
        public bint is_strict
        public FX_CHECK_METHOD bi_fx_check
        public bint gap_as_kl
        public bint bi_end_is_peak
        public bint bi_allow_sub_peak

    cpdef CBiConfig copy(self)
