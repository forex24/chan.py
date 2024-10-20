# cython: language_level=3
from libc.stdlib cimport malloc, free
from typing import List, Optional, Union, overload

from Common.CEnum cimport FX_TYPE, KLINE_DIR
from KLine.KLine cimport CKLine

from .Bi cimport CBi
from .BiConfig cimport CBiConfig

cdef class CBiList:
    cdef:
        public list bi_list
        public CKLine last_end
        public CBiConfig config
        public list free_klc_lst

    cpdef bint try_create_first_bi(self, CKLine klc)
    cpdef bint update_bi(self, CKLine klc, CKLine last_klc, bint cal_virtual)
    cpdef bint can_update_peak(self, CKLine klc)
    cpdef bint update_peak(self, CKLine klc, bint for_virtual=*)
    cpdef bint update_bi_sure(self, CKLine klc)
    cpdef void delete_virtual_bi(self)
    cpdef bint try_add_virtual_bi(self, CKLine klc, bint need_del_end=*)
    cpdef void add_new_bi(self, CKLine pre_klc, CKLine cur_klc, bint is_sure=*)
    cpdef bint satisfy_bi_span(self, CKLine klc, CKLine last_end)
    cpdef int get_klc_span(self, CKLine klc, CKLine last_end)
    cpdef bint can_make_bi(self, CKLine klc, CKLine last_end, bint for_virtual=*)
    cpdef bint try_update_end(self, CKLine klc, bint for_virtual=*)
    cpdef CKLine get_last_klu_of_last_bi(self)

cdef bint check_top(CKLine klc, bint for_virtual)
cdef bint check_bottom(CKLine klc, bint for_virtual)
cdef bint end_is_peak(CKLine last_end, CKLine cur_end)
