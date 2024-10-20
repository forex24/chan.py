# cython: language_level=3
cimport cython
from typing import List

from Bi.BiList cimport CBiList
from Common.CEnum cimport BI_DIR, SEG_TYPE
from .EigenFX cimport CEigenFX
from .SegConfig cimport CSegConfig
from .SegListComm cimport CSegListComm
from .Seg cimport CSeg

cdef class CSegListChan(CSegListComm):
    cdef:
        CEigenFX up_eigen
        CEigenFX down_eigen

    def __cinit__(self, CSegConfig seg_config=CSegConfig(), SEG_TYPE lv=SEG_TYPE.BI):
        super().__init__(seg_config=seg_config, lv=lv)
        self.up_eigen = CEigenFX(BI_DIR.UP, lv=self.lv)
        self.down_eigen = CEigenFX(BI_DIR.DOWN, lv=self.lv)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void do_init(self):
        while self.lst and not self.lst[-1].is_sure:
            self._remove_last_seg()
        if self.lst and self.lst[-1].eigen_fx and self.lst[-1].eigen_fx.ele[-1] and not self.lst[-1].eigen_fx.ele[-1].lst[-1].is_sure:
            self._remove_last_seg()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update(self, CBiList bi_lst):
        self.do_init()
        cdef int start_idx = 0 if not self.lst else self.lst[-1].end_bi.idx + 1
        self.cal_seg_sure(bi_lst, start_idx)
        self.collect_left_seg(bi_lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void cal_seg_sure(self, CBiList bi_lst, int begin_idx):
        cdef:
            BI_DIR last_seg_dir = self.lst[-1].dir if self.lst else None
            CEigenFX fx_eigen
            object bi
        
        for bi in bi_lst[begin_idx:]:
            fx_eigen = self._process_bi(bi, last_seg_dir)
            
            if not self.lst:
                last_seg_dir = self._determine_first_seg_dir(bi)
            
            if fx_eigen and self._treat_fx_eigen(fx_eigen, bi_lst):
                return

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef CEigenFX _process_bi(self, object bi, BI_DIR last_seg_dir):
        if bi.is_down() and last_seg_dir != BI_DIR.UP and self.up_eigen.add(bi):
            return self.up_eigen
        if bi.is_up() and last_seg_dir != BI_DIR.DOWN and self.down_eigen.add(bi):
            return self.down_eigen
        return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef BI_DIR _determine_first_seg_dir(self, object bi):
        if self.up_eigen.ele[1] is not None and bi.is_down():
            self.down_eigen.clear()
            return BI_DIR.DOWN
        if self.down_eigen.ele[1] is not None and bi.is_up():
            self.up_eigen.clear()
            return BI_DIR.UP
        return None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef bint _treat_fx_eigen(self, CEigenFX fx_eigen, CBiList bi_lst):
        cdef:
            object end_status
            int end_bi_idx
            bint is_sure
        
        end_status = fx_eigen.can_be_end(bi_lst)
        end_bi_idx = fx_eigen.GetPeakBiIdx()
        
        if end_status in (True, None):
            is_sure = end_status is not None and fx_eigen.all_bi_is_sure()
            if self.add_new_seg(bi_lst, end_bi_idx, is_sure=is_sure):
                self.lst[-1].eigen_fx = fx_eigen
                if is_sure:
                    self.cal_seg_sure(bi_lst, end_bi_idx + 1)
                return True
            self.cal_seg_sure(bi_lst, end_bi_idx + 1)
        else:
            self.cal_seg_sure(bi_lst, fx_eigen.lst[1].idx)
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _remove_last_seg(self):
        cdef:
            CSeg last_seg
            object bi
        last_seg = self.lst.pop()
        for bi in last_seg.bi_list:
            bi.parent_seg = None
        if last_seg.pre:
            last_seg.pre.next = None
