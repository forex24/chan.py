# cython: language_level=3
cimport cython
from typing import Generic, List, TypeVar, Union, overload

from Bi.Bi cimport CBi
from Bi.BiList cimport CBiList
from Common.CEnum cimport BI_DIR, LEFT_SEG_METHOD, SEG_TYPE
from Common.ChanException cimport CChanException, ErrCode

from .Seg cimport CSeg
from .SegConfig cimport CSegConfig

SUB_LINE_TYPE = TypeVar('SUB_LINE_TYPE', CBi, "CSeg")

cdef class CSegListComm(Generic[SUB_LINE_TYPE]):
    cdef:
        public list lst
        public SEG_TYPE lv
        public CSegConfig config

    def __cinit__(self, CSegConfig seg_config=CSegConfig(), SEG_TYPE lv=SEG_TYPE.BI):
        self.lst = []
        self.lv = lv
        self.config = seg_config

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint left_bi_break(self, CBiList bi_lst):
        if not self.lst:
            return False
        cdef:
            CBi last_seg_end_bi = self.lst[-1].end_bi
            CBi bi
        return any((last_seg_end_bi.is_up() and bi._high() > last_seg_end_bi._high()) or
                   (last_seg_end_bi.is_down() and bi._low() < last_seg_end_bi._low())
                   for bi in bi_lst[last_seg_end_bi.idx+1:])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void collect_first_seg(self, CBiList bi_lst):
        if len(bi_lst) < 3:
            return
        if self.config.left_method == LEFT_SEG_METHOD.PEAK:
            self._collect_first_seg_peak(bi_lst)
        elif self.config.left_method == LEFT_SEG_METHOD.ALL:
            self._collect_first_seg_all(bi_lst)
        else:
            raise CChanException(f"unknown seg left_method = {self.config.left_method}", ErrCode.PARA_ERROR)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _collect_first_seg_peak(self, CBiList bi_lst):
        cdef:
            double _high = max(bi._high() for bi in bi_lst)
            double _low = min(bi._low() for bi in bi_lst)
            CBi peak_bi
        if abs(_high-bi_lst[0].get_begin_val()) >= abs(_low-bi_lst[0].get_begin_val()):
            peak_bi = FindPeakBi(bi_lst, is_high=True)
            assert peak_bi is not None
            self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.UP, split_first_seg=False, reason="0seg_find_high")
        else:
            peak_bi = FindPeakBi(bi_lst, is_high=False)
            assert peak_bi is not None
            self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.DOWN, split_first_seg=False, reason="0seg_find_low")
        self.collect_left_as_seg(bi_lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _collect_first_seg_all(self, CBiList bi_lst):
        cdef BI_DIR _dir = BI_DIR.UP if bi_lst[-1].get_end_val() >= bi_lst[0].get_begin_val() else BI_DIR.DOWN
        self.add_new_seg(bi_lst, bi_lst[-1].idx, is_sure=False, seg_dir=_dir, split_first_seg=False, reason="0seg_collect_all")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void collect_left_seg_peak_method(self, CBi last_seg_end_bi, CBiList bi_lst):
        cdef CBi peak_bi
        if last_seg_end_bi.is_down():
            peak_bi = FindPeakBi(bi_lst[last_seg_end_bi.idx+3:], is_high=True)
            if peak_bi and peak_bi.idx - last_seg_end_bi.idx >= 3:
                self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.UP, reason="collectleft_find_high")
        else:
            peak_bi = FindPeakBi(bi_lst[last_seg_end_bi.idx+3:], is_high=False)
            if peak_bi and peak_bi.idx - last_seg_end_bi.idx >= 3:
                self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.DOWN, reason="collectleft_find_low")
        last_seg_end_bi = self.lst[-1].end_bi

        self.collect_left_as_seg(bi_lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void collect_segs(self, CBiList bi_lst):
        cdef:
            CBi last_bi = bi_lst[-1]
            CBi last_seg_end_bi = self.lst[-1].end_bi
            CBi peak_bi
        if last_bi.idx-last_seg_end_bi.idx < 3:
            return
        if last_seg_end_bi.is_down() and last_bi.get_end_val() <= last_seg_end_bi.get_end_val():
            peak_bi = FindPeakBi(bi_lst[last_seg_end_bi.idx+3:], is_high=True)
            if peak_bi:
                self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.UP, reason="collectleft_find_high_force")
                self.collect_left_seg(bi_lst)
        elif last_seg_end_bi.is_up() and last_bi.get_end_val() >= last_seg_end_bi.get_end_val():
            peak_bi = FindPeakBi(bi_lst[last_seg_end_bi.idx+3:], is_high=False)
            if peak_bi:
                self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=BI_DIR.DOWN, reason="collectleft_find_low_force")
                self.collect_left_seg(bi_lst)
        elif self.config.left_method == LEFT_SEG_METHOD.ALL:
            self.collect_left_as_seg(bi_lst)
        elif self.config.left_method == LEFT_SEG_METHOD.PEAK:
            self.collect_left_seg_peak_method(last_seg_end_bi, bi_lst)
        else:
            raise CChanException(f"unknown seg left_method = {self.config.left_method}", ErrCode.PARA_ERROR)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void collect_left_seg(self, CBiList bi_lst):
        if len(self.lst) == 0:
            self.collect_first_seg(bi_lst)
        else:
            self.collect_segs(bi_lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void collect_left_as_seg(self, CBiList bi_lst):
        cdef:
            CBi last_bi = bi_lst[-1]
            CBi last_seg_end_bi = self.lst[-1].end_bi
        if last_seg_end_bi.idx+1 >= len(bi_lst):
            return
        if last_seg_end_bi.dir == last_bi.dir:
            self.add_new_seg(bi_lst, last_bi.idx-1, is_sure=False, reason="collect_left_1")
        else:
            self.add_new_seg(bi_lst, last_bi.idx, is_sure=False, reason="collect_left_0")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint try_add_new_seg(self, CBiList bi_lst, int end_bi_idx, bint is_sure=True, BI_DIR seg_dir=None, bint split_first_seg=True, str reason="normal"):
        cdef:
            CBi peak_bi
            int bi1_idx
            CBi bi1, bi2
        if len(self.lst) == 0 and split_first_seg and end_bi_idx >= 3:
            peak_bi = FindPeakBi(bi_lst[end_bi_idx-3::-1], bi_lst[end_bi_idx].is_down())
            if peak_bi:
                if (peak_bi.is_down() and (peak_bi._low() < bi_lst[0]._low() or peak_bi.idx == 0)) or \
                   (peak_bi.is_up() and (peak_bi._high() > bi_lst[0]._high() or peak_bi.idx == 0)):
                    self.add_new_seg(bi_lst, peak_bi.idx, is_sure=False, seg_dir=peak_bi.dir, reason="split_first_1st")
                    self.add_new_seg(bi_lst, end_bi_idx, is_sure=False, reason="split_first_2nd")
                    return True
        bi1_idx = 0 if len(self.lst) == 0 else self.lst[-1].end_bi.idx+1
        bi1 = bi_lst[bi1_idx]
        bi2 = bi_lst[end_bi_idx]
        self.lst.append(CSeg(len(self.lst), bi1, bi2, is_sure=is_sure, seg_dir=seg_dir, reason=reason))

        if len(self.lst) >= 2:
            self.lst[-2].next = self.lst[-1]
            self.lst[-1].pre = self.lst[-2]
        self.lst[-1].update_bi_list(bi_lst, bi1_idx, end_bi_idx)
        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint add_new_seg(self, CBiList bi_lst, int end_bi_idx, bint is_sure=True, BI_DIR seg_dir=None, bint split_first_seg=True, str reason="normal"):
        try:
            return self.try_add_new_seg(bi_lst, end_bi_idx, is_sure, seg_dir, split_first_seg, reason)
        except CChanException as e:
            if e.errcode == ErrCode.SEG_END_VALUE_ERR and len(self.lst) == 0:
                return False
            raise e
        except Exception as e:
            raise e

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update(self, CBiList bi_lst):
        raise NotImplementedError("Subclass must implement abstract method")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint exist_sure_seg(self):
        return any(seg.is_sure for seg in self.lst)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef CBi FindPeakBi(Union[CBiList, List[CBi]] bi_lst, bint is_high):
    cdef:
        double peak_val = float("-inf") if is_high else float("inf")
        CBi peak_bi = None
        CBi bi
    for bi in bi_lst:
        if (is_high and bi.get_end_val() >= peak_val and bi.is_up()) or (not is_high and bi.get_end_val() <= peak_val and bi.is_down()):
            if bi.pre and bi.pre.pre and ((is_high and bi.pre.pre.get_end_val() > bi.get_end_val()) or (not is_high and bi.pre.pre.get_end_val() < bi.get_end_val())):
                continue
            peak_val = bi.get_end_val()
            peak_bi = bi
    return peak_bi
