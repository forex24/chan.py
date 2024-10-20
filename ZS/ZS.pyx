# cython: language_level=3
cimport cython
from typing import Generic, List, Optional, TypeVar

from Bi.Bi cimport CBi
from BuySellPoint.BSPointConfig cimport CPointConfig
from Common.ChanException cimport CChanException, ErrCode
from Common.func_util cimport has_overlap
from KLine.KLine_Unit cimport CKLine_Unit
from Seg.Seg cimport CSeg

LINE_TYPE = TypeVar('LINE_TYPE', CBi, CSeg)

cdef class CZS(Generic[LINE_TYPE]):
    cdef:
        public bint __is_sure
        public list __sub_zs_lst
        public CKLine_Unit __begin
        public LINE_TYPE __begin_bi
        public double __low
        public double __high
        public double __mid
        public CKLine_Unit __end
        public LINE_TYPE __end_bi
        public double __peak_high
        public double __peak_low
        public LINE_TYPE __bi_in
        public LINE_TYPE __bi_out
        public list __bi_lst
        dict _memoize_cache

    def __cinit__(self, list lst=None, bint is_sure=True):
        self.__is_sure = is_sure
        self.__sub_zs_lst = []
        self._memoize_cache = {}

        if lst is None:
            return

        self.__begin = lst[0].get_begin_klu()
        self.__begin_bi = lst[0]

        self.update_zs_range(lst)

        self.__peak_high = float("-inf")
        self.__peak_low = float("inf")
        for item in lst:
            self.update_zs_end(item)

        self.__bi_in = None
        self.__bi_out = None
        self.__bi_lst = []

    cpdef void clean_cache(self):
        self._memoize_cache = {}

    @property
    def is_sure(self): return self.__is_sure

    @property
    def sub_zs_lst(self): return self.__sub_zs_lst

    @property
    def begin(self): return self.__begin

    @property
    def begin_bi(self): return self.__begin_bi

    @property
    def low(self): return self.__low

    @property
    def high(self): return self.__high

    @property
    def mid(self): return self.__mid

    @property
    def end(self): return self.__end

    @property
    def end_bi(self): return self.__end_bi

    @property
    def peak_high(self): return self.__peak_high

    @property
    def peak_low(self): return self.__peak_low

    @property
    def bi_in(self): return self.__bi_in

    @property
    def bi_out(self): return self.__bi_out

    @property
    def bi_lst(self): return self.__bi_lst

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_zs_range(self, list lst):
        self.__low = max(bi._low() for bi in lst)
        self.__high = min(bi._high() for bi in lst)
        self.__mid = (self.__low + self.__high) / 2
        self.clean_cache()

    cpdef bint is_one_bi_zs(self):
        assert self.end_bi is not None
        return self.begin_bi.idx == self.end_bi.idx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_zs_end(self, LINE_TYPE item):
        self.__end = item.get_end_klu()
        self.__end_bi = item
        if item._low() < self.peak_low:
            self.__peak_low = item._low()
        if item._high() > self.peak_high:
            self.__peak_high = item._high()
        self.clean_cache()

    def __str__(self):
        cdef str _str = f"{self.begin_bi.idx}->{self.end_bi.idx}"
        cdef str _str2 = ",".join([str(sub_zs) for sub_zs in self.sub_zs_lst])
        return f"{_str}({_str2})" if _str2 else _str

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint combine(self, CZS zs2, str combine_mode):
        if zs2.is_one_bi_zs():
            return False
        if self.begin_bi.seg_idx != zs2.begin_bi.seg_idx:
            return False
        if combine_mode == 'zs':
            if not has_overlap(self.low, self.high, zs2.low, zs2.high, equal=True):
                return False
            self.do_combine(zs2)
            return True
        elif combine_mode == 'peak':
            if has_overlap(self.peak_low, self.peak_high, zs2.peak_low, zs2.peak_high):
                self.do_combine(zs2)
                return True
            else:
                return False
        else:
            raise CChanException(f"{combine_mode} is unsupport zs conbine mode", ErrCode.PARA_ERROR)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void do_combine(self, CZS zs2):
        if len(self.sub_zs_lst) == 0:
            self.__sub_zs_lst.append(self.make_copy())
        self.__sub_zs_lst.append(zs2)

        self.__low = min([self.low, zs2.low])
        self.__high = max([self.high, zs2.high])
        self.__peak_low = min([self.peak_low, zs2.peak_low])
        self.__peak_high = max([self.peak_high, zs2.peak_high])
        self.__end = zs2.end
        self.__bi_out = zs2.bi_out
        self.__end_bi = zs2.end_bi
        self.clean_cache()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint try_add_to_end(self, LINE_TYPE item):
        if not self.in_range(item):
            return False
        if self.is_one_bi_zs():
            self.update_zs_range([self.begin_bi, item])
        self.update_zs_end(item)
        return True

    cpdef bint in_range(self, LINE_TYPE item):
        return has_overlap(self.low, self.high, item._low(), item._high())

    cpdef bint is_inside(self, CSeg seg):
        return seg.start_bi.idx <= self.begin_bi.idx <= seg.end_bi.idx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple is_divergence(self, CPointConfig config, LINE_TYPE out_bi=None):
        if not self.end_bi_break(out_bi):
            return False, None
        cdef double in_metric = self.get_bi_in().cal_macd_metric(config.macd_algo, is_reverse=False)
        cdef double out_metric
        if out_bi is None:
            out_metric = self.get_bi_out().cal_macd_metric(config.macd_algo, is_reverse=True)
        else:
            out_metric = out_bi.cal_macd_metric(config.macd_algo, is_reverse=True)

        if config.divergence_rate > 100:
            return True, out_metric/in_metric
        else:
            return out_metric <= config.divergence_rate*in_metric, out_metric/in_metric

    cpdef void init_from_zs(self, CZS zs):
        self.__begin = zs.begin
        self.__end = zs.end
        self.__low = zs.low
        self.__high = zs.high
        self.__peak_high = zs.peak_high
        self.__peak_low = zs.peak_low
        self.__begin_bi = zs.begin_bi
        self.__end_bi = zs.end_bi
        self.__bi_in = zs.bi_in
        self.__bi_out = zs.bi_out

    cpdef CZS make_copy(self):
        cdef CZS copy = CZS(lst=None, is_sure=self.is_sure)
        copy.init_from_zs(zs=self)
        return copy

    cpdef bint end_bi_break(self, LINE_TYPE end_bi=None):
        if end_bi is None:
            end_bi = self.get_bi_out()
        assert end_bi is not None
        return (end_bi.is_down() and end_bi._low() < self.low) or \
            (end_bi.is_up() and end_bi._high() > self.high)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple out_bi_is_peak(self, int end_bi_idx):
        assert len(self.bi_lst) > 0
        if self.bi_out is None:
            return False, None
        cdef double peak_rate = float("inf")
        cdef double r
        for bi in self.bi_lst:
            if bi.idx > end_bi_idx:
                break
            if (self.bi_out.is_down() and bi._low() < self.bi_out._low()) or (self.bi_out.is_up() and bi._high() > self.bi_out._high()):
                return False, None
            r = abs(bi.get_end_val()-self.bi_out.get_end_val())/self.bi_out.get_end_val()
            if r < peak_rate:
                peak_rate = r
        return True, peak_rate

    cpdef LINE_TYPE get_bi_in(self):
        assert self.bi_in is not None
        return self.bi_in

    cpdef LINE_TYPE get_bi_out(self):
        assert self.__bi_out is not None
        return self.__bi_out

    cpdef void set_bi_in(self, LINE_TYPE bi):
        self.__bi_in = bi
        self.clean_cache()

    cpdef void set_bi_out(self, LINE_TYPE bi):
        self.__bi_out = bi
        self.clean_cache()

    cpdef void set_bi_lst(self, list bi_lst):
        self.__bi_lst = bi_lst
        self.clean_cache()
