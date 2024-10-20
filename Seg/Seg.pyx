# cython: language_level=3
cimport cython
from libc.stdlib cimport malloc, free
from typing import Generic, List, Optional, TypeVar

from Bi.Bi cimport CBi
from Common.CEnum cimport BI_DIR, MACD_ALGO, TREND_LINE_SIDE
from Common.ChanException cimport CChanException, ErrCode
from KLine.KLine_Unit cimport CKLine_Unit
from Math.TrendLine cimport CTrendLine

from .EigenFX cimport CEigenFX

LINE_TYPE = TypeVar('LINE_TYPE', CBi, "CSeg")

cdef class CSeg(Generic[LINE_TYPE]):
    cdef:
        public int idx
        public LINE_TYPE start_bi
        public LINE_TYPE end_bi
        public bint is_sure
        public BI_DIR dir
        public list zs_lst
        public CEigenFX eigen_fx
        public int seg_idx
        public CSeg parent_seg
        public CSeg pre
        public CSeg next
        public object bsp
        public list bi_list
        public str reason
        public CTrendLine support_trend_line
        public CTrendLine resistance_trend_line
        public bint ele_inside_is_sure

    def __cinit__(self, int idx, LINE_TYPE start_bi, LINE_TYPE end_bi, bint is_sure=True, BI_DIR seg_dir=None, str reason="normal"):
        assert start_bi.idx == 0 or start_bi.dir == end_bi.dir or not is_sure, f"{start_bi.idx} {end_bi.idx} {start_bi.dir} {end_bi.dir}"
        self.idx = idx
        self.start_bi = start_bi
        self.end_bi = end_bi
        self.is_sure = is_sure
        self.dir = end_bi.dir if seg_dir is None else seg_dir

        from ZS.ZS import CZS
        self.zs_lst = []

        self.eigen_fx = None
        self.seg_idx = None  # 线段的线段用
        self.parent_seg = None  # 在哪个线段里面
        self.pre = None
        self.next = None

        from BuySellPoint.BS_Point import CBS_Point
        self.bsp = None  # 尾部是不是买卖点

        self.bi_list = []  # 仅通过self.update_bi_list来更新
        self.reason = reason
        self.support_trend_line = None
        self.resistance_trend_line = None
        if end_bi.idx - start_bi.idx < 2:
            self.is_sure = False
        self.check()

        self.ele_inside_is_sure = False

    cpdef void set_seg_idx(self, int idx):
        self.seg_idx = idx

    cpdef void check(self):
        if not self.is_sure:
            return
        if self.is_down():
            if self.start_bi.get_begin_val() < self.end_bi.get_end_val():
                raise CChanException(f"下降线段起始点应该高于结束点! idx={self.idx}", ErrCode.SEG_END_VALUE_ERR)
        elif self.start_bi.get_begin_val() > self.end_bi.get_end_val():
            raise CChanException(f"上升线段起始点应该低于结束点! idx={self.idx}", ErrCode.SEG_END_VALUE_ERR)
        if self.end_bi.idx - self.start_bi.idx < 2:
            raise CChanException(f"线段({self.start_bi.idx}-{self.end_bi.idx})长度不能小于2! idx={self.idx}", ErrCode.SEG_LEN_ERR)

    def __str__(self):
        return f"{self.start_bi.idx}->{self.end_bi.idx}: {self.dir}  {self.is_sure}"

    cpdef void add_zs(self, object zs):
        self.zs_lst = [zs] + self.zs_lst  # 因为中枢是反序加入的

    @cython.cdivision(True)
    cpdef double cal_klu_slope(self):
        assert self.end_bi.idx >= self.start_bi.idx
        return (self.get_end_val()-self.get_begin_val())/(self.get_end_klu().idx-self.get_begin_klu().idx)/self.get_begin_val()

    @cython.cdivision(True)
    cpdef double cal_amp(self):
        return (self.get_end_val()-self.get_begin_val())/self.get_begin_val()

    cpdef int cal_bi_cnt(self):
        return self.end_bi.idx-self.start_bi.idx+1

    cpdef void clear_zs_lst(self):
        self.zs_lst = []

    cpdef double _low(self):
        return self.end_bi.get_end_klu().low if self.is_down() else self.start_bi.get_begin_klu().low

    cpdef double _high(self):
        return self.end_bi.get_end_klu().high if self.is_up() else self.start_bi.get_begin_klu().high

    cpdef bint is_down(self):
        return self.dir == BI_DIR.DOWN

    cpdef bint is_up(self):
        return self.dir == BI_DIR.UP

    cpdef double get_end_val(self):
        return self.end_bi.get_end_val()

    cpdef double get_begin_val(self):
        return self.start_bi.get_begin_val()

    cpdef double amp(self):
        return abs(self.get_end_val() - self.get_begin_val())

    cpdef CKLine_Unit get_end_klu(self):
        return self.end_bi.get_end_klu()

    cpdef CKLine_Unit get_begin_klu(self):
        return self.start_bi.get_begin_klu()

    cpdef int get_klu_cnt(self):
        return self.get_end_klu().idx - self.get_begin_klu().idx + 1

    cpdef double cal_macd_metric(self, MACD_ALGO macd_algo, bint is_reverse):
        if macd_algo == MACD_ALGO.SLOPE:
            return self.Cal_MACD_slope()
        elif macd_algo == MACD_ALGO.AMP:
            return self.Cal_MACD_amp()
        else:
            raise CChanException(f"unsupport macd_algo={macd_algo} of Seg, should be one of slope/amp", ErrCode.PARA_ERROR)

    @cython.cdivision(True)
    cpdef double Cal_MACD_slope(self):
        cdef:
            CKLine_Unit begin_klu = self.get_begin_klu()
            CKLine_Unit end_klu = self.get_end_klu()
        if self.is_up():
            return (end_klu.high - begin_klu.low)/end_klu.high/(end_klu.idx - begin_klu.idx + 1)
        else:
            return (begin_klu.high - end_klu.low)/begin_klu.high/(end_klu.idx - begin_klu.idx + 1)

    @cython.cdivision(True)
    cpdef double Cal_MACD_amp(self):
        cdef:
            CKLine_Unit begin_klu = self.get_begin_klu()
            CKLine_Unit end_klu = self.get_end_klu()
        if self.is_down():
            return (begin_klu.high-end_klu.low)/begin_klu.high
        else:
            return (end_klu.high-begin_klu.low)/begin_klu.low

    cpdef void update_bi_list(self, list bi_lst, int idx1, int idx2):
        cdef:
            int bi_idx
            LINE_TYPE bi
        for bi_idx in range(idx1, idx2+1):
            bi = bi_lst[bi_idx]
            bi.parent_seg = self
            self.bi_list.append(bi)
        if len(self.bi_list) >= 3:
            self.support_trend_line = CTrendLine(self.bi_list, TREND_LINE_SIDE.INSIDE)
            self.resistance_trend_line = CTrendLine(self.bi_list, TREND_LINE_SIDE.OUTSIDE)

    cpdef object get_first_multi_bi_zs(self):
        return next((zs for zs in self.zs_lst if not zs.is_one_bi_zs()), None)

    cpdef object get_final_multi_bi_zs(self):
        return next((zs for zs in self.zs_lst[::-1] if not zs.is_one_bi_zs()), None)

    cpdef int get_multi_bi_zs_cnt(self):
        return sum(not zs.is_one_bi_zs() for zs in self.zs_lst)
