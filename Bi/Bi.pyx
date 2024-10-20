# cython: language_level=3
cimport cython
from libc.math cimport fabs
from typing import List, Optional

from Common.cache cimport make_cache
from Common.CEnum cimport BI_DIR, BI_TYPE, DATA_FIELD, FX_TYPE, MACD_ALGO
from Common.ChanException cimport CChanException, ErrCode
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

    def __cinit__(self, CKLine begin_klc, CKLine end_klc, int idx, bint is_sure):
        self.__dir = None
        self.__idx = idx
        self.__type = BI_TYPE.STRICT

        self.set(begin_klc, end_klc)

        self.__is_sure = is_sure
        self.__sure_end = []

        self.__seg_idx = -1

        self.parent_seg = None  # 在哪个线段里面
        self.bsp = None  # 尾部是不是买卖点

        self.next = None
        self.pre = None

        self._memoize_cache = {}

    cpdef void clean_cache(self):
        self._memoize_cache = {}

    @property
    def begin_klc(self): return self.__begin_klc

    @property
    def end_klc(self): return self.__end_klc

    @property
    def dir(self): return self.__dir

    @property
    def idx(self): return self.__idx

    @property
    def type(self): return self.__type

    @property
    def is_sure(self): return self.__is_sure

    @property
    def sure_end(self): return self.__sure_end

    @property
    def klc_lst(self):
        cdef CKLine klc = self.begin_klc
        while True:
            yield klc
            klc = klc.next
            if klc is None or klc.idx > self.end_klc.idx:
                break

    @property
    def klc_lst_re(self):
        cdef CKLine klc = self.end_klc
        while True:
            yield klc
            klc = klc.pre
            if klc is None or klc.idx < self.begin_klc.idx:
                break

    @property
    def seg_idx(self): return self.__seg_idx

    cpdef void set_seg_idx(self, int idx):
        self.__seg_idx = idx

    def __str__(self):
        return f"{self.dir}|{self.begin_klc} ~ {self.end_klc}"

    cpdef void check(self):
        try:
            if self.is_down():
                assert self.begin_klc.high > self.end_klc.low
            else:
                assert self.begin_klc.low < self.end_klc.high
        except Exception as e:
            raise CChanException(f"{self.idx}:{self.begin_klc[0].time}~{self.end_klc[-1].time}笔的方向和收尾位置不一致!", ErrCode.BI_ERR) from e

    cpdef void set(self, CKLine begin_klc, CKLine end_klc):
        self.__begin_klc = begin_klc
        self.__end_klc = end_klc
        if begin_klc.fx == FX_TYPE.BOTTOM:
            self.__dir = BI_DIR.UP
        elif begin_klc.fx == FX_TYPE.TOP:
            self.__dir = BI_DIR.DOWN
        else:
            raise CChanException("ERROR DIRECTION when creating bi", ErrCode.BI_ERR)
        self.check()
        self.clean_cache()

    @make_cache
    cpdef double get_begin_val(self):
        return self.begin_klc.low if self.is_up() else self.begin_klc.high

    @make_cache
    cpdef double get_end_val(self):
        return self.end_klc.high if self.is_up() else self.end_klc.low

    @make_cache
    cpdef CKLine_Unit get_begin_klu(self):
        if self.is_up():
            return self.begin_klc.get_peak_klu(is_high=False)
        else:
            return self.begin_klc.get_peak_klu(is_high=True)

    @make_cache
    cpdef CKLine_Unit get_end_klu(self):
        if self.is_up():
            return self.end_klc.get_peak_klu(is_high=True)
        else:
            return self.end_klc.get_peak_klu(is_high=False)

    @make_cache
    cpdef double amp(self):
        return fabs(self.get_end_val() - self.get_begin_val())

    @make_cache
    cpdef int get_klu_cnt(self):
        return self.get_end_klu().idx - self.get_begin_klu().idx + 1

    @make_cache
    cpdef int get_klc_cnt(self):
        assert self.end_klc.idx == self.get_end_klu().klc.idx
        assert self.begin_klc.idx == self.get_begin_klu().klc.idx
        return self.end_klc.idx - self.begin_klc.idx + 1

    @make_cache
    cpdef double _high(self):
        return self.end_klc.high if self.is_up() else self.begin_klc.high

    @make_cache
    cpdef double _low(self):
        return self.begin_klc.low if self.is_up() else self.end_klc.low

    @make_cache
    cpdef double _mid(self):
        return (self._high() + self._low()) / 2  # 笔的中位价

    @make_cache
    cpdef bint is_down(self):
        return self.dir == BI_DIR.DOWN

    @make_cache
    cpdef bint is_up(self):
        return self.dir == BI_DIR.UP

    cpdef void update_virtual_end(self, CKLine new_klc):
        self.append_sure_end(self.end_klc)
        self.update_new_end(new_klc)
        self.__is_sure = False

    cpdef void restore_from_virtual_end(self, CKLine sure_end):
        self.__is_sure = True
        self.update_new_end(new_klc=sure_end)
        self.__sure_end = []

    cpdef void append_sure_end(self, CKLine klc):
        self.__sure_end.append(klc)

    cpdef void update_new_end(self, CKLine new_klc):
        self.__end_klc = new_klc
        self.check()
        self.clean_cache()

    cpdef double cal_macd_metric(self, MACD_ALGO macd_algo, bint is_reverse):
        if macd_algo == MACD_ALGO.AREA:
            return self.Cal_MACD_half(is_reverse)
        elif macd_algo == MACD_ALGO.PEAK:
            return self.Cal_MACD_peak()
        elif macd_algo == MACD_ALGO.FULL_AREA:
            return self.Cal_MACD_area()
        elif macd_algo == MACD_ALGO.DIFF:
            return self.Cal_MACD_diff()
        elif macd_algo == MACD_ALGO.SLOPE:
            return self.Cal_MACD_slope()
        elif macd_algo == MACD_ALGO.AMP:
            return self.Cal_MACD_amp()
        elif macd_algo == MACD_ALGO.AMOUNT:
            return self.Cal_MACD_trade_metric(DATA_FIELD.FIELD_TURNOVER, cal_avg=False)
        elif macd_algo == MACD_ALGO.VOLUMN:
            return self.Cal_MACD_trade_metric(DATA_FIELD.FIELD_VOLUME, cal_avg=False)
        elif macd_algo == MACD_ALGO.VOLUMN_AVG:
            return self.Cal_MACD_trade_metric(DATA_FIELD.FIELD_VOLUME, cal_avg=True)
        elif macd_algo == MACD_ALGO.AMOUNT_AVG:
            return self.Cal_MACD_trade_metric(DATA_FIELD.FIELD_TURNOVER, cal_avg=True)
        elif macd_algo == MACD_ALGO.TURNRATE_AVG:
            return self.Cal_MACD_trade_metric(DATA_FIELD.FIELD_TURNRATE, cal_avg=True)
        elif macd_algo == MACD_ALGO.RSI:
            return self.Cal_Rsi()
        else:
            raise CChanException(f"unsupport macd_algo={macd_algo}, should be one of area/full_area/peak/diff/slope/amp", ErrCode.PARA_ERROR)

    @make_cache
    cpdef double Cal_Rsi(self):
        cdef:
            list rsi_lst = []
            CKLine klc
            CKLine_Unit klu
        for klc in self.klc_lst:
            rsi_lst.extend(klu.rsi for klu in klc.lst)
        return 10000.0/(min(rsi_lst)+1e-7) if self.is_down() else max(rsi_lst)

    @make_cache
    cpdef double Cal_MACD_area(self):
        cdef:
            double _s = 1e-7
            CKLine klc
            CKLine_Unit klu
        for klc in self.klc_lst:
            for klu in klc.lst:
                _s += fabs(klu.macd.macd)
        return _s

    @make_cache
    cpdef double Cal_MACD_peak(self):
        cdef:
            double peak = 1e-7
            CKLine klc
            CKLine_Unit klu
        for klc in self.klc_lst:
            for klu in klc.lst:
                if fabs(klu.macd.macd) > peak:
                    if self.is_down() and klu.macd.macd < 0:
                        peak = fabs(klu.macd.macd)
                    elif self.is_up() and klu.macd.macd > 0:
                        peak = fabs(klu.macd.macd)
        return peak

    cpdef double Cal_MACD_half(self, bint is_reverse):
        if is_reverse:
            return self.Cal_MACD_half_reverse()
        else:
            return self.Cal_MACD_half_obverse()

    @make_cache
    cpdef double Cal_MACD_half_obverse(self):
        cdef:
            double _s = 1e-7
            CKLine_Unit begin_klu = self.get_begin_klu()
            double peak_macd = begin_klu.macd.macd
            CKLine klc
            CKLine_Unit klu
        for klc in self.klc_lst:
            for klu in klc.lst:
                if klu.idx < begin_klu.idx:
                    continue
                if klu.macd.macd*peak_macd > 0:
                    _s += fabs(klu.macd.macd)
                else:
                    break
            else:  # 没有被break，继续找写一个KLC
                continue
            break
        return _s

    @make_cache
    cpdef double Cal_MACD_half_reverse(self):
        cdef:
            double _s = 1e-7
            CKLine_Unit begin_klu = self.get_end_klu()
            double peak_macd = begin_klu.macd.macd
            CKLine klc
            CKLine_Unit klu
        for klc in self.klc_lst_re:
            for klu in klc[::-1]:
                if klu.idx > begin_klu.idx:
                    continue
                if klu.macd.macd*peak_macd > 0:
                    _s += fabs(klu.macd.macd)
                else:
                    break
            else:  # 没有被break，继续找写一个KLC
                continue
            break
        return _s

    @make_cache
    cpdef double Cal_MACD_diff(self):
        cdef:
            double _max = float("-inf")
            double _min = float("inf")
            CKLine klc
            CKLine_Unit klu
            double macd
        for klc in self.klc_lst:
            for klu in klc.lst:
                macd = klu.macd.macd
                if macd > _max:
                    _max = macd
                if macd < _min:
                    _min = macd
        return _max-_min

    @make_cache
    cpdef double Cal_MACD_slope(self):
        cdef:
            CKLine_Unit begin_klu = self.get_begin_klu()
            CKLine_Unit end_klu = self.get_end_klu()
        if self.is_up():
            return (end_klu.high - begin_klu.low)/end_klu.high/(end_klu.idx - begin_klu.idx + 1)
        else:
            return (begin_klu.high - end_klu.low)/begin_klu.high/(end_klu.idx - begin_klu.idx + 1)

    @make_cache
    cpdef double Cal_MACD_amp(self):
        cdef:
            CKLine_Unit begin_klu = self.get_begin_klu()
            CKLine_Unit end_klu = self.get_end_klu()
        if self.is_down():
            return (begin_klu.high-end_klu.low)/begin_klu.high
        else:
            return (end_klu.high-begin_klu.low)/begin_klu.low

    cpdef double Cal_MACD_trade_metric(self, str metric, bint cal_avg=False):
        cdef:
            double _s = 0
            int cnt = 0
            CKLine klc
            CKLine_Unit klu
            object metric_res
        for klc in self.klc_lst:
            for klu in klc.lst:
                metric_res = klu.trade_info.metric.get(metric)
                if metric_res is None:
                    return 0.0
                _s += metric_res
                cnt += 1
        return _s / cnt if cal_avg else _s
