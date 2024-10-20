# cython: language_level=3
import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fmin, fmax

from Common.CEnum import DATA_FIELD, TRADE_INFO_LST, TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Math.BOLL cimport BOLL_Metric, BollModel
from Math.Demark cimport CDemarkEngine, CDemarkIndex
from Math.KDJ cimport KDJ
from Math.MACD cimport CMACD, CMACD_item
from Math.RSI cimport RSI
from Math.TrendModel cimport CTrendModel

from .TradeInfo cimport CTradeInfo

cdef class CKLine_Unit:
    cdef:
        public object kl_type
        public CTime time
        public double close, open, high, low
        public CTradeInfo trade_info
        public CDemarkIndex demark
        public list sub_kl_list
        public CKLine_Unit sup_kl
        object __klc
        public dict trend
        public int limit_flag
        public CKLine_Unit pre, next
        int __idx

    def __cinit__(self, dict kl_dict, bint autofix=False):
        self.kl_type = None
        self.time = kl_dict[DATA_FIELD.FIELD_TIME]
        self.close = kl_dict[DATA_FIELD.FIELD_CLOSE]
        self.open = kl_dict[DATA_FIELD.FIELD_OPEN]
        self.high = kl_dict[DATA_FIELD.FIELD_HIGH]
        self.low = kl_dict[DATA_FIELD.FIELD_LOW]

        self.check(autofix)

        self.trade_info = CTradeInfo(kl_dict)

        self.demark = CDemarkIndex()

        self.sub_kl_list = []
        self.sup_kl = None

        self.__klc = None

        self.trend = {}

        self.limit_flag = 0
        self.pre = None
        self.next = None

        self.set_idx(-1)

    def __deepcopy__(self, memo):
        cdef dict _dict = {
            DATA_FIELD.FIELD_TIME: self.time,
            DATA_FIELD.FIELD_CLOSE: self.close,
            DATA_FIELD.FIELD_OPEN: self.open,
            DATA_FIELD.FIELD_HIGH: self.high,
            DATA_FIELD.FIELD_LOW: self.low,
        }
        for metric in TRADE_INFO_LST:
            if metric in self.trade_info.metric:
                _dict[metric] = self.trade_info.metric[metric]
        obj = CKLine_Unit(_dict)
        obj.demark = self.demark.__deepcopy__(memo)
        obj.trend = self.trend.__deepcopy__(memo)
        obj.limit_flag = self.limit_flag
        if hasattr(self, "macd"):
            obj.macd = self.macd.__deepcopy__(memo)
        if hasattr(self, "boll"):
            obj.boll = self.boll.__deepcopy__(memo)
        if hasattr(self, "rsi"):
            obj.rsi = self.rsi.__deepcopy__(memo)
        if hasattr(self, "kdj"):
            obj.kdj = self.kdj.__deepcopy__(memo)
        obj.set_idx(self.idx)
        memo[id(self)] = obj
        return obj

    @property
    def klc(self):
        assert self.__klc is not None
        return self.__klc

    def set_klc(self, klc):
        self.__klc = klc

    @property
    def idx(self):
        return self.__idx

    def set_idx(self, int idx):
        self.__idx = idx

    def __str__(self):
        return f"{self.idx}:{self.time}/{self.kl_type} open={self.open} close={self.close} high={self.high} low={self.low} {self.trade_info}"

    cpdef void check(self, bint autofix=False):
        cdef double min_val = fmin(fmin(fmin(self.low, self.open), self.high), self.close)
        cdef double max_val = fmax(fmax(fmax(self.low, self.open), self.high), self.close)
        
        if self.low > min_val:
            if autofix:
                self.low = min_val
            else:
                raise CChanException(f"{self.time} low price={self.low} is not min of [low={self.low}, open={self.open}, high={self.high}, close={self.close}]", ErrCode.KL_DATA_INVALID)
        if self.high < max_val:
            if autofix:
                self.high = max_val
            else:
                raise CChanException(f"{self.time} high price={self.high} is not max of [low={self.low}, open={self.open}, high={self.high}, close={self.close}]", ErrCode.KL_DATA_INVALID)

    cpdef void add_children(self, CKLine_Unit child):
        self.sub_kl_list.append(child)

    cpdef void set_parent(self, CKLine_Unit parent):
        self.sup_kl = parent

    def get_children(self):
        return iter(self.sub_kl_list)

    cpdef double _low(self):
        return self.low

    cpdef double _high(self):
        return self.high

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void set_metric(self, list metric_model_lst):
        cdef object metric_model
        for metric_model in metric_model_lst:
            if isinstance(metric_model, CMACD):
                self.macd = metric_model.add(self.close)
            elif isinstance(metric_model, CTrendModel):
                if metric_model.type not in self.trend:
                    self.trend[metric_model.type] = {}
                self.trend[metric_model.type][metric_model.T] = metric_model.add(self.close)
            elif isinstance(metric_model, BollModel):
                self.boll = metric_model.add(self.close)
            elif isinstance(metric_model, CDemarkEngine):
                self.demark = metric_model.update(idx=self.idx, close=self.close, high=self.high, low=self.low)
            elif isinstance(metric_model, RSI):
                self.rsi = metric_model.add(self.close)
            elif isinstance(metric_model, KDJ):
                self.kdj = metric_model.add(self.high, self.low, self.close)

    cpdef object get_parent_klc(self):
        assert self.sup_kl is not None
        return self.sup_kl.klc

    cpdef bint include_sub_lv_time(self, str sub_lv_t):
        cdef CKLine_Unit sub_klu
        if self.time.to_str() == sub_lv_t:
            return True
        for sub_klu in self.sub_kl_list:
            if sub_klu.time.to_str() == sub_lv_t:
                return True
            if sub_klu.include_sub_lv_time(sub_lv_t):
                return True
        return False

    cpdef void set_pre_klu(self, CKLine_Unit pre_klu):
        if pre_klu is None:
            return
        pre_klu.next = self
        self.pre = pre_klu
