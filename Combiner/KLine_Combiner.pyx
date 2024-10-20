# cython: language_level=3
cimport cython
from libc.math cimport fmax, fmin
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from typing import Generic, Iterable, List, Optional, Self, TypeVar, Union, overload

from Common.cache import make_cache
from Common.CEnum import FX_TYPE, KLINE_DIR
from Common.ChanException import CChanException, ErrCode
from KLine.KLine_Unit cimport CKLine_Unit

from .Combine_Item cimport CCombine_Item

T = TypeVar('T')

cdef class CKLine_Combiner(Generic[T]):
    cdef:
        public int time_begin
        public int time_end
        public double high
        public double low
        public list lst
        public KLINE_DIR dir
        public FX_TYPE fx
        public CKLine_Combiner pre
        public CKLine_Combiner next
        dict _memoize_cache

    def __cinit__(self, T kl_unit, KLINE_DIR _dir):
        cdef CCombine_Item item = CCombine_Item(kl_unit)
        self.time_begin = item.time_begin
        self.time_end = item.time_end
        self.high = item.high
        self.low = item.low

        self.lst = [kl_unit]  # 本级别每一根单位K线

        self.dir = _dir
        self.fx = FX_TYPE.UNKNOWN
        self.pre = None
        self.next = None

        self._memoize_cache = {}

    cpdef void clean_cache(self):
        self._memoize_cache = {}

    @property
    def pre(self):
        assert self.pre is not None
        return self.pre

    @property
    def next(self):
        return self.next

    cpdef CKLine_Combiner get_next(self):
        assert self.next is not None
        return self.next

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef KLINE_DIR test_combine(self, CCombine_Item item, bint exclude_included=False, object allow_top_equal=None):
        if (self.high >= item.high and self.low <= item.low):
            return KLINE_DIR.COMBINE
        if (self.high <= item.high and self.low >= item.low):
            if allow_top_equal == 1 and self.high == item.high and self.low > item.low:
                return KLINE_DIR.DOWN
            elif allow_top_equal == -1 and self.low == item.low and self.high < item.high:
                return KLINE_DIR.UP
            return KLINE_DIR.INCLUDED if exclude_included else KLINE_DIR.COMBINE
        if (self.high > item.high and self.low > item.low):
            return KLINE_DIR.DOWN
        if (self.high < item.high and self.low < item.low):
            return KLINE_DIR.UP
        else:
            raise CChanException("combine type unknown", ErrCode.COMBINER_ERR)

    cpdef void add(self, T unit_kl):
        # only for deepcopy
        self.lst.append(unit_kl)

    cpdef void set_fx(self, FX_TYPE fx):
        # only for deepcopy
        self.fx = fx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef KLINE_DIR try_add(self, T unit_kl, bint exclude_included=False, object allow_top_equal=None):
        # allow_top_equal = None普通模式
        # allow_top_equal = 1 被包含，顶部相等不合并
        # allow_top_equal = -1 被包含，底部相等不合并
        cdef:
            CCombine_Item combine_item = CCombine_Item(unit_kl)
            KLINE_DIR _dir

        _dir = self.test_combine(combine_item, exclude_included, allow_top_equal)
        if _dir == KLINE_DIR.COMBINE:
            self.lst.append(unit_kl)
            if isinstance(unit_kl, CKLine_Unit):
                unit_kl.set_klc(self)
            if self.dir == KLINE_DIR.UP:
                if combine_item.high != combine_item.low or combine_item.high != self.high:  # 处理一字K线
                    self.high = fmax(self.high, combine_item.high)
                    self.low = fmax(self.low, combine_item.low)
            elif self.dir == KLINE_DIR.DOWN:
                if combine_item.high != combine_item.low or combine_item.low != self.low:  # 处理一字K线
                    self.high = fmin(self.high, combine_item.high)
                    self.low = fmin(self.low, combine_item.low)
            else:
                raise CChanException(f"KLINE_DIR = {self.dir} err!!! must be {KLINE_DIR.UP}/{KLINE_DIR.DOWN}", ErrCode.COMBINER_ERR)
            self.time_end = combine_item.time_end
            self.clean_cache()
        # 返回UP/DOWN/COMBINE给KL_LIST，设置下一个的方向
        return _dir

    cpdef T get_peak_klu(self, bint is_high):
        # 获取最大值 or 最小值所在klu/bi
        return self.get_high_peak_klu() if is_high else self.get_low_peak_klu()

    @make_cache
    cpdef T get_high_peak_klu(self):
        cdef:
            T kl
            CCombine_Item item
        for kl in reversed(self.lst):
            item = CCombine_Item(kl)
            if item.high == self.high:
                return kl
        raise CChanException("can't find peak...", ErrCode.COMBINER_ERR)

    @make_cache
    cpdef T get_low_peak_klu(self):
        cdef:
            T kl
            CCombine_Item item
        for kl in reversed(self.lst):
            item = CCombine_Item(kl)
            if item.low == self.low:
                return kl
        raise CChanException("can't find peak...", ErrCode.COMBINER_ERR)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void update_fx(self, CKLine_Combiner _pre, CKLine_Combiner _next, bint exclude_included=False, object allow_top_equal=None):
        # allow_top_equal = None普通模式
        # allow_top_equal = 1 被包含，顶部相等不合并
        # allow_top_equal = -1 被包含，底部相等不合并
        self.set_next(_next)
        self.set_pre(_pre)
        _next.set_pre(self)
        if exclude_included:
            if _pre.high < self.high and _next.high <= self.high and _next.low < self.low:
                if allow_top_equal == 1 or _next.high < self.high:
                    self.fx = FX_TYPE.TOP
            elif _next.high > self.high and _pre.low > self.low and _next.low >= self.low:
                if allow_top_equal == -1 or _next.low > self.low:
                    self.fx = FX_TYPE.BOTTOM
        elif _pre.high < self.high and _next.high < self.high and _pre.low < self.low and _next.low < self.low:
            self.fx = FX_TYPE.TOP
        elif _pre.high > self.high and _next.high > self.high and _pre.low > self.low and _next.low > self.low:
            self.fx = FX_TYPE.BOTTOM
        self.clean_cache()

    def __str__(self):
        return f"{self.time_begin}~{self.time_end} {self.low}->{self.high}"

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> List[T]: ...

    def __getitem__(self, index: Union[slice, int]) -> Union[List[T], T]:
        return self.lst[index]

    def __len__(self):
        return len(self.lst)

    def __iter__(self) -> Iterable[T]:
        yield from self.lst

    cpdef void set_pre(self, CKLine_Combiner _pre):
        self.pre = _pre
        self.clean_cache()

    cpdef void set_next(self, CKLine_Combiner _next):
        self.next = _next
        self.clean_cache()
