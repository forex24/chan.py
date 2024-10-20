# cython: language_level=3
import cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport fmax, fmin

from Combiner.KLine_Combiner cimport CKLine_Combiner
from Common.CEnum import FX_CHECK_METHOD, FX_TYPE, KLINE_DIR
from Common.ChanException import CChanException, ErrCode
from Common.func_util import has_overlap
from KLine.KLine_Unit cimport CKLine_Unit

cdef class CKLine(CKLine_Combiner):
    cdef:
        public int idx
        public object kl_type

    def __cinit__(self, CKLine_Unit kl_unit, int idx, KLINE_DIR _dir=KLINE_DIR.UP):
        super().__init__(kl_unit, _dir)
        self.idx = idx
        self.kl_type = kl_unit.kl_type
        kl_unit.set_klc(self)

    def __str__(self):
        cdef str fx_token = ""
        if self.fx == FX_TYPE.TOP:
            fx_token = "^"
        elif self.fx == FX_TYPE.BOTTOM:
            fx_token = "_"
        return f"{self.idx}th{fx_token}:{self.time_begin}~{self.time_end}({self.kl_type}|{len(self.lst)}) low={self.low} high={self.high}"

    cpdef list GetSubKLC(self):
        cdef:
            CKLine_Unit klu
            CKLine_Unit sub_klu
            CKLine last_klc = None
        for klu in self.lst:
            for sub_klu in klu.get_children():
                if sub_klu.klc != last_klc:
                    last_klc = sub_klu.klc
                    yield sub_klu.klc

    cpdef double get_klu_max_high(self):
        return max(x.high for x in self.lst)

    cpdef double get_klu_min_low(self):
        return min(x.low for x in self.lst)

    cpdef bint has_gap_with_next(self):
        assert self.next is not None
        return not has_overlap(self.get_klu_min_low(), self.get_klu_max_high(), self.next.get_klu_min_low(), self.next.get_klu_max_high(), equal=True)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint check_fx_valid(self, CKLine item2, FX_CHECK_METHOD method, bint for_virtual=False):
        assert self.next is not None and item2.pre is not None
        assert self.pre is not None
        assert item2.idx > self.idx

        cdef:
            double item2_high, self_low, item2_low, cur_high
            bint is_top = self.fx == FX_TYPE.TOP

        if is_top:
            assert for_virtual or item2.fx == FX_TYPE.BOTTOM
            if for_virtual and item2.dir != KLINE_DIR.DOWN:
                return False

            if method == FX_CHECK_METHOD.HALF:
                item2_high = fmax(item2.pre.high, item2.high)
                self_low = fmin(self.low, self.next.low)
            elif method == FX_CHECK_METHOD.LOSS:
                item2_high = item2.high
                self_low = self.low
            elif method in (FX_CHECK_METHOD.STRICT, FX_CHECK_METHOD.TOTALLY):
                if for_virtual:
                    item2_high = fmax(item2.pre.high, item2.high)
                else:
                    assert item2.next is not None
                    item2_high = fmax(fmax(item2.pre.high, item2.high), item2.next.high)
                self_low = fmin(fmin(self.pre.low, self.low), self.next.low)
            else:
                raise CChanException("bi_fx_check config error!", ErrCode.CONFIG_ERROR)

            if method == FX_CHECK_METHOD.TOTALLY:
                return self.low > item2_high
            else:
                return self.high > item2_high and item2.low < self_low

        else:
            assert for_virtual or item2.fx == FX_TYPE.TOP
            if for_virtual and item2.dir != KLINE_DIR.UP:
                return False

            if method == FX_CHECK_METHOD.HALF:
                item2_low = fmin(item2.pre.low, item2.low)
                cur_high = fmax(self.high, self.next.high)
            elif method == FX_CHECK_METHOD.LOSS:
                item2_low = item2.low
                cur_high = self.high
            elif method in (FX_CHECK_METHOD.STRICT, FX_CHECK_METHOD.TOTALLY):
                if for_virtual:
                    item2_low = fmin(item2.pre.low, item2.low)
                else:
                    assert item2.next is not None
                    item2_low = fmin(fmin(item2.pre.low, item2.low), item2.next.low)
                cur_high = fmax(fmax(self.pre.high, self.high), self.next.high)
            else:
                raise CChanException("bi_fx_check config error!", ErrCode.CONFIG_ERROR)

            if method == FX_CHECK_METHOD.TOTALLY:
                return self.high < item2_low
            else:
                return self.low < item2_low and item2.high > cur_high
