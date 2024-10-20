# cython: language_level=3
cimport cython
from Common.ChanException import CChanException, ErrCode

cdef class CCombine_Item:
    cdef:
        public int time_begin
        public int time_end
        public double high
        public double low

    def __cinit__(self, item):
        from Bi.Bi import CBi
        from KLine.KLine_Unit import CKLine_Unit
        from Seg.Seg import CSeg

        if isinstance(item, CBi):
            self.time_begin = item.begin_klc.idx
            self.time_end = item.end_klc.idx
            self.high = item._high()
            self.low = item._low()
        elif isinstance(item, CKLine_Unit):
            self.time_begin = item.time
            self.time_end = item.time
            self.high = item.high
            self.low = item.low
        elif isinstance(item, CSeg):
            self.time_begin = item.start_bi.begin_klc.idx
            self.time_end = item.end_bi.end_klc.idx
            self.high = item._high()
            self.low = item._low()
        else:
            raise CChanException(f"{type(item)} is unsupported sub class of CCombine_Item", ErrCode.COMMON_ERROR)

    def __str__(self):
        return f"CCombine_Item(time_begin={self.time_begin}, time_end={self.time_end}, high={self.high}, low={self.low})"

    def __repr__(self):
        return self.__str__()
