from enum import Enum
from typing import Literal

cdef class DATA_SRC(Enum):
    pass

cdef class KL_TYPE(Enum):
    pass

cdef class KLINE_DIR(Enum):
    pass

cdef class FX_TYPE(Enum):
    pass

cdef class BI_DIR(Enum):
    pass

cdef class BI_TYPE(Enum):
    pass

BSP_MAIN_TYPE = Literal['1', '2', '3']

cdef class BSP_TYPE(Enum):
    cpdef BSP_MAIN_TYPE main_type(self)

cdef class AUTYPE(Enum):
    pass

cdef class TREND_TYPE(Enum):
    pass

cdef class TREND_LINE_SIDE(Enum):
    pass

cdef class LEFT_SEG_METHOD(Enum):
    pass

cdef class FX_CHECK_METHOD(Enum):
    pass

cdef class SEG_TYPE(Enum):
    pass

cdef class MACD_ALGO(Enum):
    pass

cdef class DATA_FIELD:
    cdef:
        readonly str FIELD_TIME
        readonly str FIELD_OPEN
        readonly str FIELD_HIGH
        readonly str FIELD_LOW
        readonly str FIELD_CLOSE
        readonly str FIELD_VOLUME
        readonly str FIELD_TURNOVER
        readonly str FIELD_TURNRATE

cdef list TRADE_INFO_LST
