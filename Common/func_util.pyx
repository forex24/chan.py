# cython: language_level=3
import cython
from libc.math cimport isnan, isinf
from Common.CEnum cimport BI_DIR, KL_TYPE

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint kltype_lt_day(KL_TYPE _type):
    return _type.value < KL_TYPE.K_DAY.value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint kltype_lte_day(KL_TYPE _type):
    return _type.value <= KL_TYPE.K_DAY.value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void check_kltype_order(list type_list):
    cdef:
        int last_lv = type_list[0].value
        KL_TYPE kl_type
    for kl_type in type_list[1:]:
        assert kl_type.value < last_lv, "lv_list的顺序必须从大级别到小级别"
        last_lv = kl_type.value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef BI_DIR revert_bi_dir(BI_DIR dir):
    return BI_DIR.DOWN if dir == BI_DIR.UP else BI_DIR.UP

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint has_overlap(double l1, double h1, double l2, double h2, bint equal=False):
    return h2 >= l1 and h1 >= l2 if equal else h2 > l1 and h1 > l2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double str2float(str s):
    try:
        return float(s)
    except ValueError:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object _parse_inf(object v):
    if isinstance(v, float):
        if isnan(v) or isinf(v):
            if v > 0:
                return 'float("inf")'
            else:
                return 'float("-inf")'
    return v

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint is_trading_time(int hour, int minute):
    if 9 <= hour < 15:
        if hour == 9 and minute < 30:
            return False
        if hour == 11 and minute >= 30:
            return False
        if 12 <= hour < 13:
            return False
        return True
    return False

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef bint is_trading_day(int weekday):
    return 0 <= weekday <= 4

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double round_to(double x, double base=0.01):
    return round(x / base) * base
