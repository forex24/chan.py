# cython: language_level=3
import cython
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free

from Common.CEnum import BI_DIR, TREND_LINE_SIDE

cdef struct Point:
    int x
    double y

cdef struct Line:
    Point p
    double slope

cdef double cal_slope(Point p1, Point p2):
    if p1.x != p2.x:
        return (p1.y - p2.y) / (p1.x - p2.x)
    return float('inf')

cdef double cal_dis(Line line, Point p):
    return fabs(line.slope * p.x - p.y + line.p.y - line.slope * line.p.x) / sqrt(line.slope * line.slope + 1)

cdef Line cal_tl(Point* c_p, int c_p_size, BI_DIR _dir, TREND_LINE_SIDE side):
    cdef:
        Point p = c_p[0]
        double peak_slope
        int idx = 1
        int point_idx
        Point p2
        double slope

    if side == TREND_LINE_SIDE.INSIDE:
        peak_slope = 0
    elif _dir == BI_DIR.UP:
        peak_slope = float('inf')
    else:
        peak_slope = -float('inf')

    for point_idx in range(1, c_p_size):
        p2 = c_p[point_idx]
        slope = cal_slope(p, p2)
        if (_dir == BI_DIR.UP and slope < 0) or (_dir == BI_DIR.DOWN and slope > 0):
            continue
        if side == TREND_LINE_SIDE.INSIDE:
            if (_dir == BI_DIR.UP and slope > peak_slope) or (_dir == BI_DIR.DOWN and slope < peak_slope):
                peak_slope = slope
                idx = point_idx + 1
        else:
            if (_dir == BI_DIR.UP and slope < peak_slope) or (_dir == BI_DIR.DOWN and slope > peak_slope):
                peak_slope = slope
                idx = point_idx + 1

    return Line(p=p, slope=peak_slope), idx

cdef class CTrendLine:
    cdef:
        Line line
        TREND_LINE_SIDE side

    def __init__(self, list lst, TREND_LINE_SIDE side=TREND_LINE_SIDE.OUTSIDE):
        self.side = side
        self.cal(lst)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void cal(self, list lst):
        cdef:
            double bench = float('inf')
            Point* all_p
            int all_p_size
            Point* c_p
            int c_p_size
            Line line
            int idx
            double dis
            int i

        if self.side == TREND_LINE_SIDE.INSIDE:
            all_p_size = len(lst)
            all_p = <Point*>malloc(all_p_size * sizeof(Point))
            for i in range(all_p_size):
                all_p[i] = Point(lst[i].get_begin_klu().idx, lst[i].get_begin_val())
        else:
            all_p_size = len(lst)
            all_p = <Point*>malloc(all_p_size * sizeof(Point))
            for i in range(all_p_size):
                all_p[i] = Point(lst[i].get_end_klu().idx, lst[i].get_end_val())

        c_p_size = all_p_size
        c_p = <Point*>malloc(c_p_size * sizeof(Point))
        for i in range(c_p_size):
            c_p[i] = all_p[i]

        while True:
            line, idx = cal_tl(c_p, c_p_size, lst[-1].dir, self.side)
            dis = 0
            for i in range(all_p_size):
                dis += cal_dis(line, all_p[i])
            if dis < bench:
                bench = dis
                self.line = line
            c_p = &c_p[idx]
            c_p_size -= idx
            if c_p_size == 1:
                break

        free(all_p)
        free(c_p)
