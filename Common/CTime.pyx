# cython: language_level=3
from datetime import datetime
from libc.time cimport time_t, tm, mktime
from cpython.datetime cimport datetime, timedelta

cdef extern from "time.h":
    tm* localtime(const time_t* timer)

cdef class CTime:
    cdef:
        public int year, month, day, hour, minute, second
        public bint auto
        public double ts

    def __cinit__(self, int year, int month, int day, int hour, int minute, int second=0, bint auto=False):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.auto = auto  # 自适应对天的理解
        self.set_timestamp()  # set self.ts

    def __str__(self):
        if self.hour == 0 and self.minute == 0:
            return f"{self.year:04}/{self.month:02}/{self.day:02}"
        else:
            return f"{self.year:04}/{self.month:02}/{self.day:02} {self.hour:02}:{self.minute:02}"

    def to_str(self):
        return self.__str__()

    cpdef str toDateStr(self, str splt=''):
        return f"{self.year:04}{splt}{self.month:02}{splt}{self.day:02}"

    cpdef CTime toDate(self):
        return CTime(self.year, self.month, self.day, 0, 0, auto=False)

    cdef void set_timestamp(self):
        cdef:
            tm timeinfo
            time_t timestamp

        if self.hour == 0 and self.minute == 0 and self.auto:
            timeinfo.tm_year = self.year - 1900
            timeinfo.tm_mon = self.month - 1
            timeinfo.tm_mday = self.day
            timeinfo.tm_hour = 23
            timeinfo.tm_min = 59
            timeinfo.tm_sec = self.second
            timeinfo.tm_isdst = -1
        else:
            timeinfo.tm_year = self.year - 1900
            timeinfo.tm_mon = self.month - 1
            timeinfo.tm_mday = self.day
            timeinfo.tm_hour = self.hour
            timeinfo.tm_min = self.minute
            timeinfo.tm_sec = self.second
            timeinfo.tm_isdst = -1

        timestamp = mktime(&timeinfo)
        self.ts = timestamp

    def __gt__(self, CTime t2):
        return self.ts > t2.ts

    def __ge__(self, CTime t2):
        return self.ts >= t2.ts

    def __lt__(self, CTime t2):
        return self.ts < t2.ts

    def __le__(self, CTime t2):
        return self.ts <= t2.ts

    def __eq__(self, CTime t2):
        return self.ts == t2.ts

    def __ne__(self, CTime t2):
        return self.ts != t2.ts

    @staticmethod
    def now():
        cdef:
            time_t current_time
            tm* timeinfo

        time(&current_time)
        timeinfo = localtime(&current_time)

        return CTime(
            timeinfo.tm_year + 1900,
            timeinfo.tm_mon + 1,
            timeinfo.tm_mday,
            timeinfo.tm_hour,
            timeinfo.tm_min,
            timeinfo.tm_sec
        )

    @staticmethod
    def from_timestamp(double timestamp):
        cdef:
            time_t ts = <time_t>timestamp
            tm* timeinfo

        timeinfo = localtime(&ts)

        return CTime(
            timeinfo.tm_year + 1900,
            timeinfo.tm_mon + 1,
            timeinfo.tm_mday,
            timeinfo.tm_hour,
            timeinfo.tm_min,
            timeinfo.tm_sec
        )

    def to_datetime(self):
        return datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)

    @staticmethod
    def from_datetime(dt):
        return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    def __add__(self, timedelta delta):
        cdef datetime dt = self.to_datetime() + delta
        return CTime.from_datetime(dt)

    def __sub__(self, other):
        if isinstance(other, CTime):
            return self.to_datetime() - other.to_datetime()
        elif isinstance(other, timedelta):
            dt = self.to_datetime() - other
            return CTime.from_datetime(dt)
        else:
            raise TypeError("Unsupported operand type for -")
