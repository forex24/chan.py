# cython: language_level=3
cimport cython
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem
from libc.stdlib cimport malloc, free
from typing import Dict, Optional

from Common.CEnum cimport TRADE_INFO_LST

cdef class CTradeInfo:
    cdef:
        public dict metric

    def __cinit__(self, dict info):
        self.metric = {}
        cdef:
            str metric_name
            object value
            void* value_ptr
        for metric_name in TRADE_INFO_LST:
            value_ptr = PyDict_GetItem(info, metric_name)
            if value_ptr != NULL:
                value = <object>value_ptr
                PyDict_SetItem(self.metric, metric_name, value)
            else:
                PyDict_SetItem(self.metric, metric_name, None)

    def __str__(self):
        return " ".join([f"{metric_name}:{value}" for metric_name, value in self.metric.items()])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef dict to_dict(self):
        return self.metric

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object get(self, str key, object default=None):
        cdef void* value_ptr = PyDict_GetItem(self.metric, key)
        if value_ptr != NULL:
            return <object>value_ptr
        return default

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void set(self, str key, object value):
        PyDict_SetItem(self.metric, key, value)
