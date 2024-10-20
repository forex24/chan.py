# cython: language_level=3
import abc
from typing import Iterable

from KLine.KLine_Unit cimport CKLine_Unit

cdef class CCommonStockApi:
    cdef:
        public str code
        public str name
        public bint is_stock
        public object k_type
        public object begin_date
        public object end_date
        public str autype

    def __cinit__(self, str code, object k_type, object begin_date, object end_date, str autype):
        self.code = code
        self.name = None
        self.is_stock = None
        self.k_type = k_type
        self.begin_date = begin_date
        self.end_date = end_date
        self.autype = autype
        self.SetBasciInfo()

    @abc.abstractmethod
    def get_kl_data(self) -> Iterable[CKLine_Unit]:
        pass

    @abc.abstractmethod
    def SetBasciInfo(self):
        pass

    @classmethod
    @abc.abstractmethod
    def do_init(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def do_close(cls):
        pass
