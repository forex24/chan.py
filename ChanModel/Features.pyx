# cython: language_level=3
from cpython.dict cimport PyDict_GetItem, PyDict_SetItem, PyDict_Update, PyDict_Items
from cpython.float cimport PyFloat_FromDouble
from libc.math cimport isnan
from typing import Optional, Dict, Tuple, Iterator

cdef class CFeatures:
    cdef:
        dict __features

    def __cinit__(self, dict initFeat=None):
        self.__features = {} if initFeat is None else dict(initFeat)

    def __iter__(self) -> Iterator[Tuple[str, float]]:
        return self.__features.items()

    def __getitem__(self, str k) -> float:
        cdef void* result = PyDict_GetItem(self.__features, k)
        if result is NULL:
            raise KeyError(k)
        return <object>result

    cpdef void add_feat(self, object inp1, object inp2=None):
        if inp2 is None:
            if not isinstance(inp1, dict):
                raise TypeError("When inp2 is None, inp1 must be a dictionary")
            PyDict_Update(self.__features, inp1)
        else:
            if not isinstance(inp1, str):
                raise TypeError("When inp2 is provided, inp1 must be a string")
            PyDict_SetItem(self.__features, inp1, inp2)

    cpdef dict to_dict(self):
        return dict(self.__features)

    cpdef object get(self, str key, object default=None):
        cdef void* result = PyDict_GetItem(self.__features, key)
        if result is NULL:
            return default
        return <object>result

    cpdef bint has_nan(self):
        cdef:
            str key
            object value
        for key, value in PyDict_Items(self.__features):
            if isinstance(value, float) and isnan(value):
                return True
        return False

    cpdef void remove_nan(self):
        cdef:
            str key
            object value
            dict new_features = {}
        for key, value in PyDict_Items(self.__features):
            if not (isinstance(value, float) and isnan(value)):
                PyDict_SetItem(new_features, key, value)
        self.__features = new_features

    cpdef CFeatures copy(self):
        return CFeatures(self.__features.copy())

    def __str__(self):
        return str(self.__features)

    def __repr__(self):
        return f"CFeatures({self.__features})"
