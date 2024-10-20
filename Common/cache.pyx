# cython: language_level=3
import cython
import inspect
import types
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF, Py_DECREF

cdef class make_cache:
    cdef:
        object func
        str func_key

    def __cinit__(self, func):
        self.func = func

        fargspec = inspect.getfullargspec(func)
        if len(fargspec.args) != 1 or fargspec.args[0] != "self":
            raise Exception("@memoize must be `(self)`")

        # set key for this function
        self.func_key = str(func)

    def __get__(self, instance, cls):
        if instance is None:
            raise Exception("@memoize's must be bound")

        if not hasattr(instance, "_memoize_cache"):
            setattr(instance, "_memoize_cache", {})

        return types.MethodType(self, instance)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, *args, **kwargs):
        cdef:
            object instance = args[0]
            dict cache = instance._memoize_cache
            PyObject* result_ptr

        result_ptr = PyDict_GetItem(cache, self.func_key)
        if result_ptr != NULL:
            return <object>result_ptr

        result = self.func(*args, **kwargs)
        PyDict_SetItem(cache, self.func_key, result)
        return result

cdef extern from "Python.h":
    PyObject* PyDict_GetItem(object p, object key) except NULL
    int PyDict_SetItem(object p, object key, object val) except -1
