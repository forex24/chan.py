cdef class make_cache:
    cdef:
        object func
        str func_key

    cpdef __get__(self, instance, cls)
    cpdef __call__(self, *args, **kwargs)

cdef extern from "Python.h":
    ctypedef struct PyObject

    PyObject* PyDict_GetItem(object p, object key) except NULL
    int PyDict_SetItem(object p, object key, object val) except -1
