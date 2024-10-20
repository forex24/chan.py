# cython: language_level=3
cimport cython
import pandas as pd
from cpython.dict cimport PyDict_SetItem

from Common.CEnum cimport DATA_FIELD, KL_TYPE
from Common.ChanException cimport CChanException, ErrCode
from Common.CTime cimport CTime
from Common.func_util cimport str2float
from KLine.KLine_Unit cimport CKLine_Unit

from .CommonStockAPI cimport CCommonStockApi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict create_item_dict(list data, list column_name):
    cdef:
        int i
        dict result = {}
        object value
    for i in range(len(data)):
        value = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else float(data[i])
        PyDict_SetItem(result, column_name[i], value)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef CTime parse_time_column(object inp):
    return CTime(inp.year, inp.month, inp.day, inp.hour, inp.minute)

cdef class DATAFRAME_API(CCommonStockApi):
    cdef:
        public list columns
        public int time_column_idx
        public object df
        public object begin_date
        public object end_date

    def __cinit__(self, object code, object k_type=KL_TYPE.K_DAY, object begin_date=None, object end_date=None, object autype=None):
        self.columns = [
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
            DATA_FIELD.FIELD_TIME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]  # 每一列字段
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        self.df = code  # 传入的DataFrame
        super().__init__(code, k_type, begin_date, end_date, autype)
        self.begin_date = pd.to_datetime(begin_date) if begin_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_kl_data(self):
        cdef:
            list data
            dict item_dict
            object row_time
        if self.df is None:
            raise CChanException("DataFrame is not provided", ErrCode.SRC_DATA_NOT_FOUND)

        for _, row in self.df.iterrows():
            data = row.tolist()
            if len(data) != len(self.columns):
                raise CChanException("DataFrame format error", ErrCode.SRC_DATA_FORMAT_ERROR)
            row_time = data[self.time_column_idx]
            if self.begin_date is not None and row_time < self.begin_date:
                continue
            if self.end_date is not None and row_time > self.end_date:
                continue
            item_dict = create_item_dict(data, self.columns)
            yield CKLine_Unit(item_dict)

    cpdef SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
