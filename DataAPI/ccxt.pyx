# cython: language_level=3
from datetime import datetime

import ccxt

from Common.CEnum cimport AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime cimport CTime
from Common.func_util cimport kltype_lt_day, str2float
from KLine.KLine_Unit cimport CKLine_Unit

from .CommonStockAPI cimport CCommonStockApi

cdef dict _dict = {
    "time": DATA_FIELD.FIELD_TIME,
    "open": DATA_FIELD.FIELD_OPEN,
    "high": DATA_FIELD.FIELD_HIGH,
    "low": DATA_FIELD.FIELD_LOW,
    "close": DATA_FIELD.FIELD_CLOSE,
}

cdef list GetColumnNameFromFieldList(str fields):
    return [_dict[x] for x in fields.split(",")]

cdef class CCXT(CCommonStockApi):
    cdef:
        bint is_connect

    def __cinit__(self, str code, object k_type=KL_TYPE.K_DAY, object begin_date=None, object end_date=None, object autype=AUTYPE.QFQ):
        super().__init__(code, k_type, begin_date, end_date, autype)

    cpdef get_kl_data(self):
        cdef:
            str fields = "time,open,high,low,close"
            object exchange = ccxt.binance()
            str timeframe = self.__convert_type()
            object since_date = exchange.parse8601(f'{self.begin_date}T00:00:00')
            list data = exchange.fetch_ohlcv(self.code, timeframe, since=since_date)
            list item_data
            object time_obj
            str time_str

        for item in data:
            time_obj = datetime.fromtimestamp(item[0] / 1000)
            time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                item[1],
                item[2],
                item[3],
                item[4]
            ]
            yield CKLine_Unit(self.create_item_dict(item_data, GetColumnNameFromFieldList(fields)), autofix=True)

    cpdef SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    cdef str __convert_type(self):
        cdef dict _dict = {
            KL_TYPE.K_DAY: '1d',
            KL_TYPE.K_WEEK: '1w',
            KL_TYPE.K_MON: '1M',
            KL_TYPE.K_5M: '5m',
            KL_TYPE.K_15M: '15m',
            KL_TYPE.K_30M: '30m',
            KL_TYPE.K_60M: '1h',
        }
        return _dict[self.k_type]

    cdef object parse_time_column(self, str inp):
        cdef:
            int year, month, day, hour, minute
        if len(inp) == 10:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = minute = 0
        elif len(inp) == 17:
            year = int(inp[:4])
            month = int(inp[4:6])
            day = int(inp[6:8])
            hour = int(inp[8:10])
            minute = int(inp[10:12])
        elif len(inp) == 19:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = int(inp[11:13])
            minute = int(inp[14:16])
        else:
            raise ValueError(f"unknown time column from TradingView:{inp}")
        return CTime(year, month, day, hour, minute, auto=not kltype_lt_day(self.k_type))

    cdef dict create_item_dict(self, list data, list column_name):
        cdef:
            int i
            dict result = {}
        for i in range(len(data)):
            data[i] = self.parse_time_column(data[i]) if i == 0 else str2float(data[i])
        for i in range(len(column_name)):
            result[column_name[i]] = data[i]
        return result
