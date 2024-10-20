# cython: language_level=3
import baostock as bs
from libc.stdlib cimport malloc, free

from Common.CEnum cimport AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime cimport CTime
from Common.func_util cimport kltype_lt_day, str2float
from KLine.KLine_Unit cimport CKLine_Unit

from .CommonStockAPI cimport CCommonStockApi

cdef dict _dict = {
    "time": DATA_FIELD.FIELD_TIME,
    "date": DATA_FIELD.FIELD_TIME,
    "open": DATA_FIELD.FIELD_OPEN,
    "high": DATA_FIELD.FIELD_HIGH,
    "low": DATA_FIELD.FIELD_LOW,
    "close": DATA_FIELD.FIELD_CLOSE,
    "volume": DATA_FIELD.FIELD_VOLUME,
    "amount": DATA_FIELD.FIELD_TURNOVER,
    "turn": DATA_FIELD.FIELD_TURNRATE,
}

cdef object create_item_dict(list data, list column_name):
    cdef:
        int i
        dict result = {}
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    for i in range(len(column_name)):
        result[column_name[i]] = data[i]
    return result

cdef object parse_time_column(str inp):
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
        raise ValueError(f"unknown time column from baostock:{inp}")
    return CTime(year, month, day, hour, minute)

cdef list GetColumnNameFromFieldList(str fields):
    return [_dict[x] for x in fields.split(",")]

cdef class CBaoStock(CCommonStockApi):
    cdef:
        bint is_connect

    def __cinit__(self, str code, object k_type=KL_TYPE.K_DAY, object begin_date=None, object end_date=None, object autype=AUTYPE.QFQ):
        super().__init__(code, k_type, begin_date, end_date, autype)

    cpdef get_kl_data(self):
        cdef:
            str fields
            dict autype_dict
            object rs
            list row_data
            dict item_dict
        
        if kltype_lt_day(self.k_type):
            if not self.is_stock:
                raise ValueError("没有获取到数据，注意指数是没有分钟级别数据的！")
            fields = "time,open,high,low,close"
        else:
            fields = "date,open,high,low,close,volume,amount,turn"
        
        autype_dict = {AUTYPE.QFQ: "2", AUTYPE.HFQ: "1", AUTYPE.NONE: "3"}
        rs = bs.query_history_k_data_plus(
            code=self.code,
            fields=fields,
            start_date=self.begin_date,
            end_date=self.end_date,
            frequency=self.__convert_type(),
            adjustflag=autype_dict[self.autype],
        )
        if rs.error_code != '0':
            raise ValueError(rs.error_msg)
        while rs.error_code == '0' and rs.next():
            row_data = rs.get_row_data()
            item_dict = create_item_dict(row_data, GetColumnNameFromFieldList(fields))
            yield CKLine_Unit(item_dict)

    cpdef SetBasciInfo(self):
        cdef:
            object rs
            list row_data
        rs = bs.query_stock_basic(code=self.code)
        if rs.error_code != '0':
            raise ValueError(rs.error_msg)
        row_data = rs.get_row_data()
        self.name = row_data[1]
        self.is_stock = (row_data[4] == '1')

    @classmethod
    def do_init(cls):
        if not cls.is_connect:
            cls.is_connect = bs.login()

    @classmethod
    def do_close(cls):
        if cls.is_connect:
            bs.logout()
            cls.is_connect = False

    cdef str __convert_type(self):
        cdef dict _dict = {
            KL_TYPE.K_DAY: 'd',
            KL_TYPE.K_WEEK: 'w',
            KL_TYPE.K_MON: 'm',
            KL_TYPE.K_5M: '5',
            KL_TYPE.K_15M: '15',
            KL_TYPE.K_30M: '30',
            KL_TYPE.K_60M: '60',
        }
        return _dict[self.k_type]
