import pandas as pd

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else float(data[i])
    return dict(zip(column_name, data))


def parse_time_column(inp):
    return CTime(inp.year, inp.month, inp.day, inp.hour, inp.minute)


class DATAFRAME_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
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
        super(DATAFRAME_API, self).__init__(code, k_type, begin_date, end_date, autype)
        self.begin_date = pd.to_datetime(begin_date) if begin_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
    def get_kl_data(self):
        if self.df is None:
            raise CChanException("DataFrame is not provided", ErrCode.SRC_DATA_NOT_FOUND)

        for _, row in self.df.iterrows():
            data = row.tolist()
            if len(data) != len(self.columns):
                raise CChanException("DataFrame format error", ErrCode.SRC_DATA_FORMAT_ERROR)
            if self.begin_date is not None and data[self.time_column_idx] < self.begin_date:
                continue
            if self.end_date is not None and data[self.time_column_idx] > self.end_date:
                continue
            yield CKLine_Unit(create_item_dict(data, self.columns))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass