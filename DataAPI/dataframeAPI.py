import os
from datetime import datetime

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit
import pandas as pd
from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fields: str):
    _dict = {
        "timestamp": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
    }
    return [_dict[x] for x in fields.split(",")]


def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if i == 0 else str2float(data[i])
    return dict(zip(column_name, data))
    
def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
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
        raise Exception(f"unknown time column from csv:{inp}")
    return CTime(year, month, day, hour, minute)


class DATAFRAME_API(CCommonStockApi):
    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            DATA_FIELD.FIELD_VOLUME,
        ] 
        super(DATAFRAME_API, self).__init__(code, k_type, begin_date, end_date, autype)
        self.df = code
        self.begin_date = pd.to_datetime(begin_date) if begin_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

    def get_kl_data(self):
        #cur_path = os.path.dirname(os.path.realpath(__file__))
        #file_path = f"{cur_path}/{self.code}.parquet"  # 文件路径
        #if not os.path.exists(file_path):
        #    raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)
        
        fields = "timestamp,open,high,low,close"
        #df = pd.read_parquet(file_path, engine='fastparquet')
        
        # 确保 DataFrame 的索引是 datetime 类型
        self.df.index = pd.to_datetime(self.df.index)
        
        for index, row in self.df.iterrows():
            #print("index type:", type(index))
            #print("begin type:", type(self.begin_date))
            time_obj = index  # 假设时间列是第一列
            if self.begin_date is not None and time_obj < self.begin_date:
                continue
            if self.end_date is not None and time_obj > self.end_date:
                continue
            time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                row['open'],
                row['high'],
                row['low'],
                row['close']
            ]
            yield CKLine_Unit(create_item_dict(item_data, GetColumnNameFromFieldList(fields)))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass
