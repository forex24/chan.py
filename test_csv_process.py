import json
from typing import Dict, TypedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import sys
import os

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.PlotDriver import CPlotDriver
from pathlib import Path

DEFAULT_MAX_COUNT = 400000
DEFAULT_MIN_COUNT = 100000  # 设置最小分割大小
OVERLAP_RATIO = 0.2  # 20% 重叠

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_all_symbols(directory):
        # 查找目录下有多少个不同的 symbol
    symbols = set()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            symbol = filename.split('_')[0]
            symbols.add(symbol)
    return symbols


def chan_lab(df, symbol):
    """
    对输入数据打标策略产出的买卖点的特征
    start向前减一个月,end向后加一个月
    用于数据warmup
    """
    #print(df.info())
    #print('symbol:', symbol)
    code = df
    begin_time = None
    end_time = None
    data_src = DATA_SRC.DATAFRAME
    lv_list = [KL_TYPE.K_1M]

    config = CChanConfig({
        "trigger_step": True,  # 打开开关！
        "bi_strict": True,
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 1,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    last_bsp_list=[]
    # 跑策略，保存买卖点的特征
    for chan_snapshot in chan.step_load():
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_bsp()
        if not bsp_list:
            continue
        last_bsp = bsp_list[-1]

        if len(last_bsp_list) != 0:
            last = last_bsp_list[-1]
            if last is not None:
                if last_bsp.klu.time == last.klu.time:
                    continue
        last_bsp_list.append(last_bsp)

    last_bsp_df = pd.DataFrame([
                {
                    'is_buy': bsp.is_buy,
                    'bsp_type': bsp.type2str(),
                    'bi_idx': bsp.bi.idx if bsp.bi else None,
                    'time': bsp.klu.time,
                } for bsp in last_bsp_list
            ])
    start = pd.to_datetime(df['timestamp'].iloc[0])
    end = pd.to_datetime(df['timestamp'].iloc[-1])
    directory = f"{symbol}_save_{start.strftime('%Y%m%d%H%M%S')}_{end.strftime('%Y%m%d%H%M%S')}"
    mkdir_p(directory)
    last_bsp_df.to_csv(os.path.join(directory, "last_bsp.csv"), index=False)
    chan[0].to_csv(directory)


def resample_timeframe(df, timeframe):
    df = df.set_index('timestamp')
    df_resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    df_resampled.dropna(inplace=True)
    return df_resampled

def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.sort_values(by='timestamp',inplace=True)
    return df

def split_df(df, freq):
    length = len(df)
    overlap_size = int(DEFAULT_MAX_COUNT * OVERLAP_RATIO)
    step_size = DEFAULT_MAX_COUNT - overlap_size

    print(f'freq: {freq}, length: {length}, splits: {(length - overlap_size) // step_size + 1}')

    start_idx = 0
    split_num = 0
    split_dfs = []

    while start_idx < length:
        end_idx = min(start_idx + DEFAULT_MAX_COUNT, length)
        split_df = df.iloc[start_idx:end_idx].copy()
        split_dfs.append(split_df)

        start_idx += step_size
        split_num += 1

    # 检查最后一个分割是否小于最小大小
    if len(split_dfs[-1]) < DEFAULT_MIN_COUNT and len(split_dfs) > 1:
        # 将最后一个分割合并到前一个
        split_dfs[-2] = pd.concat([split_dfs[-2], split_dfs[-1]])
        split_dfs.pop()
        split_num -= 1

    # 打印每个分割的信息
    for i, split_df in enumerate(split_dfs):
        print(f"Split {i} for {freq}, rows: {len(split_df)}")

    print(f"Total splits for {freq}: {split_num}")


    return split_dfs

def process_split(split_df, freq, i, symbol):
    start = split_df['timestamp'].iloc[0]
    end = split_df['timestamp'].iloc[-1]
    symbol_info = f"{symbol}_{freq}_{i}"
    print(f"Processing {symbol_info} from {start} to {end}")
    return chan_lab(split_df, symbol_info)

def label(df, symbol, start, end):
    try:
        chan_lab(df, symbol)
    except Exception as e:
        print(f"Error processing {symbol} from {start} to {end}: {e}")

def parse_symbol(symbol):
    df = load_csv(symbol)
    timeframe=['1d','8h','6h','4h','2h','1h','30min','15min','5min','3min', '1min']
    #timeframe=['1min']
    all_splits = {}
    for freq in timeframe:
        freq_df = resample_timeframe(df, freq)
        all_splits[freq] = split_df(freq_df, freq)

    # 现在 all_splits 是一个字典，键是时间频率，值是该频率下的 DataFrame 列表
    # 您可以根据需要使用这些分割后的 DataFrame
    for freq, splits in all_splits.items():
        print(f"Frequency: {freq}, Number of splits: {len(splits)}")

    n_cores = cpu_count()
    print(f'系统的核心数是：{n_cores}')
    multi_work = Parallel(n_jobs=n_cores, backend='loky')
    tasks = []

    # 如果您需要对每个分割进行处理，可以使用以下代码：
    for freq, splits in all_splits.items():
        for i, split_df in enumerate(splits):
            #print(split_df.head())
            split_df.reset_index(inplace=True)
            #print(split_df.head())
            start = split_df['timestamp'].iloc[0]
            end = split_df['timestamp'].iloc[-1]
            print(f"Processing {freq} split {i} from {start} to {end}")
            # 在这里调用 chan_lab 函数
            #chan_lab(split_df, f"{symbol}_{freq}_{i}")
            #split_df['timetamp'] = str(split_df['timestamp'])
            print(split_df.info())
            #label(split_df, f"{symbol}_{freq}_{i}", start, end)
            tasks.append(delayed(label)(split_df, f"{symbol}_{freq}_{i}", start, end))

    res = multi_work(tasks)
    print(res)



if __name__ == "__main__":
    # 获取命令行参数，默认为 'eurusd'
    symbols = get_all_symbols('/opt/data')
    for symbol in symbols:
        parse_symbol(symbol)

