import json
from typing import Dict, TypedDict
from datetime import datetime, timedelta    
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

import os
import pandas as pd
from glob import glob
import re
import shutil

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

def split_dataframe(df, freq):
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
        all_splits[freq] = split_dataframe(freq_df, freq)

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

def merge_csv_files(directory):
    # 获取目录下所有的子目录
    subdirs = glob(os.path.join(directory, '*_*_*_save_*_*'))
    
    # 创建一个字典来存储不同时间周期和文件类型的数据框
    timeframe_file_dfs = {}
    
    for subdir in subdirs:
        # 从目录名中提取信息
        dir_name = os.path.basename(subdir)
        match = re.match(r'(.+)_(\w+)_(\d+)_save_(\d+)_(\d+)', dir_name)
        if match:
            symbol, timeframe, segment, start_time, end_time = match.groups()
        else:
            print(f"警告：无法解析目录名 {dir_name}")
            continue
        
        # 获取子目录中的所有CSV文件
        csv_files = glob(os.path.join(subdir, '*.csv'))
        
        for file in csv_files:
            file_name = os.path.basename(file)
            
            # 读取CSV文件
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            key = (symbol, timeframe, file_name)
            if key not in timeframe_file_dfs:
                timeframe_file_dfs[key] = []
            
            timeframe_file_dfs[key].append((int(segment), df))
    
    # 合并并保存每个时间周期和文件类型的数据
    for (symbol, timeframe, file_name), dfs in timeframe_file_dfs.items():
        # 按段号排序
        sorted_dfs = sorted(dfs, key=lambda x: x[0])
        sorted_dfs = [df for _, df in sorted_dfs]
        
        if 'timestamp' in sorted_dfs[0].columns:
            merged_df = pd.DataFrame()
            for i, df in enumerate(sorted_dfs):
                if 'is_sure' in df.columns:
                    if i == 0:  # 第一段
                        df = df.loc[~(df['is_sure'] == False).cumsum().astype(bool)]
                    elif i == len(sorted_dfs) - 1:  # 最后一段
                        df = df.loc[df['is_sure'].cumsum() > 0]
                    else:  # 中间段
                        df = df[df['is_sure']]
                
                if not merged_df.empty:
                    # 验证重叠部分
                    overlap = pd.merge(merged_df, df, on=list(df.columns))
                    if not overlap.empty and not overlap.equals(merged_df.tail(len(overlap))):
                        print(f"警告：在 {symbol}_{timeframe}_{file_name} 中发现不一致的重叠部分")
                    
                    # 移除重叠部分
                    df = df[~df['timestamp'].isin(merged_df['timestamp'])]
                
                merged_df = pd.concat([merged_df, df])
            
            # 最终排序和去重
            merged_df.sort_values('timestamp', inplace=True)
            merged_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        else:
            # 如果没有timestamp列，直接合并所有数据框
            merged_df = pd.concat(sorted_dfs, ignore_index=True)
            merged_df.drop_duplicates(inplace=True)
        
        # 保存合并后的CSV文件
        output_file = os.path.join(directory, f'{symbol}_merged_{timeframe}_{file_name}')
        merged_df.to_csv(output_file, index=False)
        print(f'已保存合并文件：{output_file}')


def create_test_data(base_dir):
    # 创建测试目录结构
    symbol = 'EURUSD'
    timeframe = '5min'
    for i in range(3):
        dir_name = f"{symbol}_{timeframe}_{i}_save_20230101000000_20230101235959"
        os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        
        # 创建测试CSV文件
        start_time = datetime(2023, 1, 1) + timedelta(hours=i*8)
        end_time = start_time + timedelta(hours=10)
        times = pd.date_range(start_time, end_time, freq='5min')
        
        df = pd.DataFrame({
            'timestamp': times,
            'value': range(len(times)),
            'is_sure': [True] * (len(times) - 60) + [False] * 60
        })
        
        df.to_csv(os.path.join(base_dir, dir_name, 'test.csv'), index=False)

def test_merge_csv_files():
    # 创建临时测试目录
    test_dir = 'test_merge_csv'
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # 创建测试数据
        #create_test_data(test_dir)
        
        # 运行合并函数
        merge_csv_files('/opt/data')
        
        # 验证结果
        result_file = os.path.join(test_dir, 'EURUSD_merged_5min_test.csv')
        if not os.path.exists(result_file):
            print("测试失败：未生成合并文件")
            return
        
        result_df = pd.read_csv(result_file)
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        
        # 检查时间范围
        expected_start = datetime(2023, 1, 1)
        expected_end = datetime(2023, 1, 2, 1, 55)
        actual_start = result_df['timestamp'].min()
        actual_end = result_df['timestamp'].max()
        
        if actual_start != expected_start or actual_end != expected_end:
            print(f"测试失败：时间范围不正确。预期：{expected_start} 到 {expected_end}，实际：{actual_start} 到 {actual_end}")
            return
        
        # 检查数据点数量
        expected_count = 312  # (26 hours - 2 hours of overlap) / 5 minutes
        if len(result_df) != expected_count:
            print(f"测试失败：数据点数量不正确。预期：{expected_count}，实际：{len(result_df)}")
            return
        
        # 检查 is_sure 字段处理
        if result_df['is_sure'].iloc[0] != True or result_df['is_sure'].iloc[-1] != True:
            print("测试失败：is_sure 字段处理不正确")
            return
        
        print("测试通过：merge_csv_files 函数工作正常")
    
    finally:
        # 清理测试目录
        shutil.rmtree(test_dir)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '/opt/data'


    symbols = get_all_symbols(directory)
    print(symbols)
    for symbol in symbols:
        symbol_directory = f'{directory}/{symbol}'
        parse_symbol(symbol_directory)
    #    merge_csv_files(symbol_directory)
