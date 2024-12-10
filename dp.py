"""
数据处理：
1. 数据分割处理
2. 合并数据
"""
import argparse
from typing import Dict, TypedDict
from datetime import datetime, timedelta    
from dateutil.relativedelta import relativedelta
from joblib import Parallel, delayed, cpu_count
import pandas as pd
import logging
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

DEFAULT_MAX_COUNT = 200000  # 设置最大分割大小
DEFAULT_MIN_COUNT = 10000  # 设置最小分割大小
OVERLAP_RATIO = 0.3  # 30% 重叠


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def chan_lab(df, symbol, output_dir):
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

    # 跑策略，保存买卖点的特征
    for _ in chan.step_load():
        continue
    start = pd.to_datetime(df['timestamp'].iloc[0])
    end = pd.to_datetime(df['timestamp'].iloc[-1])
    directory = f"{symbol}_save_{start.strftime('%Y%m%d%H%M%S')}_{end.strftime('%Y%m%d%H%M%S')}"
    directory = os.path.join(output_dir, directory)
    ensure_directory_exists(directory)
    chan[0].to_csv(directory)  # 使用新的输出目录

def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.sort_values(by='timestamp',inplace=True)
    return df

def split_dataframe(df):
    length = len(df)
    overlap_size = int(DEFAULT_MAX_COUNT * OVERLAP_RATIO)
    step_size = DEFAULT_MAX_COUNT - overlap_size

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
    #for i, split_df in enumerate(split_dfs):
    #    print(f"Split {i} for {freq}, rows: {len(split_df)}")
    
    logging.info(f"Total splits: {split_num}")


    return split_dfs

def label(df, symbol, start, end, output_dir):
    try:
        chan_lab(df, symbol, output_dir)
    except Exception as e:
        logging.error(f"Error processing {symbol} from {start} to {end}: {e}")

def parse_symbol(symbol, input_directory, output_directory):
    csv_file = f"{symbol}.csv"
    file_path = os.path.join(input_directory, csv_file)
    if not os.path.exists(file_path):
        logging.warning(f"No CSV file found for symbol {symbol}")
        return
    
    try:
        df = load_csv(file_path)
        
        all_splits = split_dataframe(df)

        n_cores = cpu_count()
        logging.info(f'系统的核心数是：{n_cores}')
        multi_work = Parallel(n_jobs=n_cores, backend='loky')
        tasks = []


        for i, split_df in enumerate(all_splits):
            if len(split_df) == 0:
                logging.warning(f"警告：{symbol}_{i} 的数据框为空")
                continue
            split_df.reset_index(inplace=True)
            start = split_df['timestamp'].iloc[0]
            end = split_df['timestamp'].iloc[-1]
            
            symbol_freq_dir = os.path.join(output_directory, symbol)
            ensure_directory_exists(symbol_freq_dir)
            
            tasks.append(delayed(label)(split_df, f"{symbol}_{i}", start, end, symbol_freq_dir))

        multi_work(tasks)

    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    
    args = parser.parse_args()

    root_directory = '/opt/data'
    input_directory = os.path.join(root_directory, 'raw_data')
    output_directory = os.path.join(root_directory, 'split_data')
    ensure_directory_exists(output_directory)


    if args.symbol:
        parse_symbol(args.symbol, input_directory, output_directory)






