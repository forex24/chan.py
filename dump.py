"""
数据处理：
1. 数据分割处理
2. 多周期数据处理
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

def get_all_symbols(directory):
    symbols = set()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # 移除 .csv 后缀
            name_without_extension = filename[:-4]
            # 分割文件名取第一部分作为符号名
            symbol = name_without_extension.split('_')[0]
            symbols.add(symbol)
    return symbols


def chan_lab(df, symbol, output_dir):
    """
    对输入数据打标策略产出的买卖点的特征
    start向前减一个月,end向后加一个月
    用于数据warmup
    """
    #print(df.head(40))
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
    
    df_resampled.drop_duplicates(inplace=True)
    df_resampled.reset_index(inplace=True)
    df_resampled.sort_values(by='timestamp',inplace=True)
    return df_resampled

def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    #df.set_index('timestamp', inplace=True)
    #df.drop_duplicates(inplace=True)
    #df.reset_index(inplace=True)
    #df.sort_values(by='timestamp',inplace=True)
    return df

def split_dataframe(df, freq):
    #length = len(df)
    #overlap_size = int(DEFAULT_MAX_COUNT * OVERLAP_RATIO)
    #step_size = DEFAULT_MAX_COUNT - overlap_size

    #print(f'freq: {freq}, length: {length}, splits: {(length - overlap_size) // step_size + 1}')

    #start_idx = 0
    #split_num = 0
    split_dfs = []

    #while start_idx < length:
    #    end_idx = min(start_idx + DEFAULT_MAX_COUNT, length)
    #    split_df = df.iloc[start_idx:end_idx].copy()
    #    split_dfs.append(split_df)

    #    start_idx += step_size
    #    split_num += 1

    # 检查最后一个分割是否小于最小大小
    #if len(split_dfs[-1]) < DEFAULT_MIN_COUNT and len(split_dfs) > 1:
    #    # 将最后一个分割合并到前一个
    #    split_dfs[-2] = pd.concat([split_dfs[-2], split_dfs[-1]])
    #    split_dfs.pop()
    #    split_num -= 1

    # 打印每个分割的信息
    #for i, split_df in enumerate(split_dfs):
    #    print(f"Split {i} for {freq}, rows: {len(split_df)}")
    
    #logging.info(f"Total splits for {freq}: {split_num}")
    split_dfs.append(df)

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
        
        timeframe=['1min']
        all_splits = {}
        for freq in timeframe:
            #freq_df = resample_timeframe(df, freq)
            #freq_df.to_csv(os.path.join(output_directory, f'{symbol}_{freq}.csv'))
            all_splits[freq] = split_dataframe(df, freq)
            #for index, check_df in enumerate(all_splits[freq]):
            #    check_df.to_csv(os.path.join(output_directory, f'{symbol}_{freq}_{index}.csv'))

        n_cores = cpu_count()
        logging.info(f'系统的核心数是：{n_cores}')
        multi_work = Parallel(n_jobs=n_cores, backend='loky')
        tasks = []

        for freq, splits in all_splits.items():
            for i, split_df in enumerate(splits):
                if len(split_df) == 0:
                    logging.warning(f"警告：{symbol}_{freq}_{i} 的数据框为空")
                    continue
                split_df.reset_index(inplace=True)
                start = split_df['timestamp'].iloc[0]
                end = split_df['timestamp'].iloc[-1]
                
                symbol_freq_dir = os.path.join(output_directory, symbol)
                ensure_directory_exists(symbol_freq_dir)

                #print(split_df.head(40))
                
                tasks.append(delayed(label)(split_df, f"{symbol}_{freq}_{i}", start, end, symbol_freq_dir))
                #label(split_df, f"{symbol}_{freq}_{i}", start, end, symbol_freq_dir)

        multi_work(tasks)

    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {str(e)}")

def export_config_to_json(config: CChanConfig, output_path: str):
    """
    将配置导出为JSON文件
    Args:
        config: CChanConfig对象
        output_path: JSON文件保存路径
    """
    import json
    try:
        json_str = config.to_json()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(json_str), f, indent=4, ensure_ascii=False)
        logging.info(f"配置已成功导出到: {output_path}")
    except Exception as e:
        logging.error(f"导出配置失败: {str(e)}")

def load_config_from_json(json_path: str) -> CChanConfig:
    """
    从JSON文件加载配置
    Args:
        json_path: JSON配置文件路径
    Returns:
        CChanConfig对象
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return CChanConfig.from_json(json_str)
    except Exception as e:
        logging.error(f"加载配置失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 创建默认配置
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

    # 导出配置到JSON文件
    config_path = os.path.join(os.path.dirname(__file__), 'chan_config.json')
    export_config_to_json(config, config_path)


    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--root', default='/opt/data', help='Root directory for data (default: /opt/data)')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    parser.add_argument('--config', help='Path to config JSON file (optional)', default=config_path)
    
    args = parser.parse_args()

    # 如果提供了配置文件，则使用提供的配置
    if args.config and args.config != config_path:
        config = load_config_from_json(args.config)

    root_directory = args.root
    input_directory = os.path.join(root_directory, 'raw_data')
    output_directory = os.path.join(root_directory, 'split_data')
    ensure_directory_exists(output_directory)

    if args.symbol:
        parse_symbol(args.symbol, input_directory, output_directory)
    else:
        symbols = get_all_symbols(input_directory)
        print(symbols)
        for symbol in symbols:
            parse_symbol(symbol, input_directory, output_directory)







