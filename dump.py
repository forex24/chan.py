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
import json

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
    #directory = f"{symbol}_save_{start.strftime('%Y%m%d%H%M%S')}_{end.strftime('%Y%m%d%H%M%S')}"
    directory = f"{symbol}"
    directory = os.path.join(output_dir, directory)
    ensure_directory_exists(directory)
    chan[0].to_csv(directory)  # 使用新的输出目录


def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def split_dataframe(df, freq):
    split_dfs = []
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
        
        timeframe=['1m']
        all_splits = {}
        for freq in timeframe:
            all_splits[freq] = split_dataframe(df, freq)


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
                
                tasks.append(delayed(label)(split_df, f"{symbol}_{freq}", start, end, symbol_freq_dir))

        multi_work(tasks)

    except Exception as e:
        logging.error(f"Error processing symbol {symbol}: {str(e)}")

def dump_config(config: CChanConfig) -> str:
    """
    Dumps the config content as JSON
    
    Args:
        config: CChanConfig instance
    
    Returns:
        str: JSON formatted string of the config
    """
    def convert_value(v):
        """Helper function to convert values to JSON serializable format"""
        if hasattr(v, 'name'):  # Handle enum types
            return v.name
        elif isinstance(v, (str, int, float, bool, type(None))):
            return v
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [convert_value(item) for item in v]
        return str(v)  # Convert any other types to string

    config_dict = {
        "bi_conf": {
            "bi_algo": config.bi_conf.bi_algo,
            "is_strict": config.bi_conf.is_strict,
            "gap_as_kl": config.bi_conf.gap_as_kl,
            "bi_end_is_peak": config.bi_conf.bi_end_is_peak,
            "bi_allow_sub_peak": config.bi_conf.bi_allow_sub_peak,
            "bi_fx_check": convert_value(config.bi_conf.bi_fx_check)
        },
        "seg_conf": {
            "seg_algo": config.seg_conf.seg_algo,
            "left_method": convert_value(config.seg_conf.left_method)
        },
        "zs_conf": {k: convert_value(v) for k, v in vars(config.zs_conf).items()},
        "bs_point_conf": {
            "b_conf": {k: convert_value(v) for k, v in vars(config.bs_point_conf.b_conf).items()} if hasattr(config.bs_point_conf, 'b_conf') else {},
            "s_conf": {k: convert_value(v) for k, v in vars(config.bs_point_conf.s_conf).items()} if hasattr(config.bs_point_conf, 's_conf') else {}
        },
        "seg_bs_point_conf": {
            "b_conf": {k: convert_value(v) for k, v in vars(config.seg_bs_point_conf.b_conf).items()} if hasattr(config.seg_bs_point_conf, 'b_conf') else {},
            "s_conf": {k: convert_value(v) for k, v in vars(config.seg_bs_point_conf.s_conf).items()} if hasattr(config.seg_bs_point_conf, 's_conf') else {}
        },
    }
    
    # Remove any non-serializable objects
    for section_name, section in config_dict.items():
        if isinstance(section, dict):
            for key in list(section.keys()):
                if not isinstance(section[key], (str, int, float, bool, list, dict, type(None))):
                    del section[key]
    
    return json.dumps(config_dict, indent=2, ensure_ascii=False)

def export_config_to_json2(config: CChanConfig, output_path: str):
    """
    将配置导出为JSON文件
    Args:
        config: CChanConfig对象
        output_path: JSON文件保存路径
    """
    import json
    try:
        json_str = dump_config(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json.loads(json_str), f, indent=4, ensure_ascii=False)
        logging.info(f"配置已成功导出到: {output_path}")
    except Exception as e:
        logging.error(f"导出配置失败: {str(e)}")

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
    dump_config(config)


    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--root', default='/opt/data', help='Root directory for data (default: /opt/data)')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    parser.add_argument('--config', help='Path to config JSON file (optional)', default=config_path)
    
    args = parser.parse_args()

    root_directory = args.root
    input_directory = os.path.join(root_directory, 'raw_data')
    output_directory = os.path.join(root_directory, 'split_data')
    ensure_directory_exists(output_directory)

    if args.symbol:
        parse_symbol(args.symbol, input_directory, output_directory)
        config_path = os.path.join(output_directory, 'chan_config.json')
        export_config_to_json(config, config_path)
        config_path = os.path.join(output_directory, 'chan_config2.json')
        export_config_to_json(config, config_path)
    else:
        symbols = get_all_symbols(input_directory)
        print(symbols)
        for symbol in symbols:
            parse_symbol(symbol, input_directory, output_directory)







