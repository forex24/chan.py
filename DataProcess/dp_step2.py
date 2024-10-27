
import argparse
import logging
import os
import pandas as pd
from glob import glob
import re
import numpy as np
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.WARN, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义每种文件类型需要比较的字段
FILE_TYPE_FIELDS = {
    'bi_list.csv': ['begin_time', 'end_time', 'dir', 'high', 'low'],
    'bs_point_lst.csv': ['begin_time', 'bsp_type'],
    'kline_list.csv': ['begin_time', 'end_time', 'dir', 'high', 'low', 'fx'],
    'bs_point_history.csv': ['is_buy', 'bsp_type', 'begin_time'],
    'seg_bs_point_history.csv': ['is_buy', 'bsp_type', 'begin_time'],
    'seg_bs_point_lst.csv': ['begin_time', 'bsp_type'],
    'seg_list.csv': ['begin_time', 'end_time', 'dir', 'high', 'low'],
    'segseg_list.csv': ['begin_time', 'end_time', 'dir', 'high', 'low'],
    'segzs_list.csv': ['begin_time', 'end_time', 'begin_bi_time', 'end_bi_time', 'bi_in_time', 'bi_out_time'],
    'zs_list.csv': ['begin_time', 'end_time', 'begin_bi_time', 'end_bi_time', 'bi_in_time', 'bi_out_time'],
}


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

def get_comparison_fields(file_name):
    for key, fields in FILE_TYPE_FIELDS.items():
        if key in file_name:
            return fields
    return None  # 如果没有匹配的文件类型，返回None

def compare_dataframes(df1, df2, comparison_fields=None):
    if comparison_fields:
        # 只选择两个DataFrame中都存在的列
        common_fields = [field for field in comparison_fields if field in df1.columns and field in df2.columns]
        if not common_fields:
            logging.warning(f"警告：没有找到共同的比较字段。df1列：{df1.columns.tolist()}，df2列：{df2.columns.tolist()}")
            return False
        df1 = df1[common_fields]
        df2 = df2[common_fields]
    else:
        # 如果没有指定比较字段，使用两个DataFrame的共同列
        common_fields = df1.columns.intersection(df2.columns)
        df1 = df1[common_fields]
        df2 = df2[common_fields]
    
    return df1.equals(df2)

def output_differences(df1, df2, name1, name2, comparison_fields, output_dir):
    logging.info(f"比较 {name1} 和 {name2}")
    logging.info(f"{name1} 大小: {df1.shape}, {name2} 大小: {df2.shape}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 只比较两个DataFrame中都存在的指定字段
    if comparison_fields:
        common_fields = [field for field in comparison_fields if field in df1.columns and field in df2.columns]
    else:
        common_fields = df1.columns.intersection(df2.columns)
    
    df1 = df1[common_fields].copy()
    df2 = df2[common_fields].copy()
    
    # 确保索引是相同的
    common_index = df1.index.intersection(df2.index)
    df1 = df1.loc[common_index]
    df2 = df2.loc[common_index]
    
    # 找出不一致的行
    diff_mask = ~(df1 == df2).all(axis=1)
    
    if diff_mask.any():
        diff_count = diff_mask.sum()
        logging.info(f"\n发现 {diff_count} 行不一致")
        
        # 创建输出文件
        output_file = os.path.join(output_dir, f"diff_{name1}_{name2}.txt")
        with open(output_file, 'w') as f:
            f.write(f"比较 {name1} 和 {name2}\n")
            f.write(f"{name1} 大小: {df1.shape}, {name2} 大小: {df2.shape}\n")
            f.write(f"比较的字段: {', '.join(common_fields)}\n")
            f.write(f"不一致的行数: {diff_count}\n\n")
            
            for idx in diff_mask[diff_mask].index:
                f.write(f"时间 {idx}:\n")
                f.write(f"{name1} 数据:\n")
                f.write(df1.loc[idx].to_string() + "\n")
                f.write(f"\n{name2} 数据:\n")
                f.write(df2.loc[idx].to_string() + "\n")
                f.write("\n差异:\n")
                diff = df1.loc[idx] != df2.loc[idx]
                f.write(diff[diff].to_string() + "\n")
                f.write("-" * 40 + "\n")
        
        logging.warning(f"详细的差异信息已写入文件: {output_file}")
    else:
        logging.info("未发现差异。")
    
    logging.info("\n" + "="*50 + "\n")  # 分隔线

def get_time_column(df):
    """
    确定DataFrame中使用的时间列名称。
    
    :param df: DataFrame
    :return: 时间列名称，如果没有找到则返回None
    """
    time_columns = ['timestamp', 'begin_time', 'time']
    for col in time_columns:
        if col in df.columns:
            return col
    return None

def remove_all_nan_rows(df):
    """
    删除DataFrame中全是NaN的行
    """
    return df.dropna(how='all')

def clean_dataframe(df):
    """
    清理DataFrame：删除全是NaN的行，并将剩余的NaN值替换为None
    """
    df = df.dropna(how='all')
    return df.where(pd.notnull(df), None)

def merge_csv_files(input_directory, output_directory):
    # 获取目录下所有的子目录
    subdirs = glob(os.path.join(input_directory, '*_*_*_save_*_*'))
    
    # 创建一个字典来存储不同时间周期和文件类型的数据框
    timeframe_file_dfs = {}
    
    for subdir in subdirs:
        # 从目录名中提取信息
        dir_name = os.path.basename(subdir)
        match = re.match(r'(.+)_(\w+)_(\d+)_save_(\d+)_(\d+)', dir_name)
        if match:
            symbol, timeframe, segment, start_time, end_time = match.groups()
        else:
            logging.warning(f"警告：无法解析目录名 {dir_name}")
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
        logging.info(f"处理 {symbol}_{timeframe}_{file_name}")
        
        comparison_fields = get_comparison_fields(file_name)
        if comparison_fields is None:
            logging.warning(f"警告：未找到 {file_name} 的比较字段配置，将比较所有字段")
        
        sorted_dfs = sorted(dfs, key=lambda x: x[0])

        merged_df = pd.DataFrame()
        for i, (file_path, df) in enumerate(sorted_dfs):
            df = clean_dataframe(df)
            
            time_col = get_time_column(df)
            if time_col is None:
                logging.warning(f"警告：在文件 {file_path} 中未找到时间列，跳过此文件")
                continue
            
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.dropna(subset=[time_col])
            df = df.set_index(time_col).sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            if merged_df.empty:
                merged_df = df
                continue

            # 找到初始重叠部分
            overlap_start = max(merged_df.index.min(), df.index.min())
            overlap_end = min(merged_df.index.max(), df.index.max())
            
            # 计算需要删除的时间范围
            overlap_duration = overlap_end - overlap_start
            remove_duration = overlap_duration / 4

            # 调整重叠区间
            new_overlap_start = overlap_start + remove_duration
            new_overlap_end = overlap_end - remove_duration

            logging.info(f"初始重叠部分：从 {overlap_start} 到 {overlap_end}")
            logging.info(f"调整后重叠部分：从 {new_overlap_start} 到 {new_overlap_end}")
            
            # 截取调整后的重叠部分
            overlap_merged = merged_df.loc[new_overlap_start:new_overlap_end]
            overlap_df = df.loc[new_overlap_start:new_overlap_end]
            
            logging.info(f"调整后重叠部分大小：merged_df: {overlap_merged.shape}, df: {overlap_df.shape}")
            
            # 验证重叠部分
            if not compare_dataframes(overlap_merged, overlap_df, comparison_fields):
                segment = i
                name1 = f"{symbol}_{timeframe}_{file_name}_merged_data"
                name2 = f"{symbol}_{timeframe}_{file_name}_segment_{segment}"
                logging.warning(f"警告：在 {symbol}_{timeframe}_{file_name} 的第 {segment} 段中发现不一致的重叠部分")
                output_differences(
                    overlap_merged, 
                    overlap_df, 
                    name1, 
                    name2, 
                    comparison_fields,
                    output_dir=os.path.join(output_directory, "overlap_diff_output")
                )
            
            # 合并非重叠部分和调整后的重叠部分
            merged_df = pd.concat([merged_df[merged_df.index < new_overlap_start], 
                                   overlap_merged,
                                   df[df.index > new_overlap_end]])

        # 最终排序和去重
        merged_df = merged_df.sort_index()
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
        merged_df = clean_dataframe(merged_df)
        merged_df.reset_index(inplace=True)

        # 保存合并后的CSV文件
        output_file = os.path.join(output_directory, f'{symbol}_merged_{timeframe}_{file_name}')
        merged_df.to_csv(output_file, index=False)
        logging.info(f'已保存合并文件：{output_file}')

def parse_time(time_str):
    """
    解析时间字符串，返回datetime对象
    """
    try:
        return pd.to_datetime(time_str)
    except:
        logging.warning(f"无法解析时间: {time_str}")
        return None

def merge_single_symbol_csv_files(input_directory, output_directory, symbol):
    input_directory = os.path.join(input_directory, symbol)
    output_directory = os.path.join(output_directory, symbol)
    ensure_directory_exists(output_directory)
    logging.info(f"输入目录: {input_directory}")
    logging.info(f"输出目录: {output_directory}")
    merge_csv_files(input_directory, output_directory)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--root', default='/opt/data', help='Root directory for data (default: /opt/data)')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    
    args = parser.parse_args()

    root_directory = args.root
    logging.info(f"root_directory: {root_directory}")
    input_directory = os.path.join(root_directory, 'split_data')
    output_directory = os.path.join(root_directory, 'merged_data')

    if args.symbol:
        logging.info(f"parse symbol: {args.symbol}")
        merge_single_symbol_csv_files(input_directory, output_directory, args.symbol)
    else:
        symbols = get_all_symbols(os.path.join(root_directory, 'raw_data'))
        logging.info(f"Symbols: {symbols}")
        for symbol in symbols:
            merge_single_symbol_csv_files(input_directory, output_directory, symbol)
