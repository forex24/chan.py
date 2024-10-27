import os
import pandas as pd
import argparse
from typing import List, Dict, Tuple
import glob
import re

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

def get_symbols(directory: str) -> List[str]:
    """获取目录下的所有symbol（子目录名）"""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def load_csv(file_path: str) -> pd.DataFrame:
    """加载CSV文件返回DataFrame"""
    df = pd.read_csv(file_path)
    time_column = 'begin_time' if 'begin_time' in df.columns else 'time'
    df[time_column] = pd.to_datetime(df[time_column])
    return df

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, fields: List[str]) -> Tuple[bool, str]:
    """比较两个DataFrame的指定字段"""
    # 检查并报告缺失的列
    missing_columns_df1 = set(fields) - set(df1.columns)
    missing_columns_df2 = set(fields) - set(df2.columns)
    
    if missing_columns_df1 or missing_columns_df2:
        message = "Columns mismatch. "
        if missing_columns_df1:
            message += f"Missing in first file: {missing_columns_df1}. "
        if missing_columns_df2:
            message += f"Missing in second file: {missing_columns_df2}. "
        return False, message

    # 只使用两个数据框中都存在的列
    common_fields = list(set(fields) & set(df1.columns) & set(df2.columns))
    
    if not common_fields:
        return False, "No common fields to compare"

    df1_sorted = df1.sort_index()
    df2_sorted = df2.sort_index()

    common_index = df1_sorted.index.intersection(df2_sorted.index)
    
    if common_index.empty:
        return False, "No common indices to compare"

    df1_common = df1_sorted.loc[common_index, common_fields]
    df2_common = df2_sorted.loc[common_index, common_fields]

    is_equal = df1_common.equals(df2_common)

    if is_equal:
        return True, "Files are identical"
    else:
        # 找出不同的行
        diff_mask = (df1_common != df2_common).any(axis=1)
        diff_indices = diff_mask[diff_mask].index
        
        # 获取第一个不同的行的索引
        first_diff_index = diff_indices[0] if len(diff_indices) > 0 else None
        
        if first_diff_index is not None:
            diff_row1 = df1_common.loc[first_diff_index]
            diff_row2 = df2_common.loc[first_diff_index]
            diff_fields = [field for field in common_fields if diff_row1[field] != diff_row2[field]]
            
            message = f"Files differ. First difference at index {first_diff_index}. "
            message += f"Different fields: {diff_fields}. "
            message += f"Values in file1: {diff_row1[diff_fields].to_dict()}. "
            message += f"Values in file2: {diff_row2[diff_fields].to_dict()}."
        else:
            message = "Files differ, but no specific differences found. This might be due to NaN values or floating-point precision issues."
        
        return False, message

def compare_symbol_files(dir1: str, dir2: str, symbol: str):
    """比较单个symbol的所有文件"""
    symbol_dir1 = os.path.join(dir1, symbol)
    symbol_dir2 = os.path.join(dir2, symbol)
    
    files1 = glob.glob(os.path.join(symbol_dir1, f"{symbol}_merged_*.csv"))
    
    for file1 in files1:
        file_name = os.path.basename(file1)
        file2 = os.path.join(symbol_dir2, file_name)
        
        if not os.path.exists(file2):
            print(f"File {file_name} does not exist in the second directory")
            continue
        
        # 使用更新的正则表达式提取 file_type
        match = re.search(r'_merged_(?:\d+[dhm]|[1-9]\d*min)_(.+\.csv)$', file_name)
        if match:
            file_type = match.group(1)
        else:
            print(f"Unable to extract file type from {file_name}")
            continue
        
        if file_type not in FILE_TYPE_FIELDS:
            print(f"Unknown file type: {file_type}")
            continue
        
        fields = FILE_TYPE_FIELDS[file_type]
        
        try:
            df1 = load_csv(file1)
            df2 = load_csv(file2)
        except Exception as e:
            print(f"Error loading files for {file_name}: {str(e)}")
            continue

        try:
            is_same, message = compare_dataframes(df1, df2, fields)
            print(f"Comparing {file_name}: {message}")
        except Exception as e:
            print(f"Error comparing {file_name}: {str(e)}")
            print(f"Fields for this file type: {fields}")
            print(f"Columns in file1: {df1.columns.tolist()}")
            print(f"Columns in file2: {df2.columns.tolist()}")

def main(dir1: str, dir2: str):
    symbols1 = set(get_symbols(dir1))
    symbols2 = set(get_symbols(dir2))
    
    common_symbols = symbols1.intersection(symbols2)
    
    print(f"Found {len(common_symbols)} common symbols")
    
    for symbol in common_symbols:
        print(f"\nComparing files for symbol: {symbol}")
        compare_symbol_files(dir1, dir2, symbol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare CSV files in two directories')
    parser.add_argument('dir1', help='Path to the first directory')
    parser.add_argument('dir2', help='Path to the second directory')
    
    args = parser.parse_args()
    
    main(args.dir1, args.dir2)
