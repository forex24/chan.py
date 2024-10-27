import os
import pandas as pd
import argparse
from typing import List, Dict, Tuple
import glob

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
    """加载CSV文件��返回DataFrame"""
    df = pd.read_csv(file_path)
    time_column = 'begin_time' if 'begin_time' in df.columns else 'time'
    df[time_column] = pd.to_datetime(df[time_column])
    return df

def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, fields: List[str]) -> Tuple[bool, str]:
    """比较两个DataFrame的指定字段"""
    if df1.empty and df2.empty:
        return True, "Both dataframes are empty"
    
    if df1.empty or df2.empty:
        return False, "One dataframe is empty while the other is not"
    
    time_column = 'begin_time' if 'begin_time' in df1.columns else 'time'
    
    df1_sorted = df1.sort_values(time_column).set_index(time_column)
    df2_sorted = df2.sort_values(time_column).set_index(time_column)
    
    common_index = df1_sorted.index.intersection(df2_sorted.index)
    
    if common_index.empty:
        return False, "No common timestamps found"
    
    df1_common = df1_sorted.loc[common_index, fields]
    df2_common = df2_sorted.loc[common_index, fields]
    
    if df1_common.equals(df2_common):
        return True, "Data is identical for common timestamps"
    else:
        diff_count = (df1_common != df2_common).sum().sum()
        return False, f"Found {diff_count} differences in the specified fields"

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
        
        file_type = file_name.split('_')[-1]
        if file_type not in FILE_TYPE_FIELDS:
            print(f"Unknown file type: {file_type}")
            continue
        
        fields = FILE_TYPE_FIELDS[file_type]
        
        df1 = load_csv(file1)
        df2 = load_csv(file2)
        
        is_same, message = compare_dataframes(df1, df2, fields)
        print(f"Comparing {file_name}: {message}")

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
