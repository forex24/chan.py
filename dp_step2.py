import os
import pandas as pd
from glob import glob
import re

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
        # 如果只有一个分段，不需要合并
        if len(dfs) == 1:
            merged_df = dfs[0][1]
        else:
            # 按段号排序
            sorted_dfs = sorted(dfs, key=lambda x: x[0])
            sorted_dfs = [df for _, df in sorted_dfs]
            
            merged_df = pd.DataFrame()
            for i, df in enumerate(sorted_dfs):
                if 'is_sure' in df.columns:
                    if i == 0:  # 第一段
                        df = df.loc[~(df['is_sure'] == False).iloc[::-1].cumsum().iloc[::-1].astype(bool)]
                    elif i == len(sorted_dfs) - 1:  # 最后一段
                        df = df.loc[df['is_sure'].cumsum() > 0]
                    else:  # 中间段
                        df = df[df['is_sure']]
                
                if not merged_df.empty:
                    # 找到重叠部分
                    if 'timestamp' in df.columns:
                        overlap_start = merged_df['timestamp'].max()
                        overlap_end = df['timestamp'].min()
                        overlap_merged = merged_df[merged_df['timestamp'] >= overlap_start]
                        overlap_df = df[df['timestamp'] <= overlap_end]
                    else:
                        overlap_merged = merged_df.tail(min(len(merged_df), len(df)))
                        overlap_df = df.head(min(len(merged_df), len(df)))
                    
                    # 验证重叠部分
                    if not overlap_merged.equals(overlap_df):
                        print(f"警告：在 {symbol}_{timeframe}_{file_name} 中发现不一致的重叠部分")
                        # 这里可以添加更详细的不一致信息输出
                    
                    # 移除重叠部分
                    if 'timestamp' in df.columns:
                        df = df[df['timestamp'] > overlap_end]
                
                merged_df = pd.concat([merged_df, df])
        
        # 最终排序和去重
        if 'timestamp' in merged_df.columns:
            merged_df.sort_values('timestamp', inplace=True)
            merged_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        else:
            merged_df.drop_duplicates(inplace=True)
        
        # 保存合并后的CSV文件
        output_file = os.path.join(directory, f'{symbol}_merged_{timeframe}_{file_name}')
        merged_df.to_csv(output_file, index=False)
        print(f'已保存合并文件：{output_file}')

def main():
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = '/opt/data'
    
    merge_csv_files(directory)
        

if __name__ == "__main__":
    main()
