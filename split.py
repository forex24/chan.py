import pandas as pd
import sys
import os

DEFAULT_MAX_COUNT = 500000
DEFAULT_MIN_COUNT = 200000  # 设置最小分割大小
OVERLAP_RATIO = 0.2  # 20% 重叠

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
    #print(df.head())
    #print(df.tail())
    return df

def split_df(df, freq, output_dir):
    length = len(df)
    overlap_size = int(DEFAULT_MAX_COUNT * OVERLAP_RATIO)
    step_size = DEFAULT_MAX_COUNT - overlap_size

    print(f'freq: {freq}, length: {length}, splits: {(length - overlap_size) // step_size + 1}')

    start_idx = 0
    split_num = 0
    split_dfs = []

    while start_idx < length:
        end_idx = min(start_idx + DEFAULT_MAX_COUNT, length)
        split_df = df.iloc[start_idx:end_idx]
        split_dfs.append(split_df)

        start_idx += step_size
        split_num += 1

    # 检查最后一个分割是否小于最小大小
    if len(split_dfs[-1]) < DEFAULT_MIN_COUNT and len(split_dfs) > 1:
        # 将最后一个分割合并到前一个
        split_dfs[-2] = pd.concat([split_dfs[-2], split_dfs[-1]])
        split_dfs.pop()
        split_num -= 1

    # 保存分割后的 DataFrame
    for i, split_df in enumerate(split_dfs):
        output_filename = f"{output_dir}/{freq}_split_{i}.csv"
        split_df.to_csv(output_filename)
        print(f"Saved {output_filename}, rows: {len(split_df)}")

    print(f"Total splits for {freq}: {split_num}")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    # 获取命令行参数，默认为 'eurusd'
    if len(sys.argv) > 1:
        symbol = sys.argv[1]
    else:
        symbol = '/opt/data/eurusd.csv'

    # 获取起始年份，默认为 2000
    if len(sys.argv) > 2:
        start_year = int(sys.argv[2])
    else:
        start_year = 2000

    # 创建输出目录
    output_dir = f"split_data_{os.path.basename(symbol).split('.')[0]}"
    ensure_dir(output_dir)

    df = load_csv(symbol)
    timeframe=['1d','8h','6h','4h','2h','1h','30min','15min','5min','3min', '1min']
    for freq in timeframe:
        freq_df = resample_timeframe(df, freq)
        split_df(freq_df, freq, output_dir)
