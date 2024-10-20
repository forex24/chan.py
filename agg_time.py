from pathlib import Path
import pandas as pd
import sys
import os
import json
from typing import Dict, TypedDict
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from joblib import Parallel, delayed, cpu_count

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
            basename = os.path.basename(filename)
            symbol  = os.path.splitext(basename)[0]
            symbols.add(symbol)
    return symbols

# resample different timeframes
def resample_timeframe(df, timeframe):
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
    return df

if __name__ == "__main__":
    # 获取命令行参数，默认为 'eurusd'
    base_dir = os.path.join(os.getcwd(),'csv')
    symbols = get_all_symbols(base_dir)
    timeframe=['1d','8h','6h','4h','2h','1h','30min','15min','5min','3min']
    for freq in timeframe:
        for symbol in symbols:
            df = load_csv(os.path.join(base_dir, f'{symbol}.csv'))
            freq_df = resample_timeframe(df, freq)
            freq_df.to_csv(os.path.join(base_dir, f'{symbol}_{freq}.csv'))