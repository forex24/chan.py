import argparse
import os
import mplfinance as mpf
import pandas as pd


def show(df):
    mpf.plot(df, type='candle')

def load_csv(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index=df.timestamp
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files for symbol analysis.')
    parser.add_argument('--root', default='/opt/data', help='Root directory for data (default: /opt/data)')
    parser.add_argument('--symbol', help='Specific symbol to process (optional)')
    
    args = parser.parse_args()

    root_directory = args.root
    input_directory = os.path.join(root_directory, 'raw_data')

    if args.symbol:
        csv_file = f"{args.symbol}.csv"
        file_path = os.path.join(input_directory, csv_file)
        df = load_csv(file_path)
        show(df)