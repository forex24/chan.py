import os
import sys
import pandas as pd


def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# 获取命令行参数，默认为当前目录
if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = os.getcwd()

def check_dup(directory):
    # 查找目录下有多少个不同的 symbol
    symbols = set()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            symbol = filename.split('_')[0]
            symbols.add(symbol)
    
    # 对每个 symbol 进行处理
    for symbol in symbols:
        # 初始化一个空的 DataFrame
        combined_df = pd.DataFrame()
    
        # 遍历目录下的所有文件
        for filename in os.listdir(directory):
            if filename.startswith(f'{symbol}_feature') and filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                # 读取 CSV 文件
                df = pd.read_csv(file_path)
                # 将读取的 DataFrame 拼接到 combined_df 中
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
        # 去重
        combined_df.drop_duplicates(inplace=True)
        combined_df = combined_df.sort_values(by='open_time')
        col = combined_df.pop('label')
        combined_df.insert(loc= 0 , column= 'label', value= col)
    
        # 检查 open_time 列中是否存在重复
        duplicates_in_open_time = combined_df[combined_df.duplicated('open_time', keep=False)]
    
        # 打印结果
        if duplicates_in_open_time.empty:
            print(f"No duplicates found in 'open_time' column for symbol {symbol}.")
        else:
            print(f"Duplicates found in 'open_time' column for symbol {symbol}:")
            print(duplicates_in_open_time)
            duplicates_in_open_time = duplicates_in_open_time.sort_values(by='open_time')
    
            # 将重复项写入 "{symbol}_all_dup.csv"
            dup_file_path = os.path.join(directory, f'{symbol}_all_dup.csv')
            duplicates_in_open_time.to_csv(dup_file_path, index=False)
            print(f"Duplicates for symbol {symbol} have been written to {dup_file_path}")
    
        # 将去重后的 DataFrame 写入 "{symbol}_all_feature.csv"
        output_file_path = os.path.join(directory, f'{symbol}_all_feature.csv')
        combined_df.to_csv(output_file_path, index=False)
        print(f"Combined DataFrame for symbol {symbol} has been written to {output_file_path}")
