import os
import sys
import pandas as pd

# 获取命令行参数，默认为当前目录
if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = os.getcwd()

# 初始化一个空的 DataFrame
combined_df = pd.DataFrame()

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.startswith('feature') and filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        # 读取 CSV 文件
        df = pd.read_csv(file_path)
        # 将读取的 DataFrame 拼接到 combined_df 中
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 去重
combined_df.drop_duplicates(inplace=True)

# 检查 open_time 列中是否存在重复
duplicates_in_open_time = combined_df[combined_df.duplicated('open_time', keep=False)]

# 打印结果
if duplicates_in_open_time.empty:
    print("No duplicates found in 'open_time' column.")
else:
    print("Duplicates found in 'open_time' column:")
    print(duplicates_in_open_time)

    # 将重复项写入 "feature_dup.csv"
    dup_file_path = os.path.join(directory, 'feature_dup.csv')
    duplicates_in_open_time.to_csv(dup_file_path, index=False)
    print(f"Duplicates have been written to {dup_file_path}")

# 将去重后的 DataFrame 写入 "feature_all.csv"
output_file_path = os.path.join(directory, 'feature_all.csv')
combined_df.to_csv(output_file_path, index=False)
print(f"Combined DataFrame has been written to {output_file_path}")
