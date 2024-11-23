import os
from typing import Optional
import numpy as np
import pandas as pd
from ChanModel.CDataSet import CDataSet
from sklearn.utils import resample # type: ignore

class CAutoGluon_DataSet(CDataSet):
    def __init__(self, tag="autogluon"):
        """
        初始化 CAutoGluon_DataSet 对象，用于存储和管理训练集、测试集、预测集，以及特征名等信息。

        :param tag: 数据集的标签
        """
        super().__init__(data=None, tag=tag)
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.pred_data = pd.DataFrame()
        self.label_column = 'label'
        self.feature_names = []
        self.feature_stats = None

    def load_train_data(self, csv_path):
        """
        从 CSV 文件加载训练数据，并与现有数据合并。

        :param csv_path: 训练数据 CSV 文件路径
        """
        new_data = pd.read_csv(csv_path, low_memory=False)
        # print(f"加载训练数据: {csv_path}, 列名: {new_data.columns.tolist()}")
        self._update_feature_names(new_data)
        self.train_data = self._merge_data(self.train_data, new_data)
        self._ensure_column_consistency()  # 确保列名一致性

    def load_test_data(self, csv_path):
        """
        从 CSV 文件加载测试数据，并与现有数据合并。

        :param csv_path: 测试数据 CSV 文件路径
        """
        new_data = pd.read_csv(csv_path, low_memory=False)
        # print(f"加载测试数据: {csv_path}, 列名: {new_data.columns.tolist()}")
        self._update_feature_names(new_data)
        self.test_data = self._merge_data(self.test_data, new_data)
        self._ensure_column_consistency()  # 确保列名一致性

    def load_pred_data(self, csv_path):
        """
        从 CSV 文件加载预测数据，并与现有数据合并。

        :param csv_path: 预测数据 CSV 文件路径
        """
        new_data = pd.read_csv(csv_path, low_memory=False)
        # print(f"加载预测数据: {csv_path}, 列名: {new_data.columns.tolist()}")
        self._update_feature_names(new_data)
        self.pred_data = self._merge_data(self.pred_data, new_data)
        self._ensure_column_consistency()  # 确保列名一致性

    def _update_feature_names(self, data):
        """
        更新特征名称列表，确保与数据集中的列对齐。

        :param data: 新加载的数据
        """
        columns = data.columns.tolist()
        if self.label_column in columns:
            columns.remove(self.label_column)

        if not self.feature_names:
            self.feature_names = columns
        else:
            # 不需要再进行一致性检查，因为我们将合并所有特征
            self.feature_names = list(set(self.feature_names).union(columns))

    def _merge_data(self, existing_data, new_data):
        """
        根据列名合并两个数据集，保留所有列，对于缺失的列用 NaN 填充。

        :param existing_data: 已存在的数据集
        :param new_data: 新加载的数据集
        :return: 合并后的数据集
        """
        # 获取合并前的样本数量
        existing_count = len(existing_data)
        new_count = len(new_data)

        # 获取现有数据和新数据的列名
        existing_columns = existing_data.columns.tolist()
        new_columns = new_data.columns.tolist()
        
        # 确保 'label' 列在最前面
        if 'label' in existing_columns:
            existing_columns.remove('label')
        if 'label' in new_columns:
            new_columns.remove('label')
        
        # 重新排列列名
        sorted_existing_columns = ['label'] + sorted(existing_columns)
        sorted_new_columns = ['label'] + sorted(new_columns)

        # 检查并删除全部为 NA 的列
        existing_data = existing_data.dropna(axis=1, how='all')
        new_data = new_data.dropna(axis=1, how='all')
        
        # 合并数据时保留所有列
        combined_data = pd.concat([
            existing_data.reindex(columns=sorted_existing_columns), 
            new_data.reindex(columns=sorted_new_columns)
        ], ignore_index=True, sort=False)

        # 获取合并后的样本数量
        combined_count = len(combined_data)

        # 打印合并前和合并后的样本数量
        print(f"合并前 - 现有数据集样本数: {existing_count}, 新数据集样本数: {new_count}"
              +"------"+ f"合并后 - 总样本数: {combined_count}")

        return combined_data

    def _ensure_column_consistency(self):
        """
        确保训练集、测试集和预测集的列名一致。
        如果测试集和预测集中缺少训练集的列，则添加这些列，并用 NaN 填充。
        """
        all_columns = set(self.train_data.columns).union(self.test_data.columns).union(self.pred_data.columns)
        
        # 遍历每个数据集，确保所有列都存在
        for df in [self.train_data, self.test_data, self.pred_data]:
            # 重新索引以包括所有列，并填充缺失值
            df_reindexed = df.reindex(columns=all_columns, fill_value=pd.NA)
            
            # 更新原始数据框
            if df is self.train_data:
                self.train_data = df_reindexed
            elif df is self.test_data:
                self.test_data = df_reindexed
            elif df is self.pred_data:
                self.pred_data = df_reindexed

    def get_count(self):
        """
        获取训练集样本总数。
        """
        return len(self.train_data)

    def get_pos_count(self):
        """
        获取训练集中的正样本数量。
        """
        return self.train_data[self.train_data[self.label_column] == 1].shape[0]

    def get_label(self) -> list:
        """
        获取训练集的标签数据。

        :return: 标签列表
        """
        return self.train_data[self.label_column].tolist()

    def to_dataframe(self, data_type="train"):
        """
        将指定的数据集转换为 Pandas DataFrame 格式。

        :param data_type: 数据类型，可选 "train"、"test"、"pred"
        :return: 转换后的 DataFrame
        """
        if data_type == "train":
            return self.train_data
        elif data_type == "test":
            return self.test_data
        elif data_type == "pred":
            return self.pred_data
        else:
            raise ValueError("数据类型无效，请选择 'train'、'test' 或 'pred'。")

    def balance_data(self, method='downsample', target_ratio=1.0, balance_test=False, balance_pred=False):
        """
        对数据进行平衡处理。

        :param method: 平衡方法，'downsample' 表示下采样，'upsample' 表示上采样
        :param target_ratio: 正负样本比例目标，1.0 表示正负样本数量相等。0 表示不做平衡处理。
        :param balance_test: 是否平衡测试集
        :param balance_pred: 是否平衡预测集
        """

        # 定义平衡数据集的内部函数
        def balance(df):
            if df.empty or target_ratio == 0:
                # 若数据为空或目标比例为0，则直接返回原始数据
                print(f"未做平衡处理。正样本数: {len(df[df[self.label_column] == 1])}，负样本数: {len(df[df[self.label_column] == 0])}")
                return df
            
            # 分离正负样本
            positive_samples = df[df[self.label_column] == 1]
            negative_samples = df[df[self.label_column] == 0]

            # 当前正负样本数量
            current_positive_count = len(positive_samples)
            current_negative_count = len(negative_samples)

            # 执行下采样或上采样
            if method == 'downsample':
                desired_negative_count = int(current_positive_count * target_ratio)
                if desired_negative_count < current_negative_count:
                    negative_samples = resample(
                        negative_samples, replace=False, n_samples=desired_negative_count, random_state=42
                    )
            elif method == 'upsample':
                desired_positive_count = int(current_negative_count * target_ratio)
                if desired_positive_count > current_positive_count:
                    positive_samples = resample(
                        positive_samples, replace=True, n_samples=desired_positive_count, random_state=42
                    )
            else:
                raise ValueError("无效的平衡方法，必须是 'downsample' 或 'upsample'。")

            # 合并样本并返回平衡后的数据集
            balanced_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
            print(f"平衡前 - 正样本数: {current_positive_count}，负样本数: {current_negative_count}" +"------"+
                f"平衡后 - 正样本数: {len(balanced_df[balanced_df[self.label_column] == 1])}，负样本数: {len(balanced_df[balanced_df[self.label_column] == 0])}")
            
            return balanced_df

        # 平衡训练集
        self.train_data = balance(self.train_data)
        # print(f"平衡后的训练集总样本数量: {len(self.train_data)}")

        # 根据需求平衡测试集和预测集
        if balance_test:
            self.test_data = balance(self.test_data)
            # print(f"平衡后的测试集总样本数量: {len(self.test_data)}")
            
        if balance_pred:
            self.pred_data = balance(self.pred_data)
            # print(f"平衡后的预测集总样本数量: {len(self.pred_data)}")

    def calculate_feature_stats(self, dataset: Optional[pd.DataFrame] = None, replace_outliers: bool = False) -> pd.DataFrame:
        """计算特征的均值、方差、标准差、最大值、最小值、中位数和中位数绝对偏差，并返回结果。若启用，使用留一法替换异常值为 pd.NA。"""

        def _calculate_mad(series: pd.Series) -> float:
            """计算中位数绝对偏差（MAD）。"""
            series = pd.to_numeric(series, errors='coerce')  # 将数据转换为数值类型，无法转换的值为 NaN
            median = series.median()  # 计算中位数
            mad = (series - median).abs().median()  # 计算中位数绝对偏差
            return mad

        if dataset is None:
            dataset = self.train_data

        # 如果有标签列，删除标签列
        if self.label_column in dataset.columns:
            dataset = dataset.drop(columns=[self.label_column])

        # 仅保留数值列，过滤掉非数值列
        dataset = dataset.select_dtypes(include=[np.number])

        # 如果启用替换异常值功能
        if replace_outliers:
            for feature in dataset.columns:
                values = dataset[feature].values
                outliers = []
                
                # 对每个数据点进行留一法计算 z-score
                for i in range(len(values)):
                    leave_one_mean = (np.sum(values) - values[i]) / (len(values) - 1)
                    leave_one_std = np.sqrt(((np.sum((values - leave_one_mean)**2) - (values[i] - leave_one_mean)**2) / (len(values) - 1)))
                    
                    if leave_one_std != 0:  # 避免除以零的情况
                        z_score = (values[i] - leave_one_mean) / leave_one_std
                        if (abs(z_score) > 3 and values[i] > 5):  # 识别 z-score 超过 ±3 的异常值
                            outliers.append(i)
                
                # 替换异常值并打印信息
                if outliers:
                    replaced_values = dataset.loc[outliers, feature]  # 获取被替换的异常值
                    outlier_pairs = list(zip(outliers, replaced_values))  # 将索引和对应的值配对
                    print(f"正在替换特征 '{feature}' 中的异常值{len(outlier_pairs)}条，详情如下：")
                    for idx, value in outlier_pairs:
                        print(f"    索引位置: {idx}，被替换的值: {value}")
                    # 执行替换
                    dataset.loc[outliers, feature] = pd.NA

        # 计算统计指标
        stats = {
            'variance': dataset.var(),
            'std_dev': dataset.std(),
            'mad': dataset.apply(lambda x: _calculate_mad(x)),  # 手动计算 MAD
            'median': dataset.apply(lambda x: pd.to_numeric(x, errors='coerce').median()),  # 转换为数值再计算中位数
            'mean': dataset.mean(),
            'max': dataset.max(),
            'min': dataset.min(),
        }
        
        # 将统计结果合并为 DataFrame
        stats_df = pd.DataFrame(stats)
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'index'}, inplace=True)

        # 保存计算结果
        self.feature_stats = stats_df
        return stats_df

    def load_all_data_from_directory(self, directory_path):
        """
        从指定目录中批量加载所有数据文件，分别加载到训练集、测试集和预测集中。

        :param directory_path: 包含数据文件的目录路径
        """
        print(f"开始从目录 '{directory_path}' 加载所有数据文件...")
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if not filename.endswith('.csv'):
                print(f"忽略非 CSV 文件: {filename}")
                continue
            
            # 根据文件名加载相应的数据集
            if '_train' in filename or '_total' in filename:
                print(f"检测到训练数据文件: {filename}")
                self.load_train_data(file_path)
            elif '_test' in filename:
                print(f"检测到测试数据文件: {filename}")
                self.load_test_data(file_path)
            elif '_pred' in filename:
                print(f"检测到预测数据文件: {filename}")
                self.load_pred_data(file_path)
            else:
                print(f"文件名 '{filename}' 未能匹配到任何数据集，将忽略此文件。")

        # 重新确保所有数据集的列名一致
        self._ensure_column_consistency()
        print("所有数据加载完成。")
