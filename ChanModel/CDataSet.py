import abc
import pandas as pd
import numpy as np
from typing import List, Union
from scipy.sparse import csr_matrix


class CDataSet(abc.ABC):

    def __init__(self, data: Union[pd.DataFrame, np.ndarray, csr_matrix], tag: str = "tmp", feature_meta: List[str] = None):
        """
        初始化数据集对象。

        :param data: 具体的数据内容，支持 DataFrame, numpy array 或稀疏矩阵
        :param tag: 数据集的标签，默认为 "tmp"
        :param feature_meta: 特征元数据列表，用于列名
        """
        self.data = data
        self.tag = tag
        self.feature_meta = feature_meta or []

        # 如果 data 是 DataFrame，直接设置 feature_meta
        if isinstance(data, pd.DataFrame) and not self.feature_meta:
            self.feature_meta = list(data.columns)

    @abc.abstractmethod
    def get_count(self) -> int:
        """
        获取样本总数。

        :return: 样本总数
        """
        if isinstance(self.data, pd.DataFrame):
            return len(self.data)
        elif isinstance(self.data, (np.ndarray, csr_matrix)):
            return self.data.shape[0]

    @abc.abstractmethod
    def get_pos_count(self) -> int:
        """
        获取正样本数量。

        :return: 正样本数量
        """
        labels = self.get_label()
        return sum(1 for label in labels if label > 0)

    @abc.abstractmethod
    def get_label(self) -> List[float]:
        """
        获取标签数据。

        :return: 标签数组
        """
        if isinstance(self.data, pd.DataFrame):
            return self.data['label'].tolist()
        elif isinstance(self.data, np.ndarray) or isinstance(self.data, csr_matrix):
            return self.data[:, -1].toarray().flatten() if isinstance(self.data, csr_matrix) else self.data[:, -1].tolist()

    def to_dataframe(self) -> pd.DataFrame:
        """
        将数据转换为 Pandas DataFrame 格式。
        
        :return: 转换后的 DataFrame
        """
        if isinstance(self.data, pd.DataFrame):
            df = self.data.copy()
        else:
            features = self.data.toarray() if isinstance(self.data, csr_matrix) else self.data
            df = pd.DataFrame(features, columns=self.feature_meta)
            if 'label' not in df.columns:
                df['label'] = self.get_label()
        
        return df

    def load_data(self, new_data: Union[pd.DataFrame, np.ndarray, csr_matrix]):
        """
        加载新的数据。

        :param new_data: 要加载的数据
        """
        self.data = new_data
        # 如果新数据是 DataFrame，更新 feature_meta
        if isinstance(new_data, pd.DataFrame):
            self.feature_meta = list(new_data.columns)

    def load_feature_meta(self, meta: List[str]):
        """
        加载特征元数据。

        :param meta: 特征元数据的列表
        """
        if not isinstance(meta, list) or not all(isinstance(col, str) for col in meta):
            raise ValueError("特征元数据应为字符串列表。")
        self.feature_meta = meta
        if isinstance(self.data, pd.DataFrame):
            self.data.columns = meta

