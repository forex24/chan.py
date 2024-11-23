import abc
from typing import Tuple, List

from ChanModel.CDataSet import CDataSet


class CModelGenerator(abc.ABC):
    """
    抽象模型生成类，定义模型训练、预测和数据集管理的方法。
    """

    @abc.abstractmethod
    def train(self, train_set: "CDataSet", test_set: "CDataSet") -> None:
        """
        训练模型。

        :param train_set: 训练数据集
        :param test_set: 测试数据集
        """
        pass

    @abc.abstractmethod
    def create_train_test_set(self, sample_iter) -> Tuple["CDataSet", "CDataSet"]:
        """
        创建训练集和测试集。

        :param sample_iter: 样本迭代器
        :return: 训练集和测试集的元组
        """
        pass

    @abc.abstractmethod
    def save_model(self):
        """
        保存模型到指定路径。
        """
        pass

    @abc.abstractmethod
    def load_model(self) -> int:
        """
        加载模型并返回所需特征维度。

        :return: 特征维度
        """
        pass

    @abc.abstractmethod
    def predict(self, dataSet: "CDataSet") -> List[float]:
        """
        对给定数据集进行预测。

        :param dataSet: 输入数据集
        :return: 预测结果列表
        """
        pass

    @abc.abstractmethod
    def create_data_set(self, feature_arr: List[List[float]]) -> "CDataSet":
        """
        从特征数组创建数据集。

        :param feature_arr: 描述 N 个样本 M 个特征的二维数组
        :return: 创建的数据集
        """
        pass
