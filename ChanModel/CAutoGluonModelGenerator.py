import time
import os
from typing import Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, # type: ignore
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, matthews_corrcoef)

from ChanModel.CModelGenerator import CModelGenerator
from ChanModel.CAutoGluon_DataSet import CAutoGluon_DataSet


class CAutoGluonModelGenerator:
    def __init__(self, dataset: Optional[CAutoGluon_DataSet] = None, model_save_path: str = '..\\autogluon_model', threshold = 0.25):
        self.model_save_path = model_save_path
        self.model = None
        self.dataset = dataset
        self.label_column = 'label'
        self.train_data = None
        self.test_data = None
        self.pred_data = None
        self.feature_importance = None
        self.feature_stats = None
        self.save_path: Optional[str] = None
        self.required_columns = None
        self.threshold = threshold

    def _get_data(self, data_type: str) -> pd.DataFrame:
        """从数据集中获取训练、测试或预测数据。"""
        if self.dataset:
            return self.dataset.to_dataframe(data_type=data_type)
        raise ValueError("未提供 CAutoGluon_DataSet 实例或数据集参数")

    def train(self, train_set: Optional[pd.DataFrame] = None, test_set: Optional[pd.DataFrame] = None,
            pred_set: Optional[pd.DataFrame] = None, model_save_path: Optional[str] = None,
            label_column: Optional[str] = 'label', time_limit: Optional[int] = None,
            presets_type: Optional[int] = 4) -> None:
        """训练模型并评估性能。"""
        self.train_data = train_set if train_set is not None else self._get_data(data_type="train")
        self.test_data = test_set if test_set is not None else self._get_data(data_type="test")
        self.pred_data = pred_set if pred_set is not None else self._get_data(data_type="pred")

        self.label_column = label_column or self.dataset.label_column
        model_save_path = model_save_path or self.model_save_path
        # os.makedirs(model_save_path, exist_ok=True)

        # 确保标签列数据类型一致
        self.train_data[self.label_column] = self.train_data[self.label_column].astype('float64')
        self.test_data[self.label_column] = self.test_data[self.label_column].astype('float64')
        self.feature_stats = self.calculate_feature_stats(self.train_data)  # 去除特征异常值

        start_time = time.time()

        # 根据 presets 参数设置对应的值
        preset_mapping = {
            1: 'ignore_text',                 # 1: 忽略文本数据的处理（最快）
            2: 'optimize_for_deployment',     # 2: 优化模型以适应部署的需求（较快）
            3: 'good_quality',                # 3: 在质量和资源使用之间取得平衡（中等速度）
            4: 'medium_quality',              # 4: 默认的预设，适合大多数情况（中等速度）
            5: 'high_quality',                # 5: 提供高质量的预测，适度考虑效率（较慢）
            6: 'best_quality',                # 6: 获取最准确的整体预测器（最慢）
            7: 'interpretable'                # 7: 优先选择可解释性强的模型（速度可变，取决于模型复杂度）
        }

        # 在 train 方法中
        preset_str = preset_mapping.get(presets_type, 'medium_quality')  # 默认是'medium_quality'

        self.model = TabularPredictor(
            label=self.label_column,
            path=model_save_path,
            eval_metric='roc_auc'
            # eval_metric='accuracy'
        ).fit(
            train_data=self.train_data,
            tuning_data=self.test_data,
            time_limit=time_limit,
            verbosity=2,  # 日志等级
            ag_args_fit={'num_gpus': 1},
            presets=preset_str,  # 使用选择的预设
            # use_bag_holdout=True,  # bagged模式单独添加调优集
            hyperparameters = {
                'XGB': {'tree_method': 'hist'},
            }
        )
        
        total_time = time.time() - start_time
        print(f"总训练时间: {total_time:.2f} seconds")

        threshold_metrics_df = self.evaluate_multiple_thresholds(self.pred_data)
        print(threshold_metrics_df.to_string())
        self.plot_threshold_metrics(threshold_metrics_df)
        self.merge_and_plot_feature_stats()

    def predict(self, dataset: Optional[pd.DataFrame] = None) -> List[float]:
        """生成预测结果。"""
        data = dataset if dataset is not None else self._get_data(data_type="pred")
        predictions = self.model.predict(data)
        return predictions.tolist()

    def evaluate(self, dataset: Optional[pd.DataFrame] = None, label_column: Optional[str] = None) -> dict:
        """评估模型性能。"""
        data = dataset if dataset is not None else self._get_data(data_type="test")
        label_column = label_column or self.dataset.label_column
        performance = self.model.evaluate(data, label=label_column)
        return performance

    def save_model(self, save_path: Optional[str] = None) -> None:
        """保存模型到指定路径。"""
        if self.model:
            save_path = save_path or self.model_save_path
            self.model.save(path=save_path)
            print(f"模型已保存到路径：{save_path}")
        else:
            raise ValueError("模型尚未训练，无法保存。")

    def load_model(self, load_path: str) -> int:
        """
        加载已保存的模型。

        此方法根据给定的路径加载预先保存的模型，并返回模型的特征数量。
        如果指定的路径不存在该模型文件，将抛出 FileNotFoundError。

        参数:
            load_path (str): 模型文件的路径。

        返回:
            model: 模型

        抛出:
            FileNotFoundError: 如果路径不存在模型文件，抛出此异常。
        """
        if os.path.exists(load_path):
            self.model = TabularPredictor.load(load_path)
            self.model_save_path = load_path
            return self.model
        else:
            raise FileNotFoundError("模型文件不存在，请检查路径。")

    def fill_missing_columns(self, pred_data: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失的特征列。

        根据模型所需的特征列，检查预测数据中的缺失列并用默认值（NaN）填充。
        返回补全后的数据。

        参数:
            pred_data (pd.DataFrame): 输入的预测数据。

        返回:
            pd.DataFrame: 填充缺失列后的数据。
        """
        # 获取模型的所有特征列
        feature_metadata = self.model.feature_metadata
        feature_lst = feature_metadata.get_features()
        self.required_columns = feature_lst  # 保存模型所需的特征列
        feature_num = len(feature_lst)
        # print(f"加载的模型所需的特征列: {self.required_columns}")
        # print(f"原始预测数据列: {pred_data.columns.tolist()}")
        # 填充缺失的列，用 NaN 填充缺失列（你可以根据需要选择其他默认值）
        for col in self.required_columns:
            if col not in pred_data.columns:
                # print(f"缺少列: {col}，将使用 NaN 填充")
                pred_data[col] = np.nan  # 或者使用其他默认值，例如 0
            # else:
                # print(f"预测数据中已包含列: {col}")
        # print(f"补全后的数据列: {pred_data.columns.tolist()}")
        return pred_data

    def predict_probs(self, pred_data: pd.DataFrame = None, model=None) -> np.ndarray:
        """
        计算模型的预测概率。
        
        参数:
            pred_data (pd.DataFrame): 包含预测数据的数据框。
            model: 要评估的模型，默认为 None（使用 self.model）。
            
        返回:
            np.ndarray: 预测的概率数组。
        """
        if pred_data is None:
            pred_data = self.pred_data

        if model is None:
            model = self.model

        # 使用 fill_missing_columns 方法填充缺失的特征列
        pred_data = self.fill_missing_columns(pred_data)

        # 如果有标签列，删除标签列；否则直接使用特征列
        if self.label_column in pred_data.columns:
            X_test = pred_data.drop(columns=[self.label_column])
        else:
            X_test = pred_data
        
        # 获取预测概率
        y_pred_probs_df = model.predict_proba(X_test)
        y_pred_probs = y_pred_probs_df.iloc[:, 1].to_numpy()  # 取出阳性类的概率

        return y_pred_probs

    def evaluate_model(self, pred_data: pd.DataFrame = None, model=None, threshold: float = 0.5) -> dict:
        """评估模型在预测数据上的表现，支持调整阈值。
        
        参数:
            pred_data (pd.DataFrame): 包含预测数据和真实标签的数据框，默认为 None。
            model: 要评估的模型，默认为 None（使用 self.model）。
            threshold (float): 预测为正类的概率阈值，默认为0.5。
            
        返回:
            dict: 包含性能指标的字典。
        """
        # 如果没有传入预测数据，则使用 self.pred_data
        if pred_data is None:
            pred_data = self.pred_data
                
        # 如果没有传入模型，则使用 self.model
        if model is None:
            model = self.model

        if self.label_column not in pred_data.columns:
            print(f"错误: 在 pred_data 中找不到列 '{self.label_column}'")
            return {}

        y_true = pred_data[self.label_column].astype(int)  # 保持为 pandas Series

        # 调用 predict_proba 方法计算预测概率
        y_pred_probs = self.predict_probs(pred_data, model)

        # 使用概率而不是二值结果计算 AUC
        auc_score = roc_auc_score(y_true, y_pred_probs)

        # 根据给定的阈值进行预测
        binary_predictions = (y_pred_probs >= threshold).astype(int)

        # 计算性能指标
        # auc_score = roc_auc_score(y_true, binary_predictions)
        accuracy = accuracy_score(y_true, binary_predictions)
        # balanced_accuracy = balanced_accuracy_score(y_true, binary_predictions)
        precision = precision_score(y_true, binary_predictions, zero_division=0.0)
        recall = recall_score(y_true, binary_predictions)
        f1 = f1_score(y_true, binary_predictions)
        mcc = matthews_corrcoef(y_true, binary_predictions)

        # 创建一个字典来存储性能指标
        metrics = {
            'accuracy': accuracy,
            # 'balanced_accuracy': balanced_accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'mcc': mcc,
            'roc_auc': auc_score,
        }

        return metrics  # 返回性能指标

    def evaluate_multiple_thresholds(self, pred_data: pd.DataFrame = None, model=None) -> pd.DataFrame:
        """评估多个阈值下的模型性能指标，并返回结果表格。
        
        参数:
            pred_data (pd.DataFrame): 包含预测数据和真实标签的数据框，默认为 None。
            model: 要评估的模型，默认为 None（使用 self.model）。
            
        返回:
            pd.DataFrame: 包含不同阈值下的性能指标的 DataFrame。
        """
        if pred_data is None:
            pred_data = self.pred_data

        if model is None:
            model = self.model

        thresholds = [round(i * 0.1, 1) for i in range(11)]  # 生成 0.0 到 1.0 的阈值
        results = []

        for threshold in thresholds:
            metrics = self.evaluate_model(pred_data, model, threshold)
            results.append({
                'Threshold': threshold,
                'ROC_AUC': metrics['roc_auc'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Accuracy': metrics['accuracy'],
                # 'Balanced Accuracy': metrics['balanced_accuracy'],
                'F1 Score': metrics['f1'],
                'MCC': metrics['mcc'],
            })

        # 创建一个 DataFrame 来存储结果
        results_df = pd.DataFrame(results)
        return results_df

    def print_metrics(self, metrics: dict) -> None:
        """打印模型性能指标。
        
        参数:
            metrics (dict): 包含性能指标的字典。
        """
        print("=====================================")
        print(f"准确率（accuracy）: {metrics['accuracy']:.2%}")
        # print(f"平衡准确率（balanced_accuracy）: {metrics['balanced_accuracy']:.2%}")
        print(f"F1 分数（F1 Score）: {metrics['f1']:.2%}")
        print(f"精确率（precision）: {metrics['precision']:.2%}")
        print(f"召回率（recall）: {metrics['recall']:.2%}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print("=====================================")

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

    def set_feature_importance(self, dataset: Optional[pd.DataFrame] = None, model = None) -> pd.DataFrame:
        """设置并返回特征重要性。"""
        if model is None:
            model = self.model

        if model:
            if dataset is None:
                dataset = self.train_data
                
            # 提取特征重要性
            self.feature_importance = model.feature_importance(dataset).reset_index()
            # self.feature_importance.rename(columns={'index': 'feature'}, inplace=True)  # index重命名为feature。inplace=True 表示直接在原数据框上进行修改
            # print("feature_importance:", self.feature_importance)
            
            return self.feature_importance
        else:
            print("错误: 没有模型，无法获取特征重要性。")
            return None

    def merge_and_plot_feature_stats(self, dataset: Optional[pd.DataFrame] = None) -> None:
        """合并特征重要性和特征统计量，并绘制图形。"""
        if dataset is None:
            dataset = self.train_data
            
        # 计算特征统计量
        # self.feature_stats = self.calculate_feature_stats(dataset)
        # print("feature_stats:", self.feature_stats)
        self.feature_importance = self.set_feature_importance(dataset)
        # 合并并打印结果
        merged_stats = self.feature_importance.merge(self.feature_stats, on='index', how='left')
        # print("Merged Feature Importance and Stats:", merged_stats)
        merged_stats_str = merged_stats.to_string(index=False, float_format="{:.6f}".format)
        print("Merged Feature Importance and Stats:\n", merged_stats_str)
        
        # 绘图
        self._plot_feature_importance(20, self.save_path)

    def _plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """绘制特征重要性图。
        
        参数:
            top_n (int): 要显示的特征数量，默认为 10。
        """
        if self.feature_importance is not None:
            # 选择特征和重要性列，并确保它们存在
            if 'importance' in self.feature_importance.columns:
                feature_importance = self.feature_importance[['index', 'importance', 'stddev', 'p_value', 'p99_high', 'p99_low', 'n']]
                feature_importance.columns = ['feature', 'importance', 'stddev', 'p_value', 'p99_high', 'p99_low', 'n']
                
                # 添加 -log10(p_value) 列
                feature_importance = feature_importance.copy()  # 确保是副本
                feature_importance['log_p_value'] = -np.log10(feature_importance['p_value'])
                
                # 排序并选择前 top_n 个特征
                feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(top_n)

                # 设置图形大小
                fig, ax = plt.subplots(figsize=(12, len(feature_importance) * 0.5))
                
                # 绘制横向条形图，带误差线
                bars = ax.barh(feature_importance['feature'], feature_importance['importance'],
                            xerr=feature_importance['stddev'], color='skyblue', label='Importance', capsize=5, height=0.6)

                # 添加 p99_high 和 p99_low 的竖线
                for idx, row in feature_importance.iterrows():
                    ax.vlines(row['p99_high'], idx - 0.3, idx + 0.3, colors='green', linestyles='--', label='p99_high' if idx == 0 else "")
                    ax.vlines(row['p99_low'], idx - 0.3, idx + 0.3, colors='red', linestyles='--', label='p99_low' if idx == 0 else "")

                # 在柱子上标注 n 值和 -log_p_value
                for idx, row in feature_importance.iterrows():
                    ax.text(row['importance'] + row['stddev'] + 0.01, idx, f"{row['log_p_value']:.2f}", va='center', color='black', fontsize=8)

                # 设置标签和标题
                ax.set_xlim(left=0.0)  # 确保x轴从0.0开始
                ax.set_xlabel('Importance')
                ax.set_ylabel('Feature')
                ax.set_title(f'Top {top_n} Feature Importances')
                ax.legend(loc='lower right')  # 设置图例位置为右下角
                ax.invert_yaxis()  # 反转 Y 轴

                plt.tight_layout()  # 自动调整布局
                # 保存图像（如果提供了路径）
                if save_path:
                    plt.savefig(save_path, format='png')
                    print(f"图像已保存至 {save_path}")

                plt.show()
            else:
                print("错误: 没有找到 'importance' 列。")
        else:
            print("错误: 请先调用 set_feature_importance 方法以设置特征重要性。")

    def plot_threshold_metrics(self, threshold_metrics_df: pd.DataFrame) -> None:
        """绘制不同阈值下性能指标的曲线图。
        
        参数:
            threshold_metrics_df (pd.DataFrame): 包含阈值和性能指标的数据框。
        """
        plt.figure(figsize=(12, 8))

        # 绘制精确率
        plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['ROC_AUC'], label='ROC_AUC', marker='o')
        # 绘制精确率
        plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['Precision'], label='Precision', marker='o')
        # 绘制召回率
        plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['Recall'], label='Recall', marker='o')
        # 绘制准确率
        plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['Accuracy'], label='Accuracy', marker='o')
        # 绘制平衡准确率
        # plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['Balanced Accuracy'], label='Balanced Accuracy', marker='o')
        # 绘制F1分数
        plt.plot(threshold_metrics_df['Threshold'], threshold_metrics_df['F1 Score'], label='F1 Score', marker='o')

        plt.title('Performance Metrics vs. Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Performance Metrics')
        plt.xticks(threshold_metrics_df['Threshold'])  # 设置 X 轴刻度
        plt.legend()
        plt.grid()
        plt.show()