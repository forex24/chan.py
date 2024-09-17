import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularPredictor
import argparse
import sys

# 增加递归深度限制
sys.setrecursionlimit(1500)

# 1. 数据读取
def load_data(file_path):
    data = pd.read_parquet(file_path, engine='fastparquet')
    return data

# 2. 数据预处理
def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X.fillna(X.mean(), inplace=True)
    return X, y

# 3. 处理正负样本数量差距过大
def handle_imbalanced_data(y):
    pos_weight = (y == 0).sum() / (y == 1).sum()
    return pos_weight

# 4. 特征工程
def feature_engineering(X):
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 添加多项式特征
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_scaled)
    
    return X_poly

# 5. 使用AutoGluon进行超参数调优和模型训练
def train_with_autogluon(X_train, y_train, X_test, y_test):
    train_data = pd.DataFrame(X_train)
    train_data['label'] = y_train
    
    test_data = pd.DataFrame(X_test)
    test_data['label'] = y_test
    
    predictor = TabularPredictor(label='label').fit(train_data, presets='best_quality')
    
    return predictor

# 6. 模型评估
def evaluate_model(predictor, X_test, y_test):
    test_data = pd.DataFrame(X_test)
    test_data['label'] = y_test
    
    y_pred = predictor.predict(test_data.drop(columns=['label']))
    y_pred_proba = predictor.predict_proba(test_data.drop(columns=['label']))[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# 7. 绘图
def plot_feature_importance(predictor, feature_names):
    importance = predictor.feature_importance(test_data.drop(columns=['label']))
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

# 8. 保存模型
def save_model(predictor, model_path):
    predictor.save(model_path)
    print(f"Model saved to {model_path}")

# 9. 加载模型
def load_model(model_path):
    predictor = TabularPredictor.load(model_path)
    print(f"Model loaded from {model_path}")
    return predictor

# 主函数
def main():
    parser = argparse.ArgumentParser(description='Train a classification model using AutoGluon.')
    parser.add_argument('file_path', type=str, help='Path to the input Parquet file.')
    parser.add_argument('--save_path', type=str, default='autogluon_model', help='Path to save the trained model.')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load a pre-trained model.')
    args = parser.parse_args()
    
    # 1. 加载数据
    data = load_data(args.file_path)
    
    # 2. 数据预处理
    X, y = preprocess_data(data)
    
    # 3. 处理正负样本数量差距过大
    pos_weight = handle_imbalanced_data(y)
    
    # 4. 特征工程
    X_engineered = feature_engineering(X)
    
    # 5. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42, stratify=y)
    
    if args.load_path:
        # 加载预训练模型
        predictor = load_model(args.load_path)
    else:
        # 使用AutoGluon进行超参数调优和模型训练
        predictor = train_with_autogluon(X_train, y_train, X_test, y_test)
        
        # 保存模型
        save_model(predictor, args.save_path)
    
    # 模型评估
    evaluate_model(predictor, X_test, y_test)
    
    # 绘制特征重要性
    plot_feature_importance(predictor, data.columns[:-1])

if __name__ == "__main__":
    main()
