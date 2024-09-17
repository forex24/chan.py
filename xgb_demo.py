import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

# 1. 数据读取
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 2. 数据预处理
def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X.fillna(X.mean(), inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y

# 3. 处理正负样本数量差距过大
def handle_imbalanced_data(y):
    pos_weight = (y == 0).sum() / (y == 1).sum()
    return pos_weight

# 4. 创建DMatrix
def create_dmatrix(X, y):
    dtrain = xgb.DMatrix(X, label=y)
    return dtrain

# 5. 模型训练
def train_model(dtrain, pos_weight):
    params = {
        'scale_pos_weight': pos_weight,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = xgb.train(params, dtrain)
    return model

# 6. 保存模型
def save_model(model, file_path):
    model.save_model(file_path)
    print(f"Model saved to {file_path}")

# 7. 加载模型
def load_model(file_path):
    model = xgb.Booster()
    model.load_model(file_path)
    print(f"Model loaded from {file_path}")
    return model

# 8. 模型评估
def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest)
    y_pred_labels = np.round(y_pred)
    
    accuracy = accuracy_score(y_test, y_pred_labels)
    conf_matrix = confusion_matrix(y_test, y_pred_labels)
    class_report = classification_report(y_test, y_pred_labels)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    # ROC曲线
    fpr, tpr, _ = roc_curve(y_test, y_pred)
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

# 9. 绘图
def plot_feature_importance(model, feature_names):
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': list(importance.values())})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.show()

# 主函数
def main():
    # 1. 加载数据
    data = load_data('your_dataset.csv')
    
    # 2. 数据预处理
    X, y = preprocess_data(data)
    
    # 3. 处理正负样本数量差距过大
    pos_weight = handle_imbalanced_data(y)
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. 创建DMatrix
    dtrain = create_dmatrix(X_train, y_train)
    
    # 6. 模型训练
    model = train_model(dtrain, pos_weight)
    
    # 7. 保存模型
    save_model(model, 'xgboost_model.json')
    
    # 8. 加载模型
    loaded_model = load_model('xgboost_model.json')
    
    # 9. 模型评估
    evaluate_model(loaded_model, X_test, y_test)
    
    # 10. 绘制特征重要性
    plot_feature_importance(loaded_model, data.columns[:-1])

if __name__ == "__main__":
    main()