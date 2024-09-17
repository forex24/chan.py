import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from category_encoders import TargetEncoder
import joblib
from imblearn.over_sampling import SMOTE

# 读取数据
data = pd.read_csv('your_dataset.csv')

# 查看数据的前几行
print(data.head())

# 处理缺失值
data = data.dropna()

# 分离特征和目标变量
X = data.drop('target_column', axis=1)
y = data['target_column']

# 如果有分类变量，可以使用One-Hot编码
X = pd.get_dummies(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 SMOTE 进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 计算正负样本的比例
scale_pos_weight = sum(y_resampled == 0) / sum(y_resampled == 1)


# 特征选择
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 创建交互特征
X_train['interaction_feature'] = X_train['feature1'] * X_train['feature2']
X_test['interaction_feature'] = X_test['feature1'] * X_test['feature2']

# 生成多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# 假设数据中有分类变量，使用目标编码
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# 初始化XGBoost分类器
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1, max_depth=3)

# 训练模型
model.fit(X_train_poly, y_train)

# 使用K折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_poly, y_train, cv=kfold, scoring='accuracy')

# 计算交叉验证的准确率
cv_accuracy = cv_scores.mean()
print(f'Cross-Validation Accuracy: {cv_accuracy}')

# 在测试集上进行预测
y_pred = model.predict(X_test_poly)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# 打印分类报告
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 获取特征重要性
feature_importance = model.feature_importances_

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=X_train.columns)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.show()

# 保存模型
joblib.dump(model, 'xgboost_classifier_model.pkl')
