#feature_2024_06_28_2024_10_04.libsvm
## 将libsvm转为dataframe  
from sklearn.datasets import load_svmlight_file  
from pandas import DataFrame  
import pandas as pd  
  
X_train, y_train = load_svmlight_file("feature_2024_06_28_2024_10_04.libsvm")  
mat = X_train.todense()   
  
df1 = pd.DataFrame(mat)  
df1.columns = ['sepal_length',  'sepal_width',  'petal_length',  'petal_width']  
  
df2 = pd.DataFrame(y_train)  
df2.columns = ['target']  
  
df = pd.concat([df2, df1], axis=1)      # 第一列为target  
df.to_csv("df_data.txt", index=False)  
