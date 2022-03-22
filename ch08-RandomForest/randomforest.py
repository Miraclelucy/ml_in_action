import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

#随机森林分类算法，对iris数据执行分类预测
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.8 #随机划分训练集和测试集
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2) #随机森林分类模型
y, _ = pd.factorize(train['species'])  #向量化
clf.fit(train[features], y)

preds = iris.target_names[clf.predict(test[features])]  #执行预测
#画出混淆矩阵，横向为预测值，纵向为真实值
print(pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds']))

#随机森林回归算法，对boston房价数据执行回归预测
boston_house = load_boston()
boston_feature_name = boston_house.feature_names
boston_features = boston_house.data
boston_target = boston_house.target
#随机划分80%训练集和20%测试集
x_train,x_test,y_train,y_test = train_test_split(boston_features,boston_target,test_size=0.2)

clf = RandomForestRegressor(n_estimators=15)  #随机森林回归模型
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
print("MSE = %f" %mean_squared_error(y_test, preds)) #计算MSE均方误差
print("MAPE = %f" %mape(y_test,preds)) #计算MAPE平均百分比误差
