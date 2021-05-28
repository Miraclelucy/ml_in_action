import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

#训练数据
x = np.array(list(range(1, 11))).reshape(-1, 1)
y = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05]).ravel()

model1 = DecisionTreeRegressor(max_depth=1) #最大深度为1的决策回归树
model2 = DecisionTreeRegressor(max_depth=3) #最大深度为3的决策回归树
model3 = linear_model.LinearRegression() #线性回归
model1.fit(x, y)
model2.fit(x, y)
model3.fit(x, y)

# Predict
X_test = np.arange(0.0, 10.0, 0.01)[:, np.newaxis] #0.0到10.0每隔0.01取值，共1000个点作为测试数据
y_1 = model1.predict(X_test) #对1000个点执行预测
y_2 = model2.predict(X_test) #对1000个点执行预测
y_3 = model3.predict(X_test) #对1000个点执行预测

# Plot the results
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black",c="darkorange", label="data") #测试数据的散点图
plt.plot(X_test, y_1, color="cornflowerblue",label="max_depth=1", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=3", linewidth=2)
plt.plot(X_test, y_3, color='red', label='liner regression', linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()
