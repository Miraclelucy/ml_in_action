# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# 读取训练集数据
def get_training_data(file_path):
    orig_data = np.loadtxt(file_path,skiprows=1) #忽略第一行的标题
    cols = orig_data.shape[1]
    return (orig_data, orig_data[:, :cols - 1], orig_data[:, cols-1:])

#展示实际数据点以及拟合出的曲线图
def showlinercurve(y_hat, sample_training_set):
    x, y = sample_training_set[:, 1], sample_training_set[:, 2]
    plt.scatter(x, y, color='b', marker='x',label="sample data")
    plt.plot(x, y_hat, 'r', color="r",label="regression curve")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('liner regression curve')
    plt.legend()
    plt.show()

# 导入岭回归模型 - l2正则化
from sklearn.linear_model import Ridge

def RidgeRegression():
    training_data_include_y, X, y = get_training_data('data.txt')
    # 创建线性回归模型对象
    model = Ridge()
    # 使用训练数据 X -特征矩阵 和标签 y 对模型进行拟合
    model.fit(X, y)
    # 对模型进行预测
    y_hat = model.predict(X)
    showlinercurve(y_hat, training_data_include_y)
 
if __name__ == '__main__':
    RidgeRegression()
