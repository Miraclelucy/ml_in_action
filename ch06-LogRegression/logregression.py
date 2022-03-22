# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def loadDataSet():
    x_data = []; y_data = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        x_data.append([float(lineArr[0]),float(lineArr[1])])
        y_data.append([float(lineArr[2])])
    return x_data,y_data

#定义sigmoid方法
def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))

#使用梯度下降法训练逻辑回归模型
def log_regress(x_train, y_train, alpha, maxcycles):
    # 对x_train在左边补齐x0=1，读进来的x只有x1,x2...
    x_train = np.hstack([np.ones((len(x_train), 1)), x_train])
    initial_theta = np.ones(x_train.shape[1]).reshape(x_train.shape[1], 1) #初始化theta为全1
    # 梯度下降法求出参数theta
    theta = gradient_descending(x_train, y_train, initial_theta, alpha, maxcycles)
    return theta

#梯度下降的过程
def gradient_descending(X, Y, initial_theta, alpha, maxcycles, epsilon=1e-5):
    theta = initial_theta
    cycle = 0
    while cycle < maxcycles:
        last_theta = theta #上一次计算的theta
        theta = theta - alpha * dJ(theta, X, Y)
        #如果前后两次计算的损失函数差小于epsilon，则学习结束
        if (abs(Jtheta(theta, X, Y)-Jtheta(last_theta, X, Y)) < epsilon):
            print("abs<epsilon")
            break
        cycle += 1
    return theta

#计算代价函数
def Jtheta(theta, X, Y):
    # sigmoid计算出的假设函数值
    h_theta = sigmoid(X.dot(theta))
    # 计算代价函数
    Jtheta = - np.sum(Y*np.log(h_theta)+(1-np.array(Y))*np.log(1-np.array(h_theta))) / len(Y)
    return Jtheta

#代价函数的梯度
def dJ(theta, X, Y):
    return (X.T.dot(sigmoid(X.dot(theta)) - Y)) / len(Y)

#对测试集预测
def predict(X_predict,theta):
    #左端补0后，通过sigmoid函数计算概率
    X_predict = np.hstack([np.ones((len(X_predict), 1)), X_predict])
    probability = sigmoid(X_predict.dot(theta))
    # 概率大于0.5则分类为1，否则为0
    return np.array(probability >= 0.5, dtype='int')

#展示实际数据点以及拟合出的曲线图
def showlinercurve(theta, x_data,y_data):
    x_data0 = []; y_data0 = []
    x_data1 = []; y_data1 = []
    #将0和1两个类别分开，以便用不同的颜色图示
    for i in range(len(y_data)):
        if int(y_data[i][0]) == 1:
            x_data1.append(x_data[i][0])
            y_data1.append(x_data[i][1])
        else:
            x_data0.append(x_data[i][0])
            y_data0.append(x_data[i][1])
    x = np.arange(-3,3,0.1) #设定分割曲线的横坐标范围
    z = (-theta[0] - theta[1] * x)/theta[2]  #最佳拟合分割线，即决策边界
    plt.scatter(x_data0, y_data0, color='y', marker='.', label="label 0 data")
    plt.scatter(x_data1, y_data1, color='b', marker='x', label="label 1 data")
    plt.plot(x, z, 'r', color="r",label="decision boundary")
    plt.legend()
    plt.show()

if __name__=="__main__":
    x_data,y_data=loadDataSet() #读取数据
    data_len=len(x_data)
    x_train=x_data[0:round(data_len*0.8)] #前80%数据做训练集，后20%数据做测试集
    x_test=x_data[round(data_len*0.8):] #前80%数据做训练集，后20%数据做测试集
    y_train = y_data[0:round(data_len * 0.8)]
    y_test = y_data[round(data_len * 0.8):]
    alpha=0.01 #定义学习率
    maxcycles=10000 #定义最大迭代次数
    theta=log_regress(x_train,y_train,alpha,maxcycles)
    y_predict=predict(x_test,theta) #对测试集预测
    print(accuracy_score(y_test, y_predict)) #计算预测精确度
    showlinercurve(theta,x_data,y_data)