# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

def model_svm(Xdata, ylabel, kernel, C,gamma='scale'):
    # 将数据按0和1两个类别分开，以便用不同的颜色和图形标识
    X0 = [] #存储类别为0的数据
    X1 = [] #存储类别为1的数据
    for i in range(len(ylabel)):
        if int(ylabel[i]) == 1: #类别为1
            X1.append(Xdata[i])
        else: #类别为0
            X0.append(Xdata[i])
    X0 = np.array(X0)
    X1 = np.array(X1)

    '''
    SVM模型，参数C是惩罚系数, 即对误差的宽容度。C越高，说明越不能容忍出现误差,容易过拟合。C越小，容易欠拟合。C过大或过小，泛化能力变差。
    C与松弛变量的关系：减小惩罚系数C，松弛变量会变大。    
    gamma是kernel选择'rbf','poly'的核函数参数。隐含地决定了数据映射到新的特征空间后的分布，gamma越大，支持向量越少，
    gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度。
    '''
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    model = clf.fit(Xdata, ylabel)
    draw_svm(clf,X0,X1)

    return model  #返回训练模型

def draw_svm(model,X0,X1):
    # 画出所有的样本点
    plt.title('Support Vector Machine / C = %s, gamma= %s' %(model.get_params()['C'],model.get_params()['gamma']))
    plt.scatter(X0[:, 0], X0[:, 1], c='g', marker='s', label="class 0")
    plt.scatter(X1[:, 0], X1[:, 1], c='b', label="class 1")

    ax = plt.gca()
    xlimit = ax.get_xlim()
    ylimit = ax.get_ylim()
    xx = np.linspace(xlimit[0], xlimit[1], 100)
    yy = np.linspace(ylimit[0], ylimit[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    H = ax.contour(XX, YY, Z, colors='r', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.clabel(H, inline=True, fontsize=10)
    plt.legend()
    plt.show()

def predict_svm(model,data,label):
    ypred = model.predict(data)
    print("real label: ", label)
    print("pred label: ", ypred)
    print("prediction accuracy score: ", model.score(data, label))

if __name__=="__main__":
    # 读取线性SVM数据
    data = pd.read_csv('lineardata.csv', header=None)
    data_train = data.sample(frac=0.8) #80%的数据作为训练集
    data_test = data[~data.index.isin(data_train.index)] #剩下的20%数据作为测试集

    #提取训练集数据和标签
    Xdata = data_train.values[:, :2]
    ylabel = data_train.values[:, 2]

    #线性SVM，参数C分别取不同值，观察区别.
    #线性SVM还有LinearSVC方法，可以尝试看看跟SVC(kernel='linear')有什么区别
    clf=model_svm(Xdata, ylabel, kernel='linear', C=0.0001)
    predict_svm(clf, data_test.values[:, :2],data_test.values[:, 2])
    clf=model_svm(Xdata, ylabel, kernel='linear', C=1)
    predict_svm(clf, data_test.values[:, :2],data_test.values[:, 2])

    #读取多项式SVM数据
    data = pd.read_csv('polydata.csv', header=None)
    data_train = data.sample(frac=0.8)  # 80%的数据作为训练集
    data_test = data[~data.index.isin(data_train.index)]  # 剩下的20%数据作为测试集
    Xdata = data.values[:, :2]
    ylabel = data.values[:, 2]

    # 多项式SVM，参数C分别取不同值，观察区别
    clf = model_svm(Xdata, ylabel, kernel='poly',C=0.1)
    predict_svm(clf, data_test.values[:, :2],data_test.values[:, 2])
    clf = model_svm(Xdata, ylabel, kernel='poly', C=1)
    predict_svm(clf, data_test.values[:, :2],data_test.values[:, 2])

    # 生成200个训练数据集和40个测试数据集，2维正态分布，2个样本特征，协方差系数为3，生成的数据按分位数分为两类
    Xdata, ylabel = make_gaussian_quantiles(n_samples=200, n_features=2, n_classes=2, cov=3)
    Xdata_test, ylabel_test = make_gaussian_quantiles(n_samples=40, n_features=2, n_classes=2, cov=3)

    # 基于高斯核函数的SVM，参数C分别取不同值，观察区别
    clf = model_svm(Xdata, ylabel, kernel='rbf', C=0.1,gamma='auto')
    predict_svm(clf, Xdata_test,ylabel_test)
    clf = model_svm(Xdata, ylabel, kernel='rbf', C=1,gamma='auto')
    predict_svm(clf, Xdata_test, ylabel_test)



