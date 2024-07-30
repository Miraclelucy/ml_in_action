# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 生成决策树
def create_tree(trainingdata):
    data = trainingdata.iloc[:, :-1]  # 特征矩阵
    labels = trainingdata.iloc[:, -1]  # 标签
    trainedtree = tree.DecisionTreeClassifier(criterion="entropy")  # 分类决策树
    trainedtree.fit(data, labels)  # 训练
    data_feature_names = ["年龄", "有工作", "有自己的房子", "信贷情况"]
    data_labels = labels.unique()
    show_tree(trainedtree, data_feature_names, data_labels)
    return trainedtree

def data2vectoc(data):
    names = data.columns[:-1]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes
    return data

def show_tree(model,feature_names,labels):
    # 用图片画出
    plt.figure(figsize=(15, 10))  #
    a = tree.plot_tree(model,
                       feature_names=feature_names,
                       class_names=labels,
                       rounded=True,
                       filled=True,
                       fontsize=16)
    plt.show()

if __name__ == "__main__":
    data = pd.read_table('bank.txt', header=None, sep='\t')  # 读取训练数据
    trainingvec = data2vectoc(data)  # 向量化数据
    decisionTree = create_tree(trainingvec)  # 创建决策树
    testVec = [0, 0, 1, 1]  # 中年、否、是、好
    print(decisionTree.predict(np.array(testVec).reshape(1, -1)))  # 预测
