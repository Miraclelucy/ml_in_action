# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import tree
import graphviz # 使用conda安装graphviz，安装后需要将路径加到系统环境变量PATH中


# 生成决策树
def create_tree(trainingdata):
    data = trainingdata.iloc[:, :-1]  # 特征矩阵
    labels = trainingdata.iloc[:, -1]  # 标签
    trainedtree = tree.DecisionTreeClassifier(criterion="entropy")  # 分类决策树
    trainedtree.fit(data, labels)  # 训练
    return trainedtree


def showtree2pdf(trainedtree, finename):
    dot_data = tree.export_graphviz(trainedtree, out_file=None)  # 将树导出为Graphviz格式
    graph = graphviz.Source(dot_data)
    graph.render(finename) # 保存树图到文件，默认格式是pdf

def data2vectoc(data):
    names = data.columns[:-1]
    for i in names:
        col = pd.Categorical(data[i])
        data[i] = col.codes
    return data


if __name__ == "__main__":
    data = pd.read_table('tennis.txt', header=None, sep='\t')  # 读取训练数据
    trainingvec = data2vectoc(data)  # 向量化数据
    decisionTree = create_tree(trainingvec)  # 创建决策树
    showtree2pdf(decisionTree, "tennis")  # 图示决策树
    testVec = [0, 0, 1, 1]  # 天气晴、气温冷、湿度高、风力强
    print(decisionTree.predict(np.array(testVec).reshape(1, -1)))  # 预测
