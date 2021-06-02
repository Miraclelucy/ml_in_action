# -*- coding: utf-8 -*-
# 手写数字识别 - 利用sklearn框架中的Kneighborsclassifier实现knn分类
from os import listdir

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# 将32x32的二进制图像转换为1x1024的二维数组
def img2vector(filename):
    returnvect = np.zeros((1, 1024))  # 构建一个1行1024列的二维数组
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()  # 读取文件的一行
        for j in range(32):
            returnvect[0, 32 * i + j] = int(linestr[j])
    return returnvect


def getdata(data_dir):
    labels = []
    datafilelist = listdir(data_dir)  # 文件名构成的字符串列表['0_0.txt', '0_1.txt', ...]
    m = len(datafilelist)  # 训练集中m的值为1934
    data = np.zeros((m, 1024))
    for i in range(m):
        filenamestr = datafilelist[i]  # 获取完整的文件名
        filestr = filenamestr.split('.')[0]  # 获取文件名前缀
        classnumstr = int(filestr.split('_')[0])  # 获取这个文件的实际数字
        labels.append(classnumstr)
        data[i, :] = img2vector(data_dir + '/%s' % filenamestr)
    return data, labels # 返回数据和标签


if __name__ == "__main__":
    data_train, label_train = getdata('digits/trainingDigits')  # 获取训练集
    data_test, label_test = getdata('digits/testDigits')  # 获取测试集

    # 执行交叉验证
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        score = cross_val_score(knn, data_train, label_train, cv=10, scoring='accuracy')
        scores.append(score.mean())

    # 绘图显示k值与预测准确度的关系
    plt.plot(k_range, scores)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    # 选取交叉验证时最好结果的k值，开始预测
    k = scores.index(max(scores)) + 1
    print("model reaches max accuracy: %f when k = %d" % (max(scores), k))
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(data_train, label_train)
    label_predict = model.predict(data_test)
    print("total test data number:", len(data_test))
    print("the prediction accuracy is:", accuracy_score(label_test, label_predict))
