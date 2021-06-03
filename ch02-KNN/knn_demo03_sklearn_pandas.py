# -*- coding: utf-8 -*-
# 手写数字识别 - 利用pandas处理数据，sklearn框架中的Kneighborsclassifier实现knn分类
# 本示例运行时间约需要1分钟，pandas处理数据还是太慢了，待后续优化。
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# 1.准备数据
def get_data(path):
    datafilelist = os.listdir(path)
    img = np.zeros((len(datafilelist), 1024))  # # 训练集中len(datafilelist)的值为1934
    lables = []  # lables是一个列表,训练集中len(lables)是1934，也是文件的总数
    for i in range(len(datafilelist)):
        filename = datafilelist[i]  # 获取文件名
        txt = pd.read_csv(path + f'/{filename}', header=None)  # txt是32x1的数组，即32行1列，每一列是长度为32的0-1字符串
        num = np.zeros((1, 1024))  # 用来存储构成一个数字的所有二进制数据
        for m in range(32):  # 
            for n in range(32):  # 
                num[0, 32 * m + n] = txt.iloc[m, 0][n]  # 
        img[i, :] = num  # 给1934x1024二位数组的每一行赋值
        filelabel = filename.split('_')[0]  # 获取真实数字
        lables.append(filelabel)
    return img, lables


if __name__ == "__main__":
    start_time = time.perf_counter()
    train_img, train_lables = get_data("digits/trainingDigits")
    test_img, test_lables = get_data("digits/testDigits")

    # 交叉验证 获取最优的K
    k_range = range(1, 20)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        score = cross_val_score(knn, train_img, train_lables, cv=10, scoring='accuracy')
        scores.append(score.mean())

    # 绘图显示k值与预测准确度的关系
    plt.plot(k_range, scores)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    # 利用KNN算法实现手写数字识别
    k = scores.index(max(scores)) + 1
    print("model reaches max accuracy: %f when k = %d" % (max(scores), k))
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(train_img, train_lables)
    predict_label = model.predict(test_img)
    print("total test data number:", len(test_img))
    print("the prediction accuracy is:", accuracy_score(test_lables, predict_label))

    end_time = time.perf_counter()
    print('Running time: %s Seconds' % (end_time - start_time))
