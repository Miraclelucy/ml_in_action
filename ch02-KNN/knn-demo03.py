# -*- coding: utf-8 -*-
# 手写数字识别 - 利用pandas处理数据，sklearn框架中的Kneighborsclassifier实现knn分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1.准备数据
def get_data(path):
    fileList = os.listdir(path)
    data = pd.DataFrame()
    img = []
    lables = []
    for i in range(len(fileList)):
        filename = fileList[i]  # 获取文件名
        txt = pd.read_csv(path + f'/{filename}', header=None)
        num = ''
        for j in range(txt.shape[0]):
            num += txt.iloc[i,:]
        img.append(num[0])
        filelabel = filename.split('_')[0] # 获取真实数字
        lables.append(filelabel)
    data['img'] = img
    data['lables'] = lables
    return data

# 2.分析并展示数据
