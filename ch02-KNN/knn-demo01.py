
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1.准备数据
rawdata = {'电影名称': ['无问西东', '后来的我们', '前任3', '红海行动', '唐人街探案', '战狼'],
           '打斗镜头': [1, 5, 12, 108, 112, 115],
           '接吻镜头': [101, 89, 97, 5, 9, 8],
           '电影类型': ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片']}
movie_data = pd.DataFrame(rawdata)

# 2.分析并展示数据
plt.scatter(movie_data['打斗镜头'], movie_data['接吻镜头'])
plt.xlabel('打斗镜头')
plt.ylabel('接吻镜头')
plt.show()

# 3.演示KNN算法
new_data = [24, 67]
k = 3
dist = np.sum((movie_data.iloc[:6, 1:3] - new_data) ** 2, axis=1) ** 0.5
dist_label = pd.DataFrame({'dist': dist, 'label': movie_data.iloc[:6, 3]})

dr = dist_label.sort_values(by='dist')[:k]


# 4.KNN算法封装
def classify0(inx, dataset, k):
    # 1.求欧式距离
    dist = np.sum((dataset.iloc[:6, 1:3] - inx) ** 2, axis=1) ** 0.5
    # 2.距离和标签组合
    dist_label = pd.DataFrame({'dist': dist, 'label': movie_data.iloc[:6, 3]})
    # 3.按距离对标签排序，取最小的k个
    dr = dist_label.sort_values(by='dist')[:k]
    # 4.统计不同标签的个数
    re = dr.loc[:, 'label'].value_counts()
    # 5.返回数量最多的那个标签
    return re.index[0]


# 5.利用KNN进行电影分类
re = classify0(new_data, movie_data, 3)
