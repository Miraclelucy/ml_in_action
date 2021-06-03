# 约会分类 - 从零实现KNN算法，并进行约会数据分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1.准备数据
dataTesting = pd.read_table('datingTestSet.txt', header=None)
dataTesting.head()  # 默认返回前5行的数据

# 2.分析并展示数据
Colors = []
for i in range(dataTesting.shape[0]):
    m = dataTesting.iloc[i, -1]
    if m == 'largeDoses':
        Colors.append('red')
    if m == 'smallDoses':
        Colors.append('orange')
    if m == 'didntLike':
        Colors.append('black')

pl = plt.figure(figsize=(12, 8))
fig1 = pl.add_subplot(221)
plt.scatter(dataTesting.iloc[:, 0], dataTesting.iloc[:, 1], marker='.', c=Colors)
plt.xlabel('每年飞行里程数')
plt.ylabel('玩游戏视频所占时间比')

fig2 = pl.add_subplot(222)
plt.scatter(dataTesting.iloc[:, 0], dataTesting.iloc[:, 2], marker='.', c=Colors)
plt.xlabel('每年飞行里程数')
plt.ylabel('每周消费冰淇淋公升数')

fig3 = pl.add_subplot(223)
plt.scatter(dataTesting.iloc[:, 1], dataTesting.iloc[:, 2], marker='.', c=Colors)
plt.xlabel('玩游戏视频所占时间比')
plt.ylabel('每周消费冰淇淋公升数')
plt.show()

# 示例：一张figure里面生成多张子图
x = np.arange(0, 100)
pl2 = plt.figure(figsize=(12, 8))
ax1 = pl2.add_subplot(3, 1, (1, 2))
ax1.plot(x, x ** 2)
ax2 = pl2.add_subplot(3, 1, 3)
ax2.plot(x, x)
plt.show()


# 3.数据归一化有多种方法，如0-1标准化、Z-score标准化、Sigmoid压缩法等。这里采用0-1标准化
def autonorm(dataset):
    min = dataset.min()
    max = dataset.max()
    res = (dataset - min) / (max - min)
    return res


dataTestingNorm = pd.concat([autonorm(dataTesting.iloc[:, 0:3]), dataTesting.iloc[:, 3]], axis=1)
dataTestingNorm.head()


# 4.切分训练集和测试集
def randSplit(dataset, rate=0.9):
    m = int(dataset.shape[0] * rate)
    traindata = dataset.iloc[:m, :]
    testdata = dataset.iloc[m:, :]
    return traindata, testdata


traindata, testdata = randSplit(dataTestingNorm)


# 5.利用KNN算法进行约会数据分类
def datingClass(train, test, k):
    n = train.shape[1] - 1
    m = test.shape[0]
    count = 0
    for i in range(m):
        dist = np.sum((train.iloc[:, : n] - test.iloc[i, :n]) ** 2, axis=1) ** 0.5 # 求欧式距离
        dist_l = pd.DataFrame({'dist': dist, 'label': train.iloc[:, n]})
        dr = dist_l.sort_values(by='dist')[:k]
        re = dr.loc[:, 'label'].value_counts()
        if re.index[0] == test.iloc[i, n]:
            count += 1
    print(f'模型预测准确率为{count / m}')
    return count / m


datingClass(traindata, testdata, 5)
