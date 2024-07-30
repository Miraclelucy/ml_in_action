# author:无形忍者
# time :2024-07-24
# description: 决策树对葡萄酒质量进行分类
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

plt.style.use('ggplot')
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# sep参数默认逗号
red_df = pd.read_csv('winequality-red.csv', sep=';')
white_df = pd.read_csv('winequality-white.csv', sep=';')
print("红酒数据集共有 {} 行 {} 列".format(red_df.shape[0],red_df.shape[1]))
print("白酒数据集共有 {} 行 {} 列".format(white_df.shape[0],white_df.shape[1]))
score = red_df.groupby("quality").agg({"fixed_acidity": lambda x: len(x)})
score = score.reset_index()
score.columns = ["quality","count"]

# 观察数据
sns.barplot(x = 'quality', y = 'count', data = score, hue="count", palette="rocket",legend=False)
#plt.show()

# 数据集区分“高质量红酒”与非“高质量红酒”
red_df["GoodWine"] = red_df.quality.apply(lambda x: "good" if x >=6 else 'not')
print(red_df.head()) # 展示前5条数据
print(red_df.columns) # 展示前5条数据

# 特征
X = np.array(red_df[red_df.columns[:11]])
# 分类标签
y = np.array(red_df.GoodWine)

# 划分训练集和测试集
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3)

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

# 开始建模 -- 没有使用编码的标签
clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=10,splitter="best")
clf = clf.fit(Xtrain, Ytrain)
# 画出训练模型
feature_names = red_df.columns[:11].values.tolist()
# show_tree(clf, feature_names, Ytrain)

# 探索重要特征
feature_importances_list= [*zip(feature_names,clf.feature_importances_)]
print(feature_importances_list)

# 返回预测的准确accuracy
score = clf.score(Xtest,Ytest)
print("模型准确率得分 {}".format(score))

# 模型优化-随机设置参数
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                  random_state=10,
                                 splitter="random",
                                  min_samples_leaf=10,
                                  min_samples_split=10)
clf = clf.fit(Xtrain, Ytrain)
# 返回预测的准确accuracy
score = clf.score(Xtest,Ytest)
print("模型优化-随机设置参数后，准确率得分 {}".format(score))

# 模型优化-roc曲线调参-max_depth
# test = []
# for i in range(15):
#     clf = tree.DecisionTreeClassifier(max_depth=i+1,criterion="entropy",random_state=10,splitter="random")
#     clf = clf.fit(Xtrain, Ytrain)
#     score = clf.score(Xtest, Ytest)
#     test.append(score)
# plt.plot(range(1,16),test,color="red",label="max_depth")
# plt.legend()
# plt.show()

# 模型优化-roc曲线调参-max_depth
# test = []
# for i in range(40):
#     clf = tree.DecisionTreeClassifier(min_samples_leaf=i+1,criterion="entropy",random_state=10,splitter="random")
#     clf = clf.fit(Xtrain, Ytrain)
#     score = clf.score(Xtest, Ytest)
#     test.append(score)
# plt.plot(range(1,41),test,color="red",label="min_samples_leaf")
# plt.legend()
# plt.show()

# 模型优化-网格优化-max_depth
# param_grid = {'max_depth':np.arange(1, 30, 1)}
#
# clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                  random_state=10,
#                                  splitter="random",
#                                   min_samples_leaf=10,
#                                   min_samples_split=10
#                                  )
# GS = GridSearchCV(clf,param_grid,cv=10)
# GS.fit(Xtrain, Ytrain)
# print("best_params_ {}".format(GS.best_params_))

# # 模型优化-网格优化-min_samples_leaf
# param_grid = {'min_samples_leaf':np.arange(1, 50, 5)}
#
# clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                   random_state=10,
#                                   splitter="random",
#                                   min_samples_split=10
#                                  )
# GS = GridSearchCV(clf,param_grid,cv=10)
# GS.fit(Xtrain, Ytrain)
# print("best_params_ {}".format(GS.best_params_))

# 模型优化-网格优化-min_samples_split
# param_grid = {'min_samples_split':np.arange(2, 50, 5)}
#
# clf = tree.DecisionTreeClassifier(criterion='entropy',
#                                  random_state=10,
#                                  max_depth=15,
#                                   splitter="random",
#                                   min_samples_leaf=36
#                                  )
# GS = GridSearchCV(clf,param_grid,cv=10)
# GS.fit(Xtrain, Ytrain)
# print("best_params_ {}".format(GS.best_params_))

#
clf = tree.DecisionTreeClassifier(criterion='entropy',
                                 random_state=10,
                                 max_depth=10,
                                  splitter="random",
                                  min_samples_leaf=36,
                                  min_samples_split=3
                                 )
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest,Ytest)
print("模型优化-网格参数搜索优化后，准确率得分 {}".format(score))
