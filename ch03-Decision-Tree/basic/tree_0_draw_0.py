# -*- coding: utf-8 -*-
# 去不去打羽毛球问题 - 从零实现决策树并进行分类；从零绘制树
# 本示例运行时间约需要5秒钟
from math import log  # 对数函数
import operator
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False


def calc_shannonent(dataset):  # 计算熵
    numentries = len(dataset)
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys(): labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1  # 统计每个标签的数据总数
    shannonent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / numentries  # 每个标签的数据总数/所有数据总数
        shannonent -= prob * log(prob, 2)  # 对数函数
    return shannonent


def split_dataset(dataset, axis, value):
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:  # 统计某一条件下的子数据集
            reducedfeatvec = featvec[:axis]
            reducedfeatvec.extend(featvec[axis + 1:])  # 把除了特征值value之外的其他特征值拼接起来
            retdataset.append(reducedfeatvec)
    return retdataset


def choose_bestfeature_tosplit(dataset):
    numfeatures = len(dataset[0]) - 1  # 最后一列是标签
    baseentropy = calc_shannonent(dataset)  # 计算信息熵
    bestinfogain = 0.0
    bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in
                    dataset]  # 数据集所有行的第i个特征，比如i=0时，值为列表['晴', '晴', '阴', '雨', '雨', '雨', '阴', '晴', '晴', '雨', '晴', '阴', '阴', '雨']
        uniquevals = set(featlist)  # 列举第i个特征的所有值，比如i=0时，值为集合{'晴', '阴', '雨'}
        newentropy = 0.0
        for value in uniquevals:  # 遍历特征集合中的所有值
            subdataset = split_dataset(dataset, i, value)  # 获取某一条件下的子数据集
            prob = len(subdataset) / float(len(dataset))
            newentropy += prob * calc_shannonent(subdataset)  # 计算条件熵
        infogain = baseentropy - newentropy  # 计算信息增益 = 信息熵- 条件熵
        if infogain > bestinfogain:
            bestinfogain = infogain
            bestfeature = i
    return bestfeature  # 返回被选中的特征


def majority_cnt(classlist):
    classcount = {}
    for vote in classlist:
        if vote not in classcount.keys(): classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def createtree(dataset, labels):  # 生成决策树
    classlist = [example[-1] for example in dataset]  # 取出标签
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]  # 当所有数据的类别都一样的是时候停止分裂
    if len(dataset[0]) == 1:
        return majority_cnt(classlist)  # 当没有更多特征的时候停止分裂
    bestfeat = choose_bestfeature_tosplit(dataset)
    bestfeatlabel = labels[bestfeat]  # 获取最好的特征对应的值
    mytree = {bestfeatlabel: {}}
    del (labels[bestfeat])
    featvalues = [example[bestfeat] for example in dataset]
    uniquevals = set(featvalues)
    for value in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlabel][value] = createtree(split_dataset(dataset, bestfeat, value), sublabels)
    return mytree


def classify(inputtree, featlabels, testvec):  # 决策树执行预测
    firststr = list(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)
    key = testvec[featindex]
    valueoffeat = seconddict[key]
    if isinstance(valueoffeat, dict):
        classlabel = classify(valueoffeat, featlabels, testvec)
    else:
        classlabel = valueoffeat
    return classlabel


def getnumleafs(mytree):
    numleafs = 0
    firststr = list(mytree.keys())[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':  # 节点是不是字典类型, 不是的话就是叶子节点
            numleafs += getnumleafs(seconddict[key])
        else:
            numleafs += 1
    return numleafs


def gettreedepth(mytree):
    maxdepth = 0
    firststr = list(mytree.keys())[0]
    seconddict = mytree[firststr]
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':  # 节点是不是字典类型, 不是的话就是叶子节点
            thisdepth = 1 + gettreedepth(seconddict[key])
        else:
            thisdepth = 1
        if thisdepth > maxdepth: maxdepth = thisdepth
    return maxdepth


def plotnode(nodetxt, centerpt, parentpt, nodetype):
    arrow_args = dict(arrowstyle="<-")
    createplot.ax1.annotate(nodetxt, xy=parentpt, xycoords='axes fraction',
                            xytext=centerpt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodetype, arrowprops=arrow_args)


def plotmidtext(cntrpt, parentpt, txtstring):
    xmid = (parentpt[0] - cntrpt[0]) / 2.0 + cntrpt[0]
    ymid = (parentpt[1] - cntrpt[1]) / 2.0 + cntrpt[1]
    createplot.ax1.text(xmid, ymid, txtstring, va="center", ha="center", rotation=30)


def plottree(mytree, parentpt, nodetxt):
    decisionnode = dict(boxstyle="sawtooth", fc="0.8")
    leafnode = dict(boxstyle="round4", fc="0.8")
    numleafs = getnumleafs(mytree)
    firststr = list(mytree.keys())[0]
    cntrpt = (plottree.xoff + (1.0 + float(numleafs)) / 2.0 / plottree.totalw, plottree.yoff)
    plotmidtext(cntrpt, parentpt, nodetxt)
    plotnode(firststr, cntrpt, parentpt, decisionnode)
    seconddict = mytree[firststr]
    plottree.yoff = plottree.yoff - 1.0 / plottree.totald
    for key in seconddict.keys():
        if type(seconddict[key]).__name__ == 'dict':  # 节点是不是字典类型, 不是的话就是叶子节点
            plottree(seconddict[key], cntrpt, str(key))
        else:  # 打印叶子节点
            plottree.xoff = plottree.xoff + 1.0 / plottree.totalw
            plotnode(seconddict[key], (plottree.xoff, plottree.yoff), cntrpt, leafnode)
            plotmidtext((plottree.xoff, plottree.yoff), cntrpt, str(key))
    plottree.yoff = plottree.yoff + 1.0 / plottree.totald


def createplot(intree):
    fig = plt.figure(1, facecolor='yellow')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createplot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plottree.totalw = float(getnumleafs(intree))
    plottree.totald = float(gettreedepth(intree))
    plottree.xoff = -0.5 / plottree.totalw
    plottree.yoff = 1.0
    plottree(intree, (0.5, 1.0), '')
    plt.title("decision tree")
    plt.show()


if __name__ == "__main__":
    fr = open('tennis.txt', encoding='UTF-8')  # 读取训练数据
    trainingdata = [inst.strip().split('\t') for inst in fr.readlines()]
    print(type(trainingdata[0]))
    labels = ['天气', '气温', '湿度', '风力']  # 特性的名称
    decisiontree = createtree(trainingdata, labels)  # 生成决策树
    createplot(decisiontree)  # 图示决策树
    labels = ['天气', '气温', '湿度', '风力']
    testdata = ['晴', '冷', '高', '强']  # 天气晴、气温冷、湿度高、风力强
    print(classify(decisiontree, labels, testdata))  # 对testVec执行预测并打印结果
