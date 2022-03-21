# -*- coding: utf-8 -*-
'''
'''
from numpy import *
from os import listdir
import codecs
import jieba
import re
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from itertools import chain

def segment2word(doc: str):
    stop_words = codecs.open("stop_list.txt", "r", "UTF-8").read().splitlines()
    doc = re.sub('[\t\r\n]', ' ', doc)
    word_list = list(jieba.cut(doc.strip()))  #jieba分词
    out_str = ''
    for word in word_list:  #去停用词
        if word == ' ' or word == '':
            continue
        if word not in stop_words:
            out_str += word.strip()
            out_str += ' '
    segments = out_str.strip().split(sep=' ')
    return segments

def getDatafromDir(data_dir):
    docLists = []
    docLabels = [f for f in listdir(data_dir) if f.endswith('.txt')]
    for doc in docLabels:
        try:
            filepath=data_dir + "/" + doc
            wordList = segment2word(codecs.open(filepath, "r", "UTF-8").read())
            docLists.append(wordList)
        except:
            print("handling file %s is error!!" %filepath)
    return docLists

def spamEmailTest():
    spamDocList=getDatafromDir("./email/spam")  #读取垃圾邮件并分词
    hamDocList = getDatafromDir("./email/ham")  #读取正常邮件并分词
    fullDocList = spamDocList + hamDocList
    # 添加标签，垃圾邮件标记为1，正常邮件标记为0
    classList = array([1] * len(spamDocList)+[0]*len(hamDocList))

    # 获取前500个出现最频繁的词
    frequencyList = Counter(chain(*fullDocList))
    topWords = [w[0] for w in frequencyList.most_common(500)]

    # 以500个最频繁词为基础，以其中每个词在每份邮件中出现的次数建立向量
    vector = []
    for docList in fullDocList:
        vector.append(list(map(lambda x: docList.count(x), topWords)))
    vector = array(vector)

    model = MultinomialNB()  #选取贝叶斯MultinomialNB为训练模型
    model.fit(vector, classList)  #喂数据训练模型

    dataList=[]
    test_dir = "./email/test"
    docLabels = [f for f in listdir(test_dir) if f.endswith('.txt')]
    for doc in docLabels:
        try:
            filepath = test_dir + "/" + doc
            dataList = segment2word(codecs.open(filepath, "r", "UTF-8").read())
        except:
            print("handling file %s is error!!" % filepath)
        testVector = array(tuple(map(lambda x: dataList.count(x), topWords)))
        predicted_label = model.predict(testVector.reshape(1, -1))  #执行预测
        if(predicted_label == 1):
            print("%s is spam mail" %doc)
        else:
            print("%s is NOT spam mail" % doc)

if __name__=="__main__":
    spamEmailTest()
