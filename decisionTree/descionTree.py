from math import log
import operator
import tree_plotter
import time
import os

# 决策树算法


# 计算香农熵值,即该样本集合的纯度
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按给定特征划分数据集
# axis：第几个特征；value；该特征值是多少
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择当前最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEnt = calcShannonEnt(dataSet)
    baseInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 获得该属性对应几个分类值
        newEnt = 0.0
        for value in uniqueVals:  # 对该属性每个分类值所对应的样本集合求熵并最终加权
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEnt += prob * calcShannonEnt(subDataSet)
        infoGain = baseEnt - newEnt  # 信息增益
        if(infoGain > baseInfoGain):
            baseInfoGain = infoGain
            bestFeature = i
    return bestFeature


# 返回该集合中值最多的label
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassConut = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassConut[0][0]


# 创建决策树
# 决策树停止条件：1、当前集合下label类别完全相同，返回该label
#               2、遍历完所有特征值时，当下label类别还没有完全相同，选择最多的label值
# 递归树是一个字典
def createTree(dataSet, labels):
    featLabels = labels[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 条件1
        return classList[0]
    if len(dataSet[0]) == 1:  # 条件2   当len(dataSet[0])==1时，说明当下只剩一个特征属性，不用再进一步划分了
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择当前最优划分属性
    bestFeatLabel = featLabels[bestFeat]
    myTree = {bestFeatLabel: {}}  # 在当前最优属性为key，其value也是一个字典，对应该属性的多个值
    del(featLabels[bestFeat])  # 从label集合中删除当前选择的属性
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:  # 对每个分支集合，再递归创建
        sublabels = featLabels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value),
            sublabels)
    return myTree


# 对传进来的testVec进行分类
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = featLabels[0]
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if secondDict[key].__class__.__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


# 使用pickle模块来存储决策树
def storeTree(inputTree, fileName):
    import pickle
    fw = open(fileName, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


# 提取决策树
def grabTree(fileName):
    import pickle
    fr = open(fileName, "rb")
    return pickle.load(fr)


def createTestDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


