# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 增加一个x0特征默认为1，后面其对应的权重即系数也为1
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(x):
    return 1.0/(1+exp(-x))


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


# 梯度上升算法
# maxCycles次迭代更新回归系数，每次迭代都要处理整个数据集
def gredAscnt(dataMat, classLabels):
    print('采用梯度上升算法！！！\n')
    dataMatrix = mat(dataMat)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001  # 步长
    maxCycles = 300  # 循环次数
    weights = ones((n, 1))  # 系数向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error  # 梯度上升法更新系数，w = w + alpha * grad(f(w))
        if (k+1) % 100 == 0:
            print('已经过', k+1, '轮迭代，当前回归系数:\n', weights, '\n')
    weightsList = [x[0] for x in weights.tolist()]  # 将系数矩阵转化为一维系数列表
    return weightsList


# 随机梯度上升算法
# m(样本数量)次迭代，一次只处理一个样本的数据来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + [alpha * error * x for x in dataMatrix[i]]
    print('采用随机梯度上升算法，回归系数:\n', weights)
    return weights


# 改进的随机梯度上升算法
# 增加了迭代次数；每次对alpha进行了调整
def stocGradAscent1(dataMatrix, classLabels, numIter=200):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + i + j) + 0.01  # 每次迭代调整，缓解系数的高频波动
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机取样本来更新回归系数，减少周期性波动;取完后从列表中删除
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + [alpha * error * x for x in dataMatrix[randIndex]]
            dataIndex.remove(dataIndex[randIndex])
    print('采用改进的随机梯度上升算法，回归系数:\n', weights)
    return weights


# 画出最佳拟合的函数直线;2-dimension
def plotBestFit(dataMat, labelMat, weights):
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = ((-weights[0] - weights[1] * x) / weights[2])
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


