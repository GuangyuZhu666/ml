# coding=utf-8

from numpy import *


# 根据某个属性对数据划分，阈值两边的数据不同label
# dimen，某特征列；threshVal，划分阈值；threshIneq,比较规则
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


# 单层决策树生成函数
# 错误率最小的单层决策树
# D(m,1)向量: 每个数据的权重
def builStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j)*stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print('split: dim %d, thresh %.2f, thresh ineq: %s, the weighted error is %3.f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=10):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    print('-------------------------- Train begins !!! ---------------------------\n')
    for i in range(numIt):
        print('------------------------ Number of cycles : %d ------------------------' % (i+1))
        bestStump, error, classEst = builStump(dataArr, classLabels, D)
        # print('bestStump: ', bestStump)
        # print('error: ', error)
        # print('classEst: \n', classEst)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))  # alpha = 1/2 * ln((1-e)/e);  max(...)防止错误率为0时除数为0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print('mat(classLabels).T: ', mat(classLabels).T)
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)  # 为下一次迭代计算新的数据权重分布
        # print('D: ', D)                                # if该样本被正确分类，调整权重 D(t+1) = D(t)*exp(-alpha) / sum(D)
        D = multiply(D, exp(expon))                    # if该样本被错误分类，调整权重 D(t+1) = D(t)*exp(alpha) / sum(D)
        D = D/D.sum()
        aggClassEst += alpha * classEst  # 记录每个数据点的类别估计累计值
        # print('aggClassEst: \n', aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # sign(), >0的值都设为1，<0的值设为-1
        errorRate = aggErrors.sum()/m
        print('total error: {}\n'.format(errorRate))
        if(errorRate == 0.0):
            break
    print('---------- train finished !!!!!! ----------\nfinal stumps:', weakClassArr)
    return weakClassArr, aggClassEst


# 对单个数据判断分类
def adaClassify(dataToClass, classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    numFeature = len(open(fileName).readline().split('\t'))
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeature-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
