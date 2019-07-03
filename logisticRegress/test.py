#coding=utf-8
from logisticRegress import *


# 野马死亡率预测案例
def colicTest():
    frTrain = open('data/horseColicTraining.txt')
    frTest = open('data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f\n" % errorRate)
    return errorRate


def multiTest():
    print('------ 野马死亡率预测案例 ------')
    numTests = 3
    errorSum = 0.0
    for k in range(numTests):
        print('--- test%d ---' % (k+1))
        errorSum += colicTest()
    print("------ 最终测试结果 ------\nafter %d iterations, the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == '__main__':

    # dataMat, labelMat = loadDataSet()
    # # weightResult = gredAscnt(dataMat, labelMat)
    # # weightResult = stocGradAscent0(dataMat, labelMat)
    # weightResult = stocGradAscent1(dataMat, labelMat, 200)
    # plotBestFit(dataMat, labelMat, weightResult)

    # 野马死亡率测试
    multiTest()




