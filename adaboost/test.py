from numpy import *
from adaboost import *
import rocPlotter

if __name__ == '__main__':
    # dataMat = matrix([[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]])
    # classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    # D = mat(ones((5, 1))/5)
    # testData = [[0, 0], [5, 5]]
    # classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
    # result = adaClassify(testData, classifierArr)
    # print('\ntestData: ', testData)
    # print('classify result: \n', result)

    dataArr, labelArr = loadDataSet('data/horseColicTraining2.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 300)
    testArr, testLabelArr = loadDataSet('data/horseColicTest2.txt')
    pridiction = adaClassify(testArr, classifierArray)
    print('\ntest prediction:\n', pridiction)
    m = shape(testLabelArr)[0]
    errArr = mat(ones((m, 1)))
    errorRate = errArr[pridiction!=mat(testLabelArr).T].sum() / (m*1.0)
    print('test errorRate: ', errorRate)
    rocPlotter.plotROC(aggClassEst.T, labelArr)


