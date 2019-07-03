import svmMLiA
from numpy import *

if __name__ == '__main__':

    # smo simple test
    dataArr, labelArr = svmMLiA.loadDataSet('data/testSet.txt')
    b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas)


