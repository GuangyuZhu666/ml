# coding=utf-8
from kMeans import *


if __name__ == '__main__':

    k = 4

    # dataSet = mat(loadData('data/testSet.txt'))
    # myCentroids, clusterAssing = kMeans(dataSet, k)

    dataSet = mat(loadData('data/testSet2.txt'))
    myCentroids, clusterAssing = bitKmeans(dataSet, k)

    xlim = [-6, 6]
    ylim = [-6, 6]
    plotter(dataSet, k, myCentroids, clusterAssing, xlim, ylim)




