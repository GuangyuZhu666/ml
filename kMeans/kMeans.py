from numpy import *
import matplotlib.pyplot as plt

# kMeans 算法

# 加载数据
def loadData(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(lambda x: float(x), curLine))
        dataMat.append(fltLine)
    return dataMat


# 计算两个向量的欧式距离
def distEclud(vecA, vecB):
    vecC = list(map(lambda x, y: x-y, vecA, vecB))
    return sqrt(sum(power(vecC, 2)))


# 构建簇质心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


# kMeans
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:  # 进行多轮循环直到元素分簇不再变化，即clusterAssment不再变化
        clusterChanged = False
        for i in range(m):  # 给每个元素分簇
            minDist = inf
            minIndex = -1
            for j in range(k):  # 对该元素计算和每个簇质心的距离
                distJI = distMeas(centroids[j], dataSet[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True  # 当变化时True,如果某轮这里没执行，while循环终止
            clusterAssment[i, :] = minIndex, minDist**2
        print('--- 进行一轮迭代计算出{}个簇的质心分布: ---'.format(k))
        print(centroids)
        for cent in range(k):  # 对每个簇重新计算质心
            a = nonzero(clusterAssment[:, 0].A == cent)[0]
            ptsInClust = dataSet[a]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# 二分K-均值聚类算法
def bitKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]  # 第一次只有一个簇时，簇心为均值
    centList = [centroid0]  # 放簇心
    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :]) ** 2  # 计算所有元素到质心的距离平方即SSE值
    numsDiv = 1
    while(len(centList) < k):  # 不停分簇，直到达到指定簇数
        print('--------- 开始第{0}次划分簇，当前簇数：{1}，目标簇数：{2} ---------'.format(numsDiv, len(centList), k))
        lowestSSE = inf
        for i in range(len(centList)):  # 尝试对每一簇进行划分、判断
            print('------ 开始尝试对第{0}簇进行划分 ------'.format(i))
            ptsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClusterAss = kMeans(ptsInCluster, 2, distMeas)  # 尝试对该簇一分为二
            sseSplit = sum(splitClusterAss[:, 1])  # 该簇一分为二后，划分后的这两簇的总sse值
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])  # 其他簇总sse值
            print('sseSplit，sseNotSplit，sse = {0},{1},{2}'.format(sseSplit, sseNotSplit, sseSplit+sseNotSplit))
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClusterAss.copy()
                lowestSSE = sseSplit + sseNotSplit
            print('------ 结束对第{0}簇尝试的划分 ------'.format(i))
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # 更新簇心分布；新簇心号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit  # 更新簇心分布；这部分还保留原来的簇心号
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # 更改簇心；原来的
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # 更新元素簇分布
        print('--------- 结束第{0}次划分簇，本次划分后结果如下 ---------'.format(numsDiv))
        print('the bestCentToSplit is : {0}'.format(bestCentToSplit))
        print('the length of best clustAss is : {0}'.format(len(bestClustAss)))
        print('cents numbers now is :{0}，aim:{1}.\n'.format(len(centList), k))
        numsDiv += 1
    print('！！！最终{0}个簇的质心分布 ！！！'.format(k))
    print(centList)
    return mat(centList), clusterAssment


# 画散点图分布
def plotter(dataSet, k, myCentroids, clusterAssing, xlim, ylim):

    colors = ['black', 'gray', 'orange', 'coral', 'blue', 'pink', 'purple', 'yellow', 'green', 'red']

    p1 = plt.subplot(2, 1, 1)
    p2 = plt.subplot(2, 1, 2)

    p1.set_title('original datas')
    p1.set_xlim(xlim[0], xlim[1])
    p1.set_ylim(ylim[0], ylim[1])
    p1.plot(dataSet[:, 0], dataSet[:, 1], 'bo')

    p2.set_title('clustered by kMeans k={0}'.format(k))
    p1.set_xlim(xlim[0], xlim[1])
    p1.set_ylim(ylim[0], ylim[1])

    for cent in range(k):  # 对每个簇重新计算质心
        xIndexs = nonzero(clusterAssing[:, 0].A == cent)[0]
        ptsInClust = dataSet[xIndexs]
        clusterColor = colors.pop()
        p2.scatter([x[0] for x in ptsInClust.tolist()], [x[1] for x in ptsInClust.tolist()], marker='o',
                   color=clusterColor)

    centX = [x[0] for x in myCentroids.tolist()]
    centY = [x[1] for x in myCentroids.tolist()]
    p2.scatter(centX, centY, marker='+', color='black')

    # p2.legend(loc='upper right')
    plt.show()



