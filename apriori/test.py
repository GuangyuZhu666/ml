#coding=utf-8

from apriori import *

dataSet = loadDataSet()
print('测试数据:', dataSet)
L, suppData = apriori(dataSet, minSupport=0.5)
print('候选项集:', L)
print('支持度表:', suppData)
print('关联规则:')
rules = generateRules(L, suppData, minConf=0.7)
print(rules)


# 例子：找毒蘑菇的公共特征，数据中包含特征值2的为毒蘑菇，我们只需找到包含2的频繁项集
# mushDataSet = [line.split() for line in open('data/mushroom.dat').readlines()]
# L, suppData = apriori(mushDataSet, minSupport=0.3)
# for item in L[3]:
#     if item.intersection('2'):
#         print(item, 'conf', suppData[item])
