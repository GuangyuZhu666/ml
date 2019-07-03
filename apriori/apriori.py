# coding=utf-8
from numpy import *


def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


# 创建只包含一个元素的候选项集的集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))  # use frozenset so we can use it as a key in a dict.
                               # [ frozenset([1]), frozenset([2]), ... ]


# 在数据集D中计算Ck序列中每个候选项的支持度
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData  # retList,满足最小支持度的候选项集；supportData,所有候选项的支持度


# 两两合并，将大小为k-1的两个合并成大小为k的候选项集
def aprioriGen(Lk, k):  # creates Ck
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # if first k-2 elements are equal
                retList.append(Lk[i] | Lk[j])  # set union
    return retList


# 满足最小支持度要求的候选项集L ,例: [ [ frozenset([0]), frozenset([1]), frozenset([2]) ],                 L[0]
#                                     [  frozenset([0, 1]), frozenset([1, 2]) ],                          L[1]
#                                     [  frozenset([0, 1, 2]) ],                                          L[2]
#                                     [ ]                                                  ]              L[3]最后为空
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0): # 当上一个候选项项集不为空，当前就可以继续合并
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = scanD(D, Ck, minSupport)  # scan DB to get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


# 关联规则生成函数
def generateRules(L, supportData, minConf=0.7):  # supportData is a dict coming from scanD
    bigRuleList = []
    for i in range(1, len(L)):  # only get the sets with two or more items
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  # freqSet: frozeset([0, 1]) => H1 :[frozenset([0]), frozenset([1])]
            if (i > 1):  # 超过两个以上元素的候选项集
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 两元素候选项集,直接计算
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


# 对每个候选项集，计算所有可能的规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = []  # create new list to return
    for conseq in H:  # (freqSet - conseq)  --》 conseq
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # calc confidence
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH  # 之后合并用的


# 由上一层的规则，合并到下一层的规则
#       23->01      13->02      12->03           合并尾巴
#             3->012      1->023
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # try further merging
        Hmp1 = aprioriGen(H, m + 1)  # create Hm+1 new candidates 得到该候选项集所可以得到的所有长度为m+1的序列
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):  # need at least two sets to merge
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)






# def pntRules(ruleList, itemMeaning):
#     for ruleTup in ruleList:
#         for item in ruleTup[0]:
#             print
#             itemMeaning[item]
#         print
#         "           -------->"
#         for item in ruleTup[1]:
#             print
#             itemMeaning[item]
#         print
#         "confidence: %f" % ruleTup[2]
#         print  # print a blank line


# from time import sleep
# from votesmart import votesmart
#
# votesmart.apikey = 'a7fa40adec6f4a77178799fae4441030'
#
#
# # votesmart.apikey = 'get your api key first'
# def getActionIds():
#     actionIdList = [];
#     billTitleList = []
#     fr = open('recent20bills.txt')
#     for line in fr.readlines():
#         billNum = int(line.split('\t')[0])
#         try:
#             billDetail = votesmart.votes.getBill(billNum)  # api call
#             for action in billDetail.actions:
#                 if action.level == 'House' and \
#                         (action.stage == 'Passage' or action.stage == 'Amendment Vote'):
#                     actionId = int(action.actionId)
#                     print
#                     'bill: %d has actionId: %d' % (billNum, actionId)
#                     actionIdList.append(actionId)
#                     billTitleList.append(line.strip().split('\t')[1])
#         except:
#             print
#             "problem getting bill %d" % billNum
#         sleep(1)  # delay to be polite
#     return actionIdList, billTitleList
#
#
# def getTransList(actionIdList, billTitleList):  # this will return a list of lists containing ints
#     itemMeaning = ['Republican', 'Democratic']  # list of what each item stands for
#     for billTitle in billTitleList:  # fill up itemMeaning list
#         itemMeaning.append('%s -- Nay' % billTitle)
#         itemMeaning.append('%s -- Yea' % billTitle)
#     transDict = {}  # list of items in each transaction (politician)
#     voteCount = 2
#     for actionId in actionIdList:
#         sleep(3)
#         print
#         'getting votes for actionId: %d' % actionId
#         try:
#             voteList = votesmart.votes.getBillActionVotes(actionId)
#             for vote in voteList:
#                 if not transDict.has_key(vote.candidateName):
#                     transDict[vote.candidateName] = []
#                     if vote.officeParties == 'Democratic':
#                         transDict[vote.candidateName].append(1)
#                     elif vote.officeParties == 'Republican':
#                         transDict[vote.candidateName].append(0)
#                 if vote.action == 'Nay':
#                     transDict[vote.candidateName].append(voteCount)
#                 elif vote.action == 'Yea':
#                     transDict[vote.candidateName].append(voteCount + 1)
#         except:
#             print
#             "problem getting actionId: %d" % actionId
#         voteCount += 2
#     return transDict, itemMeaning