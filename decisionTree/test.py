from descionTree import *

if __name__ == '__main__':
    # 测试1
    testDataSet, labels = createTestDataSet()
    testTree = createTree(testDataSet, labels)
    # fileName = time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime(time.time())) + "_" + 'tree.txt'
    # fullpath = os.path.join("trees", fileName)
    # storeTree(testTree, fullpath)
    # print(grabTree(fullpath))
    print(testTree)
    tree_plotter.createPlot(testTree)
    print(classify(testTree, labels, [1, 1]))
    print(classify(testTree, labels, [0, 0]))
    print(classify(testTree, labels, [1, 0]))
    print(classify(testTree, labels, [0, 1]))

    # 测试2；隐形眼镜数据集测试
    # fr = open('data/lenses.txt')
    # lenses = [inst.strip().split("\t") for inst in fr.readlines()]
    # lenses_labels = ['age', 'prescript', 'astigmatic', 'tear_rate']
    # lenses_tree = createTree(lenses, lenses_labels)
    # print(lenses_tree)
    # tree_plotter.createPlot(lenses_tree)




