from kNN import *

if __name__ == '__main__':
    trains, trains_labels = createDataSet()
    # test = [0, 0.2]
    k = 3
    test = []
    while(True):
        param0 = input('输入第一个特征值：')
        if(param0 == 'exit'):
            break
        param1 = input('输入第二个特征值：')
        test.append(float(param0))
        test.append(float(param1))
        print('{0}的kNN预测(k={1})结果是'.format(test, k) + classify(test, trains, trains_labels, k))
        test.clear()