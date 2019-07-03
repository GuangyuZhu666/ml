# coding=utf-8
from sklearn.datasets import load_digits  # 数据集
from sklearn.preprocessing import LabelBinarizer  # 标签二值化
from sklearn.model_selection import train_test_split  # 数据集分割
import numpy as np
import pylab as pl  # 数据可视化


# 激活函数sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sigmoid函数的导数
def dsigmoid(x):
    return x * (1 - x)


class NeuralNetwork:

    # 初始化权值，范围 -1 ~ 1
    def __init__(self, layers):  # 这里是三层网络，列表[64,100,10]表示输入，隐藏，输出层的单元个数

        self.V = np.random.random((layers[0] + 1, layers[1])) * 2 - 1  # 隐藏层权值(65,100)，之所以是65，因为有偏置对应的权重W
        self.W = np.random.random((layers[1], layers[2])) * 2 - 1  # (100,10)

    # lr为学习率，epochs为迭代的次数
    def train(self, X, y, X_test, y_test, lr=0.1, epochs=10000):
        temp = np.ones([X.shape[0], X.shape[1] + 1])  # 为数据集添加偏置
        temp[:, 0:-1] = X
        X = temp  # 这里最后一列为偏置

        # 进行权值训练更新
        for n in range(epochs + 1):
            i = np.random.randint(X.shape[0])  # 每次从训练集中随机选取一行数据(一个样本)进行训练，更新权值
            x = X[i]
            x = np.atleast_2d(x)  # 转为二维数据

            L1 = sigmoid(np.dot(x, self.V))  # 点积;  隐层输出 = sigmoid（输入层输入 x 权重） , (1,64)*(64,100) = (1,100)
            L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出 = sigmoid(隐层输出 x 权重) , (1,100)*(100,10) = (1,10)

            # delta
            L2_delta = (y[i] - L2) * dsigmoid(L2)  # 输出层梯度项Gj, (1,10)
            L1_delta = L2_delta.dot(self.W.T) * dsigmoid(L1)  # 隐层梯度项Eh, (1,100)

            # 更新权重
            self.W += lr * L1.T.dot(L2_delta)  # w = w + ^w = w + 学习率 * 隐层输出 * 输出层梯度项Gj
            self.V += lr * x.T.dot(L1_delta)  # v = v + ^v = v + 学习率 * 输入层输出 * 隐层梯度项Eh

            # 每训练1000次预测准确率
            if n != 0 and n % 1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):  # 用验证集去测试
                    res = self.predict(X_test[j])  # 返回预测结果,（1,10）
                    predictions.append(res)  # 将预测结果加入列表，稍后与真实结果做对比
                    # print('预测结果：', result, '; 实际结果：', y_test[j])

                accuracy = np.mean(np.equal(predictions, y_test))  # 预测结果和真实结果对比：求准确度平均值
                print('已训练:', n, '次, 当前在验证集上的准确率:', accuracy)

    # 预测单个样本，返回预测结果即输出层输出，(1, 10)
    def predict(self, x):
        temp = np.ones(len(x) + 1)  # 为数据添加偏置;注意这里x是一维的
        temp[0:-1] = x
        x = temp
        x = np.atleast_2d(x)

        L1 = sigmoid(np.dot(x, self.V))  # 隐层输出
        L2 = sigmoid(np.dot(L1, self.W))  # 输出层输出
        # print(L2)
        result = np.argmax(L2)  # 转换成对应标签
        return result


def test():
    digits = load_digits()  # 载入数据
    # print(digits.data.shape)  # 打印数据集大小(1797L, 64L）
    # pl.gray()  # 灰度化图片
    # pl.matshow(digits.images[0])  # 显示第1张图片，上面的数字是0
    # pl.show()

    X = digits.data  # 数据
    y = digits.target  # 标签

    # 数据归一化,一般是x=(x-x.min)/x.max-x.min
    X -= X.min()
    X /= X.max()

    # 创建神经网络
    nm = NeuralNetwork([64, 100, 10])
    X_train, X_test, y_train, y_test = train_test_split(X, y)  # 默认分割：3:1

    # 标签二值化  比如: 0 -> [1,0,0,0,0,0,0,0,0,0]
    #                 1 -> [0,1,0,0,0,0,0,0,0,0]
    #                 ...... 有几个标签就映射为多大长度的列表,当前标签位置置1,其他位置置0
    labels_train = LabelBinarizer().fit_transform(y_train)
    # print(labels_train[0:10])

    print('------ start training ------')
    nm.train(X_train, labels_train, X_test, y_test, epochs=20000)
    print('------ end training ------\n')

    print('------ start testing  ------')
    for i in range(10):
        rand = np.random.randint(X_test.shape[0])
        test = X_test[rand]
        label_true = y_test[rand]
        label_test = nm.predict(test)
        print('测试', i, ', 预测结果:', label_test, ', 实际结果:', label_true)
    print('------ end testing ------')


if __name__ == '__main__':
    test()
