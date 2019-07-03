from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets

'''
  鸢尾花3分类问题
'''

svc = svm.SVC(kernel='linear')
# 鸢尾花数据集是sklearn自带的
iris = datasets.load_iris()
# 只提取前面两列数据作为特征
X = iris.data[:,:2]
y = iris.target
# 基于这些数据训练出一个支持向量分离器SVC
svc.fit(X, y)

# 将预测结果可视化
# 因为鸢尾花是3分类问题，我们要对样本和预测结果均用三种颜色区分开
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


def plot_estimator(estimator, X, y):  # 这个函数的作用是基于分类器，对预测结果与原始标签进行可视化。
    estimator.fit(X, y)
    # 确定网格最大最小值作为边界
    x_min, x_max = X[:, 0].min()-.1, X[:, 0].max()+.1
    y_min, y_max = X[:, 1].min()-.1, X[:, 1].max()+.1

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    # np.meshgrid 产生网格节点, X轴坐标xx, Y轴坐标yy
    # 例：a = np.array([1, 2, 3])
    # b = np.array([7, 8])
    # 返回list,有两个元素,第一个元素是X轴的取值,第二个元素是Y轴的取值
    # res = np.meshgrid(a, b)
    # 返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]

    # 基于分离器，对网格节点做预测
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])  # ravel()展平成一维数组;
                                                          # np.c_将两个特征合并到一个数组里,[ [f1, f2] [f1, f2] ...]
    # 对预测结果上色
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 对原始训练样本上色
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')  # 坐标轴适应数据量 plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_estimator(svc, X, y)