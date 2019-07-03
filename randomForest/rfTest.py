# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data[:, :2]
y = iris.target
RF = RandomForestClassifier(n_estimators=100, n_jobs=4, oob_score=True)
ET = ExtraTreesClassifier(n_estimators=100, n_jobs=4, oob_score=True, bootstrap=True)
RF.fit(x, y)
ET.fit(x, y)

h = 0.02  # step size

camp_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
camp_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

for classifier in ['RandomForestClassifier', 'ExtraTreesClassifier']:
    if classifier == 'RandomForestClassifier':
        Z = RF.predict(np.c_[xx.ravel(), yy.ravel()])
        print(classifier, ' score: ', RF.score(x, y))
    else:
        Z = ET.predict(np.c_[xx.ravel(), yy.ravel()])
        print(classifier, ' score: ', ET.score(x, y))
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=camp_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=camp_bold, edgecolors='k', s=20)
    plt.xlim(x_min, x_max)
    plt.title(classifier)
    plt.show()







