# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 16:03:05 2018

@author: gateway
"""
import matplotlib
matplotlib.use('Agg')

from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np
from mpl_toolkits.mplot3d import Axes3D



dataset = datasets.load_iris()
data = dataset.data
target  = dataset.target

clf = KMeans(n_clusters=8)

result = clf.fit(data)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
clf.fit(data)
labels = clf.labels_

ax.scatter(data[:, 3], data[:, 0], data[:, 2],
               c=labels.astype(np.float), edgecolor='k')

#ax.w_xaxis.set_ticklabels([])
#ax.w_yaxis.set_ticklabels([])
#ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title("3d plot")
ax.dist = 12
fig.show()
fig.savefig('foo.png')