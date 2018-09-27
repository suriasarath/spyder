import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans

style.use('ggplot')

x = [1,1.5,3.7,4,5,6,7,8]
y = [1,2,3,7,8,7,7,8]

array = np.array([x,y])
array_t = np.transpose(array)

kmeans = KMeans(n_clusters=2)
kmeans.fit(array_t)
l = kmeans.predict([[5,6]])

centroids = kmeans.cluster_centers_
label = kmeans.labels_

colors = ["g.","r.","c.","y."]

for i in range(len(array_t)):
    plt.plot(array_t[i][0],array_t[i][1],markersize=30,marker = 'o')
    
    
plt.scatter(centroids[:,0],centroids[:,1],label = label[i],marker='x',s= 150,linewidths = 5, zorder = 10)

plt.show()