import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data=np.array([
    [1,2],
    [1.5,1.8],
    [5,8],[8,8],[8,10],
    [1,0.6],[9,11]
])
Kmeans=KMeans(n_clusters=3,random_state=42)
Kmeans.fit(data)
labels=Kmeans.predict(data)
centers=Kmeans.cluster_centers_
colors=["red","green","orange"]
for i in range(len(data)):
    plt.scatter(data[i][0],data[i][1],
        c=colors[labels[i]],label=f"point{i}")
plt.scatter(centers[:,0],centers[:,1],
    c="blue",marker="X",s=200,label="centeriods")
plt.title("K_means clustering")
plt.legend()
plt.grid(True)
plt.show()
    